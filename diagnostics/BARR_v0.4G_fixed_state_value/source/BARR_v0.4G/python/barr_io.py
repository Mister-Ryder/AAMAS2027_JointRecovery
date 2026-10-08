"""Original-weight graph I/O and full-graph validation. No model/solver dependency."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import numpy as np

MAX_TOTAL = (1 << 60) - 1

@dataclass
class Instance:
    weights: np.ndarray
    edges: np.ndarray
    owners: np.ndarray
    initial: np.ndarray
    metadata: dict

    @property
    def n(self) -> int:
        return len(self.weights)

    def validate(self, selected) -> np.ndarray:
        ids = np.asarray(selected)
        if ids.size == 0:
            ids = np.empty(0, dtype=np.int64)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("selected must be a one-dimensional integer ID list")
        ids = ids.astype(np.int64, copy=False)
        if np.any(ids < 0) or np.any(ids >= self.n) or len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate or out-of-range selected IDs")
        mask = np.zeros(self.n, dtype=bool)
        mask[ids] = True
        if self.edges.size and np.any(mask[self.edges[:, 0]] & mask[self.edges[:, 1]]):
            raise ValueError("solution is infeasible on the original complete graph")
        return ids

    def objective(self, selected) -> float:
        ids = self.validate(selected)
        value = math.fsum(float(self.weights[v]) for v in ids)
        if not math.isfinite(value):
            raise ValueError("original objective overflow")
        return value

    def ticks(self, tick_seconds: float) -> list[int]:
        if not math.isfinite(tick_seconds) or tick_seconds <= 0:
            raise ValueError("tick must be positive and finite")
        # Match the existing project's scalar Python round(float/tick) convention.
        scaled = [float(w) / tick_seconds for w in self.weights]
        if any(not math.isfinite(w) or w > MAX_TOTAL for w in scaled):
            raise ValueError("integer domain overflow")
        result = [int(round(w)) for w in scaled]
        if sum(result) > MAX_TOTAL:
            raise ValueError("sum of integer weights exceeds 2^60-1")
        return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _integers(value, name: str) -> np.ndarray:
    a = np.asarray(value)
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must use an integer dtype")
    if np.issubdtype(a.dtype, np.unsignedinteger) and a.size and int(a.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds int64")
    return a.astype(np.int64)


def common_greedy(weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n = len(weights)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[int(u)].append(int(v)); adj[int(v)].append(int(u))
    candidates = []
    for alpha in (0.0, 0.5, 1.0):
        order = sorted(range(n), key=lambda v: (-float(weights[v]) / (1 + len(adj[v])) ** alpha, -float(weights[v]), v))
        blocked = np.zeros(n, dtype=bool); selected = []
        for v in order:
            if not blocked[v]:
                selected.append(v); blocked[v] = True; blocked[adj[v]] = True
        ids = tuple(sorted(selected))
        candidates.append((math.fsum(float(weights[v]) for v in ids), ids))
    return np.asarray(max(candidates)[1], dtype=np.int64)


def load_npz(path: Path, start: str = "warm") -> Instance:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        w = np.asarray(z["weights"], dtype=np.float64)
        if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite, nonnegative and one-dimensional")
        if "edges" in z:
            e = _integers(z["edges"], "edges")
            if e.shape != (len(e), 2):
                raise ValueError("edges must have shape (m,2)")
        elif "edge_u" in z and "edge_v" in z:
            u, v = _integers(z["edge_u"], "edge_u"), _integers(z["edge_v"], "edge_v")
            if u.ndim != 1 or v.shape != u.shape:
                raise ValueError("edge arrays must have equal one-dimensional shape")
            e = np.column_stack((u, v))
        else:
            raise ValueError("missing edges or edge_u/edge_v")
        if e.size and (np.any(e < 0) or np.any(e >= len(w)) or np.any(e[:, 0] == e[:, 1])):
            raise ValueError("out-of-range edge or self-loop")
        e = np.sort(e, axis=1)
        if len(np.unique(e, axis=0)) != len(e):
            raise ValueError("duplicate edges (including reverse duplicates)")
        if len(e):
            e = e[np.lexsort((e[:, 1], e[:, 0]))]
        own = _integers(z["agents"], "agents") if "agents" in z else np.full(len(w), -1, dtype=np.int64)
        if own.shape != w.shape:
            raise ValueError("agents length differs from weights")
        metadata = {"graph_id": path.stem, "source_group": None, "split": None}
        for key in metadata:
            if key in z:
                if z[key].shape != ():
                    raise ValueError(f"{key} must be a scalar")
                metadata[key] = str(z[key].item())
        if start == "cold":
            initial = np.empty(0, dtype=np.int64)
            metadata["initial_origin"] = "empty"
        elif start == "warm":
            if "initial_mask" in z:
                mask = np.asarray(z["initial_mask"])
                if mask.shape != w.shape or not np.all((mask == 0) | (mask == 1)):
                    raise ValueError("invalid initial_mask")
                initial = np.flatnonzero(mask).astype(np.int64)
                metadata["initial_origin"] = "saved_initial_mask"
            else:
                initial = common_greedy(w, e)
                metadata["initial_origin"] = "common_three_greedy"
        else:
            raise ValueError("start must be warm or cold")
    instance = Instance(w, e, own, initial, metadata)
    instance.objective(initial)
    return instance


def write_native(path: Path, instance: Instance, ticks: list[int]) -> None:
    with Path(path).open("w", encoding="ascii", newline="\n") as f:
        f.write(f"BARR1 {instance.n} {len(instance.edges)}\n")
        for w, raw, owner in zip(ticks, instance.weights, instance.owners):
            f.write(f"{w} {float(raw):.17g} {int(owner)}\n")
        for u, v in instance.edges:
            f.write(f"{u} {v}\n")
        f.write(str(len(instance.initial)) + "\n")
        for v in instance.initial:
            f.write(f"{v}\n")


def write_metis(path: Path, initial_path: Path, instance: Instance, ticks: list[int]) -> None:
    # Same narrow pinned-upstream p1 domain retained by the supplied repository.
    # Also used conservatively for p>1, but not a proof that concurrent code is UB-free.
    if instance.n == 0 or instance.n > 32760 or sum(ticks) > (1 << 40) or any(w <= 0 for w in ticks):
        raise ValueError("outside conservative external-CHILS comparison domain (0<n<=32760, positive ticks, total<=2^40)")
    adj = [[] for _ in range(instance.n)]
    for u, v in instance.edges:
        adj[int(u)].append(int(v)); adj[int(v)].append(int(u))
    with Path(path).open("w", encoding="ascii", newline="\n") as f:
        f.write(f"{instance.n} {len(instance.edges)} 10\n")
        for v, row in enumerate(adj):
            f.write(str(ticks[v]) + " " + " ".join(str(u + 1) for u in sorted(row)) + "\n")
    Path(initial_path).write_text("".join(f"{v+1}\n" for v in instance.initial), encoding="ascii")
