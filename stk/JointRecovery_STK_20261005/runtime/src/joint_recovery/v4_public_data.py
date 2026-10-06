"""Native-integer public inputs for V4, independent of frozen V3 code.

No download, optimization, labels, or float conversion is performed here.
Amazon VR edge records are made into a simple undirected graph explicitly;
PACE vertex cover is interpreted as MIS on the SAME graph, not its complement.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import tarfile
from typing import Dict, Optional, Tuple
import zipfile

import numpy as np

PACE_OFFICIAL_MD5 = "33e9a2b7a064dce16ed1c5d1718a97ba"
MAX_VERTEX_WEIGHT = (1 << 63) - 1


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _integer_weights(values) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in ("i", "u", "O"):
        raise ValueError("Native weights must be a one-dimensional integer array")
    checked = []
    for w in values:
        if isinstance(w, (bool, np.bool_)) or not isinstance(w, (int, np.integer)):
            raise ValueError("Do not coerce native weights from floating point")
        w = int(w)
        if not 0 < w <= MAX_VERTEX_WEIGHT:
            raise ValueError("Each positive weight must fit signed int64")
        checked.append(w)
    result = np.asarray(checked, dtype=np.int64)
    result.setflags(write=False)
    return result


def exact_objective(weights, mask) -> int:
    """Python-integer sum; never np.dot/int64 accumulation or float rounding."""
    weights = _integer_weights(weights)
    mask = np.asarray(mask)
    if mask.shape != weights.shape or not np.isin(mask, (0, 1)).all():
        raise ValueError("Membership must be a binary vector with one entry per vertex")
    return sum(int(w) for w, selected in zip(weights, mask) if selected)


def mask_from_ids(n: int, ids, one_based: bool = True) -> np.ndarray:
    ids = list(ids)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in ids):
        raise ValueError("Vertex IDs must be integers")
    ids = [int(v) - int(one_based) for v in ids]
    if len(ids) != len(set(ids)) or any(v < 0 or v >= n for v in ids):
        raise ValueError("Membership IDs must be unique and in range")
    mask = np.zeros(n, dtype=np.int8)
    mask[ids] = 1
    return mask


def _simple_edges(n: int, edge_rows: np.ndarray):
    edge_rows = np.asarray(edge_rows)
    if edge_rows.size == 0:
        edge_rows = np.empty((0, 2), dtype=np.int64)
    if edge_rows.ndim != 2 or edge_rows.shape[1] != 2 or edge_rows.dtype.kind not in ("i", "u"):
        raise ValueError("Edge records must be integer endpoint pairs")
    if n >= (1 << 31) or np.any(edge_rows < 0) or np.any(edge_rows >= n):
        raise ValueError("Endpoint is out of range, or graph exceeds int32 vertex IDs")
    if np.any(edge_rows[:, 0] == edge_rows[:, 1]):
        raise ValueError("Self loops require an explicit problem transform; do not discard them")
    ordered = np.sort(edge_rows.astype(np.int64, copy=False), axis=1)
    unique = np.unique(ordered, axis=0)
    u = np.asarray(unique[:, 0], dtype=np.int32)
    v = np.asarray(unique[:, 1], dtype=np.int32)
    u.setflags(write=False); v.setflags(write=False)
    return u, v, len(edge_rows) - len(unique)


def labeled_topology_sha256(n: int, u, v) -> str:
    """Sorted one-based edge text + n, matching the V4 input-only inspection."""
    h = hashlib.sha256((str(n) + "\n").encode("ascii"))
    for start in range(0, len(u), 10000):
        rows = ["%d %d" % (int(a) + 1, int(b) + 1)
                for a, b in zip(u[start:start + 10000], v[start:start + 10000])]
        if start:
            h.update(b"\n")
        h.update("\n".join(rows).encode("ascii"))
    return h.hexdigest()


@dataclass(frozen=True)
class PublicGraph:
    graph_id: str
    corpus: str
    family: str
    weights: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    initial_mask: Optional[np.ndarray]
    cliques: Tuple[np.ndarray, ...]
    metadata: Dict

    @property
    def n(self) -> int:
        return len(self.weights)

    @property
    def m(self) -> int:
        return len(self.edge_u)

    def validate_mask(self, mask) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.shape != (self.n,) or not np.isin(mask, (0, 1)).all():
            raise ValueError("Membership has incorrect shape or non-binary values")
        selected = mask.astype(np.int8)
        if np.any(selected[self.edge_u] & selected[self.edge_v]):
            raise ValueError("Returned membership conflicts on an original graph edge")
        return selected

    def objective(self, mask) -> int:
        return exact_objective(self.weights, self.validate_mask(mask))


def _read_integer_rows(data: bytes, width: int) -> np.ndarray:
    rows = []
    for number, line in enumerate(data.decode("ascii").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != width or any(re.fullmatch(r"[+-]?\d+", value) is None for value in fields):
            raise ValueError("Malformed integer row at line %d" % number)
        row = [int(value) for value in fields]
        if any(not -(1 << 63) <= value <= MAX_VERTEX_WEIGHT for value in row):
            raise ValueError("Input integer is outside signed int64")
        rows.append(row)
    return np.asarray(rows, dtype=np.int64).reshape((-1, width))


def _clique_check(n, u, v, cliques):
    """Check all supplied factors; incomplete covers are reported, not invented."""
    edge_keys = u.astype(np.int64) * n + v
    covered = np.zeros(len(u), dtype=bool)
    for clique in cliques:
        if len(clique) != len(np.unique(clique)) or np.any(clique < 0) or np.any(clique >= n):
            raise ValueError("Clique contains duplicate or invalid vertex IDs")
        clique = np.sort(clique)
        for i, a in enumerate(clique[:-1]):
            keys = int(a) * n + clique[i + 1:]
            positions = np.searchsorted(edge_keys, keys)
            if np.any(positions >= len(edge_keys)):
                raise ValueError("Supplied factor is not a clique in the original graph")
            if np.any(edge_keys[positions] != keys):
                raise ValueError("Supplied factor is not a clique in the original graph")
            covered[positions] = True
    return dict(clique_count=len(cliques), cliques_checked=True,
                clique_edge_cover_count=int(covered.sum()),
                clique_edge_cover_complete=bool(covered.all()),
                uncovered_edge_count=int((~covered).sum()))


def load_vr_archive(path, graph_id: Optional[str] = None, *,
                    max_edge_records: int = 20000000, verify_cliques: bool = True,
                    max_semantic_bytes: int = 512 << 20) -> PublicGraph:
    """Parse one native Amazon tar.gz, without extracting files or using LP labels.

    max_edge_records/max_semantic_bytes are explicit memory guards, not selections.
    A rejected archive still belongs in the experiment coverage manifest.
    """
    path = Path(path)
    archive_sha = _file_sha(path)
    required = {"instance_name.txt", "conflict_graph.txt", "node_weights.txt", "solution.txt", "cliques.txt"}
    files = {}
    semantic_bytes = 0
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            key = Path(member.name.replace("\\", "/")).name
            if key not in required:
                continue
            if not member.isfile() or key in files:
                raise ValueError("Duplicate or non-regular semantic archive member")
            semantic_bytes += member.size
            if member.size < 0 or semantic_bytes > max_semantic_bytes:
                raise ValueError("Semantic archive bytes exceed the declared input memory guard")
            files[key] = archive.extractfile(member).read()
    if set(files) != required:
        raise ValueError("VR archive is missing required semantic members: %s" % sorted(required - set(files)))
    source_id = files["instance_name.txt"].decode("ascii").strip()
    if graph_id is not None and source_id != graph_id:
        raise ValueError("Archive instance_name does not match the declared graph ID")
    if re.fullmatch(r"(?:MT|MW|MR|CW|CR)-[A-Za-z0-9-]+", source_id) is None:
        raise ValueError("Unexpected VR source ID")
    lines = files["conflict_graph.txt"].splitlines()
    if not lines or len(_read_integer_rows(lines[0], 2)) != 1:
        raise ValueError("Conflict graph requires a single n m header")
    header = _read_integer_rows(lines[0], 2)
    n, record_count = map(int, header[0])
    if not 0 < n < (1 << 31) or not 0 <= record_count <= max_edge_records:
        raise ValueError("Graph exceeds the declared input memory guard")
    edges = _read_integer_rows(b"\n".join(lines[1:]), 2)
    if len(edges) != record_count:
        raise ValueError("Raw edge row count does not match the source header")
    u, v, duplicates = _simple_edges(n, edges - 1)
    weight_rows = _read_integer_rows(files["node_weights.txt"], 2)
    if len(weight_rows) != n or set(weight_rows[:, 0].tolist()) != set(range(1, n + 1)):
        raise ValueError("Weights require each original vertex ID exactly once")
    weights = _integer_weights(weight_rows[np.argsort(weight_rows[:, 0]), 1])
    initial_ids = _read_integer_rows(files["solution.txt"], 1).ravel()
    initial = mask_from_ids(n, initial_ids)
    cliques = []
    for line in files["cliques.txt"].splitlines():
        if not line.strip():
            continue
        fields = line.split()
        values = _read_integer_rows(b"\n".join(fields), 1).ravel() - 1
        values.setflags(write=False)
        cliques.append(values)
    check = _clique_check(n, u, v, cliques) if verify_cliques else dict(cliques_checked=False)
    topology = labeled_topology_sha256(n, u, v)
    h = hashlib.sha256(bytes.fromhex(topology)); h.update(weights.astype("<i8").tobytes())
    metadata = dict(source_archive_sha256=archive_sha, source_id=source_id,
                    semantic_bytes=semantic_bytes,
                    source_member_sha256={k: _sha(b) for k, b in sorted(files.items())},
                    declared_edge_records=record_count, unique_undirected_edges=len(u),
                    duplicate_edge_records=duplicates, weights_origin="native positive integers",
                    weight_scale=1, weight_sum_exact=sum(map(int, weights)),
                    labeled_topology_sha256=topology, weighted_graph_sha256=h.hexdigest(),
                    original_node_ids="1..n; parser arrays use ID-1",
                    source_cluster="amazon-vr-anonymized-corpus; city/route overlap unresolved",
                    nominal_family=source_id.split("-", 1)[0], actual_agent_ownership_available=False,
                    official_lp_loads_used=False, **check)
    graph = PublicGraph(source_id, "Amazon-MWIS-VR", metadata["nominal_family"],
                        weights, u, v, initial, tuple(cliques), metadata)
    graph.validate_mask(initial)
    initial.setflags(write=False)
    metadata["initial_weight_exact"] = graph.objective(initial)
    metadata["initial_feasible"] = True
    return graph


def load_pace_archive(path, *, expected_md5: Optional[str] = PACE_OFFICIAL_MD5,
                      expected_count: Optional[int] = 200) -> Tuple[PublicGraph, ...]:
    """Official 200 files; duplicate labeled topologies share a cluster/split.

    For synthetic parser guards only, explicitly use expected_md5/count=None.
    Original odd/even challenge split is recorded, not claimed as our untouched
    or externally untrained split. V4 topology inspection has already occurred.
    """
    payload = Path(path).read_bytes()
    if expected_md5 is not None and hashlib.md5(payload).hexdigest() != expected_md5:
        raise ValueError("PACE archive differs from the published MD5")
    graphs = []
    archive_sha = _sha(payload)
    ids = set()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in sorted(archive.namelist()):
            name = Path(member).name
            if not name.endswith((".gr", ".hgr")):
                continue
            match = re.fullmatch(r"vc-exact_(\d{3})\.(?:gr|hgr)", name)
            if match is None or name in ids:
                raise ValueError("Unexpected or duplicate PACE graph member")
            ids.add(name)
            raw = archive.read(member)
            lines = [line.strip() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith(b"c")]
            if not lines:
                raise ValueError("PACE graph is missing its header")
            fields = lines[0].split()
            if len(fields) != 4 or fields[:2] != [b"p", b"td"]:
                raise ValueError("PACE requires p td n m")
            n, records = map(int, fields[2:])
            edges = _read_integer_rows(b"\n".join(lines[1:]), 2)
            if n <= 0 or records < 0 or len(edges) != records:
                raise ValueError("Invalid PACE declared graph size")
            u, v, duplicates = _simple_edges(n, edges - 1)
            weights = _integer_weights(np.ones(n, dtype=np.int64))
            topology = labeled_topology_sha256(n, u, v)
            number = int(match.group(1))
            metadata = dict(source_archive_sha256=archive_sha, source_member=member,
                            source_member_sha256=_sha(raw), declared_edge_records=records,
                            unique_undirected_edges=len(u), duplicate_edge_records=duplicates,
                            weights_origin="original unweighted VC/MIS; all weights exactly one",
                            transform="VC complement membership on same topology; no graph complement",
                            labeled_topology_sha256=topology, cluster_id="pace-topology-" + topology,
                            historical_split="development-odd" if number % 2 else "challenge-test-even",
                            input_topology_inspected_in_v4_design=True,
                            external_checkpoint_training_overlap="unverified")
            graphs.append(PublicGraph("pace2019-" + match.group(1), "PACE-2019-VC", "PACE",
                                      weights, u, v, None, (), metadata))
    if not graphs or (expected_count is not None and len(graphs) != expected_count):
        raise ValueError("PACE coverage differs from the declared file count")
    cluster_split = {}
    for graph in graphs:
        key = graph.metadata["cluster_id"]
        old = cluster_split.setdefault(key, graph.metadata["historical_split"])
        if old != graph.metadata["historical_split"]:
            raise ValueError("Duplicate topology crosses the historical odd/even split; freeze a group split instead")
    return tuple(graphs)


def write_weighted_metis(graph: PublicGraph, path) -> Dict:
    """Exact native integer text. Never apply V3's float-rounding scale=1000."""
    rows = [[] for _ in range(graph.n)]
    for a, b in zip(graph.edge_u, graph.edge_v):
        rows[int(a)].append(int(b) + 1); rows[int(b)].append(int(a) + 1)
    with Path(path).open("w", encoding="ascii", newline="\n") as stream:
        stream.write("%d %d 10\n" % (graph.n, graph.m))
        for w, neighbors in zip(graph.weights, rows):
            stream.write(" ".join(map(str, [int(w)] + sorted(neighbors))) + "\n")
    return dict(weight_scale=1, per_vertex_rounding_max=0,
                output_sha256=_sha(Path(path).read_bytes()),
                objective_sum_exact=sum(map(int, graph.weights)),
                fits_signed_int64_total=sum(map(int, graph.weights)) <= MAX_VERTEX_WEIGHT,
                fits_unsigned_int32_total=sum(map(int, graph.weights)) < (1 << 32))


def read_vertex_id_solution(path, graph: PublicGraph) -> np.ndarray:
    """CHILS CLI -o/-i contract: one selected original 1-based ID per line.

    An empty file is the feasible empty set, not a successful solver exit receipt.
    Exit status, deadline and file creation freshness belong to the runner.
    """
    ids = _read_integer_rows(Path(path).read_bytes(), 1).ravel()
    return graph.validate_mask(mask_from_ids(graph.n, ids))


def write_vertex_id_solution(graph: PublicGraph, mask, path) -> Dict:
    """CHILS CLI warm start in original IDs; validate before emitting."""
    mask = graph.validate_mask(mask)
    text = "".join("%d\n" % (int(i) + 1) for i in np.flatnonzero(mask))
    with Path(path).open("w", encoding="ascii", newline="\n") as stream:
        stream.write(text)
    return dict(output_sha256=_file_sha(Path(path)), selected_count=int(mask.sum()),
                objective_exact=graph.objective(mask), id_base=1)


def read_binary_membership(path, graph: PublicGraph) -> np.ndarray:
    """KaMIS mmwis-style binary membership, exactly one value per vertex."""
    mask = _read_integer_rows(Path(path).read_bytes(), 1).ravel()
    return graph.validate_mask(mask)
