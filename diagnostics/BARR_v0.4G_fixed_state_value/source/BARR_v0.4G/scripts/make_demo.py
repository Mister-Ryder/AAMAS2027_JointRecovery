#!/usr/bin/env python3
"""Synthetic single-terminal contact conflict graphs; NOT STK or orbital propagation."""
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from barr_io import common_greedy

def generate(path: Path, n: int, seed: int, source: str, split: str) -> None:
    rng = np.random.default_rng(seed)
    satellites = rng.integers(0, max(6, n // 80), size=n, dtype=np.int64)
    stations = rng.integers(0, max(3, n // 300), size=n, dtype=np.int64)
    start = rng.uniform(0, 3600, n)
    end = start + rng.uniform(20, 240, n)
    es = set()
    # An owner is not a day-long clique: only interval overlap on that resource conflicts.
    for resource in (satellites, stations):
        for key in np.unique(resource):
            ids = np.flatnonzero(resource == key)
            active = []
            for v in ids[np.argsort(start[ids])]:
                active = [u for u in active if end[u] > start[v]]
                for u in active:
                    es.add((min(int(u), int(v)), max(int(u), int(v))))
                active.append(int(v))
    edges = np.asarray(sorted(es), dtype=np.int64).reshape(-1, 2)
    weights = end - start
    initial = np.zeros(n, dtype=np.uint8)
    initial[common_greedy(weights, edges)] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weights=weights, edges=edges, agents=satellites,
                        stations=stations, start_seconds=start, end_seconds=end, initial_mask=initial,
                        graph_id=path.stem, source_group=source, split=split, origin="synthetic_interval_contacts_NOT_STK")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "examples/demo")
    p.add_argument("--vertices", type=int, default=1000)
    a = p.parse_args()
    if a.vertices < 1:
        p.error("vertices must be positive")
    rows = []
    for i, split in enumerate(("TRAIN", "TRAIN", "VAL", "TEST")):
        path = a.out / f"contacts_{i:02d}.npz"
        generate(path, a.vertices, 10600 + i, f"synthetic_source_{i:02d}", split)
        rows.append(dict(graph=str(path.resolve()), graph_id=path.stem, source_group=f"synthetic_source_{i:02d}", split=split))
    with (a.out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(a.out / "manifest.csv")
