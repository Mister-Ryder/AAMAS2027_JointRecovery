#!/usr/bin/env python3
"""Offline counterfactual action labeling on TRAIN / VAL sources only.

Runs every proposed action on the same frozen parental state and same workpoint.
This extra work is NOT deployment performance, nor an exact optimum oracle.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from barr_io import load_npz, atomic_json, sha256
from run import parser as run_parser, execute

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    p.add_argument("--split", choices=("TRAIN", "VAL", "DEV"), required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "build/barr_solver")
    p.add_argument("--max-groups", type=int, default=500)
    p.add_argument("--recovery-backend", choices=("hybrid", "factor", "branch"), default="hybrid")
    p.add_argument("--challengers", type=int, default=12)
    a = p.parse_args()
    rows = list(csv.DictReader(a.manifest.open()))
    selected_rows = [row for row in rows if row["split"].upper() == a.split]
    if not selected_rows:
        p.error("no matching split")
    a.out.mkdir(parents=True, exist_ok=False)
    records = []; count = 0
    with (a.out / "labels.jsonl").open("w", encoding="utf-8") as labels:
        for row in selected_rows:
            graph = Path(row["graph"])
            if not graph.is_absolute():
                graph = a.manifest.resolve().parent / graph
            instance = load_npz(graph)
            if str(instance.metadata["split"]).upper() != a.split or instance.metadata["source_group"] != row["source_group"]:
                raise ValueError("manifest and NPZ source/split disagree")
            for seed in a.seeds:
                run_dir = a.out / f"run_{len(records):04d}"
                args = run_parser().parse_args(["--graph", str(graph), "--out", str(run_dir), "--seconds", str(a.seconds),
                                               "--binary", str(a.binary), "--seed", str(seed), "--trace", "--rank", "random",
                                               "--challengers", str(a.challengers), "--recovery-backend", a.recovery_backend])
                result = execute(args)
                records.append({"graph": str(graph), "source_group": row["source_group"], "seed": seed,
                                "status": result["status"], "graph_sha256": sha256(graph)})
                trace_path = run_dir / "counterfactual.jsonl"
                if trace_path.is_file():
                    used = 0
                    for line in trace_path.read_text().splitlines():
                        try:
                            group = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # killed writer's incomplete final line; count in collection report
                        if used >= a.max_groups:
                            break
                        group["executor_contract"] = result["executor_contract"]
                        group.update(source_group=row["source_group"], graph_sha256=sha256(graph), split=a.split, seed=seed)
                        labels.write(json.dumps(group, allow_nan=False) + "\n"); count += 1; used += 1
    atomic_json(a.out / "collection.json", {"schema": "barr_labels_collection_v1", "runs": records, "groups": count,
                                          "extra_counterfactual_work": True, "labels_are_bounded_outcomes_not_optima": True})
    print(a.out / "labels.jsonl")
