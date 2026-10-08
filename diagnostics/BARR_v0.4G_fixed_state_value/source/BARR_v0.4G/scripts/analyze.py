#!/usr/bin/env python3
"""Source-equal summaries, missing native outcomes remain N/A; paired deltas."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

def summarize(rows):
    # seed mean -> graph mean -> physical-source mean. No successful-subset ranking.
    cells = defaultdict(list)
    for row in rows:
        cells[(row["variant"], row["allowance_seconds"], row["source_group"], row["graph_id"])].append(row)
    by_method = defaultdict(list)
    for (variant, budget, source, graph), trials in cells.items():
        complete = all(t.get("status") == "COMPLETE" and t.get("complete_native_quality") is not None for t in trials)
        by_method[(variant, budget)].append((source, graph, trials, complete))
    result = []
    for (variant, budget), graphs in sorted(by_method.items()):
        sources = defaultdict(list)
        for source, graph, trials, complete in graphs:
            sources[source].append(trials)
        full = all(v[3] for v in graphs)
        def source_equal(field):
            return statistics.mean(statistics.mean(statistics.mean(float(t[field]) for t in graph) for graph in group) for group in sources.values())
        count = sum(len(v[2]) for v in graphs)
        successes = sum(t["status"] == "COMPLETE" for v in graphs for t in v[2])
        result.append({"variant": variant, "allowance_seconds": budget, "sources": len(sources), "cells": count,
                       "complete_cells": successes, "complete_quality": source_equal("complete_native_quality") if full else None,
                       "caller_seconds": source_equal("caller_observed_seconds"),
                       "native_quality_not_imputed": True})
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); rows = [json.loads(s) for s in a.results.read_text().splitlines()]
    # An aborted matrix must not masquerade as a complete matrix.
    plan_path = a.results.parent / "plan.json"
    planned = len(json.loads(plan_path.read_text())["cells"]) if plan_path.is_file() else None
    keys = [(r["variant"], r["source_group"], r["graph_id"], r["allowance_seconds"], r["seed"], r.get("threads", 1)) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate experiment cells; do not treat repeated runs as extra seeds")
    summary = summarize(rows)
    matrix_complete = planned is not None and planned == len(rows)
    if not matrix_complete:
        for row in summary:
            row["complete_quality"] = None
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["variant"])
        writer.writeheader()
        for row in summary:
            writer.writerow({k: "N/A" if v is None else v for k, v in row.items()})
    # Per-instance/seed paired differences, including missing/unplanned controls.
    by_key = {(r["variant"], r["source_group"], r["graph_id"], r["allowance_seconds"], r["seed"], r.get("threads", 1)): r for r in rows}
    pairs = []
    for r in rows:
        if r["variant"] != "barr":
            continue
        for other in sorted({v["variant"] for v in rows} - {"barr"}):
            q = by_key.get((other, r["source_group"], r["graph_id"], r["allowance_seconds"], r["seed"], r.get("threads", 1)))
            valid = q is not None and r["status"] == q["status"] == "COMPLETE"
            pairs.append(dict(source_group=r["source_group"], graph_id=r["graph_id"], budget=r["allowance_seconds"], seed=r["seed"],
                              comparator=other, paired_difference=r["complete_native_quality"]-q["complete_native_quality"] if valid else "N/A",
                              pair_status="COMPLETE" if valid else "MISSING_OR_NOT_PLANNED"))
    if pairs:
        with (a.out / "paired_differences.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    report = {"summary": summary, "observed_cells": len(rows), "planned_cells": planned,
              "matrix_complete": matrix_complete, "missing_are_not_zero": True,
              "inference": "descriptive only; no significant-superiority conclusion from this script"}
    (a.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(a.out / "summary.csv")
