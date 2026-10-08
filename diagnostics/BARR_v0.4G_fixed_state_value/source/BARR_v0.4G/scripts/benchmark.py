#!/usr/bin/env python3
"""Full factorial runs with recorded failures. No best-of-seeds selection."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from barr_io import load_npz, atomic_json
from run import parser as run_parser, execute
VARIANTS = {
    "local": ["--method", "local"], "fusion": ["--method", "fusion"],
    "barr": ["--method", "barr"],
    "barr-factor": ["--method", "barr", "--recovery-backend", "factor"],
    "barr-branch": ["--method", "barr", "--recovery-backend", "branch"],
    "independent": ["--rank", "independent"],
    "random-rank": ["--rank", "random"], "random-domain": ["--random-domains"],
    "greedy-repair": ["--greedy-repair"], "no-decompose": ["--no-decompose", "--recovery-backend", "branch"],
    "fixed-k": ["--fixed-k"], "gnn": ["--rank", "gnn"],
    "chils-p1": ["--method", "chils", "--chils-population", "1"],
    "chils-p16": ["--method", "chils", "--chils-population", "16"]}
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=VARIANTS, default=["local", "fusion", "barr", "random-domain", "greedy-repair"])
    p.add_argument("--budgets", type=float, nargs="+", default=[10, 30, 60])
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--binary", type=Path, default=ROOT / "build/barr_solver")
    p.add_argument("--chils", type=Path, default=ROOT / "external/CHILS/CHILS")
    p.add_argument("--chils-step", type=float, default=.1)
    p.add_argument("--model", type=Path)
    p.add_argument("--split", choices=("TRAIN", "VAL", "DEV", "TEST"), default="VAL")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--execute", action="store_true", help="otherwise save only the run plan")
    a = p.parse_args()
    if a.split == "TEST" and not a.allow_test:
        p.error("TEST is locked; freeze all methods/parameters, then pass --allow-test")
    if "gnn" in a.methods and not a.model:
        p.error("gnn requires --model")
    rows = list(csv.DictReader(a.manifest.open()))
    sources = {}
    for row in rows:
        source, split = row["source_group"], row["split"].upper()
        if source in sources and sources[source] != split:
            raise ValueError("same physical source appears in multiple splits")
        sources[source] = split
    rows = [r for r in rows if r["split"].upper() == a.split]
    if not rows:
        p.error("no selected graphs")
    plans = []
    for row in rows:
        graph = Path(row["graph"])
        if not graph.is_absolute():
            graph = a.manifest.resolve().parent / graph
        graph_data = load_npz(graph)
        if graph_data.metadata["source_group"] != row["source_group"] or str(graph_data.metadata["split"]).upper() != a.split:
            raise ValueError("NPZ and manifest disagree")
        for budget in a.budgets:
            for seed in a.seeds:
                for method in a.methods:
                    opts = ["--graph", str(graph), "--out", str(a.out / f"cell_{len(plans):05d}"),
                            "--seconds", str(budget), "--seed", str(seed), "--threads", str(a.threads),
                            "--binary", str(a.binary), "--chils", str(a.chils), "--chils-step", str(a.chils_step)] + VARIANTS[method]
                    if method == "gnn":
                        opts += ["--model", str(a.model)]
                    plans.append({"variant": method, "options": opts, "source_group": row["source_group"], "graph_id": row["graph_id"]})
    a.out.mkdir(parents=True, exist_ok=False)
    atomic_json(a.out / "plan.json", {"schema": "barr_plan_v1", "cells": plans, "split": a.split,
                                    "test_explicitly_unlocked": a.allow_test, "allowance_protocol": "barr_run_v1"})
    if a.execute:
        with (a.out / "results.jsonl").open("w", encoding="utf-8") as f:
            for i, plan in enumerate(plans):
                result = execute(run_parser().parse_args(plan["options"]))
                result["variant"] = plan["variant"]
                f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"); f.flush()
                print(f"{i+1}/{len(plans)} {plan['variant']} {result['status']}", flush=True)
