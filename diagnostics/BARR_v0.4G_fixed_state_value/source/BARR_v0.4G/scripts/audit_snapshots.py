#!/usr/bin/env python3
"""Sample frozen TRAIN/DEV/VAL kernels reproducibly and audit both backends.

Failed, timed-out, uncompiled and unsupported cases are retained. This is an
OFFLINE mechanism study; it must not be substituted for deployment timings.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from barr_io import atomic_json, sha256


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="run.py output with --snapshots")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "build/barr_probe")
    p.add_argument("--max-snapshots", type=int, default=20)
    p.add_argument("--seed", type=int, default=1703)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--boundary", type=int, default=10)
    args = p.parse_args()
    run = json.loads((args.run / "result.json").read_text())
    if str(run.get("split", "")).upper() not in ("TRAIN", "VAL", "DEV"):
        p.error("exploratory audits require TRAIN/VAL/DEV, not TEST/unknown")
    paths = sorted((args.run / "kernels").glob("*.barrk"))
    if not paths or args.max_snapshots < 1:
        p.error("no snapshots or invalid sample size")
    selected = sorted(random.Random(args.seed).sample(paths, min(args.max_snapshots, len(paths))))
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for path in selected:
        for backend in ("factor", "branch"):
            output = args.out / f"{path.stem}_{backend}.json"
            cmd = [str(args.binary.resolve()), "--kernel", str(path.resolve()), "--output", str(output.resolve()),
                   "--backend", backend, "--seconds", str(args.seconds), "--width", str(args.width), "--boundary", str(args.boundary)]
            row = dict(snapshot=path.name, snapshot_sha256=sha256(path), backend=backend, command=cmd)
            try:
                child = subprocess.run(cmd, capture_output=True, text=True, timeout=args.seconds+3)
                row["exit_code"] = child.returncode
                if child.returncode == 0:
                    row["result"] = json.loads(output.read_text())
                else:
                    row["error"] = child.stderr
            except subprocess.TimeoutExpired:
                row["error"] = "external timeout"
            records.append(row)
    statuses = {}
    for row in records:
        if "result" not in row:
            status = "error"
        else:
            r = row["result"]
            status = r["factor"]["status"] if row["backend"] == "factor" else ("unsupported" if not r["supported"] else "exact" if r["exact"] else "bounded")
        key = row["backend"]+":"+status; statuses[key] = statuses.get(key, 0)+1
    atomic_json(args.out / "audit_manifest.json", {"schema": "barr_snapshot_audits_v1", "run_result_sha256": sha256(args.run / "result.json"),
        "source_group": run.get("source_group"), "split": run.get("split"), "graph_sha256": run.get("graph_sha256"),
        "binary_sha256": sha256(args.binary), "available_snapshots": len(paths), "sampled_snapshots": len(selected),
        "sampling_seed": args.seed, "diagnostic_not_deployment": True, "status_counts": statuses, "records": records})
    print(json.dumps(statuses, indent=2))

if __name__ == "__main__":
    main()
