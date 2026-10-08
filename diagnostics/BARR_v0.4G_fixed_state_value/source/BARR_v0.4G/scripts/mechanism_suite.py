#!/usr/bin/env python3
"""Controlled, frozen-state mechanism experiments. NOT STK and NOT a CHILS ranking.

Creates connected non-bipartite low-coupling-width families, pure high-order
improvement motifs, and unary-indistinguishable controls. All expected gains are
specified analytically before execution. Failed/unsupported runs remain in JSON.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from barr_io import atomic_json, sha256


def frozen(path: Path, weights, edges, p, q, base) -> None:
    edges = sorted({tuple(sorted(e)) for e in edges})
    p, q, base = set(p), set(q), set(base)
    lines = [f"BARRK1 {len(weights)} {len(edges)}"]
    for v, w in enumerate(weights):
        color = 0 if v in p else 1 if v in q else -1
        freq = (int(v in p) + int(v in q)) / 2
        lines.append(f"{w} {w} -1 {color} {int(v in base)} {int(v in p)} {int(v in q)} {freq} {v}")
    lines.extend(f"{u} {v}" for u, v in edges)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_chain(path: Path, count: int, dense: bool = False, block_size: int = 1) -> None:
    # C_j is K_{r,r}: exact optimum 2*r. Each outsider has weight 3*r
    # and touches every vertex in C_i and C_{i+1}. This separates the
    # complexity of the backbone itself from the width of its responses.
    if count < 1 or block_size < 1:
        raise ValueError("positive chain and block sizes required")
    r = block_size; p=[]; q=[]; weights=[]; edges=[]
    for j in range(count+1):
        left=list(range(2*r*j,2*r*j+r)); right=list(range(2*r*j+r,2*r*(j+1)))
        p.extend(left);q.extend(right);weights.extend([2]*r+[1]*r)
        edges.extend((u,v) for u in left for v in right)
    start=len(weights);weights += [3*r]*count
    for i in range(count):
        edges.extend((start+i,u) for j in (i,i+1) for u in range(2*r*j,2*r*(j+1)))
    if dense:
        edges.extend((start+i,start+j) for i in range(count) for j in range(i+1,count))
    frozen(path, weights, edges, p, q, p)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "build/barr_probe")
    p.add_argument("--seconds", type=float, default=5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    fixtures = []
    for count in (4, 8, 16, 24, 32, 64, 128):
        path = args.out / f"chain_{count:03d}.barrk"; make_chain(path, count)
        fixtures.append(dict(name=path.stem, file=path.name, family="connected_triangle_chain", k=count,
                             expected_gain=count-2, expected_width=1, expected_min_positive_size=3))
    for r in (4,16):
        path=args.out/f"block_chain_064_r{r:02d}.barrk";make_chain(path,64,block_size=r)
        fixtures.append(dict(name=path.stem,file=path.name,family="dense_bipartite_backbone_chain",k=64,block_size=r,
                             expected_gain=r*62,expected_width=1,expected_min_positive_size=3))
    for count in (3, 4, 6, 8, 12):
        path = args.out / f"order_{count:02d}.barrk"
        # A single edge backbone. Choosing any challenger removes both endpoints.
        frozen(path, [2*count-1, 1]+[2]*count, [(0, 1)]+[(u, v) for u in (0, 1) for v in range(2, count+2)], [0], [1], [0])
        fixtures.append(dict(name=path.stem, file=path.name, family="pure_high_order", k=count,
                             expected_gain=1, expected_width=count-1, expected_min_positive_size=count))
    for name, edges, gain in (("shared", [(0,2),(0,3)], 2), ("separate", [(0,2),(1,3)], 0)):
        path = args.out / f"unary_{name}.barrk"; frozen(path, [10,10,6,6], edges, [0,1], [0,1], [0,1])
        fixtures.append(dict(name=path.stem, file=path.name, family="same_unary_controls", k=2, expected_gain=gain))
    path = args.out / "dense_32.barrk"; make_chain(path, 32, dense=True)
    fixtures.append(dict(name=path.stem, file=path.name, family="width_limit_negative_control", k=32, expected_gain=0))
    atomic_json(args.out / "preregistered_fixture_expectations.json", {"source": "analytic constructions, not observations",
        "fixtures": fixtures, "random_selection_or_posthoc_filtering": False})
    rows = []
    for fixture in fixtures:
        for backend in ("factor", "branch"):
            name = fixture["name"]+"_"+backend; output = args.out / (name+".json")
            # The larger caps on tiny motif audits are diagnostic, not deployment defaults.
            cmd = [str(args.binary.resolve()), "--kernel", str((args.out / fixture["file"]).resolve()),
                   "--output", str(output.resolve()), "--backend", backend, "--seconds", str(args.seconds),
                   "--nodes", "128", "--width", "14", "--boundary", "14", "--entries", "2000000", "--enumerate-limit", "12"]
            row = {**fixture, "backend_requested": backend, "snapshot_sha256": sha256(args.out / fixture["file"]), "command": cmd}
            try:
                run = subprocess.run(cmd, capture_output=True, text=True, timeout=args.seconds+3)
                row.update(exit_code=run.returncode)
                (args.out / (name+".stderr")).write_text(run.stderr)
                if run.returncode == 0:
                    row["result"] = json.loads(output.read_text())
                    result = row["result"]
                    if not result["feasible"] or not result["lower_ticks"] <= result["base_ticks"]+fixture["expected_gain"] <= result["upper_ticks"]:
                        row["invariant_failure"] = True
                    if result["exact"] and result["lower_ticks"]-result["base_ticks"] != fixture["expected_gain"]:
                        row["invariant_failure"] = True
                else:
                    row["error"] = run.stderr
            except subprocess.TimeoutExpired:
                row["error"] = "external diagnostic timeout; no complete result"
            rows.append(row)
    atomic_json(args.out / "results.json", {"schema": "barr_controlled_mechanisms_v1", "binary_sha256": sha256(args.binary),
        "synthetic_only": True, "comparison_is_frozen_kernel_not_end_to_end_solver": True, "rows": rows})
    failed = any(row.get("invariant_failure") for row in rows)
    print(json.dumps({"rows": len(rows), "invariant_failure": failed, "results": str(args.out / "results.json")}, indent=2))
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
