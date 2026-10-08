#!/usr/bin/env python3
"""Single run, with the SAME end-to-end accounting for BARR and external CHILS.

The allowance starts before graph loading, shared initialization and serialization.
Native final output, external fallback and late/failed outcomes are separate fields.
This is an ordinary measured subprocess allowance, not a physical hard-real-time proof.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import resource
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from barr_io import load_npz, atomic_json, sha256, write_native, write_metis


def executor_contract(args) -> dict:
    return {"algorithm": "BARR-0.3", "backend": args.recovery_backend,
            "kernel_nodes": args.kernel_nodes, "kernel_seconds": args.kernel_seconds,
            "tick_seconds": args.tick,
            "factor_width": args.factor_width, "factor_boundary": args.factor_boundary,
            "factor_entries": args.factor_entries, "decompose": not args.no_decompose,
            "greedy_repair": args.greedy_repair}


def execute(args) -> dict:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = {"schema": "barr_run_v1", "method": args.method, "seed": args.seed,
            "allowance_seconds": args.seconds, "threads": args.threads, "start": args.start,
            "complete_native_quality": None, "fallback_quality": None, "on_time": False,
            "status": "ERROR", "error": None, "command": None, "native": None,
            "protocol": "load-initialize-export-launch-parse-validate charged; report-write separately"}
    g = None
    binary = Path(args.chils if args.method == "chils" else args.binary).resolve()
    try:
        if not math.isfinite(args.seconds) or not 0 < args.seconds <= 31536000:
            raise ValueError("seconds must be finite and in (0,31536000]")
        g = load_npz(Path(args.graph), args.start)
        data.update(g.metadata)
        data.update(graph_sha256=sha256(Path(args.graph)), vertices=g.n, edges=len(g.edges),
                    initial_quality=g.objective(g.initial), tick_seconds=args.tick,
                    binary_sha256=sha256(binary))
        ticks = g.ticks(args.tick)
        output = out / "native_result.json"
        checkpoint = out / "checkpoint.json"
        trace = out / "counterfactual.jsonl"
        if args.method == "chils":
            inp, warm, selected = out / "input.graph", out / "warm.sol", out / "native.sol"
            write_metis(inp, warm, g, ticks)
        else:
            inp = out / "input.barr"
            write_native(inp, g, ticks)
        # The same fixed rule for both methods. No historical 0.70*D comparison.
        reserve = max(.03, min(.5, .05 * args.seconds))
        native_budget = args.seconds - (time.perf_counter() - started) - reserve
        data["preparation_seconds"] = time.perf_counter() - started
        if native_budget <= 0:
            raise TimeoutError("allowance consumed before native launch")
        if args.method == "chils":
            cmd = [str(binary), "-g", str(inp), "-i", str(warm), "-o", str(selected),
                   "-p", str(args.chils_population), "-c", str(args.threads), "-r", str(args.seed),
                   "-s", str(args.chils_step), "-t", str(native_budget)]
        else:
            cmd = [str(binary), "--input", str(inp), "--output", str(output),
                   "--seconds", str(native_budget), "--seed", str(args.seed), "--threads", str(args.threads),
                   "--population", str(args.population), "--mode", args.method, "--rank", args.rank,
                   "--kernel-nodes", str(args.kernel_nodes), "--kernel-seconds", str(args.kernel_seconds),
                   "--challengers", str(args.challengers), "--proposals", str(args.proposals),
                   "--execute-top", str(args.execute_top), "--checkpoint", str(checkpoint),
                   "--events", str(out / "events.jsonl"),
                   "--recovery-backend", args.recovery_backend, "--factor-width", str(args.factor_width),
                   "--factor-boundary", str(args.factor_boundary), "--factor-entries", str(args.factor_entries)]
            data["recovery_backend"] = args.recovery_backend
            data["executor_contract"] = executor_contract(args)
            if args.snapshots:
                if str(g.metadata.get("split", "")).lower() not in ("train", "val", "dev"):
                    raise ValueError("mechanism snapshots require TRAIN/VAL/DEV, not TEST/unknown")
                cmd += ["--kernel-snapshots", str(out / "kernels")]
                data["diagnostic_snapshot_run"] = True
            if args.model:
                cmd += ["--model", str(Path(args.model).resolve())]
                data["model_sha256"] = sha256(Path(args.model))
                metadata_path = Path(str(args.model) + ".json")
                if metadata_path.is_file():
                    model_meta = json.loads(metadata_path.read_text())
                    if model_meta.get("executor_contract") != executor_contract(args):
                        raise ValueError("model executor/workpoint differs; recollect and retrain for v0.3 backend")
                    if model_meta.get("model_sha256") != data["model_sha256"]:
                        raise ValueError("model metadata hash mismatch")
                    source = g.metadata.get("source_group")
                    split = str(g.metadata.get("split", "")).upper()
                    forbidden = set(model_meta.get("train_sources", []))
                    if split == "TEST":
                        forbidden |= set(model_meta.get("validation_sources", []))
                    if split in ("TEST", "VAL", "DEV") and source in forbidden:
                        raise ValueError("model training/selection source overlaps evaluation source")
                    data["model_provenance_checked"] = True
                else:
                    data["model_provenance_checked"] = False
                    if str(g.metadata.get("split", "")).upper() == "TEST":
                        raise ValueError("TEST evaluation requires the exported model provenance .json sidecar")
            if args.trace:
                if str(g.metadata.get("split", "")).lower() not in ("train", "val", "dev"):
                    raise ValueError("counterfactual collection requires explicit TRAIN/VAL/DEV graph split, not TEST/unknown")
                cmd += ["--trace", str(trace)]
            for name in ("fixed_k", "random_domains", "no_decompose", "greedy_repair"):
                if getattr(args, name):
                    cmd.append("--" + name.replace("_", "-"))
        data["command"] = cmd
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
            wall_start = time.perf_counter()
            cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
            child = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
            try:
                child.wait(timeout=max(.001, args.seconds - (time.perf_counter() - started) - reserve / 2))
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
                raise TimeoutError("native process exceeded shared caller allowance")
            finally:
                if child.poll() is None:
                    child.kill(); child.wait()
            data["subprocess_seconds"] = time.perf_counter() - wall_start
            cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            data["subprocess_cpu_seconds"] = (cpu_after.ru_utime + cpu_after.ru_stime) - (cpu_before.ru_utime + cpu_before.ru_stime)
        if child.returncode != 0:
            raise RuntimeError(f"native return code {child.returncode}; see stderr.log")
        if args.method == "chils":
            if not selected.is_file():
                raise RuntimeError("CHILS produced no membership file")
            ids = [int(line) - 1 for line in selected.read_text().split()]
            native = {"selected": ids, "population": args.chils_population, "upstream_cli": True}
        else:
            native = json.loads(output.read_text())
            if native.get("schema") != "barr_result_v1":
                raise ValueError("unexpected native schema")
            ids = native["selected"]
        quality = g.objective(ids)
        native_ticks = sum(ticks[int(v)] for v in ids)
        if "tick_value" in native and native["tick_value"] != native_ticks:
            raise ValueError("native integer objective mismatch")
        # A quantized external solver may be microscopically worse in original units.
        # Preserve its true native quality but return the better feasible schedule.
        best_ids = ids if quality >= data["initial_quality"] else g.initial.tolist()
        data["native"] = native
        data["native_quality_observed"] = quality
        data["returned_selected"] = best_ids
        data["returned_quality"] = g.objective(best_ids)
        data["caller_observed_seconds"] = time.perf_counter() - started
        data["on_time"] = data["caller_observed_seconds"] <= args.seconds
        data["status"] = "COMPLETE" if data["on_time"] else "LATE"
        if data["on_time"]:
            data["complete_native_quality"] = quality
        if args.trace:
            atomic_json(out / "trace_metadata.json", {**g.metadata, "graph_sha256": data["graph_sha256"],
                        "binary_sha256": data["binary_sha256"], "seed": args.seed,
                        "trace_file": trace.name, "training_target": "bounded marginal gain beyond exact fusion; not optimal gain",
                        "collection_extra_time_not_deployment": True})
    except Exception as exc:
        data["error"] = f"{type(exc).__name__}: {exc}"
        data["status"] = "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR"
        if g is not None:
            fallback = g.initial.tolist()
            checkpoint = out / "checkpoint.json"
            try:
                if checkpoint.is_file():
                    trial = json.loads(checkpoint.read_text())["selected"]
                    if g.objective(trial) >= g.objective(fallback):
                        fallback = trial
            except (OSError, ValueError, KeyError, TypeError):
                pass
            data["returned_selected"] = fallback
            data["fallback_quality"] = g.objective(fallback)
            data["returned_quality"] = data["fallback_quality"]
        data["caller_observed_seconds"] = time.perf_counter() - started
        data["fallback_observed_before_deadline"] = g is not None and data["caller_observed_seconds"] <= args.seconds
    # Observe the decision, not merely native discovery. Report-file serialization is separate.
    report_start = time.perf_counter()
    atomic_json(out / "result.json", data)
    data["report_write_seconds"] = time.perf_counter() - report_start
    return data


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new output directory, must not already exist")
    p.add_argument("--method", choices=("barr", "fusion", "local", "chils"), default="barr")
    p.add_argument("--binary", type=Path, default=ROOT / "build/barr_solver")
    p.add_argument("--chils", type=Path, default=ROOT / "external/CHILS/CHILS")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--chils-population", type=int, default=16)
    p.add_argument("--chils-step", type=float, default=0.1, help="phase slice; tune only on DEV, not claimed published default")
    p.add_argument("--start", choices=("warm", "cold"), default="warm")
    p.add_argument("--tick", type=float, default=1e-6)
    p.add_argument("--rank", choices=("heuristic", "independent", "random", "gnn"), default="heuristic")
    p.add_argument("--model", type=Path)
    p.add_argument("--recovery-backend", choices=("hybrid", "factor", "branch"), default="hybrid")
    p.add_argument("--factor-width", type=int, default=10)
    p.add_argument("--factor-boundary", type=int, default=10)
    p.add_argument("--factor-entries", type=int, default=262144)
    p.add_argument("--snapshots", action="store_true", help="export lossless frozen kernels; diagnostic overhead is charged")
    p.add_argument("--kernel-nodes", type=int, default=128)
    p.add_argument("--kernel-seconds", type=float, default=.03)
    p.add_argument("--challengers", type=int, default=12)
    p.add_argument("--proposals", type=int, default=6)
    p.add_argument("--execute-top", type=int, default=2)
    for opt in ("fixed-k", "random-domains", "no-decompose", "greedy-repair", "trace"):
        p.add_argument("--" + opt, action="store_true")
    return p

if __name__ == "__main__":
    args = parser().parse_args()
    result = execute(args)
    print(json.dumps({k: v for k, v in result.items() if k not in ("native", "returned_selected")}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "COMPLETE" else 2)
