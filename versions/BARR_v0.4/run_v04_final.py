"""Explicit frozen v0.4 full protocol. Native source and tested binaries are unchanged."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess

SOLVER_SHA = "8dba1d02ad3da42f49d3a9b6d0858a6bdbdffb0e28c9fa41e4f5130ab0bab656"
OPTIONS = [
    "--threads", "1", "--population", "4", "--mode", "pair", "--rank", "heuristic",
    "--recovery-backend", "hybrid", "--kernel-nodes", "128", "--kernel-seconds", "0.03",
    "--challengers", "12", "--proposals", "6", "--execute-top", "2", "--factor-width", "10",
    "--factor-boundary", "10", "--factor-entries", "262144", "--local-iterations", "64",
    "--local-seconds", "0.025", "--gate-fraction", "0.05", "--gate-warmup", "30",
    "--gate-cooldown", "0.5", "--event-stale", "8", "--pair-slice", "0.5",
    "--pair-max-seeds", "0", "--pair-fusion-every", "4", "--pair-policy", "fusion-refine",
    "--pair-component", "full",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Frozen v0.4 full algorithm; native search defaults to360 seconds")
    parser.add_argument("--input", type=Path, required=True, help="Verified BARR1 input including the frozen common initial mask")
    parser.add_argument("--run-dir", type=Path, required=True, help="New output directory; existing evidence is never overwritten")
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parent / "source/BARR_v0.4-final/build/barr_solver")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seconds", type=float, default=360.)
    parser.add_argument("--cpu", type=int, help="Bind the whole single-thread run to one allowed server core")
    parser.add_argument("--dry-run", action="store_true", help="Validate frozen binary and display command without any search or writes")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 360.:
        parser.error("The final research protocol requires at least360 native search seconds")
    if args.seed < 0 or (args.cpu is not None and args.cpu < 0):
        parser.error("Seed and CPU must be nonnegative")
    binary, graph, output = args.binary.resolve(), args.input.resolve(), args.run_dir.resolve()
    if not graph.is_file() or not binary.is_file():
        parser.error("Input and the frozen Linux binary must exist")
    if digest(binary) != SOLVER_SHA:
        parser.error("Binary differs from the actual tested and archived E-full executable")
    if output.exists():
        parser.error("Choose a fresh run directory to preserve existing results")
    command = [str(binary), "--input", str(graph), "--output", str(output / "native_result.json"),
               "--seconds", str(args.seconds), "--seed", str(args.seed), "--events", str(output / "events.jsonl"),
               "--checkpoint", str(output / "checkpoint.json")] + OPTIONS
    if args.dry_run:
        print(json.dumps(dict(command=command, binary_sha256=SOLVER_SHA, native_search_started=False), indent=2))
        return
    if not hasattr(os, "sched_setaffinity"):
        parser.error("Execute the tested Linux binary on the server; this machine may inspect it with --dry-run")
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):
            parser.error("Requested CPU is outside this process's allowed affinity")
        os.sched_setaffinity(0, {args.cpu})
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(command=command, binary_sha256=SOLVER_SHA, native_input_sha256=digest(graph),
                   seconds=args.seconds, threads=1, population=4, mode="pair", pair_component="full",
                   cpu=args.cpu, diagnostic_snapshots=False, preparation_separate=True,
                   binary_and_native_source_unchanged=True)
    (output / "launch_protocol.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
        completed = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr,
                                   stdin=subprocess.DEVNULL, check=False)
    print(json.dumps(dict(returncode=completed.returncode, output_directory=str(output))))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
