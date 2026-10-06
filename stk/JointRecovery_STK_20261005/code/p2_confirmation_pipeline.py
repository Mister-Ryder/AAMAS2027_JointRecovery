"""Wait for sealed P1 development, then run every declared P2 source cell.

Only the existing benchmark driver's evaluate stage is dispatched. No fitting,
fixed-call replay or recalibration is repeated; failed validation results never
filter the test sources or alter the frozen deadlines.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import p1_benchmark_driver as benchmark

SCHEMA = "joint_recovery_stk_p2_confirmation_pipeline_v1"
WORKPOINTS = [10, 50, 200, 1000]
BLOCKS = (("validation", [6, 7]), ("test", [8, 9, 10, 11]))
EXPECTED_ALL_GRAPHS = 36
EXPECTED_ALL_CELLS = 648
THREAD_ENV = dict(benchmark.THREAD_ENVIRONMENT, PYTHONUNBUFFERED="1")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    benchmark.write_json(path, value)


def progress(out, status, **fields):
    record = dict(schema=SCHEMA, status=status, updated_utc=utc_now(), **fields)
    write_json(out / "progress.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def under(path, root):
    path.resolve().relative_to(root.resolve())


def wait_for_followup(args, out):
    completion_path = args.followup_root / "completion.json"
    began = time.monotonic()
    progress(out, "waiting_for_completed_p1_followup", completion_path=str(completion_path),
             no_repeated_fit_fixed_calls_or_calibration=True)
    while not completion_path.is_file():
        if time.monotonic() - began > args.wait_timeout_seconds:
            raise TimeoutError("P1 followup did not seal before wait timeout; no P2 run started")
        time.sleep(args.poll_seconds)
    completed = read_json(completion_path)
    if completed.get("status") != "ALL_CORE_P1_DEVELOPMENT_COMPARISONS_COMPLETE":
        raise ValueError("P1 followup completion status is not the sealed complete development stage")
    budget_path = args.followup_root / "calibration" / "frozen_budgets.json"
    fixed_summary = args.followup_root / "fixed_calls" / "summary.json"
    if completed.get("fit_completion_sha256") != sha(args.fit_root / "completion.json"):
        raise ValueError("Followup completion no longer binds the four completed fits")
    if completed.get("frozen_budgets_sha256") != sha(budget_path):
        raise ValueError("Followup frozen budget binding differs")
    if completed.get("fixed_call_summary_sha256") != sha(fixed_summary):
        raise ValueError("Completed fixed-call result binding differs; do not repeat it")
    fixed = read_json(fixed_summary)
    if fixed.get("status") != "FIXED_CALL_COLD_CLONE_LABEL_REPLAY_COMPLETE" or fixed.get("sources") != [4, 5]:
        raise ValueError("P1 development fixed-call stage is incomplete or uses a different split")
    fit_receipt = benchmark.sealed_fit_receipt(args.fit_root)
    for record in fit_receipt["checkpoints"]:
        if sha(record["path"]) != record["sha256_from_fit_completion"]:
            raise ValueError("A sealed P1 final checkpoint changed")
    frozen = read_json(budget_path)
    budgets, budget_binding = benchmark.load_budgets(budget_path, fit_receipt, WORKPOINTS, 17)
    calibration_root = args.followup_root / "calibration"
    rows = read_json(calibration_root / "actual_run_rows.json")
    calibration_summary = read_json(calibration_root / "summary.json")
    if (len(rows) != 72 or calibration_summary.get("expected_cells") != 72 or
            calibration_summary.get("received_cells") != 72 or calibration_summary.get("stage") != "calibrate"):
        raise ValueError("The original complete 72-cell caller calibration is required, never repeated here")
    # Bind the executed development CLI and its actual-policy dependency to the
    # files this pipeline will invoke. Adding this wrapper cannot change either.
    calibration_cli_paths = {row["command"][1] for row in rows}
    if len(calibration_cli_paths) != 1:
        raise ValueError("Calibration used multiple actual-policy CLI sources")
    calibration_cli = Path(next(iter(calibration_cli_paths)))
    actual_cli = Path(__file__).resolve().with_name("p1_fit_and_allocate.py")
    actual_policy = actual_cli.with_name("p1_actual_policy.py")
    if sha(calibration_cli) != sha(actual_cli) or sha(calibration_cli.with_name("p1_actual_policy.py")) != sha(actual_policy):
        raise ValueError("P2 must retain the exact common execution code used by development calibration")
    return dict(followup_completion_sha256=sha(completion_path), fit=fit_receipt,
                frozen_budget_binding=budget_binding, frozen_budgets=budgets,
                calibration_rows_sha256=sha(calibration_root / "actual_run_rows.json"),
                completed_fixed_call_summary_sha256=sha(fixed_summary),
                actual_cli=str(actual_cli), actual_cli_sha256=sha(actual_cli),
                actual_policy_sha256=sha(actual_policy), calibration_cli=str(calibration_cli),
                budget_file=str(budget_path), native_workpoints_ms=WORKPOINTS,
                action_seed=frozen["action_seed"], no_repeated_development_experiments=True)


def wait_for_graphs(args, out):
    graphs_dir = args.dataset_root / "graphs"
    began = time.monotonic()
    last_error = None
    progress(out, "waiting_for_all_predeclared_p2_graphs", required_sources=[6, 7, 8, 9, 10, 11],
             required_npz_json_pairs=EXPECTED_ALL_GRAPHS, graphs_dir=str(graphs_dir),
             no_opportunity_or_winner_selection=True)
    while True:
        try:
            graphs = benchmark.graphs_for_sources(graphs_dir, [6, 7, 8, 9, 10, 11])
            if len(graphs) != EXPECTED_ALL_GRAPHS:
                raise ValueError("P2 requires exactly 36 declared NPZ/JSON graph pairs")
            for graph in graphs:
                if sha(graph["graph_path"]) != graph["graph_sha256_from_metadata"]:
                    raise ValueError("Graph NPZ transfer is incomplete or differs from metadata: " + graph["graph_id"])
            return graphs
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
            last_error = str(error)
        if time.monotonic() - began > args.wait_timeout_seconds:
            raise TimeoutError("Declared P2 graph wait timed out: " + str(last_error))
        time.sleep(args.poll_seconds)


def run_block(name, sources, args, out, prerequisites):
    block_out = out / name
    driver = Path(__file__).resolve().with_name("p1_benchmark_driver.py")
    expected_cells = len(sources) * 6 * 6 * 3
    command = [sys.executable, str(driver), "evaluate", "--dataset-root", str(args.dataset_root),
               "--runtime-root", str(args.runtime_root), "--fit-root", str(args.fit_root),
               "--graphs-dir", str(args.dataset_root / "graphs"),
               "--actual-cli", prerequisites["actual_cli"], "--chils", str(args.chils),
               "--chils-source", str(args.chils_source), "--budgets-file", prerequisites["budget_file"],
               "--budgets-ms", "10", "50", "200", "1000", "--action-seed", "17",
               "--sources", *map(str, sources), "--out", str(block_out)]
    stdout_path, stderr_path = out / (name + ".stdout.log"), out / (name + ".stderr.log")
    began = time.monotonic()
    environment = dict(os.environ)
    environment.update(THREAD_ENV)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment,
                                   stdin=subprocess.DEVNULL)
        progress(out, "evaluating_" + name, sources=sources, expected_cells=expected_cells,
                 child_pid=process.pid, driver_progress=str(block_out / "progress.json"),
                 native_workpoints_ms=WORKPOINTS, no_filtered_graph_or_policy=True,
                 reused_budget_file_sha256=prerequisites["frozen_budget_binding"]["sha256"])
        returncode = process.wait()
    summary_path = block_out / "summary.json"
    rows_path = block_out / "actual_run_rows.json"
    summary = read_json(summary_path) if summary_path.is_file() else None
    rows = read_json(rows_path) if rows_path.is_file() else []
    failed_or_unknown = sum(bool(row.get("failure_or_unknown_outcome")) for row in rows)
    native_failed = sum(row.get("stop_reason") == "failed" for row in rows)
    complete_matrix = bool(summary and summary.get("status") == "ACTUAL_SERIAL_MATRIX_COMPLETE" and
                           summary.get("expected_cells") == expected_cells and len(rows) == expected_cells)
    receipt = dict(block=name, sources=sources, expected_cells=expected_cells, received_cells=len(rows),
                   complete_declared_matrix=complete_matrix, driver_returncode=returncode,
                   failed_or_unknown_runs=failed_or_unknown, native_failed_stop_runs=native_failed,
                   missed_return_runs=sum(row.get("missed_return_sample") is True for row in rows),
                   zero_delivered_gain_runs=sum(row.get("strict_delivered_gain_including_failed_runs_seconds") == 0
                                               for row in rows),
                   elapsed_seconds=time.monotonic() - began, command=command,
                   stdout=str(stdout_path), stderr=str(stderr_path),
                   rows_sha256=sha(rows_path) if rows_path.is_file() else None,
                   summary_sha256=sha(summary_path) if summary_path.is_file() else None,
                   all_declared_outcomes_preserved=True,
                   requires_failure_status=bool(returncode != 0 or not complete_matrix or failed_or_unknown or native_failed))
    write_json(out / (name + "_completion.json"), receipt)
    return receipt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "runtime-root", "fit-root", "followup-root", "out", "chils", "chils-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.)
    parser.add_argument("--wait-timeout-seconds", type=float, default=86400.)
    parser.add_argument("--cpu-index", type=int, default=0,
                        help="Choose one CPU from this job's inherited affinity; never modify another job")
    args = parser.parse_args()
    for name in ("dataset_root", "runtime_root", "fit_root", "followup_root", "out", "chils", "chils_source"):
        setattr(args, name, getattr(args, name).resolve())
    if any(not math.isfinite(v) or v <= 0 for v in (args.poll_seconds, args.wait_timeout_seconds)):
        parser.error("Positive finite polling and wait timeout required")
    if args.poll_seconds > 60:
        parser.error("Poll interval must not exceed 60 seconds")
    if args.cpu_index < 0:
        parser.error("Nonnegative CPU index required")
    under(args.out, args.dataset_root)
    for source in (args.fit_root, args.followup_root, args.dataset_root / "graphs"):
        if args.out == source or source in args.out.parents:
            parser.error("Pipeline output must not be inside existing fit/followup/graph inputs")
    return args


def main():
    args = parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("P2 pipeline requires a fresh independent output directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    block_receipts = []
    try:
        prerequisites = wait_for_followup(args, out)
        graphs = wait_for_graphs(args, out)
        if hasattr(os, "sched_getaffinity"):
            available = sorted(os.sched_getaffinity(0))
            cpu = available[args.cpu_index % len(available)]
            os.sched_setaffinity(0, {cpu})
            cpu_affinity = [cpu]
        else:
            cpu_affinity = None
        protocol = dict(schema=SCHEMA, frozen_utc=utc_now(), code_sha256=sha(__file__),
                        driver_sha256=sha(Path(__file__).resolve().with_name("p1_benchmark_driver.py")),
                        prerequisites=prerequisites, graphs=graphs,
                        blocks=[dict(name=name, sources=sources, expected_cells=len(sources) * 108)
                                for name, sources in BLOCKS],
                        expected_all_graphs=EXPECTED_ALL_GRAPHS, expected_all_cells=EXPECTED_ALL_CELLS,
                        native_workpoints_ms=WORKPOINTS, policies=benchmark.POLICIES,
                        CPU_affinity=cpu_affinity, thread_environment=THREAD_ENV,
                        within_each_block_serial_CPU_one=True, validation_then_test=True,
                        no_retraining_no_recalibration_no_result_repair=True,
                        validation_failure_does_not_cancel_or_filter_test=True,
                        comparison_units_are_physical_source_groups=True,
                        changed_dates_or_seeds_do_not_prove_statistical_independence=True)
        write_json(out / "protocol.json", protocol)
        for name, sources in BLOCKS:
            # A validation nonzero result is recorded, never passed as check=True
            # and never used to skip/resize the predeclared test block.
            try:
                receipt = run_block(name, sources, args, out, prerequisites)
            except Exception as error:
                receipt = dict(block=name, sources=sources, expected_cells=len(sources) * 108,
                               complete_declared_matrix=False, requires_failure_status=True,
                               failure=dict(type=type(error).__name__, message=str(error)),
                               no_retry_or_result_repair=True)
                write_json(out / (name + "_completion.json"), receipt)
            block_receipts.append(receipt)
        unchanged_budget = sha(prerequisites["budget_file"]) == prerequisites["frozen_budget_binding"]["sha256"]
        unchanged_fit = sha(args.fit_root / "protocol.json") == prerequisites["fit"]["fit_protocol_sha256"]
        failed = any(receipt["requires_failure_status"] for receipt in block_receipts) or not unchanged_budget or not unchanged_fit
        completion = dict(schema=SCHEMA, status="complete_with_failures" if failed else "complete",
                          pipeline_completed=True, completed_utc=utc_now(),
                          elapsed_seconds=time.monotonic() - started, blocks=block_receipts,
                          all_declared_cells_completed=all(r["complete_declared_matrix"] for r in block_receipts),
                          expected_all_cells=EXPECTED_ALL_CELLS, received_cells=sum(r.get("received_cells", 0) for r in block_receipts),
                          original_frozen_budget_unchanged=unchanged_budget, original_fit_protocol_unchanged=unchanged_fit,
                          no_repeated_development_fixed_calls_or_72_calibration=True,
                          validation_failed_results_did_not_filter_test=True,
                          protocol_sha256=sha(out / "protocol.json"), all_zero_late_failed_results_preserved=True)
        write_json(out / "completion.json", completion)
        progress(out, completion["status"], completion_file=str(out / "completion.json"),
                 expected_cells=EXPECTED_ALL_CELLS, received_cells=completion["received_cells"])
        return 1 if failed else 0
    except Exception as error:
        failure = dict(schema=SCHEMA, status="blocked_before_or_during_prerequisites", completed=False,
                       failure=dict(type=type(error).__name__, message=str(error)),
                       elapsed_seconds=time.monotonic() - started, existing_results_not_repaired=True,
                       no_repeated_development_experiments=True)
        write_json(out / "failure.json", failure)
        progress(out, failure["status"], failure_file=str(out / "failure.json"))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
