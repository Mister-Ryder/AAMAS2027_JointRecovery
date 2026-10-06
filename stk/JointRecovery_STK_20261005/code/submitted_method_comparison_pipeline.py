"""Append full-controller and published whole-graph comparisons without changing P2.

All cells are serial, use the four existing fits and the existing three caller
deadlines, and preserve failed or late outcomes. No outcome selects a source,
checkpoint, deadline, request menu, or algorithm configuration.
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

import p1_benchmark_driver as base

SCHEMA = "joint_recovery_stk_submitted_comparison_v1"
BLOCKS = (("validation", (6, 7)), ("test", (8, 9, 10, 11)))
POLICIES = (("FullCapacity-seed17", "FullCapacity", 17),
            ("FullCapacity-seed29", "FullCapacity", 29),
            ("FullCheapSummary-seed17", "FullCheapSummary", 17),
            ("FullCheapSummary-seed29", "FullCheapSummary", 29),
            ("FullP1", "FullP1", 17), ("FullGreedy", "FullGreedy", 17),
            ("CHILS-p1", "CHILS-p1", 17), ("HiGHS-MILP", "HiGHS-MILP", 17))
FIELDS = tuple(dict.fromkeys(base.METRIC_FIELDS + (
    "initial_mask_sha256", "strict_on_time_gain_beyond_paid_shared_prefix_seconds",
    "completed_requests", "actual_native_calls", "observed_completed_native_calls",
    "requests_without_observed_completed_response")))
EXPECTED_CELLS = 864


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    base.write_json(path, value)


def read(path):
    return base.read_json(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wait_complete(root, filename, timeout, label):
    began = time.monotonic()
    path = root / filename
    while not path.is_file():
        if time.monotonic() - began > timeout:
            raise TimeoutError(label + " did not finish; no comparison matrix started")
        time.sleep(15)
    return read(path)


def command_for(cell, args, run_dir):
    common = ["--runtime-root", str(args.runtime_root), "--graph", cell["graph_path"],
              "--out", str(run_dir), "--deadline-seconds", format(cell["deadline_seconds"], ".17g"),
              "--chils", str(args.chils), "--chils-source", str(args.chils_source)]
    if cell["policy"].startswith("Full"):
        command = [sys.executable, str(args.full_cli), *common,
                   "--fit-root", str(args.fit_root), "--policy", cell["policy"],
                   "--fit-seed", str(cell["fit_seed"]), "--budgets-ms", "10", "50", "200", "1000",
                   "--max-calls", "8", "--action-seed", "17", "--cpu-index", "0"]
        return command, run_dir / "full_joint_recovery.json", "joint_recovery_stk_full_actual_v1"
    command = [sys.executable, str(args.published_cli), *common, "--method", cell["policy"],
               "--budgets-file", str(args.budgets_file)]
    if cell["policy"] == "CHILS-p1":
        command += ["--numeric-gate", str(args.numeric_root / "gates" / (cell["graph_id"] + ".json"))]
    return command, run_dir / "actual_policy.json", "joint_recovery_stk_published_wholegraph_actual_v1"


def execute(cell, args, block_root):
    run_dir = block_root / "runs" / cell["run_id"]
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    logs = block_root / "logs"
    logs.mkdir(exist_ok=True)
    stdout_path, stderr_path = logs / (cell["run_id"] + ".stdout.txt"), logs / (cell["run_id"] + ".stderr.txt")
    if run_dir.exists():
        raise FileExistsError("Never replace an existing declared cell: " + str(run_dir))
    command, report_path, expected_schema = command_for(cell, args, run_dir)
    environment = dict(os.environ, **base.THREAD_ENVIRONMENT, PYTHONUNBUFFERED="1")
    started = time.monotonic()
    row = dict(cell, command=command, started_utc=now(), actual_policy_path=str(report_path),
               stdout_path=str(stdout_path), stderr_path=str(stderr_path),
               thread_environment=base.THREAD_ENVIRONMENT, process_watchdog_timeout=False)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment,
                                       start_new_session=(os.name == "posix"), stdin=subprocess.DEVNULL)
            try:
                row["cli_exit_code"] = process.wait(timeout=cell["deadline_seconds"] + 120.)
            except subprocess.TimeoutExpired:
                base.kill_owned_process(process)
                row.update(cli_exit_code=process.returncode, process_watchdog_timeout=True)
        except OSError as exc:
            row.update(cli_exit_code=None, launch_error=str(exc))
    row["whole_cli_wall_seconds_including_resident_setup"] = time.monotonic() - started
    row["report_state"] = "missing"
    if report_path.is_file():
        try:
            report = read(report_path)
            g = report.get("graph", {})
            row["report_identity_matches_cell"] = bool(
                report.get("schema") == expected_schema and
                g.get("graph_id") == cell["graph_id"] and g.get("source_group") == cell["source_group"] and
                g.get("npz_sha256") == cell["graph_sha256_from_metadata"] and
                report.get("policy") == cell["policy"] and report.get("fit_seed") == cell["fit_seed"] and
                report.get("deadline_seconds") == cell["deadline_seconds"])
            row.update(actual_report_schema=report.get("schema"), actual_report_status=report.get("status"),
                       report_sha256=sha(report_path), report_state="received")
            row.update({field: report.get(field) for field in FIELDS})
            row["resident_setup_fields"] = {k: v for k, v in report.items() if "setup" in k}
            row["event_count"] = len(report.get("events", []))
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            row.update(report_state="invalid", report_read_error=str(exc))
    row["report_usable"] = bool(row["report_state"] == "received" and row["cli_exit_code"] == 0 and
                               not row["process_watchdog_timeout"] and row.get("report_identity_matches_cell") and
                               base.finite_number(row.get("external_caller_observed_return_seconds")) and
                               base.finite_number(row.get("actual_gain_seconds")) and
                               isinstance(row.get("initial_mask_sha256"), str) and
                               len(row["initial_mask_sha256"]) == 64)
    strict = row.get("strict_on_time_gain_seconds")
    row["strict_delivered_gain_including_failed_runs_seconds"] = float(strict) if row["report_usable"] and base.finite_number(strict) else 0.
    row["failure_or_unknown_outcome"] = not row["report_usable"]
    write(block_root / "receipts" / (cell["run_id"] + ".json"), row)
    return row


def summarize(rows, sources):
    metrics = ("strict_delivered_gain_including_failed_runs_seconds", "returned_value_seconds",
               "initial_value_seconds", "actual_gain_seconds", "paid_shared_prefix_gain_seconds",
               "actual_gain_beyond_paid_shared_prefix_seconds", "external_caller_observed_return_seconds")
    summary = []
    for budget in ("wide", "half", "quarter"):
        for label, _, _ in POLICIES:
            selected = [r for r in rows if r["budget_id"] == budget and r["policy_label"] == label]
            source_rows = []
            for source in sources:
                current = [r for r in selected if r["source"] == source]
                means = {m: sum(r[m] for r in current) / len(current)
                         if current and all(base.finite_number(r.get(m)) for r in current) else None for m in metrics}
                source_rows.append(dict(source=source, source_group="JR-SOURCE-r%03d" % source,
                                        declared_runs=6, received_runs=len(current), metrics=means,
                                        failure_or_unknown_runs=sum(r["failure_or_unknown_outcome"] for r in current),
                                        missed_return_runs=sum(r.get("missed_return_sample") is True for r in current),
                                        native_failed_runs=sum(r.get("stop_reason") == "failed" for r in current)))
            mean = {m: sum(s["metrics"][m] for s in source_rows) / len(source_rows)
                    if all(s["metrics"][m] is not None for s in source_rows) else None for m in metrics}
            summary.append(dict(budget_id=budget, policy_label=label, source_values=source_rows,
                                equal_physical_source_mean=mean, physical_source_count=len(sources)))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "runtime-root", "fit-root", "budgets-file", "original-p2-root",
                 "numeric-root", "extension-protocol", "chils", "chils-source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--full-cli", type=Path, default=Path(__file__).with_name("full_joint_recovery_deadline.py"))
    parser.add_argument("--published-cli", type=Path, default=Path(__file__).with_name("published_wholegraph_deadline.py"))
    parser.add_argument("--wait-timeout-seconds", type=float, default=86400.)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.out.relative_to(args.dataset_root)
    if not math.isfinite(args.wait_timeout_seconds) or args.wait_timeout_seconds <= 0:
        parser.error("Positive wait timeout required")
    return args


def main():
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh extension job; never overwrite partial original or extension results")
    args.out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    write(args.out / "progress.json", dict(schema=SCHEMA, status="waiting_for_original_p2_and_numeric_gate", updated_utc=now()))
    original = wait_complete(args.original_p2_root, "completion.json", args.wait_timeout_seconds, "Original P2")
    if original.get("pipeline_completed") is not True:
        raise ValueError("Original P2 has not sealed; do not contend with its serial timing experiment")
    guard = wait_complete(args.numeric_root, "completion.json", args.wait_timeout_seconds, "New CHILS numeric scope")
    if guard.get("status") != "WHOLEGRAPH_NUMERIC_GUARD_PASS" or guard.get("pass_graphs") != 48:
        raise ValueError("Published CHILS requires actual new-domain numeric clearance; original results preserved")
    fit = base.sealed_fit_receipt(args.fit_root)
    for checkpoint in fit["checkpoints"]:
        if sha(checkpoint["path"]) != checkpoint["sha256_from_fit_completion"]:
            raise ValueError("A checkpoint differs from its frozen fit")
    budgets, binding = base.load_budgets(args.budgets_file, fit, [10, 50, 200, 1000], 17)
    proposal = read(args.extension_protocol)
    if proposal["frozen_budgets_sha256"] != binding["sha256"] or proposal["fit_completion_sha256"] != fit["completion_sha256"]:
        raise ValueError("Extension inputs differ from the protocol frozen without inspecting test outcomes")
    graphs = base.graphs_for_sources(args.dataset_root / "graphs", [6, 7, 8, 9, 10, 11])
    if len(graphs) != 36 or any(sha(g["graph_path"]) != g["graph_sha256_from_metadata"] for g in graphs):
        raise ValueError("Every original validation/test graph is required, without result-dependent selection")
    for cli in (args.full_cli, args.published_cli, Path(__file__)):
        expected = proposal["code_sha256"][cli.name]
        if sha(cli) != expected:
            raise ValueError("Comparison implementation differs from the separate frozen extension protocol")
    if hasattr(os, "sched_getaffinity"):
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
        affinity = [cpu]
    else:
        affinity = None
    protocol = dict(schema=SCHEMA, frozen_utc=now(), extension_proposal_sha256=sha(args.extension_protocol),
                    original_p2_completion_sha256=sha(args.original_p2_root / "completion.json"),
                    numeric_gate_completion_sha256=sha(args.numeric_root / "completion.json"),
                    fit=fit, budgets=binding, graphs=graphs, policies=POLICIES,
                    expected_cells=EXPECTED_CELLS, cpu_affinity=affinity,
                    full_max_calls=8, full_remaining_head=0, original_p2_changed=False,
                    no_retraining_no_recalibration_no_result_selection=True)
    write(args.out / "protocol.json", protocol)
    # One fixed development integration check, never used to choose a model,
    # source, deadline or algorithm setting; original P2 has already stopped.
    development = base.graphs_for_sources(args.dataset_root / "graphs", [4])[0]
    smoke_rows = []
    for label, policy in (("FullCapacity-seed17", "FullCapacity"), ("HiGHS-MILP", "HiGHS-MILP")):
        cell = dict(development, **budgets[0], policy_label=label, policy=policy, fit_seed=17,
                    run_id=development["graph_id"] + "__wide__" + label + "__integration_smoke")
        smoke_rows.append(execute(cell, args, args.out / "development_integration_smoke"))
    write(args.out / "development_integration_smoke" / "actual_run_rows.json", smoke_rows)
    smoke_pass = bool(all(r["report_usable"] and r.get("stop_reason") != "failed" for r in smoke_rows) and
                      len({r.get("initial_mask_sha256") for r in smoke_rows}) == 1)
    write(args.out / "development_integration_smoke" / "completion.json",
          dict(status="INTEGRATION_CONTRACT_PASS" if smoke_pass else "INTEGRATION_CONTRACT_FAILED",
               all_original_outputs_preserved=True, fixed_development_graph=development["graph_id"],
               used_for_performance_selection=False))
    if not smoke_pass:
        raise RuntimeError("Fixed development integration failed; no holdout extension cells started")
    total, block_records = 0, []
    for block, sources in BLOCKS:
        block_root = args.out / block
        block_root.mkdir()
        cells = []
        for g in graphs:
            if g["source"] not in sources:
                continue
            for budget in budgets:
                for label, policy, seed in POLICIES:
                    cells.append(dict(g, **budget, policy_label=label, policy=policy, fit_seed=seed,
                                      run_id=g["graph_id"] + "__" + budget["budget_id"] + "__" + label))
        write(block_root / "declared_matrix.json", cells)
        rows = []
        for cell in cells:
            try:
                row = execute(cell, args, block_root)
            except Exception as exc:
                row = dict(cell, failure_or_unknown_outcome=True, report_usable=False,
                           strict_delivered_gain_including_failed_runs_seconds=0.,
                           failure=dict(type=type(exc).__name__, message=str(exc)))
                write(block_root / "receipts" / (cell["run_id"] + ".json"), row)
            rows.append(row)
            total += 1
            write(block_root / "actual_run_rows.json", rows)
            write(args.out / "progress.json", dict(schema=SCHEMA, status="evaluating_" + block,
                  updated_utc=now(), completed_cells=total, expected_cells=EXPECTED_CELLS,
                  block_completed_cells=len(rows), block_expected_cells=len(cells),
                  failed_or_unknown_runs=sum(r["failure_or_unknown_outcome"] for r in rows)))
        # Verify the common initial S across policies without suppressing any run.
        starts = {}
        for row in rows:
            mask = row.get("initial_mask_sha256")
            if mask:
                starts.setdefault(row["graph_id"], set()).add(mask)
        mismatch = {g: sorted(v) for g, v in starts.items() if len(v) != 1}
        failed = sum(r["failure_or_unknown_outcome"] for r in rows)
        summary = dict(schema=SCHEMA, status="SUBMITTED_COMPARISON_MATRIX_COMPLETE", block=block,
                       sources=list(sources), expected_cells=len(cells), received_cells=len(rows),
                       failed_or_unknown_runs=failed, initial_mask_mismatches=mismatch,
                       source_equal_summary=summarize(rows, sources), all_failed_late_zero_results_preserved=True)
        write(block_root / "summary.json", summary)
        block_records.append(dict(block=block, expected_cells=len(cells), received_cells=len(rows),
                                  failed_or_unknown_runs=failed, initial_mask_mismatches=mismatch,
                                  rows_sha256=sha(block_root / "actual_run_rows.json")))
    unchanged = sha(args.budgets_file) == binding["sha256"] and sha(args.fit_root / "completion.json") == fit["completion_sha256"]
    status = "complete" if unchanged and not any(b["failed_or_unknown_runs"] or b["initial_mask_mismatches"] for b in block_records) else "complete_with_failures"
    write(args.out / "completion.json", dict(schema=SCHEMA, status=status, pipeline_completed=True,
          completed_utc=now(), expected_cells=EXPECTED_CELLS, received_cells=total, blocks=block_records,
          frozen_fit_and_budgets_unchanged=unchanged, elapsed_seconds=time.monotonic() - began))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
