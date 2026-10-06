"""Serial actual-policy calibration/evaluation for the new STK dataset.

Every matrix cell is a fresh CLI caller with a freshly paid common warm prefix.
No P0 outcomes are used to select a graph, request, policy or deadline. The
driver imports no model code; it schedules the sealed actual-policy CLI only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = "joint_recovery_stk_p1_actual_benchmark_v1"
CALIBRATION_SOURCES = (4, 5)
CALIBRATION_CAP_SECONDS = 60.0
DELIVERY_MARGIN_MULTIPLIER = 1.2
MAX_CALLS = 4
EXPECTED_CONFIGS = {(view, gap, 150) for view in ("R8", "R12") for gap in (170, 340, 680)}
POLICIES = (("Capacity-seed17", "ResidualCapacity", 17),
            ("Capacity-seed29", "ResidualCapacity", 29),
            ("CheapSummary-seed17", "ResidualCheapSummary", 17),
            ("CheapSummary-seed29", "ResidualCheapSummary", 29),
            ("P1", "P1", 17), ("Greedy", "Greedy", 17))
METRIC_FIELDS = ("external_caller_observed_return_seconds", "controller_return_sample_seconds",
                 "missed_return_sample", "actual_gain_seconds", "paid_shared_prefix_gain_seconds",
                 "actual_gain_beyond_paid_shared_prefix_seconds", "strict_on_time_gain_seconds",
                 "returned_value_seconds", "initial_value_seconds", "stop_reason")
THREAD_ENVIRONMENT = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                      "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, probability):
    """Linear empirical quantile, equivalent to NumPy's default linear method."""
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0 <= probability <= 1 or any(not math.isfinite(value) for value in ordered):
        raise ValueError("Finite nonempty sample and valid quantile required")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def under_root(path, root):
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Output must remain inside the declared new dataset root") from exc


def sealed_fit_receipt(fit_root):
    protocol_path, completion_path = fit_root / "protocol.json", fit_root / "completion.json"
    protocol, completion = read_json(protocol_path), read_json(completion_path)
    if completion.get("status") != "ACTUAL_P1_FIT_COMPLETE":
        raise ValueError("Only an actually completed, sealed four-fit root can be benchmarked")
    protocol_sha = sha(protocol_path)
    if completion.get("protocol_sha256") != protocol_sha:
        raise ValueError("Fit completion does not bind the current sealed protocol")
    if set(protocol.get("fit_sources", [])) != {0, 1, 2, 3} or protocol.get("P2_validation_sources") != [6, 7] or protocol.get("P2_test_sources") != [8, 9, 10, 11]:
        raise ValueError("Fit source split differs from the declared protocol")
    if completion.get("P2_validation_or_test_used") is not False:
        raise ValueError("Fit receipt does not confirm validation/test exclusion")
    required = {(variant, seed) for _, variant, seed in POLICIES if variant.startswith("Residual")}
    found = {(entry["variant"], entry["seed"]) for entry in completion["fits"]}
    if found != required:
        raise ValueError("All four predeclared Capacity/CheapSummary fits are required")
    checkpoints = []
    for variant, seed in sorted(required):
        checkpoint = fit_root / (variant + "-seed%d" % seed) / "final.pt"
        if not checkpoint.is_file():
            raise ValueError("Sealed checkpoint is missing: " + str(checkpoint))
        entry = next(item for item in completion["fits"] if (item["variant"], item["seed"]) == (variant, seed))
        checkpoints.append({"variant": variant, "seed": seed, "path": str(checkpoint),
                            "sha256_from_fit_completion": entry["sha256"]})
    return {"fit_root": str(fit_root), "fit_protocol_sha256": protocol_sha,
            "completion_sha256": sha(completion_path), "checkpoints": checkpoints,
            "normalization": protocol["normalization"], "model_source_sha256": protocol["model_source_sha256"]}


def graphs_for_sources(graphs_dir, sources):
    graphs = []
    for path in sorted(graphs_dir.glob("*.json")):
        metadata = read_json(path)
        if not isinstance(metadata, dict) or "graph_id" not in metadata:
            continue
        replicate = metadata.get("replicate_id", "")
        if not (len(replicate) == 4 and replicate[0] == "r" and replicate[1:].isdigit()):
            continue
        number = int(replicate[1:])
        if number not in sources:
            continue
        npz_path = path.with_suffix(".npz").resolve()
        if not npz_path.is_file():
            raise ValueError("Graph metadata has no matching NPZ: " + str(path))
        graph = {"source": number, "source_group": metadata["source_group"], "graph_id": metadata["graph_id"],
                 "split": metadata["split"], "station_view": metadata["station_view"],
                 "ground_gap_seconds": metadata["ground_gap_seconds"], "satellite_gap_seconds": metadata["satellite_gap_seconds"],
                 "graph_path": str(npz_path), "graph_sha256_from_metadata": metadata["npz_sha256"],
                 "contacts_sha256_from_metadata": metadata["contacts_sha256"],
                 "parameters_sha256_from_metadata": metadata["parameters_sha256"]}
        graphs.append(graph)
    for source in sources:
        current = [graph for graph in graphs if graph["source"] == source]
        configs = {(graph["station_view"], graph["ground_gap_seconds"], graph["satellite_gap_seconds"]) for graph in current}
        if len(current) != 6 or configs != EXPECTED_CONFIGS:
            raise ValueError("All six fixed R8/R12 × g170/g340/g680 graphs required for r%03d" % source)
        groups = {graph["source_group"] for graph in current}
        if groups != {"JR-SOURCE-r%03d" % source}:
            raise ValueError("Physical source binding differs from the declared dataset")
        expected_split = "train" if source in CALIBRATION_SOURCES else "validation" if source in (6, 7) else "test"
        if any(graph["split"].lower() != expected_split for graph in current):
            raise ValueError("Graph/source split mismatch")
    return sorted(graphs, key=lambda graph: (graph["source"], graph["graph_id"]))


def freeze_budgets(rows, protocol_sha, fit_receipt, workpoints=(200, 1000), action_seed=17):
    # Zero gain, late return, failed native search and fallback reports remain in
    # this distribution. Only a missing/invalid caller report blocks freezing.
    invalid = [row["run_id"] for row in rows if not row["report_usable_for_calibration"]]
    if invalid:
        return {"schema": SCHEMA, "status": "CALIBRATION_INCOMPLETE_BUDGETS_NOT_FROZEN",
                "invalid_or_missing_report_runs": invalid, "all_runs_retained": True}
    policy_costs = []
    for label, _, _ in POLICIES:
        selected = [row["external_caller_observed_return_seconds"] for row in rows if row["policy_label"] == label]
        if len(selected) != len(CALIBRATION_SOURCES) * 6:
            raise ValueError("Calibration must have all 12 graph/source outcomes for every policy")
        policy_costs.append({"policy_label": label, "samples": len(selected), "p95_seconds": percentile(selected, .95),
                             "maximum_observed_seconds": max(selected), "observed_seconds": selected})
    largest = max(item["p95_seconds"] for item in policy_costs)
    if largest <= 0:
        raise ValueError("Calibration cannot freeze a nonpositive caller budget")
    wide = largest * DELIVERY_MARGIN_MULTIPLIER
    return {"schema": SCHEMA, "status": "BUDGETS_FROZEN_FROM_DEVELOPMENT_ACTUAL_CALLERS",
            "frozen_utc": datetime.now(timezone.utc).isoformat(), "fit_protocol_sha256": fit_receipt["fit_protocol_sha256"],
            "calibration_protocol_sha256": protocol_sha, "calibration_sources": list(CALIBRATION_SOURCES),
            "physical_source_count": len(CALIBRATION_SOURCES), "native_workpoints_ms": list(workpoints),
            "action_seed": action_seed,
            "calibration_predeclared_deadline_seconds": CALIBRATION_CAP_SECONDS, "max_calls": MAX_CALLS,
            "cost_field": "external_caller_observed_return_seconds", "quantile_method": "linear empirical p95",
            "largest_policy_p95_seconds": largest, "delivery_margin_multiplier": DELIVERY_MARGIN_MULTIPLIER,
            "policy_distributions": policy_costs,
            "budgets": [{"budget_id": "wide", "deadline_seconds": wide},
                        {"budget_id": "half", "deadline_seconds": wide * .5},
                        {"budget_id": "quarter", "deadline_seconds": wide * .25}],
            "guarantees_zero_lateness": False, "test_outcomes_used": False,
            "note": "The 60-second ceiling is a predeclared calibration starting value, not an assurance of completion. Tight budgets may fail to pay the common prefix; retain them as execution pressure."}


def load_budgets(path, fit_receipt, workpoints, action_seed):
    frozen = read_json(path)
    if frozen.get("status") != "BUDGETS_FROZEN_FROM_DEVELOPMENT_ACTUAL_CALLERS":
        raise ValueError("Evaluation requires an actually frozen calibration budget file")
    if frozen["fit_protocol_sha256"] != fit_receipt["fit_protocol_sha256"]:
        raise ValueError("Frozen budgets and model fit are not bound to the same protocol")
    if frozen["calibration_sources"] != [4, 5] or frozen["max_calls"] != MAX_CALLS:
        raise ValueError("Frozen budget provenance differs from the declared calibration")
    if frozen.get("native_workpoints_ms") != list(workpoints) or frozen.get("action_seed") != action_seed:
        raise ValueError("Evaluation must retain the frozen native menu and action seed")
    budgets = frozen["budgets"]
    if [budget["budget_id"] for budget in budgets] != ["wide", "half", "quarter"]:
        raise ValueError("All three frozen budget levels are required")
    wide = budgets[0]["deadline_seconds"]
    if not finite_number(wide) or wide <= 0 or not math.isclose(budgets[1]["deadline_seconds"], .5 * wide) or not math.isclose(budgets[2]["deadline_seconds"], .25 * wide):
        raise ValueError("Frozen budgets must be positive Dwide, 0.5Dwide and 0.25Dwide")
    return budgets, {"path": str(path), "sha256": sha(path), "contents": frozen}


def kill_owned_process(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    else:
        # The child owns its native work; /T stops only its process descendants.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
        process.wait(timeout=10)


def execute_cell(cell, args, stage_out):
    run_dir = stage_out / "runs" / cell["run_id"]
    receipt_path = stage_out / "receipts" / (cell["run_id"] + ".json")
    if receipt_path.exists() and args.resume:
        return read_json(receipt_path)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("A prior unreceipted child output exists; preserve it and use a new stage output: " + str(run_dir))
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    log_dir = stage_out / "logs"
    log_dir.mkdir(exist_ok=True)
    stdout_path, stderr_path = log_dir / (cell["run_id"] + ".stdout.txt"), log_dir / (cell["run_id"] + ".stderr.txt")
    command = [args.python, str(args.actual_cli), "actual-policy", "--runtime-root", str(args.runtime_root),
               "--graph", cell["graph_path"], "--fit-root", str(args.fit_root), "--chils", str(args.chils),
               "--chils-source", str(args.chils_source), "--policy", cell["policy"], "--fit-seed", str(cell["fit_seed"]),
               "--action-seed", str(args.action_seed), "--deadline-seconds", format(cell["deadline_seconds"], ".17g"),
               "--max-calls", str(MAX_CALLS), "--budgets-ms", *map(str, args.budgets_ms), "--out", str(run_dir), "--device", "cpu"]
    environment = dict(os.environ)
    environment.update(THREAD_ENVIRONMENT)
    began_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        options = {"start_new_session": True} if os.name == "posix" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment, **options)
            try:
                exit_code = process.wait(timeout=cell["deadline_seconds"] + args.process_timeout_margin_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_owned_process(process)
                exit_code = process.returncode
            except BaseException:
                kill_owned_process(process)
                raise
        except OSError as exc:
            stderr.write("Driver launch failure: " + str(exc) + "\n")
            exit_code = None
    wall_seconds = time.monotonic() - started
    row = {**cell, "started_utc": began_utc, "cli_exit_code": exit_code, "process_watchdog_timeout": timed_out,
           "whole_cli_wall_seconds_including_resident_setup": wall_seconds,
           "actual_policy_path": str(run_dir / "actual_policy.json"),
           "stdout_path": str(stdout_path), "stderr_path": str(stderr_path), "command": command,
           "thread_environment": THREAD_ENVIRONMENT, "driver_does_not_substitute_cli_wall_for_caller_cost": True}
    report_path = run_dir / "actual_policy.json"
    if report_path.is_file():
        try:
            report = read_json(report_path)
            graph_binding = report.get("graph", {})
            row["report_identity_matches_cell"] = bool(
                graph_binding.get("graph_id") == cell["graph_id"]
                and graph_binding.get("source_group") == cell["source_group"]
                and graph_binding.get("npz_sha256") == cell["graph_sha256_from_metadata"]
                and report.get("policy") == cell["policy"]
                and report.get("fit_seed") == cell["fit_seed"]
                and report.get("deadline_seconds") == cell["deadline_seconds"])
            row["actual_report_schema"] = report.get("schema")
            row["actual_report_status"] = report.get("status")
            for field in METRIC_FIELDS:
                row[field] = report.get(field)
            row["resident_setup_fields"] = {key: value for key, value in report.items()
                                             if "setup" in key and isinstance(value, (int, float, dict))}
            row["initial_global_greedy_preparation_seconds"] = report.get("initial_global_greedy", {}).get("preparation_seconds")
            row["initial_greedy_setup_is_contained_in_deployment_resident_setup_do_not_add_twice"] = True
            row["event_count"] = len(report.get("events", []))
            row["report_sha256"] = sha(report_path)
            row["report_state"] = "received"
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            row["report_state"] = "invalid"
            row["report_read_error"] = str(exc)
    else:
        row["report_state"] = "missing"
    caller = row.get("external_caller_observed_return_seconds")
    row["report_usable_for_calibration"] = bool(row["report_state"] == "received" and exit_code == 0 and not timed_out
                                                and finite_number(caller) and caller >= 0
                                                and finite_number(row.get("actual_gain_seconds"))
                                                and row.get("report_identity_matches_cell") is True
                                                and row["actual_report_schema"] == "joint_recovery_stk_p1_actual_v1")
    # Preserve the failure itself. Strict delivered gain is zero for an absent
    # or invalid caller outcome, while raw gain/latency remain missing, not fake.
    strict = row.get("strict_on_time_gain_seconds")
    row["strict_delivered_gain_including_failed_runs_seconds"] = float(strict) if row["report_usable_for_calibration"] and finite_number(strict) else 0.0
    row["failure_or_unknown_outcome"] = not row["report_usable_for_calibration"]
    write_json(receipt_path, row)
    return row


def summarize(rows, sources, budget_ids):
    summaries = []
    metrics = ("strict_delivered_gain_including_failed_runs_seconds", "actual_gain_seconds",
               "paid_shared_prefix_gain_seconds", "actual_gain_beyond_paid_shared_prefix_seconds",
               "external_caller_observed_return_seconds")
    for budget_id in budget_ids:
        for label, _, _ in POLICIES:
            current = [row for row in rows if row["budget_id"] == budget_id and row["policy_label"] == label]
            source_values = []
            for source in sources:
                source_rows = [row for row in current if row["source"] == source]
                means, missing_counts = {}, {}
                for metric in metrics:
                    values = [row.get(metric) for row in source_rows if finite_number(row.get(metric))]
                    missing_counts[metric] = len(source_rows) - len(values)
                    # Unknown values are not silently removed from reported
                    # source means; strict delivered gain already accounts for
                    # their zero delivered outcome with an explicit failure flag.
                    means[metric] = sum(values) / len(values) if values and len(values) == len(source_rows) else None
                source_values.append({"source": "r%03d" % source, "source_group": "JR-SOURCE-r%03d" % source,
                                      "graph_runs": len(source_rows), "metrics": means, "missing_metric_runs": missing_counts,
                                      "failed_or_unknown_runs": sum(row["failure_or_unknown_outcome"] for row in source_rows),
                                      "native_failed_stop_runs": sum(row.get("stop_reason") == "failed" for row in source_rows),
                                      "missed_return_runs": sum(row.get("missed_return_sample") is True for row in source_rows)})
            overall = {metric: sum(item["metrics"][metric] for item in source_values) / len(source_values)
                       if source_values and all(item["metrics"][metric] is not None for item in source_values) else None
                       for metric in metrics}
            summaries.append({"budget_id": budget_id, "policy_label": label, "physical_sources": len(sources),
                              "equal_source_mean_metrics": overall, "source_values": source_values,
                              "graph_or_state_count_is_not_independent_sample_size": True})
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "evaluate"))
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--fit-root", required=True, type=Path)
    parser.add_argument("--graphs-dir", type=Path)
    parser.add_argument("--actual-cli", type=Path, default=Path(__file__).resolve().with_name("p1_fit_and_allocate.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--chils", required=True, type=Path)
    parser.add_argument("--chils-source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sources", nargs="+", type=int)
    parser.add_argument("--budgets-file", type=Path)
    parser.add_argument("--budgets-ms", nargs="+", type=int, default=[200, 1000])
    parser.add_argument("--action-seed", type=int, default=17)
    parser.add_argument("--process-timeout-margin-seconds", type=float, default=120.0,
                        help="Driver process watchdog allowance for resident setup; not part of D")
    parser.add_argument("--resume", action="store_true", help="Continue an identical frozen matrix; retain prior failures without replacing them")
    parser.add_argument("--plan-only", action="store_true", help="Write the frozen job matrix without launching policies")
    args = parser.parse_args(argv)
    if args.stage == "calibrate":
        if args.sources is not None and tuple(sorted(set(args.sources))) != CALIBRATION_SOURCES:
            parser.error("Calibration uses exactly development r004/r005")
        args.sources = list(CALIBRATION_SOURCES)
    else:
        if not args.budgets_file or not args.sources:
            parser.error("Evaluation needs frozen budgets and explicit validation or test sources")
        sources = set(args.sources)
        if not (sources.issubset({6, 7}) or sources.issubset({8, 9, 10, 11})):
            parser.error("Evaluate one declared VALIDATION or TEST source block; do not mix or use train sources")
        args.sources = sorted(sources)
    if args.budgets_ms not in ([200, 1000], [10, 50, 200, 1000]):
        parser.error("Declare a shared native menu 200/1000 or 10/50/200/1000 ms, in increasing order")
    if not finite_number(args.process_timeout_margin_seconds) or args.process_timeout_margin_seconds <= 0:
        parser.error("Positive finite process watchdog allowance required")
    return args


def main(argv=None):
    args = parse_args(argv)
    for name in ("dataset_root", "runtime_root", "fit_root", "actual_cli", "chils", "chils_source"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.graphs_dir = (args.graphs_dir or args.dataset_root / "graphs").resolve(strict=True)
    args.out = args.out.resolve()
    under_root(args.out, args.dataset_root)
    fit_receipt = sealed_fit_receipt(args.fit_root)
    graphs = graphs_for_sources(args.graphs_dir, args.sources)
    if args.stage == "calibrate":
        budgets = [{"budget_id": "calibration_60s", "deadline_seconds": CALIBRATION_CAP_SECONDS}]
        budget_receipt = None
    else:
        args.budgets_file = args.budgets_file.resolve(strict=True)
        budgets, budget_receipt = load_budgets(args.budgets_file, fit_receipt, args.budgets_ms, args.action_seed)
    cells = []
    for graph_index, graph in enumerate(graphs):
        offset = graph_index % len(POLICIES)
        rotated_policies = POLICIES[offset:] + POLICIES[:offset]
        for budget in budgets:
            for label, policy, seed in rotated_policies:
                cells.append({**graph, **budget, "policy_label": label, "policy": policy, "fit_seed": seed,
                              "run_id": graph["graph_id"] + "--" + label + "--" + budget["budget_id"]})
    protocol = {"schema": SCHEMA, "stage": args.stage, "sources": args.sources, "fit": fit_receipt,
                "driver_sha256": sha(__file__), "actual_cli_sha256": sha(args.actual_cli),
                "runtime_root": str(args.runtime_root), "chils": str(args.chils), "chils_source": str(args.chils_source),
                "python": args.python, "action_seed": args.action_seed, "max_calls": MAX_CALLS,
                "native_workpoints_ms": args.budgets_ms, "budgets": budgets, "frozen_budget_receipt": budget_receipt,
                "thread_environment": THREAD_ENVIRONMENT, "serial_policy_jobs": True,
                "fresh_cli_and_paid_warm_each_cell": True, "outcome_selection": "none; all six configurations/all six policies",
                "policy_order": "cyclic rotation by ordered graph index, declared before outcomes",
                "driver_process_watchdog_margin_seconds": args.process_timeout_margin_seconds,
                "whole_cli_wall_includes_resident_setup_but_D_uses_actual_reported_caller_return": True,
                "failed_late_and_zero_outcomes_retained": True,
                "budget_guarantees_zero_lateness": False, "matrix_cells": len(cells), "jobs": cells}
    protocol_sha = canonical_sha(protocol)
    protocol_path = args.out / "driver_protocol.json"
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or not protocol_path.is_file() or read_json(protocol_path).get("protocol_identity_sha256") != protocol_sha:
            raise FileExistsError("Use a fresh output directory, or --resume with the identical frozen protocol")
    args.out.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        write_json(protocol_path, {**protocol, "protocol_identity_sha256": protocol_sha, "created_utc": datetime.now(timezone.utc).isoformat()})
    if args.plan_only:
        print(json.dumps({"status": "MATRIX_FROZEN_NOT_EXECUTED", "cells": len(cells), "protocol": str(protocol_path)}))
        return 0
    rows = []
    for index, cell in enumerate(cells, 1):
        print(json.dumps({"event": "begin", "job": index, "total": len(cells), "run_id": cell["run_id"]}), flush=True)
        row = execute_cell(cell, args, args.out)
        rows.append(row)
        write_json(args.out / "actual_run_rows.json", rows)
        write_json(args.out / "progress.json", {"completed_cells": len(rows), "expected_cells": len(cells),
                                               "failure_or_unknown_runs": sum(item["failure_or_unknown_outcome"] for item in rows)})
        print(json.dumps({"event": "end", "job": index, "run_id": cell["run_id"], "report_state": row["report_state"],
                          "caller_seconds": row.get("external_caller_observed_return_seconds"),
                          "strict_gain_seconds": row["strict_delivered_gain_including_failed_runs_seconds"]}), flush=True)
    if sha(args.fit_root / "protocol.json") != fit_receipt["fit_protocol_sha256"]:
        raise RuntimeError("Fit protocol changed during execution; preserved runs cannot be called a sealed benchmark")
    summaries = summarize(rows, args.sources, [budget["budget_id"] for budget in budgets])
    write_json(args.out / "summary.json", {"schema": SCHEMA, "stage": args.stage, "status": "ACTUAL_SERIAL_MATRIX_COMPLETE",
               "expected_cells": len(cells), "received_cells": len(rows), "all_failures_retained": True,
               "physical_source_count": len(args.sources), "source_group_mean_results": summaries,
               "independent_unit": "physical source; six graph configurations and two fit seeds are not independent sources"})
    fields = ("run_id", "source_group", "graph_id", "policy_label", "budget_id", "deadline_seconds", *METRIC_FIELDS,
              "strict_delivered_gain_including_failed_runs_seconds", "failure_or_unknown_outcome", "report_state", "cli_exit_code",
              "whole_cli_wall_seconds_including_resident_setup", "actual_policy_path")
    with (args.out / "actual_run_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    frozen = None
    if args.stage == "calibrate":
        frozen = freeze_budgets(rows, protocol_sha, fit_receipt, args.budgets_ms, args.action_seed)
        write_json(args.out / "frozen_budgets.json", frozen)
    success = not any(row["failure_or_unknown_outcome"] for row in rows)
    print(json.dumps({"status": "MATRIX_COMPLETE" if success else "MATRIX_COMPLETE_WITH_FAILED_OR_UNKNOWN_RUNS",
                      "runs": len(rows), "out": str(args.out), "budget_status": frozen.get("status") if frozen else "reused_frozen_budgets"}), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
