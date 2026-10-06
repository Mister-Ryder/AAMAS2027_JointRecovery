"""Completion-selected quality allowance, independent of sealed short-budget jobs.

DEV: source r004, every six configurations and eight unchanged policies, 60s.
Only incomplete cells advance to 120s then 300s; every attempt is retained.
The largest first-completion level (300 for unresolved cells) freezes one D.
VAL/TEST: wait for the original submitted job to exit; main table uses selectedD,
and independent curve runs use unique 10/30/60/120/selectedD allowances. Each
graph retains serial methods/points on one CPU. No fitting or gain selection.
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

import p1_benchmark_driver as base
import submitted_method_comparison_pipeline as submitted

SCHEMA = "joint_recovery_stk_quality_comparison_v1"
ALLOWANCE_SCHEMA = "joint_recovery_stk_quality_allowance_v1"
LADDER = (60., 120., 300.)
POLICIES = submitted.POLICIES
BLOCKS = submitted.BLOCKS
FIELDS = tuple(dict.fromkeys(submitted.FIELDS + (
    "solver_procedure_finished", "native_complete_output_produced", "native_complete_output_count",
    "native_complete_output_value_seconds", "complete_result_produced", "solution_quality_value_seconds",
    "legit_no_native_required", "no_native_required_reason", "failed_completed_requests",
    "observed_verified_native_responses")))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return submitted.sha(path)


def write(path, value):
    base.write_json(path, value)


def read(path):
    return base.read_json(path)


def wait_file(path, timeout):
    began = time.monotonic()
    while not path.is_file():
        if time.monotonic() - began >= timeout:
            raise TimeoutError("Required completion/allowance not available: " + str(path))
        time.sleep(15.)
    return read(path)


def original_job_active(root):
    """Read only process identity; do not touch other projects or kill jobs."""
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = directory.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
        except (OSError, ProcessLookupError):
            continue
        if (any(Path(arg).name == "submitted_method_comparison_pipeline.py" for arg in argv if arg) and
                "--out" in argv and argv.index("--out") + 1 < len(argv) and
                Path(argv[argv.index("--out") + 1]).resolve() == root):
            return True
    return False


def wait_original(args):
    completion = args.original_comparison_root / "completion.json"
    record = wait_file(completion, args.wait_timeout_seconds)
    if record.get("pipeline_completed") is not True:
        raise ValueError("Original short-allowance comparison has not sealed")
    began = time.monotonic()
    while original_job_active(args.original_comparison_root):
        if time.monotonic() - began >= args.wait_timeout_seconds:
            raise TimeoutError("Original comparison process is still active; CPU40 quality matrix not started")
        time.sleep(5.)
    return dict(path=str(completion), sha256=sha(completion), original_process_no_longer_active=True)


def pin_cpu(cpu):
    if os.name != "posix" or cpu not in os.sched_getaffinity(0):
        raise ValueError("Requested dedicated Linux CPU is unavailable: " + str(cpu))
    os.sched_setaffinity(0, {cpu})


def implementation_receipts(args):
    paths = [Path(__file__), args.full_cli, args.published_cli,
             Path(__file__).with_name("published_wholegraph_deadline.py"),
             Path(__file__).with_name("p0_recovery_probes.py"),
             Path(__file__).with_name("p1_actual_policy.py"),
             Path(__file__).with_name("p1_fit_and_allocate.py"),
             Path(__file__).with_name("p1_benchmark_driver.py"),
             Path(__file__).with_name("submitted_method_comparison_pipeline.py")]
    return {path.name: sha(path) for path in paths}


def common_inputs(args):
    plan = read(args.quality_plan)
    if plan.get("schema") != "joint_recovery_stk_quality_evaluation_plan_v1":
        raise ValueError("Separate declared quality evaluation plan required")
    gate = read(args.numeric_root / "completion.json")
    if gate.get("status") != "WHOLEGRAPH_NUMERIC_GUARD_PASS" or gate.get("pass_graphs") != 48:
        raise ValueError("Independent actual wholegraph numeric guard must have passed")
    fit = base.sealed_fit_receipt(args.fit_root)
    for checkpoint in fit["checkpoints"]:
        if sha(checkpoint["path"]) != checkpoint["sha256_from_fit_completion"]:
            raise ValueError("Quality comparison may only use unchanged sealed fits")
    return dict(quality_plan_path=str(args.quality_plan), quality_plan_sha256=sha(args.quality_plan),
                fit=fit, code_sha256=implementation_receipts(args),
                numeric_gate_completion_sha256=sha(args.numeric_root / "completion.json"),
                development_sources=[4], development_allowance_ladder_seconds=list(LADDER),
                policies=POLICIES, workpoints_ms=[10, 50, 200, 1000], max_calls=8, action_seed=17,
                no_retraining=True, no_model_or_algorithm_configuration_change=True,
                selection_uses_gain=False, test_outcomes_used=False,
                original_short_allowance_protocol_not_changed=True)


def freeze_protocol(out, identity, resume):
    path = out / "protocol.json"
    digest = base.canonical_sha(identity)
    if path.is_file():
        if not resume or read(path).get("identity_sha256") != digest:
            raise ValueError("Resume requires an identical independent quality protocol")
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Quality output directory must be fresh")
        out.mkdir(parents=True, exist_ok=True)
        write(path, dict(schema=SCHEMA, frozen_utc=now(), identity_sha256=digest, **identity))


class MembershipChecker:
    """One graph decoded at a time; validate produced members, not new physics."""
    def __init__(self):
        self.path = None

    def load(self, cell):
        import numpy as np
        path = cell["graph_path"]
        if path != self.path:
            if sha(path) != cell["graph_sha256_from_metadata"]:
                raise ValueError("Frozen original graph changed")
            with np.load(path, allow_pickle=False) as archive:
                self.weights = np.asarray(archive["weights"], dtype=np.float64)
                self.edges = np.asarray(archive["edges"], dtype=np.int64).reshape(-1, 2)
            self.path = path
        return np

    def validate(self, raw, value):
        import numpy as np
        if not isinstance(raw, list) or len(raw) != len(set(raw)) or any(type(v) is not int for v in raw):
            return False, None
        if any(v < 0 or v >= len(self.weights) for v in raw):
            return False, None
        mask = np.zeros(len(self.weights), dtype=np.int8)
        mask[raw] = 1
        if np.any(mask[self.edges[:, 0]] & mask[self.edges[:, 1]]):
            return False, None
        actual = math.fsum(float(self.weights[v]) for v in sorted(raw))
        valid = base.finite_number(value) and math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-7)
        return valid, hashlib.sha256(mask.tobytes()).hexdigest()


def command_for(cell, args, run_dir, allowance_file):
    common = ["--runtime-root", str(args.runtime_root), "--graph", cell["graph_path"],
              "--out", str(run_dir), "--deadline-seconds", repr(cell["deadline_seconds"]),
              "--chils", str(args.chils), "--chils-source", str(args.chils_source)]
    if cell["policy"].startswith("Full"):
        return ([sys.executable, str(args.full_cli), *common, "--fit-root", str(args.fit_root),
                 "--policy", cell["policy"], "--fit-seed", str(cell["fit_seed"]),
                 "--budgets-ms", "10", "50", "200", "1000", "--max-calls", "8",
                 "--action-seed", "17", "--cpu-index", "0"],
                run_dir / "full_joint_recovery.json", "joint_recovery_stk_full_actual_v1")
    command = [sys.executable, str(args.published_cli), *common, "--method", cell["policy"],
               "--quality-allowance", str(allowance_file)]
    if cell["policy"] == "CHILS-p1":
        command += ["--numeric-gate", str(args.numeric_root / "gates" / (cell["graph_id"] + ".json"))]
    return command, run_dir / "actual_policy.json", "joint_recovery_stk_quality_published_actual_v1"


def classify(row, report, cell, checker):
    checker.load(cell)
    reasons = []
    g = report.get("graph", {})
    identity = (g.get("graph_id") == cell["graph_id"] and g.get("source_group") == cell["source_group"] and
                g.get("npz_sha256") == cell["graph_sha256_from_metadata"] and
                report.get("policy") == cell["policy"] and report.get("fit_seed") == cell["fit_seed"] and
                report.get("deadline_seconds") == cell["deadline_seconds"] and report.get("schema") == row["expected_schema"])
    initial_valid, initial_sha = checker.validate(report.get("original_members"), report.get("initial_value_seconds"))
    returned_valid, _ = checker.validate(report.get("selected_members"), report.get("returned_value_seconds"))
    initial_valid = initial_valid and initial_sha == report.get("initial_mask_sha256")
    usable = bool(identity and row.get("cli_exit_code") == 0 and not row["process_watchdog_timeout"] and
                  base.finite_number(report.get("external_caller_observed_return_seconds")) and
                  base.finite_number(report.get("actual_gain_seconds")))
    if not usable:
        reasons.append("missing_failed_or_identity_invalid_caller_report")
    if not initial_valid:
        reasons.append("original_incumbent_not_full_graph_valid")
    if not returned_valid:
        reasons.append("returned_membership_not_full_graph_valid")
    caller = report.get("external_caller_observed_return_seconds")
    late = not base.finite_number(caller) or caller > cell["deadline_seconds"]
    if late:
        reasons.append("caller_return_late_or_unknown")
    if report.get("stop_reason") != "finished":
        reasons.append("procedure_not_finished")
    native_candidates, full_candidates, verified_local_native = [], [], []
    for event in report.get("events", []):
        if event.get("kind") != "candidate" or event.get("admitted_before_deadline") is not True:
            continue
        valid, _ = checker.validate(event.get("members"), event.get("value_seconds"))
        if valid:
            full_candidates.append(event)
            diagnostic = event.get("diagnostics", {})
            if (event.get("source") == "published_wholegraph_native" and
                    diagnostic.get("native_called") is True and diagnostic.get("native_output_present") is True and
                    diagnostic.get("status") == "returned_verified_complete_membership"):
                native_candidates.append(event)
            local_diagnostic = event.get("actual_request", {}).get("diagnostics", {})
            if (event.get("source") == "actual_native_request" and
                    local_diagnostic.get("native_called") is True and
                    local_diagnostic.get("native_output_valid") is True):
                verified_local_native.append(event)
    finished = next((event for event in report.get("events", []) if event.get("kind") == "finished"), {})
    starts = [event for event in report.get("events", []) if event.get("kind") == "request_started"]
    nonempty_native_required = any(event.get("empty_scope_request") is False for event in starts)
    calls = finished.get("actual_calls", [])
    actual_native_calls = (sum(int(call.get("diagnostics", {}).get("native_called") is True) for call in calls)
                           if finished else None)
    failed_calls = sum(call.get("failed") is True for call in calls)
    legitimate_no_native = bool(cell["policy"].startswith("Full") and finished and
                               not nonempty_native_required and actual_native_calls == 0)
    no_native_reason = None
    if legitimate_no_native:
        no_native_reason = ("no_scoped_requests" if finished.get("scoped_requests") == 0 else
                            "only_empty_recovery_domains_attempted" if starts else
                            str(finished.get("controller_stop_reason", "no_executable_request")))
    if cell["policy"].startswith("Full"):
        if not full_candidates:
            reasons.append("no_caller_validated_complete_common_or_native_candidate")
        if nonempty_native_required and not verified_local_native:
            reasons.append("nonempty_native_attempts_without_actual_valid_native_response")
    elif not native_candidates:
        reasons.append("no_actual_complete_feasible_native_output_external_S_is_fallback_only")
    complete = not reasons
    row.update({field: report.get(field) for field in FIELDS})
    row.update(report_identity_matches_cell=identity, report_usable=usable,
               initial_membership_full_graph_valid=initial_valid, returned_membership_full_graph_valid=returned_valid,
               native_complete_output_produced=bool(native_candidates), native_complete_output_count=len(native_candidates),
               native_complete_output_value_seconds=native_candidates[-1]["value_seconds"] if native_candidates else None,
               complete_result_produced=complete, completion_failure_reasons=reasons,
               solution_quality_value_seconds=report["returned_value_seconds"] if complete else None,
               solver_procedure_finished=report.get("stop_reason") == "finished", missed_return_sample=late,
               legit_no_native_required=legitimate_no_native, no_native_required_reason=no_native_reason,
               actual_native_calls=actual_native_calls, observed_verified_native_responses=len(verified_local_native),
               failed_completed_requests=failed_calls,
               failure_or_unknown_outcome=not usable,
               strict_delivered_gain_including_failed_runs_seconds=(report.get("strict_on_time_gain_seconds", 0.)
                    if usable and initial_valid and returned_valid and not late else 0.),
               complete_result_gain_seconds=report["actual_gain_seconds"] if complete else None)


def execute(cell, args, root, allowance_file, checker):
    receipt = root / "receipts" / (cell["run_id"] + ".json")
    if receipt.is_file():
        if not args.resume:
            raise FileExistsError("Never overwrite a quality attempt receipt")
        row = read(receipt)
        if any(row.get(key) != value for key, value in cell.items()):
            raise ValueError("Existing quality attempt identity differs")
        return row
    run_dir = root / "runs" / cell["run_id"]
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    command, report_path, expected_schema = command_for(cell, args, run_dir, allowance_file)
    row = dict(cell, command=command, expected_schema=expected_schema, started_utc=now(),
               actual_policy_path=str(report_path), process_watchdog_timeout=False,
               quality_allowance_sha256=sha(allowance_file), cpu_affinity=sorted(os.sched_getaffinity(0)),
               report_usable=False, complete_result_produced=False, native_complete_output_produced=False,
               failure_or_unknown_outcome=True, solution_quality_value_seconds=None,
               strict_delivered_gain_including_failed_runs_seconds=0.)
    began = time.monotonic()
    try:
        if run_dir.exists():
            raise FileExistsError("Unreceipted attempt directory retained; do not overwrite it")
        with (logs / (cell["run_id"] + ".stdout.txt")).open("w", encoding="utf-8") as stdout, \
                (logs / (cell["run_id"] + ".stderr.txt")).open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                      env=dict(os.environ, **base.THREAD_ENVIRONMENT, PYTHONUNBUFFERED="1"),
                      stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                row["cli_exit_code"] = process.wait(timeout=cell["deadline_seconds"] + 120.)
            except subprocess.TimeoutExpired:
                base.kill_owned_process(process)
                row.update(cli_exit_code=process.returncode, process_watchdog_timeout=True)
            except BaseException:
                base.kill_owned_process(process)
                raise
        if report_path.is_file():
            report = read(report_path)
            row.update(report_state="received", report_sha256=sha(report_path),
                       actual_report_schema=report.get("schema"), actual_report_status=report.get("status"),
                       resident_setup_fields={key: value for key, value in report.items() if "setup" in key})
            classify(row, report, cell, checker)
        else:
            row.update(report_state="missing", completion_failure_reasons=["actual_caller_report_missing"])
    except Exception as error:
        row.update(failure=dict(type=type(error).__name__, message=str(error)), report_usable=False,
                   complete_result_produced=False, failure_or_unknown_outcome=True,
                   completion_failure_reasons=["attempt_failed_or_output_invalid"], solution_quality_value_seconds=None,
                   strict_delivered_gain_including_failed_runs_seconds=0.)
    row["whole_cli_wall_seconds_including_resident_setup"] = time.monotonic() - began
    write(receipt, row)
    return row


def matrix_cell(graph, label, policy, seed, seconds, selected=None):
    return dict(graph, policy_label=label, policy=policy, fit_seed=seed,
                budget_id="quality" if selected is None or seconds == selected else "curve-%gs" % seconds,
                is_primary_quality_allowance=selected is not None and seconds == selected,
                deadline_seconds=seconds, run_id=graph["graph_id"] + "__quality-%gs__" % seconds + label)


def save_rows(root, rows):
    write(root / "actual_run_rows.json", rows)
    fields = tuple(dict.fromkeys(("run_id", "source", "source_group", "graph_id", "policy_label", "budget_id",
              "deadline_seconds", "is_primary_quality_allowance", *FIELDS,
              "initial_membership_full_graph_valid", "returned_membership_full_graph_valid",
              "completion_failure_reasons", "complete_result_gain_seconds", "report_usable", "failure_or_unknown_outcome",
              "strict_delivered_gain_including_failed_runs_seconds", "whole_cli_wall_seconds_including_resident_setup")))
    with (root / "actual_run_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def initial_mismatches(rows):
    masks = {}
    for row in rows:
        if row.get("initial_membership_full_graph_valid"):
            masks.setdefault(row["graph_id"], set()).add(row["initial_mask_sha256"])
    return {key: sorted(values) for key, values in masks.items() if len(values) != 1}


def summarize(rows, sources, deadline):
    metrics = ("solution_quality_value_seconds", "returned_value_seconds", "initial_value_seconds",
               "strict_delivered_gain_including_failed_runs_seconds", "actual_gain_seconds",
               "paid_shared_prefix_gain_seconds", "actual_gain_beyond_paid_shared_prefix_seconds",
               "external_caller_observed_return_seconds", "complete_result_gain_seconds")
    results = []
    for label, _, _ in POLICIES:
        source_values = []
        for source in sources:
            current = [row for row in rows if row["source"] == source and row["policy_label"] == label and
                       row["deadline_seconds"] == deadline]
            means = {metric: math.fsum(row[metric] for row in current) / len(current)
                     if current and all(base.finite_number(row.get(metric)) for row in current) else None for metric in metrics}
            source_values.append(dict(source=source, source_group="JR-SOURCE-r%03d" % source,
                 declared_runs=6, received_runs=len(current), metrics=means,
                 complete_result_runs=sum(row.get("complete_result_produced") is True for row in current),
                 incomplete_result_runs=sum(row.get("complete_result_produced") is not True for row in current),
                 native_complete_output_runs=sum(row.get("native_complete_output_produced") is True for row in current),
                 missed_return_runs=sum(row.get("missed_return_sample") is True for row in current),
                 failed_or_unknown_runs=sum(row.get("failure_or_unknown_outcome") is True for row in current)))
        mean = {metric: math.fsum(source["metrics"][metric] for source in source_values) / len(source_values)
                if all(source["metrics"][metric] is not None for source in source_values) else None for metric in metrics}
        results.append(dict(budget_id="quality", deadline_seconds=deadline, policy_label=label, source_values=source_values,
                            equal_physical_source_mean=mean, physical_source_count=len(sources),
                            incomplete_quality_rows_not_imputed_with_external_fallback=True))
    return results


def development(args, common):
    pin_cpu(args.development_cpu_id)
    graphs = base.graphs_for_sources(args.dataset_root / "graphs", [4])
    identity = dict(common, stage="development_completion_threshold", graphs=graphs,
                    cpu_affinity=[args.development_cpu_id], expected_initial_cells=48,
                    maximum_attempts=144, retries_only_for_incomplete_results=True)
    freeze_protocol(args.out, identity, args.resume)
    allowance_file = args.out / "quality_allowance_protocol.json"
    allowance = dict(schema=ALLOWANCE_SCHEMA, status="QUALITY_ALLOWANCE_PROTOCOL",
                     stage="development_completion_threshold", development_sources=[4],
                     allowed_deadline_seconds=list(LADDER), quality_plan_sha256=common["quality_plan_sha256"],
                     selection_uses_gain=False, test_outcomes_used=False)
    if allowance_file.is_file() and read(allowance_file) != allowance:
        raise ValueError("Development allowance protocol changed")
    if not allowance_file.is_file():
        write(allowance_file, allowance)
    rows, thresholds = [], []
    checker = MembershipChecker()
    for graph in graphs:
        for label, policy, seed in POLICIES:
            attempts = []
            complete = False
            for seconds in LADDER:
                cell = matrix_cell(graph, label, policy, seed, seconds)
                row = execute(cell, args, args.out, allowance_file, checker)
                rows.append(row)
                attempts.append(dict(run_id=row["run_id"], deadline_seconds=seconds,
                                     complete_result_produced=row["complete_result_produced"],
                                     completion_failure_reasons=row.get("completion_failure_reasons", [])))
                save_rows(args.out, rows)
                write(args.out / "progress.json", dict(schema=SCHEMA, status="checking_development_completion",
                      updated_utc=now(), attempted_cells=len(rows), completed_base_cells=len(thresholds), expected_base_cells=48))
                if row["complete_result_produced"]:
                    complete = True
                    break
            thresholds.append(dict(graph_id=graph["graph_id"], policy_label=label,
                 first_complete_deadline_seconds=seconds if complete else None,
                 required_allowance_seconds=seconds if complete else 300.,
                 complete_within_declared_ladder=complete, attempts=attempts,
                 unresolved_after_300_seconds_is_NA=not complete))
    mismatch = initial_mismatches(rows)
    if mismatch:
        write(args.out / "initial_mask_mismatches.json", mismatch)
        raise ValueError("Development methods did not share identical original S; no quality allowance frozen")
    selected = max(60., max(row["required_allowance_seconds"] for row in thresholds))
    frozen = dict(schema=ALLOWANCE_SCHEMA, status="QUALITY_ALLOWANCE_PROTOCOL",
        stage="frozen_from_development_completion", frozen_utc=now(), development_sources=[4],
        allowed_development_deadline_seconds=list(LADDER), selected_deadline_seconds=selected,
        evaluation_deadline_seconds=sorted(set((10., 30., 60., 120., selected))),
        primary_table_deadline_seconds=selected, curve_points_are_independent_actual_runs=True,
        selection_rule="max first-complete allowance across all 48 fixed cells; unresolved at300 contributes300",
        completion_thresholds=thresholds, selection_uses_gain=False, test_outcomes_used=False,
        quality_plan_sha256=common["quality_plan_sha256"], fit_completion_sha256=common["fit"]["completion_sha256"],
        code_sha256=common["code_sha256"], development_protocol_sha256=sha(args.out / "protocol.json"),
        development_rows_sha256=sha(args.out / "actual_run_rows.json"),
        unresolved_development_cells=[row for row in thresholds if not row["complete_within_declared_ladder"]],
        guarantees_all_methods_complete=False, no_unbounded_allowance_growth=True)
    write(args.out / "selected_allowance.json", frozen)
    write(args.out / "summary.json", dict(schema=SCHEMA, status="QUALITY_DEVELOPMENT_COMPLETION_THRESHOLD_COMPLETE",
          initial_cells=48, recorded_attempts=len(rows), selected_deadline_seconds=selected,
          completion_thresholds=thresholds, no_gain_based_allowance_selection=True))
    write(args.out / "completion.json", dict(schema=SCHEMA, status="QUALITY_DEVELOPMENT_COMPLETION_THRESHOLD_COMPLETE",
          pipeline_completed=True, selected_allowance_sha256=sha(args.out / "selected_allowance.json"),
          development_rows_sha256=sha(args.out / "actual_run_rows.json"), completed_utc=now()))


def evaluate(args, common):
    args.out.mkdir(parents=True, exist_ok=True) if not args.out.exists() else None
    # No CPU40 work is started while the original primary comparison runs.
    original = wait_original(args)
    dev_complete = wait_file(args.development_root / "completion.json", args.wait_timeout_seconds)
    frozen_path = args.development_root / "selected_allowance.json"
    frozen = read(frozen_path)
    if (dev_complete.get("status") != "QUALITY_DEVELOPMENT_COMPLETION_THRESHOLD_COMPLETE" or
            dev_complete.get("selected_allowance_sha256") != sha(frozen_path) or
            frozen.get("schema") != ALLOWANCE_SCHEMA or frozen.get("status") != "QUALITY_ALLOWANCE_PROTOCOL" or
            frozen.get("stage") != "frozen_from_development_completion" or
            frozen.get("development_sources") != [4] or frozen.get("selected_deadline_seconds") not in LADDER or
            frozen.get("selection_uses_gain") is not False or frozen.get("test_outcomes_used") is not False):
        raise ValueError("Actually sealed development-only quality allowance required")
    if (frozen["fit_completion_sha256"] != common["fit"]["completion_sha256"] or
            frozen["code_sha256"] != common["code_sha256"] or
            frozen["quality_plan_sha256"] != common["quality_plan_sha256"]):
        raise ValueError("Fits, implementations or quality plan changed between development and holdout")
    if frozen["development_rows_sha256"] != sha(args.development_root / "actual_run_rows.json"):
        raise ValueError("Frozen development completion evidence changed")
    seconds = frozen["selected_deadline_seconds"]
    curve_deadlines = sorted(set((10., 30., 60., 120., seconds)))
    if frozen.get("evaluation_deadline_seconds") != curve_deadlines:
        raise ValueError("Frozen quality curve points changed")
    sources = args.sources or ([6, 7] if args.block == "validation" else [8, 9, 10, 11]
                              if args.block == "test" else [6, 7, 8, 9, 10, 11])
    graphs = base.graphs_for_sources(args.dataset_root / "graphs", sources)
    expected_cells = len(graphs) * len(POLICIES) * len(curve_deadlines)
    pin_cpu(args.evaluation_cpu_id)
    identity = dict(common, stage="quality_holdout_comparison", graphs=graphs,
                    cpu_affinity=[args.evaluation_cpu_id], original_short_comparison=original,
                    selected_allowance_sha256=sha(frozen_path), selected_deadline_seconds=seconds,
                    evaluation_sources=sources, evaluation_deadline_seconds=curve_deadlines,
                    expected_cells=expected_cells, primary_table_uses_selected_allowance=True,
                    each_curve_point_has_an_independent_actual_run=True)
    freeze_protocol(args.out, identity, args.resume)
    copied_allowance = args.out / "selected_allowance.json"
    if copied_allowance.exists() and read(copied_allowance) != frozen:
        raise ValueError("Existing selected allowance differs")
    if not copied_allowance.exists():
        write(copied_allowance, frozen)
    checker, block_records, flattened_summary, curve_summary, total = MembershipChecker(), [], [], [], 0
    for block, declared_sources in BLOCKS:
        block_sources = [source for source in declared_sources if source in sources]
        if not block_sources:
            continue
        root = args.out / block
        root.mkdir(exist_ok=True)
        cells = [matrix_cell(graph, label, policy, seed, deadline, selected=seconds)
                 for graph in graphs if graph["source"] in block_sources
                 for deadline in curve_deadlines for label, policy, seed in POLICIES]
        write(root / "declared_matrix.json", cells)
        rows = []
        for cell in cells:
            row = execute(cell, args, root, copied_allowance, checker)
            rows.append(row)
            total += 1
            save_rows(root, rows)
            write(args.out / "progress.json", dict(schema=SCHEMA, status="evaluating_quality_" + block,
                  updated_utc=now(), completed_cells=total, expected_cells=expected_cells,
                  block_completed_cells=len(rows), block_expected_cells=len(cells), selected_deadline_seconds=seconds,
                  complete_result_runs=sum(row["complete_result_produced"] for row in rows)))
        mismatch = initial_mismatches(rows)
        summaries = summarize(rows, block_sources, seconds)
        curves = [item for deadline in curve_deadlines for item in summarize(rows, block_sources, deadline)]
        summary = dict(schema=SCHEMA, status="QUALITY_COMPARISON_MATRIX_COMPLETE", block=block,
                       sources=block_sources, expected_cells=len(cells), received_cells=len(rows),
                       selected_deadline_seconds=seconds, source_group_mean_results=summaries,
                       evaluation_deadline_seconds=curve_deadlines, time_curve_source_group_mean_results=curves,
                       initial_mask_mismatches=mismatch,
                       complete_result_runs=sum(row["complete_result_produced"] for row in rows),
                       incomplete_result_runs=sum(not row["complete_result_produced"] for row in rows),
                       all_failed_late_zero_and_native_missing_results_preserved=True,
                       primary_quality_metric="solution_quality_value_seconds; N/A when complete output missing")
        write(root / "summary.json", summary)
        block_records.append(dict(block=block, expected_cells=len(cells), received_cells=len(rows),
                                  complete_result_runs=summary["complete_result_runs"], initial_mask_mismatches=mismatch,
                                  rows_sha256=sha(root / "actual_run_rows.json")))
        flattened_summary.extend(dict(block=block, **row) for row in summaries)
        curve_summary.extend(dict(block=block, **row) for row in curves)
    write(args.out / "summary.json", dict(schema=SCHEMA, status="QUALITY_COMPARISON_MATRIX_COMPLETE",
          expected_cells=expected_cells, received_cells=total, selected_deadline_seconds=seconds,
          blocks=block_records, source_group_mean_results=flattened_summary,
          evaluation_deadline_seconds=curve_deadlines, time_curve_source_group_mean_results=curve_summary,
          validation_and_test_not_pooled=True, no_algorithm_or_model_changed=True,
          all_failed_late_and_incomplete_results_preserved=True))
    write(args.out / "completion.json", dict(schema=SCHEMA, status="QUALITY_COMPARISON_MATRIX_COMPLETE",
          pipeline_completed=True, expected_cells=expected_cells, received_cells=total, blocks=block_records,
          selected_allowance_sha256=sha(copied_allowance), completed_utc=now(),
          frozen_fit_completion_unchanged=sha(args.fit_root / "completion.json") == common["fit"]["completion_sha256"]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dev", "evaluate"))
    for name in ("dataset-root", "runtime-root", "fit-root", "numeric-root", "quality-plan", "chils", "chils-source", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("original-comparison-root", "development-root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--full-cli", type=Path, default=Path(__file__).with_name("full_joint_recovery_deadline.py"))
    parser.add_argument("--published-cli", type=Path, default=Path(__file__).with_name("quality_published_wholegraph.py"))
    parser.add_argument("--development-cpu-id", type=int, default=42)
    parser.add_argument("--evaluation-cpu-id", type=int, default=40)
    parser.add_argument("--block", choices=("all", "validation", "test"), default="all")
    parser.add_argument("--sources", type=int, nargs="+", choices=(6, 7, 8, 9, 10, 11))
    parser.add_argument("--wait-timeout-seconds", type=float, default=86400.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.out.relative_to(args.dataset_root)
    if (args.stage == "evaluate" and (args.original_comparison_root is None or args.development_root is None)):
        parser.error("Evaluation requires the completed original comparison root and independent quality development root")
    if args.development_cpu_id == 40 or not math.isfinite(args.wait_timeout_seconds) or args.wait_timeout_seconds <= 0:
        parser.error("Development must use another dedicated CPU; positive bounded wait required")
    if args.sources:
        args.sources = sorted(set(args.sources))
        if ((args.block == "validation" and any(source not in (6, 7) for source in args.sources)) or
                (args.block == "test" and any(source not in (8, 9, 10, 11) for source in args.sources))):
            parser.error("Source shard must belong to the requested block")
    return args


def main(argv=None):
    args = parse_args(argv)
    if os.name != "posix":
        raise RuntimeError("Quality comparisons run only in the existing Linux cloud runtime")
    common = common_inputs(args)
    if args.stage == "dev":
        development(args, common)
    else:
        evaluate(args, common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
