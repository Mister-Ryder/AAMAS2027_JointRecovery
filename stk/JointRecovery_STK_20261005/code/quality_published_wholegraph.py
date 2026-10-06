"""Quality-allowance wholegraph controls; the sealed short-budget code is reused.

This version changes only the evaluation allowance contract. No imported module
globals, solver configuration, graph rewards, initial incumbent, model or search
rule are patched. Actual native output is distinguished from external fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import published_wholegraph_deadline as execution

SCHEMA = "joint_recovery_stk_quality_published_actual_v1"
ALLOWANCE_SCHEMA = "joint_recovery_stk_quality_allowance_v1"
LADDER = (60., 120., 300.)


def require_quality_allowance(path, seconds):
    record = execution.read_json(path)
    if (record.get("schema") != ALLOWANCE_SCHEMA or record.get("status") != "QUALITY_ALLOWANCE_PROTOCOL" or
            record.get("test_outcomes_used") is not False or record.get("selection_uses_gain") is not False):
        raise ValueError("Independent quality allowance protocol required; short p95 calibration is not this protocol")
    stage = record.get("stage")
    if stage == "development_completion_threshold":
        if record.get("development_sources") != [4] or tuple(record.get("allowed_deadline_seconds", [])) != LADDER:
            raise ValueError("The fixed source4, 60/120/300 development allowance ladder is required")
        allowed = LADDER
    elif stage == "frozen_from_development_completion":
        selected = record.get("selected_deadline_seconds")
        expected = sorted(set((10., 30., 60., 120., selected))) if selected in LADDER else []
        allowed = tuple(record.get("evaluation_deadline_seconds", []))
        if record.get("development_sources") != [4] or list(allowed) != expected:
            raise ValueError("Quality allowance was not frozen from the declared development ladder")
    else:
        raise ValueError("Unknown quality allowance stage")
    if seconds not in allowed:
        raise ValueError("Use an exact deadline declared by the separate quality allowance protocol")
    return dict(path=str(Path(path).resolve()), sha256=execution.p0.sha_file(path),
                stage=stage, deadline_seconds=seconds, budget_id="quality",
                original_p95_budget_redefined=False, selection_uses_gain=False)


def native_completion(returned):
    events = returned.get("events", [])
    native_candidates = [event for event in events if event.get("kind") == "candidate" and
                         event.get("source") == "published_wholegraph_native" and
                         event.get("admitted_before_deadline") is True and
                         event.get("diagnostics", {}).get("native_called") is True and
                         event.get("diagnostics", {}).get("native_output_present") is True and
                         event.get("diagnostics", {}).get("status") == "returned_verified_complete_membership"]
    reasons = []
    if returned.get("stop_reason") != "finished":
        reasons.append("wholegraph_procedure_not_finished")
    if returned.get("missed_return_sample"):
        reasons.append("caller_return_late")
    if not native_candidates:
        reasons.append("no_actual_complete_feasible_native_membership_received_and_validated")
    complete = not reasons
    return dict(native_complete_output_produced=bool(native_candidates),
                native_complete_output_count=len(native_candidates),
                native_complete_output_value_seconds=(native_candidates[-1]["value_seconds"] if native_candidates else None),
                solver_procedure_finished=returned.get("stop_reason") == "finished",
                complete_result_produced=complete, completion_failure_reasons=reasons,
                external_original_fallback_is_not_native_output=True,
                solution_quality_value_seconds=returned["returned_value_seconds"] if complete else None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-root", "graph", "quality-allowance", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--method", choices=("CHILS-p1", "HiGHS-MILP"), required=True)
    parser.add_argument("--deadline-seconds", type=float, required=True)
    for name in ("chils", "chils-source", "numeric-gate"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if os.name != "posix" or not math.isfinite(args.deadline_seconds) or args.deadline_seconds < 10.:
        parser.error("Linux execution and a separately declared quality/curve allowance >=10 seconds required")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Quality wholegraph output must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    resident_started = time.perf_counter()
    allowance = require_quality_allowance(args.quality_allowance, args.deadline_seconds)
    api = execution.p0.load_runtime(args.runtime_root)
    graph, metadata = execution.p0.load_graph(args.graph, api)
    original, initial_record = execution.p0.global_incumbent(graph, api)
    initial_mask = execution.np.zeros(graph.n, dtype=execution.np.int8)
    initial_mask[list(original)] = 1
    initial_sha = hashlib.sha256(initial_mask.tobytes()).hexdigest()
    domain = execution.tick_domain(graph, metadata)
    solver_api = None
    if args.method == "CHILS-p1":
        solver_receipt = execution.verify_chils_gate(args, graph, metadata, domain)
    else:
        solver_api, solver_receipt = execution.resident_highs()
    setup_seconds = time.perf_counter() - resident_started
    external_started = time.perf_counter()
    returned, cleanup, cleanup_record = execution.run_wholegraph(graph, original, args, api, solver_api)
    returned["external_caller_observed_return_seconds"] = time.perf_counter() - external_started
    returned["missed_return_sample"] = returned["external_caller_observed_return_seconds"] > args.deadline_seconds
    returned["strict_on_time_gain_seconds"] = 0. if returned["missed_return_sample"] else returned["actual_gain_seconds"]
    cleanup.join()  # Finish owned cleanup before the next serial matrix cell.
    returned.update(schema=SCHEMA, status="QUALITY_PUBLISHED_WHOLEGRAPH_ATTEMPT_RECORDED",
                    graph=metadata, deployment_resident_setup_seconds=setup_seconds,
                    initial_global_greedy=initial_record, initial_mask_sha256=initial_sha,
                    quality_allowance_binding=allowance, microsecond_serialization=domain,
                    solver_source_and_binary=solver_receipt, paper_code_source_sha256=api["reference_sha256"],
                    implementation_sha256=execution.p0.sha_file(__file__),
                    reused_execution_implementation_sha256=execution.p0.sha_file(execution.__file__),
                    imported_execution_globals_modified=False,
                    p0_graph_and_initialization_sha256=execution.p0.sha_file(execution.p0.__file__),
                    solver_library_loading_in_resident_setup=True,
                    setup_excludes_python_process_and_initial_module_import_startup=True,
                    initial_global_greedy_preparation_already_in_resident_setup_do_not_add_twice=True,
                    cpu_affinity=sorted(os.sched_getaffinity(0)), threads_environment=execution.THREAD_ENV,
                    python_version=platform.python_version(), isolated_cleanup_after_function_return=cleanup_record,
                    citation=("https://doi.org/10.4230/LIPIcs.SEA.2025.22" if args.method == "CHILS-p1" else "https://highs.dev/"))
    returned.update(native_completion(returned))
    execution.p0.write_json(out / "actual_policy.json", returned)
    print(json.dumps(dict(method=args.method, graph_id=metadata["graph_id"],
                          complete_result_produced=returned["complete_result_produced"],
                          native_complete_output_produced=returned["native_complete_output_produced"],
                          solution_quality_value_seconds=returned["solution_quality_value_seconds"],
                          caller_seconds=returned["external_caller_observed_return_seconds"],
                          missed_return_sample=returned["missed_return_sample"])), flush=True)


if __name__ == "__main__":
    main()
