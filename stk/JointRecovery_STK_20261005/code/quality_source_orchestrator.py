"""Independent source-parallel launcher/merger for the sealed quality pipeline.

Only execution placement changes: VAL r006/r007 on CPU40/44, then TEST
r008/r009/r010/r011 on CPU40/44/48/52. Within each source, existing method,
time-point, checkpoint, caller timing and failure semantics remain unchanged.
Child logs are outside child output roots. No result selects dispatch order.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import quality_method_comparison_pipeline as quality

SCHEMA = "joint_recovery_stk_quality_source_orchestrator_v1"
BINDING_SCHEMA = "joint_recovery_stk_quality_source_execution_binding_v1"
ASSIGNMENT = {6: 40, 7: 44, 8: 40, 9: 44, 10: 48, 11: 52}
STAGES = (("validation", (6, 7)), ("test", (8, 9, 10, 11)))
SUPERVISOR_CPU = 55
LAUNCHER_CPUS = tuple(range(40, 56))


def validate_binding(binding, common):
    if (binding.get("schema") != BINDING_SCHEMA or
            binding.get("status") != "QUALITY_SOURCE_PARALLEL_EXECUTION_BINDING" or
            binding.get("quality_plan_sha256") != common["quality_plan_sha256"] or
            binding.get("code_sha256") != common["code_sha256"] or
            binding.get("source_cpu_assignment") != {"r%03d" % source: cpu for source, cpu in ASSIGNMENT.items()} or
            binding.get("supervisor_cpu") != SUPERVISOR_CPU or
            binding.get("launcher_allowed_cpu_ids") != list(LAUNCHER_CPUS) or
            binding.get("source_parallel_overrides_original_serial_cpu40") is not True or
            binding.get("algorithms_and_allowance_unchanged") is not True):
        raise ValueError("Explicit source-parallel execution binding differs from the sealed algorithms/declared CPU schedule")
    if binding.get("orchestrator_sha256") is not None and binding["orchestrator_sha256"] != quality.sha(__file__):
        raise ValueError("Orchestrator implementation differs from its execution binding")


def frozen_allowance(args, common):
    completed = quality.wait_file(args.development_root / "completion.json", args.wait_timeout_seconds)
    path = args.development_root / "selected_allowance.json"
    frozen = quality.read(path)
    selected = frozen.get("selected_deadline_seconds")
    if (completed.get("status") != "QUALITY_DEVELOPMENT_COMPLETION_THRESHOLD_COMPLETE" or
            completed.get("pipeline_completed") is not True or completed.get("selected_allowance_sha256") != quality.sha(path) or
            frozen.get("schema") != quality.ALLOWANCE_SCHEMA or frozen.get("status") != "QUALITY_ALLOWANCE_PROTOCOL" or
            frozen.get("stage") != "frozen_from_development_completion" or
            frozen.get("development_sources") != [4] or selected not in quality.LADDER or
            frozen.get("evaluation_deadline_seconds") != sorted(set((10., 30., 60., 120., selected))) or
            frozen.get("selection_uses_gain") is not False or frozen.get("test_outcomes_used") is not False or
            frozen.get("quality_plan_sha256") != common["quality_plan_sha256"] or
            frozen.get("fit_completion_sha256") != common["fit"]["completion_sha256"] or
            frozen.get("code_sha256") != common["code_sha256"] or
            frozen.get("development_rows_sha256") != quality.sha(args.development_root / "actual_run_rows.json")):
        raise ValueError("Actually sealed development allowance and unchanged core implementations required")
    return frozen, path


def child_command(args, block, source):
    command = [sys.executable, str(args.pipeline_cli), "evaluate"]
    for name in ("dataset_root", "runtime_root", "fit_root", "numeric_root", "quality_plan", "development_root",
                 "original_comparison_root", "chils", "chils_source", "full_cli", "published_cli"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    command += ["--out", str(args.out / "shards" / ("r%03d" % source)), "--block", block,
                "--sources", str(source), "--evaluation-cpu-id", str(ASSIGNMENT[source]),
                "--wait-timeout-seconds", repr(args.wait_timeout_seconds)]
    return command


def owned_descendant_groups(process):
    """Cleanup only this launched source subtree, including owned native groups."""
    parents = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            status = directory.joinpath("status").read_text()
            parent = next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:"))
            parents[int(directory.name)] = int(parent)
        except (OSError, StopIteration, ValueError):
            continue
    owned = {process.pid}
    while True:
        children = {pid for pid, parent in parents.items() if parent in owned}
        if children.issubset(owned):
            break
        owned.update(children)
    groups = set()
    for pid in owned:
        try:
            group = os.getpgid(pid)
            if group in owned and group != os.getpgrp():
                groups.add(group)
        except ProcessLookupError:
            pass
    return groups


def stop_owned(process):
    groups = owned_descendant_groups(process)
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2.)
    except subprocess.TimeoutExpired:
        pass
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
        process.wait(timeout=2.)


def run_stage(args, block, sources):
    running, receipts = [], []
    logs = args.out / "child_logs"
    logs.mkdir(exist_ok=True)
    try:
        for source in sources:
            command = child_command(args, block, source)
            record = dict(source=source, source_group="JR-SOURCE-r%03d" % source, block=block,
                          cpu=ASSIGNMENT[source], command=command, started_utc=quality.now(),
                          child_out=str(args.out / "shards" / ("r%03d" % source)))
            stdout_path, stderr_path = logs / ("r%03d.stdout.txt" % source), logs / ("r%03d.stderr.txt" % source)
            record.update(stdout_path=str(stdout_path), stderr_path=str(stderr_path))
            stdout, stderr = stdout_path.open("w", encoding="utf-8"), stderr_path.open("w", encoding="utf-8")
            try:
                # Python's source pipeline checks its inherited affinity before
                # pinning; this explicit pre-exec change prevents CPU55 leakage.
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                          env=dict(os.environ, **quality.base.THREAD_ENVIRONMENT, PYTHONUNBUFFERED="1"),
                          start_new_session=True,
                          preexec_fn=lambda cpu=ASSIGNMENT[source]: os.sched_setaffinity(0, {cpu}))
                record["pid"] = process.pid
                running.append((process, stdout, stderr, record, time.monotonic()))
            except Exception as error:
                stdout.close()
                stderr.close()
                record.update(cli_exit_code=None, launch_failure=dict(type=type(error).__name__, message=str(error)))
                receipts.append(record)
        while running:
            next_running = []
            for process, stdout, stderr, record, began in running:
                result = process.poll()
                if result is None:
                    next_running.append((process, stdout, stderr, record, began))
                else:
                    stdout.close()
                    stderr.close()
                    record.update(cli_exit_code=result, completed_utc=quality.now(),
                                  child_process_wall_seconds=time.monotonic() - began)
                    receipts.append(record)
                    quality.write(args.out / "source_receipts" / ("r%03d.json" % record["source"]), record)
            running = next_running
            quality.write(args.out / "progress.json", dict(schema=SCHEMA, status="running_" + block,
                 updated_utc=quality.now(), active_sources=[entry[3]["source"] for entry in running],
                 exited_sources=[record["source"] for record in receipts],
                 source_results_not_used_to_select_execution=True))
            if running:
                time.sleep(5.)
    except BaseException:
        for process, stdout, stderr, record, _ in running:
            stop_owned(process)
            stdout.close()
            stderr.close()
            record.update(cli_exit_code=process.returncode, interrupted=True, completed_utc=quality.now())
            quality.write(args.out / "source_receipts" / ("r%03d.json" % record["source"]), record)
        raise
    for record in receipts:
        quality.write(args.out / "source_receipts" / ("r%03d.json" % record["source"]), record)
    return receipts


def merge_block(args, block, sources, frozen, graphs, receipts):
    root = args.out / block
    root.mkdir(exist_ok=True)
    selected, deadlines = frozen["selected_deadline_seconds"], frozen["evaluation_deadline_seconds"]
    expected = [quality.matrix_cell(graph, label, policy, seed, deadline, selected=selected)
                for graph in graphs if graph["source"] in sources for deadline in deadlines
                for label, policy, seed in quality.POLICIES]
    expected_by_id = {row["run_id"]: row for row in expected}
    rows, issues, shard_results = [], [], []
    by_source = {record["source"]: record for record in receipts}
    for source in sources:
        shard = args.out / "shards" / ("r%03d" % source)
        path = shard / block / "actual_run_rows.json"
        received = quality.read(path) if path.is_file() else []
        if not isinstance(received, list):
            issues.append("invalid_raw_row_container:r%03d" % source)
            received = []
        rows.extend(received)  # Preserve partial and failed rows, never impute.
        completion_path = shard / "completion.json"
        completion = quality.read(completion_path) if completion_path.is_file() else {}
        protocol_path = shard / "protocol.json"
        protocol = quality.read(protocol_path) if protocol_path.is_file() else {}
        expected_source = 6 * len(quality.POLICIES) * len(deadlines)
        complete = bool(by_source.get(source, {}).get("cli_exit_code") == 0 and
                        completion.get("status") == "QUALITY_COMPARISON_MATRIX_COMPLETE" and
                        completion.get("pipeline_completed") is True and
                        completion.get("expected_cells") == expected_source and
                        completion.get("received_cells") == expected_source and len(received) == expected_source and
                        protocol.get("evaluation_sources") == [source] and
                        protocol.get("cpu_affinity") == [ASSIGNMENT[source]] and
                        protocol.get("selected_allowance_sha256") == quality.sha(args.out / "selected_allowance.json") and
                        protocol.get("code_sha256") == frozen["code_sha256"])
        shard_results.append(dict(source=source, cpu=ASSIGNMENT[source], complete_declared_matrix=complete,
                                  expected_cells=expected_source, received_cells=len(received),
                                  completion_path=str(completion_path), row_path=str(path)))
        if not complete:
            issues.append("child_incomplete_or_execution_binding_mismatch:r%03d" % source)
    seen = set()
    for row in rows:
        identifier = row.get("run_id")
        declared = expected_by_id.get(identifier)
        if declared is None:
            issues.append("unexpected_cell:" + str(identifier))
        elif any(row.get(key) != value for key, value in declared.items()):
            issues.append("cell_identity_mismatch:" + str(identifier))
        if identifier in seen:
            issues.append("duplicate_cell:" + str(identifier))
        seen.add(identifier)
    missing = sorted(set(expected_by_id) - seen)
    mismatches = quality.initial_mismatches(rows)
    if mismatches:
        issues.append("common_original_S_mismatch")
    complete = not issues and not missing and len(rows) == len(expected)
    # Standard tables retain only the native recorded observations; coverage
    # gaps are explicit and keep report CLI from claiming a completed block.
    quality.save_rows(root, rows)
    quality.write(root / "declared_matrix.json", expected)
    main = quality.summarize(rows, sources, selected)
    curves = [entry for deadline in deadlines for entry in quality.summarize(rows, sources, deadline)]
    summary = dict(schema=quality.SCHEMA,
        status="QUALITY_COMPARISON_MATRIX_COMPLETE" if complete else "QUALITY_COMPARISON_MATRIX_INCOMPLETE",
        block=block, sources=list(sources), expected_cells=len(expected), received_cells=len(rows),
        selected_deadline_seconds=selected, evaluation_deadline_seconds=deadlines,
        source_group_mean_results=main, time_curve_source_group_mean_results=curves,
        initial_mask_mismatches=mismatches, complete_result_runs=sum(row.get("complete_result_produced") is True for row in rows),
        incomplete_result_runs=sum(row.get("complete_result_produced") is not True for row in rows),
        all_failed_late_zero_and_native_missing_results_preserved=True,
        primary_quality_metric="solution_quality_value_seconds; N/A when complete output missing",
        source_shards=shard_results, missing_declared_cells=missing, execution_issues=issues,
        source_parallel_execution_only_no_algorithm_change=True)
    quality.write(root / "summary.json", summary)
    return summary


def final_receipt(args, frozen, summaries):
    selected, deadlines = frozen["selected_deadline_seconds"], frozen["evaluation_deadline_seconds"]
    expected_cells = 36 * len(quality.POLICIES) * len(deadlines)
    complete = len(summaries) == 2 and all(row["status"] == "QUALITY_COMPARISON_MATRIX_COMPLETE" for row in summaries)
    blocks = [dict(block=row["block"], expected_cells=row["expected_cells"], received_cells=row["received_cells"],
                   complete_result_runs=row["complete_result_runs"], initial_mask_mismatches=row["initial_mask_mismatches"],
                   rows_sha256=quality.sha(args.out / row["block"] / "actual_run_rows.json")) for row in summaries]
    status = "QUALITY_COMPARISON_MATRIX_COMPLETE" if complete else "QUALITY_COMPARISON_MATRIX_INCOMPLETE"
    summary = dict(schema=quality.SCHEMA, status=status, expected_cells=expected_cells,
        received_cells=sum(row["received_cells"] for row in summaries), selected_deadline_seconds=selected,
        evaluation_deadline_seconds=deadlines, blocks=blocks,
        source_group_mean_results=[dict(block=row["block"], **entry) for row in summaries for entry in row["source_group_mean_results"]],
        time_curve_source_group_mean_results=[dict(block=row["block"], **entry) for row in summaries for entry in row["time_curve_source_group_mean_results"]],
        validation_and_test_not_pooled=True, no_algorithm_or_model_changed=True,
        all_failed_late_and_incomplete_results_preserved=True, source_parallel_execution=True,
        within_each_graph_all_methods_and_timepoints_serial_single_CPU=True,
        source_cpu_assignment={"r%03d" % source: cpu for source, cpu in ASSIGNMENT.items()},
        execution_binding_sha256=quality.sha(args.execution_binding))
    quality.write(args.out / "summary.json", summary)
    quality.write(args.out / "completion.json", dict(schema=quality.SCHEMA, status=status,
        pipeline_completed=complete, expected_cells=expected_cells, received_cells=summary["received_cells"],
        blocks=blocks, selected_allowance_sha256=quality.sha(args.out / "selected_allowance.json"),
        completed_utc=quality.now(), execution_binding_sha256=quality.sha(args.execution_binding),
        frozen_fit_completion_unchanged=quality.sha(args.fit_root / "completion.json") == frozen["fit_completion_sha256"]))
    return complete


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "runtime-root", "fit-root", "numeric-root", "quality-plan", "development-root",
                 "original-comparison-root", "chils", "chils-source", "out", "execution-binding"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--pipeline-cli", type=Path, default=Path(quality.__file__))
    parser.add_argument("--full-cli", type=Path, default=Path(__file__).with_name("full_joint_recovery_deadline.py"))
    parser.add_argument("--published-cli", type=Path, default=Path(__file__).with_name("quality_published_wholegraph.py"))
    parser.add_argument("--wait-timeout-seconds", type=float, default=86400.)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.out.relative_to(args.dataset_root)
    if (args.pipeline_cli != Path(quality.__file__).resolve() or
            not math.isfinite(args.wait_timeout_seconds) or args.wait_timeout_seconds <= 0):
        parser.error("Use the imported unchanged quality pipeline and a positive wait timeout")
    return args


def main(argv=None):
    args = parse_args(argv)
    if os.name != "posix":
        raise RuntimeError("Source orchestration runs only in the existing Linux cloud environment")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh source-parallel output; preserve prior/partial shards")
    args.out.mkdir(parents=True, exist_ok=True)
    inherited = sorted(os.sched_getaffinity(0))
    if not set(LAUNCHER_CPUS).issubset(inherited):
        raise ValueError("Launcher must initially permit CPU40 through55 before supervisor pinning")
    os.sched_setaffinity(0, {SUPERVISOR_CPU})
    common = quality.common_inputs(args)
    binding = quality.read(args.execution_binding)
    validate_binding(binding, common)
    quality.write(args.out / "progress.json", dict(schema=SCHEMA, status="waiting_for_development_and_original_short_comparison",
                  updated_utc=quality.now(), supervisor_cpu=SUPERVISOR_CPU))
    frozen, source_allowance = frozen_allowance(args, common)
    original = quality.wait_original(args)  # Includes the original process exit.
    shutil.copyfile(source_allowance, args.out / "selected_allowance.json")
    graphs = quality.base.graphs_for_sources(args.dataset_root / "graphs", [6, 7, 8, 9, 10, 11])
    protocol = dict(schema=SCHEMA, frozen_utc=quality.now(), **common,
        execution_binding_path=str(args.execution_binding), execution_binding_sha256=quality.sha(args.execution_binding),
        execution_binding=binding, orchestrator_sha256=quality.sha(__file__),
        launcher_inherited_allowed_cpu_ids=inherited, supervisor_cpu=SUPERVISOR_CPU,
        source_cpu_assignment={"r%03d" % source: cpu for source, cpu in ASSIGNMENT.items()},
        original_short_comparison=original, selected_allowance_sha256=quality.sha(source_allowance),
        selected_deadline_seconds=frozen["selected_deadline_seconds"],
        evaluation_deadline_seconds=frozen["evaluation_deadline_seconds"], graphs=graphs,
        expected_cells=36 * len(quality.POLICIES) * len(frozen["evaluation_deadline_seconds"]),
        source_parallel_overrides_original_serial_cpu40=True, only_execution_placement_changed=True,
        cross_source_parallelism_changes_host_load=True, within_each_graph_methods_and_points_share_one_CPU=True,
        source_results_not_used_to_select_dispatch=True)
    quality.write(args.out / "protocol.json", protocol)
    summaries = []
    for block, sources in STAGES:
        receipts = run_stage(args, block, sources)
        summary = merge_block(args, block, sources, frozen, graphs, receipts)
        summaries.append(summary)
        if summary["status"] != "QUALITY_COMPARISON_MATRIX_COMPLETE":
            # Do not promote partial validation to successful completion or
            # silently launch test after an incomplete source pipeline.
            if block == "validation":
                summaries.append(merge_block(args, "test", (8, 9, 10, 11), frozen, graphs, []))
            break
    return 0 if final_receipt(args, frozen, summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
