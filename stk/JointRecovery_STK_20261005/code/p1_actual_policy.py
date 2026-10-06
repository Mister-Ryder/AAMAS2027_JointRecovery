"""P1 Linux CPU caller-owned deadline execution; no teacher label inputs.

Resident graph/model and an already feasible original S are setup. The decision
timer includes process launch, menu/scopes/common warm, inference, native calls,
all parsing/checking, candidate IPC and the caller's final membership checks.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import threading
import time

import numpy as np

import p0_recovery_probes as p0


def scan_request_admission(order, budgets, conditional_costs, deadline, max_requests, clock=time.perf_counter):
    """Scan every common rank; an unstartable request spends no call quota.

    The generator resumes after each actual execution, so its next remaining
    time reflects that execution. Empty scopes are executor requests and are
    counted separately from native process calls by the caller.
    """
    executed = 0
    for index in order:
        if executed >= max_requests:
            break
        remaining = deadline - clock()
        budget = budgets[index]
        expected = float(conditional_costs.get(str(budget), budget / 1000. + .05))
        admitted = remaining > expected
        yield dict(index=index, admitted=admitted, actual_remaining_seconds=max(0., remaining),
                   expected_conditional_seconds=expected)
        if admitted:
            executed += 1
        if remaining <= 0:
            break


def worker(graph, original, args, model, scale, conditional_costs, api, model_api, channel, started, deadline):
    try:
        os.setsid()  # This isolated process group only contains this NEW job.
        prepared = time.perf_counter()
        actions, coverage, snapshot = p0.prepare_state(graph, original, args.action_seed, api)
        prefix_done = time.perf_counter()
        channel.put(dict(kind="candidate", source="paid_common_prefix",
                         members=snapshot["best_selected"], value_seconds=snapshot["best_value_seconds"],
                         worker_ready_seconds=prefix_done - started))
        scope_map = {scope.action_index: (scope, record) for scope, record in actions}
        views, descriptors = [], []
        for scope, descriptor in actions:
            warm = frozenset(snapshot["warm_by_action"][str(scope.action_index)])
            for budget_ms in args.budgets_ms:
                point = model_api["Workpoint"]("native-%dms" % budget_ms, "seconds", budget_ms / 1000.,
                    float(conditional_costs.get(str(budget_ms), budget_ms / 1000. + .05)))
                views.append(model_api["RequestView"](model_api["Request"](scope, point), warm))
                descriptors.append((scope, descriptor, budget_ms))
        inference_started = time.perf_counter()
        context = model_api["ControllerState"](snapshot["original_value_seconds"], snapshot["best_gain_seconds"], 0., ())
        if model is not None and views:
            summary = args.policy == "ResidualCheapSummary"
            packer = model_api["summary_pack"] if summary else model_api["pack"]
            batch = packer(graph, tuple(views), context, scale=scale, device="cpu",
                           static_cache=model_api["Cache"](summary_only=summary))
            with model_api["torch"].no_grad():
                predictions = (model(batch, return_details=True)["raw_gain"] * scale).detach().cpu().tolist()
        else:
            predictions = []
            for view, (_, descriptor, _) in zip(views, descriptors):
                known = (api["objective"](graph, view.warm_start) if args.policy == "Greedy"
                         else descriptor["P1_recovery_seconds"])
                predictions.append(view.request.scope.immediate_gain + known)
        inference_done = time.perf_counter()
        order = sorted(range(len(predictions)), key=lambda i: (
            -float(predictions[i]), views[i].request.scope.action_index, -descriptors[i][2]))
        calls = []
        numeric_audit = p0.native_source_audit(args.chils_source)
        native = p0.NativeCHILS(args.chils, numeric_audit, api, Path(args.out) / "temporary_native_calls")
        executed_requests = 0
        native_calls = 0
        empty_scope_requests = 0
        skipped_requests = 0
        for admission in scan_request_admission(order, [d[2] for d in descriptors], conditional_costs,
                                                 deadline, args.max_calls):
            index = admission["index"]
            view = views[index]
            scope, descriptor, budget_ms = descriptors[index]
            if not admission["admitted"]:
                skipped_requests += 1
                calls.append(dict(action_index=scope.action_index, workpoint_ms=budget_ms,
                                  status="not_admitted_insufficient_remaining", native_called=False,
                                  skipped=True, executed_request=False,
                                  actual_remaining_seconds=admission["actual_remaining_seconds"],
                                  expected_conditional_seconds=admission["expected_conditional_seconds"]))
                continue
            # Each requested action/workpoint retains the same snapshot warm.
            # No method gains hidden sequential peer prewarming in this P1.
            row = p0.request_row(graph, scope, descriptor, snapshot, budget_ms, 0, native, api)
            executed_requests += 1
            native_calls += int(row["diagnostics"]["native_called"])
            empty_scope_requests += int(not scope.replacements)
            row.update(skipped=False, executed_request=True, empty_scope_request=not scope.replacements,
                       actual_remaining_before_admission_seconds=admission["actual_remaining_seconds"])
            calls.append(row)
            channel.put(dict(kind="candidate", source="actual_native_request",
                             members=row["complete_feasible_members"], value_seconds=row["complete_value_seconds"],
                             worker_ready_seconds=time.perf_counter() - started, actual_request=row))
        channel.put(dict(kind="finished", worker_elapsed_seconds=time.perf_counter() - started,
                         coverage=coverage, scoped_requests=len(views), actual_calls=calls,
                         skipped_requests=skipped_requests, executed_requests=executed_requests,
                         actual_native_calls=native_calls, empty_scope_requests=empty_scope_requests,
                         scan_all_ranked_requests_before_call_quota_exhausted=True,
                         menu_scope_common_prefix_seconds=prefix_done - prepared,
                         inference_and_feature_preparation_seconds=inference_done - inference_started,
                         frozen_remaining_time_head_zero=True, peers_share_same_cold_snapshot=True))
    except BaseException as error:
        channel.put(dict(kind="failed", failure=dict(type=type(error).__name__, message=str(error)),
                         worker_elapsed_seconds=time.perf_counter() - started))


def guarded_complete_validation(graph, raw, expected_value, deadline, api):
    values = tuple(raw)
    if len(values) != len(set(values)) or any(not isinstance(v, int) or isinstance(v, bool) or
                                             v < 0 or v >= graph.n for v in values):
        raise ValueError("Exact unique complete original IDs required")
    selected = frozenset(values)
    for index, vertex in enumerate(values):
        if index % 16 == 0 and time.perf_counter() >= deadline:
            return None
        if graph.adjacency[vertex] & selected:
            raise ValueError("Caller received an infeasible full original-graph schedule")
    actual = api["objective"](graph, selected)
    if not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError("Caller objective differs from received member rescore")
    if time.perf_counter() >= deadline:
        return None
    return selected, actual


def cleanup_worker(process, channel):
    if process.is_alive():
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    process.join(timeout=.5)
    if process.is_alive():
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.join(timeout=.2)
    channel.close()


def run_actual_policy(graph, original, args, model, scale, conditional_costs, api, model_api):
    if os.name != "posix":
        raise RuntimeError("Caller-owned process-group execution requires actual Linux cloud")
    context = mp.get_context("fork")
    started = time.perf_counter()
    deadline = started + args.deadline_seconds
    best = frozenset(original)
    initial_value = api["objective"](graph, best)
    best_value = initial_value
    channel = context.Queue()
    process = context.Process(target=worker, args=(graph, original, args, model, scale, conditional_costs,
                                                   api, model_api, channel, started, deadline))
    process.daemon = True
    process.start()
    events = []
    stop_reason = "deadline_reached"
    while time.perf_counter() < deadline:
        try:
            event = channel.get(timeout=max(.0001, deadline - time.perf_counter()))
        except queue.Empty:
            break
        received = time.perf_counter()
        event["caller_received_seconds"] = received - started
        if event["kind"] == "candidate":
            validation_started = time.perf_counter()
            checked = guarded_complete_validation(graph, event["members"], event["value_seconds"], deadline, api)
            event["caller_complete_validation_seconds"] = time.perf_counter() - validation_started
            event["caller_validated_ready_seconds"] = time.perf_counter() - started
            event["admitted_before_deadline"] = checked is not None
            if checked is not None and checked[1] > best_value:
                best, best_value = checked
        elif event["kind"] in ("finished", "failed"):
            stop_reason = event["kind"]
            events.append(event)
            break
        events.append(event)
    # The caller keeps an already validated incumbent while the worker may
    # still be searching. Reaping the isolated worker is asynchronous cleanup,
    # never charged as if an unavailable improvement had already been returned.
    threading.Thread(target=cleanup_worker, args=(process, channel), daemon=True).start()
    prefix_values = [event["value_seconds"] for event in events if event["kind"] == "candidate" and
                     event["source"] == "paid_common_prefix" and event.get("admitted_before_deadline")]
    paid_prefix_gain = max(0., max([initial_value] + prefix_values) - initial_value)
    receipt = dict(schema="joint_recovery_stk_p1_actual_v1", policy=args.policy, fit_seed=args.fit_seed,
                   action_seed=args.action_seed, deadline_seconds=args.deadline_seconds,
                   original_members=sorted(original), selected_members=sorted(best), initial_value_seconds=initial_value,
                   returned_value_seconds=best_value, actual_gain_seconds=max(0., best_value - initial_value),
                   paid_shared_prefix_gain_seconds=paid_prefix_gain,
                   actual_gain_beyond_paid_shared_prefix_seconds=max(0., best_value - initial_value - paid_prefix_gain),
                   max_native_calls=args.max_calls, stop_reason=stop_reason, events=events,
                   original_incumbent_held_by_caller=True, preparation_prefix_inference_native_IPC_fullcheck_inside_D=True,
                   graph_model_loading_and_initial_global_greedy_are_resident_setup=True,
                   source_group_independence_not_implied_by_additional_states=True,
                   physical_hard_deadline_guaranteed=False,
                   candidate_internal_availability_different_from_caller_receipt=True)
    receipt["controller_return_sample_seconds"] = time.perf_counter() - started
    return receipt


def actual_cli(args, api, model_api):
    if args.deadline_seconds <= 0 or not math.isfinite(args.deadline_seconds):
        raise ValueError("Positive finite shared deadline required")
    if len(args.budgets_ms) != len(set(args.budgets_ms)) or any(b not in (10, 50, 200, 1000) for b in args.budgets_ms):
        raise ValueError("Explicit unique frozen native workpoints required")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Actual policy output must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    resident_started = time.perf_counter()
    graph, metadata = p0.load_graph(args.graph, api)
    original, initial_record = p0.global_incumbent(graph, api)
    model, scale, conditional_costs, checkpoint_receipt = None, 1., {}, None
    # Native source audit and binary hashing are deployment setup, repeated in
    # the child currently and therefore conservatively paid inside D too.
    native_receipt = p0.native_source_audit(args.chils_source)
    native_receipt["binary_sha256"] = p0.sha_file(args.chils)
    torch = model_api["torch"]
    torch.set_num_threads(1)
    if args.fit_root:
        fit_root = Path(args.fit_root).resolve()
        protocol = json.loads((fit_root / "protocol.json").read_text(encoding="utf-8"))
        if protocol["model_source_sha256"] != model_api["model_source_sha256"]:
            raise ValueError("Actual model runtime differs from frozen new-STK fit")
        scale = protocol["normalization"]["scale_seconds"]
        conditional_costs = protocol["conditional_cost_p95_seconds"]
        if args.policy in ("ResidualCapacity", "ResidualCheapSummary"):
            path = fit_root / (args.policy + "-seed%d" % args.fit_seed) / "final.pt"
            checkpoint = torch.load(path, map_location="cpu")
            if checkpoint["protocol_sha256"] != p0.sha_file(fit_root / "protocol.json"):
                raise ValueError("Actual predictor does not bind frozen P1 protocol")
            model = model_api["build"](args.policy)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            checkpoint_receipt = dict(path=str(path), sha256=p0.sha_file(path), scale_seconds=scale)
    elif args.policy in ("ResidualCapacity", "ResidualCheapSummary"):
        raise ValueError("Learned actual policy requires the new sealed P1 fit")
    setup_seconds = time.perf_counter() - resident_started
    external_started = time.perf_counter()
    returned = run_actual_policy(graph, original, args, model, scale, conditional_costs, api, model_api)
    # This is observed by the external caller after the allocation function
    # actually returned, including its Python return boundary.
    returned["external_caller_observed_return_seconds"] = time.perf_counter() - external_started
    returned["missed_return_sample"] = returned["external_caller_observed_return_seconds"] > args.deadline_seconds
    returned["strict_on_time_gain_seconds"] = (0.0 if returned["missed_return_sample"] else returned["actual_gain_seconds"])
    returned.update(graph=metadata, deployment_resident_setup_seconds=setup_seconds,
                    initial_global_greedy=initial_record, predictor=checkpoint_receipt,
                    native_source_and_binary=native_receipt,
                    paper_code_source_sha256=api["reference_sha256"])
    p0.write_json(out / "actual_policy.json", returned)
    print(json.dumps(dict(policy=args.policy, graph_id=graph.name, actual_gain_seconds=returned["actual_gain_seconds"],
                          caller_seconds=returned["controller_return_sample_seconds"],
                          missed_return=returned["missed_return_sample"])), flush=True)
