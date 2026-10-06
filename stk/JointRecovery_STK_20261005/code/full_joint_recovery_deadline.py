"""Isolated full online STK recovery controller, with sealed P1 predictors.

Dynamic marginal/cost priorities, same-action verified warm updates, spent
requests and exact observable embedding reuse are restored. Remaining-head
conditioning is explicitly frozen zero, as in the new-STK fit. This is not a
claim of a trained remaining-time head or a byte-identical old-paper recipe.
Imports perform no fit, native call, dataset load or experiment launch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

import p0_recovery_probes as p0
import p1_fit_and_allocate as p1
from p1_actual_policy import cleanup_worker, guarded_complete_validation

SCHEMA = "joint_recovery_stk_full_actual_v1"
POLICIES = ("FullCapacity", "FullCheapSummary", "FullGreedy", "FullP1")
MENU = (10, 50, 200, 1000)
MAX_CALLS = 8


def observable_unit_key(ordered_replacements, factors, warm, scale):
    """Same structural key as the frozen factorized packer; graph/model bind separately."""
    return (tuple(ordered_replacements), tuple(tuple(f) for f in factors),
            frozenset(warm), float(scale))


def request_key(request):
    return (int(request["scope"].action_index), p0.CAP, int(request["budget_ms"]))


def priority_order(predictions, best_gain, costs):
    if len(predictions) != len(costs) or any(not math.isfinite(float(v)) for v in predictions):
        raise ValueError("One finite prediction per pending request required")
    if any(not math.isfinite(c) or c <= 0 for c in costs):
        raise ValueError("Positive finite common conditional costs required")
    return sorted(range(len(predictions)), key=lambda i: (
        -max(0., float(predictions[i]) - best_gain) / costs[i],
        -max(0., float(predictions[i])) / costs[i], i))


def run_allocation(requests, snapshot, conditional_costs, deadline, predict, execute,
                   admit, reward, on_start, clock=time.perf_counter, max_calls=MAX_CALLS):
    """Shared closed loop; every response must be acknowledged by its real caller.

    The engine is also usable with a fake clock/executor for the one small
    non-native guard. The actual execute callback validates original memberships,
    and admit waits for the independent parent before a response warms its action.
    """
    state = copy.deepcopy(snapshot)
    spent = {tuple(key) for key in state["spent_requests"]}
    calls, rounds = [], []
    reason = "query_cap"
    for round_index in range(max_calls):
        remaining = deadline - clock()
        pending = [r for r in requests if request_key(r) not in spent and
                   conditional_costs[str(r["budget_ms"])] < remaining and r["budget_ms"] / 1000. < remaining]
        if not pending:
            reason = "decision_deadline" if remaining <= 0 else "no_affordable_unspent_request"
            break
        inference_started = clock()
        predictions, profile = predict(pending, state)
        inference_finished = clock()
        costs = [conditional_costs[str(r["budget_ms"])] for r in pending]
        order = priority_order(predictions, state["best_gain_seconds"], costs)
        # Recheck after paid packing/inference; skip a now-unaffordable rank
        # without spending the eight-call quota or forgetting cheaper ranks.
        selected = next((i for i in order if costs[i] < deadline - clock() and
                         pending[i]["budget_ms"] / 1000. < deadline - clock()), None)
        round_record = dict(round_index=round_index, best_gain_before_seconds=state["best_gain_seconds"],
                            inference_and_preparation_seconds=inference_finished - inference_started,
                            remaining_after_inference_seconds=max(0., deadline - inference_finished),
                            pending_keys=[list(request_key(r)) for r in pending],
                            prediction_seconds=[float(v) for v in predictions],
                            priority_keys=[list(request_key(pending[i])) for i in order], **profile)
        rounds.append(round_record)
        if selected is None:
            reason = "no_affordable_request_after_inference"
            break
        request = pending[selected]
        key = request_key(request)
        spent.add(key)  # Includes a failed or late launch, without a free retry.
        started = clock()
        on_start(request, state, dict(round_index=round_index, request_key=list(key),
                 actual_remaining_seconds=max(0., deadline - started),
                 predicted_gain_seconds=float(predictions[selected]),
                 predicted_marginal_per_second=max(0., float(predictions[selected]) - state["best_gain_seconds"]) / costs[selected],
                 expected_conditional_seconds=costs[selected]))
        # p0.request_row sees the prior snapshot's spent set, not its own just
        # admitted key. The controller's external spent set already contains it.
        row = execute(request, state)
        acknowledged = admit(request, row)
        timely = bool(acknowledged and acknowledged["admitted_before_deadline"] and clock() < deadline)
        warm_before = frozenset(state["warm_by_action"][str(key[0])])
        warm_after = warm_before
        if timely:
            recovered = frozenset(row["returned_recovery_members"])
            if reward(recovered) > reward(warm_before):
                warm_after = recovered
                state["warm_by_action"][str(key[0])] = sorted(recovered)
            state["best_selected"] = list(acknowledged["best_selected"])
            state["best_value_seconds"] = float(acknowledged["best_value_seconds"])
            state["best_gain_seconds"] = max(0., state["best_value_seconds"] - state["original_value_seconds"])
        state["spent_requests"] = [list(k) for k in sorted(spent)]
        row.update(controller_round=round_index, request_key=list(key),
                   acknowledged_before_deadline=timely,
                   warm_changed=warm_after != warm_before, warm_after_members=sorted(warm_after),
                   controller_best_gain_after_seconds=state["best_gain_seconds"],
                   executed_request=True, empty_scope_request=not request["scope"].replacements)
        calls.append(row)
        if not timely:
            reason = "caller_admission_deadline"
            break
    return dict(actual_calls=calls, allocation_rounds=rounds, controller_stop_reason=reason,
                spent_requests=[list(k) for k in sorted(spent)], executed_requests=len(calls),
                actual_native_calls=sum(int(r["diagnostics"]["native_called"]) for r in calls),
                empty_scope_requests=sum(int(r["empty_scope_request"]) for r in calls),
                final_controller_gain_seconds=state["best_gain_seconds"])


class PredictionSession:
    def __init__(self, graph, model, policy, scale, api, model_api):
        self.graph, self.model, self.policy, self.scale = graph, model, policy, scale
        self.api, self.model_api = api, model_api
        self.summary = policy == "FullCheapSummary"
        self.static = model_api["Cache"](summary_only=self.summary) if model is not None else None
        self.embeddings = model_api["EmbeddingCache"]() if model is not None and not self.summary else None

    def __call__(self, requests, state):
        views = tuple(self.model_api["RequestView"](self.model_api["Request"](r["scope"],
            self.model_api["Workpoint"]("native-%dms" % r["budget_ms"], "seconds", r["budget_ms"] / 1000.,
                                      r["expected_cost_seconds"])),
            frozenset(state["warm_by_action"][str(r["scope"].action_index)])) for r in requests)
        if self.model is None:
            predictions = [r["scope"].immediate_gain + (self.api["objective"](self.graph, view.warm_start)
                if self.policy == "FullGreedy" else r["descriptor"]["P1_recovery_seconds"])
                for r, view in zip(requests, views)]
            return predictions, dict(encoded_units=0, reused_units=0, cache_applicable=False,
                                     frozen_remaining_time_head_zero=True)
        context = self.model_api["ControllerState"](state["original_value_seconds"],
            state["best_gain_seconds"], 0., tuple((a, cap, "native-%dms" % b) for a, cap, b in state["spent_requests"]))
        if self.embeddings is not None:
            self.embeddings.bind(self.graph, self.model)
        packer = self.model_api["summary_pack"] if self.summary else self.model_api["pack"]
        kwargs = dict(scale=self.scale, device="cpu", static_cache=self.static)
        if self.embeddings is not None:
            kwargs["embedding_cache"] = self.embeddings
        batch = packer(self.graph, views, context, **kwargs)
        if self.embeddings is not None:
            # Validate the actual frozen packer key, rather than using an
            # action-id/graph-id approximation to permit reuse.
            mapping = batch["request_unit"].tolist()
            for i, view in enumerate(views):
                factors = self.static.get(self.graph, view.request.scope, self.scale)[0]
                expected = observable_unit_key(view.request.scope.replacements, factors, view.warm_start, self.scale)
                if batch["unit_keys"][mapping[i]] != expected:
                    raise ValueError("Model cache key differs from exact observable structural unit")
        with self.model_api["torch"].no_grad():
            details = self.model(batch, return_details=True, embedding_cache=self.embeddings) if self.embeddings is not None else self.model(batch, return_details=True)
            values = (details["raw_gain"] * self.scale).detach().cpu().tolist()
        new = len(batch["new_unit_indices"]) if self.embeddings is not None else 0
        units = batch["unit_count"] if self.embeddings is not None else 0
        return values, dict(encoded_units=new, reused_units=units - new, cache_applicable=self.embeddings is not None,
                            cache_bound_to_exact_graph_model_parameter_versions=True,
                            frozen_remaining_time_head_zero=True)


def wait_ack(feedback, token, deadline):
    while time.perf_counter() < deadline:
        try:
            response = feedback.get(timeout=max(.0001, deadline - time.perf_counter()))
        except queue.Empty:
            return None
        if response["ack_token"] != token:
            raise ValueError("Caller acknowledgement request identity changed")
        return response
    return None


def worker(graph, original, args, model, scale, costs, api, model_api, channel, feedback, started, deadline):
    try:
        os.setsid()
        prepared = time.perf_counter()
        actions, coverage, snapshot = p0.prepare_state(graph, original, args.action_seed, api)
        prefix_done = time.perf_counter()
        channel.put(dict(kind="candidate", source="paid_common_prefix", ack_token="prefix",
                         members=snapshot["best_selected"], value_seconds=snapshot["best_value_seconds"],
                         worker_ready_seconds=prefix_done - started))
        prefix_ack = wait_ack(feedback, "prefix", deadline)
        if prefix_ack is None or not prefix_ack["admitted_before_deadline"]:
            channel.put(dict(kind="finished", controller_stop_reason="paid_prefix_caller_deadline",
                             actual_calls=[], allocation_rounds=[], executed_requests=0, actual_native_calls=0,
                             empty_scope_requests=0, worker_elapsed_seconds=time.perf_counter() - started))
            return
        snapshot["best_selected"] = prefix_ack["best_selected"]
        snapshot["best_value_seconds"] = prefix_ack["best_value_seconds"]
        snapshot["best_gain_seconds"] = max(0., prefix_ack["best_value_seconds"] - snapshot["original_value_seconds"])
        predictor = PredictionSession(graph, model, args.policy, scale, api, model_api)
        native = p0.NativeCHILS(args.chils, p0.native_source_audit(args.chils_source), api,
                                Path(args.out) / "temporary_native_calls")
        requests = [dict(scope=scope, descriptor=descriptor, budget_ms=b,
                         expected_cost_seconds=costs[str(b)]) for scope, descriptor in actions for b in args.budgets_ms]

        def on_start(request, state, record):
            channel.put(dict(kind="request_started", source="actual_native_request", **record,
                             empty_scope_request=not request["scope"].replacements,
                             actual_warm_members=state["warm_by_action"][str(request["scope"].action_index)],
                             snapshot_best_gain_seconds=state["best_gain_seconds"],
                             worker_ready_seconds=time.perf_counter() - started))

        def execute(request, state):
            return p0.request_row(graph, request["scope"], request["descriptor"], state,
                                  request["budget_ms"], 0, native, api)

        def admit(request, row):
            token = "%d:%d" % (request["scope"].action_index, request["budget_ms"])
            channel.put(dict(kind="candidate", source="actual_native_request", ack_token=token,
                             members=row["complete_feasible_members"], value_seconds=row["complete_value_seconds"],
                             actual_request=row, worker_ready_seconds=time.perf_counter() - started))
            return wait_ack(feedback, token, deadline)

        result = run_allocation(requests, snapshot, costs, deadline, predictor, execute, admit,
                                lambda members: api["objective"](graph, members), on_start)
        channel.put(dict(kind="finished", coverage=coverage, scoped_requests=len(requests), **result,
                         menu_scope_common_prefix_seconds=prefix_done - prepared,
                         worker_elapsed_seconds=time.perf_counter() - started,
                         dynamic_gstar_warm_spent_and_cost_priorities=True, frozen_remaining_time_head_zero=True))
    except BaseException as error:
        channel.put(dict(kind="failed", failure=dict(type=type(error).__name__, message=str(error)),
                         worker_elapsed_seconds=time.perf_counter() - started))


def run_full_policy(graph, original, args, model, scale, costs, api, model_api):
    if os.name != "posix":
        raise RuntimeError("Actual caller-owned process-group execution requires Linux")
    context = mp.get_context("fork")
    started = time.perf_counter()
    deadline = started + args.deadline_seconds
    best = frozenset(original)
    initial_value = api["objective"](graph, best)
    best_value = initial_value
    channel, feedback = context.Queue(), context.Queue()
    process = context.Process(target=worker, args=(graph, original, args, model, scale, costs,
                              api, model_api, channel, feedback, started, deadline))
    process.daemon = True
    process.start()
    events, stop_reason = [], "deadline_reached"
    while time.perf_counter() < deadline:
        try:
            event = channel.get(timeout=max(.0001, deadline - time.perf_counter()))
        except queue.Empty:
            break
        event["caller_received_seconds"] = time.perf_counter() - started
        if event["kind"] == "candidate":
            check_started = time.perf_counter()
            checked = guarded_complete_validation(graph, event["members"], event["value_seconds"], deadline, api)
            event["caller_complete_validation_seconds"] = time.perf_counter() - check_started
            event["caller_validated_ready_seconds"] = time.perf_counter() - started
            event["admitted_before_deadline"] = checked is not None
            if checked is not None and checked[1] > best_value:
                best, best_value = checked
            # Feedback occurs only AFTER parent full validation; its IPC and
            # membership serialization are charged to the same D.
            feedback.put(dict(ack_token=event["ack_token"], admitted_before_deadline=checked is not None,
                              best_selected=sorted(best), best_value_seconds=best_value))
        elif event["kind"] in ("finished", "failed"):
            stop_reason = event["kind"]
            events.append(event)
            break
        events.append(event)
    threading.Thread(target=cleanup_worker, args=(process, channel), daemon=True).start()
    # Queue closure does not discard the caller's independently held schedule.
    feedback.close()
    prefix_values = [e["value_seconds"] for e in events if e["kind"] == "candidate" and
                     e["source"] == "paid_common_prefix" and e.get("admitted_before_deadline")]
    prefix_gain = max(0., max([initial_value] + prefix_values) - initial_value)
    starts = [e for e in events if e["kind"] == "request_started"]
    candidates = [e for e in events if e["kind"] == "candidate" and e["source"] == "actual_native_request"]
    finished = next((e for e in events if e["kind"] == "finished"), {})
    record = dict(schema=SCHEMA, status="FULL_JOINT_RECOVERY_ACTUAL_COMPLETE",
        policy=args.policy, fit_seed=args.fit_seed, action_seed=args.action_seed,
        deadline_seconds=args.deadline_seconds, original_members=sorted(original), selected_members=sorted(best),
        initial_value_seconds=initial_value, returned_value_seconds=best_value,
        actual_gain_seconds=max(0., best_value - initial_value), paid_shared_prefix_gain_seconds=prefix_gain,
        actual_gain_beyond_paid_shared_prefix_seconds=max(0., best_value - initial_value - prefix_gain),
        max_native_calls=MAX_CALLS, stop_reason=stop_reason, events=events,
        executed_requests=len(starts), completed_requests=len(candidates),
        actual_native_calls=(finished.get("actual_native_calls") if finished else None),
        observed_completed_native_calls=sum(int(e["actual_request"]["diagnostics"]["native_called"]) for e in candidates),
        requests_without_observed_completed_response=len(starts) - len(candidates),
        empty_scope_requests=sum(int(e["empty_scope_request"]) for e in starts),
        caller_validated_native_responses=sum(int(e.get("admitted_before_deadline", False)) for e in candidates),
        failed_completed_requests=sum(int(e["actual_request"]["failed"]) for e in candidates),
        dynamic_gstar_warm_spent_and_cost_priorities=True,
        same_action_warm_updates_only_after_parent_validation_ack=True,
        exact_observable_embedding_cache_with_changed_warm_reencoding=True,
        remaining_time_head_feature_frozen_zero=True, trained_remaining_time_adaptation_established=False,
        byte_identical_old_paper_recipe=False, sealed_new_STK_predictor_online_migration=True,
        original_incumbent_held_by_caller=True, preparation_prefix_inference_native_IPC_fullcheck_inside_D=True,
        graph_model_loading_and_initial_global_greedy_are_resident_setup=True,
        physical_hard_deadline_guaranteed=False)
    record["controller_return_sample_seconds"] = time.perf_counter() - started
    return record


def actual_cli(args, api, model_api):
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Full controller output must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    resident_started = time.perf_counter()
    graph, metadata = p0.load_graph(args.graph, api)
    original, initial_record = p0.global_incumbent(graph, api)
    initial_mask = np.zeros(graph.n, dtype=np.int8)
    initial_mask[list(original)] = 1
    initial_mask_sha = hashlib.sha256(initial_mask.tobytes(order="C")).hexdigest()
    fit_root = Path(args.fit_root).resolve()
    completion = json.loads((fit_root / "completion.json").read_text(encoding="utf-8"))
    protocol = json.loads((fit_root / "protocol.json").read_text(encoding="utf-8"))
    protocol_sha = p0.sha_file(fit_root / "protocol.json")
    if completion["status"] != "ACTUAL_P1_FIT_COMPLETE" or len(completion["fits"]) != 4 or completion["protocol_sha256"] != protocol_sha:
        raise ValueError("All four original sealed fits required")
    if protocol["model_source_sha256"] != model_api["model_source_sha256"]:
        raise ValueError("Full controller predictor runtime differs from sealed fits")
    scale = float(protocol["normalization"]["scale_seconds"])
    costs = {str(b): float(protocol["conditional_cost_p95_seconds"][str(b)]) for b in MENU}
    if not math.isfinite(scale) or scale <= 0 or any(not math.isfinite(c) or c <= 0 for c in costs.values()):
        raise ValueError("Finite positive sealed normalization and costs required")
    model, checkpoint_receipt = None, None
    model_api["torch"].set_num_threads(1)
    if args.policy in ("FullCapacity", "FullCheapSummary"):
        variant = "ResidualCapacity" if args.policy == "FullCapacity" else "ResidualCheapSummary"
        path = fit_root / (variant + "-seed%d" % args.fit_seed) / "final.pt"
        expected = next(row["sha256"] for row in completion["fits"] if row["variant"] == variant and row["seed"] == args.fit_seed)
        if p0.sha_file(path) != expected:
            raise ValueError("Selected checkpoint differs from sealed completion")
        checkpoint = model_api["torch"].load(path, map_location="cpu")
        if checkpoint["protocol_sha256"] != protocol_sha or not checkpoint["remaining_time_feature_frozen_zero"] or checkpoint["scale_seconds"] != scale:
            raise ValueError("Predictor normalization/head protocol changed")
        model = model_api["build"](variant)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        checkpoint_receipt = dict(path=str(path), sha256=expected, scale_seconds=scale, variant=variant)
    native_receipt = p0.native_source_audit(args.chils_source)
    native_receipt["binary_sha256"] = p0.sha_file(args.chils)
    setup_seconds = time.perf_counter() - resident_started
    external_started = time.perf_counter()
    returned = run_full_policy(graph, original, args, model, scale, costs, api, model_api)
    returned["external_caller_observed_return_seconds"] = time.perf_counter() - external_started
    returned["missed_return_sample"] = returned["external_caller_observed_return_seconds"] > args.deadline_seconds
    returned["strict_on_time_gain_seconds"] = 0. if returned["missed_return_sample"] else returned["actual_gain_seconds"]
    returned["strict_on_time_gain_beyond_paid_shared_prefix_seconds"] = 0. if returned["missed_return_sample"] else returned["actual_gain_beyond_paid_shared_prefix_seconds"]
    returned.update(graph=metadata, deployment_resident_setup_seconds=setup_seconds,
        initial_mask_sha256=initial_mask_sha,
        initial_global_greedy=initial_record, predictor=checkpoint_receipt, native_source_and_binary=native_receipt,
        fit_protocol_sha256=protocol_sha, native_workpoints_ms=list(MENU),
        paper_code_source_sha256=api["reference_sha256"], full_controller_source_sha256=p0.sha_file(__file__),
        common_parent_validator_source_sha256=p0.sha_file(Path(__file__).with_name("p1_actual_policy.py")),
        cpu_affinity=sorted(os.sched_getaffinity(0)), source_group_independence_not_implied_by_additional_states=True)
    p0.write_json(out / "full_joint_recovery.json", returned)
    print(json.dumps(dict(policy=args.policy, graph_id=graph.name, strict_gain_seconds=returned["strict_on_time_gain_seconds"],
                          caller_seconds=returned["external_caller_observed_return_seconds"],
                          missed_return=returned["missed_return_sample"])), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-root", "fit-root", "graph", "out", "chils", "chils-source"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--fit-seed", type=int, choices=p1.FIT_SEEDS, default=17)
    parser.add_argument("--action-seed", type=int, default=17)
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--budgets-ms", type=int, nargs="+", default=list(MENU))
    parser.add_argument("--max-calls", type=int, choices=(MAX_CALLS,), default=MAX_CALLS)
    parser.add_argument("--cpu-index", type=int, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.deadline_seconds) or args.deadline_seconds <= 0 or tuple(args.budgets_ms) != MENU or args.action_seed != 17:
        parser.error("Positive fixed D, exact menu 10 50 200 1000 and action seed17 required")
    return args


def main():
    args = parse_args()
    if os.name != "posix":
        raise RuntimeError("Actual full controller runs only in the existing Linux cloud runtime")
    available = sorted(os.sched_getaffinity(0))
    if args.cpu_index < 0 or args.cpu_index >= len(available):
        raise ValueError("CPU index must select an actually available CPU")
    os.sched_setaffinity(0, {available[args.cpu_index]})
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    api = p0.load_runtime(args.runtime_root)
    model_api = p1.load_model_api(args.runtime_root)
    from joint_recovery.v4_factorized_model import DecisionEmbeddingCache
    model_api["EmbeddingCache"] = DecisionEmbeddingCache
    actual_cli(args, api, model_api)


if __name__ == "__main__":
    main()
