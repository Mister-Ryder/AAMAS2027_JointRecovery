"""Source-only V4 fresh split and actual warm-aligned label collection API.

No CLI launches solvers or training. The optional CLI emits specifications
only. A caller must freeze and supply its calibrated workpoints/deadline,
backend and collection lifecycle before invoking collect_label_groups.
Alternatives clone immutable states; their outcomes never warm their peers.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np

from joint_recovery.core import _validate_state
from joint_recovery.v4_budgeted_recovery import (
    ControllerState, RepairAttempt, Request, RequestView, Workpoint,
    _check_recovery, objective)
try:
    from .v4_neighborhoods import coordination_cells
    from .v4_residual_common import ExecutedWarmScopeCache as NeighborhoodScopeCache
except ImportError:
    from v4_neighborhoods import coordination_cells
    from v4_residual_common import ExecutedWarmScopeCache as NeighborhoodScopeCache


CAPS = (256,)
NATIVE_SLICES = (.01, .05, .2)
SPLITS = {"training": (20420000, 3), "validation": (20430000, 1)}
CHILS_SHA256 = "19610c03f334c6267f94543ad3053d792cba56e9ae211fceb6c36f21750c88a0"


def _cells():
    cells = []
    for size in (64, 256):
        for topology in ("erdos", "components", "bipartite"):
            for density in (.08, .20, .45):
                cells.append(dict(domain="menu", replacements=size, menus=8,
                    density=density, coupling=.04, topology=topology))
    for passes in (48, 96):
        for grounds in (2, 4, 8):
            cells.append(dict(domain="resource", satellites=8, grounds=grounds,
                              passes=passes, alternatives=6))
    return cells


def fresh_split_specs(split):
    """Declare 72 train / 24 validation graphs; no confirmation generation API."""
    if split not in SPLITS:
        raise ValueError("Only fresh training and validation specifications are authorized")
    start, repetitions = SPLITS[split]
    return tuple(dict(cell, split=split, seed=start + repetition*24 + index)
                 for repetition in range(repetitions)
                 for index, cell in enumerate(_cells()))


def specification_manifest():
    groups = {name: fresh_split_specs(name) for name in SPLITS}
    all_seeds = [s["seed"] for specs in groups.values() for s in specs]
    if len(set(all_seeds)) != 96:
        raise AssertionError("Fresh graph splits overlap")
    return dict(status="specifications_only_not_a_frozen_execution_queue",
        splits=groups, confirmation_graphs_generated=0,
        cells_per_repetition=24, graph_split_before_actions_states_queries=True,
        no_graphs_constructed=True, no_backend_calls=True, no_training=True)


def construct_fresh_case(spec):
    """Explicit caller action; uses original families, never chooses by outcome."""
    expected = fresh_split_specs(spec.get("split"))
    if dict(spec) not in expected:
        raise ValueError("Graph specification is outside the complete declared fresh split")
    try:
        from .v3_domains import construct_case
    except ImportError:
        from v3_domains import construct_case
    # Original domain construction/old action generation is resident dataset
    # preparation, not free per-decision proposal work. The new menu is built
    # afresh inside collect_label_groups and its actual preparation is timed.
    graph, selected, _old_actions, metadata = construct_case(dict(spec), max_actions=16)
    return graph, selected, metadata


def split_graph_weight(spec):
    """Equal domains, equal graphs within domain, fixed before outcomes."""
    specs = fresh_split_specs(spec.get("split"))
    if dict(spec) not in specs:
        raise ValueError("Loss weighting requires an exact declared fresh specification")
    count = sum(item["domain"] == spec["domain"] for item in specs)
    return .5/count


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def graph_identity(graph):
    """New explicit native schema; integer rewards never float-cast for identity."""
    return _sha_json(dict(schema="v4_native_graph_v1", rewards_dtype=str(graph.weights.dtype),
        weights=[w.item() for w in graph.weights], agents=[int(a) for a in graph.agents],
        edges=sorted((v, u) for v in range(graph.n) for u in graph.adjacency[v] if v < u)))


@dataclass(frozen=True)
class CollectionConfig:
    deadline_seconds: float
    workpoints: tuple
    calibration_sha256: str
    history_max_queries: int = 1
    caps: tuple = CAPS
    require_chils_receipt: bool = True

    def __post_init__(self):
        object.__setattr__(self, "caps", tuple(self.caps))
        object.__setattr__(self, "workpoints", tuple(self.workpoints))
        if not isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("Positive calibrated complete decision deadline required")
        if self.caps != CAPS:
            raise ValueError("The first source protocol fixes fixedR256")
        if (tuple(w.amount for w in self.workpoints) != NATIVE_SLICES or
                any(w.kind != "seconds" for w in self.workpoints) or
                len({w.name for w in self.workpoints}) != 3):
            raise ValueError("All three distinct calibrated native seconds points required")
        if (len(self.calibration_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in self.calibration_sha256)):
            raise ValueError("An explicit calibration receipt SHA256 is required")
        if (not isinstance(self.history_max_queries, Integral) or
                isinstance(self.history_max_queries, bool) or self.history_max_queries != 1):
            raise ValueError("The first protocol freezes one actually launched round-robin prefix request")
        if not isinstance(self.require_chils_receipt, bool):
            raise ValueError("Executor receipt requirement must be explicit boolean")


@dataclass(frozen=True)
class LabelState:
    name: str
    original: frozenset
    best_selected: frozenset
    initial_value: float
    best_gain: float
    warm_by_action: tuple
    spent_requests: tuple
    prefix_elapsed_seconds: float
    remaining_seconds: float

    def view(self, request):
        warm = dict(self.warm_by_action)[request.scope.action_index]
        return RequestView(request, warm & frozenset(request.scope.replacements))

    def controller_state(self):
        return ControllerState(self.initial_value, self.best_gain,
                               self.remaining_seconds, self.spent_requests)

    def record(self):
        return dict(name=self.name, original=sorted(self.original),
            best_selected=sorted(self.best_selected), initial_value=self.initial_value,
            best_gain=self.best_gain,
            warm_by_action=[dict(action_index=i, recovered=sorted(s)) for i, s in self.warm_by_action],
            spent_requests=[list(k) for k in self.spent_requests],
            prefix_elapsed_seconds=self.prefix_elapsed_seconds,
            remaining_seconds=self.remaining_seconds)


def _clock(clock, previous=None):
    now = float(clock())
    if not isfinite(now) or (previous is not None and now < previous):
        raise ValueError("Finite monotonic label timestamps required")
    return now


def _execute_clone(graph, request, state, backend, clock, absolute_deadline=None, require_chils_receipt=True):
    """One actual execution in a cloned state; no original state is mutated."""
    view = state.view(request); scope = request.scope; point = request.workpoint
    row = dict(request_key=list(request.key), action_index=scope.action_index,
        cap=scope.cap, workpoint=point.name, native_seconds=point.amount,
        action_inserts=list(scope.inserts),
        action_extra_releases=sorted(scope.displaced-frozenset(
            v for c in scope.inserts for v in graph.adjacency[c]&state.original)),
        expected_seconds=point.expected_seconds, scope_nodes=list(scope.replacements),
        base=sorted(scope.base), displaced=sorted(scope.displaced), immediate_gain=scope.immediate_gain,
        actual_warm_start=sorted(view.warm_start), state_sha256=_sha_json(state.record()),
        context_best_gain=state.best_gain, context_remaining_seconds=state.remaining_seconds,
        context_spent_requests=[list(k) for k in state.spent_requests],
        launched=False, spent=False, available=False, membership_valid=False,
        raw_supervision_valid=False, returned_recovery=None, signed_return_delta=None,
        returned_gain=0, admitted_gain=0, on_time=False, elapsed_seconds=None,
        failure=None, diagnostics={},call_started_timestamp=None,
        absolute_history_deadline=absolute_deadline)
    if request.key in state.spent_requests:
        row["status"] = "already_spent_in_history"; return row
    if (state.remaining_seconds <= 0 or point.amount > state.remaining_seconds or
            point.expected_seconds > state.remaining_seconds):
        row["status"] = "not_startable"; return row
    row["available"] = True
    try:
        started = _clock(clock)
    except Exception as error:
        row.update(status="timestamp_before_launch_failed",
                   failure=dict(type=type(error).__name__, message=str(error)))
        return row
    call_remaining = (state.remaining_seconds if absolute_deadline is None else
                      min(state.remaining_seconds, max(0., absolute_deadline-started)))
    row['call_started_timestamp']=started
    row["actual_remaining_before_call"] = call_remaining
    if point.amount > call_remaining or point.expected_seconds > call_remaining:
        row.update(available=False, status="not_startable_before_history_launch")
        return row
    row.update(launched=True, spent=True, status="request_failed")
    try:
        result = backend(graph, scope, point, view.warm_start, call_remaining)
        if not isinstance(result, RepairAttempt) or result.recovered is None:
            raise ValueError("A materialized RepairAttempt membership is required")
        actual = _check_recovery(graph, scope, result.recovered)
        signed = objective(graph, scope.base | actual) - state.initial_value
        algebra = scope.immediate_gain + objective(graph, actual)
        if ((graph.weights.dtype.kind in "iu" and signed != algebra) or
                (graph.weights.dtype.kind not in "iu" and
                 not np.isclose(signed, algebra, rtol=1e-12, atol=1e-9))):
            raise ValueError("Fixed-base signed objective identity is inconsistent")
        row.update(membership_valid=True, returned_recovery=sorted(actual),
                   signed_return_delta=signed, returned_gain=max(0, signed),
                   candidate_value=objective(graph, scope.base | actual),
                   recovery_value=objective(graph, actual),
                   previous_warm_value=objective(graph, dict(state.warm_by_action)[scope.action_index]))
        diagnostics = dict(result.diagnostics or {})
        json.dumps(diagnostics, allow_nan=False)
        if require_chils_receipt and diagnostics.get("binary_sha256") != CHILS_SHA256:
            raise ValueError("Actual pinned CHILS receipt is missing or differs from the frozen executor")
        row.update(status=str(result.status), diagnostics=diagnostics)
    except Exception as error:
        row.update(failure=dict(type=type(error).__name__, message=str(error)),
                   status="request_failed", diagnostics={})
    try:
        ready = _clock(clock, started)
        row["elapsed_seconds"] = ready - started
        row["validation_ready_timestamp"] = ready
        row["on_time"] = row["failure"] is None and ready-started < call_remaining
        if absolute_deadline is not None:
            row["on_time"] = row["on_time"] and ready < absolute_deadline
    except Exception as error:
        row.update(failure=dict(type=type(error).__name__, message=str(error)),
                   status="timestamp_failed", on_time=False)
    row["raw_supervision_valid"] = row["membership_valid"] and row["failure"] is None
    row["admitted_gain"] = row["returned_gain"] if row["on_time"] else 0
    return row


def _freeze_state(name, original, best_selected, best_value, initial, warm, spent, started, now, deadline):
    return LabelState(name, original, best_selected, initial, max(0, best_value-initial),
        tuple(sorted(warm.items())), tuple(sorted(spent)), now-started, max(0., deadline-now))


def _loss_weights(rows, graph_weight, state_weight=.5):
    valid = sum(row["raw_supervision_valid"] for row in rows)
    mask_valid = sum(row["raw_supervision_valid"] and bool(row["scope_nodes"]) for row in rows)
    available = sum(row["available"] for row in rows)
    for row in rows:
        row["regression_request_weight"] = graph_weight*state_weight/max(1, valid) if row["raw_supervision_valid"] else 0.
        row["auxiliary_request_weight"] = (graph_weight*state_weight/max(1, mask_valid)
            if row["raw_supervision_valid"] and row["scope_nodes"] else 0.)
        row["ranking_request_weight"] = graph_weight*state_weight/max(1, available) if row["available"] else 0.


def collect_label_groups(graph, selected, *, seed, config, backend,
        resource_cliques=(), graph_weight=1., clock=perf_counter):
    """Collect two immutable same-state groups, using only real history outputs.

    This is counterfactual finite-label collection, NOT timed policy evaluation.
    Each alternative resets its remaining budget to its common snapshot. The
    history prefix executes first, independently of all alternative labels.
    Never invoke this API implicitly from a source/plan or dry-guard command.
    """
    if not isinstance(config, CollectionConfig):
        raise TypeError("An explicit frozen collection configuration is required")
    if not isfinite(graph_weight) or graph_weight <= 0:
        raise ValueError("Positive predeclared graph loss weight required")
    input_identity = graph_identity(graph)
    started = _clock(clock); deadline = started + config.deadline_seconds
    if not isfinite(deadline): raise ValueError("Finite absolute preparation deadline required")
    selected = tuple(selected)
    if (any(not isinstance(v, Integral) or isinstance(v, bool) for v in selected) or
            len(selected) != len(set(selected))):
        raise ValueError("Original membership requires exact distinct integer IDs")
    original = _validate_state(graph, selected); initial = objective(graph, original)
    best_selected, best_value = original, initial
    cells = coordination_cells(graph, original, seed=seed)
    cache = NeighborhoodScopeCache(graph, original, resource_cliques)
    warm = {}; requests = []; coverage = []; known = []; previous = started
    for index, (action, info) in enumerate(cells):
        info = dict(info, cell_index=index)
        if action is None:
            coverage.append(info); continue
        info.update(inserts=list(action.inserts), releases=list(action.releases), scopes=[])
        warm[index] = frozenset(); seen = {}
        for cap in config.caps:
            scope = cache.scope(index, action, cap)
            if scope.replacements in seen:
                info["scopes"].append(dict(cap=cap, status="actual_scope_alias", alias_cap=seen[scope.replacements]))
                continue
            seen[scope.replacements] = cap
            actual = _check_recovery(graph, scope, cache.initial_known_warm(scope))
            value = objective(graph, scope.base | actual)
            actual_value = objective(graph, actual)
            previous_value = objective(graph, warm[index])
            ready = _clock(clock, previous); previous = ready
            timely = ready < deadline
            known.append(dict(action_index=index, cap=cap, recovered=sorted(actual),
                signed_delta=value-initial, validation_ready_elapsed=ready-started,
                validation_ready_timestamp=ready,on_time=timely))
            if timely:
                if actual_value > previous_value: warm[index] = actual
                if value > best_value: best_selected, best_value = scope.base | actual, value
            info["scopes"].append(dict(cap=cap, status="prepared" if timely else "prepared_after_deadline",
                nodes=list(scope.replacements), base=sorted(scope.base), displaced=sorted(scope.displaced)))
            requests.extend(Request(scope, point) for point in config.workpoints)
        coverage.append(info)
    ready = _clock(clock, previous)
    initial_state = _freeze_state("initial_executed_greedy", original, best_selected, best_value,
                                 initial, warm, set(), started, ready, deadline)
    history = []; spent = set(); history_executed = 0
    history_point = next(point for point in config.workpoints if point.amount == .05)
    ordered = [r for r in requests if r.scope.cap == 256 and r.workpoint == history_point]
    for request in ordered:
        if history_executed >= config.history_max_queries:
            history.append(dict(request_key=list(request.key), action_index=request.scope.action_index,
                status="outside_fixed_history_prefix", launched=False, spent=False, available=False,
                on_time=False, failure=None))
            continue
        now = _clock(clock, ready); ready = now
        context = _freeze_state("round_robin_prefix", original, best_selected, best_value,
                               initial, warm, spent, started, now, deadline)
        row = _execute_clone(graph, request, context, backend, clock, absolute_deadline=deadline,
                             require_chils_receipt=config.require_chils_receipt)
        history.append(row)
        if row["spent"]:
            spent.add(request.key); history_executed += 1
        if row["on_time"]:
            actual = frozenset(row["returned_recovery"])
            if row["recovery_value"] > row["previous_warm_value"]:
                warm[request.scope.action_index] = actual
            value = row["candidate_value"]
            if value > best_value: best_selected, best_value = request.scope.base | actual, value
        # Preserve complete planned RR coverage, including unstartable tails;
        # those rows are not free executed/spent calls.
    ready = _clock(clock, ready)
    history_state = _freeze_state("round_robin_actual_history", original, best_selected,
        best_value, initial, warm, spent, started, ready, deadline)
    groups = []
    for state in (initial_state, history_state):
        before = _sha_json(state.record())
        rows = [_execute_clone(graph, request, state, backend, clock,
                               require_chils_receipt=config.require_chils_receipt) for request in requests]
        if before != _sha_json(state.record()) or any(row["state_sha256"] != before for row in rows):
            raise AssertionError("Alternative peer mutated the immutable shared context")
        _loss_weights(rows, graph_weight)
        groups.append(dict(state=state.record(), state_sha256=before, state_loss_weight=.5,
                           graph_loss_weight=graph_weight, alternatives=rows))
    if graph_identity(graph) != input_identity:
        raise ValueError("Backend mutated the source graph during counterfactual label collection")
    result = dict(status="actual_finite_labels_not_learning_or_policy_evidence", graph_sha256=input_identity,
        calibration_sha256=config.calibration_sha256, deadline_seconds=config.deadline_seconds,
        preparation_started_timestamp=started,preparation_deadline_timestamp=deadline,
        history_max_queries=config.history_max_queries, coverage=coverage,
        known_warm_observations=known, history=history, groups=groups,
        cells_expected=11, confirmation_graphs_generated=0,
        timing_scope="conditional executor + full returned-membership validation/rescore; full policy inference evaluated separately",
        no_unexecuted_greedy_teacher_warm=True, executed_common_greedy_prefix=True, fixed_scope_cap=256, peer_outcomes_never_update_peers=True,
        executor_receipt_required=config.require_chils_receipt,
        executor_sha256=CHILS_SHA256 if config.require_chils_receipt else None,
        dry_guard_mocks_only=not config.require_chils_receipt,
        history_actual_calls=sum(row["launched"] for row in history),
        alternative_actual_calls=sum(row["launched"] for group in groups for row in group["alternatives"]),
        physical_hard_deadline_certified=False, learning_advantage_established=False)
    json.dumps(result, allow_nan=False)
    return result


def collect_fresh_spec(spec, *, config, backend, clock=perf_counter):
    """Explicit reusable caller API, preserving a failed declared graph slot.

    Graph construction is a separate dataset/setup cost. Do not invoke before
    the parent freezes protocol/cost/source and authorizes actual collection.
    """
    weight = split_graph_weight(spec)
    if not config.require_chils_receipt:
        raise ValueError("Fresh collection must require actual pinned CHILS receipts; mocks are tiny guards only")
    setup = _clock(clock)
    try:
        graph, selected, metadata = construct_fresh_case(spec)
    except Exception as error:
        return dict(status="fresh_declared_graph_construction_failed", spec=dict(spec),
            graph_loss_weight=weight, groups=[], included_in_coverage=True,
            failure=dict(type=type(error).__name__, message=str(error)),
            confirmation_graphs_generated=0, learning_advantage_established=False)
    setup_seconds = _clock(clock, setup)-setup
    try:
        result = collect_label_groups(graph, selected, seed=spec["seed"]+1000,
            config=config, backend=backend, graph_weight=weight, clock=clock)
    except Exception as error:
        return dict(status="fresh_declared_graph_collection_failed", spec=dict(spec),
            graph_sha256=graph_identity(graph), graph_loss_weight=weight,
            graph_construction_seconds=setup_seconds, source_metadata=metadata,
            groups=[], included_in_coverage=True,
            failure=dict(type=type(error).__name__, message=str(error)),
            incomplete_case_labels_not_certified=True, confirmation_graphs_generated=0,
            learning_advantage_established=False)
    result.update(spec=dict(spec), source_metadata=metadata,
                  graph_construction_seconds=setup_seconds,
                  dataset_construction_outside_decision_timer=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True)
    args = parser.parse_args(); destination = Path(args.plan_json)
    if destination.exists(): raise FileExistsError("Never overwrite an existing split plan")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(specification_manifest(), indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__": main()
