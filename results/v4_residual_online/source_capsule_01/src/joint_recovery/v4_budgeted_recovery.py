"""Finite joint-recovery requests with auditable soft-deadline admission.

New V4 prototype only. No V3 source, checkpoint, solver or outcome is changed.
Predictors receive observable scopes and already executed warm starts, never
an offline teacher. Every call, including failures and late calls, is spent.
Only a real recovery received, validated and rescored before the absolute
deadline can improve the on-time schedule. This is NOT a physical hard-time
guarantee: actual return latency and misses must also be measured externally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum, isfinite
from numbers import Integral
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .core import Action, Graph, _validate_action, _validate_state, is_feasible
from .v4_factors import observable_factors


@dataclass(frozen=True)
class NativeIntegerGraph:
    """V4-only graph view preserving native integer rewards, unlike core.Graph.

    Compatible with core feasibility/action utilities without changing their
    frozen source. Actual objective uses Python integers, not an int64 sum.
    Prediction/solver float conversions are separately declared boundaries.
    """
    weights: np.ndarray
    agents: np.ndarray
    adjacency: tuple[frozenset[int],...]
    name: str = "native-integer-graph"

    def __post_init__(self):
        weights=np.asarray(self.weights)
        if weights.ndim!=1 or weights.dtype.kind not in "iu" or np.any(weights<=0):
            raise ValueError("Native positive integer weight array required")
        if any(any(not isinstance(v,Integral) or isinstance(v,bool) for v in row) for row in self.adjacency):
            raise ValueError("Native adjacency requires exact integer IDs")
        checked=Graph(np.ones(len(weights)),self.agents,self.adjacency,self.name)
        weights=np.array(weights,copy=True); weights.setflags(write=False)
        object.__setattr__(self,"weights",weights)
        object.__setattr__(self,"agents",checked.agents)
        object.__setattr__(self,"adjacency",checked.adjacency)
        object.__setattr__(self,"_max_weight",float(weights.max()) if len(weights) else 1.)

    @property
    def n(self): return len(self.weights)


def objective(graph: Graph, selected: Iterable[int]) -> float:
    if getattr(graph.weights,"dtype",None) is not None and graph.weights.dtype.kind in "iu":
        # Native public integer objectives remain exact even above 2**53.
        return sum(int(graph.weights[v]) for v in sorted(selected))
    return float(fsum(float(graph.weights[v]) for v in sorted(selected)))


@dataclass(frozen=True)
class Workpoint:
    """Native repair setting and development-calibrated full request cost.

    ``amount`` is in the units of ``kind``; it is never silently converted.
    ``expected_seconds`` is a shared calibration, not a guaranteed runtime.
    Backend I/O and validation overhead belong in that calibration and timer.
    """
    name: str
    kind: str
    amount: float
    expected_seconds: float

    def __post_init__(self):
        if not self.name or self.kind not in ("seconds", "nodes", "iterations"):
            raise ValueError("A named seconds/nodes/iterations workpoint is required")
        if not isfinite(self.amount) or self.amount < 0:
            raise ValueError("Native repair amount must be finite and nonnegative")
        if self.kind == "seconds" and self.amount <= 0:
            raise ValueError("A seconds workpoint must be strictly positive")
        if self.kind != "seconds" and self.amount != int(self.amount):
            raise ValueError("Nodes/iterations require exact integer amounts")
        if not isfinite(self.expected_seconds) or self.expected_seconds <= 0:
            raise ValueError("A positive calibrated request cost is required")


@dataclass(frozen=True)
class RecoveryScope:
    action_index: int
    inserts: tuple[int, ...]
    cap: int
    base: frozenset[int]
    displaced: frozenset[int]
    replacements: tuple[int, ...]
    eligible_before_cap: int
    immediate_gain: float
    resource_cliques: tuple[tuple[int, ...], ...] = ()

    def canonical_edges(self, graph: Graph) -> tuple[tuple[int, int], ...]:
        """Exact local orientation/order, independent of set serialization."""
        local = {v: i for i, v in enumerate(self.replacements)}
        return tuple(sorted((i, local[u]) for i, v in enumerate(self.replacements)
                            for u in graph.adjacency[v] if u in local and i < local[u]))


class ScopeCache:
    """One fixed graph/incumbent; cap changes never change the action/base."""
    def __init__(self, graph: Graph, selected: frozenset[int], resource_cliques=()):
        self.graph, self.selected = graph, frozenset(selected)
        self.resource_cliques = tuple(tuple(q) for q in resource_cliques)
        if any(any(not isinstance(v,Integral) or isinstance(v,bool) for v in q) for q in self.resource_cliques):
            raise ValueError("Resource memberships require exact integer vertex IDs")
        self._actions, self._scopes = {}, {}

    def scope(self, index: int, action: Action, cap: int) -> RecoveryScope:
        if int(cap) != cap or cap < 0:
            raise ValueError("Replacement cap must be an exact nonnegative integer")
        cap = int(cap)
        if index not in self._actions:
            _validate_action(self.graph, self.selected, action)
            removed = frozenset(v for u in action.inserts for v in self.graph.adjacency[u]
                                if v in self.selected)
            base = (self.selected-removed) | frozenset(action.inserts)
            candidates = set(removed)
            for v in removed:
                candidates.update(self.graph.adjacency[v])
            eligible = tuple(sorted((v for v in candidates if v not in base and
                                     not (self.graph.adjacency[v] & base)),
                                    key=lambda v: (-self.graph.weights[v].item(), v)))
            q = objective(self.graph, action.inserts)-objective(self.graph, removed)
            self._actions[index] = (tuple(action.inserts), base, removed, eligible, q)
        inserts, base, removed, eligible, q = self._actions[index]
        if inserts != tuple(action.inserts):
            raise ValueError("A scope-cache action index cannot change its commitment")
        key = (index, cap)
        if key not in self._scopes:
            replacements = eligible[:cap]
            pool = frozenset(replacements)
            cliques = set()
            for raw in self.resource_cliques:
                if len(raw) != len(set(raw)) or any(v < 0 or v >= self.graph.n for v in raw):
                    raise ValueError("Invalid supplied resource clique")
                members = tuple(sorted(pool.intersection(raw)))
                if len(members) < 2:
                    continue
                if any(v not in self.graph.adjacency[u] for i, u in enumerate(members)
                       for v in members[i+1:]):
                    raise ValueError("Supplied resource factor is not a conflict clique")
                cliques.add(members)
            self._scopes[key] = RecoveryScope(index, inserts, cap, base, removed,
                replacements, len(eligible), q, tuple(sorted(cliques)))
        return self._scopes[key]


@dataclass(frozen=True)
class Request:
    scope: RecoveryScope
    workpoint: Workpoint

    @property
    def key(self):
        return (self.scope.action_index, self.scope.cap, self.workpoint.name)


@dataclass(frozen=True)
class RequestView:
    request: Request
    warm_start: frozenset[int]


@dataclass(frozen=True)
class ControllerState:
    incumbent_value: float
    best_gain: float
    remaining_seconds: float
    spent_requests: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class FeasibleCandidate:
    scope: RecoveryScope
    recovered: frozenset[int]


@dataclass(frozen=True)
class PriorityEvaluation:
    """Point-predicted/proxy accepted gains and actual cheap feasible outputs.

    Cheap-LB policies must expose the recovery sets they actually computed;
    those are admitted on the same validation/clock boundary, not discarded to
    weaken a baseline. Values are relative to the ORIGINAL incumbent. The
    controller subtracts its current best gain to form a positive point-marginal
    approximation; this is not E[max(realized_gain-best, 0)]. A value predictor
    conditioned on the current warm start cannot simultaneously supply a new
    cheap candidate: that would invalidate the execution input it predicted.
    Classical proxies may expose cheap masks and declare the conditioning flag
    false. ``selection_order`` is a complete index permutation for an explicit
    scheduling policy, such as round robin; otherwise values/cost choose.
    """
    gains: Sequence[float]
    feasible_candidates: Sequence[FeasibleCandidate] = ()
    semantics: str = "point prediction of finite executable gain"
    current_warm_start_conditioned: bool = True
    selection_order: Sequence[int] | None = None


@dataclass(frozen=True)
class RepairAttempt:
    recovered: frozenset[int] | None
    status: str = "returned"
    diagnostics: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class AttemptReceipt:
    key: tuple[int, int, str]
    native_budget_kind: str
    native_budget_amount: float
    expected_seconds: float
    remaining_before_call: float
    elapsed_seconds: float
    status: str
    valid: bool
    admitted_on_time: bool
    gain_if_valid: float | None
    warm_start: frozenset[int]
    recovered: frozenset[int] | None
    diagnostics: Mapping


@dataclass(frozen=True)
class DecisionReceipt:
    selected: frozenset[int]
    on_time_gain: float
    incumbent_value: float
    controller_return_sample_seconds: float
    deadline_seconds: float
    missed_return_sample: bool
    stop_reason: str
    proposals: int
    prepared_requests: int
    attempts: tuple[AttemptReceipt, ...]
    spent_requests: tuple[tuple[int, int, str], ...]
    cheap_candidates_observed: int
    cheap_candidates_admitted: int
    stage_seconds: Mapping[str, float]
    physical_hard_deadline_certified: bool = False
    declared_requests: int = 0
    duplicate_scope_requests_skipped: int = 0


def _check_recovery(graph: Graph, scope: RecoveryScope, recovered) -> frozenset[int]:
    values = tuple(recovered)
    if any(not isinstance(v, Integral) or isinstance(v, bool) for v in values):
        raise ValueError("Recovery memberships require exact integer vertex IDs")
    if len(values) != len(set(values)):
        raise ValueError("Recovery membership contains duplicate vertices")
    selected = frozenset(int(v) for v in values)
    if not selected <= frozenset(scope.replacements):
        raise ValueError("Recovered vertices lie outside the fixed scope")
    if not is_feasible(graph, selected) or any(graph.adjacency[v] & scope.base for v in selected):
        raise ValueError("Recovered set is incompatible")
    if not is_feasible(graph, scope.base | selected):
        raise ValueError("Materialized recovery schedule is infeasible")
    return selected


def _safe_diagnostics(value) -> dict:
    """Malformed optional metadata never discards a spent execution receipt."""
    if not isinstance(value, Mapping):
        return {"invalid_diagnostics_type": type(value).__name__}
    try:
        return dict(value)
    except Exception as error:
        return {"invalid_diagnostics_type": type(value).__name__,
                "diagnostics_exception_type": type(error).__name__}


def run_budgeted_recovery(graph: Graph, selected: Iterable[int],
        proposal_factory: Callable, backend: Callable, priority: Callable,
        deadline_seconds: float, caps=(64, 256, 1024), workpoints=(),
        max_queries=8, resource_cliques=(), clock=perf_counter,
        deduplicate_scopes=True) -> DecisionReceipt:
    """Execute a finite action/range/budget allocation with on-time admission.

    proposal_factory(graph, original_incumbent) -> actions
    priority(graph, request_views, ControllerState) -> PriorityEvaluation
    backend(graph, scope, Workpoint, actual_warm_start, remaining_seconds)
        -> RepairAttempt

    All callbacks execute INSIDE this decision timer. Backend setup/cold-versus-
    resident lifecycle is separately declared and matched by the experiment.
    A backend must honor the supplied remaining wall time where possible, but
    this caller assumes soft time limits and discards late improvements.
    """
    if not isfinite(deadline_seconds) or deadline_seconds <= 0:
        raise ValueError("Decision deadline must be positive seconds")
    if int(max_queries) != max_queries or max_queries < 0:
        raise ValueError("Finite nonnegative query cap required")
    caps = tuple(caps); workpoints = tuple(workpoints)
    if not caps or tuple(sorted(set(caps))) != caps or any(int(c) != c or c < 0 for c in caps):
        raise ValueError("Caps must be distinct increasing nonnegative integers")
    if not workpoints or len({w.name for w in workpoints}) != len(workpoints):
        raise ValueError("Distinct workpoint names are required")
    started = clock()
    if not isfinite(started):
        raise ValueError("A finite initial decision timestamp is required")
    last_clock = started; deadline = started+deadline_seconds
    if not isfinite(deadline):
        raise ValueError("The absolute decision deadline must be finite")
    stages, attempts, spent, warm = {}, [], set(), {}
    best = frozenset(selected); best_gain = 0.; initial_value = 0.
    proposal_count = 0; requests = []; cheap_seen = cheap_admitted = 0
    declared_requests=duplicate_skipped=0
    def now():
        nonlocal last_clock
        current = clock()
        if not isfinite(current) or current < last_clock:
            raise ValueError("A finite monotonic decision clock is required")
        last_clock = current
        return current
    def finish(reason):
        current = now()
        return DecisionReceipt(best, best_gain, initial_value, current-started,
            deadline_seconds, current > deadline, reason, proposal_count,
            len(requests), tuple(attempts), tuple(sorted(spent)), cheap_seen,
            cheap_admitted, dict(stages), False,declared_requests,duplicate_skipped)
    # Incumbent checks and rescore are charged. If these exceed the deadline,
    # the original incumbent remains available; no new objective is admitted.
    t = now(); incumbent = _validate_state(graph, best)
    initial_value = objective(graph, incumbent)
    stages["incumbent_validation"] = now()-t
    if now() >= deadline:
        return finish("incumbent_validation_deadline")
    t = now(); actions = tuple(proposal_factory(graph, incumbent))
    proposal_count = len(actions); stages["proposal"] = now()-t
    declared_requests=proposal_count*len(caps)*len(workpoints)
    if len({tuple(a.inserts) for a in actions}) != len(actions):
        raise ValueError("Proposal menu contains duplicate commitments")
    if now() >= deadline:
        return finish("proposal_deadline")
    if not actions:
        return finish("no_actions")
    t = now(); cache = ScopeCache(graph, incumbent, resource_cliques)
    for index, action in enumerate(actions):
        warm[index] = frozenset()
        seen_pools=set()
        for cap in caps:
            scope = cache.scope(index, action, cap)
            if now() >= deadline:
                stages["scope_preparation"] = now()-t
                return finish("scope_deadline")
            if deduplicate_scopes and scope.replacements in seen_pools:
                duplicate_skipped+=len(workpoints); continue
            seen_pools.add(scope.replacements)
            requests.extend(Request(scope, w) for w in workpoints)
    stages["scope_preparation"] = now()-t
    while len(attempts) < max_queries:
        remaining = deadline-now()
        if remaining <= 0:
            return finish("decision_deadline")
        pending = [r for r in requests if r.key not in spent and
                   r.workpoint.expected_seconds <= remaining and
                   (r.workpoint.kind != "seconds" or r.workpoint.amount <= remaining)]
        if not pending:
            return finish("no_affordable_unspent_request")
        views = tuple(RequestView(r, warm[r.scope.action_index] & frozenset(r.scope.replacements))
                      for r in pending)
        state = ControllerState(initial_value, best_gain, remaining, tuple(sorted(spent)))
        t = now(); evaluated = priority(graph, views, state)
        stages["representation_inference_priority"] = stages.get("representation_inference_priority",0.)+now()-t
        if len(evaluated.gains) != len(pending) or any(not isfinite(float(v)) for v in evaluated.gains):
            raise ValueError("Priority must return one finite gain per observable request")
        if evaluated.current_warm_start_conditioned and evaluated.feasible_candidates:
            raise ValueError("Conditioned predictions cannot alter their own execution warm start")
        if evaluated.selection_order is not None:
            order = tuple(evaluated.selection_order)
            if any(not isinstance(i, Integral) or isinstance(i, bool) for i in order) or (
                    sorted(order) != list(range(len(pending)))):
                raise ValueError("Explicit selection order must be a complete request permutation")
        if now() >= deadline:
            return finish("priority_deadline")
        # A strong cheap policy is allowed to keep all actual feasible repairs
        # it computed. Their validation and objective rescore are charged here.
        t = now()
        valid_scopes = {id(r.scope) for r in pending}
        for candidate in evaluated.feasible_candidates:
            cheap_seen += 1
            if id(candidate.scope) not in valid_scopes:
                raise ValueError("Cheap candidate does not belong to current scopes")
            recovery = _check_recovery(graph, candidate.scope, candidate.recovered)
            schedule = candidate.scope.base | recovery
            candidate_value = objective(graph, schedule)
            recovery_value = objective(graph, recovery)
            index = candidate.scope.action_index
            previous_warm_value = objective(graph, warm[index])
            if now() >= deadline:
                stages["cheap_validation"] = stages.get("cheap_validation",0.)+now()-t
                return finish("cheap_validation_deadline")
            if recovery_value > previous_warm_value:
                warm[index] = recovery
            gain = max(0., candidate_value-initial_value)
            if gain > best_gain:
                best, best_gain = schedule, gain; cheap_admitted += 1
        stages["cheap_validation"] = stages.get("cheap_validation",0.)+now()-t
        # Primary order is predicted positive marginal return / common expected
        # cost. An absolute-value tie-break retains informative LB ordering even
        # when a cheap policy has already materialized its own lower bounds.
        def ordering(i):
            predicted = float(evaluated.gains[i]); cost = pending[i].workpoint.expected_seconds
            return (max(0.,predicted-best_gain)/cost, predicted/cost, -i)
        chosen = (int(order[0]) if evaluated.selection_order is not None else
                  max(range(len(pending)), key=ordering))
        request = pending[chosen]; scope = request.scope
        remaining = deadline-now()
        if remaining <= 0 or request.workpoint.expected_seconds > remaining or (
                request.workpoint.kind == "seconds" and request.workpoint.amount > remaining):
            return finish("selection_deadline")
        # Spent BEFORE launch: a timeout, invalid output or late result is not
        # retried for free and cannot be mistaken for an unevaluated request.
        warmstart = warm[scope.action_index] & frozenset(scope.replacements)
        call_started = now(); remaining = deadline-call_started
        if remaining <= 0 or request.workpoint.expected_seconds > remaining or (
                request.workpoint.kind == "seconds" and request.workpoint.amount > remaining):
            return finish("before_launch_deadline")
        spent.add(request.key)
        result = None; canonical_recovery = None
        valid = admitted = False; gain = None
        try:
            result = backend(graph, scope, request.workpoint, warmstart, remaining)
            if not isinstance(result, RepairAttempt):
                raise TypeError("Backend must return a RepairAttempt")
            status = str(result.status)
        except Exception as error:
            status = "backend_exception:"+type(error).__name__
            result = RepairAttempt(None, status, {"exception_type": type(error).__name__})
        if result.recovered is not None:
            try:
                recovery = _check_recovery(graph, scope, result.recovered)
                canonical_recovery = recovery
                schedule = scope.base | recovery
                candidate_value = objective(graph, schedule)
                recovery_value = objective(graph, recovery)
                previous_warm_value = objective(graph, warm[scope.action_index])
                valid = True; gain = max(0.,candidate_value-initial_value)
                # Completion includes full real membership validation/rescore.
                admitted = now() < deadline
                if admitted:
                    if recovery_value > previous_warm_value:
                        warm[scope.action_index] = recovery
                    if gain > best_gain:
                        best, best_gain = schedule, gain
                else:
                    status = "validated_late:"+status
            except (ValueError, IndexError, TypeError) as error:
                status = "invalid_output:"+type(error).__name__
                canonical_recovery = None
        call_ended = now()
        attempts.append(AttemptReceipt(request.key, request.workpoint.kind,
            request.workpoint.amount, request.workpoint.expected_seconds, remaining,
            call_ended-call_started, status, valid, admitted, gain, warmstart,
            canonical_recovery, _safe_diagnostics(result.diagnostics)))
        stages["repair_validation"] = stages.get("repair_validation",0.)+call_ended-call_started
        if call_ended >= deadline:
            return finish("repair_or_validation_deadline")
    return finish("query_cap")


def classical_priority(mode="upper",inferred_factors=True):
    """Cost-aware strong nonlearned values, through the same timed interface.

    By default ``upper``/``p1`` use the SAME inferred+supplied clique cover as
    the sparse model; inference/cache lookup is charged inside this callback.
    ``inferred_factors=False`` is an explicitly weaker supplied/edge diagnostic.
    ``upper`` uses a disjoint clique partition; uncovered nodes stay singleton.
    ``lower`` computes all three degree-penalty greedy recoveries and exposes
    their actual feasible masks to the controller. This computation is charged.
    """
    if mode not in ("upper", "p1", "lower", "immediate", "round_robin"):
        raise ValueError("Unknown classical priority")
    computed={}; offered=set(); cached_graph=None
    def evaluate(graph, views, state):
        nonlocal cached_graph
        if cached_graph is not graph or not state.spent_requests:
            computed.clear(); offered.clear(); cached_graph=graph
        candidates, values = [], []
        for view in views:
            scope = view.request.scope
            identity = (scope.action_index,scope.inserts,scope.base,scope.displaced,
                        scope.replacements,scope.resource_cliques,scope.immediate_gain)
            if identity not in computed:
                nodes = frozenset(scope.replacements)
                factors=scope.resource_cliques
                if mode in ("upper","p1") and inferred_factors:
                    _,local_factors=observable_factors(graph,scope)
                    factors=tuple(tuple(scope.replacements[v] for v in q) for q in local_factors)
                best=None
                if mode == "lower":
                    recoveries = []
                    for exponent in (0., .5, 1.):
                        order = sorted(nodes, key=lambda v: (-float(graph.weights[v])/
                            (1+len(graph.adjacency[v]&nodes))**exponent, -float(graph.weights[v]), v))
                        recovery = set()
                        for v in order:
                            if not (graph.adjacency[v]&recovery): recovery.add(v)
                        recoveries.append(frozenset(recovery))
                    best = max(recoveries, key=lambda s:(objective(graph,s),tuple(sorted(s))))
                    value = max(0.,scope.immediate_gain+objective(graph,best))
                elif mode == "upper":
                    left = set(nodes); bound = 0
                    for clique in sorted(factors, key=lambda q:(-len(q),q)):
                        block = left.intersection(clique)
                        if block:
                            best_vertex=max(block,key=lambda v:graph.weights[v])
                            bound+=objective(graph,(best_vertex,)); left.difference_update(block)
                    bound += objective(graph,left)
                    value = max(0.,scope.immediate_gain+bound)
                elif mode == "p1":
                    maximum_load = {v:1 for v in nodes}; covered = set()
                    for clique in factors:
                        for v in clique: maximum_load[v] = max(maximum_load[v],len(clique))
                        covered.update((min(u,v),max(u,v)) for i,u in enumerate(clique) for v in clique[i+1:])
                    for u in nodes:
                        for v in graph.adjacency[u]&nodes:
                            if u < v and (u,v) not in covered:
                                maximum_load[u] = max(maximum_load[u],2); maximum_load[v] = max(maximum_load[v],2)
                    value = max(0.,scope.immediate_gain+fsum(float(graph.weights[v])/maximum_load[v] for v in sorted(nodes)))
                elif mode == "immediate": value = max(0.,scope.immediate_gain)
                else: value = 0.
                # Keep actual scope object alive; no id-only cache aliases.
                computed[identity] = (value,best,scope)
            value,best,_=computed[identity]
            if best is not None and identity not in offered:
                candidates.append(FeasibleCandidate(scope,best)); offered.add(identity)
            values.append(value)
        order = None
        if mode == "round_robin":
            # Rotate actions before the next scope/workpoint for that action;
            # failed/late requests count as turns just like successful calls.
            turns = {}
            for action, cap, name in state.spent_requests:
                turns[action] = turns.get(action, 0)+1
            order = tuple(sorted(range(len(views)), key=lambda i:(
                turns.get(views[i].request.scope.action_index, 0),
                views[i].request.scope.action_index, views[i].request.scope.cap, i)))
        return PriorityEvaluation(tuple(values),tuple(candidates),
            "classical "+mode+" proxy, common calibrated cost", False, order)
    return evaluate
