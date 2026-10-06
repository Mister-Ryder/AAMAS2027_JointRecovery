"""Strong paid greedy-menu sweep with partial feasible deadline admission.

This is a classical comparator, not a new learned method. No solver/outcome
cache or label is read. Unlike a batch priority callback it commits each actual
partial mask after validation/rescore, preserving it across a later overrun.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

from .core import _validate_state
from .v4_budgeted_recovery import ScopeCache,_check_recovery,objective


@dataclass(frozen=True)
class GreedyEvent:
    action_index: int
    action_identity: tuple
    requested_cap: int
    representative_cap: int
    actual_scope_size: int
    source: str
    exponent: float | None
    recovered: frozenset[int]
    gain_if_valid: float
    validated_sample_seconds: float
    on_time: bool
    improved_incumbent: bool


@dataclass(frozen=True)
class AnytimeGreedyReceipt:
    selected: frozenset[int]
    on_time_gain: float
    incumbent_value: float
    controller_return_sample_seconds: float
    deadline_seconds: float
    missed_return_sample: bool
    stop_reason: str
    proposals: int
    declared_scopes: int
    visited_unique_scopes: int
    duplicate_scopes_skipped: int
    admitted_improvements: int
    events: tuple[GreedyEvent,...]
    stage_seconds: dict
    schedule: str
    physical_hard_deadline_certified: bool = False
    solver_calls: int = 0
    partial_gain_checks: int = 0
    full_partial_validations: int = 0
    partial_validations_skipped: int = 0


def action_identity(action):
    """Root's wider action versions bind BOTH commitments and releases."""
    return tuple(action.inserts),tuple(getattr(action,"releases",()))


def run_anytime_greedy(graph,selected,proposal_factory,deadline_seconds,
        caps=(64,256,1024),exponents=(0.,.5,1.),resource_cliques=(),
        scope_cache_factory=ScopeCache,known_warm_factory=None,clock=perf_counter):
    """Largest-scope-first action sweep; each accepted partial mask is real.

    Actions retain the shared proposal order. Within each cap, all actions are
    visited; cap order is descending. A scope's known-original warm candidate
    (when supplied) is materialized first, then fresh greedy runs e=0,.5,1.
    GreedyDirect uses exponents=(0,). This explicit fixed schedule is not chosen
    from outcomes. Root may pass NeighborhoodScopeCache without this module
    importing/editing that root-owned action/proposal implementation.

    All validation/proposal/scope/sort/greedy/materialization/rescore costs are
    timed. The known warm is only already observed original/executed membership,
    never an unexecuted greedy outcome. Actual return is externally measured.
    A paid running-value gate avoids full rescore of partials that cannot improve
    the retained result. Native integer gates are exact; floating gates include
    a conservative forward-rounding margin. Every admitted mask still receives
    complete physical validation and objective rescore.
    """
    if not isfinite(deadline_seconds) or deadline_seconds<=0:
        raise ValueError("Positive finite total wall deadline required")
    caps=tuple(caps); exponents=tuple(float(e) for e in exponents)
    if not caps or tuple(sorted(set(caps)))!=caps or any(isinstance(c,bool) or int(c)!=c or c<0 for c in caps):
        raise ValueError("Distinct increasing nonnegative integer caps required")
    if not exponents or any(not isfinite(e) or e<0 for e in exponents) or len(set(exponents))!=len(exponents):
        raise ValueError("Distinct finite nonnegative greedy exponents required")
    started=clock()
    if not isfinite(started) or not isfinite(started+deadline_seconds):
        raise ValueError("Finite initial clock/deadline required")
    last=started; deadline=started+deadline_seconds
    best=frozenset(selected); best_gain=0.; initial_value=0
    stages={}; events=[]; action_count=declared=visited=duplicates=improvements=0
    gain_checks=full_checks=skipped_checks=0
    seen=set()

    def now():
        nonlocal last
        current=clock()
        if not isfinite(current) or current<last: raise ValueError("Finite monotonic clock required")
        last=current; return current

    def finish(reason):
        end=now()
        return AnytimeGreedyReceipt(best,best_gain,initial_value,end-started,deadline_seconds,
            end>deadline,reason,action_count,declared,visited,duplicates,improvements,
            tuple(events),dict(stages),"descending-cap/action-menu-order/known-warm-then-exponent-order",
            False,0,gain_checks,full_checks,skipped_checks)

    def admit(scope,action,requested_cap,representative,raw_recovery,source,exponent):
        nonlocal best,best_gain,improvements
        tick=now()
        # Full physical membership and sorted objective rescore are inside the
        # ready boundary; a fast running sum alone is not an admitted result.
        recovery=_check_recovery(graph,scope,raw_recovery)
        candidate=scope.base|recovery
        gain=max(0.,objective(graph,candidate)-initial_value)
        ready=now(); timely=ready<deadline; changed=timely and gain>best_gain
        if changed:
            best,best_gain=candidate,gain; improvements+=1
        events.append(GreedyEvent(scope.action_index,action_identity(action),requested_cap,
            representative,len(scope.replacements),source,exponent,recovery,gain,
            ready-started,timely,changed))
        stages["materialize_validate_rescore"]=stages.get("materialize_validate_rescore",0.)+ready-tick
        return timely

    tick=now(); incumbent=_validate_state(graph,best); initial_value=objective(graph,incumbent)
    stages["incumbent_validation"]=now()-tick
    if now()>=deadline: return finish("incumbent_validation_deadline")
    tick=now(); actions=tuple(proposal_factory(graph,incumbent))
    stages["proposal"]=now()-tick; action_count=len(actions); declared=action_count*len(caps)
    if len(set(map(action_identity,actions)))!=action_count:
        raise ValueError("Duplicate action commitment/release versions")
    if now()>=deadline: return finish("proposal_deadline")
    if not actions: return finish("no_actions")
    tick=now(); cache=scope_cache_factory(graph,incumbent,resource_cliques)
    stages["scope_cache_setup"]=now()-tick
    if now()>=deadline: return finish("scope_cache_setup_deadline")
    known_warm=(known_warm_factory if known_warm_factory is not None else
                getattr(cache,"initial_known_warm",lambda scope:frozenset()))
    for cap in reversed(caps):
        for index,action in enumerate(actions):
            if now()>=deadline: return finish("before_scope_deadline")
            tick=now(); scope=cache.scope(index,action,cap)
            stages["scope_preparation"]=stages.get("scope_preparation",0.)+now()-tick
            if now()>=deadline: return finish("scope_preparation_deadline")
            identity=(index,scope.replacements)
            if identity in seen:
                duplicates+=1; continue
            seen.add(identity); visited+=1
            representative=min(c for c in caps if c>=len(scope.replacements))
            # Known warm construction itself is charged, not just its mask.
            tick=now(); warm=known_warm(scope)
            stages["known_warm_preparation"]=stages.get("known_warm_preparation",0.)+now()-tick
            if now()>=deadline: return finish("known_warm_preparation_deadline")
            if not admit(scope,action,cap,representative,warm,"known_warm",None):
                return finish("known_warm_validation_deadline")
            pool=frozenset(scope.replacements)
            tick=now(); degrees={v:len(graph.adjacency[v]&pool) for v in pool}
            stages["degree_preparation"]=stages.get("degree_preparation",0.)+now()-tick
            if now()>=deadline: return finish("degree_preparation_deadline")
            for exponent in exponents:
                tick=now()
                order=sorted(pool,key=lambda v:(
                    -graph.weights[v].item() if exponent==0 else -float(graph.weights[v])/(1+degrees[v])**exponent,
                    -graph.weights[v].item(),v))
                stages["greedy_ordering"]=stages.get("greedy_ordering",0.)+now()-tick
                if now()>=deadline: return finish("greedy_ordering_deadline")
                recovery=set(); running_value=0
                for vertex in order:
                    if now()>=deadline: return finish("before_vertex_deadline")
                    tick=now(); compatible=not(graph.adjacency[vertex]&recovery)
                    if compatible:
                        recovery.add(vertex)
                        running_value+=graph.weights[vertex].item()
                    stages["greedy_construction"]=stages.get("greedy_construction",0.)+now()-tick
                    if now()>=deadline: return finish("greedy_construction_deadline")
                    if compatible:
                        gain_checks+=1
                        estimated=scope.immediate_gain+running_value
                        if graph.weights.dtype.kind in "iu":
                            potentially_better=estimated>best_gain
                        else:
                            # Positive weights bound the possible base sum by
                            # 2*original+abs(q); the extra n-dependent term also
                            # covers the sequential running-value additions.
                            margin=(64+4*len(recovery))*2.**-52*(
                                2*abs(initial_value)+abs(scope.immediate_gain)+abs(running_value))
                            potentially_better=estimated+margin>best_gain
                        if not potentially_better:
                            skipped_checks+=1; continue
                        full_checks+=1
                        if not admit(scope,action,cap,representative,recovery,"greedy_partial",exponent):
                            return finish("partial_validation_deadline")
                # The final partial already IS the complete returned recovery;
                # do not charge/re-admit the same mask only to label completion.
    return finish("sweep_complete")
