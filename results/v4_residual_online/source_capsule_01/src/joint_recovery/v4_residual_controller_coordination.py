"""Versioned coordination hooks over the byte-pinned Stage-2 controller.

Reserved R256 revision: action identity, scope construction and paid executed common warm handling
are derived. Request selection, current-view prediction, admission, validation,
failure/spent semantics and classical priorities retain the original function.
No shared module/global is patched. The imported derivation is resident setup;
the experiment must declare its cold/resident lifecycle and measure the actual
external return time. This is not a physical hard-deadline guarantee.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

from . import v4_budgeted_recovery as core
from experiments.v4_residual_common import ExecutedWarmScopeCache


FROZEN_CONTROLLER_SHA256 = "b4b5bbd3d8abdeba2fcb9a117c54f161786c2deccc032764e792d5899bd3d1ad"


def action_identity(action):
    """Canonical C/E identity; true vertex semantics are checked by the cache."""
    identity=[]
    for field in ("inserts","releases"):
        values=tuple(getattr(action,field))
        if any(not isinstance(v,Integral) or isinstance(v,bool) for v in values):
            raise ValueError("Coordination IDs must be exact integers")
        values=tuple(sorted(map(int,values)))
        if len(values)!=len(set(values)):
            raise ValueError("Coordination IDs must be distinct")
        identity.append(values)
    inserts,releases=identity
    if len(inserts)>8 or not(inserts or releases):
        raise ValueError("A bounded nonempty commitment/release version is required")
    return inserts,releases


@dataclass(frozen=True)
class CoordinationScope(core.RecoveryScope):
    releases: tuple = ()

    @property
    def action_identity(self): return self.inserts,self.releases


@dataclass(frozen=True)
class KnownWarmObservation:
    action_identity: tuple
    action_index: int
    cap: int
    recovered: frozenset
    gain_if_valid: float
    validated_sample_seconds: float
    admitted_on_time: bool


@dataclass(frozen=True)
class CoordinationDecisionReceipt(core.DecisionReceipt):
    action_identities: tuple = ()
    known_warm_observations: tuple = ()


class _CoordinationCache:
    """One decision; explicit versions never alias through equal C alone."""
    def __init__(self,graph,selected,resource_cliques,factory,known_warm_factory):
        if factory is None:
            raise ValueError("An explicit coordinated scope cache factory is required")
        self.graph,self.selected=graph,frozenset(selected)
        self.raw=factory(graph,self.selected,resource_cliques)
        if not isinstance(self.raw, ExecutedWarmScopeCache):
            raise ValueError('Reserved cache must execute the common greedy prefix')
        self.known_warm_factory=known_warm_factory
        self.identities={}; self.records={}; self.prefixes={}

    def scope(self,index,action,cap):
        identity=action_identity(action)
        if index in self.identities and self.identities[index]!=identity:
            raise ValueError("A coordinated action index cannot change C/E")
        self.identities[index]=identity
        key=(index,identity,cap)
        if key not in self.records:
            raw=self.raw.scope(index,action,cap)
            if not isinstance(raw,core.RecoveryScope):
                raise TypeError("A coordinated cache must return RecoveryScope")
            C,E=map(frozenset,identity)
            mandatory=frozenset(v for c in C for v in self.graph.adjacency[c]&self.selected)
            D=mandatory|E; B=(self.selected-D)|C
            if (raw.action_index!=index or raw.cap!=cap or raw.inserts!=identity[0]
                    or raw.displaced!=D or raw.base!=B or not E<=self.selected
                    or C&self.selected or raw.immediate_gain!=
                    core.objective(self.graph,C)-core.objective(self.graph,D)):
                raise ValueError("Cache output does not bind the declared fixed C/E/base")
            ids=tuple(raw.replacements)
            if any(not isinstance(v,Integral) or isinstance(v,bool) or v<0 or v>=self.graph.n for v in ids):
                raise ValueError("Replacement IDs must be valid exact integers")
            if len(ids)!=len(set(ids)) or len(ids)>cap or raw.eligible_before_cap<len(ids):
                raise ValueError("Replacement pool must be a bounded unique prefix")
            if any(v in B or self.graph.adjacency[v]&B for v in ids):
                raise ValueError("Every replacement must be outside and compatible with the fixed base")
            for other_cap,other in self.prefixes.get(index,{}).items():
                small,large=(ids,other) if cap<=other_cap else (other,ids)
                if small!=large[:len(small)]:
                    raise ValueError("Caps must be prefixes of one fixed ordered pool")
            self.prefixes.setdefault(index,{})[cap]=ids
            self.records[key]=CoordinationScope(**vars(raw),releases=identity[1])
        return self.records[key]

    def initial_known_warm(self,scope):
        if self.known_warm_factory is not None or not isinstance(self.raw,ExecutedWarmScopeCache):
            raise ValueError('Reserved production warm must be the explicit executed common prefix')
        recovery=core._check_recovery(self.graph,scope,self.raw.initial_known_warm(scope))
        return recovery



_KNOWN_WARM_BODY = """
known_started = now()
recovery = cache.initial_known_warm(scope)
schedule = scope.base | recovery
candidate_value = objective(graph, schedule)
recovery_value = objective(graph, recovery)
previous_warm_value = objective(graph, warm[index])
gain = max(0., candidate_value-initial_value)
known_ready = now()
stages['known_warm_validation'] = stages.get('known_warm_validation',0.)+known_ready-known_started
known_warm_observations.append(KnownWarmObservation(
    scope.action_identity, index, cap, recovery, gain, known_ready-started, known_ready<deadline))
if known_ready >= deadline:
    stages['scope_preparation'] = now()-t
    return finish('known_warm_deadline')
if recovery_value > previous_warm_value:
    warm[index] = recovery
if gain > best_gain:
    best, best_gain = schedule, gain
"""


def _dump(node): return ast.dump(node,include_attributes=False)


def _derive_controller():
    actual=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    if actual!=FROZEN_CONTROLLER_SHA256:
        raise RuntimeError("The Stage-2 controller bytes differ from the reviewed source")
    tree=ast.parse(inspect.getsource(core.run_budgeted_recovery))
    function=tree.body[0]; original_ast=_dump(function)
    function.name="run_coordinated_recovery"
    function.args.kwonlyargs.extend([ast.arg(arg="scope_cache_factory"),ast.arg(arg="known_warm_factory")])
    function.args.kw_defaults.extend([ast.Constant(value=None),ast.Constant(value=None)])
    replacements={
        _dump(ast.parse("stages, attempts, spent, warm = {}, [], set(), {}").body[0]):
            ast.parse("stages, attempts, spent, warm = {}, [], set(), {}\ncoordination_actions=()\nknown_warm_observations=[]").body,
        _dump(ast.parse("actions = tuple(proposal_factory(graph, incumbent))").body[0]):
            ast.parse("actions = tuple(proposal_factory(graph, incumbent))\ncoordination_actions=tuple(action_identity(a) for a in actions)").body,
        _dump(ast.parse("cache = ScopeCache(graph, incumbent, resource_cliques)").body[0]):
            ast.parse("cache = _CoordinationCache(graph,incumbent,resource_cliques,scope_cache_factory,known_warm_factory)").body,
        _dump(ast.parse("requests.extend(Request(scope, w) for w in workpoints)").body[0]):
            ast.parse(_KNOWN_WARM_BODY+"\nrequests.extend(Request(scope,w) for w in workpoints)").body
    }
    hits={key:0 for key in replacements}; identity_hits=receipt_hits=0
    class Transform(ast.NodeTransformer):
        def visit_Assign(self,node):
            key=_dump(node)
            if key in replacements: hits[key]+=1; return replacements[key]
            return self.generic_visit(node)
        def visit_Expr(self,node):
            key=_dump(node)
            if key in replacements: hits[key]+=1; return replacements[key]
            return self.generic_visit(node)
        def visit_If(self,node):
            nonlocal identity_hits
            expected=ast.parse("if len({tuple(a.inserts) for a in actions}) != len(actions):\n raise ValueError('unused')").body[0]
            if _dump(node.test)==_dump(expected.test):
                node.test=ast.parse("len(set(coordination_actions)) != len(actions)",mode="eval").body
                node.body=ast.parse("raise ValueError('Proposal menu contains duplicate C/E action versions')").body
                identity_hits+=1
            return self.generic_visit(node)
        def visit_Call(self,node):
            nonlocal receipt_hits
            if isinstance(node.func,ast.Name) and node.func.id=="DecisionReceipt":
                node.func.id="CoordinationDecisionReceipt"
                node.args.extend([ast.Name(id="coordination_actions",ctx=ast.Load()),
                    ast.Call(func=ast.Name(id="tuple",ctx=ast.Load()),
                        args=[ast.Name(id="known_warm_observations",ctx=ast.Load())],keywords=[])])
                receipt_hits+=1
            return self.generic_visit(node)
    tree=Transform().visit(tree); ast.fix_missing_locations(tree)
    if any(count!=1 for count in hits.values()) or identity_hits!=1 or receipt_hits!=1:
        raise RuntimeError("Coordination derivation did not match its finite reviewed hooks")
    namespace=dict(vars(core))
    namespace.update(action_identity=action_identity,_CoordinationCache=_CoordinationCache,
        KnownWarmObservation=KnownWarmObservation,CoordinationDecisionReceipt=CoordinationDecisionReceipt,
        __name__=__name__)
    exec(compile(tree,__file__+":derived", "exec"),namespace)
    receipt=dict(original_controller_sha256=actual,
        original_function_ast_sha256=hashlib.sha256(original_ast.encode()).hexdigest(),
        derived_function_ast_sha256=hashlib.sha256(_dump(tree.body[0]).encode()).hexdigest(),
        matched_statement_hooks=sum(hits.values()),matched_identity_hooks=identity_hits,
        matched_receipt_hooks=receipt_hits,shared_module_globals_modified=False)
    return namespace["run_coordinated_recovery"],receipt


_run_coordinated_recovery,DERIVATION_RECEIPT=_derive_controller()

def run_coordinated_recovery(*args, caps=(256,), known_warm_factory=None,
        scope_cache_factory=ExecutedWarmScopeCache, **kwargs):
    """Production entry enforces fixed scope/warm even for an empty action menu."""
    if tuple(caps) != (256,) or known_warm_factory is not None or not callable(scope_cache_factory):
        raise ValueError('Reserved entry requires fixedR256 and the explicit executed common prefix')
    return _run_coordinated_recovery(*args, caps=(256,), known_warm_factory=None,
        scope_cache_factory=scope_cache_factory, **kwargs)
