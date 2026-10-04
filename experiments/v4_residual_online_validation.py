"""Reserved fixedR256 residual: sealed24 validation-only online policies.

Five architectures/all three fits and six strong classical controls share the
same paid action domain, native CHILS repair and total wall deadlines. Source
certified availability and strictly validated interface return are different
metrics. Late solver/final outputs never become an on-time interface return.
"""
from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import dataclass,is_dataclass
from datetime import datetime,timezone
from hashlib import sha256
import inspect
import json
from math import isfinite
from numbers import Integral
from pathlib import Path
import platform
import sys
import time
from types import FunctionType

import numpy as np
import torch

_REPOSITORY_ROOT=Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:sys.path.insert(0,str(_REPOSITORY_ROOT))

from joint_recovery.core import _validate_state,is_feasible
from joint_recovery import v4_budgeted_recovery as core
from joint_recovery import v4_residual_controller_coordination as coordinated
from joint_recovery import v4_residual_model as learned
from joint_recovery.v4_model_fast import classical_priority_fast
try:
    from . import v4_residual_fit as fit
    from .v4_neighborhoods import coordination_cells
    from .v4_residual_common import ExecutedWarmScopeCache as NeighborhoodScopeCache,already_paid_lower
    from .v4_backends import chils_backend,CHILS_SHA256
except ImportError:
    from experiments import v4_residual_fit as fit
    from experiments.v4_neighborhoods import coordination_cells
    from experiments.v4_residual_common import ExecutedWarmScopeCache as NeighborhoodScopeCache,already_paid_lower
    from experiments.v4_backends import chils_backend,CHILS_SHA256


DEADLINES=(.139,.556,2.221)
CAPS=(256,)
CLASSICS=('Immediate','Upper','P1','Lower','RoundRobin','AnytimeGreedyPortfolio')
FIT_SEEDS=(17,29,43)
VARIANTS=('ResidualCapacity','ResidualFreeOccupancy','ResidualNoAux','ResidualNoWarmMembership','ResidualCheapSummary')
MAX_QUERIES=8
CALIBRATION_SHA256='495dab54ad7ca75e8a543338eb71da1049e20afdd35ad18825669d37d54c5bec'
MODULE_PATHS={
    'joint_recovery':'src/joint_recovery/__init__.py',
    'joint_recovery.core':'src/joint_recovery/core.py',
    'joint_recovery.v4_budgeted_recovery':'src/joint_recovery/v4_budgeted_recovery.py',
    'joint_recovery.v4_residual_controller_coordination':'src/joint_recovery/v4_residual_controller_coordination.py',
    'joint_recovery.v4_factors':'src/joint_recovery/v4_factors.py',
    'joint_recovery.v4_factors_fast':'src/joint_recovery/v4_factors_fast.py',
    'joint_recovery.v4_model':'src/joint_recovery/v4_model.py',
    'joint_recovery.v4_model_fast':'src/joint_recovery/v4_model_fast.py',
    'joint_recovery.v4_factorized_model':'src/joint_recovery/v4_factorized_model.py',
    'joint_recovery.v4_residual_model':'src/joint_recovery/v4_residual_model.py',
    'experiments.v4_residual_common':'experiments/v4_residual_common.py',
    'experiments.v4_residual_online_validation':'experiments/v4_residual_online_validation.py',
    'experiments.v4_residual_fit':'experiments/v4_residual_fit.py',
    'experiments.v4_residual_training_data':'experiments/v4_residual_training_data.py',
    'experiments.v4_neighborhoods':'experiments/v4_neighborhoods.py',
    'experiments.v4_backends':'experiments/v4_backends.py',
    'experiments.v3_solvers':'experiments/v3_solvers.py',
    'experiments.v3_published_baselines':'experiments/v3_published_baselines.py'}
_ACTIVE_OUT=None


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def json_ready(value):
    if isinstance(value,np.generic):return value.item()
    if is_dataclass(value):return {name:json_ready(item) for name,item in vars(value).items()}
    if isinstance(value,dict):return {str(name):json_ready(item) for name,item in value.items()}
    if isinstance(value,(tuple,list)):return [json_ready(item) for item in value]
    if isinstance(value,(frozenset,set)):return sorted(value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return dict(unserializable_type=type(value).__name__,representation=repr(value))


def _dump(node):return ast.dump(node,include_attributes=False)


def _coordination_ast():
    """Capture the existing reviewed derivation in an isolated namespace."""
    captured=[];namespace=dict(coordinated._derive_controller.__globals__)
    original_compile=compile
    def capture(tree,*args,**kwargs):captured.append(copy.deepcopy(tree));return original_compile(tree,*args,**kwargs)
    namespace['compile']=capture
    creator=FunctionType(coordinated._derive_controller.__code__,namespace)
    function,receipt=creator()
    if receipt!=coordinated.DERIVATION_RECEIPT or len(captured)!=1:
        raise RuntimeError('Reviewed coordinated derivation identity changed')
    return captured[0],dict(function.__globals__)


def _add_observer(tree,namespace,expected_hits,name):
    original=_dump(tree);tree=copy.deepcopy(tree);tree.body[0].name=name
    tree.body[0].args.kwonlyargs.append(ast.arg(arg='admission_observer'))
    tree.body[0].args.kw_defaults.append(ast.Constant(value=None))
    hits=0
    class Transform(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal hits
            target=node.targets[0] if len(node.targets)==1 else None
            if (isinstance(target,ast.Tuple) and len(target.elts)==2 and
                all(isinstance(item,ast.Name) for item in target.elts) and
                tuple(item.id for item in target.elts)==('best','best_gain') and
                isinstance(node.value,ast.Tuple) and len(node.value.elts)==2 and
                isinstance(node.value.elts[1],ast.Name) and node.value.elts[1].id=='gain'):
                hits+=1
                # The original now()/full validation/rescore has already run.
                # Retain the immutable references; do not copy large masks here.
                timestamp='last_clock'
                kind=('common_prefix','cheap_policy','native_search')[hits-1]
                follow=ast.parse('if admission_observer is not None:\n admission_observer(best,best_gain,'+timestamp+','+repr(kind)+')').body[0]
                return [node,follow]
            return self.generic_visit(node)
    tree=Transform().visit(tree);ast.fix_missing_locations(tree)
    if hits!=expected_hits:raise RuntimeError('Finite observer hooks did not match reviewed admissions')
    private=dict(namespace);exec(compile(tree,__file__+':observed','exec'),private)
    return private[name],dict(original_ast_sha256=sha256(original.encode()).hexdigest(),
        observed_ast_sha256=sha256(_dump(tree).encode()).hexdigest(),admission_hooks=hits,
        extra_clock_calls=0,shared_globals_changed=False)


_TREE,_GLOBALS=_coordination_ast()
run_coordinated_observed,COORDINATION_OBSERVER_DERIVATION=_add_observer(_TREE,_GLOBALS,3,'run_coordinated_observed')


@dataclass(frozen=True)
class OnlineCase:
    case_id: str
    spec: dict
    graph: object
    selected: frozenset
    cliques: tuple
    graph_weight: float
    input_status: str
    file_bindings: dict


def observable_projection(dataset):
    """Offline labels are replayed for provenance, then excluded from policy data."""
    if dataset.split!='validation' or len(dataset.cases)!=24:raise ValueError('Only the exact validation24 is authorized')
    return tuple(OnlineCase(case.case_id,dict(case.spec),case.graph,case.selected,case.cliques,
        case.graph_weight,case.status,dict(case.file_bindings)) for case in dataset.cases)


def unit_plan(cases):
    methods=tuple((variant,seed) for variant in VARIANTS for seed in FIT_SEEDS)+tuple((method,None) for method in CLASSICS)
    if len(cases)!=24 or len({case.case_id for case in cases})!=24:raise ValueError('Fixed24 graph registry required')
    return tuple(dict(unit_index=index,case_id=case.case_id,deadline_seconds=D,method=method,fit_seed=seed)
        for index,(case,D,method,seed) in enumerate((case,D,method,seed)
            for case in cases for D in DEADLINES for method,seed in methods))


class DecisionLog:
    """Paid immutable reference ledger; persistence is measurement logging later."""
    def __init__(self,original,started,deadline,graph=None):
        self.original=original;self.started=started;self.deadline=deadline;self.graph=graph
        self.admissions=[];self.admission_kinds=[];self.scopes={};self.raw_attempts=[];self.coverage=();self.actions=()
    def admission(self,selected,gain,ready,kind='unspecified'):
        if not isinstance(selected,frozenset) or not isfinite(ready):raise ValueError('Immutable actual certified membership required')
        self.admissions.append((selected,gain,ready,ready<self.deadline))
        self.admission_kinds.append(kind)
    def scope_factory(self,graph,S,factors):
        log=self
        class RecordedCache(NeighborhoodScopeCache):
            def scope(self,index,action,cap):
                scope=super().scope(index,action,cap);log.scopes[(index,cap)]=scope;return scope
        return RecordedCache(graph,S,factors)
    def proposal(self,graph,S,seed):
        cells=coordination_cells(graph,S,seed)
        if len(cells)!=11:raise ValueError('Full eleven-cell coverage changed')
        self.coverage=tuple(cells)  # References only; all large serialization later.
        self.actions=tuple(action for action,cell in cells if action is not None)
        return self.actions
    def backend(self,backend,require_binary_receipt=False):
        def execute(graph,scope,point,warm,remaining):
            row=dict(scope=scope,workpoint=point,warm_start=warm,remaining_before=remaining,
                started=time.perf_counter(),raw_return=None,exception=None)
            self.raw_attempts.append(row)
            try:
                result=backend(graph,scope,point,warm,remaining)
                if isinstance(result,core.RepairAttempt):
                    recovered=result.recovered
                    if recovered is not None:
                        try:recovered=tuple(recovered)
                        except TypeError:pass  # Source guard retains invalid scalar output.
                    result=core.RepairAttempt(recovered,result.status,result.diagnostics)
                    row['raw_return']=result
                    if require_binary_receipt and (not isinstance(result.diagnostics,dict) or
                            result.diagnostics.get('binary_sha256')!=CHILS_SHA256):
                        raise ValueError('Actual repair receipt is not the frozen native CHILS kernel')
                row['raw_return']=result
                return result
            except Exception as error:
                row['exception']=dict(type=type(error).__name__,message=str(error));raise
            finally:row['backend_returned']=time.perf_counter()
        return execute


def _final_membership(graph,selected):
    values=tuple(selected)
    if any(not isinstance(v,Integral) or isinstance(v,bool) for v in values) or len(values)!=len(set(values)):
        raise ValueError('Final membership has nonexact/duplicate IDs')
    checked=_validate_state(graph,frozenset(values))
    return checked,core.objective(graph,checked)


def deadline_outputs(log,receipt_selected,final_selected,final_value,initial_value,final_ready,valid):
    timely=[record for record in log.admissions if record[3]]
    certified=timely[-1][0] if timely else log.original
    certified_gain=timely[-1][1] if timely else 0
    returned=final_selected if valid and final_ready<log.deadline else log.original
    returned_gain=max(0,final_value-initial_value) if valid and final_ready<log.deadline else 0
    return dict(certified_selected=certified,certified_gain=certified_gain,
        final_return_selected=returned,final_return_gain=returned_gain,
        final_return_validated_by_D=bool(valid and final_ready<log.deadline),
        source_certified_available_by_D=True,source_improvement_certified_by_D=bool(timely),
        final_validation_ready_seconds=final_ready-log.started)


def make_policy(method,model,device):
    if method in VARIANTS:return learned.priority_callback_residual(model,device=device)
    if method=='AnytimeGreedyPortfolio':return None
    if method=='Lower':
        return lambda graph,views,state:core.PriorityEvaluation(
            tuple(already_paid_lower(graph,v.request.scope,v.warm_start) for v in views),(),
            'already paid executed common warm; no repeated greedy sweep',True)
    mode={'Immediate':'immediate','Upper':'upper','P1':'p1','RoundRobin':'round_robin'}[method]
    return classical_priority_fast(mode)


def run_unit(case,method,seed,deadline,model,backend,workpoints,device='cuda',clock=time.perf_counter,
        require_binary_receipt=True):
    """All policy/cache construction, source execution and final checks are paid."""
    if case.graph is None:
        return dict(status='declared_input_unavailable',case_id=case.case_id,method=method,fit_seed=seed,
            deadline_seconds=deadline,graph_weight=case.graph_weight,domain=case.spec['domain'],
            certified_gain=0,final_return_gain=0,certified_relative_gain=0.,final_return_relative_gain=0.,
            source_certified_available_by_D=False,final_return_validated_by_D=False,
            initial_value=None,source_receipt=None,trace=None,failure='original graph unavailable; not replaced')
    if device=='cuda':torch.cuda.synchronize()
    started=clock();absolute_deadline=started+deadline
    if not isfinite(started) or not isfinite(absolute_deadline) or not isfinite(deadline) or deadline<=0:
        raise ValueError('Finite positive original wall deadline and initial clock required')
    log=DecisionLog(case.selected,started,absolute_deadline,case.graph)
    source=None;failure=None;initial=None;checked=case.selected;value=None
    try:
        policy=make_policy(method,model,device)
        policy_ready=clock();remaining=absolute_deadline-policy_ready
        if remaining<=0:raise TimeoutError('Policy/cache construction exhausted deadline')
        proposals=lambda g,S:log.proposal(g,S,case.spec['seed']+1000)
        source=run_coordinated_observed(case.graph,case.selected,proposals,log.backend(backend,require_binary_receipt),policy,
            remaining,caps=CAPS,workpoints=workpoints,
            max_queries=0 if method=='AnytimeGreedyPortfolio' else MAX_QUERIES,resource_cliques=case.cliques,
            scope_cache_factory=log.scope_factory,admission_observer=log.admission,clock=clock)
        # Existing source has already validated admissions. This additional full
        # final check/rescore is part of the strict actual return metric too.
        initial=source.incumbent_value
        checked,value=_final_membership(case.graph,source.selected)
        final_valid=True
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));final_valid=False
        initial=core.objective(case.graph,case.selected) if initial is None else initial
    if device=='cuda':torch.cuda.synchronize()
    ready=clock()
    outputs=deadline_outputs(log,source.selected if source else case.selected,checked,
        value if value is not None else initial,initial,ready,final_valid)
    denominator=max(1,initial)
    row=dict(status='complete_unit' if failure is None else 'unit_failed_retained',
        case_id=case.case_id,method=method,fit_seed=seed,deadline_seconds=deadline,
        graph_weight=case.graph_weight,domain=case.spec['domain'],initial_value=initial,
        policy_cache_setup_seconds=(policy_ready-started) if 'policy_ready' in locals() else None,
        source_clock_seconds=source.controller_return_sample_seconds if source else None,
        outer_return_seconds=ready-started,outer_deadline_miss=ready>=absolute_deadline,
        final_validation_ready_timestamp=ready,
        final_selected_valid=final_valid,raw_final_value=value,
        certified_relative_gain=outputs['certified_gain']/denominator,
        final_return_relative_gain=outputs['final_return_gain']/denominator,
        source_receipt=source,trace=log,failure=failure,**outputs,
        physical_hard_deadline_certified=False)
    add_prefix_decomposition(row)
    return row


def add_prefix_decomposition(row):
    """Actual same-unit timely common prefix, separated from native value.

    The ledger is captured inside the paid validation; no greedy is rerun to
    create a counterfactual floor. Strict-return failure still yields zero.
    """
    log=row.get('trace');records=log.admissions if log else ()
    kinds=log.admission_kinds if log else ()
    if len(records)!=len(kinds):raise ValueError('Every observed admission requires its actual stage')
    prefix=[r for r,k in zip(records,kinds) if k=='common_prefix' and r[3]]
    known=prefix[-1] if prefix else None
    gain=known[1] if known else 0
    row['common_prefix_selected']=known[0] if known else (log.original if log else frozenset())
    row['common_prefix_certified_gain']=gain
    row['certified_gain_above_prefix']=max(0,row.get('certified_gain',0)-gain)
    row['returned_prefix_component']=min(gain,row.get('final_return_gain',0))
    row['returned_gain_above_prefix']=max(0,row.get('final_return_gain',0)-gain)
    row['prefix_semantics']='actual same-unit timely executed prefix; scope preparation substage, never an extra stage sum'
    return row


def actual_receipt_boundary(row,received):
    """The caller samples immediately after run_unit really returns, before I/O.

    Final-validation readiness remains a separate diagnostic. It cannot stand
    in for this external interface-receipt time in the primary metric.
    """
    log=row.get('trace')
    if log is None:
        row['actual_interface_receipt_seconds']=None
        row['primary_boundary']='actual caller receipt, unavailable graph retained'
        return row
    elapsed=received-log.started
    if not isfinite(received) or elapsed<row['final_validation_ready_seconds']:
        raise ValueError('Finite monotonic actual receipt must follow final-validation readiness')
    row['validation_ready_before_D']=row['final_return_validated_by_D']
    row['actual_interface_receipt_seconds']=elapsed
    row['actual_interface_receipt_timestamp']=received
    row['outer_return_seconds']=elapsed;row['outer_deadline_miss']=received>=log.deadline
    row['primary_boundary']='actual caller receipt after complete run_unit return'
    if received>=log.deadline:
        row['final_return_selected']=log.original;row['final_return_gain']=0
        row['final_return_relative_gain']=0.;row['final_return_validated_by_D']=False
    add_prefix_decomposition(row)
    return row


def _flatten_memberships(groups,preserve_order=False):
    groups=tuple(groups);pointer=np.cumsum([0]+[len(group) for group in groups],dtype=np.int64)
    values=np.fromiter((v for group in groups for v in (group if preserve_order else sorted(group))),
        np.int64,count=int(pointer[-1]))
    return values,pointer


def save_unit(folder,row,original):
    """Membership IDs remain actual complete arrays; no object/pickle NPZ."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    arrays={'original_selected':np.asarray(sorted(original),np.int64)}
    for name in ('certified_selected','final_return_selected','common_prefix_selected'):
        arrays[name]=np.asarray(sorted(row.get(name,original)),np.int64)
    source=row.get('source_receipt');arrays['source_selected']=np.asarray(sorted(source.selected if source else original),np.int64)
    log=row.get('trace');admissions=log.admissions if log else ()
    arrays['admission_selected'],arrays['admission_ptr']=_flatten_memberships(record[0] for record in admissions)
    scopes=tuple(log.scopes.values()) if log else ()
    for name,groups in (('scope_R',(scope.replacements for scope in scopes)),('scope_B',(scope.base for scope in scopes)),
            ('scope_D',(scope.displaced for scope in scopes)),('scope_C',(scope.inserts for scope in scopes))):
        arrays[name],arrays[name+'_ptr']=_flatten_memberships(groups,preserve_order=name=='scope_R')
    extras=tuple(log.actions[scope.action_index].releases for scope in scopes) if log else ()
    arrays['scope_E'],arrays['scope_E_ptr']=_flatten_memberships(extras)
    attempts=tuple(getattr(source,'attempts',()))
    for name,groups in (('attempt_warm',(record.warm_start for record in attempts)),
            ('attempt_returned',(record.recovered or () for record in attempts))):
        arrays[name],arrays[name+'_ptr']=_flatten_memberships(groups)
    with (folder/'membership.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    serial={key:value for key,value in row.items() if key!='trace'}
    scope_records=[dict(vars(scope),releases=log.actions[scope.action_index].releases,
        action_identity=log.actions[scope.action_index].identity) for scope in scopes] if log else []
    raw_records=[]
    if log:
        # Logging-only exact signed audit; no after-return result enters a policy
        # or its certified/strict on-time metric.
        by_key={attempt.key:attempt for attempt in attempts}
        for raw in log.raw_attempts:
            record=dict(raw);scope=record['scope'];point=record['workpoint']
            actual=by_key.get((scope.action_index,scope.cap,point.name))
            record['source_attempt_valid']=bool(actual and actual.valid)
            record['actual_signed_gain']=None;record['q_plus_recovery_gain']=None
            if actual and actual.valid:
                record['actual_signed_gain']=core.objective(log.graph,scope.base|actual.recovered)-row['initial_value']
                record['q_plus_recovery_gain']=scope.immediate_gain+core.objective(log.graph,actual.recovered)
                record['signed_decomposition_difference']=record['q_plus_recovery_gain']-record['actual_signed_gain']
            raw_records.append(record)
    serial['trace']=dict(started_timestamp=log.started,deadline_timestamp=log.deadline,
        admissions=[dict(selected=selected,gain=gain,ready_elapsed=ready-log.started,
            ready_timestamp=ready,on_time=on_time,kind=kind)
        for (selected,gain,ready,on_time),kind in zip(admissions,log.admission_kinds)],scope_order=[vars(scope) for scope in scopes],
        full_scope_records=scope_records,raw_attempts=raw_records,coverage=log.coverage) if log else None
    serial['membership_npz_sha256']=digest(folder/'membership.npz')
    write_json(folder/'receipt.json',json_ready(serial))
    return {name:dict(sha256=digest(folder/name),bytes=(folder/name).stat().st_size)
        for name in ('membership.npz','receipt.json')}


def select_classical(rows):
    """Predeclared strict-return primary middle-D, all graphs/domain weights."""
    scores={method:0. for method in CLASSICS};certified={method:0. for method in CLASSICS};seen=set()
    for row in rows:
        if row['deadline_seconds']!=.556 or row['method'] not in CLASSICS:continue
        key=(row['case_id'],row['method'])
        if key in seen:raise ValueError('Duplicate classical validation cell')
        seen.add(key);scores[row['method']]+=row['graph_weight']*row['final_return_relative_gain']
        certified[row['method']]+=row['graph_weight']*row['certified_relative_gain']
    if len(seen)!=24*6:raise ValueError('Complete six-classic24 middle-deadline coverage required')
    selected=max(CLASSICS,key=lambda name:(scores[name],-CLASSICS.index(name)))
    return dict(selected=selected,selection_metric='equal_domain_graph_final_return_relative_gain_at_D0.556',
        strict_return_scores=scores,source_certified_scores=certified,
        tie_rule='first_in_predeclared_classic_order',confirmation_not_generated=True)


def validate_source_closure(root,path,expected_sha):
    root=Path(root).resolve()
    if digest(path)!=expected_sha:raise ValueError('Caller-pinned online source manifest differs')
    manifest=fit._read_json(path);records=manifest['files'];by_path={row['path']:row['sha256'] for row in records}
    required=set(MODULE_PATHS.values())|{'tests/test_v4_residual_online_validation.py'}
    if len(by_path)!=len(records) or not required<=set(by_path):raise ValueError('Complete online source/test closure required')
    for relative,expected in by_path.items():
        source=(root/relative).resolve()
        try:source.relative_to(root.resolve())
        except ValueError:raise ValueError('Online source path escapes capsule')
        if digest(source)!=expected:raise ValueError('Online source bytes changed: '+relative)
    origins=[]
    for name,module in tuple(sys.modules.items()):
        relative=MODULE_PATHS.get(name)
        if name=='__main__' and Path(getattr(module,'__file__','')).resolve()==Path(__file__).resolve():
            relative='experiments/v4_residual_online_validation.py'
        owned=(name=='joint_recovery' or name.startswith('joint_recovery.') or
            name.startswith('experiments.v4_') or name in MODULE_PATHS)
        if relative is None and owned and getattr(module,'__file__',None):
            # Only explicit owned project namespaces; generic third-party core
            # and generators basenames are deliberately not matched.
            try:relative=Path(module.__file__).resolve().relative_to(root).as_posix()
            except ValueError:raise ValueError('Online scientific import outside capsule: '+name)
            if relative not in by_path:raise ValueError('Unexpected scientific origin missing from capsule: '+name)
        if relative:
            source=Path(module.__file__).resolve()
            if source!=root/relative or digest(source)!=by_path[relative]:
                raise ValueError('Online actual import mismatch: module=%s path=%s'%(name,source))
            origins.append(dict(module=name,path=relative,sha256=by_path[relative]))
    if not set(MODULE_PATHS.values())<=set(row['path'] for row in origins):
        raise ValueError('Required actual online import origins missing')
    return dict(manifest_sha256=expected_sha,files=records,execution_origins=origins)


def load_models(fit_root,protocol_sha,completion_sha,collection_protocol_sha,collection_completion_sha,device):
    root=Path(fit_root)
    if digest(root/'protocol.json')!=protocol_sha or digest(root/'completion.json')!=completion_sha:
        raise ValueError('Caller-pinned complete fitting receipts differ')
    protocol=fit._read_json(root/'protocol.json');complete=fit._read_json(root/'completion.json')
    if (complete['status']!='complete_all_predeclared_fits_not_policy_evidence' or
        complete['protocol_sha256']!=protocol_sha or not complete['all_fit_seeds_retained'] or
        not complete['no_online_validation'] or complete['confirmation_graphs_generated']!=0 or
        protocol['fit_seeds']!=list(FIT_SEEDS) or not protocol['all_seeds_retained']):
        raise ValueError('Complete fixed fitting registry required')
    if protocol['variants']!=list(VARIANTS):
        raise ValueError('Only the five reserved residual architectures are allowed')
    bindings=protocol['bindings']
    if (bindings['collection_protocol_sha256']!=collection_protocol_sha or
        bindings['collection_completion_sha256']!=collection_completion_sha or
        bindings['calibration_sha256']!=CALIBRATION_SHA256):raise ValueError('Fit labels/calibration binding differs')
    entries={(entry['variant'],entry['fit_seed']):entry for entry in complete['fits']}
    if len(entries)!=len(complete['fits']):raise ValueError('Duplicate fitting entries')
    expected={(variant,seed) for variant in VARIANTS for seed in FIT_SEEDS}
    extras=set(entries)-expected
    if set(entries)!=expected:
        raise ValueError('Exactly all fifteen reserved checkpoints are required')
    models={};records=[]
    for variant in VARIANTS:
        for seed in FIT_SEEDS:
            entry=entries[(variant,seed)];folder=root/(variant+'_seed%d'%seed)
            if set(entry['files'])!={'selected.pt','epoch40.pt','history.json'}:
                raise ValueError('Each complete fit must bind exactly all three declared artifacts')
            child=fit._read_json(folder/'completion.json')
            if digest(folder/'completion.json')!=entry['completion_sha256'] or child['files']!=entry['files']:
                raise ValueError('Individual fit completion binding differs')
            if (child['status']!='complete_fixed_fit_not_online_policy_validation' or
                    child['variant']!=variant or child['fit_seed']!=seed or child['bindings']!=bindings or
                    not child['all40_epochs_retained']):raise ValueError('Individual fit registry/provenance differs')
            for name,binding in entry['files'].items():
                if name not in ('selected.pt','epoch40.pt','history.json') or digest(folder/name)!=binding['sha256'] or (folder/name).stat().st_size!=binding['bytes']:
                    raise ValueError('Complete fit artifact bytes differ')
            history=fit._read_json(folder/'history.json');epochs=history['epochs']
            if len(epochs)!=40 or [epoch['epoch'] for epoch in epochs]!=list(range(1,41)):
                raise ValueError('All forty fixed validation-selection epochs required')
            selected_epoch=fit.first_minimum_epoch([epoch['validation']['graph_domain_balanced_regret'] for epoch in epochs])
            if selected_epoch!=entry['selected_epoch'] or selected_epoch!=history['selected_epoch']:
                raise ValueError('Checkpoint is not the frozen first-minimum epoch')
            config=fit.FitConfig(variant,seed);path=folder/'selected.pt';expected_sha=entry['files']['selected.pt']['sha256']
            model,record=fit.load_bound_checkpoint(path,expected_sha,config,bindings,device)
            if record['epoch']!=selected_epoch:raise ValueError('Selected checkpoint epoch differs from complete history')
            model.eval();models[(variant,seed)]=model
            records.append(dict(variant=variant,fit_seed=seed,selected_epoch=selected_epoch,
                checkpoint_relative=variant+'_seed%d/selected.pt'%seed,checkpoint_sha256=expected_sha,
                completion_sha256=entry['completion_sha256'],history_sha256=entry['files']['history.json']['sha256']))
    return models,records


def calibration(path):
    if digest(path)!=CALIBRATION_SHA256:raise ValueError('Fixed calibration bytes changed')
    value=fit._read_json(path)
    if value['total_deadlines_seconds']!=list(DEADLINES) or value['caps']!=[64,256,1024]:
        raise ValueError('Fixed total wall budget/action-range grid changed')
    points=tuple(core.Workpoint(point['name'],point['kind'],point['amount'],point['expected_seconds']) for point in value['workpoints'])
    if [(p.amount,p.expected_seconds) for p in points]!=[(.01,.035),(.05,.075),(.2,.214)]:
        raise ValueError('Shared native workpoint/cost calibration changed')
    return points


def load_cases(root,protocol_sha,completion_sha):
    dataset=fit.load_dataset(root,'validation',expected_protocol_sha256=protocol_sha,expected_completion_sha256=completion_sha)
    return observable_projection(dataset)


def freeze(args):
    global _ACTIVE_OUT
    out=Path(args.out)
    if out.exists():raise FileExistsError('Never replace an online protocol/attempt')
    root=Path(__file__).resolve().parents[1]
    sources=validate_source_closure(root,args.source_manifest,args.source_manifest_sha256)
    cases=load_cases(args.collection_root,args.protocol_sha256,args.completion_sha256)
    _,weights=load_models(args.fit_root,args.fit_protocol_sha256,args.fit_completion_sha256,
        args.protocol_sha256,args.completion_sha256,'cpu')
    calibration(args.calibration)
    if digest(args.chils)!=CHILS_SHA256:raise ValueError('Actual native CHILS-p1 binary differs')
    plan=unit_plan(cases)
    if len(plan)!=1512:raise AssertionError('Complete online matrix changed')
    out.mkdir(parents=True);_ACTIVE_OUT=out
    protocol=dict(status='frozen_validation24_actual_online_before_execution',units=plan,expected_units=1512,
        case_registry=[dict(case_id=c.case_id,spec=c.spec,graph_weight=c.graph_weight,
            original_status=c.input_status,file_bindings=c.file_bindings) for c in cases],
        variants=VARIANTS,fit_seeds=FIT_SEEDS,classics=CLASSICS,deadlines=DEADLINES,caps=CAPS,max_queries=MAX_QUERIES,
        collector_protocol_sha256=args.protocol_sha256,collector_completion_sha256=args.completion_sha256,
        fit_protocol_sha256=args.fit_protocol_sha256,fit_completion_sha256=args.fit_completion_sha256,
        calibration_sha256=CALIBRATION_SHA256,checkpoint_records=weights,source_capsule=sources,
        coordination_observer=COORDINATION_OBSERVER_DERIVATION,
        executed_common_greedy_prefix=True,fixed_scope_cap=256,
        greedy_only_queries=0,lower_uses_already_paid_warm=True,
        chils_sha256=CHILS_SHA256,chils_population=1,chils_threads=1,chils_seed=17,
        device=args.device,cpu_threads=1,resident_graph_models_backend_factory=True,
        decision_caches='fresh per unit; all static/embedding/solver-output caches start empty',
        strict_return='actual caller receipt immediately after run_unit returns, including source output/additional full membership/native rescore/device sync/audit row construction; strictly before total wall D or original S',
        certified_available='only paid source full validation/rescore trace with actual ready strictly before outer D; separate from interface return',
        classic_selection='middleD0.556 equal-domain/graph strict-return gain/max(1,original objective); six fixed classics; first declared tie',
        persistence_cost='offline measurement logging after actual decision return; not policy computation',
        no_confirmation_graphs=True,no_teacher_input=True,no_resume_retry_or_replacement=True,
        physical_host_idle_verified=False)
    write_json(out/'protocol.json',protocol)
    write_json(out/'freeze_completion.json',dict(status='complete_online_validation_freeze_not_execution',
        expected_units=1512,protocol_sha256=digest(out/'protocol.json'),source_manifest_sha256=args.source_manifest_sha256))
    return protocol


def run(args):
    global _ACTIVE_OUT
    out=Path(args.out)
    if (out/'failure.json').exists():raise FileExistsError('Failed online attempt is preserved; no retry')
    if digest(out/'protocol.json')!=args.online_protocol_sha256:raise ValueError('Caller-pinned online protocol differs')
    protocol=fit._read_json(out/'protocol.json');closed=fit._read_json(out/'freeze_completion.json')
    if closed['protocol_sha256']!=args.online_protocol_sha256 or closed['expected_units']!=1512:
        raise ValueError('Incomplete or changed online freeze')
    if (out/'opened.json').exists() or (out/'completion.json').exists():raise FileExistsError('Online validation is exactly one attempt; no retry')
    _ACTIVE_OUT=out
    root=Path(__file__).resolve().parents[1]
    sources=validate_source_closure(root,args.source_manifest,args.source_manifest_sha256)
    if sources!=protocol['source_capsule']:raise ValueError('Online scientific closure changed after freeze')
    check_frozen_inputs(args,protocol)
    if args.device!=protocol['device']:raise ValueError('Frozen online device changed')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Frozen resident CUDA device unavailable')
    if sys.version_info[:2]!=(3,8) or str(torch.__version__).split('+')[0]!='1.11.0':
        raise RuntimeError('Actual online validation requires the calibrated Python3.8/PyTorch1.11.0 runtime')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    setup=time.perf_counter();cases=load_cases(args.collection_root,args.protocol_sha256,args.completion_sha256)
    models,weights=load_models(args.fit_root,args.fit_protocol_sha256,args.fit_completion_sha256,
        args.protocol_sha256,args.completion_sha256,args.device)
    if weights!=protocol['checkpoint_records'] or list(unit_plan(cases))!=protocol['units']:
        raise ValueError('Frozen own weights/complete unit plan changed')
    points=calibration(args.calibration);backend=chils_backend(args.chils)
    if args.device=='cuda':torch.cuda.synchronize()
    resident_setup=time.perf_counter()-setup
    write_json(out/'opened.json',dict(status='validation_online_opened_no_confirmation',
        protocol_sha256=args.online_protocol_sha256,source_manifest_sha256=args.source_manifest_sha256,
        resident_setup_seconds=resident_setup,expected_units=1512,utc=datetime.now(timezone.utc).isoformat(),
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
            device=args.device,cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None)))
    by_case={case.case_id:case for case in cases};rows=[];artifacts=[]
    with (out/'progress.jsonl').open('x',encoding='utf-8',buffering=1) as progress:
        for unit in protocol['units']:
            case=by_case[unit['case_id']];model=models.get((unit['method'],unit['fit_seed']))
            row=run_unit(case,unit['method'],unit['fit_seed'],unit['deadline_seconds'],model,backend,points,args.device)
            received=time.perf_counter()  # Actual external receipt, before any mutation or persistence.
            actual_receipt_boundary(row,received)
            row['unit_index']=unit['unit_index']
            files=save_unit(out/'units'/('unit_%04d'%unit['unit_index']),row,case.selected)
            artifacts.append(dict(unit_index=unit['unit_index'],files=files))
            small={name:json_ready(value) for name,value in row.items() if name not in (
                'source_receipt','trace','certified_selected','final_return_selected','common_prefix_selected')}
            rows.append(small)
            event=dict(status='nonfinal_completed_online_unit',completed_units=len(rows),expected_units=1512,
                case_id=unit['case_id'],method=unit['method'],fit_seed=unit['fit_seed'],deadline_seconds=unit['deadline_seconds'],
                unit_status=row['status'],artifact_sha256=files)
            progress.write(json.dumps(event,allow_nan=False)+'\n');progress.flush()
            print(json.dumps({key:value for key,value in event.items() if key!='artifact_sha256'},allow_nan=False),flush=True)
    if len(rows)!=1512:raise AssertionError('Full online matrix incomplete')
    # Byte identity is checked again; no execution is retried after an anomaly.
    validate_source_closure(root,args.source_manifest,args.source_manifest_sha256)
    final_cases=load_cases(args.collection_root,args.protocol_sha256,args.completion_sha256)
    if [c.file_bindings for c in final_cases]!=[c.file_bindings for c in cases]:raise ValueError('Collection changed during online validation')
    _,final_weights=load_models(args.fit_root,args.fit_protocol_sha256,args.fit_completion_sha256,
        args.protocol_sha256,args.completion_sha256,'cpu')
    if final_weights!=weights:raise ValueError('Complete fitting/checkpoint bytes changed during validation')
    if digest(args.chils)!=CHILS_SHA256:raise ValueError('Native binary bytes changed during online validation')
    selection=select_classical(rows)
    write_json(out/'classical_selection.json',selection)
    write_json(out/'completion.json',dict(status='complete_validation24_actual_online_not_confirmation',
        completed=True,expected_units=1512,completed_units=1512,rows=rows,artifacts=artifacts,
        protocol_sha256=args.online_protocol_sha256,opened_sha256=digest(out/'opened.json'),
        progress_sha256=digest(out/'progress.jsonl'),classical_selection_sha256=digest(out/'classical_selection.json'),
        selected_classical=selection['selected'],unit_failures=sum(row['status']!='complete_unit' for row in rows),
        no_confirmation_graphs=True,no_learning_advantage_claim=True,physical_host_idle_verified=False,
        utc=datetime.now(timezone.utc).isoformat()))


def check_frozen_inputs(args,protocol):
    for name,field in (('protocol_sha256','collector_protocol_sha256'),('completion_sha256','collector_completion_sha256'),
            ('fit_protocol_sha256','fit_protocol_sha256'),('fit_completion_sha256','fit_completion_sha256')):
        if getattr(args,name)!=protocol[field]:raise ValueError('Frozen validation input binding differs: '+name)
    if (protocol['expected_units']!=1512 or protocol['max_queries']!=MAX_QUERIES or
            protocol['deadlines']!=list(DEADLINES) or protocol['caps']!=list(CAPS) or
            protocol['variants']!=list(VARIANTS) or protocol['fit_seeds']!=list(FIT_SEEDS) or
            protocol['classics']!=list(CLASSICS) or protocol['calibration_sha256']!=CALIBRATION_SHA256 or
            protocol.get('executed_common_greedy_prefix') is not True or protocol.get('fixed_scope_cap')!=256 or
            protocol.get('greedy_only_queries')!=0 or protocol.get('lower_uses_already_paid_warm') is not True):
        raise ValueError('Frozen complete online policy/total budget design differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('freeze','run'),required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--collection-root',required=True);parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--completion-sha256',required=True);parser.add_argument('--fit-root',required=True)
    parser.add_argument('--fit-protocol-sha256',required=True);parser.add_argument('--fit-completion-sha256',required=True)
    parser.add_argument('--source-manifest',required=True);parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--calibration',required=True);parser.add_argument('--chils',required=True)
    parser.add_argument('--online-protocol-sha256');parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    if args.stage=='run' and not args.online_protocol_sha256:parser.error('run requires the frozen online protocol SHA')
    return freeze(args) if args.stage=='freeze' else run(args)


def entrypoint():
    try:return main()
    except Exception as error:
        if _ACTIVE_OUT is not None and _ACTIVE_OUT.is_dir() and not any(
                (_ACTIVE_OUT/name).exists() for name in ('completion.json','failure.json')):
            write_json(_ACTIVE_OUT/'failure.json',dict(status='failed_online_validation_no_retry',completed=False,
                exception_type=type(error).__name__,message=str(error),no_confirmation_graphs=True))
        raise


if __name__=='__main__':entrypoint()
