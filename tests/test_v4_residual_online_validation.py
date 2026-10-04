"""Finite observable/interface/deadline guards; no fresh corpus or real solver."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments import v4_residual_online_validation as online
from experiments.v4_neighborhoods import CoordinationAction,NeighborhoodScopeCache
from joint_recovery import v4_budgeted_recovery as core


def fixture(native=False):
    weights=np.asarray([10,10,12,12],np.int64)
    graph=core.NativeIntegerGraph(weights,np.asarray([0,1,0,1]),
        (frozenset([2]),frozenset([3]),frozenset([0]),frozenset([1])))
    return graph,frozenset([0,1]),CoordinationAction((),(0,1))


def case(index=0):
    graph,S,action=fixture()
    return online.OnlineCase('validation_%02d'%index,dict(seed=17+index,domain='menu' if index<12 else 'resource'),
        graph,S,(),1/24,'complete',{'fixture':'not_a_production_binding'})


def cells(action):
    return [(action,{'status':'prepared'})]+[(None,{'status':'actual_action_alias'}) for _ in range(10)]


class Clock:
    def __init__(self,step=.0001):self.value=0.;self.step=step
    def __call__(self):self.value+=self.step;return self.value


class OnlineValidationTests(unittest.TestCase):
    def test_observer_finite_hooks_preserve_frozen_shared_function_and_clock_calls(self):
        self.assertEqual(online.COORDINATION_OBSERVER_DERIVATION['admission_hooks'],3)
        self.assertEqual(online.COORDINATION_OBSERVER_DERIVATION['extra_clock_calls'],0)
        self.assertNotIn('admission_observer',online.coordinated.run_coordinated_recovery.__code__.co_varnames)
        self.assertEqual(online.digest(Path(online.core.__file__)),
            'b4b5bbd3d8abdeba2fcb9a117c54f161786c2deccc032764e792d5899bd3d1ad')

    def test_strict_deadline_return_rolls_back_while_certified_reference_retained(self):
        original=frozenset([0]);better=frozenset([1]);late=frozenset([2])
        log=online.DecisionLog(original,100.,101.)
        log.admission(better,4,100.8);log.admission(late,9,101.)
        self.assertIs(log.admissions[0][0],better)
        output=online.deadline_outputs(log,late,late,19,10,101.,True)
        self.assertIs(output['certified_selected'],better)
        self.assertEqual(output['certified_gain'],4)
        self.assertEqual(output['final_return_selected'],original)
        self.assertEqual(output['final_return_gain'],0)
        self.assertFalse(output['final_return_validated_by_D'])
        self.assertEqual(online.deadline_outputs(log,better,better,14,10,100.99,True)['final_return_gain'],4)
        row=dict(trace=log,certified_gain=4,final_return_selected=better,final_return_gain=4,
            final_return_relative_gain=.4,final_return_validated_by_D=True,final_validation_ready_seconds=.99)
        online.actual_receipt_boundary(row,101.)
        self.assertTrue(row['validation_ready_before_D']);self.assertFalse(row['final_return_validated_by_D'])
        self.assertEqual(row['final_return_gain'],0);self.assertEqual(row['certified_gain'],4)
        self.assertEqual(row['final_return_selected'],original)

    def test_exact_native_rescore_above_float_precision_and_fractional_ids_rejected(self):
        graph=core.NativeIntegerGraph(np.asarray([2**53+1,2**53+2],np.int64),np.asarray([0,1]),
            (frozenset([1]),frozenset([0])))
        old,value=online._final_membership(graph,[0]);new,newvalue=online._final_membership(graph,[1])
        self.assertEqual(newvalue-value,1);self.assertIsInstance(newvalue,int)
        for selected in ([0,0],[True],[1.0]):
            with self.assertRaises(ValueError):online._final_membership(graph,selected)

    def test_original_24_by_3_by_21_complete_plan_and_teacher_removed(self):
        source_cases=[SimpleNamespace(**dict(vars(case(i)),states=('OFFLINE_LABEL_NOT_ALLOWED',))) for i in range(24)]
        # Projection consumes only original observables and saved provenance.
        for item in source_cases:item.status=item.input_status
        dataset=SimpleNamespace(split='validation',cases=source_cases)
        projected=online.observable_projection(dataset);plan=online.unit_plan(projected)
        self.assertEqual(len(plan),1512);self.assertFalse(hasattr(projected[0],'states'))
        self.assertEqual({(p['method'],p['fit_seed']) for p in plan},
            {(v,s) for v in online.VARIANTS for s in online.FIT_SEEDS}|{(v,None) for v in online.CLASSICS})
        self.assertEqual({p['deadline_seconds'] for p in plan},set(online.DEADLINES))
        with self.assertRaises(ValueError):online.unit_plan(projected[:23])
        with self.assertRaises(ValueError):online.observable_projection(SimpleNamespace(split='training',cases=source_cases))
        unavailable=replace(projected[0],graph=None)
        row=online.run_unit(unavailable,'Upper',None,.556,None,None,(),'cpu')
        self.assertEqual(row['status'],'declared_input_unavailable');self.assertEqual(row['final_return_gain'],0)

    def test_actual_coordinated_scope_known_warm_dedup_and_raw_generator_trace(self):
        graph,S,action=fixture();observed=[]
        def repair(g,scope,point,warm,remaining):
            observed.append((scope,warm,remaining));return core.RepairAttempt(iter([2,3]),'fixture',{})
        with patch.object(online,'coordination_cells',return_value=cells(action)):
            row=online.run_unit(case(),'RoundRobin',None,1.,None,repair,
                (core.Workpoint('finite','seconds',.01,.001),),'cpu',Clock(),False)
        self.assertEqual(row['status'],'complete_unit');self.assertEqual(row['final_return_gain'],4)
        self.assertTrue(observed);self.assertEqual(observed[0][1],frozenset([2,3]))
        self.assertEqual(row['source_receipt'].duplicate_scope_requests_skipped,0)
        self.assertEqual(row['trace'].raw_attempts[0]['raw_return'].recovered,(2,3))
        self.assertTrue(row['source_improvement_certified_by_D'])
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'unit';online.save_unit(folder,row,S)
            receipt=json.loads((folder/'receipt.json').read_text())
            self.assertEqual(receipt['trace']['full_scope_records'][0]['releases'],[0,1])
            self.assertEqual(receipt['trace']['raw_attempts'][0]['actual_signed_gain'],4)
            with np.load(folder/'membership.npz',allow_pickle=False) as data:
                self.assertEqual(tuple(data['final_return_selected']),(2,3))
                self.assertEqual(tuple(data['scope_E'][:2]),(0,1))

    def test_malformed_backend_return_preserves_failure_receipt_and_original(self):
        graph,S,action=fixture()
        with patch.object(online,'coordination_cells',return_value=cells(action)):
            row=online.run_unit(case(),'RoundRobin',None,1.,None,
                lambda *args:core.RepairAttempt(42,'fixture',None),
                (core.Workpoint('finite','seconds',.01,.001),),'cpu',Clock(),False)
        self.assertEqual(row['final_return_selected'],frozenset([2,3]))
        self.assertFalse(row['source_receipt'].attempts[0].valid)
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'unit';online.save_unit(folder,row,S)
            receipt=json.loads((folder/'receipt.json').read_text())
            self.assertEqual(receipt['trace']['raw_attempts'][0]['raw_return']['recovered'],42)
            self.assertIsNone(receipt['trace']['raw_attempts'][0]['actual_signed_gain'])

    def test_actual_warm_updates_between_spent_workpoints_no_fresh_label(self):
        graph=core.NativeIntegerGraph(np.asarray([10,10,6,6,6,6],np.int64),np.asarray([0,1,0,0,1,1]),
            (frozenset([2,3]),frozenset([4,5]),frozenset([0]),frozenset([0]),frozenset([1]),frozenset([1])))
        S=frozenset([0,1]);action=CoordinationAction((),(0,1));warm_observed=[]
        item=replace(case(),graph=graph,selected=S)
        def repair(g,scope,point,warm,remaining):
            warm_observed.append(warm)
            return core.RepairAttempt(frozenset([1,2,3] if len(warm_observed)==1 else [2,3,4,5]),'fixture',{})
        with patch.object(online,'coordination_cells',return_value=cells(action)):
            row=online.run_unit(item,'RoundRobin',None,1.,None,repair,
                (core.Workpoint('one','seconds',.01,.001),core.Workpoint('two','seconds',.02,.001)),
                'cpu',Clock(),False)
        self.assertEqual(warm_observed,[S,frozenset([1,2,3])])
        self.assertEqual(row['returned_gain_above_prefix'],4)
        self.assertEqual(row['trace'].admission_kinds,['native_search','native_search'])
        self.assertEqual(row['final_return_gain'],4)
        self.assertEqual(len(row['source_receipt'].spent_requests),2)

    def test_constructor_and_final_full_validation_cost_are_inside_outer_deadline(self):
        graph,S,action=fixture();clock=Clock()
        source=SimpleNamespace(selected=frozenset([2,3]),incumbent_value=20,
            controller_return_sample_seconds=.1,attempts=())
        def controller(*args,**kwargs):
            kwargs['admission_observer'](source.selected,4,clock())
            return source
        original=online._final_membership
        def slow_check(*args):clock.value+=1.;return original(*args)
        with patch.object(online,'run_coordinated_observed',side_effect=controller), \
                patch.object(online,'_final_membership',side_effect=slow_check):
            row=online.run_unit(case(),'Upper',None,.556,None,None,(),'cpu',clock,False)
        self.assertEqual(row['certified_gain'],4);self.assertEqual(row['final_return_gain'],0)
        self.assertTrue(row['outer_deadline_miss']);self.assertFalse(row['final_return_validated_by_D'])
        with patch.object(online,'make_policy',side_effect=lambda *args:setattr(clock,'value',clock.value+1.)):
            row=online.run_unit(case(),'Upper',None,.556,None,None,(),'cpu',clock,False)
        self.assertEqual(row['final_return_gain'],0);self.assertEqual(row['failure']['type'],'TimeoutError')

    def test_ordered_recovery_scope_array_keeps_solver_vertex_order(self):
        values,pointer=online._flatten_memberships(((3,1,2),(2,0)),preserve_order=True)
        self.assertEqual(tuple(values),(3,1,2,2,0));self.assertEqual(tuple(pointer),(0,3,5))
        self.assertEqual(tuple(online._flatten_memberships(((3,1,2),))[0]),(1,2,3))

    def test_middle_deadline_strict_classic_selection_complete_all_graphs_first_tie(self):
        rows=[dict(case_id='v%d'%i,method=m,deadline_seconds=.556,graph_weight=1/24,
            final_return_relative_gain=1. if m in ('Upper','P1') else 0.,
            certified_relative_gain=10. if m=='Lower' else 0.) for i in range(24) for m in online.CLASSICS]
        chosen=online.select_classical(rows)
        self.assertEqual(chosen['selected'],'Upper');self.assertGreater(chosen['source_certified_scores']['Lower'],0)
        with self.assertRaises(ValueError):online.select_classical(rows[:-1])
        with self.assertRaises(ValueError):online.select_classical(rows+[rows[0]])

    def test_capsule_origin_thirdparty_core_allowed_owned_escape_or_missing_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'capsule';root.mkdir();records=[];modules={}
            for name,relative in online.MODULE_PATHS.items():
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# finite fixture\n')
                records.append(dict(path=relative,sha256=online.digest(path)));modules[name]=SimpleNamespace(__file__=str(path))
            test=root/'tests/test_v4_residual_online_validation.py';test.parent.mkdir();test.write_text('# finite fixture\n')
            records.append(dict(path='tests/test_v4_residual_online_validation.py',sha256=online.digest(test)))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(files=records)))
            outside=Path(temporary)/'core.py';outside.write_text('# external fixture\n')
            modules['sympy.core']=SimpleNamespace(__file__=str(outside))
            with patch.dict(online.sys.modules,modules,clear=True):
                result=online.validate_source_closure(root,manifest,online.digest(manifest))
                self.assertEqual(len(result['execution_origins']),len(online.MODULE_PATHS))
                online.sys.modules['joint_recovery.escape']=SimpleNamespace(__file__=str(outside))
                with self.assertRaises(ValueError):online.validate_source_closure(root,manifest,online.digest(manifest))
                del online.sys.modules['joint_recovery.escape'];del online.sys.modules['experiments.v4_residual_fit']
                with self.assertRaises(ValueError):online.validate_source_closure(root,manifest,online.digest(manifest))

    def test_frozen_input_tamper_and_main_failure_preserved_without_retry(self):
        args=SimpleNamespace(protocol_sha256='a',completion_sha256='b',fit_protocol_sha256='c',fit_completion_sha256='d')
        protocol=dict(collector_protocol_sha256='a',collector_completion_sha256='b',fit_protocol_sha256='c',
            fit_completion_sha256='d',expected_units=1512,max_queries=8,deadlines=list(online.DEADLINES),
            caps=list(online.CAPS),variants=list(online.VARIANTS),fit_seeds=list(online.FIT_SEEDS),
            classics=list(online.CLASSICS),calibration_sha256=online.CALIBRATION_SHA256,
            executed_common_greedy_prefix=True,fixed_scope_cap=256,greedy_only_queries=0,lower_uses_already_paid_warm=True)
        online.check_frozen_inputs(args,protocol)
        with self.assertRaises(ValueError):online.check_frozen_inputs(args,dict(protocol,expected_units=1511))
        with self.assertRaises(ValueError):online.check_frozen_inputs(args,dict(protocol,collector_completion_sha256='x'))
        with tempfile.TemporaryDirectory() as temporary,patch.object(online,'_ACTIVE_OUT',Path(temporary)), \
                patch.object(online,'main',side_effect=ValueError('fixture integrity failure')):
            with self.assertRaises(ValueError):online.entrypoint()
            path=Path(temporary)/'failure.json';before=path.read_bytes()
            self.assertFalse(json.loads(before)['completed'])
            with self.assertRaises(ValueError):online.entrypoint()
            self.assertEqual(path.read_bytes(),before)

    def test_greedy_only_runs_identical_common_prefix_and_zero_native_calls(self):
        graph,S,action=fixture();seen=[]
        def forbidden(*args):seen.append(args);raise AssertionError('greedy-only must stop after paid prefix')
        with patch.object(online,'coordination_cells',return_value=cells(action)):
            row=online.run_unit(case(),'AnytimeGreedyPortfolio',None,1.,None,forbidden,
                (core.Workpoint('finite','seconds',.01,.001),),'cpu',Clock(),False)
        self.assertEqual(row['status'],'complete_unit');self.assertEqual(seen,[])
        self.assertEqual(row['source_receipt'].attempts,())
        self.assertEqual(row['common_prefix_certified_gain'],4)
        self.assertEqual(row['returned_prefix_component'],4)
        self.assertEqual(row['returned_gain_above_prefix'],0)
        self.assertEqual(row['trace'].admission_kinds,['common_prefix'])
        self.assertEqual(row['source_receipt'].action_identities,(((),(0,1)),))

    def test_lower_uses_current_actual_warm_without_sweeps_or_extra_candidate(self):
        graph,S,action=fixture();cache=online.NeighborhoodScopeCache(graph,S)
        scope=cache.scope(0,action,256);warm=cache.initial_known_warm(scope)
        view=core.RequestView(core.Request(scope,core.Workpoint('finite','seconds',.01,.001)),warm)
        with patch('experiments.v4_residual_common.executed_greedy_warm',side_effect=AssertionError('no repeat work')):
            result=online.make_policy('Lower',None,'cpu')(graph,(view,),core.ControllerState(20,4,1.,()))
        self.assertEqual(result.gains,(4,));self.assertEqual(result.feasible_candidates,())
        self.assertTrue(result.current_warm_start_conditioned)

    def test_actual_boundary_zeroes_returned_decomposition_without_losing_prefix(self):
        original=frozenset([0]);prefix=frozenset([1]);native=frozenset([2])
        log=online.DecisionLog(original,100.,101.)
        log.admission(prefix,3,100.3,'common_prefix');log.admission(native,5,100.7,'native_search')
        row=dict(trace=log,certified_gain=5,final_return_selected=native,final_return_gain=5,
            final_return_relative_gain=.5,final_return_validated_by_D=True,final_validation_ready_seconds=.9)
        online.add_prefix_decomposition(row)
        self.assertEqual(row['common_prefix_certified_gain'],3);self.assertEqual(row['returned_gain_above_prefix'],2)
        online.actual_receipt_boundary(row,101.)
        self.assertEqual(row['final_return_gain'],0);self.assertEqual(row['returned_prefix_component'],0)
        self.assertEqual(row['returned_gain_above_prefix'],0);self.assertEqual(row['certified_gain_above_prefix'],2)
        self.assertEqual(row['common_prefix_selected'],prefix)

    def test_python38_source_no_generation_retry_or_confirmation_entry(self):
        source=Path(online.__file__).read_text();ast.parse(source,feature_version=(3,8))
        self.assertNotIn('choices=(\'freeze\',\'run\',\'confirm\')',source)
        self.assertNotIn('generator(',source);self.assertNotIn('fit_one(',source)
        self.assertEqual(online.MAX_QUERIES,8)


if __name__=='__main__':unittest.main()
