"""Finite weighted-loss/replay guards; no real collection, fitting or solver."""
import ast
from dataclasses import replace
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from experiments import v4_residual_fit as fit
from experiments import v4_residual_training_data as data
from joint_recovery.core import Graph
from joint_recovery.v4_budgeted_recovery import RepairAttempt,Workpoint


class Clock:
    def __init__(self): self.now=0.
    def __call__(self): return self.now
    def spend(self,value): self.now+=value


def fixture(native=False,late=False,absolute_start=0.):
    weights=np.asarray([10,11,12,13,17,18,19,20],dtype=np.int64 if native else np.float64)
    if native: weights[4]=2**53+1
    adjacency=[set() for _ in weights]
    for u,v in ((0,4),(1,4),(1,5),(2,5),(2,6),(3,6),(0,7),(3,7)):
        adjacency[u].add(v);adjacency[v].add(u)
    cls=fit.NativeIntegerGraph if native else Graph
    graph=cls(weights,np.arange(8)%4,tuple(map(frozenset,adjacency)),"tiny-fit-guard")
    selected=frozenset(range(4));spec=data.fresh_split_specs("training")[0]
    config=data.CollectionConfig(.556,tuple(Workpoint("slice%g"%s,"seconds",s,c)
        for s,c in zip(data.NATIVE_SLICES,(.035,.075,.214))),fit.CALIBRATION_SHA256)
    clock=Clock();clock.now=absolute_start
    def backend(g,scope,point,warm,remaining):
        clock.spend(.6 if late else .001)
        return RepairAttempt(warm,"finite_mock_only",{"binary_sha256":data.CHILS_SHA256})
    labels=data.collect_label_groups(graph,selected,seed=spec["seed"]+1000,config=config,backend=backend,
        clock=clock,graph_weight=data.split_graph_weight(spec))
    labels["spec"]=spec
    return graph,selected,spec,config,labels


def case_fixture(**kwargs):
    graph,selected,spec,config,labels=fixture(**kwargs)
    states,weight=fit.replay_case(graph,selected,(),spec,labels,config)
    return fit.LoadedCase("guard",spec,graph,selected,(),states,weight,labels["status"],{}),config,labels


def loss_fixture(raw,target,gains=None,nodes=None,mask=None,valid=None):
    raw=torch.tensor(raw,dtype=torch.float32,requires_grad=True);n=len(raw)
    assignment=torch.tensor([] if nodes is None else nodes,dtype=torch.int64)
    details=dict(raw_gain=raw,logits=torch.zeros(len(assignment),requires_grad=True))
    targets=dict(raw_gain=torch.tensor(target,dtype=torch.float32),
        gains=torch.tensor(target if gains is None else gains,dtype=torch.float32),
        valid=torch.tensor([True]*n if valid is None else valid,dtype=torch.bool),
        recovery_mask=torch.tensor([0.]*len(assignment) if mask is None else mask))
    batch=dict(request_count=n,node_request=assignment,best_gain=torch.zeros(n))
    return details,targets,batch


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,allow_nan=False),encoding="utf-8")


def dataset_fixture(root):
    project=Path(fit.__file__).resolve().parents[1]
    paths=set(fit.FIT_SOURCE_PATHS)|{"experiments/v4_residual_common.py","experiments/v4_residual_collect_training.py","experiments/v4_backends.py",
        "experiments/v3_domains.py","experiments/v3_pilot.py","experiments/v3_solvers.py",
        "experiments/v3_published_baselines.py","src/joint_recovery/efficient_core.py","src/joint_recovery/generators.py"}
    sources=dict(files=[dict(path=p,sha256=fit.sha256_file(project/p)) for p in sorted(paths)])
    calibration=json.loads((project/"docs/research_v4/CALIBRATION_FROZEN.json").read_text())
    specs=list(data.fresh_split_specs("training"))+list(data.fresh_split_specs("validation"))
    protocol=dict(status="frozen_actual_fresh_label_queue",splits=["training","validation"],complete_specs=specs,
        source_capsule=sources,calibration=calibration,executed_common_greedy_prefix=True,fixed_scope_cap=256,calibration_sha256=fit.CALIBRATION_SHA256,
        state_weight=.5,backend_reset_each_alternative=True,peer_outcomes_never_update_peers=True,
        serial_single_native_thread=True,model_fits_started=0,confirmation_graphs_generated=0)
    write_json(root/"protocol.json",protocol);entries=[]
    for i,spec in enumerate(specs):
        case_id="case_%03d"%i;folder=root/"cases"/case_id
        labels=dict(status="fresh_declared_graph_collection_failed",spec=spec,graph_sha256=None,groups=[],
            graph_loss_weight=data.split_graph_weight(spec),included_in_coverage=True,
            incomplete_case_labels_not_certified=True,learning_advantage_established=False,
            confirmation_graphs_generated=0,failure=dict(type="GuardOnly",message="no graph was generated"))
        write_json(folder/"spec.json",spec);write_json(folder/"labels.json",labels)
        files={name:dict(sha256=fit.sha256_file(folder/name),bytes=(folder/name).stat().st_size)
            for name in ("spec.json","labels.json")}
        entries.append(dict(case_id=case_id,spec=spec,status=labels["status"],graph_sha256=None,files=files,
            actual_history_calls=0,actual_alternative_calls=0))
    complete=dict(status="complete_fresh_label_coverage_not_fitting",cases=entries,expected_cases=96,
        protocol_sha256=fit.sha256_file(root/"protocol.json"),model_fits_started=0,confirmation_graphs_generated=0,
        includes_failed_and_no_startable_cases=True)
    write_json(root/"completion.json",complete)
    return protocol,complete


class FitGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_no_request_count_bias_in_state_regression(self):
        one=fit.state_loss(*loss_fixture([0.],[1.]),auxiliary_weight=0,ranking_weight=0)
        many=fit.state_loss(*loss_fixture([0.]*100,[1.]*100),auxiliary_weight=0,ranking_weight=0)
        self.assertEqual(one["total"].item(),.5);self.assertEqual(one["total"].item(),many["total"].item())

    def test_domain_graph_half_state_weight_once(self):
        total=0.
        for spec in data.fresh_split_specs("training"):
            value=2. if spec["domain"]=="menu" else 6.
            state={k:torch.tensor(value) for k in ("total","regression","ranking","auxiliary")}
            total+=fit.weighted_graph_loss((state,state),data.split_graph_weight(spec))["total"].item()
        self.assertAlmostEqual(total,4.,places=6)

    def test_empty_half_state_not_reweighted(self):
        full={k:torch.tensor(2.) for k in ("total","regression","ranking","auxiliary")}
        zero={k:torch.tensor(0.) for k in full}
        self.assertEqual(fit.weighted_graph_loss((full,zero),.25)["total"].item(),.25)

    def test_auxiliary_mean_is_per_request_not_per_node(self):
        details,targets,batch=loss_fixture([0,0],[0,0],nodes=[0]+[1]*9)
        details["logits"]=torch.tensor([0.]+[2.]*9,requires_grad=True)
        result=fit.state_loss(details,targets,batch,ranking_weight=0)
        expected=(F.softplus(torch.tensor(0.))+F.softplus(torch.tensor(2.)))/2
        self.assertAlmostEqual(result["auxiliary"].item(),expected.item(),places=6)

    def test_pair_mean_ties_and_negative_prediction_gradient(self):
        details,targets,batch=loss_fixture([-1,-2,-3],[0,1,2])
        result=fit.state_loss(details,targets,batch,auxiliary_weight=0)
        expected=F.softplus(torch.tensor([1.,2.,1.])/.2).mean()
        self.assertEqual(result["unequal_pairs"],3)
        self.assertAlmostEqual(result["ranking"].item(),expected.item(),places=6)
        result["ranking"].backward();self.assertGreater(abs(details["raw_gain"].grad[-1].item()),0)
        tied=fit.state_loss(*loss_fixture([-2,-1],[0,0]),auxiliary_weight=0)
        self.assertEqual(tied["unequal_pairs"],0)

    def test_noaux_only_removes_bce_not_signed_gradient(self):
        details,targets,batch=loss_fixture([-2.],[1.]);details["logits"]=None
        result=fit.state_loss(details,targets,batch,auxiliary_weight=0)
        self.assertEqual(result["regression"].item(),2.5);result["total"].backward()
        self.assertEqual(details["raw_gain"].grad.item(),-1.)

    def test_late_valid_and_failed_ranking_remain_separate(self):
        details,targets,batch=loss_fixture([0,0],[2,0],gains=[0,0],valid=[True,False])
        result=fit.state_loss(details,targets,batch,auxiliary_weight=0)
        self.assertEqual(result["valid_raw"],1);self.assertEqual(result["regression"].item(),1.5)
        self.assertEqual(result["unequal_pairs"],0)

    def test_ranking_subtracts_actual_current_best(self):
        details,targets,batch=loss_fixture([0,1,2],[0,1,2]);batch["best_gain"][:]=2.
        self.assertEqual(fit.state_loss(details,targets,batch,auxiliary_weight=0)["unequal_pairs"],0)

    def test_offline_selection_is_cost_adjusted_and_conditional(self):
        result=fit.offline_state_metrics([4,3],[4,3],0,[.2,.01],10)
        self.assertEqual(result["selected_index"],1);self.assertEqual(result["native_conditional_regret"],1)
        self.assertAlmostEqual(result["relative_conditional_allocation_regret"],.1)
        self.assertEqual(fit.offline_state_metrics([],[],2,[],10)["native_conditional_regret"],0)
        self.assertIsNone(fit.offline_state_metrics([],[],2,[],10)["pair_accuracy"])

    def test_first_minimum_keeps_earliest_all40_epoch(self):
        values=[3.]*40;values[3]=values[17]=1.
        self.assertEqual(fit.first_minimum_epoch(values),4)
        with self.assertRaises(ValueError): fit.first_minimum_epoch([0,float("nan")])

    def test_replay_known_history_peers_and_native_integer(self):
        for native in (False,True):
            case,config,labels=case_fixture(native=native)
            self.assertEqual(len(case.states),2);self.assertEqual(labels["history_actual_calls"],1)
            self.assertEqual(case.states[0].state_sha256,fit.sha256_json(case.states[0].record))
            if native:self.assertEqual(case.graph.weights[4].item(),2**53+1)

    def test_replay_late_raw_targets_without_warm_admission(self):
        case,config,labels=case_fixture(late=True)
        self.assertEqual(labels["history_actual_calls"],1)
        self.assertTrue(any(row["raw_supervision_valid"] and not row["on_time"] for row in case.states[0].rows))
        self.assertEqual(case.states[0].record["warm_by_action"],case.states[1].record["warm_by_action"])

    def test_absolute_uptime_time_closure_is_bounded_without_row_admission_relaxation(self):
        # Use the real collector's absolute deadline arithmetic, not a manually
        # fabricated residual: at 2**26 seconds the two subtractions differ.
        graph,selected,spec,config,labels=fixture(absolute_start=float(2**26))
        state=labels["groups"][0]["state"]
        residual=abs(state["remaining_seconds"]-
            max(0.,config.deadline_seconds-state["prefix_elapsed_seconds"]))
        self.assertGreater(residual,1e-9)
        self.assertLessEqual(residual,fit.TIME_CLOSURE_TOLERANCE_SECONDS)
        states,_=fit.replay_case(graph,selected,(),spec,labels,config)
        self.assertEqual(len(states),2)
        changed=copy.deepcopy(state)
        changed["remaining_seconds"]+=2*fit.TIME_CLOSURE_TOLERANCE_SECONDS
        with self.assertRaisesRegex(ValueError,"Remaining time closure"):
            fit._state(changed,graph,selected,config)
        # A ready time exactly at the deadline remains late in the unchanged
        # execution/replay rule; the closure tolerance is not a grace period.
        changed=copy.deepcopy(labels)
        row=next(row for row in changed["groups"][0]["alternatives"] if row["launched"])
        row["elapsed_seconds"]=row["actual_remaining_before_call"]
        with self.assertRaisesRegex(ValueError,"Saved elapsed|Strict actual validation-ready deadline"):
            fit.replay_case(graph,selected,(),spec,changed,config)
        # The same absolute comparison is replayed without subtracting D at a
        # different epoch. At this large uptime, elapsed<D even at equality.
        start=float(2**26);ticks=[]
        def preparation_boundary(*args,**kwargs):
            ticks.append(1);return start if len(ticks)==1 else start+config.deadline_seconds
        with patch.object(data,'_clock',side_effect=preparation_boundary):
            graph,selected,spec,config,late=fixture(absolute_start=start)
        self.assertTrue(late['known_warm_observations'])
        self.assertTrue(all(r['validation_ready_elapsed']<config.deadline_seconds and not r['on_time']
            for r in late['known_warm_observations']))
        self.assertEqual(late['history_actual_calls'],0);self.assertEqual(late['alternative_actual_calls'],0)
        self.assertEqual(len(fit.replay_case(graph,selected,(),spec,late,config)[0]),2)
        changed=copy.deepcopy(late);changed['known_warm_observations'][0]['on_time']=True
        with self.assertRaisesRegex(ValueError,'Known warm is admitted after'):
            fit.replay_case(graph,selected,(),spec,changed,config)

    def test_tampered_scope_state_membership_weights_fail_closed(self):
        graph,selected,spec,config,labels=fixture()
        def scope(x):x["groups"][0]["alternatives"][0]["scope_nodes"][0]=.9
        def state(x):x["groups"][0]["state"]["remaining_seconds"]+=1
        def mask(x):x["groups"][0]["alternatives"][0]["returned_recovery"]=[0,0]
        def weight(x):x["groups"][0]["alternatives"][0]["regression_request_weight"]*=2
        for alter in (scope,state,mask,weight):
            changed=copy.deepcopy(labels);alter(changed)
            with self.subTest(alter=alter.__name__),self.assertRaises(ValueError):fit.replay_case(graph,selected,(),spec,changed,config)

    def test_history_cannot_borrow_independent_deadline(self):
        graph,selected,spec,config,labels=fixture()
        changed=copy.deepcopy(labels);state=changed["groups"][1]["state"]
        state["remaining_seconds"]=.556;state["prefix_elapsed_seconds"]=0.
        changed["groups"][1]["state_sha256"]=fit.sha256_json(state)
        with self.assertRaises(ValueError):fit.replay_case(graph,selected,(),spec,changed,config)

    def test_teacher_rows_do_not_change_encoder_inputs(self):
        case,config,labels=case_fixture();fit_config=fit.FitConfig("ResidualCapacity",17);model=fit.build_model(fit_config)
        state=case.states[0];changed_rows=tuple(dict(row,signed_return_delta=999,returned_recovery=[999]) for row in state.rows)
        changed=replace(state,rows=changed_rows)
        first=fit.pack_state(case,state,model,fit_config);second=fit.pack_state(case,changed,model,fit_config)
        for key,value in first.items():
            if torch.is_tensor(value):self.assertTrue(torch.equal(value,second[key]),key)
        for key,value in first["units"].items():
            if torch.is_tensor(value):self.assertTrue(torch.equal(value,second["units"][key]),key)

    def test_all_variants_mask_warm_and_loss_interfaces(self):
        case,_,_=case_fixture()
        for variant in fit.VARIANTS:
            config=fit.FitConfig(variant,17);torch.manual_seed(17);model=fit.build_model(config)
            loss=fit.graph_loss(case,model,config);self.assertTrue(torch.isfinite(loss["total"]),variant)
            if variant=="ResidualNoWarmMembership":
                batch=fit.pack_state(case,case.states[0],model,config)
                self.assertTrue(torch.equal(batch["units"]["x"][:,3],torch.zeros_like(batch["units"]["x"][:,3])))
                self.assertTrue(torch.equal(batch["context"][:,-1:],torch.zeros_like(batch["context"][:,-1:])))
                self.assertTrue(torch.equal(batch["context"][:,-2],batch["lower"]))
            if variant in ("ResidualNoAux","ResidualCheapSummary"):self.assertEqual(config.effective_auxiliary_weight,0)

    def test_checkpoint_exact_metadata_strictload_and_finite_parameters(self):
        config=fit.FitConfig("ResidualCapacity",17);model=fit.build_model(config);bindings={"source":"guard"}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"own.pt";record=fit.checkpoint_record(model,config,4,.1,bindings);torch.save(record,path)
            loaded,result=fit.load_bound_checkpoint(path,fit.sha256_file(path),config,bindings)
            self.assertEqual(result["epoch"],4)
            for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,loaded.state_dict()[key]))
            with self.assertRaises(ValueError):fit.load_bound_checkpoint(path,"0"*64,config,bindings)
            with self.assertRaises(ValueError):fit.load_bound_checkpoint(path,fit.sha256_file(path),config,{"source":"wrong"})
            record["model_state"][next(iter(record["model_state"]))].flatten()[0]=float("nan");torch.save(record,path)
            with self.assertRaises(ValueError):fit.load_bound_checkpoint(path,fit.sha256_file(path),config,bindings)

    def test_npz_exact_schema_native_dtype_and_memberships(self):
        graph,selected,_,_,_=fixture(native=True)
        edges=np.asarray(sorted((u,v) for u in range(graph.n) for v in graph.adjacency[u] if u<v),dtype=np.int64)
        arrays=dict(weights=graph.weights,agents=graph.agents,edges=edges,selected=np.array(sorted(selected),np.int64),
            cliques_flat=np.array([],np.int64),cliques_ptr=np.array([0],np.int64))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"guard.npz";np.savez(path,**arrays)
            actual,_,_=fit.graph_from_npz(path);self.assertEqual(actual.weights.dtype,graph.weights.dtype)
            self.assertEqual(data.graph_identity(actual),data.graph_identity(graph))
            for values in (np.array([0.,1.]),np.array([0,0])):
                changed=dict(arrays,selected=values);np.savez(path,**changed)
                with self.assertRaises(ValueError):fit.graph_from_npz(path)
            changed=dict(arrays);changed["edges"]=np.vstack((edges,edges[0]));np.savez(path,**changed)
            with self.assertRaises(ValueError):fit.graph_from_npz(path)

    def test_complete_failed_registry_retains_all_fixed_graph_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset_fixture(root)
            with patch.object(data,"construct_fresh_case",side_effect=AssertionError("fresh generation")):
                kwargs=dict(expected_protocol_sha256=fit.sha256_file(root/"protocol.json"),
                    expected_completion_sha256=fit.sha256_file(root/"completion.json"))
                train=fit.load_dataset(root,"training",**kwargs);val=fit.load_dataset(root,"validation",**kwargs)
            self.assertEqual((len(train.cases),len(val.cases)),(72,24))
            self.assertAlmostEqual(sum(c.graph_weight for c in train.cases),1.)
            self.assertTrue(all(not c.states for c in train.cases+val.cases))

    def test_registry_missing_binding_and_bytes_do_not_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,complete=dataset_fixture(root)
            kwargs=dict(expected_protocol_sha256=fit.sha256_file(root/"protocol.json"),
                expected_completion_sha256=fit.sha256_file(root/"completion.json"))
            (root/"cases/case_000/labels.json").write_text("{}")
            with self.assertRaises(ValueError):fit.load_dataset(root,"training",**kwargs)
            complete["cases"].pop();write_json(root/"completion2.json",complete)
            (root/"completion.json").write_bytes((root/"completion2.json").read_bytes())
            kwargs["expected_completion_sha256"]=fit.sha256_file(root/"completion.json")
            with self.assertRaises(ValueError):fit.load_dataset(root,"training",**kwargs)

    def test_artifact_paths_source_duplicate_and_json_nan_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(ValueError):fit._bound_file(root,"../escape",{"bytes":0,"sha256":"0"*64})
            path=root/"bad.json";path.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):fit._read_json(path)
            path.write_text('{"a":NaN}')
            with self.assertRaises(ValueError):fit._read_json(path)
            path.write_text(json.dumps(dict(files=[dict(path="experiments/v4_residual_fit.py",sha256=fit.sha256_file(fit.__file__))]*2)))
            with self.assertRaises(ValueError):fit.validate_fit_sources(path,fit.sha256_file(path))

    def test_fit_settings_and_confirmation_are_not_silently_changed(self):
        for kwargs in (dict(epochs=1),dict(batch_graphs=8),dict(learning_rate=.01),dict(auxiliary_weight=0)):
            with self.assertRaises(ValueError):fit.FitConfig("ResidualCapacity",17,**kwargs)
        with self.assertRaises(ValueError):fit.FitConfig("ResidualCapacity",18)
        with self.assertRaises(ValueError):fit.load_dataset("unused","confirmation",expected_protocol_sha256="0"*64,expected_completion_sha256="0"*64)
        ast.parse(Path(fit.__file__).read_text(),feature_version=(3,8))

    def test_absolute_script_entry_help_never_collects_or_fits(self):
        project=Path(fit.__file__).resolve().parents[1]
        environment=dict(os.environ,PYTHONPATH=os.pathsep.join((str(project),str(project/"src"))))
        result=subprocess.run([sys.executable,str(Path(fit.__file__).resolve()),"--help"],
            capture_output=True,text=True,timeout=30,env=environment)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("--protocol-sha256",result.stdout)

    def test_epoch_progress_replaces_latest_and_never_certifies_completion(self):
        config=fit.FitConfig("ResidualCapacity",17);loss={key:1. for key in ("total","regression","ranking","auxiliary")}
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()) as output:
            fit.epoch_progress(temp,config,1,.3,loss,1.)
            fit.epoch_progress(temp,config,40,.2,loss,2.)
            record=json.loads((Path(temp)/"progress.json").read_text())
            self.assertEqual(record["epoch"],40);self.assertIs(record["is_final"],False)
            self.assertEqual(record["status"],"fit_epoch_update_nonfinal")
            self.assertFalse((Path(temp)/"completion.json").exists())
            self.assertFalse((Path(temp)/"progress.json.tmp").exists())
            self.assertEqual([json.loads(line)["epoch"] for line in output.getvalue().splitlines()],[1,40])
            with self.assertRaises(ValueError):fit.epoch_progress(temp,config,40,float("nan"),loss,2.)
            self.assertEqual(json.loads((Path(temp)/"progress.json").read_text()),record)


if __name__=="__main__":unittest.main()
