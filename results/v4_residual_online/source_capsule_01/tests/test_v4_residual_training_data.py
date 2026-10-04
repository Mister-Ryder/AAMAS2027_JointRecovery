"""Finite warm/snapshot/split guards; no solver, training or fresh graph run."""
import ast
from dataclasses import replace
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np

from experiments import v4_residual_training_data as data
from joint_recovery.core import Graph
from joint_recovery.v4_budgeted_recovery import NativeIntegerGraph, RepairAttempt, Workpoint


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def spend(self, amount): self.now += amount


def fixture(native=False):
    weights = np.asarray([10, 11, 12, 13, 17, 18, 19, 20],
                         dtype=np.int64 if native else np.float64)
    adjacency = [set() for _ in weights]
    for u, v in ((0,4),(1,4),(1,5),(2,5),(2,6),(3,6),(0,7),(3,7)):
        adjacency[u].add(v); adjacency[v].add(u)
    cls = NativeIntegerGraph if native else Graph
    return cls(weights, np.arange(8)%4, tuple(map(frozenset, adjacency)), "label-guard"), frozenset(range(4))


def config(deadline=1.):
    return data.CollectionConfig(deadline,
        tuple(Workpoint("s%g"%s, "seconds", s, s+.001) for s in data.NATIVE_SLICES), "0"*64,
        require_chils_receipt=False)


class TrainingDataGuards(unittest.TestCase):
    def collect(self, backend=None, graph=None, selected=None, clock=None, **kwargs):
        graph0, selected0 = fixture(); clock = clock or Clock()
        graph = graph0 if graph is None else graph
        selected = selected0 if selected is None else selected
        backend = backend or (lambda g, r, w, warm, remaining: RepairAttempt(warm, "guard"))
        return data.collect_label_groups(graph, selected, seed=19,
            config=kwargs.pop("config", config()), backend=backend, clock=clock, **kwargs)

    def test_fresh_specs_complete_disjoint_and_no_generation(self):
        with patch.object(data, "construct_fresh_case", side_effect=AssertionError("generation")):
            manifest = data.specification_manifest()
        train, val = manifest["splits"]["training"], manifest["splits"]["validation"]
        self.assertEqual((len(train), len(val)), (72, 24))
        self.assertEqual(len({s["seed"] for s in train+val}), 96)
        self.assertEqual((train[0]["seed"], val[0]["seed"]), (20420000, 20430000))
        for specs in (train, val):
            for offset in range(0, len(specs), 24):
                cells = specs[offset:offset+24]
                self.assertEqual(sum(s["domain"]=="menu" for s in cells), 18)
                self.assertEqual(sum(s["domain"]=="resource" for s in cells), 6)
        self.assertTrue(manifest["no_graphs_constructed"])

    def test_confirmation_and_outside_specs_fail_closed(self):
        for name in ("confirmation", "confirmation_iid", "test", None):
            with self.assertRaises(ValueError): data.fresh_split_specs(name)
        changed = dict(data.fresh_split_specs("training")[0], seed=20440000)
        with self.assertRaises(ValueError): data.construct_fresh_case(changed)

    def test_original_membership_never_silently_coerces_or_deduplicates(self):
        for selected in ([0,0,1],[0,1,2,3.2],[True,1,2,3]):
            with self.subTest(selected=selected),self.assertRaises(ValueError):
                self.collect(selected=selected)

    def test_graph_domain_state_request_weighting_is_predeclared(self):
        for split in ("training", "validation"):
            specs = data.fresh_split_specs(split)
            for domain in ("menu", "resource"):
                self.assertAlmostEqual(sum(data.split_graph_weight(s) for s in specs if s["domain"]==domain), .5)
        result = self.collect(graph_weight=.25)
        for group in result["groups"]:
            self.assertEqual(group["state_loss_weight"], .5)
            valid = [r for r in group["alternatives"] if r["raw_supervision_valid"]]
            if valid:
                self.assertAlmostEqual(sum(r["regression_request_weight"] for r in valid), .125)

    def test_initial_snapshot_is_paid_and_admits_known_legal_gain(self):
        clock = Clock(); original = data.coordination_cells
        def paid(*args, **kwargs):
            clock.spend(.02); return original(*args, **kwargs)
        with patch.object(data, "coordination_cells", paid): result = self.collect(clock=clock)
        first = result["groups"][0]["state"]
        self.assertAlmostEqual(first["remaining_seconds"], .98)
        self.assertGreater(first["best_gain"], 0)
        self.assertEqual(len(result["coverage"]), 11)
        self.assertTrue(any(c["status"]=="actual_action_alias" for c in result["coverage"]))

    def test_exhausted_paid_preparation_keeps_no_startable_coverage(self):
        clock = Clock(); original = data.coordination_cells; calls=[]
        def slow(*args, **kwargs):
            clock.spend(1.1); return original(*args, **kwargs)
        with patch.object(data, "coordination_cells", slow):
            result = self.collect(clock=clock, backend=lambda *a:calls.append(a))
        self.assertEqual(calls, [])
        self.assertEqual(len(result["coverage"]), 11)
        for group in result["groups"]:
            self.assertEqual(group["state"]["remaining_seconds"], 0)
            self.assertTrue(all(r["status"]=="not_startable" for r in group["alternatives"]))

    def test_first_rr_history_only_real_outcome_and_peers_do_not_mutate(self):
        clock=Clock(); calls=[]
        def backend(graph, scope, point, warm, remaining):
            calls.append((scope.action_index, point.name, warm, remaining)); clock.spend(.01)
            actual=frozenset(v for v in scope.replacements if v>=4)
            return RepairAttempt(actual, "actual_mock")
        result=self.collect(clock=clock,backend=backend)
        initial, history = result["groups"]
        self.assertEqual(result["history_actual_calls"], 1)
        self.assertTrue(result["history"][0]["launched"])
        self.assertTrue(all(not r["launched"] for r in result["history"][1:]))
        self.assertEqual(calls[0][1], "s0.05")
        # The common paid prefix already materializes the better feasible set.
        self.assertEqual(set(initial["state"]["warm_by_action"][0]["recovered"]), set(range(4,8)))
        self.assertEqual(set(history["state"]["warm_by_action"][0]["recovered"]), set(range(4,8)))
        for group in (initial, history):
            self.assertEqual({r["state_sha256"] for r in group["alternatives"]}, {group["state_sha256"]})
            same_action=[r for r in group["alternatives"] if r["action_index"]==0 and r["available"]]
            self.assertEqual(len({tuple(r["actual_warm_start"]) for r in same_action}), 1)
        self.assertTrue(any(r["status"]=="already_spent_in_history" for r in history["alternatives"]))

    def test_late_valid_outputs_keep_signed_membership_without_warming(self):
        clock=Clock()
        def backend(graph, scope, point, warm, remaining):
            clock.spend(1.1)
            return RepairAttempt(frozenset(v for v in scope.replacements if v>=4), "late")
        result=self.collect(clock=clock,backend=backend)
        first=result["history"][0]
        self.assertTrue(first["spent"]); self.assertTrue(first["membership_valid"])
        self.assertFalse(first["on_time"]); self.assertEqual(first["admitted_gain"], 0)
        self.assertIsNotNone(first["signed_return_delta"])
        self.assertEqual(result["groups"][0]["state"]["warm_by_action"], result["groups"][1]["state"]["warm_by_action"])

    def test_failed_history_is_spent_and_never_teacher_warm(self):
        result=self.collect(backend=lambda *args:RepairAttempt(None,"no_output"))
        first=result["history"][0]
        self.assertTrue(first["spent"]); self.assertFalse(first["membership_valid"])
        self.assertEqual(first["admitted_gain"], 0)
        self.assertEqual(len(result["groups"][1]["state"]["spent_requests"]), 1)

    def test_invalid_memberships_and_bad_metadata_are_serializable_failures(self):
        for recovered, diagnostics in (([0,0],{}),([4.5],{}),([True],{}),(None,{})):
            with self.subTest(recovered=recovered):
                result=self.collect(backend=lambda *args:RepairAttempt(recovered,"bad",diagnostics))
                json.dumps(result,allow_nan=False)
                self.assertTrue(all(not r["on_time"] for g in result["groups"] for r in g["alternatives"] if r["launched"]))
        result=self.collect(backend=lambda g,r,w,s,t:RepairAttempt(tuple(np.int64(v) for v in s),"badmeta",{"bad":float("nan")}))
        json.dumps(result,allow_nan=False)
        self.assertTrue(all(r["diagnostics"]=={} for g in result["groups"] for r in g["alternatives"]))

    def test_native_integer_target_is_exact_and_identity_distinct(self):
        original, S=fixture(native=True)
        weights=np.asarray([2**53,1,1,1,2**53+1,2,2,2],np.int64)
        graph=NativeIntegerGraph(weights,original.agents,original.adjacency)
        def backend(g,r,w,s,t):return RepairAttempt(frozenset(v for v in r.replacements if v>=4))
        result=self.collect(graph=graph,selected=S,backend=backend)
        row=result["history"][0]
        self.assertEqual(row["signed_return_delta"], 4)
        self.assertIsInstance(row["signed_return_delta"], int)
        altered=NativeIntegerGraph(weights+1,original.agents,original.adjacency)
        self.assertNotEqual(data.graph_identity(graph),data.graph_identity(altered))

    def test_missing_postcall_timestamp_keeps_spent_failure_receipt(self):
        graph,S=fixture();cache=data.NeighborhoodScopeCache(graph,S)
        action=data.coordination_cells(graph,S,19)[0][0];scope=cache.scope(0,action,256)
        request=data.Request(scope,config().workpoints[1]);warm=cache.initial_known_warm(scope)
        state=data.LabelState("guard",S,S,46,0,((0,warm),),(),0,1)
        ticks=iter([0.,float("nan")])
        row=data._execute_clone(graph,request,state,lambda *a:RepairAttempt(warm),lambda:next(ticks),
                                require_chils_receipt=False)
        self.assertTrue(row["spent"]);self.assertFalse(row["on_time"])
        self.assertEqual(row["status"],"timestamp_failed")
        self.assertIsNotNone(row["returned_recovery"])

    def test_failed_declared_graph_is_not_replaced(self):
        spec=data.fresh_split_specs("training")[0]
        with patch.object(data,"construct_fresh_case",side_effect=ValueError("fixture failure")):
            result=data.collect_fresh_spec(spec,config=replace(config(),require_chils_receipt=True),
                                           backend=lambda *args:None,clock=Clock())
        self.assertTrue(result["included_in_coverage"]);self.assertEqual(result["spec"],spec)
        self.assertEqual(result["groups"],[])
        graph,S=fixture()
        with patch.object(data,"construct_fresh_case",return_value=(graph,S,{"fixture":True})), \
             patch.object(data,"collect_label_groups",side_effect=ValueError("label failure")):
            failed=data.collect_fresh_spec(spec,config=replace(config(),require_chils_receipt=True),
                                          backend=lambda *args:None,clock=Clock())
        self.assertEqual(failed["status"],"fresh_declared_graph_collection_failed")
        self.assertTrue(failed["included_in_coverage"])

    def test_strict_executor_receipt_is_required_before_admission(self):
        result=self.collect(config=replace(config(),require_chils_receipt=True))
        self.assertTrue(result["executor_receipt_required"])
        self.assertTrue(all(r["admitted_gain"]==0 for g in result["groups"] for r in g["alternatives"]))
        def backed(g,r,w,s,t):return RepairAttempt(s,"pinned",{"binary_sha256":data.CHILS_SHA256})
        valid=self.collect(config=replace(config(),require_chils_receipt=True),backend=backed)
        self.assertTrue(any(r["raw_supervision_valid"] for g in valid["groups"] for r in g["alternatives"]))
        spec=data.fresh_split_specs("training")[0]
        with self.assertRaises(ValueError):data.collect_fresh_spec(spec,config=config(),backend=backed)

    def test_full_returned_membership_validation_is_in_label_timer(self):
        clock=Clock();after=[False];original=data._check_recovery
        def checked(g,r,s):
            actual=original(g,r,s)
            if after[0]:clock.spend(1.1)
            return actual
        def backend(g,r,w,s,t):after[0]=True;return RepairAttempt(s,"guard")
        with patch.object(data,"_check_recovery",checked):result=self.collect(clock=clock,backend=backend)
        first=result["history"][0]
        self.assertGreaterEqual(first["elapsed_seconds"],1.1)
        self.assertFalse(first["on_time"])
        self.assertEqual(first["admitted_gain"],0)

    def test_source_graph_mutation_rejects_the_collection(self):
        graph,S=fixture();done=[False]
        def backend(g,r,w,s,t):
            if not done[0]:
                g.weights.setflags(write=True);g.weights[7]+=1;g.weights.setflags(write=False);done[0]=True
            return RepairAttempt(s,"guard")
        with self.assertRaises(ValueError):self.collect(graph=graph,selected=S,backend=backend)

    def test_protocol_costs_cutoff_and_python38_syntax_fail_closed(self):
        with self.assertRaises(ValueError): replace(config(),history_max_queries=2)
        with self.assertRaises(ValueError): replace(config(),calibration_sha256="")
        with self.assertRaises(ValueError): replace(config(),caps=(64,256))
        with self.assertRaises(ValueError): replace(config(),deadline_seconds=float("nan"))
        ast.parse(inspect.getsource(data),feature_version=(3,8))


if __name__=="__main__":unittest.main()
