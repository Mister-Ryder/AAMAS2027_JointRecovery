"""Queue lifecycle/native input guards; no graph generation or solver calls."""
import json
from dataclasses import replace
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from experiments import v4_residual_collect_training as queue
from experiments import v3_domains, v4_backends, v4_neighborhoods, v4_residual_common
from experiments import v4_residual_training_data as data
from experiments.v4_residual_training_data import fresh_split_specs,graph_identity
from joint_recovery.v4_budgeted_recovery import NativeIntegerGraph, RepairAttempt


class FreshQueueGuards(unittest.TestCase):
    def setUp(self):
        self.calibration=Path(__file__).resolve().parents[1]/"docs/research_v4/CALIBRATION_FROZEN.json"

    def test_calibration_receipt_is_exact_and_primary(self):
        value,config=queue.validated_calibration(self.calibration)
        self.assertEqual(config.deadline_seconds,.556)
        self.assertEqual(config.calibration_sha256,queue.CALIBRATION_SHA256)
        self.assertEqual([w.expected_seconds for w in config.workpoints],[.035,.075,.214])

    def test_changed_calibration_fails_before_any_work(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"c.json";value=json.loads(self.calibration.read_text())
            value["primary_deadline_seconds"]=2.221;p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"source bytes"):queue.validated_calibration(p)

    def test_native_observable_keeps_large_rewards_and_cliques(self):
        g=NativeIntegerGraph(np.array([2**53+1,7],dtype=np.int64),np.array([0,1]),
            (frozenset([1]),frozenset([0])),"guard")
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"a.npz";queue.save_observable(p,g,{0},[(0,1)])
            with np.load(p,allow_pickle=False) as a:
                self.assertEqual(a["weights"].dtype,np.int64)
                self.assertEqual(a["weights"][0].item(),2**53+1)
                self.assertEqual(a["cliques_ptr"].tolist(),[0,2])
                self.assertEqual(a["cliques_flat"].tolist(),[0,1])
                self.assertEqual(a["edges"].tolist(),[[0,1]])

    def test_failed_declared_slot_retains_spec_weight_and_no_retry(self):
        spec=fresh_split_specs("training")[0]
        _,config=queue.validated_calibration(self.calibration)
        with tempfile.TemporaryDirectory() as folder,patch.object(queue,"construct_fresh_case",side_effect=ValueError("failed")) as make:
            row=queue.collect_case(spec,"case_000",Path(folder),config,lambda *a:None)
            data=json.loads((Path(folder)/"cases/case_000/labels.json").read_text())
            self.assertEqual(make.call_count,1)
            self.assertTrue(data["included_in_coverage"])
            self.assertEqual(data["spec"],spec)
            self.assertEqual(data["graph_loss_weight"],.5/54)
            self.assertIsNone(row["graph_sha256"])
            self.assertEqual(set(row["files"]),{"spec.json","labels.json"})

    def test_absolute_entry_collects_scope_and_references_exact_raw_files(self):
        # A file entry has no package context and includes its own directory on
        # sys.path. Exercise that fallback, including its actual action/cache
        # dispatch; --help alone cannot detect duplicate Python class objects.
        script=Path(queue.__file__).resolve()
        with patch.object(sys,"path",[str(script.parent)]+sys.path):
            entry=runpy.run_path(str(script),run_name="__absolute_collector_guard__")
        self.assertIs(entry["_fresh_domains"],v3_domains)
        self.assertIs(entry["chils_backend"],v4_backends.chils_backend)
        self.assertIs(entry["CollectionConfig"],data.CollectionConfig)
        self.assertIs(entry["collect_label_groups"],data.collect_label_groups)
        globals0=entry["collect_label_groups"].__globals__
        self.assertIs(globals0["coordination_cells"],v4_neighborhoods.coordination_cells)
        self.assertIs(globals0["NeighborhoodScopeCache"],v4_residual_common.ExecutedWarmScopeCache)
        self.assertIs(globals0["coordination_cells"].__globals__["CoordinationAction"],
            v4_residual_common.ExecutedWarmScopeCache.scope.__globals__["CoordinationAction"])

        adjacency=[set() for _ in range(8)]
        for u,v in ((0,4),(1,4),(1,5),(2,5),(2,6),(3,6),(0,7),(3,7)):
            adjacency[u].add(v);adjacency[v].add(u)
        g=NativeIntegerGraph(np.array([10,11,12,13,17,18,19,20],dtype=np.int64),
            np.arange(8)%4,tuple(map(frozenset,adjacency)),"absolute-entry-finite-guard")
        selected=frozenset(range(4));spec=fresh_split_specs("validation")[0]
        _,config=entry["validated_calibration"](self.calibration)
        config=replace(config,require_chils_receipt=False)
        now=[0.];calls=[]
        def backend(graph,scope,point,warm,remaining):
            self.assertIs(graph,g);self.assertEqual(scope.cap,256)
            calls.append(scope);now[0]+=.001
            return RepairAttempt(warm,"finite_mock")
        def actual_collection(*args,**kwargs):
            return entry["collect_label_groups"](*args,clock=lambda:now[0],**kwargs)
        # Only substitute the graph constructor and native executor. The actual
        # common prefix, scopes, history, cloned labels and receipts execute.
        collect_case=entry["collect_case"]
        with tempfile.TemporaryDirectory() as folder,patch.dict(collect_case.__globals__,
                {"construct_fresh_case":lambda _: (g,selected,{}),
                 "collect_label_groups":actual_collection}):
            row=collect_case(spec,"case_000",Path(folder),config,backend)
            labels=json.loads((Path(folder)/"cases/case_000/labels.json").read_text())
            self.assertEqual(row["status"],"actual_finite_labels_not_learning_or_policy_evidence")
            self.assertEqual(row["graph_sha256"],graph_identity(g))
            self.assertEqual(labels["cells_expected"],11)
            self.assertEqual(len(labels["groups"]),2)
            self.assertTrue(labels["executed_common_greedy_prefix"])
            self.assertGreater(len(calls),0)
            self.assertGreater(row["actual_alternative_calls"],0)
            for name,receipt in row["files"].items():
                p=Path(folder)/"cases/case_000"/name
                self.assertEqual(receipt["sha256"],queue.sha(p))
                self.assertEqual(receipt["bytes"],p.stat().st_size)
            self.assertEqual(row["actual_history_calls"],1)

    def test_wrong_runtime_gate_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"completion.json";p.write_text(json.dumps({"status":"complete_native_builds_numeric_clearance_pending"}))
            with self.assertRaises(ValueError):queue.wait_for_runtime(p)
            p.write_text(json.dumps({"status":"complete_factorized_runtime_gate_not_quality"}))
            queue.wait_for_runtime(p)

    def test_failed_runtime_never_waits_forever(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"completion.json";(p.parent/"failure.json").write_text("{}")
            with self.assertRaises(ValueError):queue.wait_for_runtime(p)


if __name__=="__main__":unittest.main()
