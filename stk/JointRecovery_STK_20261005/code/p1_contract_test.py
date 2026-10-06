"""One small P1 input/unit/allocation guard suite; no fitting/native calls."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import p0_recovery_probes as p0
import p1_fit_and_allocate as p1


class AllocationTests(unittest.TestCase):
    def test_fixed_calls_do_not_weaken_greedy_by_refusing_zero_predicted_gain(self):
        example = SimpleNamespace(key="guard", graph_id="guard", source=4,
            context=SimpleNamespace(best_gain=10.), rows=(
                dict(action_index=0, workpoint_ms=200, signed_gain_seconds=12.),
                dict(action_index=1, workpoint_ms=200, signed_gain_seconds=11.),
                dict(action_index=2, workpoint_ms=1000, signed_gain_seconds=16.),
                dict(action_index=3, workpoint_ms=1000, signed_gain_seconds=10.)))
        rows = p1.fixed_call_outcomes(example, [0., 0., 0., 0.], "Greedy")
        rows = [row for row in rows if row["request_menu"] == "pooled_workpoints"]
        self.assertEqual([r["actually_selected_unique_requests"] for r in rows], [1, 2, 4])
        self.assertEqual([r["beyond_snapshot_gain_seconds"] for r in rows], [2., 2., 6.])
        self.assertTrue(all("not_online" in r["scope"] for r in rows))

    def test_physical_source_policy(self):
        self.assertEqual(p1.source_number("JR-DUAL-r004"), 4)
        self.assertEqual(p1.source_number("JR-DUAL-r011-R12-g0170"), 11)
        self.assertFalse(set(p1.FIT_SOURCES) & set(p1.DEVELOPMENT_SOURCES))
        self.assertEqual(p1.FIT_SEEDS, (17, 29))


class ModelUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Run this one model guard on the already provisioned cloud PyTorch runtime")
        cls.api = p0.load_runtime(RUNTIME_ROOT)
        cls.model_api = p1.load_model_api(RUNTIME_ROOT)

    def fixture(self):
        api, model_api = self.api, self.model_api
        adjacency = (frozenset((2,)), frozenset((3,)), frozenset((0, 3, 5)),
                     frozenset((1, 2, 4)), frozenset((3,)), frozenset((2,)))
        graph = api["Graph"](np.asarray([4., 3., 100., 5., 2., 2.]),
                             np.asarray([0, 1, 0, 1, 1, 0]), adjacency, "contract-guard")
        scope = api["Cache"](graph, frozenset((0, 1, 5))).scope(0, model_api["Action"]((), (0, 1)), 256)
        warm = frozenset((0, 1))
        views = tuple(model_api["RequestView"](model_api["Request"](scope,
            model_api["Workpoint"]("native-%dms" % b, "seconds", b / 1000., b / 1000.)), warm) for b in (200, 1000))
        rows = tuple(dict(signed_gain_seconds=1., mean_net_marginal_seconds=1.,
                          occupancy_mean=[1., .5, .5]) for _ in views)
        return p1.Example("guard", 0, "guard", graph, {}, {0: scope}, views, rows,
                          model_api["ControllerState"](9., 0., 0., ()), {})

    def test_two_model_packers_keep_q_warm_upper_target_in_same_frozen_units(self):
        torch = self.model_api["torch"]
        example = self.fixture()
        for variant in p1.VARIANTS:
            batch = p1.observable_batch(example, variant, 10., "cpu", self.model_api)
            target = p1.targets(example, 10., "cpu", self.model_api)
            self.assertTrue(torch.allclose(batch["immediate"], torch.full((2,), -.7)))
            self.assertTrue(torch.allclose(batch["lower"], torch.full((2,), .7)))
            self.assertTrue(torch.allclose(batch["upper"], torch.full((2,), .9)))
            self.assertTrue(torch.allclose(target["raw"], torch.full((2,), .1)))
            self.assertTrue(torch.equal(batch["context"][:, 4], torch.zeros(2)))
            model = self.model_api["build"](variant)
            with torch.no_grad():
                details = model(batch, return_details=True)
                self.assertTrue(bool((details["recovery"] >= batch["lower"]).all()))
                self.assertTrue(bool((details["recovery"] <= batch["upper"]).all()))
            changed = copy.copy(example)
            changed.rows = tuple(dict(row, signed_gain_seconds=1e12) for row in example.rows)
            again = p1.observable_batch(changed, variant, 10., "cpu", self.model_api)
            self.assertTrue(torch.equal(batch["context"], again["context"]))
            self.assertTrue(torch.equal(batch["summary"], again["summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=str(Path(__file__).resolve().parents[4] / "第一篇"))
    args, test_args = parser.parse_known_args()
    RUNTIME_ROOT = args.runtime_root
    unittest.main(argv=[__file__] + test_args)
