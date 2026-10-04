"""Tiny finite interface/gradient guards; no training corpus or native solver."""
import ast
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from joint_recovery.generators import graph_from_edges
from joint_recovery.v4_budgeted_recovery import RecoveryScope, Request, RequestView, Workpoint, ControllerState, objective
from joint_recovery import v4_residual_model as residual
from joint_recovery import v4_factorized_model as factored
from joint_recovery import v4_model as original
from joint_recovery.v4_model_fast import FastStaticPackingCache


def fixture():
    graph = graph_from_edges([10, 11, 12, 13, 17, 18, 19, 20, 7], range(9),
        [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (0, 7), (3, 7), (4, 5), (5, 6), (4, 6)])
    scopes = (RecoveryScope(0, (), 256, frozenset(), frozenset(range(4)), tuple(range(8)), 8, -46., ((4, 5, 6),)),
              RecoveryScope(1, (8,), 256, frozenset({8}), frozenset(range(4)), tuple(range(8)), 8, -39., ((4, 5, 6),)))
    views = tuple(RequestView(Request(scope, Workpoint(str(seconds), "seconds", seconds, seconds + .03)), frozenset(range(4)))
                  for scope in scopes for seconds in (.01, .05, .2))
    return graph, views, ControllerState(46., 0., 1., ())


def batch_for(model, graph, views, state, **kwargs):
    packer = residual.pack_residual_summary if model.summary_only else residual.pack_residual_v4
    return packer(graph, views, state, use_warm_membership=model.use_warm_membership, **kwargs)


class ResidualModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_warm_floor_and_observable_upper_match_all_controls(self):
        graph, views, state = fixture()
        lower = objective(graph, views[0].warm_start) / original.normalization_scale(graph)
        expected = None
        for variant in residual.VARIANTS:
            model = residual.build_residual_model(variant)
            batch = batch_for(model, graph, views, state)
            self.assertEqual(batch["request_keys"], tuple(view.request.key for view in views))
            self.assertTrue(torch.equal(batch["lower"], torch.full((6,), lower)))
            self.assertTrue(bool((batch["upper"] >= batch["lower"]).all()))
            self.assertTrue(torch.equal(batch["upper"], batch["summary"][:, residual.UPPER_SUMMARY_INDEX]))
            if expected is None:
                expected = (batch["lower"], batch["upper"])
            self.assertTrue(torch.equal(batch["lower"], expected[0]))
            self.assertTrue(torch.equal(batch["upper"], expected[1]))
            details = model(batch, True)
            self.assertTrue(bool((details["recovery"] >= batch["lower"]).all()))
            self.assertTrue(bool((details["recovery"] <= batch["upper"]).all()))
            self.assertEqual(float((model.head if model.summary_only else model.residual_head)[2].bias.detach()), -4.)

    def test_known_floor_is_not_positive_gain_and_q_enters_only_offset(self):
        graph, views, state = fixture()
        model = residual.ResidualRecoveryValueNet()
        batch = residual.pack_residual_v4(graph, views, state)
        self.assertEqual(batch["unit_count"], 1)
        self.assertNotIn("context", batch["units"])
        self.assertNotIn("immediate", batch["units"])
        details = model(batch, True)
        torch.testing.assert_close(details["recovery"][:3], details["recovery"][3:], rtol=0, atol=0)
        torch.testing.assert_close(details["raw_gain"][3:] - details["raw_gain"][:3],
            batch["immediate"][3:] - batch["immediate"][:3], rtol=0, atol=1e-6)
        altered = dict(batch); altered["immediate"] = torch.full_like(batch["immediate"], -1e6)
        negative = model(altered, True)
        self.assertTrue(bool((negative["raw_gain"] < 0).all()))
        self.assertTrue(torch.equal(negative["scores"], torch.zeros(6)))
        torch.testing.assert_close(negative["recovery"], details["recovery"], rtol=0, atol=0)

    def test_warm_membership_ablation_keeps_value_floor_but_masks_ids_and_fraction(self):
        graph = graph_from_edges([10, 5, 5, 6], range(4), [])
        scope = RecoveryScope(0, (), 256, frozenset(), frozenset({0}), tuple(range(4)), 4, -10.)
        request = Request(scope, Workpoint("finite", "seconds", .05, .075))
        views = (RequestView(request, frozenset({0})), RequestView(request, frozenset({1, 2})))
        state = ControllerState(10., 0., 1., ())
        model = residual.build_residual_model("ResidualNoWarmMembership")
        batch = batch_for(model, graph, views, state)
        self.assertTrue(torch.equal(batch["units"]["x"][:, 3], torch.zeros(8)))
        self.assertTrue(torch.equal(batch["units"]["warm"], torch.zeros(8)))
        self.assertTrue(torch.equal(batch["context"][:, residual.WARM_FRACTION_INDEX], torch.zeros(2)))
        self.assertTrue(torch.equal(batch["context"][:, residual.WARM_VALUE_INDEX], batch["lower"]))
        self.assertGreater(float(batch["lower"][0]), 0.)
        details = model(batch, True)
        torch.testing.assert_close(details["recovery"][0], details["recovery"][1], rtol=0, atol=1e-6)
        # The class also masks IDs/fraction defensively when given a full pack.
        full = residual.pack_residual_v4(graph, views, state)
        torch.testing.assert_close(model(full, True)["recovery"], details["recovery"], rtol=0, atol=1e-6)

    def test_capacity_and_free_occupancy_have_identical_parameters_and_distinct_projection(self):
        torch.manual_seed(17); capacity = residual.build_residual_model("ResidualCapacity")
        torch.manual_seed(17); free = residual.build_residual_model("ResidualFreeOccupancy")
        self.assertEqual(capacity.parameter_counts(), free.parameter_counts())
        self.assertEqual(capacity.parameter_counts()["stored"], capacity.parameter_counts()["active_scoring"])
        for name, tensor in capacity.state_dict().items():
            self.assertTrue(torch.equal(tensor, free.state_dict()[name]))
        graph, views, state = fixture(); batch = residual.pack_residual_v4(graph, views, state)
        with torch.no_grad():
            for model in (capacity, free):
                for parameter in model.parameters():
                    parameter.zero_()
                model.residual_head[0].weight[0, -1] = 1.
                model.residual_head[2].weight[0, 0] = 1.
                model.residual_head[2].bias[0] = -4.
        cap = capacity(batch, True); unconstrained = free(batch, True)
        self.assertTrue(torch.equal(cap["logits"], unconstrained["logits"]))
        self.assertTrue(bool((original.factor_loads(cap["occupancy"], batch) <= 1 + 1e-6).all()))
        self.assertTrue(bool((original.factor_loads(unconstrained["occupancy"], batch) > 1).any()))
        self.assertTrue(bool((unconstrained["recovery"] > cap["recovery"]).all()))
        self.assertTrue(bool((unconstrained["recovery"] <= batch["upper"]).all()))

    def test_non_auxiliary_value_gradients_remain_when_occupancy_is_below_warm(self):
        graph, views, state = fixture(); batch = residual.pack_residual_v4(graph, views, state)
        torch.manual_seed(17); model = residual.build_residual_model("ResidualNoAux")
        # Force a very low occupancy while keeping the residual scalar path.
        with torch.no_grad():
            model.occupancy_head[2].bias.fill_(-8.)
        details = model(batch, True)
        self.assertTrue(bool((details["capacity_recovery"] < batch["lower"]).all()))
        target = batch["lower"] + .5 * (batch["upper"] - batch["lower"])
        loss = torch.nn.functional.smooth_l1_loss(details["recovery"], target)
        loss.backward()
        for parameter in (model.residual_head[2].weight, model.embed.weight):
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_binary_mask_auxiliary_and_invalid_rows_remain_compatible(self):
        graph, views, state = fixture(); batch = residual.pack_residual_v4(graph, views, state)
        outcomes = [frozenset({4, 7}), None, frozenset(range(4)), frozenset({4, 7}), None, frozenset(range(4))]
        targets = residual.execution_targets(graph, views, outcomes, on_time=[True, False, False, True, False, True])
        model = residual.build_residual_model("ResidualCapacity")
        loss = residual.recovery_value_loss(model(batch, True), targets, batch, [6], auxiliary_weight=.05)
        loss["total"].backward()
        self.assertTrue(bool(torch.isfinite(loss["total"])))
        self.assertEqual(targets["gains"].numel(), 6)
        self.assertGreater(float(model.occupancy_head[2].weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.equal(original.capacity_project_sparse(targets["recovery_mask"], batch), targets["recovery_mask"]))

    def test_invalid_actual_warm_and_fake_clique_are_rejected(self):
        graph, views, state = fixture()
        for warm in (frozenset({0, 4}), frozenset({8})):
            bad = RequestView(views[0].request, warm)
            with self.assertRaises(ValueError):
                residual.pack_residual_v4(graph, (bad,), state)
        bad_scope = replace(views[0].request.scope, resource_cliques=((0, 3),))
        bad = RequestView(Request(bad_scope, views[0].request.workpoint), views[0].warm_start)
        with self.assertRaises(ValueError):
            residual.pack_residual_v4(graph, (bad,), state)

    def test_material_bound_violation_rejected_instead_of_clipped(self):
        graph, views, state = fixture()
        class BadUpperCache(FastStaticPackingCache):
            def get(self, graph, scope, scale):
                record = super().get(graph, scope, scale)
                stats = list(record[2]); stats[residual.UPPER_SUMMARY_INDEX] = 0.
                return tuple(record[:2]) + (tuple(stats),) + tuple(record[3:])
        with self.assertRaises(ValueError):
            residual.pack_residual_v4(graph, views, state, static_cache=BadUpperCache())
        valid = residual.pack_residual_v4(graph, views, state)
        malformed = dict(valid); malformed["upper"] = valid["lower"] - 1.
        with self.assertRaises(ValueError):
            residual.ResidualRecoveryValueNet()(malformed)

    def test_empty_and_zero_width_scopes_are_exact_known_values(self):
        graph, views, state = fixture()
        empty_scope = replace(views[0].request.scope, replacements=(), resource_cliques=())
        empty = RequestView(Request(empty_scope, views[0].request.workpoint), frozenset())
        singleton_scope = replace(empty_scope, replacements=(0,))
        singleton = RequestView(Request(singleton_scope, views[0].request.workpoint), frozenset({0}))
        for variant in residual.VARIANTS:
            model = residual.build_residual_model(variant)
            batch = batch_for(model, graph, (empty, singleton), state)
            details = model(batch, True)
            self.assertTrue(torch.equal(batch["lower"], batch["upper"]))
            self.assertTrue(torch.equal(details["recovery"], batch["lower"]))
            self.assertTrue(torch.equal(details["raw_gain"], batch["immediate"] + batch["lower"]))

    def test_closed_variants_and_fixed_scope_reject_old_grid(self):
        self.assertEqual(len(residual.VARIANTS), 5)
        with self.assertRaises(ValueError):
            residual.build_residual_model("another_trial")
        graph, views, state = fixture()
        for cap in (64, 1024):
            scope = replace(views[0].request.scope, cap=cap)
            bad = RequestView(Request(scope, views[0].request.workpoint), views[0].warm_start)
            with self.assertRaises(ValueError):
                residual.pack_residual_v4(graph, (bad,), state)
        with self.assertRaises(ValueError):
            residual.pack_residual_summary(graph, (), state)

    def test_inference_cache_preserves_changed_warm_and_model_version_boundaries(self):
        graph, views, state = fixture(); model = residual.ResidualRecoveryValueNet().eval()
        cache = factored.DecisionEmbeddingCache(); cache.bind(graph, model); static = FastStaticPackingCache()
        with torch.no_grad():
            first = residual.pack_residual_v4(graph, views, state, static_cache=static, embedding_cache=cache)
            self.assertEqual(model(first, True, cache)["encoded_units"], 1)
            second = residual.pack_residual_v4(graph, views, state, static_cache=static, embedding_cache=cache)
            self.assertIsNone(second["units"])
            self.assertEqual(model(second, True, cache)["encoded_units"], 0)
            changed = RequestView(views[0].request, frozenset({4, 7}))
            third = residual.pack_residual_v4(graph, views[:1] + (changed,), state, static_cache=static, embedding_cache=cache)
            self.assertEqual(model(third, True, cache)["encoded_units"], 1)
            model.residual_head[2].bias.add_(.1)
            with self.assertRaises(ValueError):
                model(third, True, cache)
            cache.bind(graph, model)
            self.assertEqual(cache.records, {})
        uncached = residual.pack_residual_v4(graph, views, state, embedding_cache=cache)
        with self.assertRaises(ValueError):
            model(uncached, True, cache)

    def test_paid_callback_and_summary_have_no_free_graph_preparation(self):
        graph, views, state = fixture(); model = residual.ResidualRecoveryValueNet().eval()
        callback = residual.priority_callback_residual(model)
        with patch.object(model, "encode_units", wraps=model.encode_units) as encode:
            first = callback(graph, views, state)
            callback(graph, views[1:], ControllerState(46., 0., .5, (views[0].request.key,)))
            self.assertEqual(encode.call_count, 1)
            callback(graph, views, state)
            self.assertEqual(encode.call_count, 2)
        self.assertEqual(len(first.gains), 6)
        self.assertTrue(all(isinstance(value, float) for value in first.gains))
        summary = residual.ResidualSummaryValueNet()
        with patch.object(original, "pack_v4", side_effect=AssertionError("unused graph tensor preparation")):
            self.assertEqual(len(residual.priority_callback_residual(summary)(graph, views, state).gains), 6)

    def test_offline_merge_preserves_bound_scales_and_request_pointer_offsets(self):
        graph, views, state = fixture()
        for model in (residual.ResidualRecoveryValueNet(), residual.ResidualSummaryValueNet()):
            batches = [batch_for(model, graph, views, state, scale=scale) for scale in (1., 100.)]
            merged = residual.merge_residual_batches(batches)
            expected = torch.cat([model(batch, True)["raw_gain"] for batch in batches])
            # Separate and merged float32 shapes may use different accumulation
            # kernels, including Torch1.11. Numerical equivalence is distinct
            # from exact scope offsets, warm identity and feasibility checks.
            torch.testing.assert_close(model(merged, True)["raw_gain"], expected, rtol=1e-5, atol=1e-5)
            self.assertEqual(merged["request_count"], 12)
            self.assertEqual(len(merged["ptr"]), 13)
            self.assertTrue(torch.equal(merged["lower"], torch.cat([batch["lower"] for batch in batches])))
            self.assertEqual(merged["request_keys"], tuple(view.request.key for view in views) * 2)

    def test_python38_syntax_and_no_native_solver_or_teacher_import(self):
        source = Path(residual.__file__).read_text()
        tree = ast.parse(source, feature_version=(3, 8))
        imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any("fit" in name or "training" in name or "backend" in name or "solver" in name for name in imported))
        self.assertNotIn("scatter_reduce", source)
        self.assertNotIn("torch_scatter", source)


if __name__ == "__main__":
    unittest.main()
