"""Small contract guards, NOT research instances or experiment results."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import p0_recovery_probes as p0


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = p0.load_runtime(RUNTIME_ROOT)

    def test_original_decimal_reward_is_not_replaced_by_binary_endpoint_difference(self):
        starts = ["100000.123456789", "100004.123456789"]
        ends = ["100001.345678912", "100005.345678912"]
        weights = np.asarray([float(Decimal(e) - Decimal(s)) for s, e in zip(starts, ends)])
        binary = np.asarray(list(map(float, ends))) - np.asarray(list(map(float, starts)))
        self.assertFalse(np.array_equal(weights, binary))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numeric_guard.npz"
            np.savez(path, weights=weights, agents=np.asarray([0, 1]), edges=np.empty((0, 2), dtype=np.int64),
                     start_seconds=np.asarray(list(map(float, starts))), end_seconds=np.asarray(list(map(float, ends))),
                     start_decimal_seconds=np.asarray(starts), end_decimal_seconds=np.asarray(ends), graph_id="guard")
            graph, metadata = p0.load_graph(path, self.api)
        self.assertTrue(np.array_equal(graph.weights, weights))
        self.assertEqual(metadata["raw_endpoint_duration_identity"], "PASS_EXACT_ORIGINAL_DECIMAL_TO_FLOAT64")

    def fixture_scope(self):
        from experiments.v4_neighborhoods import CoordinationAction
        edges = ((0, 2), (1, 3), (2, 3), (2, 5), (3, 4))
        adjacency = [set() for _ in range(6)]
        for u, v in edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        graph = self.api["Graph"](np.asarray([4., 3., 100., 5., 2., 2.]),
                                  np.asarray([0, 1, 0, 1, 1, 0]), tuple(map(frozenset, adjacency)), "guard")
        original = frozenset((0, 1, 5))
        cache = self.api["Cache"](graph, original)
        scope = cache.scope(0, CoordinationAction((), (0, 1)), 256)
        warm = cache.initial_known_warm(scope)
        return graph, original, scope, warm

    def test_whole_fixed_base_excludes_locally_attractive_blocked_replacement(self):
        graph, original, scope, warm = self.fixture_scope()
        self.assertEqual(scope.base, frozenset((5,)))
        self.assertNotIn(2, scope.replacements)
        self.assertEqual(warm, frozenset((0, 3)))
        self.assertEqual(scope.immediate_gain + self.api["objective"](graph, warm),
                         self.api["objective"](graph, scope.base | warm) - self.api["objective"](graph, original))
        with self.assertRaises(ValueError):
            self.api["check"](graph, scope, (2,))
        bound = p0.scope_statistics(graph, scope, self.api)["U_seconds"]
        self.assertGreaterEqual(bound, self.api["objective"](graph, warm))

    def test_clone_does_not_warm_peers_and_spent_row_does_not_call_native(self):
        graph, original, scope, warm = self.fixture_scope()
        descriptor = p0.scope_statistics(graph, scope, self.api)
        descriptor["scope_sha256"] = "test-scope"
        state = dict(original=sorted(original), original_value_seconds=self.api["objective"](graph, original),
                     best_selected=sorted(scope.base | warm), best_value_seconds=self.api["objective"](graph, scope.base | warm),
                     best_gain_seconds=2., warm_by_action={"0": sorted(warm)}, spent_requests=[])
        before = json.loads(json.dumps(state))

        class GuardBackend:
            def __init__(self):
                self.calls = 0
            def run(self, graph, scope, warm, budget_ms):
                self.calls += 1
                return warm, dict(status="guard_mock_not_scientific_result", native_called=True,
                                  native_output_valid=True, full_return_seconds=0.)

        backend = GuardBackend()
        first = p0.request_row(graph, scope, descriptor, state, 200, 0, backend, self.api)
        second = p0.request_row(graph, scope, descriptor, state, 1000, 0, backend, self.api)
        self.assertEqual(state, before)
        self.assertEqual(first["actual_warm_members"], second["actual_warm_members"])
        state["spent_requests"] = [[0, 256, 200]]
        spent = p0.request_row(graph, scope, descriptor, state, 200, 0, backend, self.api)
        self.assertFalse(spent["available_for_allocation"])
        self.assertFalse(spent["actual_finite_supervision_valid"])
        self.assertEqual(backend.calls, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=str(Path(__file__).resolve().parents[4] / "第一篇"))
    args, unittest_args = parser.parse_known_args()
    RUNTIME_ROOT = args.runtime_root
    unittest.main(argv=[__file__] + unittest_args)
