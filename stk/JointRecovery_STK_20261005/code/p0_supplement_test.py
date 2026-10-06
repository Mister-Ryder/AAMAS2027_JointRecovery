"""One frozen replay/mutation guard using actual core semantics; no native."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import unittest

import numpy as np

import p0_recovery_probes as p0
import p0_supplement as supplement


class FrozenReplayTest(unittest.TestCase):
    def test_rebuild_saved_members_without_rerunning_initial_actions_or_warm(self):
        api = p0.load_runtime(RUNTIME_ROOT)
        from experiments.v4_neighborhoods import CoordinationAction
        adjacency = (frozenset((2,)), frozenset((3,)), frozenset((0, 3, 5)),
                     frozenset((1, 2, 4)), frozenset((3,)), frozenset((2,)))
        graph = api["Graph"](np.asarray([4., 3., 100., 5., 2., 2.]),
                             np.asarray([0, 1, 0, 1, 1, 0]), adjacency, "supplement-contract-guard")
        original = frozenset((0, 1, 5))
        cache = api["Cache"](graph, original)
        scope = cache.scope(0, CoordinationAction((), (0, 1)), 256)
        warm = cache.initial_known_warm(scope)
        descriptor = dict(action_index=0, inserts=[], releases=[0, 1],
                          base=sorted(scope.base), displaced=sorted(scope.displaced),
                          replacements=list(scope.replacements), q_seconds=scope.immediate_gain,
                          eligible_before_cap=scope.eligible_before_cap)
        descriptor["scope_sha256"] = p0.sha_json({key: descriptor[key] for key in
            ("inserts", "releases", "base", "displaced", "replacements", "q_seconds")})
        state = dict(original=sorted(original), original_value_seconds=api["objective"](graph, original),
                     best_selected=sorted(scope.base | warm), best_gain_seconds=2.,
                     warm_by_action={"0": sorted(warm)}, spent_requests=[])
        group = dict(snapshot=state, scopes=[descriptor], requests=[dict(snapshot_sha256=p0.sha_json(state))],
                     controller_state_identity=p0.snapshot_identity(state, [(scope, descriptor)]))
        class ReplayOnlyCache(api["Cache"]):
            def initial_known_warm(self, scope):
                raise AssertionError("Supplement must never rerun common greedy warm")
        api = dict(api, Cache=ReplayOnlyCache)
        before = p0.sha_json(group)
        reconstructed = supplement.reconstruct_frozen(graph, group, api)
        self.assertEqual(reconstructed[0][0].replacements, scope.replacements)
        self.assertEqual(before, p0.sha_json(group))
        changed = copy.deepcopy(group)
        changed["scopes"][0]["replacements"] = list(reversed(changed["scopes"][0]["replacements"]))
        with self.assertRaises(ValueError):
            supplement.reconstruct_frozen(graph, changed, api)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=str(Path(__file__).resolve().parents[4] / "第一篇"))
    args, test_args = parser.parse_known_args()
    RUNTIME_ROOT = args.runtime_root
    unittest.main(argv=[__file__] + test_args)
