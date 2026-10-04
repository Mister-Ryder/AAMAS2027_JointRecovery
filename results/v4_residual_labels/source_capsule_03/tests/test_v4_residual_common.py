"""Ten finite residual-prefix guards; no model/native/cloud/research graph."""
import ast
import hashlib
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments import v4_residual_common as common
from experiments.v4_neighborhoods import CoordinationAction, NeighborhoodScopeCache
from joint_recovery import v4_budgeted_recovery as core
from joint_recovery import v4_residual_controller_coordination as controller


class Clock:
    def __init__(self): self.t = 0.
    def __call__(self): return self.t
    def spend(self, value): self.t += value


def graph(weights, agents, edges, native=False):
    rows = [set() for _ in weights]
    for u, v in edges:
        rows[u].add(v); rows[v].add(u)
    cls = core.NativeIntegerGraph if native else core.Graph
    return cls(np.asarray(weights, dtype=np.int64 if native else np.float64),
               np.asarray(agents), tuple(frozenset(row) for row in rows))


def star():
    return graph([10, 6, 6, 6, 1], [0, 0, 1, 1, 1], [(0, 1), (0, 2), (0, 3)]), frozenset({0, 4})


def run(graph_input=None, selected=None, backend=None, priority=None, actions=None, **options):
    g, S = star()
    g = g if graph_input is None else graph_input
    S = S if selected is None else selected
    action = CoordinationAction((), tuple(sorted(S)))
    default = dict(workpoints=(core.Workpoint("one", "nodes", 0, .01),),
                   max_queries=1, clock=Clock())
    default.update(options)
    return controller.run_coordinated_recovery(g, S,
        lambda *args: [action] if actions is None else actions,
        (lambda g, scope, w, warm, remaining: core.RepairAttempt(warm)) if backend is None else backend,
        core.classical_priority("round_robin") if priority is None else priority,
        1., **default)


class ResidualCommonGuards(unittest.TestCase):
    def test_three_actual_sweeps_plus_original_floor_are_feasible_and_compared(self):
        g, S = star(); cache = common.ExecutedWarmScopeCache(g, S)
        scope = cache.scope(0, CoordinationAction((), tuple(S)), 256)
        with patch.object(common, "_check_recovery", wraps=common._check_recovery) as checks:
            warm = cache.initial_known_warm(scope)
        record = cache.preparatory_recoveries[(0, 256)]
        self.assertEqual(common.EXPONENTS, (0., .5, 1.))
        self.assertEqual(checks.call_count, 4)
        self.assertEqual([core.objective(g, values) for values in record["candidates"]], [11., 11., 11., 19.])
        self.assertEqual(warm, frozenset({1, 2, 3, 4}))
        self.assertTrue(all(core.is_feasible(g, scope.base | values) for values in record["candidates"]))

    def test_stronger_original_floor_is_not_replaced_by_any_greedy(self):
        g = graph([6, 7, 6, 4, 4, 4], [0, 1, 1, 0, 2, 1],
                  [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])
        S = frozenset({0, 2, 4}); cache = common.ExecutedWarmScopeCache(g, S)
        scope = cache.scope(0, CoordinationAction((), tuple(S)), 256)
        self.assertEqual(cache.initial_known_warm(scope), S)
        values = [core.objective(g, values) for values in cache.preparatory_recoveries[(0, 256)]["candidates"]]
        self.assertEqual(values, [16., 15., 15., 15.])

    def test_native_integer_selection_and_gain_above_two_to_53_are_exact(self):
        g = graph([(1 << 53)+1, 1 << 53, (1 << 53)+2, 1], [0, 0, 1, 1],
                  [(0, 1), (0, 2)], native=True)
        S = frozenset({0, 3})
        out = run(graph_input=g, selected=S, max_queries=0)
        self.assertEqual(out.selected, frozenset({1, 2, 3}))
        self.assertIsInstance(out.incumbent_value, int)
        self.assertEqual(out.incumbent_value, (1 << 53)+2)
        self.assertIsInstance(out.on_time_gain, int)
        self.assertEqual(out.on_time_gain, (1 << 53)+1)
        self.assertEqual(core.objective(g, out.selected), (1 << 54)+3)

    def test_forced_commitment_keeps_base_and_never_restores_mandatory_conflicts(self):
        g, S = star(); cache = common.ExecutedWarmScopeCache(g, S)
        scope = cache.scope(0, CoordinationAction((1,), (4,)), 256)
        warm = cache.initial_known_warm(scope)
        self.assertEqual(scope.base, frozenset({1}))
        self.assertEqual(warm, frozenset({2, 3, 4}))
        self.assertNotIn(0, warm); self.assertNotIn(1, warm)
        self.assertTrue(core.is_feasible(g, scope.base | warm))

    def test_cache_reuses_once_and_rejects_foreign_or_mutated_semantic_scope(self):
        g, S = star(); cache = common.ExecutedWarmScopeCache(g, S)
        scope = cache.scope(0, CoordinationAction((), tuple(S)), 256)
        with patch.object(common, "executed_greedy_warm", wraps=common.executed_greedy_warm) as sweeps:
            first = cache.initial_known_warm(scope)
            self.assertIs(first, cache.initial_known_warm(scope))
            self.assertEqual(sweeps.call_count, 1)
        for changed in (replace(scope, replacements=tuple(reversed(scope.replacements))),
                        replace(scope, base=frozenset({1})), replace(scope, inserts=(1,)),
                        replace(scope, immediate_gain=scope.immediate_gain+1),
                        replace(scope, resource_cliques=((1, 2),)), replace(scope, cap=64)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                cache.initial_known_warm(changed)
        fresh = common.ExecutedWarmScopeCache(g, S)
        with self.assertRaises(ValueError): fresh.initial_known_warm(scope)

    def test_public_entry_fixed_scope_factory_and_unexecuted_warm_fail_closed(self):
        for caps in ((64,), (1024,), (64, 256), ()):
            with self.subTest(caps=caps), self.assertRaises(ValueError):
                run(actions=[], caps=caps)
        with self.assertRaises(ValueError):
            run(actions=[], known_warm_factory=lambda scope: frozenset())
        with self.assertRaises(ValueError): run(scope_cache_factory=None)
        with self.assertRaises(ValueError): run(scope_cache_factory=NeighborhoodScopeCache)

    def test_prefix_cost_and_actual_warm_views_are_paid_and_lower_does_not_repeat(self):
        clock = Clock(); observed = []; repairs = []; original = common.executed_greedy_warm
        def paid(*args):
            result = original(*args); clock.spend(.25); return result
        def priority(g, views, state):
            observed.append((state, views))
            return core.PriorityEvaluation([100.]*len(views))
        def backend(g, scope, w, warm, remaining):
            repairs.append(warm); return core.RepairAttempt(warm)
        with patch.object(common, "executed_greedy_warm", side_effect=paid) as sweeps:
            out = run(clock=clock, priority=priority, backend=backend)
            self.assertEqual(sweeps.call_count, 1)
            state, views = observed[0]
            self.assertEqual(common.already_paid_lower(star()[0], views[0].request.scope, views[0].warm_start), 8.)
            self.assertEqual(sweeps.call_count, 1)
        self.assertEqual(repairs, [frozenset({1, 2, 3, 4})])
        self.assertEqual(state.best_gain, 8.); self.assertEqual(state.remaining_seconds, .75)
        self.assertEqual(out.stage_seconds["known_warm_validation"], .25)
        self.assertEqual(out.controller_return_sample_seconds, .25)

    def test_common_prefix_validation_at_deadline_is_retained_but_never_admitted(self):
        clock = Clock(); called = []; original = common.executed_greedy_warm
        def late(*args):
            result = original(*args); clock.spend(1.); return result
        with patch.object(common, "executed_greedy_warm", side_effect=late):
            out = run(clock=clock, backend=lambda *args: called.append(args))
        self.assertEqual(called, []); self.assertEqual(out.selected, star()[1])
        self.assertEqual(out.stop_reason, "known_warm_deadline")
        self.assertEqual(out.on_time_gain, 0.)
        self.assertEqual(len(out.known_warm_observations), 1)
        self.assertFalse(out.known_warm_observations[0].admitted_on_time)
        self.assertEqual(out.known_warm_observations[0].gain_if_valid, 8.)

    def test_failed_and_exactly_late_backend_calls_remain_spent_with_full_receipts(self):
        clock = Clock(); calls = []; original = common.executed_greedy_warm
        def paid(*args):
            result = original(*args); clock.spend(.25); return result
        def backend(g, scope, w, warm, remaining):
            calls.append((w.name, warm))
            if w.name == "failure":
                clock.spend(.25); raise RuntimeError("mock failure only")
            clock.spend(.5); return core.RepairAttempt(warm, "returned", None)
        workpoints = (core.Workpoint("failure", "nodes", 0, .01), core.Workpoint("late", "nodes", 1, .01))
        with patch.object(common, "executed_greedy_warm", side_effect=paid):
            out = run(clock=clock, backend=backend, workpoints=workpoints, max_queries=2)
        self.assertEqual([name for name, _ in calls], ["failure", "late"])
        self.assertEqual(len(out.spent_requests), 2); self.assertEqual(len(out.attempts), 2)
        self.assertEqual(out.attempts[0].status, "backend_exception:RuntimeError")
        self.assertTrue(out.attempts[1].valid); self.assertFalse(out.attempts[1].admitted_on_time)
        self.assertTrue(out.attempts[1].status.startswith("validated_late:"))
        self.assertEqual(out.attempts[1].recovered, frozenset({1, 2, 3, 4}))
        self.assertIn("invalid_diagnostics_type", out.attempts[1].diagnostics)
        self.assertEqual(out.selected, frozenset({1, 2, 3, 4}))
        self.assertEqual(out.on_time_gain, 8.)  # only the earlier validated common prefix

    def test_derivation_preserves_historical_source_and_does_not_patch_its_globals(self):
        self.assertEqual(hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                         "b4b5bbd3d8abdeba2fcb9a117c54f161786c2deccc032764e792d5899bd3d1ad")
        old = Path(core.__file__).with_name("v4_controller_coordination.py")
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),
                         "7bb0ec286ef10fccbc5898e3048af4486e9f592f97cfac615a609de5270820d0")
        self.assertIs(core.run_budgeted_recovery.__globals__["ScopeCache"], core.ScopeCache)
        self.assertEqual(controller.DERIVATION_RECEIPT["matched_statement_hooks"], 4)
        self.assertFalse(controller.DERIVATION_RECEIPT["shared_module_globals_modified"])
        ast.parse(Path(controller.__file__).read_text(), feature_version=(3, 8))
        ast.parse(Path(common.__file__).read_text(), feature_version=(3, 8))


if __name__ == "__main__": unittest.main()
