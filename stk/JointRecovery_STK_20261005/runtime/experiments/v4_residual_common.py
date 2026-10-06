"""Executed common greedy warm prefix for the reserved R256 candidate.

No graph, solver, neural fit or result is opened on import. Every recovery is
computed from the original observable graph, never a teacher label. The caller
charges this construction and complete membership validation to its deadline.
"""
from __future__ import annotations
from math import isfinite

from joint_recovery.v4_budgeted_recovery import _check_recovery, objective
from experiments.v4_neighborhoods import NeighborhoodScopeCache

CAPS = (256,)
EXPONENTS = (0., .5, 1.)

def executed_greedy_warm(graph, scope, original):
    """Same original floor plus three actually executed compatible sweeps."""
    floor = _check_recovery(graph, scope, frozenset(original) & frozenset(scope.replacements))
    pool = frozenset(scope.replacements)
    degrees = {v:len(graph.adjacency[v] & pool) for v in pool}
    candidates = [floor]
    for exponent in EXPONENTS:
        order = sorted(pool, key=lambda v:(
            -graph.weights[v].item() if exponent == 0. else
            -float(graph.weights[v]) / (1 + degrees[v])**exponent,
            -graph.weights[v].item(), v))
        recovered = set()
        for v in order:
            if not graph.adjacency[v] & recovered:
                recovered.add(v)
        candidates.append(_check_recovery(graph, scope, frozenset(recovered)))
    result = max(candidates, key=lambda values:(objective(graph, values), tuple(sorted(values))))
    if not isfinite(float(objective(graph, result))) or objective(graph, result) < objective(graph, floor):
        raise ValueError('Executed warm failed the observed feasible original floor')
    return result, tuple(candidates)

class ExecutedWarmScopeCache(NeighborhoodScopeCache):
    """An explicit new warm semantics; original cache remains immutable."""
    def __init__(self, graph, selected, resource_cliques=()):
        super().__init__(graph, selected, resource_cliques)
        self.preparatory_recoveries = {}

    def initial_known_warm(self, scope):
        if scope.cap != 256:
            raise ValueError('Every method uses the same fixed R256 prefix')
        key = (scope.action_index, scope.cap)
        declared = self._scopes.get(key)
        if declared is None or any(getattr(scope, field, None) != value for field, value in vars(declared).items()):
            raise ValueError('Executed warm must use this cache\'s exact declared original scope')
        if key not in self.preparatory_recoveries:
            selected, candidates = executed_greedy_warm(self.graph, scope, self.selected)
            self.preparatory_recoveries[key] = dict(selected=selected, candidates=candidates,
                action_index=scope.action_index, cap=scope.cap,
                original_vertices=tuple(scope.replacements),
                source='actually executed common exponents0/.5/1 plus original membership')
        record = self.preparatory_recoveries[key]
        if record['original_vertices'] != tuple(scope.replacements):
            raise ValueError('Executed prefix cannot alias different original pools')
        return record['selected']

def already_paid_lower(graph, scope, warm):
    """Lower comparator uses an already executed feasible warm, no repeat work."""
    actual = _check_recovery(graph, scope, warm)
    return max(0., scope.immediate_gain + objective(graph, actual))
