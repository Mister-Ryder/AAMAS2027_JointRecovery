"""Efficiency audit kernel: same search, local objective-delta bookkeeping.

The primary frozen kernel remains untouched. This copy changes only the final
raw-gain calculation: unchanged incumbent contacts cancel algebraically, so
gain is insertion weight minus displacement weight plus recovered weight.
Response construction, feasibility checks, order, tie rules, work units and
search bounds are shared with the original implementation. Floating-point
summation paths may differ by roundoff; the audit checks output sets exactly
and numerical gains within tolerance. Inherited satellite weights are integer
durations, for which these tested sums are exact in float64.
"""

from __future__ import annotations

import heapq
from time import perf_counter
from typing import Iterable

import numpy as np

from .core import (
    Action, Graph, Outcome, State, _popcount, _selection, _validate_state,
    build_response, is_feasible,
)


def efficient_execute(
    graph: Graph, selected: State | Iterable[int], action: Action,
    budget: int = 64, max_replacements: int = 20, trusted_state: bool = False,
) -> Outcome:
    """Frozen bounded recovery algorithm with a local objective delta."""
    if not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    start = perf_counter()
    current = _selection(selected) if trusted_state else _validate_state(graph, selected)
    response = build_response(graph, current, action, max_replacements, trusted_state=True)
    candidates = response.replacements
    size = len(candidates)
    weights = [float(graph.weights[v]) for v in candidates]
    index = {v: i for i, v in enumerate(candidates)}
    conflicts = [sum(1 << index[u] for u in graph.adjacency[v] if u in index) for v in candidates]
    order = sorted(range(size), key=lambda i: (-weights[i] / (1 + _popcount(conflicts[i])), -weights[i], candidates[i]))
    best_mask, best_weight = 0, 0.0
    warm_checks = 0
    for i in order:
        if warm_checks >= budget:
            break
        warm_checks += 1
        if not (conflicts[i] & best_mask):
            best_mask |= 1 << i
            best_weight += weights[i]
    def upper(allowed: int, weight: float) -> float:
        return weight + sum(weights[i] for i in range(size) if allowed & (1 << i))
    all_mask = (1 << size) - 1
    queue: list[tuple[float, int, int, int, float]] = []
    serial = 0
    if upper(all_mask, 0.0) > best_weight + 1e-12:
        heapq.heappush(queue, (-upper(all_mask, 0.0), serial, all_mask, 0, 0.0))
    search_expansions = 0
    while queue and warm_checks + search_expansions < budget:
        negative_bound, _, allowed, chosen, partial_weight = heapq.heappop(queue)
        search_expansions += 1
        if -negative_bound <= best_weight + 1e-12:
            continue
        if partial_weight > best_weight + 1e-12:
            best_mask, best_weight = chosen, partial_weight
        if not allowed:
            continue
        candidates_left = [i for i in order if allowed & (1 << i)]
        branch = max(candidates_left, key=lambda i: (_popcount(conflicts[i] & allowed), weights[i], -candidates[i]))
        bit = 1 << branch
        for next_allowed, next_chosen, next_weight in (
            (allowed & ~bit & ~conflicts[branch], chosen | bit, partial_weight + weights[branch]),
            (allowed & ~bit, chosen, partial_weight),
        ):
            if next_weight > best_weight + 1e-12:
                best_mask, best_weight = next_chosen, next_weight
            bound = upper(next_allowed, next_weight)
            if bound > best_weight + 1e-12:
                serial += 1
                heapq.heappush(queue, (-bound, serial, next_allowed, next_chosen, next_weight))
    recovered = frozenset(candidates[i] for i in range(size) if best_mask & (1 << i))
    raw_selected = response.base | recovered
    local_feasible = is_feasible(graph, recovered) and all(not (graph.adjacency[v] & response.base) for v in recovered)
    if not local_feasible or (not trusted_state and not is_feasible(graph, raw_selected)):
        raise AssertionError("internal recovery produced an infeasible schedule")
    raw_gain = (sum(float(graph.weights[v]) for v in action.inserts)
                - sum(float(graph.weights[v]) for v in sorted(response.displaced))
                + sum(float(graph.weights[v]) for v in sorted(recovered)))
    gain = raw_gain if raw_gain > 1e-10 else 0.0
    final = raw_selected if gain > 0.0 else current
    return Outcome(gain, raw_gain, final, raw_selected, warm_checks + search_expansions,
                   (perf_counter() - start) * 1000.0, response, warm_checks, search_expansions)


execute = efficient_execute
