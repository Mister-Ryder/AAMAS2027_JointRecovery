"""Semantics-preserving preparation for the frozen v3 recovery models.

Only observable inputs are cached, within ONE decision. No module-level cache,
teacher, outcome, candidate reordering or alternative clique heuristic is used.
The shared response preparation is available to learned and nonlearned methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from .core import (ACTION, ACTION_DEPENDENCY, DISPLACED, INCUMBENT,
                   INCOMPATIBILITY, RECOVERY_DEPENDENCY, REPLACEMENT,
                   SAME_AGENT, Response, _selection, _validate_action,
                   _validate_state, build_response, is_feasible)
from .v3_capacity import _vector


class DecisionContext:
    """One graph/incumbent decision: validate once and cache observable stats.

    Discard after that decision. ``trusted_state`` requires caller validation.
    Returned responses have exactly the original scope, order and features.
    Kernels still independently check the recovered and final selected sets.
    """
    def __init__(self, graph, selected, trusted_state=False):
        self.graph = graph
        self.selected = (_selection(selected) if trusted_state
                         else _validate_state(graph, selected))
        self._vertex_stats = {}
        self._responses = {}
        self._factors = {}

    def vertex_stats(self, nodes):
        graph = self.graph
        for v in nodes:
            if v not in self._vertex_stats:
                neighbors = graph.adjacency[v]
                degree = len(neighbors)
                owner_count = sum(graph.agents[u] == graph.agents[v] for u in neighbors)
                self._vertex_stats[v] = (degree, owner_count / max(1, degree))
        return np.asarray([self._vertex_stats[v] for v in nodes], dtype=np.float64)

    def response(self, action, max_replacements=20):
        if max_replacements < 0:
            raise ValueError("max_replacements must be nonnegative")
        key = (action.inserts, max_replacements)
        if key in self._responses:
            return self._responses[key]
        graph, current = self.graph, self.selected
        _validate_action(graph, current, action)
        actions = frozenset(action.inserts)
        removed = frozenset(v for u in actions for v in graph.adjacency[u] if v in current)
        base = (current - removed) | actions
        scope = set(removed)
        for v in removed:
            scope.update(graph.adjacency[v])
        eligible = [v for v in scope if v not in base and not (graph.adjacency[v] & base)]
        eligible.sort(key=lambda v: (-float(graph.weights[v]), v))
        replacements = tuple(eligible[:max_replacements])
        context = actions | removed | frozenset(replacements)
        boundary = set()
        remaining = current - removed
        for v in context:
            boundary.update(graph.adjacency[v] & remaining)
        incumbent = tuple(sorted(boundary))
        nodes = incumbent + action.inserts + tuple(sorted(removed)) + replacements
        # Small responses do not amortize dense relation-array setup. Reuse
        # the original constructor with already-validated incumbent instead.
        # Its extra tiny scope pass is charged inside this call, not cached
        # across decisions or omitted from timing.
        if len(nodes) <= 24:
            response = build_response(graph, current, action, max_replacements, trusted_state=True)
            self._responses[key] = response
            return response
        roles = np.asarray([INCUMBENT] * len(incumbent) + [ACTION] * len(actions)
                           + [DISPLACED] * len(removed) + [REPLACEMENT] * len(replacements),
                           dtype=np.int64)
        ids = np.asarray(nodes, dtype=np.int64)
        n = len(nodes)
        positions = {v: i for i, v in enumerate(nodes)}
        response_vertices = frozenset(nodes)
        local_neighbors = [graph.adjacency[v] & response_vertices for v in nodes]
        stats = self.vertex_stats(nodes)
        x = np.zeros((n, 10), dtype=np.float32)
        x[np.arange(n), roles] = 1.
        x[:, 4] = graph.weights[ids] / graph._max_weight
        x[:, 5] = stats[:, 0] / max(1, graph.n - 1)
        x[:, 6] = [v in current for v in nodes]
        x[:, 7] = [v in base for v in nodes]
        x[:, 8] = stats[:, 1]
        x[:, 9] = np.asarray(list(map(len, local_neighbors))) / max(1, n - 1)
        # A boolean relation tensor removes tuple/set insertion per directed
        # edge; lexsort recovers the original (source,target,type) ordering.
        typed = np.zeros((n, n, 4), dtype=np.bool_)
        for i, peers in enumerate(local_neighbors):
            if not peers:
                continue
            columns = np.fromiter((positions[v] for v in peers), dtype=np.int64)
            typed[i, columns, INCOMPATIBILITY] = True
            same = graph.agents[ids[columns]] == graph.agents[nodes[i]]
            typed[i, columns[same], SAME_AGENT] = True
        for u in action.inserts:
            peers = graph.adjacency[u] & removed
            if peers:
                columns = np.fromiter((positions[v] for v in peers), dtype=np.int64)
                i = positions[u]
                typed[i, columns, ACTION_DEPENDENCY] = True
                typed[columns, i, ACTION_DEPENDENCY] = True
        rp = np.flatnonzero(roles == REPLACEMENT)
        for u in removed:
            columns = rp[typed[positions[u], rp, INCOMPATIBILITY]]
            typed[positions[u], columns, RECOVERY_DEPENDENCY] = True
            typed[columns, positions[u], RECOVERY_DEPENDENCY] = True
        source, target, relation = np.nonzero(typed)
        edge_index = np.stack((source, target)).astype(np.int64, copy=False)
        edge_type = relation.astype(np.int64, copy=False)
        response = Response(x, edge_index, edge_type, nodes, replacements,
                            base, removed, roles, len(eligible))
        self._responses[key] = response
        return response

    def responses(self, actions, max_replacements=20):
        return [self.response(action, max_replacements) for action in actions]


def build_responses_fast(graph, selected, actions, max_replacements=20,
                         trusted_state=False):
    """One validation/cache lifetime for every action in a decision."""
    return DecisionContext(graph, selected, trusted_state).responses(actions, max_replacements)


def conflict_clique_cover_fast(adjacency, weights=None, mode="cliques"):
    """Same ordered greedy cover, with bitsets for clique intersections."""
    if mode not in ("none", "edges", "cliques"):
        raise ValueError("projection mode must be none, edges, or cliques")
    n = len(adjacency)
    neighbors = [frozenset(int(j) for j in row) for row in adjacency]
    if any(i in row or any(j < 0 or j >= n for j in row) for i, row in enumerate(neighbors)):
        raise ValueError("invalid local conflict adjacency")
    if any(i not in neighbors[j] for i, row in enumerate(neighbors) for j in row):
        raise ValueError("local conflict adjacency must be symmetric")
    edge_set = {(i, j) for i, row in enumerate(neighbors) for j in row if i < j}
    if mode == "none":
        return ()
    if mode == "edges":
        return tuple(sorted(edge_set))
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)):
        raise ValueError("clique ordering weights must be finite and size n")
    ordering = sorted(range(n), key=lambda i: (-float(w[i]), -len(neighbors[i]), i))
    masks = [sum(1 << j for j in row) for row in neighbors]
    factors, covered = set(), set()
    for seed in ordering:
        allowed = masks[seed]
        if not allowed:
            continue
        clique = [seed]
        for v in ordering:
            if allowed & (1 << v):
                clique.append(v)
                allowed &= masks[v]
        factor = tuple(sorted(clique))
        if len(factor) >= 2:
            factors.add(factor)
            covered.update((u, v) for index, u in enumerate(factor) for v in factor[index + 1:])
    factors.update(edge_set - covered)
    return tuple(sorted(factors))


def pack_v3_fast(graph, selected, responses, budget=64, scale=None, device="cpu",
                 projection="cliques", drop_replacement_edges=False, context=None):
    """Array-exact original packing with vectorized edge/owner assignments.

    Optional context may reuse identical R factors ONLY in this decision.
    Factor order and choice are identical to conflict_clique_cover. The model
    forward is unmodified, so no floating-point operation is reordered there.
    """
    del selected
    if context is not None and context.graph is not graph:
        raise ValueError("context belongs to another graph")
    count = len(responses)
    if not count:
        raise ValueError("at least one action response required")
    if scale is None:
        scale = max(1., float(np.mean(graph.weights)))
    scales, budgets = _vector(scale, count), _vector(budget, count)
    if (np.any(scales <= 0) or not np.all(np.isfinite(scales))
            or np.any(budgets < 0) or not np.all(np.isfinite(budgets))):
        raise ValueError("positive scales and nonnegative finite budgets required")
    nmax = max(len(r.node_ids) for r in responses)
    records, fmax, amax = [], 1, 1
    for r in responses:
        ids = np.asarray(r.node_ids, dtype=np.int64)
        rp = np.flatnonzero(r.roles == REPLACEMENT)
        rr_index = np.full(len(ids), -1, dtype=np.int64)
        rr_index[rp] = np.arange(len(rp))
        source, target = r.edge_index
        edge_keep = ((r.edge_type == INCOMPATIBILITY) & (rr_index[source] >= 0)
                     & (rr_index[target] >= 0))
        rr = np.zeros((len(rp), len(rp)), dtype=np.bool_)
        rr[rr_index[source[edge_keep]], rr_index[target[edge_keep]]] = True
        conflicts = [np.flatnonzero(row) for row in rr]
        mode = "none" if drop_replacement_edges else projection
        # Include actual observable incidence, so even a manually edited
        # response cannot reuse factors from a different edge relation.
        key = (tuple(ids[rp]), rr.tobytes(), mode)
        factors = context._factors.get(key) if context is not None else None
        if factors is None:
            factors = conflict_clique_cover_fast(conflicts, graph.weights[ids[rp]], mode)
            if context is not None:
                context._factors[key] = factors
        ownership = graph.agents[ids]
        distinct = np.unique(ownership)
        fmax, amax = max(fmax, len(factors)), max(amax, len(distinct))
        records.append((ids, rp, factors, ownership, distinct))
    x = np.zeros((count, nmax, 10), np.float32)
    mask = np.zeros((count, nmax), np.float32)
    weights = np.zeros((count, nmax), np.float32)
    replacement = np.zeros((count, nmax), np.float32)
    edges = np.zeros((count, 4, nmax, nmax), np.float32)
    cliques = np.zeros((count, fmax, nmax), np.float32)
    agents = np.zeros((count, amax, nmax), np.float32)
    immediate = np.zeros(count, np.float32)
    diagnostics = []
    for i, (r, record) in enumerate(zip(responses, records)):
        ids, rp, factors, ownership, distinct = record
        n = len(ids)
        x[i, :n] = r.x
        degrees = [len(graph.adjacency[int(v)]) for v in ids]
        x[i, :n, 5] = np.log1p(degrees) / np.log(65.)
        mask[i, :n] = 1.
        weights[i, :n] = graph.weights[ids] / scales[i]
        replacement[i, rp] = 1.
        immediate[i] = float(np.sum(weights[i, :n][r.roles == ACTION])
                             - np.sum(weights[i, :n][r.roles == DISPLACED]))
        source, target = r.edge_index
        keep = np.ones(len(r.edge_type), dtype=np.bool_)
        if drop_replacement_edges:
            keep &= ~((r.roles[source] == REPLACEMENT) & (r.roles[target] == REPLACEMENT))
        edges[i, r.edge_type[keep], target[keep], source[keep]] = 1.
        if factors:
            lengths = np.fromiter(map(len, factors), dtype=np.int64)
            factor_positions = np.repeat(np.arange(len(factors)), lengths)
            vertices = np.fromiter((v for factor in factors for v in factor), dtype=np.int64)
            cliques[i, factor_positions, rp[vertices]] = 1.
        agents[i, :len(distinct), :n] = distinct[:, None] == ownership[None, :]
        diagnostics.append(dict(nodes=n, replacements=len(rp), factors=len(factors),
                                factor_memberships=sum(map(len, factors)),
                                max_clique=max(map(len, factors), default=0), agents=len(distinct)))
    rcounts = replacement.sum(axis=1)
    conditioning = np.stack((np.log1p(budgets) / np.log(1025.),
                             np.log1p(budgets / np.maximum(1., rcounts)) / np.log(1025.),
                             np.log1p(rcounts) / np.log(65.)), axis=1).astype(np.float32)
    arrays = dict(x=x, mask=mask, weights=weights, replacement=replacement,
                  edges=edges, cliques=cliques, agents=agents,
                  immediate=immediate, budget_features=conditioning)
    result = {name: torch.as_tensor(array, device=device) for name, array in arrays.items()}
    result["diagnostics"] = diagnostics
    return result


@dataclass(frozen=True)
class PreparedStrongOutcome:
    gain: float
    raw_gain: float
    selected: frozenset
    raw_selected: frozenset
    expansions: int
    elapsed_ms: float
    response: object
    warm_checks: int
    search_expansions: int
    solver_info: dict


def strong_execute_prepared(context, action, response, budget=32, solve_recovery=None):
    """Reuse a prepared response with the UNMODIFIED actual strong solver.

    Caller injects experiments.v3_solvers.solve_highs to avoid introducing an
    experiment dependency into the package. Preparation MUST remain inside the
    outer decision timer; elapsed_ms here is only repair/verification time.
    Same solver seed=17, no wall deadline, node limit, fallback and acceptance
    threshold as strong_execute. Validate incumbent once via DecisionContext,
    and always validate recovery, its base compatibility and the final set.
    """
    if solve_recovery is None:
        raise ValueError("the unchanged strong solve_highs callback is required")
    started = perf_counter()
    graph, current = context.graph, context.selected
    _validate_action(graph, current, action)
    expected_base = (current - response.displaced) | frozenset(action.inserts)
    if response.base != expected_base or response.node_ids.count(action.inserts[0]) != 1:
        raise ValueError("response does not match this decision/action")
    # Require provenance from this ephemeral context, not an arbitrary stale
    # response. Identity checks do not rebuild or consult any solver output.
    if not any(response is cached and key[0] == action.inserts
               for key, cached in context._responses.items()):
        raise ValueError("prepare response through the current DecisionContext")
    recovery, info = solve_recovery(graph, response, time_limit=None, node_limit=budget, seed=17)
    recovery = frozenset(recovery)
    if (not recovery.issubset(response.replacements) or not is_feasible(graph, recovery)
            or any(graph.adjacency[v] & response.base for v in recovery)):
        raise AssertionError("Strong solver returned an infeasible recovery")
    raw_selected = response.base | recovery
    raw_gain = float(sum(graph.weights[v] for v in action.inserts)
                     - sum(graph.weights[v] for v in response.displaced)
                     + sum(graph.weights[v] for v in recovery))
    gain = max(0., raw_gain)
    final = raw_selected if gain > 1e-9 else current
    if not is_feasible(graph, final):
        raise AssertionError("Strong kernel output is infeasible")
    nodes = int(info.get("nodes", 0))
    return PreparedStrongOutcome(float(gain), raw_gain, frozenset(final), frozenset(raw_selected),
                                 nodes, (perf_counter() - started) * 1000., response,
                                 3 * len(response.replacements), nodes, info)
