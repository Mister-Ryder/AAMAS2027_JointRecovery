"""Deterministic, budgeted joint recovery for partitioned weighted graphs.

The graph partition is an agent label, not an additional feasibility constraint:
all conflicts, including within-agent conflicts, must be explicit graph edges.
No feature in this module uses an execution outcome or an optimization label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from time import perf_counter
from typing import Iterable

import numpy as np

# The provided GPU image uses Python 3.8; keep bit-mask search compatible.
_popcount = int.bit_count if hasattr(int, "bit_count") else lambda mask: bin(mask).count("1")


INCUMBENT, ACTION, DISPLACED, REPLACEMENT = range(4)
INCOMPATIBILITY, ACTION_DEPENDENCY, RECOVERY_DEPENDENCY, SAME_AGENT = range(4)
NODE_FEATURE_DIM = 10
NUM_EDGE_TYPES = 4
FEATURE_NAMES = (
    "incumbent", "action", "displaced", "replacement", "normalized_weight",
    "normalized_graph_degree", "currently_selected", "in_committed_base",
    "same_agent_neighbor_fraction", "normalized_response_conflict_degree",
)


@dataclass(frozen=True)
class Graph:
    weights: np.ndarray
    agents: np.ndarray
    adjacency: tuple[frozenset[int], ...]
    name: str = "graph"
    _max_weight: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        w = np.array(self.weights, dtype=np.float64, copy=True)
        agents = np.array(self.agents, dtype=np.int64, copy=True)
        adj = tuple(frozenset(int(j) for j in neighbors) for neighbors in self.adjacency)
        n = len(w)
        if w.ndim != 1 or agents.shape != w.shape or len(adj) != n:
            raise ValueError("weights, agents and adjacency must have the same one-dimensional size")
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("all weights must be finite and strictly positive")
        if np.any(agents < 0):
            raise ValueError("agent labels must be nonnegative")
        for i, neighbors in enumerate(adj):
            if i in neighbors or any(j < 0 or j >= n for j in neighbors):
                raise ValueError("adjacency must contain valid vertices and no self loops")
            if any(i not in adj[j] for j in neighbors):
                raise ValueError("incompatibility adjacency must be symmetric")
        w.setflags(write=False)
        agents.setflags(write=False)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "adjacency", adj)
        object.__setattr__(self, "_max_weight", float(w.max()) if n else 1.0)

    @property
    def n(self) -> int:
        return len(self.weights)


@dataclass(frozen=True)
class State:
    selected: frozenset[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected", frozenset(int(v) for v in self.selected))


@dataclass(frozen=True)
class Action:
    inserts: tuple[int, ...]

    def __post_init__(self) -> None:
        inserts = tuple(sorted(int(v) for v in self.inserts))
        if not 1 <= len(inserts) <= 2 or len(set(inserts)) != len(inserts):
            raise ValueError("an action has one or two distinct insert commitments")
        object.__setattr__(self, "inserts", inserts)


@dataclass(frozen=True)
class Response:
    x: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    node_ids: tuple[int, ...]
    replacements: tuple[int, ...]
    base: frozenset[int]
    displaced: frozenset[int]
    roles: np.ndarray
    candidate_count_before_cap: int


@dataclass(frozen=True)
class Outcome:
    gain: float
    raw_gain: float
    selected: frozenset[int]
    raw_selected: frozenset[int]
    expansions: int
    elapsed_ms: float
    response: Response
    warm_checks: int
    search_expansions: int

    @property
    def accepted(self) -> bool:
        return self.gain > 0.0

    @property
    def evaluations(self) -> int:
        """Charged work units; alias for expansions."""
        return self.expansions

    @property
    def final_schedule(self) -> frozenset[int]:
        return self.selected


def _selection(selected: State | Iterable[int]) -> frozenset[int]:
    return selected.selected if isinstance(selected, State) else frozenset(int(v) for v in selected)


def value(graph: Graph, selected: State | Iterable[int]) -> float:
    nodes = _selection(selected)
    return float(sum(float(graph.weights[v]) for v in sorted(nodes)))


def is_feasible(graph: Graph, selected: State | Iterable[int]) -> bool:
    nodes = _selection(selected)
    return all(0 <= v < graph.n and not (graph.adjacency[v] & nodes) for v in nodes)


def _validate_state(graph: Graph, selected: State | Iterable[int]) -> frozenset[int]:
    nodes = _selection(selected)
    if not is_feasible(graph, nodes):
        raise ValueError("the incumbent must be a feasible independent set")
    return nodes


def _validate_action(graph: Graph, selected: frozenset[int], action: Action) -> None:
    if any(v in selected or v < 0 or v >= graph.n for v in action.inserts):
        raise ValueError("insert commitments must be valid and currently unselected")
    if not is_feasible(graph, action.inserts):
        raise ValueError("insert commitments must be mutually compatible")
    if len(action.inserts) == 2 and graph.agents[action.inserts[0]] == graph.agents[action.inserts[1]]:
        raise ValueError("two-commitment coordination actions must cross agent partitions")


def displacement(graph: Graph, selected: State | Iterable[int], action: Action) -> frozenset[int]:
    current = _validate_state(graph, selected)
    _validate_action(graph, current, action)
    return frozenset(v for u in action.inserts for v in graph.adjacency[u] if v in current)


def immediate_gain(graph: Graph, selected: State | Iterable[int], action: Action) -> float:
    return value(graph, action.inserts) - value(graph, displacement(graph, selected, action))


def proposals(
    graph: Graph, selected: State | Iterable[int], max_actions: int = 12,
    candidate_pool: Iterable[int] | None = None, max_pool: int = 64,
    trusted_state: bool = False,
) -> list[Action]:
    """Deterministic pool containing single and compatible cross-agent pair moves.

    Ranking uses immediate insertion gain, which does not solve recovery. Up to
    half the pool is reserved for singles; the rest consists of the best pairs.
    For graphs of at most 160 vertices all unselected vertices are considered.
    Larger graphs first retain a static immediate-gain pool of ``max_pool``
    vertices. An explicit ``candidate_pool`` overrides this default; it allows
    the evaluator to advance through predeclared boundary windows without
    screening windows by teacher outcomes. Every strategy shares the pool.
    """
    current = _selection(selected) if trusted_state else _validate_state(graph, selected)
    if max_actions <= 0:
        return []
    pool = range(graph.n) if candidate_pool is None else sorted(set(int(v) for v in candidate_pool))
    available = [v for v in pool if 0 <= v < graph.n and v not in current]
    displaced = {v: graph.adjacency[v] & current for v in available}
    def score(nodes: tuple[int, ...]) -> float:
        removed = frozenset(w for v in nodes for w in displaced[v])
        return value(graph, nodes) - value(graph, removed)
    singles = sorted(((score((v,)), (v,)) for v in available), key=lambda item: (-item[0], item[1]))
    if candidate_pool is None and graph.n > 160:
        if max_pool <= 0:
            return []
        singles = singles[:max_pool]
        available = sorted(nodes[0] for _, nodes in singles)
    pairs = []
    for index, u in enumerate(available):
        for v in available[index + 1:]:
            if graph.agents[u] != graph.agents[v] and v not in graph.adjacency[u]:
                pairs.append((score((u, v)), (u, v)))
    pairs.sort(key=lambda item: (-item[0], item[1]))
    single_quota = min(len(singles), max(1, (max_actions + 1) // 2))
    chosen = singles[:single_quota] + pairs[:max_actions - single_quota]
    if len(chosen) < max_actions:
        chosen += singles[single_quota:single_quota + max_actions - len(chosen)]
    chosen.sort(key=lambda item: (-item[0], item[1]))
    return [Action(nodes) for _, nodes in chosen]


def build_response(
    graph: Graph, selected: State | Iterable[int], action: Action,
    max_replacements: int = 20, trusted_state: bool = False,
) -> Response:
    """Construct the exact action-local candidate scope used by execute.

    Recovery candidates come from displaced vertices and their excluded
    neighbors. Vertices incompatible with the committed base are removed.
    The cap is action-dependent: candidates are ordered by descending weight,
    then vertex ID, and the first ``max_replacements`` are retained. There is
    no claim that this scope contains the unrestricted optimum.
    """
    if max_replacements < 0:
        raise ValueError("max_replacements must be nonnegative")
    current = _selection(selected) if trusted_state else _validate_state(graph, selected)
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
    # Full incumbent membership remains in ``base`` for feasibility, but only
    # action-local commitments enter the learned response graph. In an
    # independent incumbent, this set is usually empty after candidate
    # feasibility filtering; displaced commitments still carry selected=1.
    context = actions | removed | frozenset(replacements)
    boundary_selected = set()
    remaining = current - removed
    for v in context:
        boundary_selected.update(graph.adjacency[v] & remaining)
    incumbent = tuple(sorted(boundary_selected))
    node_ids = incumbent + action.inserts + tuple(sorted(removed)) + replacements
    roles = np.array(
        [INCUMBENT] * len(incumbent) + [ACTION] * len(actions)
        + [DISPLACED] * len(removed) + [REPLACEMENT] * len(replacements),
        dtype=np.int64,
    )
    node_index = {v: i for i, v in enumerate(node_ids)}
    response_vertices = frozenset(node_ids)
    x = np.zeros((len(node_ids), NODE_FEATURE_DIM), dtype=np.float32)
    max_weight = graph._max_weight
    for i, v in enumerate(node_ids):
        neighbors = graph.adjacency[v]
        x[i, roles[i]] = 1.0
        x[i, 4] = graph.weights[v] / max_weight
        x[i, 5] = len(neighbors) / max(1, graph.n - 1)
        x[i, 6] = float(v in current)
        x[i, 7] = float(v in base)
        x[i, 8] = sum(graph.agents[u] == graph.agents[v] for u in neighbors) / max(1, len(neighbors))
        x[i, 9] = len(neighbors & response_vertices) / max(1, len(node_ids) - 1)
    edges: set[tuple[int, int, int]] = set()
    def add(u: int, v: int, kind: int) -> None:
        i, j = node_index[u], node_index[v]
        edges.add((i, j, kind))
        edges.add((j, i, kind))
    for u in node_ids:
        for v in sorted(graph.adjacency[u] & response_vertices):
            if u < v:
                add(u, v, INCOMPATIBILITY)
                if graph.agents[u] == graph.agents[v]:
                    add(u, v, SAME_AGENT)
    for u in action.inserts:
        for v in sorted(removed & graph.adjacency[u]):
            add(u, v, ACTION_DEPENDENCY)
    for u in sorted(removed):
        for v in replacements:
            if v in graph.adjacency[u]:
                add(u, v, RECOVERY_DEPENDENCY)
    ordered = sorted(edges)
    edge_index = np.array([(u, v) for u, v, _ in ordered], dtype=np.int64).T
    if not ordered:
        edge_index = np.empty((2, 0), dtype=np.int64)
    edge_type = np.array([kind for _, _, kind in ordered], dtype=np.int64)
    return Response(x, edge_index, edge_type, node_ids, replacements, base, removed, roles, len(eligible))


def execute(
    graph: Graph, selected: State | Iterable[int], action: Action,
    budget: int = 64, max_replacements: int = 20, trusted_state: bool = False,
) -> Outcome:
    """Apply commitments and recover a feasible subset under a shared budget.

    The measured latency includes response construction and kernel execution.
    A work unit is one candidate feasibility check in the deterministic greedy
    warm start or one expanded state in best-first branch and bound. Both are
    charged against the same budget; objective summation and graph construction
    are measured in latency but do not count as search work units. B&B uses only
    the sound sum-of-remaining-positive-weights upper bound. The incumbent is
    retained whenever the best realized update fails to improve it.
    ``trusted_state=True`` is for callers that already verified the complete
    incumbent. It skips repeated scans of untouched incumbent vertices while
    still checking commitments and the entire proposed local recovery.
    """
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
        # Every generated child already describes a feasible partial recovery.
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
    raw_gain = value(graph, raw_selected) - value(graph, current)
    gain = raw_gain if raw_gain > 1e-10 else 0.0
    final = raw_selected if gain > 0.0 else current
    return Outcome(gain, raw_gain, final, raw_selected, warm_checks + search_expansions,
                   (perf_counter() - start) * 1000.0, response, warm_checks, search_expansions)


def weighted_greedy(graph: Graph, priority: np.ndarray | None = None) -> frozenset[int]:
    """Construct a maximal feasible incumbent without optimization labels."""
    if priority is None:
        priority = graph.weights / np.sqrt(1 + np.array([len(v) for v in graph.adjacency]))
    priority = np.asarray(priority, dtype=np.float64)
    if priority.shape != (graph.n,) or not np.all(np.isfinite(priority)):
        raise ValueError("priority must be a finite vector of graph size")
    order = sorted(range(graph.n), key=lambda v: (-float(priority[v]), -float(graph.weights[v]), v))
    selected: set[int] = set()
    blocked: set[int] = set()
    for v in order:
        if v not in blocked:
            selected.add(v)
            blocked.add(v)
            blocked.update(graph.adjacency[v])
    return frozenset(selected)


def exact_recovery_value(graph: Graph, response: Response) -> float:
    """Exhaustive tiny-graph diagnostic; never used by features or execution."""
    vertices = response.replacements
    if len(vertices) > 24:
        raise ValueError("the exhaustive diagnostic is restricted to at most 24 candidates")
    best = 0.0
    for mask in range(1 << len(vertices)):
        chosen = frozenset(v for i, v in enumerate(vertices) if mask & (1 << i))
        if is_feasible(graph, chosen):
            best = max(best, value(graph, chosen))
    return best
