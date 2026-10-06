"""Independent controlled graphs and transparent schedule initializations.

Controlled graphs are abstract optimization tests, not satellite physics.
Coupling changes cross-agent edge probability; compatibility changes the
probability that otherwise sampled alternative-alternative conflicts survive.
Graph seeds must be split before constructing states or action examples.
"""

from __future__ import annotations

import numpy as np

from .core import Action, Graph, State, weighted_greedy


def graph_from_edges(weights, agents, edges, name: str = "graph") -> Graph:
    adjacency = [set() for _ in weights]
    for u, v in edges:
        if u != v:
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
    return Graph(np.asarray(weights), np.asarray(agents), tuple(frozenset(a) for a in adjacency), name)


def random_block_graph(
    seed: int, coupling: float = 0.5, compatibility: float = 0.5,
    n: int = 96, k: int = 4,
) -> Graph:
    """Random block graph with reproducible coupling and compatibility controls.

    Vertex IDs divisible by three denote latent incumbent contacts; remaining
    IDs denote latent alternatives. These design labels are never features or
    imposed starting selections. In addition to random block edges, each
    three-contact bundle contains a path incumbent--alternative and
    incumbent--alternative, leaving the two alternatives potentially jointly
    recoverable. Cross-agent edges vary with coupling. Edges between any two
    latent alternatives are retained with probability 1 - compatibility.
    The design is intended to stress recovery interactions, with all actual
    action roles determined solely by the selected state and proposed move.
    """
    if not 0 <= coupling <= 1 or not 0 <= compatibility <= 1:
        raise ValueError("coupling and compatibility must lie in [0, 1]")
    if n < 3 or k < 2 or k > n:
        raise ValueError("controlled graphs require n >= 3 and 2 <= k <= n")
    rng = np.random.default_rng(seed)
    agents = np.arange(n, dtype=np.int64) % k
    weights = rng.uniform(3.0, 20.0, size=n)
    edges: list[tuple[int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            same = agents[u] == agents[v]
            probability = 0.065 if same else 0.015 + 0.105 * coupling
            sampled = rng.random() < probability
            bundle = u // 3 == v // 3 and u % 3 == 0
            if sampled or bundle:
                if u % 3 != 0 and v % 3 != 0 and rng.random() < compatibility:
                    continue
                edges.append((u, v))
    label = f"random-block-seed{seed}-c{coupling:.2f}-p{compatibility:.2f}-n{n}-k{k}"
    return graph_from_edges(weights, agents, edges, label)


def controlled_graph(
    seed: int, coupling: float = 0.5, compatibility: float = 0.5,
    n: int = 96, k: int = 4,
) -> Graph:
    """Structured recovery bundles with independent cross-agent interventions.

    Each six-contact bundle consists of one incumbent anchor, one action-like
    alternative and up to four recovery-like alternatives. The anchor
    conflicts with every other bundle member, but the action-like alternative
    does not conflict with its own recovery alternatives. Coupling controls
    action-to-other-anchor conflicts and cross-bundle, cross-agent conflicts.
    Compatibility controls conflicts within and across recovery alternatives.
    These construction roles are withheld from learned features; runtime roles
    follow only the actual incumbent and action. Initial controlled incumbents
    deliberately prioritize anchors to expose the recovery problem.
    """
    if not 0 <= coupling <= 1 or not 0 <= compatibility <= 1:
        raise ValueError("coupling and compatibility must lie in [0, 1]")
    if n < 6 or k < 2 or k > n:
        raise ValueError("controlled bundles require n >= 6 and 2 <= k <= n")
    rng = np.random.default_rng(seed)
    ids = np.arange(n)
    agents = (ids // 6 + ids % 6) % k
    weights = rng.uniform(4.0, 12.0, size=n)
    weights[ids % 6 == 0] = rng.uniform(20.0, 30.0, size=np.count_nonzero(ids % 6 == 0))
    weights[ids % 6 == 1] = rng.uniform(12.0, 20.0, size=np.count_nonzero(ids % 6 == 1))
    bundles = (n + 5) // 6
    edges: list[tuple[int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            same_bundle = u // 6 == v // 6
            u_role, v_role = u % 6, v % 6
            conflict = False
            if same_bundle:
                if u_role == 0:
                    conflict = True
                elif u_role >= 2 and v_role >= 2:
                    conflict = rng.random() < 0.85 * (1 - compatibility)
            elif agents[u] != agents[v]:
                if (u_role == 1 and v_role == 0) or (u_role == 0 and v_role == 1):
                    conflict = rng.random() < min(0.5, 1.6 * coupling / max(1, bundles - 1))
                elif u_role >= 2 and v_role >= 2:
                    conflict = rng.random() < (0.01 + 0.09 * coupling) * (1 - compatibility)
                elif u_role == 1 and v_role == 1:
                    conflict = rng.random() < 0.03 * coupling
            if conflict:
                edges.append((u, v))
    label = f"controlled-bundle-seed{seed}-c{coupling:.2f}-p{compatibility:.2f}-n{n}-k{k}"
    return graph_from_edges(weights, agents, edges, label)


def initial_state(graph: Graph, seed: int = 0, perturbation: float = 0.45) -> frozenset[int]:
    """Seeded randomized greedy state; always feasible and maximal.

    Lognormal priority perturbation creates imperfect but plausible incumbents,
    independently of the teacher's outcomes. At perturbation zero this is the
    deterministic weighted-degree greedy incumbent.
    """
    if perturbation < 0:
        raise ValueError("perturbation must be nonnegative")
    rng = np.random.default_rng(seed)
    degree = np.array([len(a) for a in graph.adjacency], dtype=np.float64)
    priority = graph.weights / np.sqrt(1 + degree)
    priority = priority * np.exp(rng.normal(0, perturbation, graph.n))
    if graph.name.startswith("controlled-bundle-"):
        # Anchors form a feasible independent set by construction. The large
        # fixed priority bonus encodes only benchmark initialization, never
        # an outcome or a prediction feature, and creates improvable states.
        priority[np.arange(graph.n) % 6 == 0] += 1000.0
    return weighted_greedy(graph, priority)


def preference_reversal_witness(reverse: bool = False) -> tuple[Graph, frozenset[int], tuple[Action, Action]]:
    """Same individual gains and recoverable-weight sums, reversed joint values.

    Both actions gain 5 immediately and free two alternatives of weight 4.
    Exactly one freed pair conflicts; changing which pair conflicts reverses
    action preferences from (9, 13) to (13, 9), while additive predictions for
    both remain 13. This is a diagnostic fixture rather than training data.
    """
    weights = [10, 10, 15, 15, 4, 4, 4, 4]
    agents = [0, 1, 0, 1, 1, 2, 0, 2]
    edges = [(0, 2), (0, 4), (0, 5), (1, 3), (1, 6), (1, 7)]
    edges.append((6, 7) if reverse else (4, 5))
    graph = graph_from_edges(weights, agents, edges, f"preference-reversal-{int(reverse)}")
    return graph, frozenset({0, 1}), (Action((2,)), Action((3,)))


def joint_overlap_witness() -> tuple[Graph, State, Action]:
    """Cross-agent pair whose individual replacement values cannot be summed."""
    weights = [10, 10, 11, 11, 9, 9]
    agents = [0, 1, 0, 1, 1, 0]
    edges = [(0, 2), (1, 3), (0, 4), (1, 5), (4, 5)]
    graph = graph_from_edges(weights, agents, edges, "joint-replacement-conflict")
    return graph, State(frozenset({0, 1})), Action((2, 3))
