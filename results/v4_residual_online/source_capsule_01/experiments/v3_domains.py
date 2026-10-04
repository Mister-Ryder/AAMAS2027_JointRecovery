"""New split-safe domains for resource scheduling and hard joint recovery.

Resource contacts are synthetic intervals contained in generated visibility
windows. This module does not claim an orbit-propagation or measured-contact
data set. Its complete constraints are directly checked against intervals,
independently of the conflict graph used by the optimizer.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from joint_recovery.core import Action, is_feasible, proposals, weighted_greedy
from joint_recovery.generators import graph_from_edges, random_block_graph

try:
    from .v3_pilot import menu_graph
except ImportError:
    from v3_pilot import menu_graph


def satellite_resource_graph(seed, satellites=8, grounds=4, passes=48,
                             alternatives=6, horizon=7200., satellite_setup=30.,
                             ground_setup=15.):
    """Intervals obey one simultaneous contact per satellite and ground.

    A pass is a visibility window for one satellite-ground pair. Candidate
    contacts are distinct possible transmissions contained inside that window.
    Satellite agents retain their identity throughout the graph. Ground
    contention induces cross-agent incompatibilities. Setup times are imposed
    exactly before another contact sharing either resource may start.
    """
    rng = np.random.default_rng(seed)
    n = passes * alternatives
    contacts = []
    for pass_id in range(passes):
        satellite = pass_id % satellites
        ground = int(rng.integers(0, grounds))
        window_length = float(rng.uniform(360., 900.))
        start_window = float(rng.uniform(0., max(1., horizon - window_length)))
        end_window = start_window + window_length
        for alternative in range(alternatives):
            # A window may support one long transmission or several shorter
            # separately rewarded transmissions. This duration heterogeneity
            # creates a resource-valid recovery problem rather than adding
            # arbitrary incompatibility edges to the contact graph.
            duration = float(rng.uniform(.60 * window_length, .85 * window_length)
                             if alternative == 0 else
                             rng.uniform(30., min(180., .30 * window_length)))
            start = float(rng.uniform(start_window, end_window - duration))
            priority = float(rng.choice([1., 1.5, 2.], p=[.6, .3, .1]))
            contacts.append(dict(satellite=satellite, ground=ground, start=start,
                                 end=start + duration, duration=duration,
                                 priority=priority, visibility_start=start_window,
                                 visibility_end=end_window, pass_id=pass_id))
    weights = np.asarray([contact["duration"] * contact["priority"] for contact in contacts])
    agents = np.asarray([contact["satellite"] for contact in contacts])
    edges = []
    for first, one in enumerate(contacts):
        for second in range(first + 1, n):
            two = contacts[second]
            satellite_shared = one["satellite"] == two["satellite"]
            ground_shared = one["ground"] == two["ground"]
            setup = max(satellite_setup if satellite_shared else 0.,
                        ground_setup if ground_shared else 0.)
            if (satellite_shared or ground_shared) and not (
                one["end"] + setup <= two["start"] or two["end"] + setup <= one["start"]):
                edges.append((first, second))
    graph = graph_from_edges(weights, agents, edges,
                             "v3-resource-s%d-sat%d-ground%d-pass%d-alt%d" %
                             (seed, satellites, grounds, passes, alternatives))
    metadata = dict(kind="synthetic_visibility_window_contacts", seed=seed,
                    satellites=satellites, grounds=grounds, horizon=horizon,
                    satellite_setup=satellite_setup, ground_setup=ground_setup,
                    contacts=contacts)
    return graph, metadata


def check_resource_schedule(metadata, selected):
    contacts = metadata["contacts"]
    for node in selected:
        item = contacts[node]
        if item["start"] < item["visibility_start"] - 1e-8 or item["end"] > item["visibility_end"] + 1e-8:
            return False
    for resource, setup in (("satellite", metadata["satellite_setup"]),
                            ("ground", metadata["ground_setup"])):
        by_resource = {}
        for node in selected:
            item = contacts[node]
            by_resource.setdefault(item[resource], []).append(item)
        for items in by_resource.values():
            items.sort(key=lambda item: item["start"])
            if any(one["end"] + setup > two["start"] + 1e-8 for one, two in zip(items, items[1:])):
                return False
    return True


def resource_initial(graph, seed):
    rng = np.random.default_rng(seed)
    degrees = np.asarray([len(neighbors) for neighbors in graph.adjacency])
    priority = graph.weights / np.sqrt(1. + degrees)
    priority *= np.exp(rng.normal(0., .6, graph.n))
    return weighted_greedy(graph, priority)


def diverse_action_pool(graph, state, seed, max_actions=16, max_pool=64):
    """Shared pool mixes immediate candidates with fixed random exploration.

    No teacher or model score influences candidate generation. Include a
    disjoint-state seeded random half to avoid restricting recovery learning
    to only the largest immediate-gain candidates.
    """
    rng = np.random.default_rng(seed)
    available = [v for v in range(graph.n) if v not in state]
    ranked = sorted(available, key=lambda v:(-(float(graph.weights[v]) -
                    sum(graph.weights[u] for u in graph.adjacency[v] & state)), v))
    chosen = ranked[:max_pool // 2]
    remaining = [v for v in available if v not in chosen]
    rng.shuffle(remaining)
    chosen += remaining[:max_pool - len(chosen)]
    singles = [Action((v,)) for v in chosen]
    pairs = [Action((u, v)) for index, u in enumerate(chosen) for v in chosen[index + 1:]
             if graph.agents[u] != graph.agents[v] and v not in graph.adjacency[u]]
    def score(action):
        removed = frozenset(v for u in action.inserts for v in graph.adjacency[u] & state)
        return float(sum(graph.weights[v] for v in action.inserts) -
                     sum(graph.weights[v] for v in removed))
    selected = []
    for source, quota in ((singles, max_actions // 2), (pairs, max_actions - max_actions // 2)):
        source.sort(key=lambda action:(-score(action), action.inserts))
        deterministic = min(len(source), max(1, quota // 2))
        selected.extend(source[:deterministic])
        rest = source[deterministic:]
        rng.shuffle(rest)
        selected.extend(rest[:quota - deterministic])
    return selected


def graph_hash(graph):
    digest = hashlib.sha256()
    digest.update(np.asarray(graph.weights, dtype="<f8").tobytes())
    digest.update(np.asarray(graph.agents, dtype="<i8").tobytes())
    edges = [(u, v) for u, neighbors in enumerate(graph.adjacency)
             for v in sorted(neighbors) if u < v]
    digest.update(np.asarray(edges, dtype="<i8").reshape(-1, 2).tobytes())
    return digest.hexdigest()


def plan_v3(train_count=120, validation_count=36, test_count=48):
    """Seeds fixed by split before outcomes; final tests are not pilot seeds."""
    plan = []
    for split, count, start in (("train", train_count, 810000),
                                ("validation", validation_count, 820000),
                                ("test_iid", test_count, 830000),
                                ("test_size", test_count // 2, 840000),
                                ("test_coupling", test_count // 2, 850000),
                                ("test_resource", test_count, 860000)):
        for index in range(count):
            seed = start + index
            resource = split == "test_resource" or (split in ("train", "validation") and index % 3 == 0)
            if resource:
                large = split == "test_resource" and index % 2 == 1
                spec = dict(domain="resource", seed=seed, split=split,
                            satellites=12 if large else 8,
                            grounds=[2, 4, 8][index % 3], passes=96 if large else 48,
                            alternatives=8 if large else 6)
            else:
                large = split == "test_size"
                strong = split == "test_coupling"
                spec = dict(domain="menu", seed=seed, split=split,
                            replacements=64 if large else [16, 32, 48][index % 3],
                            menus=16 if large else 8,
                            density=[.08, .20, .45][(index // 3) % 3],
                            coupling=.20 if strong else [.01, .04, .08][index % 3],
                            topology=["erdos", "components", "bipartite"][(index // 9) % 3])
            plan.append(spec)
    assert len({item["seed"] for item in plan}) == len(plan)
    return plan


def construct_case(spec, max_actions=16):
    args = {key:value for key,value in spec.items() if key not in ("domain", "split")}
    if spec["domain"] == "resource":
        graph, metadata = satellite_resource_graph(**args)
        state = resource_initial(graph, spec["seed"] + 700000)
        actions = diverse_action_pool(graph, state, spec["seed"] + 900000, max_actions=max_actions)
        if not check_resource_schedule(metadata, state):
            raise AssertionError("Independent interval schedule check failed")
    else:
        graph, state, actions = menu_graph(**args)
        metadata = {"kind":"controlled_recovery_menu"}
        singles = [action for action in actions if len(action.inserts) == 1]
        pairs = [action for action in actions if len(action.inserts) == 2]
        single_quota = max_actions // 2
        actions = singles[:single_quota] + pairs[:max_actions - single_quota]
    if not is_feasible(graph, state):
        raise AssertionError("Initial graph schedule is infeasible")
    if not actions:
        raise ValueError("Domain generated no feasible commitments")
    return graph, state, actions, metadata


if __name__ == "__main__":
    for index in range(4):
        graph, metadata = satellite_resource_graph(810000 + index, grounds=2 + index * 2)
        state = resource_initial(graph, 123 + index)
        print(graph.name, graph.n, sum(map(len, graph.adjacency)) // 2,
              len(state), check_resource_schedule(metadata, state))
