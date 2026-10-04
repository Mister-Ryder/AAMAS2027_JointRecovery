"""Validation-only stress pilot for genuinely coupled, costly recovery.

This pilot contains no final-test seeds. It measures a common deterministic
execution kernel and stronger feasible recovery surrogates before choosing a
learning design. The menu construction is an optimization stress generator,
not a physical satellite-contact generator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from joint_recovery.core import Action, build_response, is_feasible
from joint_recovery.efficient_core import efficient_execute
from joint_recovery.generators import graph_from_edges


def menu_graph(seed, replacements=32, menus=12, density=.15, coupling=.04,
               topology="erdos"):
    """Many actions share an incumbent but free diverse recovery subgraphs.

    Each menu has an anchor, an insertion and R alternatives. Anchors are
    initially committed; insertion displaces its anchor. Recovery alternatives
    have heterogeneous weights, actual conflict graphs and physical-agent
    labels. Cross-menu alternatives can also conflict, so paired actions do
    not decompose automatically. Menu labels never enter learning features.
    """
    rng = np.random.default_rng(seed)
    width = replacements + 2
    n = width * menus
    agents = np.arange(n, dtype=np.int64) % max(4, menus // 2)
    weights = rng.uniform(2., 16., n)
    edges = []
    anchors, inserts = [], []
    for menu in range(menus):
        anchor, insert = menu * width, menu * width + 1
        anchors.append(anchor); inserts.append(insert)
        agents[anchor] = menu % max(4, menus // 2)
        agents[insert] = menu % max(4, menus // 2)
        weights[anchor] = rng.uniform(20., 35.)
        weights[insert] = rng.uniform(15., 30.)
        alternatives = list(range(menu * width + 2, (menu + 1) * width))
        edges.extend((anchor, other) for other in [insert] + alternatives)
        local_density = np.clip(density * rng.uniform(.5, 1.5), .01, .9)
        for first, u in enumerate(alternatives):
            for second, v in enumerate(alternatives[first + 1:], first + 1):
                if topology == "cliques":
                    conflict = first // 5 == second // 5 or rng.random() < local_density / 10
                elif topology == "bipartite":
                    conflict = (first % 2 != second % 2) and rng.random() < min(1., 2 * local_density)
                elif topology == "components":
                    conflict = first // 8 == second // 8 and rng.random() < min(1., local_density * 3)
                else:
                    conflict = rng.random() < local_density
                if conflict:
                    edges.append((u, v))
    # Cross-menu interactions matter only when two commitments free both sets.
    for first_menu in range(menus):
        first = range(first_menu * width + 2, (first_menu + 1) * width)
        for second_menu in range(first_menu + 1, menus):
            second = range(second_menu * width + 2, (second_menu + 1) * width)
            for u in first:
                for v in second:
                    if agents[u] != agents[v] and rng.random() < coupling:
                        edges.append((u, v))
    graph = graph_from_edges(weights, agents, edges,
                             "v3-menu-%s-r%d-m%d-d%.2f-c%.2f-s%d" %
                             (topology, replacements, menus, density, coupling, seed))
    actions = [Action((u,)) for u in inserts]
    pairs = []
    for first, u in enumerate(inserts):
        for v in inserts[first + 1:]:
            if agents[u] != agents[v]:
                pairs.append(Action((u, v)))
    # Independent of teacher outcomes; preserve singles and a fixed pair quota.
    rng.shuffle(pairs)
    actions += pairs[:menus]
    state = frozenset(anchors)
    if not is_feasible(graph, state):
        raise AssertionError("Generator incumbent is infeasible")
    return graph, state, actions


def response_stats(graph, response):
    vertices = frozenset(response.replacements)
    adjacency = {v: graph.adjacency[v] & vertices for v in vertices}
    unseen = set(vertices); components = []
    while unseen:
        pending = [min(unseen)]; unseen.remove(pending[0]); component = []
        while pending:
            v = pending.pop(); component.append(v)
            for u in sorted(adjacency[v] & unseen):
                unseen.remove(u); pending.append(u)
        components.append(component)
    edges = sum(len(peers) for peers in adjacency.values()) // 2
    cross = sum(graph.agents[u] != graph.agents[v]
                for u, peers in adjacency.items() for v in peers if u < v)
    triangles = sum(len(adjacency[u] & adjacency[v])
                    for u, peers in adjacency.items() for v in peers if u < v) // 3
    size = len(vertices)
    return dict(replacement_count=size, uncapped_count=response.candidate_count_before_cap,
                components=len(components), largest_component=max(map(len, components), default=0),
                rr_edges=edges, rr_density=2 * edges / max(1, size * (size - 1)),
                rr_triangles=triangles, cross_agent_edges=cross)


def immediate_delta(graph, response, action):
    return float(sum(graph.weights[v] for v in action.inserts)
                 - sum(graph.weights[v] for v in response.displaced))


def feasible_surrogates(graph, response, action, local_checks=1024):
    """Independent strong cheap controls use feasible lower bounds, not sums.

    Three degree/weight greedy starts are followed by weighted improving
    one-in/many-out exchanges and refill. All sets remain feasible. This is a
    transparent weighted independent-set local-search control; it is not
    labeled as a reproduction of a named external algorithm.
    """
    vertices = list(response.replacements); scope = frozenset(vertices)
    adjacent = {v: graph.adjacency[v] & scope for v in vertices}
    orders = [sorted(vertices, key=lambda v, p=p:
                     (-float(graph.weights[v]) / (1 + len(adjacent[v])) ** p,
                      -float(graph.weights[v]), v)) for p in (0., .5, 1.)]
    starts = []
    start_time = time.perf_counter()
    for order in orders:
        chosen = set()
        for v in order:
            if not (adjacent[v] & chosen):
                chosen.add(v)
        starts.append(chosen)
    greedy_ms = (time.perf_counter() - start_time) * 1000
    best = max(starts, key=lambda chosen: (sum(graph.weights[v] for v in chosen), -len(chosen)))
    best_value = float(sum(graph.weights[v] for v in best))
    checks = 0; selected = set(best)
    start_time = time.perf_counter()
    improved = True
    while improved and checks < local_checks:
        improved = False
        for v in orders[0]:
            if checks >= local_checks:
                break
            checks += 1
            if v in selected:
                continue
            remove = adjacent[v] & selected
            if graph.weights[v] > sum(graph.weights[u] for u in remove) + 1e-9:
                selected.difference_update(remove); selected.add(v); improved = True
                for u in orders[2]:
                    if checks >= local_checks:
                        break
                    checks += 1
                    if u not in selected and not (adjacent[u] & selected):
                        selected.add(u)
        if improved:
            candidate_value = float(sum(graph.weights[v] for v in selected))
            if candidate_value > best_value:
                best = set(selected); best_value = candidate_value
    local_ms = (time.perf_counter() - start_time) * 1000
    if not is_feasible(graph, best):
        raise AssertionError("Cheap local search produced an infeasible recovery")
    immediate = immediate_delta(graph, response, action)
    return {"Immediate":max(0., immediate),
            "AllReplacementSum":max(0., immediate + sum(graph.weights[v] for v in vertices)),
            "GreedyPortfolio":max(0., immediate + max(sum(graph.weights[v] for v in chosen) for chosen in starts)),
            "SwapLocalSearch":max(0., immediate + best_value)}, \
           {"Immediate":0., "AllReplacementSum":0., "GreedyPortfolio":greedy_ms,
            "SwapLocalSearch":greedy_ms + local_ms}, checks


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(quick=False):
    plan = []
    index = 0
    for topology in ("erdos", "components", "bipartite"):
        for replacements in (16, 32, 64):
            for density in (.08, .20, .45):
                for menus in (8, 24):
                    index += 1
                    plan.append(dict(seed=730000 + index, topology=topology,
                                     replacements=replacements, density=density,
                                     menus=menus, coupling=.03 if topology == "components" else .08))
    return plan[:1] if quick else plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/v3_pilot")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 2048, 16384])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plan = make_plan(args.quick)
    if args.limit:
        plan = plan[:args.limit]
    protocol = dict(purpose="validation-only recovery complexity pilot; no final test",
                    plan=plan, budgets=args.budgets, max_replacements=128,
                    cheap_local_checks=1024, script_sha256=file_sha(__file__),
                    kernel_sha256=file_sha(Path(__file__).resolve().parents[1] / "src/joint_recovery/efficient_core.py"))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows, action_rows = [], []
    started = time.perf_counter()
    for case_index, spec in enumerate(plan):
        graph, state, actions = menu_graph(**spec)
        preparation_start = time.perf_counter()
        responses = [build_response(graph, state, action, 128) for action in actions]
        preparation_ms = (time.perf_counter() - preparation_start) * 1000
        cheap_scores = {key:[] for key in ("Immediate", "AllReplacementSum", "GreedyPortfolio", "SwapLocalSearch")}
        cheap_times = {key:0. for key in cheap_scores}
        for action_index, (action, response) in enumerate(zip(actions, responses)):
            scores, times, checks = feasible_surrogates(graph, response, action)
            for method in cheap_scores:
                cheap_scores[method].append(scores[method]); cheap_times[method] += times[method]
            action_rows.append(dict(case=case_index, action=action_index, graph=graph.name,
                                    commitments=len(action.inserts), local_checks=checks,
                                    **response_stats(graph, response)))
        for budget in args.budgets:
            oracle_start = time.perf_counter()
            outcomes = [efficient_execute(graph, state, action, budget=budget,
                                          max_replacements=128) for action in actions]
            oracle_ms = (time.perf_counter() - oracle_start) * 1000
            gains = np.asarray([outcome.gain for outcome in outcomes])
            teacher = float(gains.max())
            common = dict(case=case_index, graph=graph.name, n=graph.n, actions=len(actions),
                          seed=spec["seed"], topology=spec["topology"], replacements=spec["replacements"],
                          menus=spec["menus"], density=spec["density"], coupling=spec["coupling"],
                          budget=budget, teacher_gain=teacher, oracle_ms=oracle_ms,
                          preparation_ms=preparation_ms,
                          total_expansions=sum(item.expansions for item in outcomes),
                          mean_expansions=float(np.mean([item.expansions for item in outcomes])),
                          mean_search_expansions=float(np.mean([item.search_expansions for item in outcomes])),
                          mean_rr_size=float(np.mean([len(response.replacements) for response in responses])),
                          mean_rr_component=float(np.mean([row["largest_component"] for row in action_rows if row["case"] == case_index])))
            for method, score in cheap_scores.items():
                selected = int(np.argmax(score)); chosen = outcomes[selected]
                if not is_feasible(graph, chosen.selected):
                    raise AssertionError("Shared execution returned an infeasible state")
                rows.append(dict(**common, method=method, selected_action=selected,
                                 realized_gain=float(chosen.gain), regret=teacher - float(chosen.gain),
                                 feasible=True, ranking_ms=cheap_times[method],
                                 online_ms=preparation_ms + cheap_times[method] + chosen.elapsed_ms,
                                 execution_ms=chosen.elapsed_ms,
                                 selected_search_expansions=chosen.search_expansions))
            rows.append(dict(**common, method="ExplicitLookahead", selected_action=int(gains.argmax()),
                             realized_gain=teacher, regret=0., feasible=True, ranking_ms=oracle_ms,
                             online_ms=oracle_ms, execution_ms=0., selected_search_expansions=0))
            print("PILOT", case_index + 1, "/", len(plan), graph.name, "B", budget,
                  "oracle_ms", round(oracle_ms, 2), "gain", round(teacher, 3), flush=True)
            for name, values in (("decisions.csv", rows), ("response_structure.csv", action_rows)):
                with (out / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    (out / "completion.json").write_text(json.dumps(dict(completed=True, cases=len(plan),
        decisions=len(rows), action_structures=len(action_rows), elapsed_seconds=time.perf_counter() - started), indent=2))


if __name__ == "__main__":
    main()
