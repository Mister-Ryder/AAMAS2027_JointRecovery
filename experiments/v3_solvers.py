"""Strong independent-set recovery controls with verified feasible outputs.

HiGHS is called through SciPy's official MILP interface. Optional KaMIS uses
the official weighted branch-and-reduce / weighted local-search binaries.
These optimize the action-local recovery scope; neither receives a learned
score or an executed-outcome label as an input.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from joint_recovery.core import is_feasible


def greedy_recovery(graph, response, exponent=1.):
    vertices = frozenset(response.replacements)
    order = sorted(vertices, key=lambda v:(-float(graph.weights[v]) /
                   (1 + len(graph.adjacency[v] & vertices)) ** exponent,
                   -float(graph.weights[v]), v))
    chosen = set()
    for node in order:
        if not (graph.adjacency[node] & chosen):
            chosen.add(node)
    return frozenset(chosen)


def greedy_portfolio(graph, response):
    choices = [greedy_recovery(graph, response, exponent) for exponent in (0., .5, 1.)]
    return max(choices, key=lambda chosen:(sum(graph.weights[v] for v in chosen), tuple(sorted(chosen))))


def local_search(graph, response, checks=4096, deadline=None, initial=None):
    scope = frozenset(response.replacements)
    selected = set(greedy_portfolio(graph, response) if initial is None else initial)
    order = sorted(scope, key=lambda node:(-float(graph.weights[node]), node))
    work = 0; changed = True
    while changed and work < checks:
        changed = False
        for node in order:
            if work >= checks or (deadline is not None and time.perf_counter() >= deadline):
                return frozenset(selected), work
            work += 1
            if node in selected:
                continue
            remove = graph.adjacency[node] & selected
            if graph.weights[node] > sum(graph.weights[peer] for peer in remove) + 1e-9:
                selected.difference_update(remove); selected.add(node); changed = True
                for candidate in order:
                    if work >= checks or (deadline is not None and time.perf_counter() >= deadline):
                        return frozenset(selected), work
                    work += 1
                    if candidate not in selected and not (graph.adjacency[candidate] & selected):
                        selected.add(candidate)
    return frozenset(selected), work


def solve_highs(graph, response, time_limit=.1, node_limit=None, seed=17):
    started = time.perf_counter()
    vertices = tuple(response.replacements); size = len(vertices)
    fallback = greedy_portfolio(graph, response)
    fallback_value = float(sum(graph.weights[v] for v in fallback))
    if not size:
        return fallback, dict(status=0, success=True, optimal=True, bound=0., gap=0.,
                              elapsed_ms=(time.perf_counter() - started) * 1000, fallback=False)
    index = {node:position for position,node in enumerate(vertices)}
    edges = [(index[u], index[v]) for u in vertices for v in graph.adjacency[u]
             if v in index and u < v]
    constraints = None
    if edges:
        columns = [node for pair in edges for node in pair]
        rows = np.repeat(np.arange(len(edges)), 2)
        matrix = coo_matrix((np.ones(len(columns)), (rows, columns)), shape=(len(edges), size)).tocsc()
        # SciPy 1.10/HiGHS requires 32-bit sparse indices on the cloud image.
        matrix.indices = matrix.indices.astype(np.int32); matrix.indptr = matrix.indptr.astype(np.int32)
        constraints = LinearConstraint(matrix, -np.inf, 1.)
    options = dict(presolve=True, mip_rel_gap=0., threads=1, random_seed=int(seed))
    if time_limit is not None:
        options["time_limit"] = max(.001, float(time_limit) - (time.perf_counter() - started))
    if node_limit is not None:
        options["node_limit"] = int(node_limit)
    # SciPy documents forwarding unknown option names verbatim to HiGHS.
    # These two HiGHS options fix reproducibility and CPU concurrency; the
    # expected forwarding warning is recorded by the protocol, not a failure.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
        result = milp(c=-graph.weights[list(vertices)], integrality=np.ones(size),
                      bounds=Bounds(np.zeros(size), np.ones(size)), constraints=constraints,
                      options=options)
    selected = fallback
    used_fallback = True
    if result.x is not None:
        candidate = frozenset(vertices[position] for position,x in enumerate(result.x) if x > .5)
        if is_feasible(graph, candidate) and sum(graph.weights[v] for v in candidate) >= fallback_value - 1e-8:
            selected = candidate; used_fallback = False
    if not is_feasible(graph, selected) or any(graph.adjacency[v] & response.base for v in selected):
        raise AssertionError("Strong solver returned an infeasible recovery")
    bound = -float(result.mip_dual_bound) if getattr(result, "mip_dual_bound", None) is not None else None
    gap = float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None
    details = dict(status=int(result.status), success=bool(result.success), optimal=int(result.status) == 0,
                   bound=bound, gap=gap, elapsed_ms=(time.perf_counter() - started) * 1000,
                   fallback=used_fallback, nodes=int(getattr(result, "mip_node_count", 0) or 0))
    return selected, details


def solve_kamis(graph, response, binary, time_limit=.1, seed=17):
    started = time.perf_counter()
    vertices = tuple(response.replacements); size = len(vertices)
    fallback = greedy_portfolio(graph, response)
    if not size:
        return fallback, dict(status=0, optimal=None, elapsed_ms=0., fallback=False, rounding_bound=0.)
    index = {node:position for position,node in enumerate(vertices)}
    neighbors = [sorted(index[v] + 1 for v in graph.adjacency[node] if v in index) for node in vertices]
    weights = np.maximum(1, np.rint(graph.weights[list(vertices)] * 1000).astype(np.int64))
    with tempfile.TemporaryDirectory(prefix="aamas-v3-kamis-") as directory:
        source = Path(directory) / "scope.graph"; output = Path(directory) / "selected.txt"
        lines = ["%d %d 10" % (size, sum(map(len, neighbors)) // 2)]
        lines.extend(" ".join(map(str, [int(weights[position])] + peers)) for position,peers in enumerate(neighbors))
        source.write_text("\n".join(lines) + "\n", encoding="ascii")
        command = [str(binary), str(source), "--time_limit=" + str(max(.001, time_limit)),
                   "--seed=" + str(seed), "--output=" + str(output)]
        try:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     timeout=max(1., time_limit + .5), check=False, text=True)
            status = process.returncode
            if output.exists():
                values = [int(value) for value in output.read_text().split()]
                if len(values) != size or any(value not in (0, 1) for value in values):
                    raise RuntimeError("Unexpected KaMIS membership output")
                candidate = frozenset(vertices[position] for position,value in enumerate(values) if value == 1)
                if not is_feasible(graph, candidate):
                    raise AssertionError("KaMIS membership output is infeasible")
                selected = max([fallback, candidate], key=lambda chosen:sum(graph.weights[v] for v in chosen))
                used_fallback = selected == fallback and candidate != fallback
            else:
                selected, used_fallback = fallback, True
        except subprocess.TimeoutExpired:
            selected, used_fallback, status = fallback, True, 124
    return selected, dict(status=status, optimal=None, elapsed_ms=(time.perf_counter() - started) * 1000,
                          fallback=used_fallback, rounding_bound=size * .0005)


def realized_gain(graph, action, response, recovery):
    if not is_feasible(graph, recovery) or any(graph.adjacency[v] & response.base for v in recovery):
        raise AssertionError("An executable local recovery is required")
    delta = float(sum(graph.weights[v] for v in action.inserts) -
                  sum(graph.weights[v] for v in response.displaced) +
                  sum(graph.weights[v] for v in recovery))
    return max(0., delta), response.base | recovery
