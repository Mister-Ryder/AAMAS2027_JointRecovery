"""V4 finite mature repair adapters; actual memberships and cost, no oracle.

SciPy/HiGHS has no x0 argument. Its solver is cold on each call; the actual
known warm recovery is retained by the adapter, not claimed passed to HiGHS.
CHILS receives that same recovery through its official native -i interface.
Both are soft-budget procedures; controller admission owns the deadline.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from joint_recovery.core import is_feasible
from joint_recovery.v4_budgeted_recovery import RepairAttempt, objective
try:
    from .v3_solvers import greedy_portfolio
    from .v3_published_baselines import solve_chils, sha256_file
except ImportError:
    from v3_solvers import greedy_portfolio
    from v3_published_baselines import solve_chils, sha256_file


CHILS_SHA256 = "19610c03f334c6267f94543ad3053d792cba56e9ae211fceb6c36f21750c88a0"


def highs_backend(graph, scope, workpoint, warm_start, remaining_seconds):
    started = time.perf_counter()
    if workpoint.kind not in ("seconds", "nodes"):
        raise ValueError("HiGHS supports seconds or node-count workpoints")
    vertices = scope.replacements
    fallback = max((frozenset(warm_start), greedy_portfolio(graph, scope)),
                   key=lambda s: (objective(graph, s), tuple(sorted(s))))
    if not vertices:
        return RepairAttempt(fallback, "empty_scope", {
            "elapsed_seconds": time.perf_counter()-started,
            "solver_warmstart": False, "retained_known_warmstart": True})
    edges = scope.canonical_edges(graph)
    size = len(vertices)
    constraints = None
    if edges:
        columns = np.asarray(edges, dtype=np.int32).reshape(-1)
        rows = np.repeat(np.arange(len(edges), dtype=np.int32), 2)
        matrix = coo_matrix((np.ones(len(columns)), (rows, columns)),
                            shape=(len(edges), size)).tocsc()
        matrix.indices = matrix.indices.astype(np.int32)
        matrix.indptr = matrix.indptr.astype(np.int32)
        constraints = LinearConstraint(matrix, -np.inf, 1.)
    prep = time.perf_counter()-started
    native_seconds = (float(workpoint.amount) if workpoint.kind == "seconds"
                      else float(remaining_seconds))
    seconds = min(native_seconds, float(remaining_seconds))-prep
    if seconds <= 0:
        return RepairAttempt(fallback, "preparation_exhausted_slice", {
            "elapsed_seconds": time.perf_counter()-started, "preparation_seconds": prep,
            "solver_warmstart": False, "retained_known_warmstart": True})
    options = dict(presolve=True, mip_rel_gap=0., threads=1, random_seed=17,
                   time_limit=seconds)
    if workpoint.kind == "nodes":
        options["node_limit"] = int(workpoint.amount)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
        solved = milp(c=-graph.weights[list(vertices)], integrality=np.ones(size),
                      bounds=Bounds(np.zeros(size), np.ones(size)),
                      constraints=constraints, options=options)
    best = fallback
    if solved.x is not None:
        candidate = frozenset(vertices[i] for i, x in enumerate(solved.x) if x > .5)
        if is_feasible(graph, candidate) and not any(graph.adjacency[v]&scope.base for v in candidate):
            if objective(graph, candidate) > objective(graph, best):
                best = candidate
    return RepairAttempt(best, "highs_status_"+str(int(solved.status)), {
        "elapsed_seconds": time.perf_counter()-started, "preparation_seconds": prep,
        "solver_seconds_requested": seconds, "solver_warmstart": False,
        "retained_known_warmstart": True, "nodes": int(getattr(solved, "mip_node_count", 0) or 0),
        "optimal": int(solved.status) == 0,
        "fallback_selected": best == fallback})


def chils_backend(binary):
    binary = Path(binary).resolve()
    if not binary.is_file():
        raise FileNotFoundError("Pinned official CHILS executable is unavailable")
    digest = sha256_file(binary)
    if digest != CHILS_SHA256:
        raise ValueError("CHILS executable does not match the independently audited pinned binary")
    def execute(graph, scope, workpoint, warm_start, remaining_seconds):
        if workpoint.kind != "seconds":
            raise ValueError("CHILS workpoint must be in seconds")
        started = time.perf_counter()
        vertices = scope.replacements
        if not vertices:
            return RepairAttempt(frozenset(), "empty_scope", {"binary_sha256": digest})
        edges = np.asarray(scope.canonical_edges(graph), dtype=np.int64).reshape(-1, 2)
        index = {v:i for i,v in enumerate(vertices)}
        initial = np.zeros(len(vertices), dtype=np.int8)
        for v in warm_start: initial[index[v]] = 1
        # Adapter conversion/export/process/output validation consumes the
        # declared slice as well as the outer decision's remaining time.
        available = min(float(workpoint.amount), float(remaining_seconds))-(time.perf_counter()-started)
        if available <= 0:
            return RepairAttempt(frozenset(warm_start), "preparation_exhausted_slice", {
                "binary_sha256": digest, "solver_warmstart": True})
        integral = np.array_equal(graph.weights[list(vertices)],
                                  np.rint(graph.weights[list(vertices)]))
        scale = 1 if integral else 1000
        mask, details = solve_chils(graph.weights[list(vertices)], edges[:,0], edges[:,1],
            binary, total_seconds=available, seed=17, initial_mask=initial,
            population=1, threads=1, weight_scale=scale)
        returned = frozenset(vertices[i] for i in np.flatnonzero(mask))
        details.update(binary_sha256=digest, solver_warmstart=True,
                       adapter_elapsed_seconds=time.perf_counter()-started)
        return RepairAttempt(returned, "chils_returned" if details.get("returned_solution")
                             else "chils_no_new_output", details)
    return execute
