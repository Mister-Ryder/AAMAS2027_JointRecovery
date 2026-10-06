"""P0: actual fixed-base, warm-aligned recovery probes on NEW STK graphs.

This program deliberately has no neural-model, old-data or calibration import.
It imports four immutable algorithm modules from the supplied paper runtime.
Native CHILS sees microsecond integer rewards; every reported reward is rescored
from the original float64 contact duration and the complete graph/base.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

SCHEMA = "joint_recovery_stk_p0_v1"
CAP = 256
TICK_SECONDS = 1e-6
MAX_NATIVE_TOTAL = 1 << 40
MAX_NATIVE_VERTICES = 32760
NATIVE_SEED = 17
GLOBAL_GREEDY_EXPONENTS = (0.0, 0.5, 1.0)
REFERENCE_FILES = (
    "src/joint_recovery/core.py",
    "src/joint_recovery/v4_budgeted_recovery.py",
    "src/joint_recovery/v4_factors.py",
    "experiments/v4_neighborhoods.py",
    "experiments/v4_residual_common.py",
)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError('Unsupported scientific JSON value: ' + type(value).__name__)


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False, default=json_default).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_runtime(root):
    root = Path(root).resolve()
    receipt = {}
    for relative in REFERENCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError("Missing read-only algorithm dependency: " + str(path))
        receipt[relative] = sha_file(path)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    from joint_recovery.core import Graph, is_feasible
    from joint_recovery.v4_budgeted_recovery import objective, _check_recovery
    from joint_recovery.v4_factors import observable_factors
    from experiments.v4_neighborhoods import coordination_cells
    from experiments.v4_residual_common import ExecutedWarmScopeCache
    return dict(Graph=Graph, feasible=is_feasible, objective=objective,
                check=_check_recovery, factors=observable_factors,
                cells=coordination_cells, Cache=ExecutedWarmScopeCache,
                reference_sha256=receipt)


def scalar(archive, key, fallback=None):
    if key not in archive:
        return fallback
    value = archive[key]
    return value.item() if value.shape == () else value.tolist()


def load_graph(path, api):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        weights = np.asarray(archive["weights"], dtype=np.float64)
        agents = np.asarray(archive["agents"], dtype=np.int64)
        if "edges" in archive:
            edges = np.asarray(archive["edges"], dtype=np.int64).reshape(-1, 2)
        else:
            edges = np.column_stack((archive["edge_u"], archive["edge_v"]))
        n = len(weights)
        if edges.size and (np.any(edges < 0) or np.any(edges >= n) or
                           np.any(edges[:, 0] >= edges[:, 1])):
            raise ValueError("Original edges must be canonical u < v and in range")
        if len({tuple(row) for row in edges.tolist()}) != len(edges):
            raise ValueError("Duplicate original graph edges")
        adjacency = [set() for _ in range(n)]
        for u, v in edges:
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
        graph_id = str(scalar(archive, "graph_id", path.stem))
        graph = api["Graph"](weights, agents, tuple(map(frozenset, adjacency)), graph_id)
        metadata = dict(graph_id=graph_id, source_group=scalar(archive, "source_group"),
                        split=scalar(archive, "split"), npz_sha256=sha_file(path),
                        vertices=n, edges=len(edges), weight_unit="original_duration_seconds",
                        agent_contract="satellite_owner_is_label_not_whole_day_capacity",
                        resource_member_CSR_is_not_static_conflict_cliques=True)
        if "start_seconds" in archive and "end_seconds" in archive:
            durations = archive["end_seconds"] - archive["start_seconds"]
            binary_error = float(np.max(np.abs(weights - durations))) if n else 0.0
            if "start_decimal_seconds" in archive and "end_decimal_seconds" in archive:
                raw_durations = np.asarray([float(Decimal(str(e)) - Decimal(str(s)))
                    for s, e in zip(archive["start_decimal_seconds"], archive["end_decimal_seconds"])])
                if not np.array_equal(weights, raw_durations):
                    raise ValueError("Rewards differ from unmodified original decimal endpoint differences")
                metadata["raw_endpoint_duration_identity"] = "PASS_EXACT_ORIGINAL_DECIMAL_TO_FLOAT64"
            elif not np.array_equal(weights, durations):
                raise ValueError("Nonidentical float endpoint differences require original decimal endpoints")
            else:
                metadata["raw_endpoint_duration_identity"] = "PASS_EXACT_FLOAT64"
            if binary_error > 1e-9:
                raise ValueError("Float endpoint views differ from original duration by more than 1 ns")
            metadata["binary_endpoint_subtraction_max_error_seconds"] = binary_error
        if "edge_types" in archive:
            types = np.asarray(archive["edge_types"])
            if types.shape != (len(edges),):
                raise ValueError("Edge-kind array differs from edge list")
            metadata["edge_kind_counts"] = {str(kind): int(np.sum(types == kind))
                                            for kind in (1, 2, 3)}
        if "contact_id" in archive:
            metadata["contact_ids_sha256"] = sha_json(archive["contact_id"].tolist())
    return graph, metadata


def native_source_audit(source):
    """Bind conservative microsecond domain to actual longlong upstream source.

    An unsupported build fails explicitly. The p1 path excludes the cooperative
    multi-population loop. This is a narrow source/domain check, not an audit of
    all compiler/runtime behavior or a physical timing guarantee.
    """
    source = Path(source).resolve()
    files = list((source / "src").rglob("*.h")) + list((source / "src").rglob("*.c"))
    files += list((source / "include").rglob("*.h"))
    if not files:
        raise FileNotFoundError("Actual CHILS src files required for integer-domain check")
    contents = {str(path.relative_to(source)): path.read_text(encoding="utf-8", errors="replace")
                for path in files}
    all_text = "\n".join(contents.values())
    # Actual pinned CHILS declares reward, cost and adjacent_weight long long.
    checks = dict(
        reward_longlong=bool(re.search(r"long\s+long\s*\*\s*W\b", all_text)),
        cost_longlong=bool(re.search(r"long\s+long\s+(?:int\s+)?cost\b", all_text)),
        adjacent_weight_longlong=bool(re.search(r"long\s+long\s*\*\s*adjacent_weight\b", all_text)),
        p1_branch="if (run_chils > 1)" in all_text,
        p1_iteration_guard="while (c < il)" in all_text,
        log_reset="ls->log_count = 0;" in all_text,
    )
    if not all(checks.values()):
        raise ValueError("CHILS microsecond source contract unsupported: " + json.dumps(checks))
    return dict(status="PASS_NARROW_SOURCE_DOMAIN_CHECK", checks=checks,
                source_files_sha256={key: sha_file(source / key) for key in sorted(contents)},
                maximum_scope_vertices=MAX_NATIVE_VERTICES,
                maximum_tick_weight_sum=MAX_NATIVE_TOTAL,
                population=1, threads=1, native_seed=NATIVE_SEED,
                reward_arithmetic="signed long long; conservative p1 weight sum <= 2**40",
                not_a_universal_native_or_physical_timing_certificate=True)


def global_incumbent(graph, api):
    """Actually execute all three full-graph greeds and keep their best."""
    started = time.perf_counter()
    degrees = [len(row) for row in graph.adjacency]
    candidates = []
    for exponent in GLOBAL_GREEDY_EXPONENTS:
        order = sorted(range(graph.n), key=lambda v: (
            -float(graph.weights[v]) / (1 + degrees[v]) ** exponent,
            -float(graph.weights[v]), v))
        selected = set()
        for vertex in order:
            if not graph.adjacency[vertex] & selected:
                selected.add(vertex)
        selected = frozenset(selected)
        if not api["feasible"](graph, selected):
            raise AssertionError("Full-graph greedy returned infeasible membership")
        candidates.append(selected)
    best = max(candidates, key=lambda members: (api["objective"](graph, members), tuple(sorted(members))))
    return best, dict(preparation_seconds=time.perf_counter() - started,
                      selected=sorted(best), objective_seconds=api["objective"](graph, best),
                      candidates=[dict(exponent=e, selected=sorted(c), objective_seconds=api["objective"](graph, c))
                                  for e, c in zip(GLOBAL_GREEDY_EXPONENTS, candidates)])


def scope_statistics(graph, scope, api):
    neighbors, factors = api["factors"](graph, scope)
    left = set(range(len(scope.replacements)))
    blocks = []
    for factor in sorted(factors, key=lambda members: (-len(members), members)):
        block = left.intersection(factor)
        if block:
            blocks.append(tuple(sorted(block)))
            left.difference_update(block)
    blocks.extend((v,) for v in sorted(left))
    upper = math.fsum(max(float(graph.weights[scope.replacements[i]]) for i in block) for block in blocks)
    loads = [1] * len(neighbors)
    for factor in factors:
        for v in factor:
            loads[v] = max(loads[v], len(factor))
    p1 = math.fsum(float(graph.weights[scope.replacements[i]]) / loads[i] for i in range(len(loads)))
    edges = scope.canonical_edges(graph)
    return dict(U_seconds=upper, P1_recovery_seconds=p1,
                local_edges=len(edges), local_vertices=len(scope.replacements),
                eligible_before_cap=scope.eligible_before_cap,
                cap_truncated=scope.eligible_before_cap > len(scope.replacements),
                local_cross_owner_edges=sum(graph.agents[scope.replacements[u]] !=
                                            graph.agents[scope.replacements[v]] for u, v in edges),
                distinct_local_owners=len(set(int(graph.agents[v]) for v in scope.replacements)),
                upper_clique_partition=[[int(scope.replacements[i]) for i in block] for block in blocks],
                U_is_structural_upper_bound_not_realized_gain=True)


def prepare_state(graph, original, action_seed, api):
    started = time.perf_counter()
    original = frozenset(original)
    initial = api["objective"](graph, original)
    best, best_value = original, initial
    cache = api["Cache"](graph, original, ())
    actions, coverage, warm = [], [], {}
    for cell_index, (action, information) in enumerate(api["cells"](graph, original, seed=action_seed)):
        record = dict(information, cell_index=cell_index)
        coverage.append(record)
        if action is None:
            continue
        scope_started = time.perf_counter()
        scope = cache.scope(cell_index, action, CAP)
        scope_ready = time.perf_counter()
        members = api["check"](graph, scope, cache.initial_known_warm(scope))
        warm_ready = time.perf_counter()
        statistics = scope_statistics(graph, scope, api)
        reward = api["objective"](graph, members)
        if reward > statistics["U_seconds"] + max(1e-8, abs(reward) * 1e-12):
            raise AssertionError("Actual warm exceeds original-graph clique-partition bound")
        candidate = scope.base | members
        actual_value = api["objective"](graph, candidate)
        if actual_value > best_value:
            best, best_value = candidate, actual_value
        descriptor = dict(action_index=cell_index, inserts=list(action.inserts), releases=list(action.releases),
                          base=sorted(scope.base), displaced=sorted(scope.displaced),
                          replacements=list(scope.replacements), q_seconds=scope.immediate_gain,
                          common_warm_members=sorted(members), common_L_seconds=reward,
                          common_prefix_scope_seconds=scope_ready - scope_started,
                          common_prefix_warm_and_validation_seconds=warm_ready - scope_ready,
                          **statistics)
        descriptor["scope_sha256"] = sha_json({key: descriptor[key] for key in
                                             ("inserts", "releases", "base", "displaced", "replacements", "q_seconds")})
        descriptor["common_greedy_candidates"] = [sorted(c) for c in
                                                   cache.preparatory_recoveries[(cell_index, CAP)]["candidates"]]
        actions.append((scope, descriptor))
        warm[cell_index] = members
        record.update(status="actual_common_prefix_executed", scope_sha256=descriptor["scope_sha256"])
    state = dict(original=sorted(original), original_value_seconds=initial,
                 best_selected=sorted(best), best_value_seconds=best_value,
                 best_gain_seconds=max(0.0, best_value - initial),
                 warm_by_action={str(index): sorted(values) for index, values in warm.items()},
                 spent_requests=[], common_prefix_seconds=time.perf_counter() - started,
                 current_warm_source="actually_executed_common_three_greeds_plus_incumbent")
    return actions, coverage, state


def snapshot_identity(state, actions):
    # Timing and cosmetic seed labels never create a new controller state.
    return sha_json(dict(original=state["original"], best_selected=state["best_selected"],
                         warm_by_action=state["warm_by_action"], spent_requests=state["spent_requests"],
                         action_descriptors=[record["scope_sha256"] for _, record in actions]))


class NativeCHILS:
    def __init__(self, binary, audit, api, temporary_root):
        self.binary = Path(binary).resolve()
        if not self.binary.is_file():
            raise FileNotFoundError("Actual CHILS executable is unavailable")
        self.sha256 = sha_file(self.binary)
        self.cpu_affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        self.audit = audit
        self.api = api
        self.temporary_root = Path(temporary_root).resolve()
        self.temporary_root.mkdir(parents=True, exist_ok=True)

    def run(self, graph, scope, warm, budget_ms):
        started = time.perf_counter()
        api = self.api
        warm = api["check"](graph, scope, warm)
        if not scope.replacements:
            return warm, dict(status="empty_scope_no_native_call", native_called=False,
                              native_budget_ms=budget_ms, validated_ready_seconds=time.perf_counter() - started,
                              full_return_seconds=time.perf_counter() - started,
                              earliest_validated_improving_membership_seconds=None)
        diagnostic = dict(native_called=False, native_budget_ms=budget_ms, seed=NATIVE_SEED,
                          population=1, threads=1, binary_sha256=self.sha256,
                          tick_seconds=TICK_SECONDS, native_slice_excludes_preparation_and_validation=True,
                          physical_hard_deadline_certified=False)
        diagnostic["inherited_cpu_affinity"] = self.cpu_affinity
        try:
            vertices = scope.replacements
            weights = np.asarray([float(graph.weights[v]) for v in vertices], dtype=np.float64)
            tick_weights = [int(round(float(w) / TICK_SECONDS)) for w in weights]
            if (any(w <= 0 for w in tick_weights) or len(vertices) > MAX_NATIVE_VERTICES or
                    sum(tick_weights) > MAX_NATIVE_TOTAL):
                raise ValueError("Microsecond native graph outside cleared positive p1 integer domain")
            rounding = [abs(w * TICK_SECONDS - float(original)) for w, original in zip(tick_weights, weights)]
            diagnostic.update(native_tick_weight_sum=sum(tick_weights),
                              tick_rounding_max_seconds=max(rounding),
                              tick_rounding_additive_bound_seconds=math.fsum(rounding),
                              rewards_rescored_with_original_float64_seconds=True)
            local = {v: i for i, v in enumerate(vertices)}
            edges = scope.canonical_edges(graph)
            neighbors = [[] for _ in vertices]
            for u, v in edges:
                neighbors[u].append(v + 1)
                neighbors[v].append(u + 1)
            with tempfile.TemporaryDirectory(prefix="p0-native-", dir=self.temporary_root) as folder:
                folder = Path(folder)
                graph_path, warm_path, output_path = folder / "input.graph", folder / "warm.txt", folder / "output.txt"
                graph_path.write_text("%d %d 10\n" % (len(vertices), len(edges)) + "\n".join(
                    " ".join(map(str, [weight] + sorted(peers))) for weight, peers in zip(tick_weights, neighbors)) + "\n",
                    encoding="ascii")
                warm_path.write_text("".join("%d\n" % (local[v] + 1) for v in sorted(warm)), encoding="ascii")
                preparation_ready = time.perf_counter()
                diagnostic["preparation_seconds"] = preparation_ready - started
                command = [str(self.binary), "-g", str(graph_path), "-i", str(warm_path),
                           "-o", str(output_path), "-p", "1", "-c", "1", "-t", str(budget_ms / 1000.0),
                           "-r", str(NATIVE_SEED)]
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_PROC_BIND="true")
                launch_started = time.perf_counter()
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, encoding="utf-8", errors="replace", env=env)
                diagnostic["native_called"] = True
                diagnostic["process_launch_seconds"] = time.perf_counter() - launch_started
                watchdog = False
                try:
                    stdout, stderr = process.communicate(timeout=budget_ms / 1000.0 + 5.0)
                except subprocess.TimeoutExpired:
                    watchdog = True
                    process.send_signal(signal.SIGTERM)
                    try:
                        stdout, stderr = process.communicate(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                process_ready = time.perf_counter()
                diagnostic.update(native_process_seconds=process_ready - launch_started,
                                  returncode=process.returncode, watchdog_signaled=watchdog,
                                  stdout=stdout, stderr=stderr, output_file_present=output_path.exists())
                parse_started = time.perf_counter()
                candidate = None
                if output_path.exists():
                    values = [int(v) for v in output_path.read_text(encoding="ascii").split()]
                    if len(values) != len(set(values)) or any(v < 1 or v > len(vertices) for v in values):
                        raise ValueError("Native output IDs must be distinct original local 1-based integers")
                    candidate = frozenset(vertices[v - 1] for v in values)
                diagnostic["parse_seconds"] = time.perf_counter() - parse_started
                validate_started = time.perf_counter()
                if candidate is not None:
                    candidate = api["check"](graph, scope, candidate)
                    candidate_reward = api["objective"](graph, candidate)
                    warm_reward = api["objective"](graph, warm)
                    if candidate_reward > warm_reward:
                        recovered = candidate
                    else:
                        recovered = warm
                    diagnostic.update(native_output_valid=True, native_recovery_members=sorted(candidate),
                                      native_output_recovery_seconds=candidate_reward,
                                      retained_warm=(recovered == warm))
                else:
                    recovered = warm
                    diagnostic.update(native_output_valid=False, native_recovery_members=None, retained_warm=True)
                full = scope.base | recovered
                if not api["feasible"](graph, full):
                    raise AssertionError("Recovery local validity did not imply complete fixed-base validity")
                diagnostic["full_schedule_reward_seconds"] = api["objective"](graph, full)
                ready = time.perf_counter()
                diagnostic.update(validation_and_rescore_seconds=ready - validate_started,
                                  validated_ready_seconds=ready - started,
                                  earliest_validated_improving_membership_seconds=(ready - started if recovered != warm else None),
                                  first_materialized_native_membership_observed_at_return=True,
                                  solver_internal_claims_are_not_earliest_verified_membership=True,
                                  status=("returned_verified" if candidate is not None else "native_no_output_warm_retained"))
                if watchdog or process.returncode != 0:
                    diagnostic["status"] = "native_failed_or_watchdog_with_verified_fallback"
        except Exception as error:
            recovered = warm
            diagnostic.update(status="native_request_failed_warm_retained",
                              failure=dict(type=type(error).__name__, message=str(error)),
                              validated_ready_seconds=time.perf_counter() - started,
                              earliest_validated_improving_membership_seconds=None)
        diagnostic["full_return_seconds"] = time.perf_counter() - started
        return recovered, diagnostic


def request_row(graph, scope, descriptor, state, budget_ms, repeat, native, api):
    started = time.perf_counter()
    warm = frozenset(state["warm_by_action"][str(scope.action_index)])
    warm_value = api["objective"](graph, warm)
    identity = sha_json(state)
    already_spent = [scope.action_index, CAP, budget_ms] in state["spent_requests"]
    if already_spent:
        recovered, diagnostic = warm, dict(status="already_spent_in_actual_history", native_called=False,
            native_budget_ms=budget_ms, full_return_seconds=0.0,
            earliest_validated_improving_membership_seconds=None)
    else:
        recovered, diagnostic = native.run(graph, scope, warm, budget_ms)
    validation_started = time.perf_counter()
    recovered = api["check"](graph, scope, recovered)
    recovery = api["objective"](graph, recovered)
    full_members = scope.base | recovered
    full_value = api["objective"](graph, full_members)
    signed = full_value - state["original_value_seconds"]
    algebra = scope.immediate_gain + recovery
    if not math.isclose(signed, algebra, rel_tol=1e-12, abs_tol=1e-7):
        raise AssertionError("Complete original objective differs from q + recovered reward")
    if identity != sha_json(state):
        raise AssertionError("A peer result mutated its immutable actual snapshot")
    ready = time.perf_counter()
    row = dict(snapshot_sha256=identity, action_index=scope.action_index,
               scope_sha256=descriptor["scope_sha256"], workpoint_ms=budget_ms, repeat=repeat,
               cap=CAP, q_seconds=scope.immediate_gain, L_seconds=warm_value,
               U_seconds=descriptor["U_seconds"], P1_recovery_seconds=descriptor["P1_recovery_seconds"],
               snapshot_best_gain_seconds=state["best_gain_seconds"],
               actual_warm_members=sorted(warm), returned_recovery_members=sorted(recovered),
               complete_feasible_members=sorted(full_members), complete_value_seconds=full_value,
               recovery_value_seconds=recovery, q_plus_recovery_seconds=algebra,
               signed_original_gain_seconds=signed,
               local_delta=(recovery - warm_value) / max(1.0, warm_value),
               beyond_snapshot_gain_seconds=max(0.0, algebra - state["best_gain_seconds"]),
               beyond_snapshot_global_fraction=max(0.0, algebra - state["best_gain_seconds"]) /
                                                max(1.0, state["original_value_seconds"]),
               complete_membership_valid=True,
               available_for_allocation=not already_spent,
               actual_finite_supervision_valid=(diagnostic["native_called"] and
                                                diagnostic.get("native_output_valid", False)),
               failed=(diagnostic["status"].startswith("native_failed") or "failure" in diagnostic or
                       diagnostic["status"] == "native_no_output_warm_retained"),
               zero_extra_gain=(recovery <= warm_value + 1e-9),
               extra_validation_and_rescore_seconds=ready - validation_started,
               complete_conditional_return_seconds=ready - started,
               common_prefix_gain_not_learning_gain=True, diagnostics=diagnostic)
    # The snapshot warm is already held/verified before the request (time 0).
    # A new native good set is not available until the final complete check.
    row["earliest_verified_more_than_snapshot_gain_seconds"] = (
        ready - started if row["beyond_snapshot_gain_seconds"] > 1e-9 else None)
    row["snapshot_verified_incumbent_available_seconds"] = 0.0
    return row


def history_snapshot(graph, actions, state, native, api, trajectory):
    cloned = json.loads(json.dumps(state))
    if not actions:
        return cloned, dict(status="no_prepared_actions", actual_call=False)
    scope, descriptor = actions[trajectory % len(actions)]
    row = request_row(graph, scope, descriptor, state, 200, 0, native, api)
    key = [scope.action_index, CAP, 200]
    cloned["spent_requests"].append(key)
    recovered = row["returned_recovery_members"]
    if row["recovery_value_seconds"] > api["objective"](graph, cloned["warm_by_action"][str(scope.action_index)]):
        cloned["warm_by_action"][str(scope.action_index)] = recovered
    if row["complete_value_seconds"] > cloned["best_value_seconds"]:
        cloned["best_selected"] = row["complete_feasible_members"]
        cloned["best_value_seconds"] = row["complete_value_seconds"]
        cloned["best_gain_seconds"] = max(0.0, cloned["best_value_seconds"] - cloned["original_value_seconds"])
    cloned["current_warm_source"] = "common_prefix_plus_one_actually_executed_predeclared_200ms_history_call"
    return cloned, row


def summarize_graph(groups, metadata, budgets, target_states):
    rows = [row for group in groups for row in group["requests"]]
    opportunity = [any(row["local_delta"] >= 0.01 and row["beyond_snapshot_gain_seconds"] > 1e-9
                       for row in group["requests"]) for group in groups]
    budget_difference = []
    for group in groups:
        by_action_repeat = {}
        for row in group["requests"]:
            by_action_repeat.setdefault((row["action_index"], row["repeat"]), {})[row["workpoint_ms"]] = row
        comparisons = {}
        if 200 in budgets and 1000 in budgets:
            for (action, repeat), points in by_action_repeat.items():
                if (200 in points and 1000 in points and not points[200]["failed"] and not points[1000]["failed"] and
                        points[200]["available_for_allocation"] and points[1000]["available_for_allocation"] and
                        points[200]["actual_finite_supervision_valid"] and points[1000]["actual_finite_supervision_valid"]):
                    comparisons.setdefault(action, []).append(points[1000]["recovery_value_seconds"] -
                                                               points[200]["recovery_value_seconds"])
        budget_difference.append(any(len(values) >= 2 and
                                     (all(v > 1e-7 for v in values) or all(v < -1e-7 for v in values))
                                     for values in comparisons.values()))
    return dict(schema=SCHEMA, graph=metadata, status="ACTUAL_P0_PROBES_COMPLETE_NO_TRAINING",
                target_unique_states=target_states, actual_unique_controller_states=len(groups),
                distinct_original_schedule_memberships=len({sha_json(g["snapshot"]["original"]) for g in groups}),
                distinct_best_schedule_memberships=len({sha_json(g["snapshot"]["best_selected"]) for g in groups}),
                state_count_is_not_independent_physical_source_count=True,
                actual_request_rows=len(rows), actual_native_calls=sum(r["diagnostics"]["native_called"] for r in rows),
                failed_rows=sum(r["failed"] for r in rows), zero_extra_gain_rows=sum(r["zero_extra_gain"] for r in rows),
                G1_states_with_local_at_least_one_percent_and_net_beyond_snapshot=sum(opportunity),
                G1_fraction=(sum(opportunity) / len(groups) if groups else 0.0),
                G1_thirty_percent_is_development_reference_not_sample_filter=True,
                G3_states_with_same_direction_repeated_200_1000ms_difference=sum(budget_difference),
                G3_repeatable_fraction=(sum(budget_difference) / len(groups) if groups else 0.0),
                G2_graph_information_advantage_established=False,
                actual_max_beyond_snapshot_seconds=max((r["beyond_snapshot_gain_seconds"] for r in rows), default=0.0),
                actual_median_complete_conditional_seconds=float(np.median(
                    [r["complete_conditional_return_seconds"] for r in rows])) if rows else None,
                budgets_ms=budgets, source_group_split_preserved=True,
                no_model_training=True, no_old_data_or_calibration=True,
                strict_end_to_end_deadline_policy_test=False, learning_advantage_established=False)


def write_rows_csv(path, groups):
    columns = ("snapshot_id", "state_kind", "action_index", "workpoint_ms", "repeat", "q_seconds", "L_seconds",
               "U_seconds", "P1_recovery_seconds", "snapshot_best_gain_seconds", "recovery_value_seconds",
               "q_plus_recovery_seconds", "local_delta", "beyond_snapshot_gain_seconds",
               "beyond_snapshot_global_fraction", "complete_conditional_return_seconds", "failed", "zero_extra_gain",
               "earliest_verified_more_than_snapshot_gain_seconds")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for group in groups:
            for row in group["requests"]:
                writer.writerow({key: group[key] if key in ("snapshot_id", "state_kind") else row[key]
                                 for key in columns})


def collect_graph(path, out_root, native, api, args):
    graph, metadata = load_graph(path, api)
    destination = Path(out_root).resolve() / metadata["graph_id"]
    destination.mkdir(parents=True, exist_ok=True)
    protocol = dict(schema=SCHEMA, graph=metadata, code_sha256=sha_file(__file__),
                    reference_sha256=api["reference_sha256"], numeric_audit=native.audit,
                    native_binary_sha256=native.sha256, seed=args.seed,
                    native_seed=NATIVE_SEED, budgets_ms=args.budgets_ms, repeats=args.repeats,
                    target_states=args.states, cap=CAP, peer_states_never_mutated=True,
                    timing_contract="native search workpoint plus all measured export/launch/parse/full validation/return cost",
                    common_prefix_and_native_executor_gains_not_learning_gains=True)
    protocol_path = destination / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise FileExistsError("Existing P0 output belongs to another graph/code/protocol")
    else:
        write_json(protocol_path, protocol)
    original, incumbent_record = global_incumbent(graph, api)
    write_json(destination / "initial_global_greedy.json", incumbent_record)
    groups = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(destination.glob("snapshot_*.json"))]
    seen = {g["controller_state_identity"] for g in groups}
    groups = sorted(groups, key=lambda g: int(g["snapshot_id"].split("-")[-1]))
    # Six independently predeclared proposal seeds give two real states each.
    # Additional attempts only fill duplicates, never filter by observed reward.
    max_trajectories = max(6, args.states * 3)
    for trajectory in range(max_trajectories):
        if len(groups) >= args.states:
            break
        seed = args.seed + trajectory * 1009
        actions, coverage, base_state = prepare_state(graph, original, seed, api)
        history_state, history = history_snapshot(graph, actions, base_state, native, api, trajectory)
        write_json(destination / ("trajectory_%03d.json" % trajectory), dict(
            trajectory=trajectory, action_seed=seed, coverage=coverage,
            scopes=[d for _, d in actions], common_prefix_state=base_state,
            history_state=history_state, actual_history_call=history))
        for kind, state in (("after_common_greedy_prefix", base_state), ("after_one_actual_history_call", history_state)):
            if len(groups) >= args.states:
                break
            identity = snapshot_identity(state, actions)
            if identity in seen:
                continue
            group = dict(snapshot_id="s-%03d" % len(groups), state_kind=kind, trajectory=trajectory,
                         controller_state_identity=identity, snapshot=state, action_seed=seed,
                         scopes=[d for _, d in actions], coverage=coverage, requests=[])
            frozen = sha_json(state)
            for scope, descriptor in actions:
                for budget_ms in args.budgets_ms:
                    for repeat in range(args.repeats):
                        row = request_row(graph, scope, descriptor, state, budget_ms, repeat, native, api)
                        group["requests"].append(row)
            if sha_json(state) != frozen:
                raise AssertionError("Immutable snapshot changed during peer enumeration")
            write_json(destination / ("snapshot_%03d.json" % len(groups)), group)
            groups.append(group)
            seen.add(identity)
            print(json.dumps(dict(graph_id=graph.name, complete_states=len(groups), target_states=args.states,
                                  actual_rows=len(group["requests"]), state_kind=kind)), flush=True)
    summary = summarize_graph(groups, metadata, args.budgets_ms, args.states)
    if len(groups) < args.states:
        summary["state_target_shortfall_reason"] = "deduplicated_actual_states_exhausted_predeclared_attempts"
    write_rows_csv(destination / "requests.csv", groups)
    write_json(destination / "summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--chils", required=True)
    parser.add_argument("--chils-source", required=True)
    parser.add_argument("--graph", action="append", default=[])
    parser.add_argument("--graphs-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--budgets-ms", nargs="+", type=int, default=[200, 1000])
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not args.graph and not args.graphs_dir:
        parser.error("--graph or --graphs-dir is required")
    if args.states < 1 or args.repeats < 1 or args.states > 144:
        parser.error("Positive finite state and repeat limits required")
    if len(args.budgets_ms) != len(set(args.budgets_ms)) or any(v not in (10, 50, 200, 1000) for v in args.budgets_ms):
        parser.error("Native workpoints must be distinct members of 10/50/200/1000 ms")
    return args


def main():
    args = parse_args()
    api = load_runtime(args.runtime_root)
    audit = native_source_audit(args.chils_source)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    native = NativeCHILS(args.chils, audit, api, out / "temporary_native_calls")
    paths = [Path(p).resolve() for p in args.graph]
    if args.graphs_dir:
        paths.extend(sorted(Path(args.graphs_dir).resolve().rglob("*.npz")))
    paths = list(dict.fromkeys(paths))
    results = []
    for path in paths:
        results.append(collect_graph(path, out, native, api, args))
    write_json(out / "batch_summary.json", dict(schema=SCHEMA, graphs=results,
               actual_graph_count=len(results), no_training=True, failure_and_zero_rows_retained=True))
    print(json.dumps(dict(status="P0_COMPLETE", graph_count=len(results), out=str(out))), flush=True)


if __name__ == "__main__":
    main()
