"""Isolated whole-STK-graph published controls under the frozen caller deadline.

Linux only. Graph/runtime/solver libraries and the identical P0 full-graph
greedy incumbent are resident setup. Export, model assembly, process launch,
search, parsing, child validation, IPC and caller revalidation are paid in D.
CHILS receives the original incumbent; SciPy 1.10.1 MILP has no x0 interface.
Neither method reads recovery-scope labels, model predictions or P2 outcomes.

CHILS requires an independently produced wholegraph source/numeric/native gate
receipt. This module never manufactures that receipt or reuses P0's cap256
integer clearance. Tick conversion is microsecond serialization only; every
delivered schedule is rescored with unchanged original float64 durations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
import warnings

# These are set before importing numerical/runtime libraries, also inherited by
# the forked worker and the official native executable.
THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
os.environ.update(THREAD_ENV)

import numpy as np
import p0_recovery_probes as p0

SCHEMA = "joint_recovery_stk_published_wholegraph_actual_v1"
GATE_SCHEMA = "joint_recovery_stk_chils_wholegraph_numeric_gate_v1"
CHILS_COMMIT = "515952724cd3dcc6c4365a340ecf0f1da782119a"
TICK_SECONDS = 1e-6
SEED = 17
SEARCH_FRACTION = .70
RETURN_RESERVE_FRACTION = .10
MIN_SEARCH_SECONDS = .001


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def require_frozen_deadline(path, seconds):
    record = read_json(path)
    if (record.get("status") != "BUDGETS_FROZEN_FROM_DEVELOPMENT_ACTUAL_CALLERS" or
            record.get("calibration_sources") != [4, 5] or
            record.get("action_seed") != SEED or record.get("test_outcomes_used") is not False):
        raise ValueError("The unchanged actual development caller-budget receipt is required")
    budgets = record.get("budgets", [])
    if [row.get("budget_id") for row in budgets] != ["wide", "half", "quarter"]:
        raise ValueError("Frozen wide/half/quarter budget entries required")
    wide = finite(budgets[0].get("deadline_seconds"))
    if (wide is None or wide <= 0 or budgets[1].get("deadline_seconds") != wide * .5 or
            budgets[2].get("deadline_seconds") != wide * .25):
        raise ValueError("Frozen budget ratios changed")
    matched = [row for row in budgets if row["deadline_seconds"] == seconds]
    if len(matched) != 1:
        raise ValueError("Use the exact frozen float, not a rounded display deadline")
    return dict(path=str(Path(path).resolve()), sha256=p0.sha_file(path),
                budget_id=matched[0]["budget_id"], deadline_seconds=seconds,
                recalibration_performed=False)


def tick_domain(graph, metadata):
    ticks = [int(round(float(w) / TICK_SECONDS)) for w in graph.weights]
    total = sum(ticks)
    maximum = max(ticks, default=0)
    bound = 8 * graph.n * (total + 129)  # Python integers, never int64 overflow.
    if (graph.n < 1 or graph.n >= 1 << 31 or 2 * metadata["edges"] >= 1 << 31 or
            any(w <= 0 for w in ticks) or total > 1 << 53 or maximum > 1 << 53 or
            bound >= 1 << 63):
        raise ValueError("Wholegraph microsecond serialization outside the declared exact-double/int64 domain")
    errors = [abs(tick * TICK_SECONDS - float(w)) for tick, w in zip(ticks, graph.weights)]
    return dict(tick_seconds=TICK_SECONDS, tick_weight_sum=total,
                max_tick_weight=maximum, safe_signed64_bound=bound,
                exact_double_limit=1 << 53, population=1, threads=1, seed=SEED,
                tick_rounding_max_seconds=max(errors),
                tick_rounding_additive_bound_seconds=math.fsum(errors),
                original_seconds_used_for_acceptance_and_scoring=True)


def verify_chils_gate(args, graph, metadata, domain):
    if not args.numeric_gate or not args.chils or not args.chils_source:
        raise ValueError("CHILS requires binary/source paths and an independent wholegraph numeric gate")
    gate = read_json(args.numeric_gate)
    if gate.get("schema") != GATE_SCHEMA or gate.get("status") != "WHOLEGRAPH_NUMERIC_GATE_PASS":
        raise ValueError("Wholegraph native gate has not passed; retain this as an unavailable baseline")
    checks = gate.get("checks")
    if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
        raise ValueError("All independent wholegraph gate checks must be explicit true booleans")
    bound_graph = gate.get("graph", {})
    for key in ("npz_sha256", "vertices", "edges"):
        if bound_graph.get(key) != metadata[key]:
            raise ValueError("Wholegraph numeric gate does not bind this original graph: " + key)
    for key in ("tick_seconds", "tick_weight_sum", "max_tick_weight", "safe_signed64_bound",
                "exact_double_limit", "population", "threads", "seed"):
        if gate.get("native", {}).get(key) != domain[key]:
            raise ValueError("Wholegraph numeric gate domain differs: " + key)
    native = gate.get("chils", {})
    if native.get("source_commit", gate.get("source_commit")) != CHILS_COMMIT:
        raise ValueError("Wholegraph gate must bind the frozen official CHILS source commit")
    binary = Path(args.chils).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError("Official CHILS executable unavailable")
    binary_sha = p0.sha_file(binary)
    if native.get("binary_sha256") != binary_sha:
        raise ValueError("Official CHILS binary differs from the independent gate")
    source = Path(args.chils_source).resolve()
    files = set()
    for directory in (source / "src", source / "include"):
        if directory.exists():
            files.update(path for path in directory.rglob("*")
                         if path.is_file() and path.suffix in (".c", ".h"))
    declared = native.get("source_files_sha256")
    actual = {path.relative_to(source).as_posix(): p0.sha_file(path) for path in sorted(files)}
    if not actual or not isinstance(declared, dict) or declared != actual:
        raise ValueError("Complete CHILS src/include .c/.h source identity differs from the gate")
    return dict(path=str(Path(args.numeric_gate).resolve()), sha256=p0.sha_file(args.numeric_gate),
                status=gate["status"], checks=checks, source_commit=CHILS_COMMIT,
                binary_sha256=binary_sha, source_files_sha256=actual,
                old_cap256_numeric_clearance_used=False,
                standalone_gate_does_not_certify_a_physical_deadline=True)


def resident_highs():
    import scipy
    import scipy.optimize._milp as wrapper
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    if scipy.__version__ != "1.10.1":
        raise ValueError("The declared resident HiGHS reference requires SciPy 1.10.1")
    package = Path(scipy.__file__).resolve().parent
    libraries = {path.relative_to(package).as_posix(): p0.sha_file(path)
                 for path in (package / "optimize").rglob("*")
                 if path.is_file() and path.suffix in (".so", ".pyd", ".dll") and
                 "highs" in str(path).lower()}
    if not libraries:
        raise ValueError("Installed native HiGHS library identity is unavailable")
    receipt = dict(scipy_version=scipy.__version__, numpy_version=np.__version__,
                   milp_source_sha256=p0.sha_file(wrapper.__file__), native_highs_sha256=libraries,
                   native_warmstart=False, x0_argument_available=False,
                   no_model_matrix_preassembled_in_resident_setup=True)
    return dict(Bounds=Bounds, LinearConstraint=LinearConstraint, milp=milp,
                coo_matrix=coo_matrix), receipt


def check_clock(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("Deadline expired before complete schedule was available")


def complete_check(graph, raw_members, expected_value, deadline, api):
    check_clock(deadline)
    members = tuple(raw_members)
    if (len(members) != len(set(members)) or
            any(type(v) is not int or v < 0 or v >= graph.n for v in members)):
        raise ValueError("Unique exact original zero-based vertex IDs required")
    selected = frozenset(members)
    for index, vertex in enumerate(members):
        if index % 16 == 0:
            check_clock(deadline)
        if graph.adjacency[vertex] & selected:
            raise ValueError("Full original-graph membership is infeasible")
    value = api["objective"](graph, selected)
    if expected_value is not None and not math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError("Original-duration objective differs from child report")
    check_clock(deadline)
    return selected, value


def search_admission(deadline, seconds):
    reserve = RETURN_RESERVE_FRACTION * seconds
    remaining = deadline - time.perf_counter()
    budget = min(SEARCH_FRACTION * seconds, remaining - reserve)
    if budget < MIN_SEARCH_SECONDS:
        raise TimeoutError("Wholegraph preparation left insufficient declared return reserve")
    return budget, remaining, reserve


def emit(channel, started, kind, **fields):
    channel.send(dict(kind=kind, worker_elapsed_seconds=time.perf_counter() - started, **fields))


def chils_search(graph, original, args, api, channel, started, deadline, diagnostic):
    preparation_started = time.perf_counter()
    folder = Path(args.out) / "wholegraph_native"
    folder.mkdir()  # This fresh wholegraph export is paid inside D.
    graph_path, warm_path, output_path = folder / "input.graph", folder / "original.txt", folder / "output.txt"
    ticks = [int(round(float(w) / TICK_SECONDS)) for w in graph.weights]
    edge_count = sum(len(row) for row in graph.adjacency) // 2
    with graph_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("%d %d 10\n" % (graph.n, edge_count))
        for index, (weight, peers) in enumerate(zip(ticks, graph.adjacency)):
            if index % 128 == 0:
                check_clock(deadline)
            stream.write(" ".join(map(str, [weight] + [v + 1 for v in sorted(peers)])) + "\n")
    with warm_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("".join("%d\n" % (v + 1) for v in sorted(original)))
    diagnostic.update(wholegraph_export_sha256=p0.sha_file(graph_path),
                      actual_native_warmstart_sha256=p0.sha_file(warm_path),
                      actual_native_warmstart_members=sorted(original), native_warmstart=True,
                      export_vertices=graph.n, export_edges=edge_count,
                      original_vertex_mapping="native_ID = original_zero_based_ID + 1",
                      export_preparation_seconds=time.perf_counter() - preparation_started)
    budget, remaining, reserve = search_admission(deadline, args.deadline_seconds)
    command = [str(Path(args.chils).resolve()), "-g", str(graph_path), "-i", str(warm_path),
               "-o", str(output_path), "-p", "1", "-c", "1", "-t", repr(budget), "-r", str(SEED)]
    diagnostic.update(command=command, native_search_budget_seconds=budget,
                      remaining_before_native_launch_seconds=remaining,
                      reserved_return_seconds=reserve)
    launch_started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace", env=os.environ.copy())
    diagnostic.update(process_launch_seconds=time.perf_counter() - launch_started, native_called=True)
    emit(channel, started, "native_started", diagnostics=dict(diagnostic))
    try:
        # Wholegraph process parsing/initialization also consumes the caller's
        # remaining time. The caller can independently kill the owned group.
        stdout, stderr = process.communicate(timeout=max(.0001, deadline - reserve - time.perf_counter()))
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        (folder / "stdout.txt").write_text(stdout, encoding="utf-8")
        (folder / "stderr.txt").write_text(stderr, encoding="utf-8")
        diagnostic.update(native_process_seconds=time.perf_counter() - launch_started,
                          native_returncode=process.returncode, native_watchdog_timeout=True)
        raise TimeoutError("Native process did not return before its reserved parsing/validation interval")
    diagnostic.update(native_process_seconds=time.perf_counter() - launch_started,
                      native_returncode=process.returncode, native_watchdog_timeout=False,
                      stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
                      stderr_sha256=hashlib.sha256(stderr.encode()).hexdigest())
    (folder / "stdout.txt").write_text(stdout, encoding="utf-8")
    (folder / "stderr.txt").write_text(stderr, encoding="utf-8")
    check_clock(deadline)
    if process.returncode != 0:
        raise RuntimeError("Official CHILS exited unsuccessfully: " + str(process.returncode))
    if not output_path.is_file():
        diagnostic["status"] = "native_no_output_original_retained"
        return None
    parse_started = time.perf_counter()
    ids = [int(token) for token in output_path.read_text(encoding="ascii").split()]
    if len(ids) != len(set(ids)) or any(v < 1 or v > graph.n for v in ids):
        raise ValueError("CHILS returned invalid complete original 1-based vertex IDs")
    diagnostic.update(parse_seconds=time.perf_counter() - parse_started,
                      native_output_sha256=p0.sha_file(output_path), native_output_present=True)
    return [v - 1 for v in ids]


def highs_search(graph, args, solver_api, channel, started, deadline, diagnostic):
    preparation_started = time.perf_counter()
    n = graph.n
    # No induced scope, cap, presaved METIS or resident sparse matrix is used.
    # Every original incompatibility generates exactly one x_u + x_v <= 1.
    edges = []
    for u, peers in enumerate(graph.adjacency):
        if u % 128 == 0:
            check_clock(deadline)
        edges.extend((u, v) for v in sorted(peers) if u < v)
    columns = np.asarray(edges, dtype=np.int32).reshape(-1)
    rows = np.repeat(np.arange(len(edges), dtype=np.int32), 2)
    matrix = solver_api["coo_matrix"]((np.ones(len(columns)), (rows, columns)),
                                      shape=(len(edges), n)).tocsc()
    matrix.indices = matrix.indices.astype(np.int32)
    matrix.indptr = matrix.indptr.astype(np.int32)
    ticks = np.asarray([int(round(float(w) / TICK_SECONDS)) for w in graph.weights], dtype=np.int64)
    objective = -ticks.astype(np.float64)
    integrality = np.ones(n, dtype=np.int32)
    bounds = solver_api["Bounds"](np.zeros(n), np.ones(n))
    constraints = solver_api["LinearConstraint"](matrix, -np.inf, 1.)
    diagnostic.update(model_construction_seconds=time.perf_counter() - preparation_started,
                      model_vertices=n, model_edge_constraints=len(edges), native_warmstart=False,
                      common_incumbent_is_external_fallback_only=True)
    budget, remaining, reserve = search_admission(deadline, args.deadline_seconds)
    options = dict(presolve=True, mip_rel_gap=0., threads=1, random_seed=SEED, time_limit=budget)
    diagnostic.update(native_search_budget_seconds=budget, options=options,
                      remaining_before_native_launch_seconds=remaining,
                      reserved_return_seconds=reserve, native_called=True)
    emit(channel, started, "native_started", diagnostics=dict(diagnostic))
    search_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = solver_api["milp"](c=objective, integrality=integrality, bounds=bounds,
                                    constraints=constraints, options=options)
    diagnostic.update(native_process_seconds=time.perf_counter() - search_started,
                      solver_warnings=[str(row.message) for row in captured],
                      highs_status=int(result.status), highs_message=str(result.message),
                      numerical_optimum_reported=int(result.status) == 0,
                      highs_node_count=int(getattr(result, "mip_node_count", 0) or 0),
                      highs_gap=finite(getattr(result, "mip_gap", None)),
                      highs_dual_bound_ticks=finite(getattr(result, "mip_dual_bound", None)),
                      dual_bound_is_descriptive_numerical_not_original_duration_certificate=True)
    check_clock(deadline)
    if int(result.status) not in (0, 1):
        raise RuntimeError("HiGHS did not finish with an optimal/time-limited incumbent status")
    if result.x is None:
        diagnostic["status"] = "native_no_membership_original_retained"
        return None
    parse_started = time.perf_counter()
    x = np.asarray(result.x, dtype=np.float64)
    if (x.shape != (n,) or not np.isfinite(x).all() or
            np.any(np.abs(x - np.rint(x)) > 1e-5) or np.any(x < -1e-5) or np.any(x > 1. + 1e-5)):
        raise ValueError("HiGHS did not return a finite full binary integral membership")
    members = np.flatnonzero(np.rint(x) == 1).tolist()
    diagnostic.update(parse_seconds=time.perf_counter() - parse_started, native_output_present=True)
    return members


def worker(graph, original, args, api, solver_api, channel, started, deadline):
    diagnostic = dict(method=args.method, seed=SEED, configured_threads=1,
                      native_called=False, complete_original_graph=True,
                      scope_cap=None, original_vertex_count=graph.n,
                      search_fraction_of_frozen_D=SEARCH_FRACTION,
                      caller_return_reserve_fraction_of_frozen_D=RETURN_RESERVE_FRACTION)
    try:
        os.setsid()  # Its native descendants belong only to this fresh job.
        emit(channel, started, "worker_started", owned_process_group=os.getpid())
        if args.method == "CHILS-p1":
            members = chils_search(graph, original, args, api, channel, started, deadline, diagnostic)
        else:
            members = highs_search(graph, args, solver_api, channel, started, deadline, diagnostic)
        if members is not None:
            checked_started = time.perf_counter()
            selected, value = complete_check(graph, members, None, deadline, api)
            diagnostic.update(worker_full_validation_seconds=time.perf_counter() - checked_started,
                              raw_candidate_value_seconds=value, raw_candidate_members=sorted(selected),
                              status="returned_verified_complete_membership")
            emit(channel, started, "candidate", source="published_wholegraph_native",
                 members=sorted(selected), value_seconds=value, diagnostics=dict(diagnostic))
        emit(channel, started, "finished", actual_calls=[dict(diagnostics=diagnostic)],
             no_extra_common_warm_prefix_executed=True)
    except BaseException as error:
        try:
            diagnostic.update(status="wholegraph_failed_original_retained",
                              failure=dict(type=type(error).__name__, message=str(error)))
            emit(channel, started, "failed", actual_calls=[dict(diagnostics=diagnostic)],
                 failure=diagnostic["failure"])
        except (BrokenPipeError, EOFError, OSError):
            pass  # Caller has already returned its independently held S.
    finally:
        channel.close()


def cleanup_owned_worker(process, receiver, group_was_confirmed, record):
    began = time.perf_counter()
    try:
        group_is_owned = group_was_confirmed
        if process.is_alive() and not group_is_owned:
            try:
                group_is_owned = os.getpgid(process.pid) == process.pid
            except ProcessLookupError:
                pass
        if group_is_owned:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.is_alive():
            process.terminate()
        process.join(timeout=.25)
        if group_is_owned:
            # Kill surviving native descendants even if the Python worker
            # already exited. This group was created exclusively by this job.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.is_alive():
            process.kill()
        process.join(timeout=.25)
        record.update(worker_reaped=not process.is_alive(), owned_group_cleanup_attempted=group_is_owned)
    except BaseException as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
    finally:
        receiver.close()
        record["cleanup_seconds_after_return_started"] = time.perf_counter() - began


def run_wholegraph(graph, original, args, api, solver_api):
    context = mp.get_context("fork")
    started = time.perf_counter()
    deadline = started + args.deadline_seconds
    best = frozenset(original)
    initial_value = api["objective"](graph, best)
    best_value = initial_value
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(graph, original, args, api, solver_api,
                                                   sender, started, deadline))
    process.daemon = True
    events, stop_reason, group_confirmed = [], "deadline_reached", False
    process.start()
    sender.close()
    while time.perf_counter() < deadline:
        if not receiver.poll(max(0., deadline - time.perf_counter())):
            break
        try:
            event = receiver.recv()
        except EOFError:
            stop_reason = "failed"
            events.append(dict(kind="failed", failure=dict(type="WorkerClosedWithoutReceipt")))
            break
        event["caller_received_seconds"] = time.perf_counter() - started
        if event["kind"] == "worker_started":
            group_confirmed = event.get("owned_process_group") == process.pid
        elif event["kind"] == "candidate":
            validation_started = time.perf_counter()
            try:
                checked = complete_check(graph, event["members"], event["value_seconds"], deadline, api)
                event["admitted_before_deadline"] = True
                if checked[1] > best_value:
                    best, best_value = checked
            except TimeoutError:
                event["admitted_before_deadline"] = False
            except Exception as error:
                event["admitted_before_deadline"] = False
                event["caller_validation_failure"] = dict(type=type(error).__name__, message=str(error))
                stop_reason = "failed"
            event["caller_complete_validation_seconds"] = time.perf_counter() - validation_started
            event["caller_validated_ready_seconds"] = time.perf_counter() - started
        elif event["kind"] in ("finished", "failed"):
            stop_reason = event["kind"]
            events.append(event)
            break
        events.append(event)
        if stop_reason == "failed":
            break
    cleanup_record = {}
    cleanup = threading.Thread(target=cleanup_owned_worker,
                               args=(process, receiver, group_confirmed, cleanup_record), daemon=True)
    cleanup.start()
    receipt = dict(schema=SCHEMA, method=args.method, policy=args.method, fit_seed=SEED,
                   fit_seed_is_compatibility_field_no_fitted_model=True, action_seed=SEED,
                   deadline_seconds=args.deadline_seconds, original_members=sorted(original),
                   selected_members=sorted(best), initial_value_seconds=initial_value,
                   returned_value_seconds=best_value, actual_gain_seconds=max(0., best_value - initial_value),
                   paid_shared_prefix_gain_seconds=0.,
                   actual_gain_beyond_paid_shared_prefix_seconds=max(0., best_value - initial_value),
                   paid_prefix_not_applicable_no_scope_prefix_executed=True,
                   stop_reason=stop_reason, events=events, max_native_calls=1,
                   native_search_budget_rule="min(0.70 * D, remaining_at_launch - 0.10 * D)",
                   original_incumbent_held_by_caller=True,
                   native_warmstart=args.method == "CHILS-p1",
                   export_model_launch_search_parse_childcheck_IPC_callercheck_inside_D=True,
                   complete_original_graph=True, recovery_scope_cap=None,
                   source_group_independence_not_implied_by_additional_configurations=True,
                   physical_hard_deadline_guaranteed=False,
                   candidate_internal_availability_different_from_caller_receipt=True,
                   cleanup_after_return_does_not_make_late_candidate_available=True)
    receipt["controller_return_sample_seconds"] = time.perf_counter() - started
    return receipt, cleanup, cleanup_record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--method", choices=("CHILS-p1", "HiGHS-MILP"), required=True)
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--budgets-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chils", type=Path)
    parser.add_argument("--chils-source", type=Path)
    parser.add_argument("--numeric-gate", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.deadline_seconds) or args.deadline_seconds <= 0:
        parser.error("Positive finite frozen caller deadline required")
    if os.name != "posix":
        parser.error("This actual caller-owned fork interface runs only on Linux")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Wholegraph output directory must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    resident_started = time.perf_counter()
    budget_receipt = require_frozen_deadline(args.budgets_file, args.deadline_seconds)
    api = p0.load_runtime(args.runtime_root)
    graph, metadata = p0.load_graph(args.graph, api)
    original, initial_record = p0.global_incumbent(graph, api)
    initial_mask = np.zeros(graph.n, dtype=np.int8)
    initial_mask[list(original)] = 1
    initial_mask_sha256 = hashlib.sha256(initial_mask.tobytes()).hexdigest()
    domain = tick_domain(graph, metadata)
    solver_api = None
    if args.method == "CHILS-p1":
        solver_receipt = verify_chils_gate(args, graph, metadata, domain)
    else:
        solver_api, solver_receipt = resident_highs()
    setup_seconds = time.perf_counter() - resident_started
    external_started = time.perf_counter()
    returned, cleanup, cleanup_record = run_wholegraph(graph, original, args, api, solver_api)
    returned["external_caller_observed_return_seconds"] = time.perf_counter() - external_started
    returned["missed_return_sample"] = returned["external_caller_observed_return_seconds"] > args.deadline_seconds
    returned["strict_on_time_gain_seconds"] = (0. if returned["missed_return_sample"] else returned["actual_gain_seconds"])
    # Reap this run before the CLI ends and the serial benchmark starts its
    # next method. Cleanup timing is declared separately from function return.
    cleanup.join()
    returned.update(graph=metadata, deployment_resident_setup_seconds=setup_seconds,
                    initial_global_greedy=initial_record, initial_mask_sha256=initial_mask_sha256,
                    frozen_budget_binding=budget_receipt,
                    microsecond_serialization=domain, solver_source_and_binary=solver_receipt,
                    paper_code_source_sha256=api["reference_sha256"],
                    implementation_sha256=p0.sha_file(__file__),
                    p0_graph_and_initialization_sha256=p0.sha_file(p0.__file__),
                    solver_library_loading_in_resident_setup=True,
                    setup_excludes_python_process_and_initial_module_import_startup=True,
                    initial_global_greedy_preparation_already_in_resident_setup_do_not_add_twice=True,
                    cpu_affinity=sorted(os.sched_getaffinity(0)),
                    threads_environment=THREAD_ENV, python_version=platform.python_version(),
                    isolated_cleanup_after_function_return=cleanup_record,
                    citation=("https://doi.org/10.4230/LIPIcs.SEA.2025.22" if args.method == "CHILS-p1"
                              else "https://highs.dev/"))
    p0.write_json(out / "actual_policy.json", returned)
    print(json.dumps(dict(method=args.method, graph_id=metadata["graph_id"],
                          actual_gain_seconds=returned["actual_gain_seconds"],
                          strict_on_time_gain_seconds=returned["strict_on_time_gain_seconds"],
                          caller_seconds=returned["external_caller_observed_return_seconds"],
                          missed_return_sample=returned["missed_return_sample"],
                          stop_reason=returned["stop_reason"])), flush=True)


if __name__ == "__main__":
    main()
