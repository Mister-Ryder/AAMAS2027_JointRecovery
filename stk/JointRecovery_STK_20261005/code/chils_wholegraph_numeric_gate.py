"""Bound and actually smoke-test original CHILS-p1 on the frozen STK whole graphs.

prepare never executes a native program and never writes a PASS gate. run is
Linux-only and performs exactly three small fixtures and one development smoke.
This is a numerical/interface clearance, not a solution-quality benchmark.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

SCHEMA = "joint_recovery_stk_chils_wholegraph_numeric_gate_v1"
COMMIT = "515952724cd3dcc6c4365a340ecf0f1da782119a"
BINARY_SHA = "19610c03f334c6267f94543ad3053d792cba56e9ae211fceb6c36f21750c88a0"
TICK_SECONDS = 1e-6
MAX_N, MAX_M, MAX_S = 20919, 232636, 9_383_000_000_000
SIGNED64_MAX, INT32_MAX, EXACT_DOUBLE = (1 << 63) - 1, (1 << 31) - 1, 1 << 53
SMOKE_GRAPH = "JR-DUAL-r004-R12-g0170"
SOURCE_LF_SHA = {
    "include/chils.h": "3d0c6e4cfb618fad8a2415bc1748751ed3314a919f0981cdbba4d7a267de9801",
    "include/chils_internal.h": "9f9632a69552c501daec03300fd4c8f2f87190904a12ec167b6929581baf179e",
    "include/graph.h": "8e9d38b0c80c68a073440da50759f5770caa36e4467cbdd282ab11cca0bdeba8",
    "include/local_search.h": "64bf1cec407f3e38f331a4c0f17d773f58e3f14988b93923f9514cadd32dbd0b",
    "src/chils.c": "5fb4fa4189c81e41d1ee6cb09cb6dc1b79ff0faab4475a2e0e095e3fc4d8a6e6",
    "src/chils_internal.c": "3957fe991dfbd0913b74f6608ead3b217cdf7bf61eb329af95a669eb176a9937",
    "src/graph.c": "4b684cf91ece5690ede3fe1fa80237e8081aef3204001e99f0db10ad11e9db26",
    "src/local_search.c": "72b9ce55083d5c4f4e5b91ba02a95cad01776b3b047c280eb96e4bca4cc3ac6e",
    "src/main.c": "d67f7700f1150c4c0faa3ff03eea3559252d16adb49fe112f70563b5ae4741d4",
}
DERIVATION = [
    "Only -p 1 -c 1 -r 17, positive simple graphs and the pinned original binary are covered; cooperative p>1 paths are excluded.",
    "The p1 c<il guard precedes c++ in local_search_explore; even default il=LLONG_MAX cannot postincrement LLONG_MAX.",
    "Cost and adjacent_weight are nonnegative subset sums bounded by S, including transient add-before-remove membership.",
    "Distinct two-vertex replacement sums are at most S; signed replacement/path differences are within [-S,S]. Weighted perturbation is at most 2*S; fixed random noise has magnitude at most 2**29.",
    "Each successful greedy update increases the integer objective by at least 1 and removes at most n vertices. Perturbation has at most 129 updates (MAX_GUESS=128). Removal log length per iteration is at most n*(S+129).",
    "Log count resets each iteration. Doubled allocation capacity is below 2*n*(S+129); 4-byte entries require below 8*n*(S+129) bytes. This must fit signed64 and the actual Linux 64-bit size_t.",
    "Queue, tightness and path masks use unique vertex IDs; n, 2*n and 2*m+1 fit signed int32. Symmetric edge pointers and objective weights are long long.",
    "Generated weighted METIS input uses positive decimal integers and LF. All parser prefixes are bounded by max(n,m,max_weight); the digit-add-before-subtract-48 intermediate is bounded by that value+48.",
    "S<2**53 additionally preserves exact integer-to-double conversions; all comparisons and membership objectives in the covered source use long long. Original duration rewards remain unchanged outside the native microsecond serialization.",
]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def graph_ids():
    return [f"JR-DUAL-r{s:03d}-{stations}-g{gap:04d}"
            for s in range(4, 12) for stations in ("R12", "R8") for gap in (170, 340, 680)]


def numeric_checks(n, m, ticks):
    S, largest = sum(ticks), max(ticks, default=0)
    return dict(
        positive_integer_weights=bool(n > 0 and all(w > 0 for w in ticks)),
        fixed_wholegraph_domain=bool(n <= MAX_N and m <= MAX_M and S <= MAX_S),
        signed64_log_allocation=bool(8 * n * (S + 129) <= SIGNED64_MAX),
        signed64_weight_and_noise=bool(2 * S + (1 << 29) <= SIGNED64_MAX),
        signed32_vertex_and_directed_edge_indices=bool(2 * n <= INT32_MAX and 2 * m + 1 <= INT32_MAX),
        signed64_decimal_parser_intermediates=bool(max(n, m, largest) + 48 <= SIGNED64_MAX),
        exact_integer_to_double=bool(S < EXACT_DOUBLE),
        signed64_graph_array_allocations=bool(max(8 * (n + 1), 8 * n, 8 * m) <= SIGNED64_MAX))


def load_graph(path):
    with np.load(path, allow_pickle=False) as a:
        if a["weights"].dtype != np.dtype("float64"):
            raise ValueError("Frozen original float64 duration archive required")
        w = np.asarray(a["weights"], dtype=np.float64).copy()
        edges = (np.asarray(a["edges"], dtype=np.int64).reshape(-1, 2).copy()
                 if "edges" in a else np.column_stack((a["edge_u"], a["edge_v"])))
        agents = np.asarray(a["agents"], dtype=np.int64).copy()
    if w.ndim != 1 or len(agents) != len(w) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Original positive finite float64 duration weights required")
    if edges.size and (np.any(edges < 0) or np.any(edges >= len(w)) or np.any(edges[:, 0] >= edges[:, 1])):
        raise ValueError("Original canonical u<v edges required")
    if len(edges) and len(np.unique(edges[:, 0] * len(w) + edges[:, 1])) != len(edges):
        raise ValueError("Simple original graph required for subset-sum proof")
    ticks = [int(round(float(x) / TICK_SECONDS)) for x in w]
    checks = numeric_checks(len(w), len(edges), ticks)
    checks["original_positive_duration_weights"] = True
    checks["original_edge_index_contract"] = True
    checks["simple_original_graph_no_duplicate_edges"] = True
    return w, edges, agents, ticks, checks


def source_proof(source, binary):
    files = sorted(f for branch in ("src", "include") for f in (source / branch).rglob("*")
                   if f.is_file() and f.suffix in (".c", ".h"))
    raw = {f.relative_to(source).as_posix(): sha_file(f) for f in files}
    lf = {f.relative_to(source).as_posix(): hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
          for f in files}
    texts = {name: (source / name).read_text(encoding="utf-8")
             for name in SOURCE_LF_SHA if (source / name).is_file()}
    ls, main = texts.get("src/local_search.c", ""), texts.get("src/main.c", "")
    graph_header = texts.get("include/graph.h", "")
    ls_header = texts.get("include/local_search.h", "")
    signatures = all(s in ls for s in ("while (c < il)", "ls->log_count = 0;", "ls->log_alloc *= 2;",
                                     "(2 * g->W[v])", "(1 << 29)", "long long best = ls->cost", "#define MAX_GUESS 128"))
    signatures = signatures and "if (run_chils > 1)" in main
    types = all(t in graph_header for t in ("int n;", "long long m;", "long long *V;", "int *E;", "long long *W;"))
    types = types and all(t in ls_header for t in ("long long cost;", "long long *adjacent_weight;", "long long log_count"))
    bsha = sha_file(binary) if binary.is_file() else None
    checks = dict(pinned_complete_source_identity=(lf == SOURCE_LF_SHA),
                  pinned_original_binary_identity=(bsha == BINARY_SHA),
                  audited_p1_loop_and_log_signatures=bool(signatures),
                  audited_longlong_weight_and_counter_types=bool(types))
    return dict(schema=SCHEMA + "_source_proof", source_commit=COMMIT,
                binary_path=str(binary), binary_sha256=bsha,
                source_root=str(source), source_files_sha256=raw,
                normalized_lf_source_files_sha256=lf, expected_normalized_lf_source_files_sha256=SOURCE_LF_SHA,
                checks=checks, derivation=DERIVATION,
                extended_domain=dict(max_vertices=MAX_N, max_edges=MAX_M, max_tick_weight_sum=MAX_S,
                    maximum_safe_signed64_bound=8 * MAX_N * (MAX_S + 129),
                    safe_signed64_limit=SIGNED64_MAX, exact_double_limit=EXACT_DOUBLE),
                limitations=["No cap256 certificate is inherited", "No p>1 clearance", "No BR64/KaMIS clearance",
                    "Allocation failure remains a failed run", "No online deadline or solution-quality advantage is established"])


def adjacency(n, edges):
    rows = [set() for _ in range(n)]
    for u, v in edges:
        rows[int(u)].add(int(v))
        rows[int(v)].add(int(u))
    return rows


def feasible(members, n, edges):
    if len(members) != len(set(members)) or any(v < 0 or v >= n for v in members):
        return False
    mask = np.zeros(n, dtype=bool)
    mask[list(members)] = True
    return not bool(np.any(mask[edges[:, 0]] & mask[edges[:, 1]])) if len(edges) else True


def exhaustive_optimum(ticks, edges):
    best, chosen = -1, ()
    for mask in range(1 << len(ticks)):
        selected = tuple(i for i in range(len(ticks)) if mask & (1 << i))
        if feasible(selected, len(ticks), edges):
            value = sum(ticks[i] for i in selected)
            if (value, selected) > (best, chosen):
                best, chosen = value, selected
    return chosen, best


def fixtures():
    # Different compatibility patterns; all totals exceed the previous 2**40
    # cap-domain ceiling. These test numeric handling, not heuristic quality.
    path = np.asarray([(i, i + 1) for i in range(7)], dtype=np.int64)
    clique = np.asarray(list(itertools.combinations(range(7), 2)), dtype=np.int64)
    mixed = np.asarray([(0, 1), (0, 2), (1, 2), (3, 4), (4, 5), (5, 6), (3, 6),
                        (2, 7), (6, 8), (7, 9)], dtype=np.int64)
    return [
        ("high_total_path", [(1 << 38) + 19 * i + 1 for i in range(8)], path),
        ("high_total_clique", [(1 << 40) + 23 * i + 1 for i in range(7)], clique),
        ("high_total_mixed", [(1 << 37) + 1009 * i + 3 for i in range(10)], mixed),
    ]


def original_incumbent(runtime_root, weights, edges, agents):
    """The same three paid-prefix greedy candidates as P0, using original rewards."""
    sys.path.insert(0, str(runtime_root / "src"))
    from joint_recovery.core import Graph, is_feasible
    from joint_recovery.v4_budgeted_recovery import objective
    rows = adjacency(len(weights), edges)
    graph = Graph(weights, agents, tuple(map(frozenset, rows)), SMOKE_GRAPH)
    started = time.perf_counter()
    candidates = []
    for exponent in (0.0, 0.5, 1.0):
        order = sorted(range(graph.n), key=lambda v: (
            -float(weights[v]) / (1 + len(rows[v])) ** exponent, -float(weights[v]), v))
        selected = set()
        for v in order:
            if not rows[v] & selected:
                selected.add(v)
        if not is_feasible(graph, selected):
            raise ValueError("Original paid-prefix candidate is infeasible")
        candidates.append(frozenset(selected))
    best = max(candidates, key=lambda members: (objective(graph, members), tuple(sorted(members))))
    receipt = dict(contract="P0 global_incumbent exponents 0,0.5,1 and identical reward/tie ordering",
                   selected=sorted(best), objective_seconds=float(objective(graph, best)),
                   preparation_seconds=time.perf_counter() - started,
                   reference_sha256={r: sha_file(runtime_root / r) for r in
                       ("src/joint_recovery/core.py", "src/joint_recovery/v4_budgeted_recovery.py")})
    return tuple(sorted(best)), receipt


def native_check(name, directory, binary, ticks, edges, warm, seconds,
                 expected_optimum=None, original_weights=None):
    directory.mkdir(parents=True, exist_ok=False)
    n, m = len(ticks), len(edges)
    if not feasible(warm, n, edges) or not all(numeric_checks(n, m, ticks).values()):
        raise ValueError("Native fixture/warm outside proved numerical domain")
    rows = adjacency(n, edges)
    graph_path, warm_path = directory / "input.graph", directory / "warm.ids"
    output_path = directory / "output.ids"
    graph_path.write_bytes((f"{n} {m} 10\n" + "\n".join(
        " ".join(map(str, [ticks[v]] + [u + 1 for u in sorted(rows[v])])) for v in range(n)) + "\n").encode("ascii"))
    warm_path.write_bytes(("".join(f"{v + 1}\n" for v in sorted(warm))).encode("ascii"))
    command = [str(binary), "-g", str(graph_path), "-i", str(warm_path), "-o", str(output_path),
               "-p", "1", "-c", "1", "-r", "17", "-t", format(seconds, ".6f")]
    record = dict(name=name, command=command, population=1, threads=1, seed=17,
                  native_budget_seconds=seconds, vertices=n, edges=m, tick_weight_sum=sum(ticks),
                  warm_membership=list(warm), warm_exact_tick_objective=sum(ticks[i] for i in warm),
                  expected_optimum_ticks=expected_optimum, status="NATIVE_CHECK_FAILED")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   cwd=directory, env=env, timeout=15, check=False)
        record["external_native_return_seconds"] = time.perf_counter() - started
        record["returncode"] = completed.returncode
        (directory / "stdout.txt").write_bytes(completed.stdout)
        (directory / "stderr.txt").write_bytes(completed.stderr)
        members = tuple(int(v) - 1 for v in output_path.read_text(encoding="ascii").split()) if output_path.is_file() else ()
        valid = bool(output_path.is_file() and feasible(members, n, edges))
        exact = sum(ticks[i] for i in members) if valid else None
        parsed = None
        for line in completed.stdout.decode("utf-8", "replace").splitlines():
            fields = line.strip().split(",")
            if len(fields) == 6:
                try:
                    if int(fields[1]) == n and int(fields[2]) == m:
                        parsed = int(fields[3])
                except ValueError:
                    pass
        checks = dict(native_exit_success=(completed.returncode == 0),
                      membership_present_and_original_graph_feasible=valid,
                      exact_reported_tick_objective_matches_membership=(valid and parsed == exact),
                      true_warm_argument_supplied=("-i" in command and warm_path.is_file()),
                      native_preserves_warm_tick_objective=(valid and exact >= record["warm_exact_tick_objective"]))
        if expected_optimum is not None:
            checks["exact_exhaustive_optimum_preserved"] = bool(valid and exact == expected_optimum)
        record.update(checks=checks, native_reported_ticks=parsed, returned_membership=list(members),
                      returned_exact_tick_objective=exact)
        if original_weights is not None and valid:
            record["returned_original_duration_seconds"] = math.fsum(float(original_weights[i]) for i in sorted(members))
            record["warm_original_duration_seconds"] = math.fsum(float(original_weights[i]) for i in sorted(warm))
            record["native_rounding_can_change_submicrosecond_preferences"] = True
        if all(checks.values()):
            record["status"] = "NATIVE_CHECK_PASS"
    except Exception as error:
        record.update(error=f"{type(error).__name__}: {error}", checks={"native_completed_and_validated": False},
                      external_native_return_seconds=time.perf_counter() - started)
        if isinstance(error, subprocess.TimeoutExpired):
            (directory / "stdout.txt").write_bytes(error.stdout or b"")
            (directory / "stderr.txt").write_bytes(error.stderr or b"")
    record["files_sha256"] = {p.name: sha_file(p) for p in directory.iterdir() if p.is_file()}
    write_json(directory / "receipt.json", record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", "--dataset-root", dest="root", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--graphs")
    p.add_argument("--chils", required=True)
    p.add_argument("--chils-source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("prepare", "run"), default="prepare")
    args = p.parse_args()
    root, runtime = Path(args.root).resolve(), Path(args.runtime_root).resolve()
    graphs = Path(args.graphs).resolve() if args.graphs else root / "graphs"
    binary, source, out = Path(args.chils).resolve(), Path(args.chils_source).resolve(), Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory; pending/failed gates remain auditable")
    out.mkdir(parents=True, exist_ok=True)
    proof = source_proof(source, binary)
    write_json(out / "source_proof.json", proof)
    abi = dict(platform=platform.system(), int_bytes=ctypes.sizeof(ctypes.c_int),
               longlong_bytes=ctypes.sizeof(ctypes.c_longlong), pointer_bytes=ctypes.sizeof(ctypes.c_void_p),
               size_t_bytes=ctypes.sizeof(ctypes.c_size_t))
    abi_ok = bool(args.mode == "run" and abi["platform"] == "Linux" and abi["int_bytes"] == 4
                  and abi["longlong_bytes"] == 8 and abi["pointer_bytes"] == 8 and abi["size_t_bytes"] == 8)
    protocol = dict(schema=SCHEMA + "_protocol", graph_ids=graph_ids(),
                    data_scope="r004/r005 development, r006/r007 validation, r008-r011 test; 48 fixed complete graphs",
                    source_commit=COMMIT, native=dict(tick_seconds=TICK_SECONDS, population=1, threads=1, seed=17),
                    domain=dict(max_vertices=MAX_N, max_edges=MAX_M, max_tick_weight_sum=MAX_S),
                    fixture_names=[f[0] for f in fixtures()], development_smoke_graph=SMOKE_GRAPH,
                    development_smoke_native_seconds=0.1, script_sha256=sha_file(__file__),
                    source_proof_sha256=sha_file(out / "source_proof.json"), abi=abi)
    write_json(out / "protocol.json", protocol)
    information, smoke_data = {}, None
    for graph_id in graph_ids():
        path = graphs / (graph_id + ".npz")
        try:
            weights, edges, agents, ticks, checks = load_graph(path)
            information[graph_id] = dict(
                graph=dict(graph_id=graph_id, npz_path=str(path), npz_sha256=sha_file(path), vertices=len(weights), edges=len(edges)),
                native=dict(tick_seconds=TICK_SECONDS, tick_weight_sum=sum(ticks), max_tick_weight=max(ticks),
                    safe_signed64_bound=8 * len(weights) * (sum(ticks) + 129), exact_double_limit=EXACT_DOUBLE,
                    safe_signed64_limit=SIGNED64_MAX, population=1, threads=1, seed=17,
                    reward_encoding="round(float64_original_duration_seconds / 1e-6); original weights unchanged",
                    max_serialization_error_seconds=max(abs(t * TICK_SECONDS - float(w)) for t, w in zip(ticks, weights)),
                    sum_serialization_error_bound_seconds=len(weights) * TICK_SECONDS / 2),
                checks=checks)
            if graph_id == SMOKE_GRAPH:
                smoke_data = weights, edges, agents, ticks
        except Exception as error:
            information[graph_id] = dict(graph=dict(graph_id=graph_id, npz_path=str(path)), native={},
                                        checks={"graph_numeric_input_available_and_valid": False},
                                        error=f"{type(error).__name__}: {error}")
    fixtures_actual, smoke_actual = [], None
    all_inputs_ok = all(all(row["checks"].values()) for row in information.values())
    source_ok = all(proof["checks"].values())
    runtime_files = ("src/joint_recovery/core.py", "src/joint_recovery/v4_budgeted_recovery.py")
    runtime_ok = all((runtime / r).is_file() for r in runtime_files)
    if args.mode == "run" and abi_ok and source_ok and all_inputs_ok and runtime_ok:
        for name, ticks, edges in fixtures():
            warm, optimum = exhaustive_optimum(ticks, edges)
            record = native_check(name, out / "native_checks" / name, binary, ticks, edges, warm, .025,
                                  expected_optimum=optimum)
            record["exhaustive_states"] = 1 << len(ticks)
            record["high_total_exceeds_old_cap_ceiling"] = sum(ticks) > (1 << 40)
            write_json(out / "native_checks" / name / "receipt.json", record)
            fixtures_actual.append(record)
        if all(r["status"] == "NATIVE_CHECK_PASS" for r in fixtures_actual):
            weights, edges, agents, ticks = smoke_data
            try:
                warm, warm_receipt = original_incumbent(runtime, weights, edges, agents)
                write_json(out / "development_warm.json", warm_receipt)
                smoke_actual = native_check(SMOKE_GRAPH, out / "native_checks" / "development_fullgraph", binary,
                    ticks, edges, warm, .1, original_weights=weights)
                smoke_actual["graph_npz_sha256"] = information[SMOKE_GRAPH]["graph"]["npz_sha256"]
                smoke_actual["development_warm_receipt_sha256"] = sha_file(out / "development_warm.json")
                smoke_actual["full_original_input_not_a_capped_response"] = True
                write_json(out / "native_checks" / "development_fullgraph" / "receipt.json", smoke_actual)
            except Exception as error:
                smoke_actual = dict(status="NATIVE_CHECK_FAILED", error=f"{type(error).__name__}: {error}")
    native_records = {}
    for path in sorted((out / "native_checks").rglob("receipt.json")) if (out / "native_checks").exists() else []:
        native_records[path.relative_to(out).as_posix()] = sha_file(path)
    fixture_ok = len(fixtures_actual) == 3 and all(r["status"] == "NATIVE_CHECK_PASS" and
                 r["high_total_exceeds_old_cap_ceiling"] for r in fixtures_actual)
    smoke_ok = bool(smoke_actual and smoke_actual["status"] == "NATIVE_CHECK_PASS")
    guard = dict(schema=SCHEMA + "_guard", mode=args.mode, source_proof_sha256=sha_file(out / "source_proof.json"),
                 protocol_sha256=sha_file(out / "protocol.json"), native_receipts_sha256=native_records,
                 actual_linux_abi=abi, checks=dict(pinned_source_and_binary=source_ok, actual_linux_abi=abi_ok,
                 all_48_wholegraph_numeric_inputs=all_inputs_ok, runtime_original_incumbent_dependencies=runtime_ok,
                 three_actual_high_total_native_fixtures=fixture_ok, actual_development_wholegraph_smoke=smoke_ok),
                 native_executions=len(fixtures_actual) + int(smoke_actual is not None),
                 fixtures=fixtures_actual, development_smoke=smoke_actual)
    guard["status"] = ("WHOLEGRAPH_NUMERIC_GUARD_PASS" if all(guard["checks"].values()) else
                       "PENDING_NATIVE_WHOLEGRAPH_CHECKS" if args.mode == "prepare" else "WHOLEGRAPH_NUMERIC_GUARD_FAILED")
    write_json(out / "guard.json", guard)
    passes = []
    for graph_id, info in information.items():
        checks = dict(proof["checks"], **info["checks"], **guard["checks"])
        status = ("WHOLEGRAPH_NUMERIC_GATE_PASS" if all(checks.values()) else
                  "PENDING_NATIVE_WHOLEGRAPH_CHECKS" if args.mode == "prepare" else "WHOLEGRAPH_NUMERIC_GATE_FAILED")
        gate = dict(schema=SCHEMA, status=status, graph=info["graph"], native=info["native"],
                    chils={k: proof[k] for k in ("source_commit", "binary_sha256", "source_files_sha256", "normalized_lf_source_files_sha256")},
                    checks=checks, evidence=dict(source_proof_sha256=sha_file(out / "source_proof.json"),
                        protocol_sha256=sha_file(out / "protocol.json"), guard_sha256=sha_file(out / "guard.json"),
                        native_receipts_sha256=native_records), scope="original CHILS-p1-c1 seed17 on the full fixed STK graph",
                    error=info.get("error"))
        write_json(out / "gates" / (graph_id + ".json"), gate)
        if status == "WHOLEGRAPH_NUMERIC_GATE_PASS":
            passes.append(graph_id)
    completion = dict(schema=SCHEMA + "_completion", status=guard["status"], mode=args.mode,
                      expected_graphs=48, pass_graphs=len(passes), native_executions=guard["native_executions"],
                      gates_sha256={p.stem: sha_file(p) for p in sorted((out / "gates").glob("*.json"))},
                      benchmark_run=False, original_source_modified=False, original_binary_modified=False,
                      old_cap_certificate_inherited=False, br64_wholegraph_cleared=False)
    write_json(out / "completion.json", completion)
    print(json.dumps({"status": completion["status"], "mode": args.mode, "pass_graphs": len(passes), "out": str(out)}, ensure_ascii=False))
    return 0 if args.mode == "prepare" or len(passes) == 48 else 2


if __name__ == "__main__":
    raise SystemExit(main())
