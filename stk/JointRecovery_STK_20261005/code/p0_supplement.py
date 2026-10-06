"""Append 50 then 10 ms probes to EXACT completed P0 snapshots.

No original/action/warm/history construction is allowed. Original files stay
unchanged. This wrapper only replays their scopes against original graph edges.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

import p0_recovery_probes as p0


def graph_paths(root):
    paths = {}
    for path in Path(root).resolve().rglob("*.npz"):
        with np.load(path, allow_pickle=False) as arrays:
            identity = str(p0.scalar(arrays, "graph_id", path.stem))
        if identity in paths:
            raise ValueError("Duplicate input graph_id")
        paths[identity] = path
    return paths


def reconstruct_frozen(graph, group, api):
    from experiments.v4_neighborhoods import CoordinationAction
    state = group["snapshot"]
    original = frozenset(state["original"])
    best = frozenset(state["best_selected"])
    if not api["feasible"](graph, original) or not api["feasible"](graph, best):
        raise ValueError("Original frozen snapshot membership is infeasible")
    initial = api["objective"](graph, original)
    if not math.isclose(initial, state["original_value_seconds"], rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError("Original frozen incumbent reward differs")
    gain = max(0., api["objective"](graph, best) - initial)
    if not math.isclose(gain, state["best_gain_seconds"], rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError("Original frozen best gain differs from materialized members")
    # This only reconstructs the recorded observable scope. Crucially, never
    # call coordination_cells, global_incumbent, initial_known_warm or history.
    cache = api["Cache"](graph, original, ())
    actions = []
    for record in group["scopes"]:
        action = CoordinationAction(tuple(record["inserts"]), tuple(record["releases"]))
        scope = cache.scope(record["action_index"], action, p0.CAP)
        for field in ("inserts", "base", "displaced"):
            if frozenset(getattr(scope, field)) != frozenset(record[field]):
                raise ValueError("Recorded frozen scope changed: " + field)
        if tuple(scope.replacements) != tuple(record["replacements"]):
            raise ValueError("Recorded original ordered recovery domain changed")
        if scope.eligible_before_cap != record["eligible_before_cap"]:
            raise ValueError("Recorded original domain truncation changed")
        if not math.isclose(scope.immediate_gain, record["q_seconds"], rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError("Recorded frozen q changed")
        warm = api["check"](graph, scope, state["warm_by_action"][str(scope.action_index)])
        if len(warm) != len(state["warm_by_action"][str(scope.action_index)]):
            raise ValueError("Recorded actual warm has duplicate memberships")
        expected_scope_hash = p0.sha_json({key: record[key] for key in
            ("inserts", "releases", "base", "displaced", "replacements", "q_seconds")})
        if expected_scope_hash != record["scope_sha256"]:
            raise ValueError("Frozen scope descriptor hash differs")
        actions.append((scope, record))
    if p0.snapshot_identity(state, actions) != group["controller_state_identity"]:
        raise ValueError("Reconstructed actual frozen controller state identity differs")
    frozen_sha = p0.sha_json(state)
    for row in group["requests"]:
        if row["snapshot_sha256"] != frozen_sha:
            raise ValueError("Original request did not clone the saved exact snapshot")
    return actions


def require_gate(path):
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    if gate.get("allow_short_probe_execution") is not True or not gate.get("reason"):
        raise ValueError("Root P0 actual opportunity decision required before short-budget probe")
    return dict(path=str(Path(path).resolve()), sha256=p0.sha_file(path), decision=gate)


def supplement_graph(summary_path, graph_path, destination, native, api, args, gate):
    source = summary_path.parent.resolve()
    old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    old_protocol_path = source / "protocol.json"
    old_protocol = json.loads(old_protocol_path.read_text(encoding="utf-8"))
    if old_summary["status"] != "ACTUAL_P0_PROBES_COMPLETE_NO_TRAINING":
        raise ValueError("Only completed actual P0 collections may be supplemented")
    if set(old_protocol["budgets_ms"]) != {200, 1000}:
        raise ValueError("Supplement requires the original complete 200/1000 ms workpoints")
    if old_protocol["repeats"] != 2:
        raise ValueError("Supplement requires original two-repeat formal P0, not a smoke")
    if old_protocol["reference_sha256"] != api["reference_sha256"]:
        raise ValueError("Replay algorithm sources differ from frozen P0 closure")
    if old_protocol["code_sha256"] != p0.sha_file(p0.__file__):
        raise ValueError("Actual P0 adapter source differs from the frozen 200/1000ms executor")
    if old_protocol["native_binary_sha256"] != native.sha256:
        raise ValueError("Native binary differs from frozen P0")
    graph, metadata = p0.load_graph(graph_path, api)
    if metadata["npz_sha256"] != old_summary["graph"]["npz_sha256"] or metadata["npz_sha256"] != old_protocol["graph"]["npz_sha256"]:
        raise ValueError("Graph bytes differ from both original frozen P0 bindings")
    destination = Path(destination).resolve() / metadata["graph_id"]
    if destination.exists():
        raise FileExistsError("Supplement graph output must be entirely fresh")
    destination.mkdir(parents=True)
    paths = sorted(source.glob("snapshot_*.json"))
    originals = {str(path): p0.sha_file(path) for path in paths}
    originals[str(summary_path)] = p0.sha_file(summary_path)
    originals[str(old_protocol_path)] = p0.sha_file(old_protocol_path)
    protocol = dict(old_protocol, budgets_ms=[200, 1000, 50, 10],
        supplement_schema="joint_recovery_stk_p0_supplement_v1", supplement_code_sha256=p0.sha_file(__file__),
        original_P0_bindings=originals, original_summary_sha256=p0.sha_file(summary_path),
        original_protocol_sha256=p0.sha_file(old_protocol_path),
        original_results_never_modified=True, exact_original_snapshot_and_warm_clones=True,
        additional_native_budget_order_ms=[50, 10], supplement_repeats=2,
        gate=gate, initial_actions_or_greedy_or_history_not_regenerated=True)
    p0.write_json(destination / "protocol.json", protocol)
    groups = []
    original_groups = []
    replay_actions = []
    snapshot_hashes = []
    original_row_hashes = []
    for index, path in enumerate(paths):
        original_group = json.loads(path.read_text(encoding="utf-8"))
        actions = reconstruct_frozen(graph, original_group, api)
        group = copy.deepcopy(original_group)
        group["supplement_original_snapshot_file_sha256"] = originals[str(path)]
        groups.append(group)
        original_groups.append(original_group)
        replay_actions.append(actions)
        snapshot_hashes.append(p0.sha_json(group["snapshot"]))
        original_row_hashes.append(p0.sha_json(group["requests"]))
    # First complete the graph's whole 50ms pass, then its whole 10ms pass.
    for budget in (50, 10):
        for index, (path, group, actions) in enumerate(zip(paths, groups, replay_actions)):
            for scope, descriptor in actions:
                for repeat in range(2):
                    row = p0.request_row(graph, scope, descriptor, group["snapshot"], budget, repeat, native, api)
                    row["supplement_source_snapshot_sha256"] = originals[str(path)]
                    group["requests"].append(row)
            if p0.sha_json(group["snapshot"]) != snapshot_hashes[index]:
                raise AssertionError("Short-budget alternative mutated the original frozen snapshot")
            original_count = len(original_groups[index]["requests"])
            if p0.sha_json(group["requests"][:original_count]) != original_row_hashes[index]:
                raise AssertionError("Original 200/1000ms result rows changed")
            group["supplement_completed_budget_stages_ms"] = [50] if budget == 50 else [50, 10]
            p0.write_json(destination / path.name, group)
            print(json.dumps(dict(graph_id=graph.name, supplement_budget_ms=budget,
                                  supplemented_states=index + 1, original_200_1000ms_rows=original_count,
                                  added_50_10ms_rows=len(group["requests"]) - original_count)), flush=True)
    for path, digest in originals.items():
        if p0.sha_file(path) != digest:
            raise AssertionError("Original P0 input changed during supplement: " + path)
    summary = p0.summarize_graph(groups, metadata, [200, 1000, 50, 10], old_summary["target_unique_states"])
    summary.update(supplement_completed=True, original_results_never_modified=True,
                   original_summary_sha256=originals[str(summary_path)],
                   original_states_and_physical_source_count_unchanged=True,
                   original_groups=len(paths), no_training=True)
    p0.write_rows_csv(destination / "requests.csv", groups)
    p0.write_json(destination / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--graphs-dir", required=True)
    parser.add_argument("--source-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chils", required=True)
    parser.add_argument("--chils-source", required=True)
    parser.add_argument("--p0-gate", required=True)
    args = parser.parse_args()
    source = Path(args.source_out).resolve()
    out = Path(args.out).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Supplement output must be independent of original P0 inputs")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Supplement output must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    api = p0.load_runtime(args.runtime_root)
    gate = require_gate(args.p0_gate)
    audit = p0.native_source_audit(args.chils_source)
    native = p0.NativeCHILS(args.chils, audit, api, out / "temporary_native_calls")
    paths = graph_paths(args.graphs_dir)
    summaries = sorted(source.rglob("summary.json"))
    results = []
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "ACTUAL_P0_PROBES_COMPLETE_NO_TRAINING":
            continue
        identity = summary["graph"]["graph_id"]
        results.append(supplement_graph(summary_path, paths[identity], out, native, api, args, gate))
    if not results:
        raise ValueError("No complete formal P0 graph found; supplement not run")
    p0.write_json(out / "batch_summary.json", dict(status="ACTUAL_P0_SHORT_SUPPLEMENT_COMPLETE",
                  graphs=results, original_results_never_modified=True,
                  added_budgets_ms=[50, 10], repeats=2, no_initial_state_or_action_or_warm_regeneration=True))


if __name__ == "__main__":
    main()
