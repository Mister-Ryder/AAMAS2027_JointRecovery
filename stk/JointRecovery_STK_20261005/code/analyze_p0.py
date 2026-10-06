"""One evidence-preserving P0 analysis, with physical sources as the grouping unit.

No predictor is trained, no execution protocol or labels are changed, and no
unfavourable/zero/unavailable request is silently dropped from the request table.
Native workpoints are not end-to-end caller deadlines. A summary-similar pair
with different outcomes is a diagnostic lead, not proof that a graph model wins.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SUMMARY_RELATIVE_TOLERANCE = 0.05
SIZE_RELATIVE_TOLERANCE = 0.10
RESULT_RELATIVE_TOLERANCE = 0.001


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def optional_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def metrics(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {"available_count": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "max": max(values) if values else None}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nNO_AVAILABLE_ROWS\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def request_available(row):
    return (row.get("available_for_allocation") is True and row.get("failed") is False
            and row.get("actual_finite_supervision_valid") is True
            and row.get("complete_membership_valid") is True)


def scope_base_diagnostics(npz_path, scopes):
    """Measure the actual displaced-neighbour sphere excluded by the full base.

    This is a structural diagnostic, not a replay or a new alternate action pool.
    No resource membership is treated as a static clique.
    """
    if not npz_path.exists():
        return {s["scope_sha256"]: {"base_blocking_status": "GRAPH_NPZ_UNAVAILABLE"} for s in scopes}
    with np.load(npz_path, allow_pickle=False) as graph:
        edges = np.asarray(graph["edges"], dtype=np.int64)
        n = len(graph["weights"])
        origins = np.r_[edges[:, 0], edges[:, 1]]
        neighbours = np.r_[edges[:, 1], edges[:, 0]]
        order = np.argsort(origins, kind="stable")
        neighbours = neighbours[order]
        pointers = np.r_[0, np.cumsum(np.bincount(origins, minlength=n))]
    result = {}
    for scope in scopes:
        key = scope["scope_sha256"]
        if key in result:
            continue
        base = np.asarray(scope["base"], dtype=np.int64)
        displaced = np.asarray(scope["displaced"], dtype=np.int64)
        pieces = [displaced] + [neighbours[pointers[v]:pointers[v + 1]] for v in displaced]
        candidates = np.unique(np.concatenate(pieces)) if pieces else np.empty(0, dtype=np.int64)
        base_mask = np.zeros(n, dtype=np.bool_); base_mask[base] = True
        blocked = np.zeros(n, dtype=np.bool_)
        for vertex in base:
            blocked[neighbours[pointers[vertex]:pointers[vertex + 1]]] = True
        outside_base = candidates[~base_mask[candidates]]
        eligible = outside_base[~blocked[outside_base]]
        expected = int(scope["eligible_before_cap"])
        # This checks the claimed diagnostic denominator, not every solver outcome.
        if len(eligible) != expected:
            raise ValueError("Recorded eligible sphere differs from frozen original graph/base")
        blocked_count = int(np.count_nonzero(blocked[outside_base]))
        result[key] = {"base_blocking_status": "MEASURED_FROM_ORIGINAL_GRAPH",
                       "actual_candidate_sphere_vertices": len(candidates),
                       "candidate_sphere_vertices_outside_base": len(outside_base),
                       "fixed_base_blocked_candidate_vertices": blocked_count,
                       "fixed_base_blocked_fraction_outside_base": blocked_count / max(1, len(outside_base)),
                       "eligible_before_cap_confirmed": expected,
                       "eligible_vertices_removed_by_cap": max(0, expected - int(scope["local_vertices"]))}
    return result


def prepare_rows(graph_id, source, group, scopes, initial_seconds, history_seconds):
    scope_by_action = {int(s["action_index"]): s for s in scopes}
    rows = []
    snapshot = group["snapshot"]
    for raw in group.get("requests", []):
        diag = raw.get("diagnostics", {})
        scope = scope_by_action.get(int(raw["action_index"]), {})
        original_value = float(snapshot["original_value_seconds"])
        local_delta = optional_float(raw.get("local_delta"))
        added = optional_float(raw.get("beyond_snapshot_gain_seconds"))
        conditional = optional_float(raw.get("complete_conditional_return_seconds"))
        prefix = optional_float(snapshot.get("common_prefix_seconds"))
        one_shot_estimate = (initial_seconds + prefix + history_seconds + conditional
                             if all(v is not None for v in (initial_seconds, prefix, history_seconds, conditional)) else None)
        row = {
            "source_group": source, "graph_id": graph_id, "snapshot_id": group["snapshot_id"],
            "controller_state_identity": group["controller_state_identity"], "state_kind": group["state_kind"],
                     "trajectory": group["trajectory"], "action_index": raw["action_index"],
            "scope_sha256": raw["scope_sha256"], "workpoint_ms": raw["workpoint_ms"], "repeat": raw["repeat"],
            "actual_warm_sha256": digest(raw["actual_warm_members"]), "native_seed": diag.get("seed"),
            "native_binary_sha256": diag.get("binary_sha256"), "inherited_cpu_affinity": diag.get("inherited_cpu_affinity"),
            "status": diag.get("status", "UNKNOWN"), "failed": raw.get("failed"),
            "zero_extra_gain": raw.get("zero_extra_gain"), "available_for_allocation": raw.get("available_for_allocation"),
            "actual_finite_supervision_valid": raw.get("actual_finite_supervision_valid"),
            "complete_membership_valid": raw.get("complete_membership_valid"),
            "eligible_for_signal_analysis": request_available(raw), "native_called": diag.get("native_called"),
            "q_seconds": raw.get("q_seconds"), "L_seconds": raw.get("L_seconds"), "U_seconds": raw.get("U_seconds"),
            "P1_recovery_seconds": raw.get("P1_recovery_seconds"), "original_global_seconds": original_value,
            "snapshot_best_gain_seconds": raw.get("snapshot_best_gain_seconds"),
            "recovery_value_seconds": raw.get("recovery_value_seconds"), "q_plus_recovery_seconds": raw.get("q_plus_recovery_seconds"),
            "signed_original_gain_seconds": raw.get("signed_original_gain_seconds"),
            "local_recovery_improvement_percent": 100 * local_delta if local_delta is not None else None,
            "beyond_snapshot_added_seconds": added,
            "beyond_snapshot_global_percent": 100 * added / max(1, original_value) if added is not None else None,
            "warm_members": raw.get("actual_warm_members"), "returned_recovery_members": raw.get("returned_recovery_members"),
            "common_prefix_seconds": prefix, "initial_global_greedy_seconds": initial_seconds,
            "actual_pre_snapshot_history_seconds": history_seconds,
            "complete_conditional_return_seconds": conditional,
            "one_shot_problem_to_return_estimate_seconds": one_shot_estimate,
            "one_shot_estimate_is_not_caller_receipt": True,
            "earliest_verified_added_gain_seconds": raw.get("earliest_verified_more_than_snapshot_gain_seconds"),
            "native_budget_seconds": float(raw["workpoint_ms"]) / 1000,
            "preparation_seconds": diag.get("preparation_seconds"), "process_launch_seconds": diag.get("process_launch_seconds"),
            "native_process_seconds": diag.get("native_process_seconds"), "parse_seconds": diag.get("parse_seconds"),
            "native_validation_and_rescore_seconds": diag.get("validation_and_rescore_seconds"),
            "extra_validation_and_rescore_seconds": raw.get("extra_validation_and_rescore_seconds"),
            "native_return_seconds": diag.get("full_return_seconds"),
            "caller_deadline_seconds": None, "caller_late": "UNMEASURED", "native_process_gt_workpoint_is_not_caller_late": True,
            "local_vertices": scope.get("local_vertices"), "local_edges": scope.get("local_edges"),
            "local_cross_owner_edges": scope.get("local_cross_owner_edges"), "distinct_local_owners": scope.get("distinct_local_owners"),
            "eligible_before_cap": scope.get("eligible_before_cap"), "cap_truncated": scope.get("cap_truncated"),
            **{key: value for key, value in scope.get("base_diagnostics", {}).items()},
        }
        rows.append(row)
    return rows


def g2_pairs(rows):
    """Within the same actual state/workpoint, compare repeated request means."""
    by_action = defaultdict(list)
    for row in rows:
        if row["eligible_for_signal_analysis"]:
            by_action[(row["workpoint_ms"], row["action_index"], row["scope_sha256"], row["actual_warm_sha256"])].append(row)
    by_budget = defaultdict(list)
    for key, repeats in by_action.items():
        representative = repeats[0]
        by_budget[key[0]].append({"row": representative, "repeats": len(repeats),
                                 "mean_recovery": float(np.mean([x["recovery_value_seconds"] for x in repeats])),
                                 "mean_total": float(np.mean([x["q_plus_recovery_seconds"] for x in repeats])),
                                 "repeat_recovery_range": max(x["recovery_value_seconds"] for x in repeats) - min(x["recovery_value_seconds"] for x in repeats)})
    pairs = []
    for budget, options in by_budget.items():
        for left, right in itertools.combinations(options, 2):
            a, b = left["row"], right["row"]
            if a["action_index"] == b["action_index"]:
                continue
            features = ("q_seconds", "L_seconds", "U_seconds", "P1_recovery_seconds")
            if any(a[x] is None or b[x] is None for x in features + ("local_vertices",)):
                continue
            size_close = abs(a["local_vertices"] - b["local_vertices"]) <= SIZE_RELATIVE_TOLERANCE * max(1, a["local_vertices"], b["local_vertices"])
            relative_distances = {x: abs(a[x] - b[x]) / max(1, abs(a[x]), abs(b[x])) for x in features}
            if not (size_close and max(relative_distances.values()) <= SUMMARY_RELATIVE_TOLERANCE):
                continue
            difference = right["mean_total"] - left["mean_total"]
            threshold = max(1e-7, RESULT_RELATIVE_TOLERANCE * max(1, abs(left["mean_recovery"]), abs(right["mean_recovery"])))
            exact_summary = a["local_vertices"] == b["local_vertices"] and all(abs(a[x] - b[x]) <= 1e-7 for x in features)
            pairs.append({"source_group": a["source_group"], "graph_id": a["graph_id"], "snapshot_id": a["snapshot_id"],
                          "workpoint_ms": budget, "left_action_index": a["action_index"], "right_action_index": b["action_index"],
                          "left_scope_sha256": a["scope_sha256"], "right_scope_sha256": b["scope_sha256"],
                          "left_repeats": left["repeats"], "right_repeats": right["repeats"],
                          "exact_summary_at_1e_minus7": exact_summary, "max_relative_summary_difference": max(relative_distances.values()),
                          "left_mean_q_plus_recovery_seconds": left["mean_total"], "right_mean_q_plus_recovery_seconds": right["mean_total"],
                          "right_minus_left_seconds": difference, "meaningful_result_difference": abs(difference) > threshold,
                          "result_difference_threshold_seconds": threshold,
                          "left_local_cross_owner_edges": a["local_cross_owner_edges"], "right_local_cross_owner_edges": b["local_cross_owner_edges"],
                          "left_local_edges": a["local_edges"], "right_local_edges": b["local_edges"],
                          "left_repeat_recovery_range_seconds": left["repeat_recovery_range"],
                          "right_repeat_recovery_range_seconds": right["repeat_recovery_range"],
                          "graph_information_necessity_proved": False})
    return pairs


def g3_pairs(rows):
    by_key = defaultdict(dict)
    for row in rows:
        if row["eligible_for_signal_analysis"]:
            key = (row["scope_sha256"], row["actual_warm_sha256"], row["native_seed"],
                   row.get("native_binary_sha256"), tuple(row.get("inherited_cpu_affinity") or []), row["repeat"])
            by_key[key][row["workpoint_ms"]] = row
    comparisons = []
    by_action = defaultdict(list)
    for key, points in by_key.items():
        if 200 not in points or 1000 not in points:
            continue
        low, high = points[200], points[1000]
        difference = float(high["recovery_value_seconds"]) - float(low["recovery_value_seconds"])
        comparisons.append({"source_group": low["source_group"], "graph_id": low["graph_id"], "snapshot_id": low["snapshot_id"],
                            "scope_sha256": key[0], "warm_sha256": key[1], "native_seed": key[2],
                            "native_binary_sha256": key[3], "inherited_cpu_affinity": list(key[4]), "repeat": key[5],
                            "action_index": low["action_index"], "recovery_200ms_seconds": low["recovery_value_seconds"],
                            "recovery_1000ms_seconds": high["recovery_value_seconds"], "difference_1000_minus_200_seconds": difference,
                            "conditional_return_200ms_seconds": low["complete_conditional_return_seconds"],
                            "conditional_return_1000ms_seconds": high["complete_conditional_return_seconds"]})
        by_action[key[:-1]].append(difference)
    repeatable = any(len(values) >= 2 and (all(x > 1e-7 for x in values) or all(x < -1e-7 for x in values))
                     for values in by_action.values())
    return comparisons, repeatable


def analyze(root, execution_roots=None):
    execution_roots = execution_roots or [root / "execution"]
    summary_paths = sorted({p.resolve() for directory in execution_roots for p in directory.rglob("summary.json")
                            if p.parent.name.startswith("JR-DUAL-")})
    destination = root / "reports" / "p0_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    all_rows, graph_rows, state_rows, similar_pairs, budget_pairs = [], [], [], [], []
    graph_reports, seen_graphs = {}, set()
    input_manifest = []
    for summary_path in summary_paths:
        parent = summary_path.parent
        summary = load(summary_path)
        metadata = summary["graph"]
        graph_id, source = metadata["graph_id"], metadata["source_group"]
        if graph_id in seen_graphs:
            raise ValueError("Multiple execution collections for the same graph; select one explicit execution root")
        seen_graphs.add(graph_id)
        groups = [load(p) for p in sorted(parent.glob("snapshot_*.json"))]
        initial_path = parent / "initial_global_greedy.json"
        initial_seconds = optional_float(load(initial_path).get("preparation_seconds")) if initial_path.exists() else None
        unique_scopes = {s["scope_sha256"]: s for group in groups for s in group["scopes"]}
        base_diag = scope_base_diagnostics(root / "graphs" / f"{graph_id}.npz", list(unique_scopes.values()))
        rows, states, pairs, comparisons = [], [], [], []
        for group in groups:
            scopes = [dict(s, base_diagnostics=base_diag.get(s["scope_sha256"], {})) for s in group["scopes"]]
            history_seconds = 0.0
            if group["state_kind"] == "after_one_actual_history_call":
                trajectory_file = parent / f"trajectory_{group['trajectory']:03d}.json"
                history_seconds = optional_float(load(trajectory_file).get("actual_history_call", {}).get("complete_conditional_return_seconds")) if trajectory_file.exists() else None
            request_rows = prepare_rows(graph_id, source, group, scopes, initial_seconds, history_seconds)
            rows.extend(request_rows)
            state_g2 = g2_pairs(request_rows); pairs.extend(state_g2)
            state_g3, repeatable = g3_pairs(request_rows); comparisons.extend(state_g3)
            valid = [r for r in request_rows if r["eligible_for_signal_analysis"]]
            opportunity = any(r["local_recovery_improvement_percent"] >= 1 and r["beyond_snapshot_added_seconds"] > 1e-9 for r in valid)
            state = {"source_group": source, "graph_id": graph_id, "snapshot_id": group["snapshot_id"],
                     "state_kind": group["state_kind"], "controller_state_identity": group["controller_state_identity"],
                     "request_rows": len(request_rows), "signal_eligible_rows": len(valid), "G1_extra_opportunity": opportunity,
                     "no_valid_actual_finite_outcomes": len(valid) == 0,
                     "G3_repeatable_200_1000ms_difference": repeatable,
                     "G2_summary_similar_pairs": len(state_g2),
                     "G2_differing_summary_similar_pairs": sum(p["meaningful_result_difference"] for p in state_g2),
                     "max_added_seconds": max((r["beyond_snapshot_added_seconds"] for r in valid), default=0.0),
                     "max_local_recovery_improvement_percent": max((r["local_recovery_improvement_percent"] for r in valid), default=0.0),
                     "max_added_global_percent": max((r["beyond_snapshot_global_percent"] for r in valid), default=0.0),
                     "not_independent_physical_source": True}
            states.append(state)
        scope_records = list(unique_scopes.values())
        costs = {}
        for budget in sorted({r["workpoint_ms"] for r in rows}):
            budget_rows = [r for r in rows if r["workpoint_ms"] == budget]
            fields = ("preparation_seconds", "process_launch_seconds", "native_process_seconds", "parse_seconds",
                      "native_validation_and_rescore_seconds", "extra_validation_and_rescore_seconds",
                      "complete_conditional_return_seconds", "earliest_verified_added_gain_seconds",
                      "one_shot_problem_to_return_estimate_seconds")
            costs[str(budget)] = {field: metrics([r[field] for r in budget_rows]) for field in fields}
        graph = {"graph_id": graph_id, "source_group": source, "split": metadata.get("split"),
                 "controller_states": len(states), "target_states": summary.get("target_unique_states"),
                 "request_rows": len(rows), "signal_eligible_rows": sum(r["eligible_for_signal_analysis"] for r in rows),
                 "states_without_valid_actual_finite_outcomes": sum(s["no_valid_actual_finite_outcomes"] for s in states),
                 "failed_rows": sum(r["failed"] is True for r in rows), "zero_extra_gain_rows": sum(r["zero_extra_gain"] is True for r in rows),
                 "unavailable_for_allocation_rows": sum(r["available_for_allocation"] is not True for r in rows),
                 "G1_opportunity_states": sum(s["G1_extra_opportunity"] for s in states),
                 "G1_fraction": sum(s["G1_extra_opportunity"] for s in states) / len(states) if states else None,
                 "G3_repeatable_states": sum(s["G3_repeatable_200_1000ms_difference"] for s in states),
                 "G3_fraction": sum(s["G3_repeatable_200_1000ms_difference"] for s in states) / len(states) if states else None,
                 "G2_similar_pairs": len(pairs), "G2_meaningfully_differing_similar_pairs": sum(p["meaningful_result_difference"] for p in pairs),
                 "G2_exact_summary_differing_pairs": sum(p["exact_summary_at_1e_minus7"] and p["meaningful_result_difference"] for p in pairs),
                 "unique_actual_scopes": len(scope_records),
                 "cap_truncated_scope_fraction": sum(s["cap_truncated"] for s in scope_records) / len(scope_records) if scope_records else None,
                 "mean_local_cross_owner_edge_fraction": float(np.mean([s["local_cross_owner_edges"] / max(1, s["local_edges"]) for s in scope_records])) if scope_records else None,
                 "mean_fixed_base_blocked_fraction": float(np.mean([x["fixed_base_blocked_fraction_outside_base"] for x in base_diag.values() if "fixed_base_blocked_fraction_outside_base" in x])) if any("fixed_base_blocked_fraction_outside_base" in x for x in base_diag.values()) else None,
                 "max_added_seconds": max((s["max_added_seconds"] for s in states), default=0.0),
                 "max_added_global_percent": max((s["max_added_global_percent"] for s in states), default=0.0),
                 "caller_deadline_result": "UNMEASURED", "graph_model_advantage_established": False}
        graph_rows.append(graph)
        graph_reports[graph_id] = {"acceptance": graph, "costs_by_native_workpoint_ms": costs,
                                   "scope_base_diagnostics": base_diag}
        state_rows.extend(states); all_rows.extend(rows); similar_pairs.extend(pairs); budget_pairs.extend(comparisons)
        input_manifest.append({"summary_path": str(summary_path), "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                               "snapshot_files": len(groups)})
    source_rows = []
    for source in sorted({g["source_group"] for g in graph_rows}):
        views = [g for g in graph_rows if g["source_group"] == source]
        g1_available = [g["G1_fraction"] for g in views if g["G1_fraction"] is not None]
        g3_available = [g["G3_fraction"] for g in views if g["G3_fraction"] is not None]
        source_rows.append({"source_group": source, "derived_graphs": len(views),
                            "G1_equal_graph_mean_fraction": float(np.mean(g1_available)) if g1_available else None,
                            "G3_equal_graph_mean_fraction": float(np.mean(g3_available)) if g3_available else None,
                            "controller_states_are_not_extra_sources": True})
    opportunities = [g["G1_fraction"] for g in graph_rows if g["G1_fraction"] is not None]
    cap_fraction = metrics([g["cap_truncated_scope_fraction"] for g in graph_rows])["mean"]
    recommendations = []
    if not graph_rows:
        recommendations.append("真实P0记录尚未到齐：保持等待，不由空报告判定无恢复机会。")
    elif not any(g["signal_eligible_rows"] for g in graph_rows):
        recommendations.append("尚无可用的实际有限监督结果：先排查native失败或不可分配原因，不能由失败行断言物理数据没有恢复机会，更不能先训练模型。")
    elif opportunities and max(opportunities) < 0.30:
        recommendations.append("已观测各图G1尚未达到30%开发参考线；零机会状态继续保留，先不启动大模型训练。")
        if cap_fraction is not None and cap_fraction > 0:
            recommendations.append("实际存在cap256截断：可优先考虑仅一次、全部对照同步的cap512诊断；截断率本身不证明被截顶点包含有益组合。")
        else:
            recommendations.append("尚无cap截断证据：若继续诊断，只考虑计划允许的一次单壳168星几何对照；同时报告已测耦合与base封锁，不承诺产生优势。")
    else:
        recommendations.append("有图出现真实额外机会：若P0完整，进入小模型同状态排序与固定调用数比较；保留CheapSummary、P1和Greedy，不把native收益归给学习。")
    if graph_rows and not any(g["G3_fraction"] and g["G3_fraction"] >= 0.20 for g in graph_rows):
        recommendations.append("尚未观察足够重复的200/1000ms有限结果差异：先检查平台、域与饱和阶段；本批可能支持恢复值排序，但不能支持预算头优势。")
    expected_graphs = 12
    analysis = {"status": "ACTUAL_P0_ANALYSIS" if graph_rows else "WAITING_FOR_ACTUAL_P0_DATA",
                "complete_expected_p0_graphs": len(graph_rows) == expected_graphs,
                "expected_p0_graphs": expected_graphs, "actual_analyzed_graphs": len(graph_rows),
                "physical_source_group_count": len(source_rows), "derived_graphs_are_not_independent_sources": True,
                "request_rows": len(all_rows), "controller_states": len(state_rows),
                "learning_advantage_established": False, "caller_strict_deadline_test_performed": False,
                "G1_rule": "available, valid actual finite result; local recovery >=1% and q+w(T)>snapshot current best gain",
                "G1_30_percent_is_development_reference_not_test_filter": True,
                "G1_fraction_is_observed_opportunity_lower_bound_when_failures_remain": True,
                "G2_rule": {"within_same_snapshot_and_workpoint": True, "size_relative_tolerance": SIZE_RELATIVE_TOLERANCE,
                            "q_L_U_P1_relative_tolerance_each": SUMMARY_RELATIVE_TOLERANCE,
                            "meaningful_result_difference_relative_to_recovery": RESULT_RELATIVE_TOLERANCE,
                            "approximate_matching_not_proof_of_graph_necessity": True},
                "G3_rule": "same frozen scope, warm, native seed and repeat; 200 vs1000ms; at least2 valid repeats same nonzero direction",
                "G3_20_percent_is_development_reference_not_test_filter": True,
                "G4_rule": "Full measured conditional cost reported. Caller deadlines absent; native workpoint overruns are not caller lateness.",
                "source_group_summaries": source_rows, "graphs": graph_reports, "recommendations": recommendations,
                "inputs": input_manifest, "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "completed_utc": datetime.now(timezone.utc).isoformat()}
    (destination / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(destination / "graph_acceptance.csv", graph_rows)
    write_csv(destination / "source_group_summary.csv", source_rows)
    write_csv(destination / "state_summary.csv", state_rows)
    write_csv(destination / "all_requests.csv", all_rows)
    write_csv(destination / "summary_similar_request_pairs.csv", similar_pairs)
    write_csv(destination / "paired_200_1000ms_results.csv", budget_pairs)
    report = ["# P0实际恢复机会与预算诊断", "", f"已分析{len(graph_rows)}/12张配置图、{len(source_rows)}个物理来源、{len(state_rows)}个实际去重控制状态、{len(all_rows)}条请求记录。",
              "同一来源的R8/R12、gap和状态均不是额外独立物理观测；目前没有训练或学习优势结论。失败、零收益与不可分配请求全部保留在all_requests.csv。", "",
              "| 图 | 状态数 | G1额外机会↑ | G3重复预算差异↑ | cap截断 | 最大额外秒数↑ | 最大完整目标增加%↑ |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for graph in graph_rows:
        pct = lambda value: "未知" if value is None else f"{100 * value:.1f}%"
        report.append(f"| {graph['graph_id']} | {graph['controller_states']} | {pct(graph['G1_fraction'])} | {pct(graph['G3_fraction'])} | {pct(graph['cap_truncated_scope_fraction'])} | {graph['max_added_seconds']:.6f} | {graph['max_added_global_percent']:.6f}% |")
    report.extend(["", "G1同时要求实际局部恢复至少1%且净收益超过该状态已验证最好值；30%只是是否继续投入的开发参考，不能删除零机会测试状态。局部恢复比例、完整调度增加秒数和完整目标比例已分别保存。没有有效有限执行结果的状态单列为未知；存在失败时，上述G1是已观察机会比例，不能把未知状态当作已证实没有机会。",
                   "", "G2只比较同状态、同workpoint的摘要相似请求；近似匹配后的有限结果差异仍可能受摘要差异和求解过程影响，不是图编码必要性或Capacity优于CheapSummary的证据。G3要求同warm、同种子、两次重复均同方向，20%也是开发参考。",
                   "", "G4保留准备、启动、native进程、解析、两层完整验证以及实际条件返回耗时。原始全图贪心和共同前缀另列。没有真实caller deadline收据，严格迟到率为未测；不能把native进程超出200/1000ms当作完整策略迟到。",
                   "", "下一步建议：", ""] + [f"- {x}" for x in recommendations])
    (destination / "P0_ANALYSIS_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "graphs": len(graph_rows), "sources": len(source_rows),
                      "requests": len(all_rows), "report": str(destination / "P0_ANALYSIS_ZH.md")}, ensure_ascii=False), flush=True)
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--execution-root", type=Path, action="append")
    args = parser.parse_args()
    analyze(args.dataset_root.resolve(), [p.resolve() for p in args.execution_root] if args.execution_root else None)
