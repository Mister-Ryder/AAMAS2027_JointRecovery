"""Extract up to three traced, actual positive P0 recoveries; never synthesize.

Cases are explanatory examples chosen from completed, explicitly selected
collections. They are not an unbiased performance estimate or evidence of a
graph learner's advantage. This tool does not change a graph, request, or label.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class GraphView:
    def __init__(self, path):
        self.path = path
        self.sha256 = file_sha256(path)
        with np.load(path, allow_pickle=False) as graph:
            columns = ("weights", "owner", "edges", "edge_types", "start_ns", "end_ns", "contact_id",
                       "satellite_id", "site_id", "antenna_id", "start_decimal_seconds", "end_decimal_seconds",
                       "duration_decimal_seconds", "epoch_utc", "source_group", "graph_id", "ground_gap_seconds",
                       "satellite_gap_seconds", "parameters_sha256", "contacts_sha256")
            self.data = {key: np.array(graph[key], copy=True) for key in columns}
        self.n = len(self.data["weights"])
        edges = self.data["edges"]
        self.codes = edges[:, 0] * self.n + edges[:, 1]

    def edge_type(self, left, right):
        if left == right:
            return 0
        code = min(left, right) * self.n + max(left, right)
        position = int(np.searchsorted(self.codes, code))
        return int(self.data["edge_types"][position]) if position < len(self.codes) and self.codes[position] == code else 0

    def total(self, vertices):
        return math.fsum(float(self.data["weights"][v]) for v in vertices)

    def independent(self, vertices):
        mask = np.zeros(self.n, dtype=np.bool_)
        mask[list(vertices)] = True
        edges = self.data["edges"]
        return not bool(np.any(mask[edges[:, 0]] & mask[edges[:, 1]]))

    def member_set(self, vertices):
        vertices = sorted(set(map(int, vertices)))
        return {"count": len(vertices), "vertex_indices": vertices,
                "contact_ids": [str(self.data["contact_id"][v]) for v in vertices],
                "duration_sum_seconds": self.total(vertices)}

    def contact(self, vertex):
        result = {"vertex_index": int(vertex), "owner_satellite_index": int(self.data["owner"][vertex]),
                  "reward_seconds": float(self.data["weights"][vertex])}
        for key in ("contact_id", "satellite_id", "site_id", "antenna_id", "start_decimal_seconds", "end_decimal_seconds", "duration_decimal_seconds"):
            result[key] = str(self.data[key][vertex])
        result["epoch_utc"] = str(self.data["epoch_utc"].item() if self.data["epoch_utc"].shape == () else self.data["epoch_utc"][vertex])
        result["start_ns"] = int(self.data["start_ns"][vertex])
        result["end_ns"] = int(self.data["end_ns"][vertex])
        return result

    def relation(self, reference, other):
        kind = self.edge_type(reference, other)
        start_r, end_r = int(self.data["start_ns"][reference]), int(self.data["end_ns"][reference])
        start_o, end_o = int(self.data["start_ns"][other]), int(self.data["end_ns"][other])
        same_satellite = self.data["satellite_id"][reference] == self.data["satellite_id"][other]
        same_site = self.data["site_id"][reference] == self.data["site_id"][other]
        return {"reference_vertex": int(reference), "other_vertex": int(other),
                "reference_contact_id": str(self.data["contact_id"][reference]),
                "other_contact_id": str(self.data["contact_id"][other]), "edge_type": kind,
                "ground_conflict": bool(kind & 1), "satellite_conflict": bool(kind & 2),
                "same_satellite_cross_site": bool(same_satellite and not same_site),
                "same_site_cross_satellite": bool(same_site and not same_satellite),
                "actual_time_overlap_seconds": max(0, min(end_r, end_o) - max(start_r, start_o)) / 1e9,
                "other_extends_left": start_o < start_r, "other_extends_right": end_o > end_r,
                "other_fully_contained_in_reference": start_r <= start_o and end_o <= end_r}


def positive_rows(batch_outs):
    candidates, sources, used_graphs = [], [], set()
    for batch_out in batch_outs:
        summary_paths = sorted(batch_out.rglob("summary.json"))
        for summary_path in summary_paths:
            summary = load_json(summary_path)
            if "graph" not in summary:
                continue
            metadata = summary["graph"]
            graph_id = metadata["graph_id"]
            if graph_id in used_graphs:
                raise ValueError("Repeated graph across selected collections; select one protocol per graph")
            used_graphs.add(graph_id)
            groups = [load_json(path) for path in sorted(summary_path.parent.glob("snapshot_*.json"))]
            expected_states = summary.get("target_unique_states")
            actual_states = summary.get("actual_unique_controller_states")
            if actual_states != len(groups):
                raise ValueError("Completed graph summary and actual snapshot file count differ")
            sources.append({"graph_id": graph_id, "source_group": metadata["source_group"],
                            "summary_path": str(summary_path), "summary_sha256": file_sha256(summary_path),
                            "target_states": expected_states, "actual_states": actual_states,
                            "deduplicated_shortfall_retained": actual_states != expected_states})
            for path in sorted(summary_path.parent.glob("snapshot_*.json")):
                group = load_json(path)
                by_action = {int(s["action_index"]): s for s in group["scopes"]}
                for request_index, row in enumerate(group.get("requests", [])):
                    if not (row.get("available_for_allocation") is True and row.get("failed") is False
                            and row.get("actual_finite_supervision_valid") is True
                            and row.get("complete_membership_valid") is True
                            and float(row.get("beyond_snapshot_gain_seconds", 0)) > 1e-9
                            and float(row["recovery_value_seconds"]) > float(row["L_seconds"]) + 1e-9):
                        continue
                    candidates.append({"metadata": metadata, "group": group, "scope": by_action[int(row["action_index"])],
                                       "row": row, "snapshot_path": path, "request_index": request_index})
    candidates.sort(key=lambda case: (-float(case["row"]["beyond_snapshot_gain_seconds"]),
                                      case["metadata"]["graph_id"], case["group"]["snapshot_id"],
                                      case["row"]["action_index"], case["row"]["workpoint_ms"], case["row"]["repeat"]))
    unique, identities = [], set()
    for case in candidates:
        row = case["row"]
        identity = (case["metadata"]["graph_id"], row["scope_sha256"],
                    tuple(row["actual_warm_members"]), tuple(row["returned_recovery_members"]))
        if identity not in identities:
            unique.append(case); identities.add(identity)
    return unique, sources, len(candidates)


def extract_case(candidate, graph):
    scope, row, group = candidate["scope"], candidate["row"], candidate["group"]
    if candidate["metadata"]["npz_sha256"] != graph.sha256:
        raise ValueError("Original graph hash differs from execution metadata")
    sets = {"B_fixed_base_including_commitments": set(scope["base"]), "D_displaced": set(scope["displaced"]),
            "C_commitments": set(scope["inserts"]), "E_declared_extra_releases": set(scope.get("releases", [])),
            "R_actual_capped_recovery_domain": set(scope["replacements"]),
            "W_actual_execution_warm": set(row["actual_warm_members"]), "T_actual_returned_recovery": set(row["returned_recovery_members"])}
    base, displaced, commitments = sets["B_fixed_base_including_commitments"], sets["D_displaced"], sets["C_commitments"]
    domain, warm, recovered = sets["R_actual_capped_recovery_domain"], sets["W_actual_execution_warm"], sets["T_actual_returned_recovery"]
    complete = base | recovered
    if not commitments <= base or not warm <= domain or not recovered <= domain or base & domain:
        raise ValueError("Recorded case violates fixed-base/recovery membership contract")
    if not graph.independent(base | warm) or not graph.independent(complete):
        raise ValueError("Actual warm or returned complete case is infeasible in original graph")
    q, warm_value, recovery_value = graph.total(commitments) - graph.total(displaced), graph.total(warm), graph.total(recovered)
    for actual, recorded in ((q, row["q_seconds"]), (warm_value, row["L_seconds"]),
                             (recovery_value, row["recovery_value_seconds"]), (graph.total(complete), row["complete_value_seconds"])):
        if not math.isclose(actual, float(recorded), rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError("Original-duration rescore differs from actual request record")
    added = q + recovery_value - float(group["snapshot"]["best_gain_seconds"])
    if added <= 1e-9 or not math.isclose(added, float(row["beyond_snapshot_gain_seconds"]), rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError("Selected case is not an actual positive increment beyond the held snapshot")
    new_vertices = recovered - warm
    removed_warm = warm - recovered
    # Explain native changes against actually displaced or superseded warm contacts.
    references = sorted(displaced | removed_warm)
    bundles = []
    for reference in references:
        neighbours = sorted(v for v in new_vertices if graph.edge_type(reference, v))
        if len(neighbours) < 2:
            continue
        compatible = all(graph.edge_type(a, b) == 0 for a, b in itertools.combinations(neighbours, 2))
        if not compatible:
            raise ValueError("Claimed recovered combination contains an original conflict")
        relations = [graph.relation(reference, v) for v in neighbours]
        bundles.append({"reference_vertex": reference, "reference_contact_id": str(graph.data["contact_id"][reference]),
                        "reference_is_displaced": reference in displaced, "reference_is_superseded_actual_warm": reference in removed_warm,
                        "reference_reward_seconds": float(graph.data["weights"][reference]),
                        "joint_recovered_vertices": neighbours, "joint_recovered_contact_ids": [str(graph.data["contact_id"][v]) for v in neighbours],
                        "joint_recovered_duration_seconds": graph.total(neighbours), "all_pairwise_compatible": True,
                        "has_left_extension": any(x["other_extends_left"] for x in relations),
                        "has_right_extension": any(x["other_extends_right"] for x in relations),
                        "has_same_satellite_cross_site": any(x["same_satellite_cross_site"] for x in relations),
                        "has_same_site_cross_satellite": any(x["same_site_cross_satellite"] for x in relations),
                        "member_relations_to_reference": relations,
                        "bundle_duration_difference_is_not_the_complete_case_gain": True})
    bundles.sort(key=lambda x: (-len(x["joint_recovered_vertices"]), -x["joint_recovered_duration_seconds"], x["reference_vertex"]))
    focus_vertices = sorted(new_vertices, key=lambda v: (-float(graph.data["weights"][v]), v))[:12]
    compatibility = [{"left": a, "right": b, "left_contact_id": str(graph.data["contact_id"][a]),
                      "right_contact_id": str(graph.data["contact_id"][b]), "edge_type": graph.edge_type(a, b),
                      "jointly_compatible": graph.edge_type(a, b) == 0}
                     for a, b in itertools.combinations(focus_vertices, 2)]
    relevant = sorted(domain | displaced | commitments | removed_warm)
    mask = np.zeros(graph.n, dtype=np.bool_); mask[relevant] = True
    edge_mask = mask[graph.data["edges"][:, 0]] & mask[graph.data["edges"][:, 1]]
    relevant_edges, relevant_types = graph.data["edges"][edge_mask], graph.data["edge_types"][edge_mask]
    conflicts = [{"left": int(u), "right": int(v), "edge_type": int(kind),
                  "ground_conflict": bool(kind & 1), "satellite_conflict": bool(kind & 2)}
                 for (u, v), kind in zip(relevant_edges, relevant_types)]
    original_value = float(group["snapshot"]["original_value_seconds"])
    return {
        "evidence_status": "ACTUAL_VALIDATED_POSITIVE_REQUEST", "graph_id": candidate["metadata"]["graph_id"],
        "source_group": candidate["metadata"]["source_group"], "snapshot_id": group["snapshot_id"],
        "controller_state_identity": group["controller_state_identity"], "state_kind": group["state_kind"],
        "action_index": row["action_index"], "workpoint_ms": row["workpoint_ms"], "repeat": row["repeat"],
        "scope_sha256": row["scope_sha256"], "native_seed": row["diagnostics"].get("seed"),
        "native_binary_sha256": row["diagnostics"].get("binary_sha256"),
        "trace": {"snapshot_path": str(candidate["snapshot_path"]), "snapshot_file_sha256": file_sha256(candidate["snapshot_path"]),
                  "request_index_in_snapshot_requests": candidate["request_index"], "request_sha256": value_sha256(row),
                  "graph_npz_path": str(graph.path), "graph_npz_sha256": graph.sha256,
                  "contacts_sha256": str(graph.data["contacts_sha256"].item()),
                  "parameters_sha256": str(graph.data["parameters_sha256"].item())},
        "member_sets": {name: graph.member_set(vertices) for name, vertices in sets.items()},
        "new_T_minus_W": graph.member_set(new_vertices), "removed_W_minus_T": graph.member_set(removed_warm),
        "q_seconds": q, "actual_W_seconds": warm_value, "actual_T_seconds": recovery_value,
        "local_recovery_improvement_percent": 100 * (recovery_value - warm_value) / max(1, warm_value),
        "snapshot_best_gain_seconds": group["snapshot"]["best_gain_seconds"], "beyond_snapshot_added_seconds": added,
        "beyond_snapshot_global_percent": 100 * added / max(1, original_value),
        "full_B_union_T_duration_seconds": graph.total(complete), "complete_original_graph_feasible": True,
        "all_T_pairwise_compatible": True, "all_T_compatible_with_entire_fixed_B": True,
        "ground_gap_seconds": int(graph.data["ground_gap_seconds"].item()),
        "satellite_gap_seconds": int(graph.data["satellite_gap_seconds"].item()),
        "contacts_in_recovery_explanation": [graph.contact(v) for v in relevant],
        "relevant_resource_conflict_edges": conflicts,
        "joint_recovery_bundles": bundles[:3], "joint_recovery_bundle_count": len(bundles),
        "largest_new_members_compatibility_pairs": compatibility,
        "complete_conditional_return_seconds": row["complete_conditional_return_seconds"],
        "earliest_verified_added_gain_seconds": row.get("earliest_verified_more_than_snapshot_gain_seconds"),
        "diagnostics": row["diagnostics"], "caller_hard_deadline_verified": False,
        "graph_learning_advantage_established": False,
        "simple_example_is_not_a_CheapSummary_comparison": True,
    }


def main(root, batch_outs, limit):
    destination = root / "reports" / "P0_STRUCTURAL_CASES"
    destination.mkdir(parents=True, exist_ok=True)
    candidates, sources, positive_row_count = positive_rows(batch_outs)
    chosen, selected_graphs = [], set()
    # Fixed rule: largest actual extra increment per graph first, then fill
    # unused unique scope/warm/return identities, up to the requested small cap.
    for candidate in candidates:
        graph_id = candidate["metadata"]["graph_id"]
        if graph_id not in selected_graphs and len(chosen) < limit:
            chosen.append(candidate); selected_graphs.add(graph_id)
    for candidate in candidates:
        if len(chosen) >= limit:
            break
        if not any(candidate is old for old in chosen):
            chosen.append(candidate)
    status = ("ACTUAL_CASES_EXTRACTED" if chosen else
              "NO_ACTUAL_POSITIVE_CASE_AVAILABLE" if sources else "WAITING_FOR_COMPLETED_P0_COLLECTION")
    index = {"status": status,
             "selected_batch_outs": [str(p) for p in batch_outs], "completed_graph_collections": sources,
             "positive_request_rows_before_deduplication": positive_row_count,
             "distinct_positive_scope_warm_return_outcomes": len(candidates), "selected_case_count": len(chosen),
             "case_selection_rule": "Largest actual added seconds; distinct graph first, then unique scope/warm/return. Illustrative selection only.",
             "cases_are_not_independent_physical_sources_or_performance_estimates": True,
             "endpoint_serialization_note": "9-digit decimal export describes the frozen numerical graph; not nanosecond physical accuracy.",
             "graph_learning_advantage_established": False, "cases": [],
             "tool_sha256": file_sha256(Path(__file__)), "created_utc": datetime.now(timezone.utc).isoformat()}
    text = ["# P0真实恢复结构案例", "", f"选定完整收据中有{positive_row_count}条真实正收益请求；去除重复的scope/warm/return后为{len(candidates)}种结果。本次提取{len(chosen)}个说明性案例。",
            "按完整已验证状态之外的实际增加秒数排序，优先不同图，最多3例；这是说明性选择，不代表总体胜率、来源独立性或学习优势。所有原始零机会和失败请求仍保存在完整实验收据。", ""]
    if not chosen:
        text.extend(["没有可提取的真实正收益案例。工具没有拼接窗口、改权重、合成恢复组合或用上界替代实际结果。",
                     "若尚无完整summary，表示实际批次尚未到齐，不能据此认定物理数据没有机会。", ""])
    for number, candidate in enumerate(chosen, 1):
        graph = GraphView(root / "graphs" / f"{candidate['metadata']['graph_id']}.npz")
        case = extract_case(candidate, graph)
        case_name = f"case_{number:02d}.json"
        (destination / case_name).write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index["cases"].append({"file": case_name, "graph_id": case["graph_id"], "source_group": case["source_group"],
                               "snapshot_id": case["snapshot_id"], "action_index": case["action_index"],
                               "beyond_snapshot_added_seconds": case["beyond_snapshot_added_seconds"],
                               "joint_recovery_bundle_count": case["joint_recovery_bundle_count"]})
        text.extend([f"## 案例 {number}：{case['graph_id']} / {case['snapshot_id']} / 请求 {case['action_index']}", "",
                     f"实际workpoint={case['workpoint_ms']}ms，repeat={case['repeat']}。W={case['actual_W_seconds']:.9f}s，T={case['actual_T_seconds']:.9f}s；局部恢复提高{case['local_recovery_improvement_percent']:.6f}%。",
                     f"相对该状态已持有最好调度，完整目标真实增加{case['beyond_snapshot_added_seconds']:.9f}s（{case['beyond_snapshot_global_percent']:.9f}%）；条件调用完整返回耗时{case['complete_conditional_return_seconds']:.6f}s。", "",
                     f"B/C/D/R/W/T完整原始顶点及contact_id见[{case_name}]({case_name})。T已在原图中验证互兼容且与整个固定B兼容；奖励来自冻结完整窗口的原端点差。"])
        if case["joint_recovery_bundles"]:
            for bundle in case["joint_recovery_bundles"]:
                text.append(f"- 参考窗口{bundle['reference_contact_id']}：原时长{bundle['reference_reward_seconds']:.9f}s；对应{len(bundle['joint_recovered_vertices'])}个实际新恢复且彼此兼容的窗口，总时长{bundle['joint_recovered_duration_seconds']:.9f}s。左端延伸={bundle['has_left_extension']}，右端延伸={bundle['has_right_extension']}，同星跨站={bundle['has_same_satellite_cross_site']}，同站跨星={bundle['has_same_site_cross_satellite']}。")
            text.extend(["组合时长与某个参考窗口的差不是完整案例净收益；实际收益以上述q+w(T)与已持有最好值比较为准。", ""])
        else:
            text.extend(["本例存在真实整体恢复增益，但未找到至少两个新T成员同时替代某个D或失去W窗口的局部见证。工具保留该事实，不合成二换一示意。", ""])
    text.extend(["这些关系见证说明实际执行发生了什么；没有证明图模型优于CheapSummary，也没有验证完整caller截止时间。下一步仍需同状态、同执行选项的实际模型与强廉价对照比较。", ""])
    (destination / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "README_ZH.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"status": index["status"], "positive_rows": positive_row_count,
                      "cases": len(chosen), "report": str(destination / "README_ZH.md")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-out", type=Path, action="append", required=True,
                        help="Explicit completed P0 collection out directory; repeat for disjoint batches.")
    parser.add_argument("--limit", type=int, default=3, choices=[0, 1, 2, 3])
    arguments = parser.parse_args()
    main(arguments.dataset_root.resolve(), [path.resolve() for path in arguments.batch_out], arguments.limit)
