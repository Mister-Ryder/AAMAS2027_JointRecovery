"""Local sealed ample-allowance quality results; no experiments or live rows.

Primary: produced complete feasible schedule objective/gain in raw seconds.
Incomplete/native-empty/failure/late rows are retained, quality stays unknown.
Caller cost is separate; strict deadline delivery is secondary, never substituted
for solver production or full-quality performance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from submitted_comparison_report import (FULL_LABELS, MEAN_LABELS, DISPLAY_ORDER,
    BLOCKS, CONFIGS, read, sha, finite, exact_mean, write_csv, CHILS_URL, SCIPY_URL, HIGHS_URL)
from jointrecovery_method_names import method_display, mapping_markdown, METHOD_MAPPING

SCHEMA = "joint_recovery_stk_quality_results_report_v1"
ALLOWANCE_SCHEMA = "joint_recovery_stk_quality_allowance_v1"
PUBLIC = ("CHILS-p1", "HiGHS-MILP")
EPS = 1e-7
METRICS = {
    "initial_S_seconds": "derived_initial_S_seconds",
    "complete_objective_seconds": "derived_complete_objective_seconds",
    "complete_gain_seconds": "derived_complete_gain_seconds",
    "complete_gain_percent": "derived_complete_gain_percent",
    "caller_seconds": "external_caller_observed_return_seconds",
    "whole_cli_seconds_including_resident_setup": "whole_cli_wall_seconds_including_resident_setup",
    "complete_result_rate": "derived_complete_result",
    "native_output_rate": "derived_native_output",
    "external_fallback_only_rate": "derived_external_fallback_only",
    "legit_no_native_required_rate": "derived_legit_no_native_required",
    "on_time_return_rate": "derived_on_time",
    "observed_late_rate": "derived_late",
    "submission_failure_rate": "derived_failure",
    "unknown_completion_rate": "derived_completion_unknown",
    "actual_native_calls": "actual_native_calls",
    "verified_native_responses": "observed_verified_native_responses",
    "failed_completed_requests": "failed_completed_requests",
    "strict_gain_seconds_secondary": "strict_delivered_gain_including_failed_runs_seconds",
    "actual_return_objective_seconds_diagnostic": "returned_value_seconds",
    "paid_prefix_seconds_diagnostic": "paid_shared_prefix_gain_seconds",
    "beyond_paid_prefix_seconds_diagnostic": "actual_gain_beyond_paid_shared_prefix_seconds",
}


def load_allowance(path, inputs):
    if not path.is_file():
        return None, ["WAITING: selected_allowance.json not available"]
    record = read(path)
    inputs.append(path)
    d = record.get("selected_deadline_seconds")
    valid = (record.get("schema") == ALLOWANCE_SCHEMA and
             record.get("status") == "QUALITY_ALLOWANCE_PROTOCOL" and
             record.get("stage") == "frozen_from_development_completion" and
             record.get("selection_uses_gain") is False and record.get("test_outcomes_used") is False and
             record.get("development_sources") == [4] and d in (60., 120., 300.) and
             record.get("evaluation_deadline_seconds") == sorted(set([10., 30., 60., 120., d])))
    return record, [] if valid else ["Selected quality allowance is not the frozen DEV-completion-only 60/120/300 protocol"]


def load_block(root, phase, allowance, inputs):
    summary_path = root / phase / "summary.json"
    rows_path = root / phase / "actual_run_rows.json"
    deadlines = allowance.get("evaluation_deadline_seconds", []) if allowance else []
    expected = len(BLOCKS[phase]) * 6 * len(FULL_LABELS) * len(deadlines) if deadlines else None
    receipt = dict(status="WAITING", expected_cells=expected, received_cells=None,
                   row_file_read=False, issues=[], summary_path=str(summary_path))
    if allowance is None or not summary_path.is_file():
        return [], receipt
    summary = read(summary_path)
    inputs.append(summary_path)
    receipt["summary_status"] = summary.get("status")
    if summary.get("status") != "QUALITY_COMPARISON_MATRIX_COMPLETE":
        return [], receipt
    if (summary.get("expected_cells") != expected or summary.get("received_cells") != expected or
            not rows_path.is_file()):
        receipt.update(status="WAITING_FOR_COMPLETE_LOCAL_MIRROR", received_cells=summary.get("received_cells"))
        return [], receipt
    values = read(rows_path)
    inputs.append(rows_path)
    receipt.update(status="COMPLETED", row_file_read=True, received_cells=len(values))
    if len(values) != expected:
        receipt.update(status="INCOMPLETE_LOCAL_MIRROR", issues=["Declared / local row count differs"])
        return [], receipt
    seen, configs, budgets = set(), defaultdict(dict), set()
    for row in values:
        key = (row.get("graph_id"), row.get("policy_label"), row.get("deadline_seconds"))
        if key in seen:
            receipt["issues"].append("Duplicate quality cell: " + str(key))
        seen.add(key)
        if row.get("source") not in BLOCKS[phase] or row.get("policy_label") not in FULL_LABELS:
            receipt["issues"].append("Unexpected quality source/policy: " + str(key))
        if row.get("deadline_seconds") not in deadlines:
            receipt["issues"].append("Quality row does not use a declared curve allowance: " + str(key))
        budgets.add(row.get("budget_id"))
        configs[row.get("source")][row.get("graph_id")] = (
            row.get("station_view"), row.get("ground_gap_seconds"), row.get("satellite_gap_seconds"))
    receipt["evaluation_deadline_seconds"] = deadlines
    for source in BLOCKS[phase]:
        if len(configs[source]) != 6 or set(configs[source].values()) != CONFIGS:
            receipt["issues"].append("Missing/changed graph configurations for r%03d" % source)
    expected_keys = {(graph, policy, d) for source in BLOCKS[phase] for graph in configs[source]
                     for policy in FULL_LABELS for d in deadlines}
    if seen != expected_keys:
        receipt["issues"].append("Declared method / source / time curve grid differs")
    if summary.get("initial_mask_mismatches"):
        receipt["issues"].append("Driver initial S mask mismatch")
    return values, receipt


def derive(raw_rows, issues):
    anchors, masks, hashes = defaultdict(list), defaultdict(set), defaultdict(set)
    for row in raw_rows:
        graph = row["graph_id"]
        if row.get("report_usable") and finite(row.get("initial_value_seconds")):
            anchors[graph].append(row["initial_value_seconds"])
        if row.get("initial_mask_sha256"):
            masks[graph].add(row["initial_mask_sha256"])
        if row.get("graph_sha256_from_metadata"):
            hashes[graph].add(row["graph_sha256_from_metadata"])
    initial = {}
    for graph in {row["graph_id"] for row in raw_rows}:
        a = anchors[graph]
        valid = bool(a) and all(math.isclose(v, a[0], rel_tol=1e-12, abs_tol=EPS) for v in a)
        if len(masks[graph]) > 1 or len(hashes[graph]) > 1 or (a and not valid):
            issues.append("Common original S/graph identity differs: " + graph)
            valid = False
        initial[graph] = a[0] if valid else None
    result = []
    for raw in raw_rows:
        row = dict(raw)
        usable = row.get("report_usable") is True
        claimed_complete = row.get("complete_result_produced") is True
        membership_valid = row.get("returned_membership_full_graph_valid") is True
        published_native_output = row.get("native_complete_output_produced") is True
        native_output = bool(published_native_output or
                             (row["policy_label"] not in PUBLIC and finite(row.get("observed_verified_native_responses")) and
                              row["observed_verified_native_responses"] > 0))
        legit_no_native = row.get("legit_no_native_required") is True
        reasons = list(row.get("completion_failure_reasons") or [])
        complete = bool(usable and claimed_complete and membership_valid and
                        row.get("initial_membership_full_graph_valid") is True and
                        row.get("stop_reason") == "finished" and row.get("missed_return_sample") is False and
                        finite(row.get("solution_quality_value_seconds")) and finite(row.get("returned_value_seconds")))
        if row["policy_label"] in PUBLIC and not published_native_output:
            complete = False
            reasons.append("No actual verified complete native output; external original S is not solver production")
        elif row["policy_label"] not in PUBLIC and not native_output and not legit_no_native:
            complete = False
            reasons.append("No verified native response and no legitimate no-native-required receipt")
        if claimed_complete and not complete:
            issues.append("Complete-result flag lacks required production/feasibility/finished receipt: " + row.get("run_id", row["graph_id"]))
        if complete and not math.isclose(row["solution_quality_value_seconds"], row["returned_value_seconds"], rel_tol=1e-12, abs_tol=EPS):
            complete = False
            issues.append("Complete quality objective differs from returned original-duration objective")
        objective = float(row["solution_quality_value_seconds"]) if complete else None
        s = initial[row["graph_id"]]
        gain = objective - s if finite(objective) and finite(s) else None
        if finite(gain) and gain < -EPS:
            issues.append("Returned quality below the common incumbent floor: " + row.get("run_id", row["graph_id"]))
        caller, deadline = row.get("external_caller_observed_return_seconds"), row.get("deadline_seconds")
        on_time = bool(usable and row.get("missed_return_sample") is False and finite(caller) and finite(deadline) and caller <= deadline)
        strict = row.get("strict_delivered_gain_including_failed_runs_seconds")
        if not finite(strict):
            issues.append("Driver strict delivery value missing; not filled with zero")
        row.update(derived_initial_S_seconds=s, derived_original_S_from_common_usable_graph=s is not None,
                   derived_complete_objective_seconds=objective, derived_complete_gain_seconds=gain,
                   derived_complete_gain_percent=100. * gain / max(1., s) if finite(gain) and finite(s) else None,
                   derived_complete_result=float(complete), derived_native_output=float(native_output),
                   derived_legit_no_native_required=float(legit_no_native),
                   derived_external_fallback_only=float(row.get("external_incumbent_fallback_only") is True or
                       (usable and not native_output and not legit_no_native)),
                   derived_on_time=float(on_time), derived_late=float(row.get("missed_return_sample") is True),
                   derived_failure=float(row.get("failure_or_unknown_outcome", not usable)),
                   derived_completion_unknown=float(not isinstance(row.get("complete_result_produced"), bool)),
                   derived_quality_status="COMPLETE_FEASIBLE_PRODUCED" if complete else "NOT_PRODUCED_OR_INCOMPLETE_NA",
                   derived_completion_failure_reasons=reasons,
                   report_scope="AMPLE_ALLOWANCE_PRIMARY_SOLUTION_QUALITY")
        result.append(row)
    return result


def aggregate_at_deadline(rows, phase, deadline, primary):
    rows = [row for row in rows if row["deadline_seconds"] == deadline]
    summary, source_rows, graph_rows = [], [], []
    for label in DISPLAY_ORDER:
        members = MEAN_LABELS.get(label, (label,))
        selected = [row for row in rows if row["policy_label"] in members]
        current_sources = []
        for source in BLOCKS[phase]:
            current = [r for r in selected if r["source"] == source]
            graphs = sorted({r["graph_id"] for r in current})
            current_graphs = []
            for graph in graphs:
                cells = [r for r in current if r["graph_id"] == graph]
                complete_grid = len(cells) == len(members) and {r["policy_label"] for r in cells} == set(members)
                values = {metric: exact_mean([r.get(field) for r in cells]) if complete_grid else None for metric, field in METRICS.items()}
                gr = dict(phase=phase, budget_id="quality", policy_label=label, source=source, graph_id=graph,
                          is_primary_quality_allowance=deadline == primary,
                          seed_count=len(members), grid_complete=complete_grid,
                          complete_quality_cells=sum(r["derived_complete_result"] for r in cells),
                          deadline_seconds=exact_mean([r.get("deadline_seconds") for r in cells]), **values)
                current_graphs.append(gr)
                graph_rows.append(gr)
            grid = len(current_graphs) == 6 and all(r["grid_complete"] for r in current_graphs)
            sr = dict(phase=phase, budget_id="quality", policy_label=label, source=source,
                      deadline_seconds=deadline, is_primary_quality_allowance=deadline == primary,
                      physical_source="r%03d" % source, graph_count=len(graphs), seed_count=len(members), grid_complete=grid,
                      declared_cells=6 * len(members), received_cells=len(current),
                      complete_quality_cells=sum(r["derived_complete_result"] for r in current),
                      **{m: exact_mean([r[m] for r in current_graphs]) if grid else None for m in METRICS})
            current_sources.append(sr)
            source_rows.append(sr)
        grid = all(r["grid_complete"] for r in current_sources)
        sr = dict(phase=phase, budget_id="quality", policy_label=label, seed_count=len(members), grid_complete=grid,
                  is_primary_quality_allowance=deadline == primary,
                  deadline_seconds=exact_mean([r.get("deadline_seconds") for r in selected]),
                  physical_source_count=len(BLOCKS[phase]), declared_cells=6 * len(BLOCKS[phase]) * len(members), received_cells=len(selected),
                  complete_quality_cells=sum(r["derived_complete_result"] for r in selected),
                  incomplete_quality_cells=sum(not r["derived_complete_result"] for r in selected),
                  observed_late_cells=sum(r["derived_late"] for r in selected),
                  submission_failure_cells=sum(r["derived_failure"] for r in selected),
                  no_native_output_cells=sum(not r["derived_native_output"] for r in selected),
                  **{m: exact_mean([r[m] for r in current_sources]) if grid else None for m in METRICS})
        summary.append(sr)
    return summary, source_rows, graph_rows


def aggregate(rows, phase, allowance):
    main, sources, graphs = [], [], []
    for deadline in allowance["evaluation_deadline_seconds"]:
        m, s, g = aggregate_at_deadline(rows, phase, deadline, allowance["selected_deadline_seconds"])
        index = {row["policy_label"]: row for row in m}
        for row in m:
            members = MEAN_LABELS.get(row["policy_label"], (row["policy_label"],))
            for metric in ("complete_objective_seconds", "complete_gain_seconds", "complete_gain_percent", "caller_seconds"):
                values = [index[label][metric] for label in members]
                known = all(finite(value) for value in values)
                row[metric + "_fit_seed_min"] = min(values) if known else None
                row[metric + "_fit_seed_max"] = max(values) if known else None
            row["fit_seed_range_is_not_CI"] = True
        main.extend(m); sources.extend(s); graphs.extend(g)
    return main, sources, graphs


def comparisons(summary, sources, comparable):
    index = {(r["phase"], r["deadline_seconds"], r["policy_label"]): r for r in summary}
    si = {(r["phase"], r["deadline_seconds"], r["policy_label"], r["source"]): r for r in sources}
    output = []
    for phase in BLOCKS:
      for deadline in sorted({r["deadline_seconds"] for r in summary if r["phase"] == phase}):
        for target in ("FullCapacity-seed17", "FullCapacity-seed29", "FullCapacity-mean17/29"):
            if (phase, deadline, target) not in index:
                continue
            for comparator in ("FullCheapSummary-mean17/29", "FullGreedy", "FullP1", *PUBLIC):
                a, b = index[(phase, deadline, target)], index[(phase, deadline, comparator)]
                av, bv = a["complete_objective_seconds"], b["complete_objective_seconds"]
                delta = av - bv if finite(av) and finite(bv) else None
                ds = []
                for source in BLOCKS[phase]:
                    av, bv = si[(phase, deadline, target, source)]["complete_objective_seconds"], si[(phase, deadline, comparator, source)]["complete_objective_seconds"]
                    ds.append(av - bv if finite(av) and finite(bv) else None)
                valid = comparable.get(phase) and finite(delta) and all(finite(v) for v in ds)
                output.append(dict(phase=phase, budget_id="quality", target=target, comparator=comparator,
                    deadline_seconds=deadline, is_primary_quality_allowance=a["is_primary_quality_allowance"],
                    comparable_complete_quality=bool(valid), delta_complete_objective_seconds=delta,
                    delta_complete_gain_seconds=delta,
                    result="HIGHER" if valid and delta > EPS else "LOWER" if valid and delta < -EPS else "TIED" if valid else "NOT_COMPARABLE_OR_NOT_PRODUCED",
                    physical_source_count=len(ds), source_differences_seconds=ds,
                    positive_source_count=sum(v > EPS for v in ds) if all(finite(v) for v in ds) else None,
                    negative_source_count=sum(v < -EPS for v in ds) if all(finite(v) for v in ds) else None))
    return output


def show(value, digits=3):
    return format(value, ".%df" % digits) if finite(value) else "N/A / 未完整产出"


def report_text(allowance, statuses, main, diffs, issues):
    lines = ["# JointRecovery（JR）：充裕预算下的完整求解质量", "",
        "**本文方法为 JointRecovery（JR，原始键 FullCapacity）。P1 / FullP1 是 Independent-replacement+CHILS 对照，不是本文方法。** 完整名称和原始键见 README_METHODS_ZH.md。", "",
        "本报告以完整求解后的原始目标值秒↑及相对共同初始方案 S 的增益↑为主。实际调用返回耗时秒↓单列；严格按时交付为次要诊断。短预算压力结果不用于替代这里的算法质量比较。", "",
        "只有完整封存的来源块才读取结果；不读取实时云端或正在进行的测试行，不训练、改模型或选择有利结果。", "",
        "## 固定求解余量与完成范围", ""]
    if allowance:
        lines += ["统一允许预算 D=%s 秒。该值来自 DEV r004 的预定 60→120→300 秒阶梯，以完成性选择并冻结，selection_uses_gain=%s、test_outcomes_used=%s；不按增益或验证/测试表现选择 D。D 是统一最大求解余量，不是每个方法实际用满的运行时长，也不证明精确最优。" % (show(allowance.get("selected_deadline_seconds"), 9), allowance.get("selection_uses_gain"), allowance.get("test_outcomes_used")), ""]
        lines += ["实际独立运行的曲线预算点：%s 秒。每点都是新运行；不复用一个长预算结果填短预算，不插值、平滑或只挑获胜预算。以下主表固定使用 selectedD，所有点在 CSV 保留。" % allowance.get("evaluation_deadline_seconds"), ""]
    else:
        lines += ["WAITING：尚未取得从 DEV 完成性冻结的统一求解余量。", ""]
    lines += ["| 来源块 | 状态 | 收到 / 预定行 | 结果行已读取 |", "|---|---|---:|---|"]
    for phase in ("test", "validation"):
        s = statuses[phase]
        lines.append("| %s | %s | %s / %s | %s |" % (phase, s["status"], s.get("received_cells"), s["expected_cells"], s["row_file_read"]))
    lines += ["", "测试独立单位为四个物理母源 r008–r011；每母源六配置等权，然后四母源等权。两种子均值先对同图两fit取均值，不把fit当新物理来源，不称种子范围为置信区间。", ""]
    for phase in ("test", "validation"):
        selected = [r for r in main if r["phase"] == phase and r["is_primary_quality_allowance"]]
        lines += ["## %s：完整求解质量（主表）" % phase, "",
                  "| 方法 | 原始 S 目标秒↑ | 完整求解目标秒↑ | 相对 S 增秒↑ | 相对 S 增益%↑ | 实际完整产出 / 全部行 |", "|---|---:|---:|---:|---:|---:|"]
        if not selected:
            lines += ["| JointRecovery（JR）及全部预定对照 | WAITING | WAITING | WAITING | WAITING | WAITING |"]
            continue
        for r in selected:
            label = method_display(r["policy_label"])
            ref = " [1]" if r["policy_label"] == "CHILS-p1" else " [2]" if r["policy_label"] == "HiGHS-MILP" else ""
            lines.append("| %s%s | %s | %s | %s | %s | %d / %d |" % (label, ref, show(r["initial_S_seconds"]), show(r["complete_objective_seconds"]), show(r["complete_gain_seconds"]), show(r["complete_gain_percent"]), r["complete_quality_cells"], r["declared_cells"]))
        lines += ["", "任何一项预定 cell 未完整产出方案时，完整质量总体均值为 N/A，不删失败后仅对成功样本求均值；逐来源/图的已知值仍全部保留在 CSV。未知质量没有补0，也没有用共同 S 冒充求解器新解。", "",
                  "### %s：实际时间成本与交付诊断（单独表）" % phase, "",
                  "| 方法 | caller 秒↓ | 含 resident setup 的 CLI 秒↓ | 完整产出率%↑ | 完整调用按时率%↑ | 提交失败数↓ | 迟到数↓ | 严格交付增秒↑（次要） |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in selected:
            lines.append("| %s | %s | %s | %s | %s | %d | %d | %s |" % (method_display(r["policy_label"]), show(r["caller_seconds"]), show(r["whole_cli_seconds_including_resident_setup"]), show(100. * r["complete_result_rate"] if finite(r["complete_result_rate"]) else None, 1), show(100. * r["on_time_return_rate"] if finite(r["on_time_return_rate"]) else None, 1), r["submission_failure_cells"], r["observed_late_cells"], show(r["strict_gain_seconds_secondary"])))
    lines += ["", "## 直接比较：JointRecovery 是否超过对照", "",
        "以下为 TEST 完整目标值差秒↑；共同 S 一致，因此完整目标值差等于增益差。仅在全部预定目标已实际完整产出且输入一致时判定高/低/平；没有完整公开求解输出时标 N/A，不能声称击败该方法。", "",
        "| JointRecovery fit | 对照 | 完整目标差秒↑ | 结论 |", "|---|---|---:|---|"]
    test_diffs = [d for d in diffs if d["phase"] == "test" and d["is_primary_quality_allowance"]]
    words = {"HIGHER": "高于", "LOWER": "低于", "TIED": "持平", "NOT_COMPARABLE_OR_NOT_PRODUCED": "N/A：未完整产出或不可比"}
    for row in test_diffs:
        lines.append("| %s | %s | %s | %s |" % (method_display(row["target"]), method_display(row["comparator"]), show(row["delta_complete_objective_seconds"]), words[row["result"]]))
    if not test_diffs:
        lines.append("| seed17、seed29及均值 | 全部对照 | WAITING | TEST 未封存 |")
    lines += ["", "## 必须保留的解释边界", "",
        "- 完整求解指程序正常结束并产出原图完整可行成员向量，不表示已经找到 MWIS 精确最优。HiGHS 没有原生完整可行候选输出时，外部父进程保留的 S 只作保底，不是求解成功。", 
        "- 所有失败、无 native 输出、未结束、迟到及零增益行均保留。原始未知值是 null；完整质量是 N/A。严格交付次要值按 driver 原记录保留失败0，不能据此填主质量值。", 
        "- 总收益包含共同贪心准备和实际 native 搜索。只有与同框架 JR-CheapSummary / Greedy-rank+CHILS 等匹配差值可描述学习请求分配的观测差，不把全部恢复收益归给学习。", 
        "- 本文完整 STK 迁移框架保留 11动作、cap256、四 workpoints、最多8请求和动态 warm/g*/spent；remaining-head 固定0，未验证剩余时间学习。完整框架不等于旧稿全部组件逐一验证。", 
        "- CHILS / HiGHS 在整图搜索，JointRecovery 受 cap256 恢复域限制；共同原图、原 S 和原时长目标一致，但搜索范围不同。HiGHS 是通用 MILP 软件参照；SciPy1.10.1 milp 无 solver 内 x0 warm-start。", 
        "- 充裕预算协议是用户提出质量优先要求后新增的独立确认，不覆盖或重定义先前已冻结的短预算实验；不追溯宣称所有后续比较在旧实验之前预注册。", "",
        "## 公开依据", "",
        "[1] Großmann, Langedal, Schulz. Concurrent Iterated Local Search for the Maximum Weight Independent Set Problem. SEA 2025，LIPIcs338:22。[正式会议页面](%s)。这里是官方单线程 p1 配置，不是多线程论文最佳配置。" % CHILS_URL, "",
        "[2] [SciPy1.10.1 milp 官方文档](%s)、[HiGHS 官方软件站](%s)。通用软件参照不冒充第二篇专门 MWIS 论文算法。" % (SCIPY_URL, HIGHS_URL)]
    if issues:
        lines += ["", "## 输入异常（不删结果）", ""] + ["- " + x for x in sorted(set(issues))]
    return "\n".join(lines) + "\n"


def run(args):
    inputs, issues = [], []
    allowance, allowance_issues = load_allowance(args.allowance_file, inputs)
    issues.extend(allowance_issues)
    statuses, rows_all, main, sources_all, graphs_all, comparable = {}, [], [], [], [], {}
    for phase in BLOCKS:
        raw, receipt = load_block(args.comparison_root, phase, allowance if not allowance_issues else None, inputs)
        phase_issues = list(receipt["issues"])
        if raw:
            rows = derive(raw, phase_issues)
            summary, sources, graphs = aggregate(rows, phase, allowance)
            main.extend(summary)
            sources_all.extend(sources)
            graphs_all.extend(graphs)
            rows_all.extend(dict(report_phase=phase, **row) for row in rows)
        if phase_issues and receipt["row_file_read"]:
            receipt["status"] = "COMPLETED_WITH_INPUT_ISSUES"
        receipt["issues"] = phase_issues
        statuses[phase] = receipt
        comparable[phase] = bool(raw) and not phase_issues and not allowance_issues
        issues.extend(phase + ": " + issue for issue in phase_issues)
    diffs = comparisons(main, sources_all, comparable)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for filename, values in (("quality_source_equal.csv", main), ("quality_per_physical_source.csv", sources_all),
            ("quality_per_graph_configuration.csv", graphs_all), ("all_completed_quality_run_rows.csv", rows_all),
            ("jointrecovery_quality_comparator_differences.csv", diffs)):
        write_csv(out / filename, values)
    (out / "JOINTRECOVERY_QUALITY_RESULTS_ZH.md").write_text(report_text(allowance, statuses, main, diffs, issues), encoding="utf-8")
    (out / "README_METHODS_ZH.md").write_text(mapping_markdown("quality"), encoding="utf-8")
    manifest = dict(schema=SCHEMA, generated_utc=datetime.now(timezone.utc).isoformat(),
        status="COMPLETED_QUALITY_TEST_REPORT" if statuses["test"]["status"] == "COMPLETED" and not allowance_issues else "WAITING_OR_INPUT_ISSUES",
        completed_block_statuses=statuses, selected_allowance=allowance,
        proposed_method_raw_key="FullCapacity", proposed_method_display="JointRecovery (JR)", method_display_mapping=METHOD_MAPPING,
        report_role="AMPLE_ALLOWANCE_PRIMARY_SOLUTION_QUALITY_CALLER_COST_SEPARATE", physical_test_sources=[8, 9, 10, 11],
        quality_null_not_zero_for_no_complete_output=True, external_S_not_solver_production=True,
        failed_late_incomplete_rows_retained=True, success_only_means_not_used=True,
        live_incomplete_row_files_not_read=True, experiments_launched=False,
        remaining_time_head_zero_not_validated=True, generator_sha256=sha(__file__),
        inputs=[dict(path=str(p.resolve()), sha256=sha(p)) for p in sorted(set(inputs))], issues=sorted(set(issues)))
    (out / "QUALITY_REPORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=manifest["status"], completed_rows=len(rows_all), report=str(out / "JOINTRECOVERY_QUALITY_RESULTS_ZH.md")), ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--allowance-file", "--selected-allowance", dest="allowance_file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/JOINTRECOVERY_QUALITY_RESULTS"))
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve()
    for name in ("comparison_root", "out"):
        value = getattr(args, name)
        setattr(args, name, value.resolve() if value.is_absolute() else (args.dataset_root / value).resolve())
    value = args.allowance_file or args.comparison_root / "selected_allowance.json"
    args.allowance_file = value.resolve() if value.is_absolute() else (args.dataset_root / value).resolve()
    args.out.relative_to(args.dataset_root / "reports")
    return args


if __name__ == "__main__":
    run(parse_args())
