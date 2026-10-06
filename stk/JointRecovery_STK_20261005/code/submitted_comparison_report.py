"""Read sealed local comparison blocks; write honest Chinese results and CSVs.

No cloud connection, experiment, model update, paper edit, plot or publication.
Incomplete blocks are WAITING and their actual_run_rows are never opened.
Driver-declared failed strict deliveries are zero; original unknowns stay null.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from jointrecovery_method_names import annotate_method_fields, method_display, mapping_markdown, METHOD_MAPPING

SCHEMA = "joint_recovery_stk_submitted_results_report_v1"
FULL_LABELS = ("FullCapacity-seed17", "FullCapacity-seed29",
               "FullCheapSummary-seed17", "FullCheapSummary-seed29", "FullP1", "FullGreedy", "CHILS-p1", "HiGHS-MILP")
OLD_LABELS = ("Capacity-seed17", "Capacity-seed29", "CheapSummary-seed17", "CheapSummary-seed29", "P1", "Greedy")
MEAN_LABELS = {"FullCapacity-mean17/29": FULL_LABELS[:2], "FullCheapSummary-mean17/29": FULL_LABELS[2:4]}
DISPLAY_ORDER = ("FullCapacity-mean17/29", *FULL_LABELS[:2], "FullCheapSummary-mean17/29", *FULL_LABELS[2:])
BLOCKS = {"validation": (6, 7), "test": (8, 9, 10, 11)}
BUDGETS = ("quarter", "half", "wide")
CONFIGS = {(view, gap, 150) for view in ("R8", "R12") for gap in (170, 340, 680)}
TOLERANCE = 1e-7
CHILS_URL = "https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2025.22"
SCIPY_URL = "https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.optimize.milp.html"
HIGHS_URL = "https://highs.dev/"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def exact_mean(values):
    return math.fsum(float(v) for v in values) / len(values) if values and all(finite(v) for v in values) else None


def write_csv(path, rows, empty_status="WAITING"):
    if not rows:
        rows = [{"status": empty_status}]
    rows = [annotate_method_fields(row) for row in rows]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {key: row.get(key) for key in fields}
            writer.writerow({key: "null" if value is None else json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (list, tuple, dict)) else value for key, value in values.items()})


def load_completed_block(root, block, inputs, old=False):
    """Only a sealed summary authorizes reading the corresponding row file."""
    directory = Path(root) / block
    summary_path, rows_path = directory / "summary.json", directory / "actual_run_rows.json"
    labels = OLD_LABELS if old else FULL_LABELS
    expected = len(BLOCKS[block]) * 6 * len(labels) * 3
    record = dict(block=block, status="WAITING", root=str(directory), expected_cells=expected,
                  received_cells=None, row_file_read=False, issues=[])
    if not summary_path.is_file():
        return [], record
    summary = read(summary_path)
    inputs.append(summary_path)
    marker = "ACTUAL_SERIAL_MATRIX_COMPLETE" if old else "SUBMITTED_COMPARISON_MATRIX_COMPLETE"
    if summary.get("status") != marker:
        record["summary_status"] = summary.get("status")
        return [], record
    if summary.get("expected_cells") != expected or summary.get("received_cells") != expected or not rows_path.is_file():
        record.update(status="WAITING_FOR_COMPLETE_LOCAL_MIRROR", summary_status=summary.get("status"),
                      received_cells=summary.get("received_cells"))
        return [], record
    rows = read(rows_path)
    inputs.append(rows_path)
    record.update(row_file_read=True, received_cells=len(rows), status="COMPLETED")
    if len(rows) != expected:
        record.update(status="INCOMPLETE_LOCAL_MIRROR", issues=["sealed summary / local row count differs"])
        return [], record
    expected_sources = set(BLOCKS[block])
    seen = set()
    graph_config = defaultdict(dict)
    for row in rows:
        key = (row.get("graph_id"), row.get("policy_label"), row.get("budget_id"))
        if key in seen:
            record["issues"].append("duplicate cell: " + str(key))
        seen.add(key)
        source = row.get("source")
        if source not in expected_sources or row.get("policy_label") not in labels or row.get("budget_id") not in BUDGETS:
            record["issues"].append("unexpected source/policy/budget cell: " + str(key))
        config = (row.get("station_view"), row.get("ground_gap_seconds"), row.get("satellite_gap_seconds"))
        graph_config[source][row.get("graph_id")] = config
    for source in expected_sources:
        if len(graph_config[source]) != 6 or set(graph_config[source].values()) != CONFIGS:
            record["issues"].append("not all six configurations: r%03d" % source)
    if summary.get("initial_mask_mismatches"):
        record["issues"].append("driver reported original S mask mismatch")
    if record["issues"]:
        record["status"] = "COMPLETED_WITH_INPUT_ISSUES"
    return rows, record


def bind_and_derive(rows, budgets, issues):
    """Keep all reported fields; annotate strict delivery and known common S."""
    anchors, masks, hashes = defaultdict(list), defaultdict(set), defaultdict(set)
    for row in rows:
        graph_id = row["graph_id"]
        if row.get("graph_sha256_from_metadata"):
            hashes[graph_id].add(row["graph_sha256_from_metadata"])
        if row.get("report_usable") or row.get("report_usable_for_calibration"):
            if finite(row.get("initial_value_seconds")):
                anchors[graph_id].append(float(row["initial_value_seconds"]))
            if row.get("initial_mask_sha256"):
                masks[graph_id].add(row["initial_mask_sha256"])
    graph_initial = {}
    for graph_id in {r["graph_id"] for r in rows}:
        values = anchors[graph_id]
        consistent = bool(values) and all(math.isclose(v, values[0], rel_tol=1e-12, abs_tol=TOLERANCE) for v in values)
        if len(masks[graph_id]) > 1 or len(hashes[graph_id]) > 1 or (values and not consistent):
            issues.append("graph original S/objective/input identity mismatch: " + graph_id)
            consistent = False
        graph_initial[graph_id] = values[0] if consistent else None
    result = []
    for raw in rows:
        row = dict(raw)
        usable = bool(row.get("report_usable", row.get("report_usable_for_calibration", False)))
        failed = bool(row.get("failure_or_unknown_outcome", not usable))
        strict = row.get("strict_delivered_gain_including_failed_runs_seconds")
        if not finite(strict):
            # Missing strict numeric input is not fabricated, even for a
            # purported complete row. Real driver failures already contain0.
            issues.append("missing driver strict delivered value: " + row.get("run_id", row["graph_id"]))
        elif strict < -TOLERANCE:
            issues.append("negative strict delivered gain")
        deadline = row.get("deadline_seconds")
        if row["budget_id"] in budgets and deadline != budgets[row["budget_id"]]:
            issues.append("exact frozen deadline differs: " + row.get("run_id", row["graph_id"]))
        caller = row.get("external_caller_observed_return_seconds")
        observed_late = row.get("missed_return_sample") is True
        on_time = bool(usable and row.get("missed_return_sample") is False and finite(caller) and finite(deadline) and caller <= deadline)
        initial = graph_initial[row["graph_id"]]
        prefix_raw = row.get("paid_shared_prefix_gain_seconds")
        if not usable or observed_late:
            prefix = 0.
        elif finite(prefix_raw):
            prefix = float(prefix_raw)
        else:
            prefix = None
        beyond = max(0., float(strict) - prefix) if finite(strict) and finite(prefix) else None
        if finite(strict) and finite(prefix) and prefix > strict + TOLERANCE:
            issues.append("strict prefix exceeds strict gain: " + row.get("run_id", row["graph_id"]))
        row.update(derived_known_original_S_value_seconds=initial,
            derived_original_S_anchor_from_usable_common_graph=initial is not None,
            derived_strict_return_objective_seconds=initial + strict if finite(initial) and finite(strict) else None,
            derived_strict_gain_percent_of_original_S=100. * strict / max(1., initial) if finite(initial) and finite(strict) else None,
            derived_strict_paid_prefix_gain_seconds=prefix,
            derived_strict_gain_beyond_paid_prefix_seconds=beyond,
            derived_usable=float(usable), derived_submission_failure_or_unknown=float(failed),
            derived_execution_failed_stop=float(row.get("stop_reason") == "failed"),
            derived_observed_late=float(observed_late), derived_late_status_unknown=float(not isinstance(row.get("missed_return_sample"), bool)),
            derived_on_time_all_declared_cells=float(on_time))
        result.append(row)
    return result


METRICS = {
    "initial_S_seconds": "derived_known_original_S_value_seconds",
    "strict_return_objective_seconds": "derived_strict_return_objective_seconds",
    "strict_gain_seconds": "strict_delivered_gain_including_failed_runs_seconds",
    "strict_gain_percent": "derived_strict_gain_percent_of_original_S",
    "strict_prefix_seconds": "derived_strict_paid_prefix_gain_seconds",
    "strict_beyond_prefix_seconds": "derived_strict_gain_beyond_paid_prefix_seconds",
    "caller_seconds": "external_caller_observed_return_seconds",
    "observed_return_objective_seconds": "returned_value_seconds",
    "observed_actual_gain_seconds": "actual_gain_seconds",
    "usable_rate": "derived_usable", "on_time_rate": "derived_on_time_all_declared_cells",
    "submission_failure_or_unknown_rate": "derived_submission_failure_or_unknown",
    "execution_failed_stop_rate": "derived_execution_failed_stop", "observed_late_rate": "derived_observed_late",
    "unknown_late_status_rate": "derived_late_status_unknown",
}


def aggregate(rows, phase, sources, labels):
    summary, per_source, per_graph = [], [], []
    groups = {label: (label,) for label in labels}
    if tuple(labels) == FULL_LABELS:
        groups.update(MEAN_LABELS)
    for budget in BUDGETS:
        for label, members in groups.items():
            selected = [r for r in rows if r["budget_id"] == budget and r["policy_label"] in members]
            source_records = []
            for source in sources:
                current = [r for r in selected if r["source"] == source]
                graphs = sorted({r["graph_id"] for r in current})
                graph_records = []
                for graph in graphs:
                    graph_rows = [r for r in current if r["graph_id"] == graph]
                    grid_complete = len(graph_rows) == len(members) and {r["policy_label"] for r in graph_rows} == set(members)
                    metrics = {name: exact_mean([r.get(field) for r in graph_rows]) if grid_complete else None
                               for name, field in METRICS.items()}
                    record = dict(phase=phase, budget_id=budget, policy_label=label, source=source,
                                  graph_id=graph, seed_count=len(members), grid_complete=grid_complete,
                                  deadline_seconds=exact_mean([r.get("deadline_seconds") for r in graph_rows]), **metrics)
                    graph_records.append(record)
                    per_graph.append(record)
                complete = len(graph_records) == 6 and all(g["grid_complete"] for g in graph_records)
                source_record = dict(phase=phase, budget_id=budget, policy_label=label,
                    source=source, physical_source="r%03d" % source, graph_count=len(graphs),
                    declared_cells=6 * len(members), received_cells=len(current), grid_complete=complete,
                    **{m: exact_mean([g[m] for g in graph_records]) if complete else None for m in METRICS})
                source_records.append(source_record)
                per_source.append(source_record)
            grid_complete = all(s["grid_complete"] for s in source_records)
            record = dict(phase=phase, budget_id=budget, policy_label=label,
                deadline_seconds=exact_mean([r.get("deadline_seconds") for r in selected]),
                physical_source_count=len(sources), seed_count=len(members), grid_complete=grid_complete,
                declared_cells=len(sources) * 6 * len(members), received_cells=len(selected),
                failure_or_unknown_count=sum(int(r["derived_submission_failure_or_unknown"]) for r in selected),
                execution_failed_stop_count=sum(int(r["derived_execution_failed_stop"]) for r in selected),
                observed_late_count=sum(int(r["derived_observed_late"]) for r in selected),
                unknown_late_status_count=sum(int(r["derived_late_status_unknown"]) for r in selected),
                **{m: exact_mean([s[m] for s in source_records]) if grid_complete else None for m in METRICS})
            summary.append(record)
    return summary, per_source, per_graph


def differences(summary, sources, comparable):
    result = []
    index = {(r["phase"], r["budget_id"], r["policy_label"]): r for r in summary}
    source_index = {(r["phase"], r["budget_id"], r["policy_label"], r["source"]): r for r in sources}
    targets = ("FullCapacity-seed17", "FullCapacity-seed29", "FullCapacity-mean17/29")
    baselines = ("FullGreedy", "FullP1", "FullCheapSummary-seed17", "FullCheapSummary-seed29",
                 "FullCheapSummary-mean17/29", "CHILS-p1", "HiGHS-MILP")
    for phase in ("validation", "test"):
        for budget in BUDGETS:
            if (phase, budget, targets[0]) not in index:
                continue
            cheap = [index[(phase, budget, b)] for b in ("FullGreedy", "FullCheapSummary-mean17/29")]
            strong = max(cheap, key=lambda r: r["strict_gain_seconds"]) if all(finite(r["strict_gain_seconds"]) for r in cheap) else None
            comparisons = list(baselines) + (["StrongCheapControl"] if strong else [])
            for target in targets:
                a = index[(phase, budget, target)]
                for baseline in comparisons:
                    b = strong if baseline == "StrongCheapControl" else index[(phase, budget, baseline)]
                    delta = a["strict_gain_seconds"] - b["strict_gain_seconds"] if finite(a["strict_gain_seconds"]) and finite(b["strict_gain_seconds"]) else None
                    ds = []
                    for source in BLOCKS[phase]:
                        av = source_index[(phase, budget, target, source)]["strict_gain_seconds"]
                        bv = source_index[(phase, budget, b["policy_label"], source)]["strict_gain_seconds"]
                        ds.append(av - bv if finite(av) and finite(bv) else None)
                    valid = comparable.get(phase, False) and finite(delta) and all(finite(d) for d in ds)
                    result.append(dict(phase=phase, budget_id=budget, target=target, comparator=baseline,
                        actual_comparator=b["policy_label"], comparable_inputs=comparable.get(phase, False),
                        delta_strict_gain_seconds=delta,
                        delta_gain_percentage_points=a["strict_gain_percent"] - b["strict_gain_percent"] if finite(a["strict_gain_percent"]) and finite(b["strict_gain_percent"]) else None,
                        target_exceeds_comparator=(delta > TOLERANCE) if valid else None,
                        result="HIGHER" if valid and delta > TOLERANCE else "LOWER" if valid and delta < -TOLERANCE else "TIED" if valid else "UNKNOWN_OR_INPUT_ISSUE",
                        physical_source_count=len(ds), source_differences_seconds=ds,
                        positive_source_count=sum(d > TOLERANCE for d in ds) if all(finite(d) for d in ds) else None,
                        negative_source_count=sum(d < -TOLERANCE for d in ds) if all(finite(d) for d in ds) else None,
                        tied_source_count=sum(abs(d) <= TOLERANCE for d in ds) if all(finite(d) for d in ds) else None,
                        comparator_choice_is_descriptive_not_policy_or_seed_selection=baseline == "StrongCheapControl"))
    return result


def waiting_rows(phase, labels, budgets):
    return [dict(phase=phase, budget_id=b, policy_label=p, status="WAITING",
                 deadline_seconds=budgets.get(b), **{m: None for m in METRICS}) for b in BUDGETS for p in labels]


def display(value, places=3, waiting=False):
    if not finite(value):
        return "WAITING" if waiting else "N/A"
    return format(value, ".%df" % places)


def method_name(label):
    return method_display(label) + (" [1]" if label == "CHILS-p1" else " [2]" if label == "HiGHS-MILP" else "")


def markdown_report(plan, plan_status, statuses, main, diffs, old_statuses, p1_status, issues):
    lines = ["# JointRecovery：效率与短预算压力诊断", "",
        "**本文方法是 JointRecovery（JR；原实现键 FullCapacity）。FullP1 / P1 为 Independent-replacement+CHILS 对照，不是本文主方法。** 详见 README_METHODS_ZH.md。", "",
        "本文件是实验结果报告，不是论文正文。此处三档短 D 用于效率与压力诊断；充裕预算下的完整求解质量主结果移至 reports/JOINTRECOVERY_QUALITY_RESULTS。原静态 max4 P2 仅为内部诊断。所有数字均使用原始完整 contact 时长秒。", "",
        "**remaining-head 固定0，尚未验证剩余时间学习；完整在线闭环不等于旧稿每一组件、旧工作协议都已验证。当前方法为已冻结 new-STK 检查点接入动态 warm/g*/spent、收益/成本排序与精确编码缓存的迁移版本。**", "",
        "## 完成范围", "", "| 来源块 | 状态 | 已收/预声明 cell | 主结果行读取 |", "|---|---|---:|---|"]
    for phase in ("test", "validation"):
        s = statuses[phase]
        lines.append("| %s | %s | %s/%s | %s |" % (phase, s["status"], s.get("received_cells") if s.get("received_cells") is not None else "未知", s["expected_cells"], s["row_file_read"]))
    lines += ["", "只有 block 的完整 summary 已签收且全量行齐才读取 actual_run_rows。尚未完成的 block 标为 WAITING，不把 live/缺失测试结果填0。", "",
        "测试独立单位为 r008–r011 **四个物理母源**；先对每源六配置等权，再对四源等权。学习模型两个 seed 各自完整汇总后再均值，两个 seed 范围不是置信区间。验证 r006/r007 两源单列，不用于本报告挑模型、D或种子。", "",
        "## 测试核心比较", ""]
    test_ready = statuses["test"]["row_file_read"] and any(r["phase"] == "test" and r.get("status") != "WAITING" for r in main)
    if not test_ready:
        lines += ["**WAITING：完整扩展 TEST 尚未取得可用完成镜像。现在不能回答完整方法是否优于 FullGreedy、CheapSummary、CHILS 或 HiGHS。**", ""]
    index = {(r["phase"], r["budget_id"], r["policy_label"]): r for r in main}
    for phase in ("test", "validation"):
        ready = statuses[phase]["row_file_read"] and any(r["phase"] == phase and r.get("status") != "WAITING" for r in main)
        if phase == "validation":
            lines += ["## 验证结果（单列，不代替测试）", ""]
        if not ready:
            if phase == "validation":
                lines += ["WAITING：完整验证结果尚未收齐。", ""]
            continue
        for budget in BUDGETS:
            example = index.get((phase, budget, DISPLAY_ORDER[0]), {})
            lines += ["### %s：%s，D=%s 秒" % (phase, budget, display(example.get("deadline_seconds"), 9)), "",
                "| 方法 | 原S目标↑ | 严格交付总目标↑ | 严格增秒↑ | 对原S增益%↑ | 严格共同前缀秒 | 前缀外净增秒↑ | caller秒↓ | 按时率%↑ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for label in DISPLAY_ORDER:
                r = index[(phase, budget, label)]
                values = [r.get(m) for m in ("initial_S_seconds", "strict_return_objective_seconds", "strict_gain_seconds", "strict_gain_percent", "strict_prefix_seconds", "strict_beyond_prefix_seconds", "caller_seconds")]
                lines.append("| %s | %s | %s |" % (method_name(label), " | ".join(display(v) for v in values), display(100. * r["on_time_rate"] if finite(r.get("on_time_rate")) else None, 1)))
            lines += ["", "| 方法 | 可用率%↑ | 提交失败/未知数↓ | 执行failed-stop数↓ | 已观察迟到数↓ | 迟到状态未知数 |", "|---|---:|---:|---:|---:|---:|"]
            for label in DISPLAY_ORDER:
                r = index[(phase, budget, label)]
                lines.append("| %s | %s | %s | %s | %s | %s |" % (method_name(label), display(100. * r["usable_rate"] if finite(r.get("usable_rate")) else None, 1), r.get("failure_or_unknown_count"), r.get("execution_failed_stop_count"), r.get("observed_late_count"), r.get("unknown_late_status_count")))
            lines += ["", "两seed均值行的计数覆盖两个fit的全部已声明cell，比例先seed/配置/来源等权；不是额外独立样本。原报告目标/耗时未知时为 N/A，不选择性删行计算均值。", ""]
    lines += ["## 直接回答：两个种子及均值是否超过对照", "",
        "下表比较严格按时交付增量的差值，单位秒↑。正/负/平是来源等权观测差，不是显著性结论。StrongCheapControl 是同一 D 下 FullGreedy 与 CheapSummary 两seed均值中更大的已观察均值，仅用于描述性更强参照，不选部署策略或种子。", "",
        "| TEST D | JointRecovery（JR） | Greedy-rank+CHILS 差↑ | JR-CheapSummary均值差↑ | CHILS-p1 [1] 差↑ | HiGHS [2] 差↑ | 最强内部控制差↑ |", "|---|---|---:|---:|---:|---:|---:|"]
    di = {(r["phase"], r["budget_id"], r["target"], r["comparator"]): r for r in diffs}
    for budget in BUDGETS:
        for target in ("FullCapacity-seed17", "FullCapacity-seed29", "FullCapacity-mean17/29"):
            current = [di.get(("test", budget, target, c)) for c in ("FullGreedy", "FullCheapSummary-mean17/29", "CHILS-p1", "HiGHS-MILP", "StrongCheapControl")]
            texts = [display(r["delta_strict_gain_seconds"], waiting=not test_ready) + ("（不可比）" if r and not r["comparable_inputs"] else "") if r else "WAITING" for r in current]
            lines.append("| %s | %s | %s |" % (budget, method_name(target), " | ".join(texts)))
    lines.append("")
    if test_ready:
        names = {key: method_display(key) for key in ("FullGreedy", "FullCheapSummary-mean17/29", "CHILS-p1", "HiGHS-MILP")}
        for budget in BUDGETS:
            for target in ("FullCapacity-seed17", "FullCapacity-seed29", "FullCapacity-mean17/29"):
                phrases = []
                for comparator, name in names.items():
                    r = di.get(("test", budget, target, comparator))
                    if not r or r["result"] == "UNKNOWN_OR_INPUT_ISSUE":
                        phrases.append("相对%s结论未知/输入不可比" % name)
                    else:
                        word = {"HIGHER": "高于", "LOWER": "低于", "TIED": "持平于"}[r["result"]]
                        phrases.append("%s%s（差%s秒，四源正/负/平=%s/%s/%s）" % (word, name, display(r["delta_strict_gain_seconds"]), r["positive_source_count"], r["negative_source_count"], r["tied_source_count"]))
                lines += ["- %s，%s：%s。" % (budget, method_name(target), "；".join(phrases))]
        lines += ["", "这些对比不自动证明 occupancy、辅助监督或图结构的因果作用。本次没有相应全面消融；两seed的均值不得掩盖某seed低于控制的结果。", ""]
    lines += ["## 收益归因、失败与公平边界", "",
        "严格总目标=已核对的共同原S目标+driver严格交付增量；caller迟到或提交缺失/不可用的严格增量为0。报告原始返回值、耗时、迟到标记等未知字段仍保留null，不能把未知耗时补0。原S可从同原图/同mask的可用peer收据确定，衍生列标记该来源；无法确定则目标/百分比N/A。", "",
        "共同三贪心前缀属于各恢复策略的已付费共同准备。前缀外净增来自实际native恢复；只有相对同框架 FullGreedy/CheapSummary 的匹配差值才能说明学习分配的观测收益，不能把所有native或prefix收益归给学习。整图CHILS/HiGHS没有执行该共同prefix，前缀外列只是该方法自己的增量，不能当相同机制的归因比较。", "",
        "提交失败/未知严格交付0按driver保留；若native执行失败但父进程已按时返回已验证prefix，该真实收益仍保留，同时单列执行failed-stop。按时率和可用率的分母包含全部已声明cell；观察迟到与迟到未知分开。", "",
        "Full策略固定11动作、cap256、四workpoints、最多8请求；CHILS/HiGHS可在完整原图搜索，搜索范围不同。双方共享原图、原时长objective、原S和完整caller D，不能把此比较说成相同子问题/相同native搜索次数。resident图/模型/solver初始化单列；构造、导出、模型/矩阵组装、推理、native、parse、原图校验、IPC、实际return均计D。整图baseline采用固定min(0.70D,remaining−0.10D) native allowance。", "",
        "HiGHS为通用MILP参照，经SciPy1.10.1 milp接口；该接口没有x0参数，本实现仅在父保留共同S作外部下界，没有solver内warm start。它不是专门的已发表MWIS学习算法；短D中无新解不能写成已证最优，也不能泛称击败全部先进方法。微秒tick是数值序列化，实际接受/评分仍为原始时长；不从tick求解状态推断原float目标的精确最优性。", "",
        "## 新增协议时间与旧内部诊断", ""]
    if plan_status == "LOADED":
        lines += ["扩展PLAN冻结时间：`%s`；状态：`%s`。PLAN明确 original_P2_already_in_progress=%s、not_a_pre_registration_before_dataset_generation=%s。即**新增比较在原P2开始之后、PLAN记载尚未查看原VAL/TEST结果时冻结**；不能称作数据生成或原P2开始之前的预注册。本报告按该实际记录披露，时间戳本身不是独立证明研究者从未接触结果。" % (plan.get("frozen_utc"), plan.get("status"), plan.get("original_P2_already_in_progress"), plan.get("not_a_pre_registration_before_dataset_generation")), ""]
    else:
        lines += ["WAITING：未取得扩展冻结PLAN；无法确认协议时间边界。", ""]
    lines += ["原P1开发回放状态：%s；仅写入 p1_fixed_call_internal_diagnostic.csv。原P2静态max4 block状态：%s。该程序一次预测、冷clone/frozen warm顺序，内部诊断写入 original_p2_static_internal_diagnostic.csv，不进入上述完整方法主性能结论。完整max8与旧静态max4不同，不把二者差值当单一组件消融。" % (p1_status, ", ".join(k + "=" + v["status"] for k, v in old_statuses.items())), ""]
    if issues:
        lines += ["## 需保留的输入异常", ""] + ["- " + issue for issue in sorted(set(issues))] + [""]
    lines += ["## 正式方法来源", "",
        "[1] Ernestine Großmann, Kenneth Langedal, Christian Schulz. Concurrent Iterated Local Search for the Maximum Weight Independent Set Problem. SEA 2025, LIPIcs338,22:1–22:18. DOI10.4230/LIPIcs.SEA.2025.22。[正式会议页面](%s)。本表为其官方单线程p1配置的实际整图移植，不是完整多线程论文最佳配置。" % CHILS_URL, "",
        "[2] [SciPy1.10.1 scipy.optimize.milp官方软件文档](%s)、[HiGHS官方软件站](%s)。表中为通用MILP求解参照，正式软件引用不伪装成另一篇专门MWIS方法论文。" % (SCIPY_URL, HIGHS_URL), "",
        "原始全量行、逐来源/配置、两seed、差值、输入hash均保存在旁边CSV/manifest；不生成新图，不改旧paper/results。"]
    return "\n".join(lines) + "\n"


def p1_diagnostics(root, inputs):
    path = Path(root) / "fixed_calls" / "summary.json"
    if not path.is_file():
        return [], "WAITING"
    record = read(path)
    inputs.append(path)
    if record.get("status") != "FIXED_CALL_COLD_CLONE_LABEL_REPLAY_COMPLETE":
        return [], "WAITING"
    return [dict(scope="INTERNAL_DEVELOPMENT_COLD_CLONE_LABEL_REPLAY_NOT_COMPLETE_ONLINE",
                 **row) for row in record.get("results", [])], "COMPLETED"


def run(args):
    inputs, issues = [], []
    plan = read(args.plan) if args.plan.is_file() else {}
    plan_status = "LOADED" if plan else "WAITING"
    if plan:
        inputs.append(args.plan)
    budgets = {r["budget_id"]: r["deadline_seconds"] for r in plan.get("budgets", [])}
    protocol_issues = []
    if plan and (plan.get("status") != "PLAN_FROZEN_BEFORE_INSPECTING_ORIGINAL_P2_VALIDATION_OR_TEST_RESULTS" or
                 plan.get("original_P2_already_in_progress") is not True or
                 plan.get("not_a_pre_registration_before_dataset_generation") is not True):
        protocol_issues.append("extension PLAN does not establish the declared after-P2-start freeze boundary")
    if args.budgets_file.is_file():
        frozen = read(args.budgets_file)
        inputs.append(args.budgets_file)
        frozen_values = {r["budget_id"]: r["deadline_seconds"] for r in frozen.get("budgets", [])}
        if (frozen.get("status") != "BUDGETS_FROZEN_FROM_DEVELOPMENT_ACTUAL_CALLERS" or
                plan.get("frozen_budgets_sha256") != sha(args.budgets_file) or frozen_values != budgets):
            protocol_issues.append("extension PLAN / original frozen caller budgets differ")
    else:
        protocol_issues.append("original frozen caller budgets file is missing")
    if set(budgets) != set(BUDGETS):
        protocol_issues.append("not all three exact frozen deadlines are present")
    issues.extend(protocol_issues)
    statuses, all_rows, main, source_rows, graph_rows, comparable = {}, [], [], [], [], {}
    for block, sources in BLOCKS.items():
        rows, status = load_completed_block(args.comparison_root, block, inputs)
        block_issues = list(status["issues"]) + protocol_issues
        if rows:
            rows = bind_and_derive(rows, budgets, block_issues)
            summaries, values, graphs = aggregate(rows, block, sources, FULL_LABELS)
            main.extend(summaries)
            source_rows.extend(values)
            graph_rows.extend(graphs)
            all_rows.extend([dict(report_phase=block, **r) for r in rows])
        else:
            main.extend(waiting_rows(block, DISPLAY_ORDER, budgets))
        status["issues"] = sorted(set(block_issues))
        if block_issues and status["row_file_read"]:
            status["status"] = "COMPLETED_WITH_INPUT_ISSUES"
        statuses[block] = status
        comparable[block] = bool(rows) and not block_issues and plan_status == "LOADED"
        issues.extend(block + ": " + issue for issue in block_issues)
    actual_main = [r for r in main if r.get("status") != "WAITING"]
    diffs = differences(actual_main, source_rows, comparable)
    old_statuses, old_summaries = {}, []
    for block, sources in BLOCKS.items():
        rows, status = load_completed_block(args.p2_root, block, inputs, old=True)
        old_statuses[block] = status
        if rows:
            diagnostic_issues = []
            rows = bind_and_derive(rows, budgets, diagnostic_issues)
            summary, _, _ = aggregate(rows, "original_static_max4_" + block, sources, OLD_LABELS)
            old_summaries.extend([dict(scope="INTERNAL_STATIC_MAX4_NOT_FULL_CONTROLLER_PRIMARY_RESULT", **r) for r in summary])
            old_statuses[block]["issues"].extend(diagnostic_issues)
        else:
            old_summaries.extend(waiting_rows("original_static_max4_" + block, OLD_LABELS, budgets))
    p1_rows, p1_status = p1_diagnostics(args.p1_root, inputs)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "main_source_equal.csv", main)
    write_csv(out / "main_per_physical_source.csv", source_rows)
    write_csv(out / "main_per_graph_configuration.csv", graph_rows)
    write_csv(out / "all_completed_actual_run_rows.csv", all_rows)
    write_csv(out / "capacity_comparator_differences.csv", diffs)
    write_csv(out / "original_p2_static_internal_diagnostic.csv", old_summaries)
    write_csv(out / "p1_fixed_call_internal_diagnostic.csv", p1_rows)
    manifest = dict(schema=SCHEMA, generated_utc=datetime.now(timezone.utc).isoformat(),
        status="COMPLETED_TEST_REPORT" if statuses["test"]["status"] == "COMPLETED" else "WAITING_OR_INPUT_ISSUES",
        completed_block_statuses=statuses, original_p2_internal_statuses=old_statuses,
        p1_internal_status=p1_status, physical_test_sources=list(BLOCKS["test"]),
        two_fit_seeds_are_not_independent_sources=True, seed_selection=False,
        remaining_time_head_zero_not_validated=True, full_framework_not_all_original_components_validated=True,
        raw_unknowns_retained_null=True, driver_explicit_failed_strict_delivery_zero_retained=True,
        live_incomplete_row_files_not_read=True, no_new_figures_or_experiments=True,
        report_role="EFFICIENCY_AND_SHORT_BUDGET_STRESS_DIAGNOSTIC_NOT_AMPLE_QUALITY_PRIMARY",
        proposed_method_raw_key="FullCapacity", proposed_method_display="JointRecovery (JR)",
        method_display_mapping=METHOD_MAPPING,
        issues=sorted(set(issues)), inputs=[dict(path=str(path.resolve()), sha256=sha(path)) for path in sorted(set(inputs))],
        generator_sha256=sha(__file__))
    (out / "SUBMITTED_METHOD_RESULTS_ZH.md").write_text(markdown_report(plan, plan_status, statuses, main, diffs, old_statuses, p1_status, issues), encoding="utf-8")
    (out / "README_METHODS_ZH.md").write_text(mapping_markdown("short"), encoding="utf-8")
    (out / "REPORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=manifest["status"], test=statuses["test"]["status"], validation=statuses["validation"]["status"],
                         completed_rows=len(all_rows), report=str(out / "SUBMITTED_METHOD_RESULTS_ZH.md")), ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset-root", type=Path, default=root)
    parser.add_argument("--comparison-root", type=Path, required=True, help="Local sealed extension out, with validation/test blocks")
    parser.add_argument("--original-p2-root", "--p2-root", dest="p2_root", type=Path)
    parser.add_argument("--p1-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--budgets-file", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve()
    def relative_to_dataset(path):
        return path.resolve() if path.is_absolute() else (args.dataset_root / path).resolve()
    args.comparison_root = relative_to_dataset(args.comparison_root)
    args.p2_root = relative_to_dataset(args.p2_root or Path("execution/p2_confirmation/out"))
    args.p1_root = relative_to_dataset(args.p1_root or Path("execution/p1_followup/out"))
    args.plan = relative_to_dataset(args.plan or Path("protocol/SUBMITTED_COMPARISON_EXTENSION_PLAN_20261005.json"))
    args.budgets_file = relative_to_dataset(args.budgets_file or Path("execution/p1_followup/out/calibration/frozen_budgets.json"))
    args.out = relative_to_dataset(args.out or Path("reports/SUBMITTED_METHOD_RESULTS"))
    args.out.relative_to(args.dataset_root / "reports")
    return args


if __name__ == "__main__":
    run(parse_args())
