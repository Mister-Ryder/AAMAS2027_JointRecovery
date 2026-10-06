"""Visible method names only: raw frozen algorithm and CSV keys stay unchanged."""
from __future__ import annotations

METHOD_MAPPING = {
    "FullCapacity": {"display": "JointRecovery（本文方法，JR；完整框架 STK 迁移）", "short": "JointRecovery (JR, ours)", "role": "PROPOSED_FULL_FRAMEWORK"},
    "FullCheapSummary": {"display": "JR-CheapSummary（简化表示消融）", "short": "JR-CheapSummary", "role": "SIMPLIFIED_REPRESENTATION_ABLATION"},
    "FullGreedy": {"display": "Greedy-rank+CHILS（贪心请求排序内部对照）", "short": "Greedy-rank+CHILS", "role": "INTERNAL_GREEDY_REQUEST_RANKING"},
    "FullP1": {"display": "Independent-replacement+CHILS（独立替代估计对照）", "short": "Independent-replacement+CHILS", "role": "INTERNAL_INDEPENDENT_REPLACEMENT_CONTROL"},
    "Capacity": {"display": "JointRecovery（静态 max4 内部诊断）", "short": "JointRecovery (static diagnostic)", "role": "INTERNAL_STATIC_MAX4_DIAGNOSTIC"},
    "CheapSummary": {"display": "JR-CheapSummary（静态简化表示诊断）", "short": "JR-CheapSummary (static)", "role": "INTERNAL_STATIC_SUMMARY_DIAGNOSTIC"},
    "Greedy": {"display": "Greedy-rank+CHILS（静态贪心排序对照）", "short": "Greedy-rank+CHILS (static)", "role": "INTERNAL_STATIC_GREEDY_CONTROL"},
    "P1": {"display": "Independent-replacement+CHILS（静态独立替代对照）", "short": "Independent-replacement+CHILS (static)", "role": "INTERNAL_STATIC_INDEPENDENT_CONTROL"},
    "CHILS-p1": {"display": "CHILS-p1（SEA 2025 公开 MWIS 算法）", "short": "CHILS-p1 (SEA 2025)", "role": "PUBLISHED_MWIS_BASELINE"},
    "HiGHS-MILP": {"display": "HiGHS-MILP（通用 MILP 软件参照）", "short": "HiGHS (generic MILP)", "role": "GENERIC_MILP_SOFTWARE_REFERENCE"},
    "StrongCheapControl": {"display": "较强内部对照（描述性参照）", "short": "Stronger internal control", "role": "DESCRIPTIVE_CONTROL_REFERENCE"},
}


def identify(label):
    text = str(label) if label is not None else ""
    for suffix in ("-mean17/29", "-seed17", "-seed29"):
        if text.endswith(suffix):
            return text[:-len(suffix)], suffix[1:]
    return text, ""


def method_display(label, english=False):
    key, variant = identify(label)
    record = METHOD_MAPPING.get(key)
    if record is None:
        return str(label)
    name = record["short" if english else "display"]
    if variant == "mean17/29":
        return name + ("; 2-seed mean" if english else "；两种子均值")
    if variant.startswith("seed"):
        return name + ("; " if english else "；") + variant
    return name


def annotate_method_fields(row):
    """Add inspectable display columns without modifying any raw key/value."""
    result = dict(row)
    label = row.get("policy_label", row.get("policy", row.get("family")))
    if label is not None:
        result["method_display"] = method_display(label)
        result["method_display_en"] = method_display(label, english=True)
        result["method_role"] = METHOD_MAPPING.get(identify(label)[0], {}).get("role", "UNMAPPED")
    for key in ("target", "comparator", "actual_comparator"):
        if row.get(key) is not None:
            result[key + "_method_display"] = method_display(row[key])
    return result


def mapping_markdown(scope="quality"):
    title = "主性能结果：充裕预算下完整求解质量" if scope == "quality" else "效率与短预算压力诊断（不是充裕求解质量主结果）"
    lines = ["# 方法名称与结果用途", "", "**本文方法是 JointRecovery（JR），原始实现键为 FullCapacity。P1 / FullP1 是独立替代估计对照，不是本文主方法。**", "", "本目录用途：" + title + "。原始键永久保留以便回溯封存协议；CSV 的 method_display、method_role 列提供可见名称。", "", "| 原始键（含 seed / mean 变体） | 可见名称 | 实验角色 |", "|---|---|---|"]
    for key, record in METHOD_MAPPING.items():
        lines.append("| `%s` | %s | %s |" % (key, record["display"], record["role"]))
    lines += ["", "JointRecovery 完整框架使用动态 g*/warm/spent、联合恢复估计和收益/成本请求排序。当前 STK 迁移版本 remaining-head 固定0，尚未验证该组件。名称统一不改变模型、算法、结果数值或冻结输入。", "", "充裕预算主结果另见 reports/JOINTRECOVERY_QUALITY_RESULTS；旧 reports/SUBMITTED_METHOD_RESULTS 为效率/压力诊断，reports/P1_RESULTS 为静态 max4 / 固定调用等内部诊断。"]
    return "\n".join(lines) + "\n"
