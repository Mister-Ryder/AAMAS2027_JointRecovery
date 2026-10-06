"""Compact scientific 2x2 PNG/PDF from completed local TEST report CSVs only.

No cloud/live-row access, experiments, paper edit, smoothing or invented D.
The stronger public algorithm/software reference for panel D is descriptive;
HiGHS remains explicitly a generic MILP reference, not a second MWIS paper.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from jointrecovery_method_names import method_display, METHOD_MAPPING

METHODS = {
    "Capacity": dict(key="FullCapacity-mean17/29", color="#BE6628", marker="o", linestyle="-", label="JointRecovery (JR, ours; mean)"),
    "CheapSummary": dict(key="FullCheapSummary-mean17/29", color="#65899E", marker="s", linestyle="--", label="JR-CheapSummary (mean)"),
    "FullGreedy": dict(key="FullGreedy", color="#294E66", marker="D", linestyle="-", label="Greedy-rank+CHILS"),
    "FullP1": dict(key="FullP1", color="#91A6B5", marker="^", linestyle="-.", label="Independent-replacement+CHILS"),
    "CHILS-p1": dict(key="CHILS-p1", color="#43647C", marker="P", linestyle=":", label="CHILS-p1 (SEA 2025)"),
    "HiGHS": dict(key="HiGHS-MILP", color="#6F7A86", marker="X", linestyle=(0, (4, 1, 1, 1)), label="HiGHS (generic MILP)"),
}
BUDGETS = ("quarter", "half", "wide")
SOURCES = (8, 9, 10, 11)
REFERENCE_CANDIDATES = ("CHILS-p1", "HiGHS-MILP")
SEED_KEYS = {"Capacity": ("FullCapacity-seed17", "FullCapacity-seed29"),
             "CheapSummary": ("FullCheapSummary-seed17", "FullCheapSummary-seed29")}
COMPARATORS = ("CheapSummary", "FullGreedy", "CHILS-p1", "HiGHS")
INK, GRID = "#26313B", "#E0E5E9"
SCHEMA = "joint_recovery_stk_submitted_figure_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value in (None, "", "null", "None", "N/A", "WAITING"):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def truth(value):
    return value is True or str(value).lower() == "true"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(root):
    manifest_path = root / "REPORT_MANIFEST.json"
    if not manifest_path.is_file():
        return None, dict(status="WAITING", reason="No completed TEST report manifest", inputs=[])
    manifest = load_json(manifest_path)
    inputs = [manifest_path]
    test = manifest.get("completed_block_statuses", {}).get("test", {})
    if manifest.get("status") != "COMPLETED_TEST_REPORT" or test.get("status") != "COMPLETED":
        return None, dict(status="WAITING", reason="TEST report incomplete or has input issues; CSVs not read", inputs=inputs)
    if manifest.get("physical_test_sources") != list(SOURCES):
        return None, dict(status="WAITING_FOR_COMPARABLE_DATA", reason="Four declared TEST physical sources required", inputs=inputs)
    paths = {name: root / filename for name, filename in
             (("main", "main_source_equal.csv"), ("source", "main_per_physical_source.csv"),
              ("difference", "capacity_comparator_differences.csv"))}
    if any(not path.is_file() for path in paths.values()):
        return None, dict(status="WAITING", reason="Completed TEST report CSV mirror is incomplete", inputs=inputs)
    data = {name: rows(path) for name, path in paths.items()}
    inputs.extend(paths.values())
    main, source, difference = {}, {}, {}
    for row in data["main"]:
        if row.get("phase") == "test":
            key = (row["policy_label"], row["budget_id"])
            if key in main:
                raise ValueError("Duplicate TEST aggregate row: " + str(key))
            main[key] = row
    for row in data["source"]:
        if row.get("phase") == "test":
            key = (row["policy_label"], row["budget_id"], int(row["source"]))
            if key in source:
                raise ValueError("Duplicate TEST physical-source row: " + str(key))
            source[key] = row
    for row in data["difference"]:
        if row.get("phase") == "test" and row.get("target") == "FullCapacity-mean17/29":
            key = (row["comparator"], row["budget_id"])
            if key in difference:
                raise ValueError("Duplicate TEST paired comparison: " + str(key))
            difference[key] = row
    required = [style["key"] for style in METHODS.values()] + [key for keys in SEED_KEYS.values() for key in keys]
    for key in required:
        for budget in BUDGETS:
            row = main.get((key, budget))
            if row is None or not truth(row.get("grid_complete")) or number(row.get("physical_source_count")) != 4:
                raise ValueError("Missing or incomplete four-source curve point: " + str((key, budget)))
            for metric in ("deadline_seconds", "strict_gain_seconds", "on_time_rate"):
                if number(row.get(metric)) is None:
                    raise ValueError("Unknown plotted metric, not imputed: " + str((key, budget, metric)))
            if not -1e-12 <= number(row["on_time_rate"]) <= 1. + 1e-12:
                raise ValueError("On-time rate outside [0,1]")
    x = [number(main[(METHODS["Capacity"]["key"], b)]["deadline_seconds"]) for b in BUDGETS]
    if not (0 < x[0] < x[1] < x[2]) or not math.isclose(x[1], 2 * x[0], rel_tol=1e-12) or not math.isclose(x[2], 2 * x[1], rel_tol=1e-12):
        raise ValueError("Expected three exact frozen quarter/half/wide deadlines")
    for key in required:
        for b, expected in zip(BUDGETS, x):
            if number(main[(key, b)]["deadline_seconds"]) != expected:
                raise ValueError("Method curve points have different D")
    for name, keys in SEED_KEYS.items():
        for b in BUDGETS:
            expected = sum(number(main[(key, b)]["strict_gain_seconds"]) for key in keys) / 2.
            if not math.isclose(expected, number(main[(METHODS[name]["key"], b)]["strict_gain_seconds"]), rel_tol=1e-12, abs_tol=1e-7):
                raise ValueError("Two-seed mean disagrees with per-seed TEST values")
    for name in COMPARATORS:
        comparator = METHODS[name]["key"]
        for b in BUDGETS:
            row = difference.get((comparator, b))
            if row is None or not truth(row.get("comparable_inputs")):
                raise ValueError("Signed comparison is absent or input-incomparable")
            value = number(row.get("delta_strict_gain_seconds"))
            expected = number(main[(METHODS["Capacity"]["key"], b)]["strict_gain_seconds"]) - number(main[(comparator, b)]["strict_gain_seconds"])
            if value is None or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-7):
                raise ValueError("Signed curve disagrees with TEST aggregate")
    reference = max(REFERENCE_CANDIDATES, key=lambda key: number(main[(key, "wide")]["strict_gain_seconds"]))
    source_gap = []
    for physical in SOURCES:
        pair = [source.get((key, "wide", physical)) for key in (METHODS["Capacity"]["key"], reference)]
        if any(row is None or not truth(row.get("grid_complete")) or number(row.get("graph_count")) != 6 for row in pair):
            raise ValueError("All six configurations required for every panel D source")
        values = [number(row.get("strict_gain_seconds")) for row in pair]
        if any(value is None for value in values):
            raise ValueError("Unknown physical-source gap, not omitted")
        source_gap.append(values[0] - values[1])
    return dict(x=x, main=main, difference=difference, source_gaps=source_gap, reference=reference,
                wide_D=x[-1], synthetic_guard=manifest.get("guard_synthetic_fixture") is True), dict(status="READY", reason="Completed comparable TEST report", inputs=inputs)


def style_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#7F8991")
        ax.spines[side].set_linewidth(.65)
    ax.tick_params(colors=INK, labelsize=7.2, width=.6, length=3.)
    ax.grid(axis="y", color=GRID, linewidth=.55, alpha=.9)
    ax.set_axisbelow(True)


def draw(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8., "axes.labelsize": 8.,
        "axes.titlesize": 8.8, "axes.titleweight": "bold", "text.color": INK,
        "axes.labelcolor": INK, "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True})
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05))
    fig.subplots_adjust(left=.095, right=.985, bottom=.18, top=.825, wspace=.35, hspace=.48)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    x = np.asarray(data["x"])
    def metric(style, field):
        return np.asarray([number(data["main"][(style["key"], b)][field]) for b in BUDGETS])
    handles = []
    for name, style in METHODS.items():
        y = metric(style, "strict_gain_seconds")
        handle, = ax_a.plot(x, y, color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                            linewidth=1.4, markersize=4.2, markeredgewidth=.7, label=style["label"])
        handles.append(handle)
        if name in SEED_KEYS:
            seeds = np.asarray([[number(data["main"][(key, b)]["strict_gain_seconds"]) for b in BUDGETS] for key in SEED_KEYS[name]])
            ax_a.fill_between(x, seeds.min(axis=0), seeds.max(axis=0), color=style["color"], alpha=.14, linewidth=0.)
        ax_c.plot(x, 100. * metric(style, "on_time_rate"), color=style["color"], marker=style["marker"],
                  linestyle=style["linestyle"], linewidth=1.4, markersize=4.2, markeredgewidth=.7)
    for name in COMPARATORS:
        style = METHODS[name]
        y = [number(data["difference"][(style["key"], b)]["delta_strict_gain_seconds"]) for b in BUDGETS]
        ax_b.plot(x, y, color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                  linewidth=1.4, markersize=4.2, markeredgewidth=.7)
    ax_b.axhline(0., color=INK, linewidth=.8, zorder=1)
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xticks(x, ["%.3f" % value for value in x])
        ax.set_xlim(x[0] - .06 * (x[-1] - x[0]), x[-1] + .06 * (x[-1] - x[0]))
        ax.set_xlabel("Decision allowance D (s)")
    ax_a.set_title("A  Strict delivered improvement", loc="left", pad=6)
    ax_a.set_ylabel("Strict gain (s) ↑")
    low, high = ax_a.get_ylim()
    ax_a.set_ylim(min(0., low), max(high, 1.))
    ax_b.set_title("B  JointRecovery − comparator", loc="left", pad=6)
    ax_b.set_ylabel("Signed strict gain gap (s) ↑")
    low, high = ax_b.get_ylim()
    ax_b.set_ylim(min(low, 0.), max(high, 0.))
    ax_c.set_title("C  Complete caller return", loc="left", pad=6)
    ax_c.set_ylabel("On-time return rate (%) ↑")
    ax_c.set_ylim(-4., 104.)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    gap = np.asarray(data["source_gaps"])
    positions = np.arange(len(SOURCES))
    for i, value in enumerate(gap):
        ax_d.bar(i, value, width=.54, color=METHODS["Capacity"]["color"] if value >= 0 else "white",
                 edgecolor=METHODS["Capacity"]["color"], linewidth=.8, hatch=None if value >= 0 else "///", zorder=3)
    extent = max(float(np.abs(gap).max()), 1.)
    ax_d.set_ylim(-1.26 * extent, 1.26 * extent)
    for position, value in zip(positions, gap):
        text = "%+.1f" % value if abs(value) < 100 else "%+.0f" % value
        ax_d.text(position, value + (.055 * extent if value >= 0 else -.055 * extent), text,
                  ha="center", va="bottom" if value >= 0 else "top", fontsize=7.2, color=INK)
    ax_d.axhline(0., color=INK, linewidth=.8, zorder=2)
    ax_d.set_xticks(positions, ["r%03d" % source for source in SOURCES])
    ax_d.set_xlabel("Held-out physical STK source")
    ax_d.set_ylabel("JointRecovery − public reference (s) ↑")
    reference_label = "CHILS-p1" if data["reference"] == "CHILS-p1" else "HiGHS (generic MILP)"
    ax_d.set_title("D  Four-source gap at fixed wide D", loc="left", pad=13)
    ax_d.text(.025, .965, "Reference: %s; D=%.3f s" % (reference_label, data["wide_D"]), transform=ax_d.transAxes,
              ha="left", va="top", fontsize=6.9, color=INK)
    for ax in axes.flat:
        style_axes(ax)
        if ax is not ax_c:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 4)); formatter.set_useOffset(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.get_offset_text().set_fontsize(7.)
    fig.legend(handles=handles, labels=[s["label"] for s in METHODS.values()], loc="upper center",
               bbox_to_anchor=(.54, .988), ncol=3, frameon=False, fontsize=7.2,
               handlelength=2.6, columnspacing=1.7, handletextpad=.6, labelspacing=.65)
    if data.get("synthetic_guard"):
        fig.text(.5, .88, "SYNTHETIC LAYOUT GUARD — NOT EXPERIMENTAL RESULTS", ha="center", fontsize=7., color="#7F8991")
    fig.text(.095, .056, "Shading: two-fit-seed min–max, not confidence intervals. Lines join only the three measured deadlines.", fontsize=6.9)
    fig.text(.095, .027, "Short-budget efficiency / stress diagnostic. Failed/late deliveries remain; ample-budget quality is reported separately.", fontsize=6.9)
    fig.canvas.draw()
    png, pdf = out / "submitted_comparison_2x2.png", out / "submitted_comparison_2x2.pdf"
    fig.savefig(png, dpi=360)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def run(args):
    root = args.results_root
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    try:
        data, receipt = prepare(root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        data, receipt = None, dict(status="WAITING_FOR_COMPARABLE_DATA", reason=str(error), inputs=[])
    if data is None:
        for path in (out / "submitted_comparison_2x2.png", out / "submitted_comparison_2x2.pdf"):
            if path.exists():
                raise RuntimeError("Existing scientific figure is preserved; current inputs are not eligible")
        write_json(out / "FIGURE_MANIFEST.json", dict(schema=SCHEMA, status=receipt["status"], reason=receipt["reason"],
                   created_utc=datetime.now(timezone.utc).isoformat(), raw_or_cloud_rows_read=False))
        (out / "FIGURE_CAPTION_ZH.md").write_text("# 完整方法测试比较图\n\nWAITING：" + receipt["reason"] + "。未绘制不完整测试数据，没有读取cloud/live原始运行行。\n", encoding="utf-8")
        print(json.dumps(dict(status=receipt["status"], reason=receipt["reason"]), ensure_ascii=False))
        return
    outputs = draw(data, out)
    reference = data["reference"]
    caption = """# 四面板科学结果图说明

**完整在线 STK 迁移框架的独立 TEST 比较。** A：三组预先冻结的实际完整决策预算 D 下，各方法的严格按时交付增量（原始 contact 完整时长秒，↑）。Capacity、CheapSummary 为 seed17/29 均值；阴影只表示这两个fit的min–max观测范围，**不是置信区间**。B：Capacity 两seed均值减去 CheapSummary均值、FullGreedy、CHILS-p1、HiGHS 的有符号严格增量差，颜色和标记对应被减去的参照，0线区分胜负。C：所有已声明cell的实际caller按时返回率（%，↑），失败/未知提交不计按时，迟到仍保留。D：在**预定wide D={wide:.9f}秒**，四个物理来源的 Capacity均值减去同一更强公共算法/软件参照 `{reference}`；wide不是事后选择对我方最有利的D。

D参照在 CHILS-p1 与 HiGHS-MILP 中，依据预定wide D的四母源等权严格增量均值取较强者，所有来源使用同一个参照，未逐源挑不同算法。CHILS-p1是SEA2025正式发表的MWIS方法；HiGHS只是通用MILP软件参照，**不能称其为另一篇专门MWIS发表方法**。D的负差值使用空心斜线柱；正负均展示，不删不利来源。A/B/C始终显示全部六类方法。

三个D的精确值来自已完成报告CSV：{deadlines}。连线与阴影边界只连接三个实际观测点，不做平滑、不额外生成预算观测。重合曲线表示实际相同或接近的观测；不为分离曲线抖动D或修改数值。

先对每母源六配置等权，再对r008–r011四母源等权；模型两seed并非两个新物理来源。所有失败、零增量和迟到均由已完成报告的driver严格交付统计保留。总收益包含共同三贪心前缀；不能将全部native/prefix收益归给学习，匹配学习分配差异主要应看B中与同框架CheapSummary/FullGreedy的比较。

remaining-head始终0，未验证其学习适应；完整闭环不等于旧稿全部组件验证。Full方法cap256、11动作、最多8请求；整图CHILS/HiGHS搜索范围不同，但原图、原S、原时长目标和完整caller D一致。HiGHS的SciPy1.10.1接口没有solver内x0 warm start。新增比较在原P2开始后、PLAN记载查看原VAL/TEST结果前冻结，不宣称更早预注册。

正式依据：[CHILS，SEA2025，DOI10.4230/LIPIcs.SEA.2025.22](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2025.22)；[SciPy1.10.1 milp官方文档](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.optimize.milp.html)、[HiGHS官方软件站](https://highs.dev/)。

PNG用于浏览，PDF为可缩放矢量图、嵌入TrueType字体；轴与图内文字使用英文，便于论文复用。统一字体DejaVu Sans、固定方法颜色、不同标记/线型、统一线宽和字号。没有改论文或运行新实验。
""".format(wide=data["wide_D"], reference=reference, deadlines=", ".join(format(v, ".17g") for v in data["x"]))
    caption = "**用途：效率与短预算压力诊断；充裕预算求解质量另见 JOINTRECOVERY_QUALITY_RESULTS。本文方法为 JointRecovery（JR），原键 FullCapacity；P1 是独立替代对照。**\n\n" + caption
    caption = caption.replace("Capacity、CheapSummary", "JointRecovery（JR）、JR-CheapSummary").replace("Capacity 两seed", "JointRecovery 两seed").replace("Capacity均值", "JointRecovery均值").replace("CheapSummary均值、FullGreedy", "JR-CheapSummary均值、Greedy-rank+CHILS").replace("同框架CheapSummary/FullGreedy", "同框架JR-CheapSummary/Greedy-rank+CHILS")
    (out / "FIGURE_CAPTION_ZH.md").write_text(caption, encoding="utf-8")
    if data.get("synthetic_guard"):
        (out / "FIGURE_CAPTION_ZH.md").write_text("【合成布局smoke，不是实验结果】\n\n" + caption, encoding="utf-8")
    figure_status = "SYNTHETIC_LAYOUT_GUARD_ONLY" if data.get("synthetic_guard") else "COMPLETED_TEST_FIGURES"
    write_json(out / "FIGURE_MANIFEST.json", dict(schema=SCHEMA, status=figure_status,
        generated_utc=datetime.now(timezone.utc).isoformat(), generator_sha256=sha(__file__),
        phase="test", physical_sources=list(SOURCES), exact_deadlines_seconds=data["x"],
        report_role="EFFICIENCY_AND_SHORT_BUDGET_STRESS_DIAGNOSTIC", method_display_mapping=METHOD_MAPPING,
        learned_seed_range="minmax17/29_not_CI", smoothed_or_added_budget_points=False,
        panel_D_budget="wide_predeclared_not_favourable_budget_selection", panel_D_reference=reference,
        panel_D_reference_rule="larger four-source-equal strict gain at fixed wide D among CHILS/HiGHS",
        HiGHS_is_generic_MILP_not_second_published_MWIS_method=True,
        figure_size_inches=[7.15, 5.05], font="DejaVu Sans", method_style_map=METHODS,
        raw_or_cloud_rows_read=False, real_completed_report_csvs_only=not data.get("synthetic_guard"),
        inputs=[dict(path=str(p.resolve()), sha256=sha(p)) for p in receipt["inputs"]],
        outputs=[dict(path=str(p.resolve()), sha256=sha(p)) for p in outputs]))
    print(json.dumps(dict(status=figure_status, outputs=[str(p) for p in outputs], reference=reference), ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("reports/SUBMITTED_METHOD_RESULTS"))
    args = parser.parse_args()
    dataset = Path(__file__).resolve().parents[1]
    args.results_root = args.results_root.resolve() if args.results_root.is_absolute() else (dataset / args.results_root).resolve()
    return args


if __name__ == "__main__":
    run(parse_args())
