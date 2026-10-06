"""Finite 2x2 quality curves from completed LOCAL quality-report CSVs only.

Each D comes from a separate real execution. No cloud, raw/live outcomes,
experiments, smoothing, success-only averages, or native-quality imputation.
Null quality still produces the observed completion/output/late curves.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "joint_recovery_stk_quality_figures_v1"
METHODS = (
    dict(key="FullCapacity-mean17/29", label="JointRecovery (JR, ours)", color="#BE6628", marker="o", line="-"),
    dict(key="FullCheapSummary-mean17/29", label="JR-CheapSummary", color="#65899E", marker="s", line="--"),
    dict(key="FullGreedy", label="Greedy-rank+CHILS", color="#294E66", marker="D", line="-."),
    dict(key="FullP1", label="Independent-replacement+CHILS", color="#91A6B5", marker="^", line=":"),
    dict(key="CHILS-p1", label="CHILS-p1 (SEA 2025)", color="#43647C", marker="P", line=(0, (5, 2))),
    dict(key="HiGHS-MILP", label="HiGHS (generic MILP)", color="#6F7A86", marker="X", line=(0, (3, 1, 1, 1))),
)
SEED_KEYS = {
    "FullCapacity-mean17/29": ("FullCapacity-seed17", "FullCapacity-seed29"),
    "FullCheapSummary-mean17/29": ("FullCheapSummary-seed17", "FullCheapSummary-seed29"),
}
METRICS = {
    "gain_percent": ("complete_gain_percent", "Complete feasible gain over S (%) ↑"),
    "gain_seconds": ("complete_gain_seconds", "Complete feasible gain over S (s) ↑"),
    "objective_seconds": ("complete_objective_seconds", "Complete feasible objective (s) ↑"),
}
PHASE_SOURCES = {"test": (8, 9, 10, 11), "validation": (6, 7)}
INK, GRID = "#26313B", "#E1E6EA"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
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


def prepare(report_root, phase):
    manifest_path = report_root / "QUALITY_REPORT_MANIFEST.json"
    inputs = []
    if not manifest_path.is_file():
        return None, dict(status="WAITING", reason="Completed local quality manifest is unavailable", inputs=inputs)
    inputs.append(manifest_path)
    manifest = read_json(manifest_path)
    block = manifest.get("completed_block_statuses", {}).get(phase, {})
    if block.get("status") != "COMPLETED":
        return None, dict(status="WAITING", reason=f"{phase} is not a completed local block; no CSV read", inputs=inputs)
    allowance = manifest.get("selected_allowance") or {}
    selected = number(allowance.get("selected_deadline_seconds"))
    deadlines = allowance.get("evaluation_deadline_seconds")
    expected_D = sorted(set([10., 30., 60., 120., selected])) if selected in (60., 120., 300.) else None
    if (allowance.get("schema") != "joint_recovery_stk_quality_allowance_v1" or
            allowance.get("status") != "QUALITY_ALLOWANCE_PROTOCOL" or
            allowance.get("stage") != "frozen_from_development_completion" or
            allowance.get("selection_uses_gain") is not False or allowance.get("test_outcomes_used") is not False or
            deadlines != expected_D):
        raise ValueError("No fixed DEV-completion-only allowance and independently executed curve grid")
    if manifest.get("quality_null_not_zero_for_no_complete_output") is not True or manifest.get("success_only_means_not_used") is not True:
        raise ValueError("Quality report must preserve nulls and the full population")
    paths = {"main": report_root / "quality_source_equal.csv",
             "source": report_root / "quality_per_physical_source.csv",
             "difference": report_root / "jointrecovery_quality_comparator_differences.csv"}
    if any(not path.is_file() for path in paths.values()):
        return None, dict(status="WAITING", reason="Completed local CSV mirror is incomplete", inputs=inputs)
    values = {key: read_csv(path) for key, path in paths.items()}
    inputs.extend(paths.values())
    main, source, differences = {}, {}, {}
    for row in values["main"]:
        if row.get("phase") == phase:
            key = (row["policy_label"], number(row["deadline_seconds"]))
            if key in main:
                raise ValueError("Duplicate aggregate point: " + str(key))
            main[key] = row
    for row in values["source"]:
        if row.get("phase") == phase:
            key = (row["policy_label"], number(row["deadline_seconds"]), int(row["source"]))
            if key in source:
                raise ValueError("Duplicate source point: " + str(key))
            source[key] = row
    for row in values["difference"]:
        if row.get("phase") == phase and row.get("target") == "FullCapacity-mean17/29":
            key = (row["comparator"], number(row["deadline_seconds"]))
            if key in differences:
                raise ValueError("Duplicate signed comparison: " + str(key))
            differences[key] = row
    all_keys = [m["key"] for m in METHODS] + [key for pair in SEED_KEYS.values() for key in pair]
    physical_sources = PHASE_SOURCES[phase]
    for key in all_keys:
        for D in deadlines:
            row = main.get((key, D))
            if (not row or not truth(row.get("grid_complete")) or
                    number(row.get("physical_source_count")) != len(physical_sources) or
                    number(row.get("received_cells")) != number(row.get("declared_cells"))):
                raise ValueError("Missing complete declared curve grid: " + str((key, D)))
            if truth(row.get("is_primary_quality_allowance")) != (D == selected):
                raise ValueError("Primary table marker differs from selectedD")
            for field in ("complete_result_rate", "native_output_rate", "observed_late_rate"):
                rate = number(row.get(field))
                if rate is None or not 0 <= rate <= 1:
                    raise ValueError("Unknown or invalid coverage denominator: " + str((key, D, field)))
            for physical in physical_sources:
                sr = source.get((key, D, physical))
                if not sr or not truth(sr.get("grid_complete")) or number(sr.get("graph_count")) != 6:
                    raise ValueError("All six configurations must remain in each physical source")
            # A null quality aggregate is legitimate; it is never replaced by 0.
            for metric in ("complete_objective_seconds", "complete_gain_seconds", "complete_gain_percent"):
                metric_value = number(row.get(metric))
                per_source = [number(source[(key, D, s)].get(metric)) for s in physical_sources]
                if metric_value is not None:
                    if any(v is None for v in per_source) or not math.isclose(metric_value, math.fsum(per_source) / len(per_source), rel_tol=1e-10, abs_tol=1e-7):
                        raise ValueError("Plotted quality is not equal-physical-source complete quality")
                    if number(row.get("incomplete_quality_cells")) != 0:
                        raise ValueError("Quality reported despite incomplete production cells")
    for comparator in ("CHILS-p1", "HiGHS-MILP"):
        for D in deadlines:
            row = differences.get((comparator, D))
            if row is None:
                raise ValueError("Missing paired comparison record, including N/A rows")
            a = number(main[("FullCapacity-mean17/29", D)].get("complete_objective_seconds"))
            b = number(main[(comparator, D)].get("complete_objective_seconds"))
            delta = number(row.get("delta_complete_objective_seconds"))
            if truth(row.get("comparable_complete_quality")):
                if a is None or b is None or delta is None or not math.isclose(delta, a - b, rel_tol=1e-10, abs_tol=1e-7):
                    raise ValueError("Signed curve differs from comparable complete-source quality")
            elif delta is not None:
                raise ValueError("Incomparable native quality must have a null signed difference")
    return dict(main=main, source=source, differences=differences, D=deadlines, selected_D=selected,
                phase=phase, physical_sources=physical_sources), dict(status="READY", inputs=inputs)


def style_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#ABB5BE")
        ax.spines[side].set_linewidth(.65)
    ax.tick_params(axis="both", colors=INK, labelsize=7.1, width=.65, length=3)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.set_axisbelow(True)


def deadline_axis(ax, deadlines, selected, labels=True):
    from matplotlib.ticker import ScalarFormatter
    ax.set_xscale("log")
    ax.set_xticks(deadlines)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlim(min(deadlines) / 1.12, max(deadlines) * 1.12)
    ax.axvline(selected, color="#AEB7BE", linestyle=(0, (2, 3)), linewidth=.65, zorder=0)
    if labels:
        ax.set_xlabel("Allowed caller budget D (s; log scale)")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def curve(ax, x, y, method, lower=None, upper=None):
    import numpy as np
    yy = np.asarray([float("nan") if v is None else v for v in y], dtype=float)
    xx = np.asarray([float("nan") if v is None else v for v in x], dtype=float)
    ax.plot(xx, yy, color=method["color"], marker=method["marker"], linestyle=method["line"],
            linewidth=1.25, markersize=4.1, markeredgewidth=.65, label=method["label"], zorder=4 if method is METHODS[0] else 3)
    if lower is not None and upper is not None:
        lo = np.asarray([float("nan") if v is None else v for v in lower], dtype=float)
        hi = np.asarray([float("nan") if v is None else v for v in upper], dtype=float)
        ax.fill_between(xx, lo, hi, color=method["color"], alpha=.12, linewidth=0, zorder=1)


def seed_range(data, key, metric):
    pair = SEED_KEYS.get(key)
    if not pair:
        return None, None
    values = [[number(data["main"][(k, D)].get(metric)) for k in pair] for D in data["D"]]
    return ([min(v) if all(x is not None for x in v) else None for v in values],
            [max(v) if all(x is not None for x in v) else None for v in values])


def render(data, out, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.8, "axes.labelsize": 7.6,
        "axes.titlesize": 8.2, "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    field, ylabel = METRICS[metric]
    D, main = data["D"], data["main"]
    fig = plt.figure(figsize=(7.8, 6.0), layout=None)
    grid = fig.add_gridspec(2, 2, left=.125, right=.985, bottom=.18, top=.855,
                           hspace=.43, wspace=.36)
    a, b, c = [fig.add_subplot(grid[i, j]) for i, j in ((0, 0), (0, 1), (1, 0))]
    dg = grid[1, 1].subgridspec(3, 1, hspace=.20)
    ds = [fig.add_subplot(dg[i, 0]) for i in range(3)]
    for ax in (a, b, c, *ds):
        style_axes(ax)
    a.set_title("(a) Allowance and complete-result quality", loc="left", fontweight="bold")
    b.set_title("(b) Actual caller cost and quality", loc="left", fontweight="bold")
    c.set_title("(c) JointRecovery minus external references", loc="left", fontweight="bold")
    ds[0].set_title("(d) Completion, native output and lateness", loc="left", fontweight="bold")
    unknown = []
    known_quality = 0
    for method in METHODS:
        key = method["key"]
        y = [number(main[(key, d)].get(field)) for d in D]
        xcost = [number(main[(key, d)].get("caller_seconds")) for d in D]
        xcost = [x if x is not None and x > 0 else None for x in xcost]
        lo, hi = seed_range(data, key, field)
        curve(a, D, y, method, lo, hi)
        curve(b, xcost, y, method, lo, hi)
        cost_lo, cost_hi = seed_range(data, key, "caller_seconds")
        if cost_lo is not None:
            for x, value, low, high in zip(xcost, y, cost_lo, cost_hi):
                if x is not None and value is not None and low is not None and high is not None:
                    b.errorbar(x, value, xerr=[[max(0., x - low)], [max(0., high - x)]],
                               color=method["color"], fmt="none", elinewidth=.65, capsize=1.7, alpha=.7)
        known_quality += sum(v is not None for v in y)
        unknown.extend(dict(policy_label=key, deadline_seconds=d, native_complete_quality="N/A")
                       for d, value in zip(D, y) if value is None)
        for ax, rate in zip(ds, ("complete_result_rate", "native_output_rate", "observed_late_rate")):
            rates = [number(main[(key, d)].get(rate)) for d in D]
            yy = [100. * (1. - v if rate == "native_output_rate" else v) if v is not None else None for v in rates]
            curve(ax, D, yy, method)
    deadline_axis(a, D, data["selected_D"])
    deadline_axis(c, D, data["selected_D"])
    for i, (ax, label) in enumerate(zip(ds, ("Complete ↑\n(%)", "No native ↓\n(%)", "Late ↓\n(%)"))):
        deadline_axis(ax, D, data["selected_D"], labels=i == 2)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 100])
        ax.set_ylabel(label, fontsize=6.8, labelpad=3)
    b.set_xscale("log")
    b.set_xlabel("Actual full caller return (s; log scale) ↓")
    for ax in (a, b):
        ax.set_ylabel(ylabel)
    c.set_ylabel("JR − reference objective (s) ↑")
    c.axhline(0, color="#858F98", linestyle="--", linewidth=.75)
    comparable_points = 0
    for method in METHODS[-2:]:
        yy, lo, hi = [], [], []
        for d in D:
            row = data["differences"][(method["key"], d)]
            value = number(row.get("delta_complete_objective_seconds")) if truth(row.get("comparable_complete_quality")) else None
            yy.append(value)
            comparable_points += int(value is not None)
            reference = number(main[(method["key"], d)].get("complete_objective_seconds"))
            fits = [number(main[(key, d)].get("complete_objective_seconds")) for key in SEED_KEYS["FullCapacity-mean17/29"]]
            valid = value is not None and reference is not None and all(v is not None for v in fits)
            lo.append(min(fits) - reference if valid else None)
            hi.append(max(fits) - reference if valid else None)
        curve(c, D, yy, dict(method, label="JR − " + method["label"]), lo, hi)
    c.legend(loc="best", fontsize=6.7, frameon=False, handlelength=2.3)
    if known_quality == 0:
        for ax in (a, b):
            ax.text(.5, .5, "No complete-result quality available\nCoverage remains visible in (d)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7.1, color="#6C7780")
    if comparable_points == 0:
        c.text(.5, .5, "No complete comparable native output\nSigned differences are N/A", ha="center", va="center",
               transform=c.transAxes, fontsize=7.1, color="#6C7780")
    handles, labels = a.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .973), ncol=3,
                        frameon=False, fontsize=7.15, columnspacing=1.4, handlelength=2.7)
    for text in legend.get_texts():
        if text.get_text().startswith("JointRecovery"):
            text.set_fontweight("bold")
    fig.text(.125, .033,
        f"{data['phase'].upper()}: {len(data['physical_sources'])} physical sources × 6 configurations; selectedD={data['selected_D']:g}s. "
        "Two-fit min–max bands/spans; not CI.\n"
        "Each point is independently executed; gaps indicate unavailable complete quality.\n"
        "Straight segments guide measured points, not intermediate estimates.",
        fontsize=6.75, color="#5C6872", va="bottom")
    stem = out / ("quality_curves_" + data["phase"])
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.025)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)
    return dict(metric=field, quality_points=sum(len(D) for _ in METHODS), known_quality_points=known_quality,
                unknown_quality_points=unknown, comparable_external_delta_points=comparable_points,
                matplotlib_version=matplotlib.__version__, outputs=[stem.with_suffix(".pdf"), stem.with_suffix(".png")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(PHASE_SOURCES), default="test")
    parser.add_argument("--metric", choices=tuple(METRICS), default="gain_percent")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    generated = dict(schema=SCHEMA, generated_utc=datetime.now(timezone.utc).isoformat(), phase=args.phase,
                     generator_sha256=sha(__file__), plots_from_completed_local_csv_only=True,
                     cloud_or_native_called=False, smoothing_or_quality_imputation=False,
                     seed_range_is_not_confidence_interval=True)
    try:
        data, receipt = prepare(args.report_root.resolve(), args.phase)
        generated.update(status=receipt["status"], inputs=[dict(path=str(p), sha256=sha(p)) for p in receipt["inputs"]])
        if data is None:
            generated["reason"] = receipt["reason"]
        else:
            rendered = render(data, out, args.metric)
            outputs = rendered.pop("outputs")
            generated.update(status="COMPLETE_QUALITY_FIGURES", selected_deadline_seconds=data["selected_D"],
                             evaluation_deadline_seconds=data["D"], **rendered,
                             outputs=[dict(path=str(p), sha256=sha(p)) for p in outputs])
            chart_rows = [dict(row) for (key, d), row in data["main"].items() if key in {m["key"] for m in METHODS}]
            write_json(out / "QUALITY_FIGURE_DATA.json", dict(phase=args.phase, physical_sources=data["physical_sources"],
                        deadline_seconds=data["D"], source_equal_chart_rows=chart_rows,
                        signed_delta_rows=list(data["differences"].values()),
                        unknown_quality_points=rendered["unknown_quality_points"]))
    except Exception as error:
        generated.update(status="QUALITY_FIGURE_INPUT_OR_RENDER_ERROR", reason=f"{type(error).__name__}: {error}")
    write_json(out / "QUALITY_FIGURE_MANIFEST.json", generated)
    (out / "QUALITY_FIGURES_ZH.md").write_text(
        "# 多时间点质量图\n\n状态：" + generated["status"] + "。\n\n"
        "独立真实执行的允许预算→完整质量、实际 caller 成本→完整质量、有符号公开参照差，以及完成率/无native/迟到率组成统一2×2面板。"
        "所有配置、物理来源和种子保持报告分母；缺完整质量留空，不填零或只筛成功。实际晚到raw质量与保底S不混入完整native质量。"
        "真实合法空域完成与无native输出可并存，覆盖指标不等同于算法增益。\n\n"
        "阴影仅为两模型fit seed min–max，不是置信区间。折线只连接已执行点，不平滑、不使用内部日志推曲线。"
        "该图是实验报告，不替代论文和完整原始报告。\n\n"
        "源码/CSV哈希、缺失点与输出见 QUALITY_FIGURE_MANIFEST.json，图用数据见 QUALITY_FIGURE_DATA.json。"
        + ("本次已基于完整本地 CSV 生成真实结果图，未启动新实验。\n" if generated["status"] == "COMPLETE_QUALITY_FIGURES"
         else "当前完整本地输入尚未满足绘图条件，保留状态收据。\n"),
        encoding="utf-8")
    print(json.dumps({"status": generated["status"], "out": str(out)}, ensure_ascii=False))
    return 2 if generated["status"] == "QUALITY_FIGURE_INPUT_OR_RENDER_ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
