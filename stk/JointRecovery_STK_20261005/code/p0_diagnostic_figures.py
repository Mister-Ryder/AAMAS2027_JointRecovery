"""Compact, source-backed static P0 diagnostic figures (PDF + PNG).

Only real analyzer CSV/JSON rows are drawn. Zeros remain visible; unavailable
signals are labelled as unavailable, and observation spreads are not CIs.
No plot represents a learned-policy result. Resource-type proportions use the
separately labelled full-graph validation report, not fabricated local cliques.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

VIEW_COLORS = {"R8": "#2F6B9A", "R12": "#CA8C28"}
RESOURCE_COLORS = {"ground_only_edges": "#2F6B9A", "satellite_only_edges": "#CA8C28", "double_resource_edges": "#AB5B40"}
INK, GREY, GRID = "#26313B", "#7F8991", "#E0E5E9"
SOURCE_MARKERS = ("o", "s", "D", "^")
GAP_MARKERS = {170: "o", 340: "s", 680: "D"}


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row.get("status") != "NO_AVAILABLE_ROWS"]


def number(row, key):
    value = row.get(key)
    if value in (None, "", "None", "null", "nan"):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def truth(row, key):
    return str(row.get(key, "")).strip().lower() == "true"


def identity(graph_id):
    match = re.search(r"-(r\d+)-(R8|R12)-g(\d+)$", str(graph_id))
    return (match.group(1), match.group(2), int(match.group(3))) if match else ("unknown", "unknown", 0)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def style():
    font = font_manager.findfont(font_manager.FontProperties(family=["Arial", "DejaVu Sans"]), fallback_to_default=True)
    family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 7.6, "axes.titlesize": 8.6,
        "axes.labelsize": 7.6, "xtick.labelsize": 7.1, "ytick.labelsize": 7.1,
        "legend.fontsize": 6.7, "axes.titleweight": "semibold", "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": GREY, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": 0.65, "lines.linewidth": 1.2,
        "lines.markersize": 3.6, "grid.color": GRID, "grid.linewidth": 0.5,
        "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return family


def panel(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def unavailable(ax, message):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            color=GREY, fontsize=8, linespacing=1.3)
    ax.set_xticks([]); ax.set_yticks([])


def export(fig, out, name):
    # Research house-style blossom is fixed to the header corner, outside data.
    fig.text(0.989, 0.994, "❋", ha="right", va="top", color=GREY, fontsize=7,
             fontfamily="DejaVu Sans")
    paths = []
    for extension in ("pdf", "png"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.035)
        paths.append(path)
    plt.close(fig)
    return paths


def config_order(graph_rows):
    present = {(identity(row.get("graph_id"))[1], identity(row.get("graph_id"))[2]) for row in graph_rows}
    return [(view, gap) for gap in (170, 340, 680) for view in ("R8", "R12") if (view, gap) in present]


def main_diagnostic(graph_rows, state_rows, request_rows, budget_pairs, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.48), layout="constrained")
    fig.suptitle("P0: actual recovery opportunities and complete conditional costs", fontsize=10, x=0.5)
    a, b, c, d = axes.ravel()
    sources = sorted({identity(row.get("graph_id"))[0] for row in graph_rows})
    source_markers = {source: SOURCE_MARKERS[i % len(SOURCE_MARKERS)] for i, source in enumerate(sources)}
    panel(a, "(a) Observed extra-opportunity states", "Ground switching gap (s)", "G1 opportunity fraction (%) ↑")
    for source in sources:
        for view in ("R8", "R12"):
            rows = sorted((row for row in graph_rows if identity(row.get("graph_id"))[:2] == (source, view)),
                          key=lambda row: identity(row["graph_id"])[2])
            values = [(identity(row["graph_id"])[2], number(row, "G1_fraction")) for row in rows]
            values = [(x, 100 * y) for x, y in values if y is not None]
            if values:
                a.plot(*zip(*values), color=VIEW_COLORS[view], marker=source_markers[source],
                       markerfacecolor=VIEW_COLORS[view] if sources.index(source) == 0 else "white",
                       linestyle="-" if sources.index(source) == 0 else "--", label=f"{view}/{source}")
    a.axhline(30, color=GREY, linestyle=":", linewidth=0.8)
    a.text(0.99, 30.8, "30%: development guide", ha="right", va="bottom", transform=a.get_yaxis_transform(), fontsize=6.3, color=GREY)
    a.set_xticks([170, 340, 680]); a.set_ylim(0, 105)
    if sources:
        a.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.3, columnspacing=0.8)
    else:
        unavailable(a, "No completed P0 graph summaries")
    panel(b, "(b) Real net increments; zeros retained", "Best observed extra linked seconds / state ↑ (symlog)", "State cumulative fraction (%)")
    unknown_states = 0
    distribution_count = 0
    maximum_gain = 0.0
    for source in sources:
        for view in ("R8", "R12"):
            group = [row for row in state_rows if identity(row.get("graph_id"))[:2] == (source, view)]
            unknown_states += sum(truth(row, "no_valid_actual_finite_outcomes") for row in group)
            values = [number(row, "max_added_seconds") for row in group if not truth(row, "no_valid_actual_finite_outcomes")]
            values = [x for x in values if x is not None]
            if not values:
                continue
            distribution_count += len(values)
            unique, counts = np.unique(values, return_counts=True)
            cumulative = 100 * np.cumsum(counts) / len(values)
            step_x = np.r_[0, unique] if unique[0] > 0 else unique
            step_y = np.r_[0, cumulative] if unique[0] > 0 else cumulative
            b.step(step_x, step_y, where="post", color=VIEW_COLORS[view],
                   linestyle="-" if sources.index(source) == 0 else "--", label=f"{view}/{source} (n={len(values)})")
            b.scatter(unique[0], cumulative[0], s=11, color=VIEW_COLORS[view], marker=source_markers[source], zorder=3)
            maximum_gain = max(maximum_gain, max(values))
    right_limit = max(1, maximum_gain * 1.12)
    for line in b.get_lines():
        xdata, ydata = line.get_data()
        line.set_data(np.r_[xdata, right_limit], np.r_[ydata, 100])
    b.set_xscale("symlog", linthresh=0.1); b.set_xlim(0, right_limit); b.set_ylim(0, 105)
    if distribution_count:
        b.legend(loc="lower right", frameon=False, fontsize=6.0, handlelength=1.3)
        b.text(0.02, 0.97, f"Unknown states: {unknown_states}; best-of-probes diagnostic", ha="left", va="top", transform=b.transAxes, fontsize=6.2, color=GREY)
    else:
        unavailable(b, "No finite state outcomes\nUnknown is not a zero")
    panel(c, "(c) Matched finite responses: 200 → 1000 ms", "Native search workpoint (ms)", "Recovery change from 200 ms (%) ↑")
    valid_pairs = []
    for row in budget_pairs:
        low, high = number(row, "recovery_200ms_seconds"), number(row, "recovery_1000ms_seconds")
        if low is not None and high is not None:
            valid_pairs.append((row, 100 * (high - low) / max(1, low)))
    if valid_pairs:
        for row, delta in valid_pairs:
            c.plot([200, 1000], [0, delta], color=GREY, alpha=0.055, linewidth=0.5, zorder=1)
        for view in ("R8", "R12"):
            values = [delta for row, delta in valid_pairs if identity(row.get("graph_id"))[1] == view]
            if values:
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                c.fill_between([200, 1000], [0, q25], [0, q75], color=VIEW_COLORS[view], alpha=0.13, zorder=2)
                c.plot([200, 1000], [0, median], color=VIEW_COLORS[view], marker="o", label=f"{view} median; IQR (n={len(values)})", zorder=3)
        c.axhline(0, color=GREY, linewidth=0.7)
        c.set_xticks([200, 1000]); c.set_xlim(155, 1045)
        deltas = [value for _, value in valid_pairs]
        if max(abs(x) for x in deltas) <= 1e-10:
            c.set_ylim(-0.1, 0.1)
            c.text(0.5, 0.75, "All recorded matched responses equal", transform=c.transAxes, ha="center", color=GREY, fontsize=6.8)
        c.legend(loc="upper left", frameon=False, fontsize=6.2)
    else:
        unavailable(c, "No valid same-warm/seed/budget pairs")
    panel(d, "(d) Complete conditional return cost", "Native search workpoint (ms)", "Return time (s) ↓; symlog")
    cost_series, positions, colors, native_points = [], [], [], []
    for budget_i, budget in enumerate((200, 1000)):
        native_points.append((budget_i + 1, budget / 1000))
        for view_i, view in enumerate(("R8", "R12")):
            values = [number(row, "complete_conditional_return_seconds") for row in request_rows
                      if identity(row.get("graph_id"))[1] == view and number(row, "workpoint_ms") == budget]
            values = [x for x in values if x is not None]
            if values:
                cost_series.append(values); positions.append(budget_i + 1 + (-0.15 if view_i == 0 else 0.15)); colors.append(VIEW_COLORS[view])
    if cost_series:
        boxes = d.boxplot(cost_series, positions=positions, widths=0.23, patch_artist=True,
                          medianprops={"color": INK, "linewidth": 1},
                          boxprops={"linewidth": 0.75}, whiskerprops={"linewidth": 0.75}, capprops={"linewidth": 0.75},
                          flierprops={"marker": ".", "markersize": 1.8, "alpha": 0.18, "markeredgecolor": GREY})
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.35); patch.set_edgecolor(color)
        d.plot(*zip(*native_points), color=GREY, linestyle=":", marker="_", markersize=10, linewidth=0.8)
        d.set_yscale("symlog", linthresh=0.001); d.set_ylim(bottom=0)
        d.set_xticks([1, 2], ["200", "1000"]); d.set_xlim(0.6, 2.4)
        d.legend(handles=[Patch(facecolor=VIEW_COLORS[v], alpha=0.35, label=v) for v in ("R8", "R12")]
                 + [Line2D([0], [0], color=GREY, linestyle=":", label="Native slice; not deadline")],
                 loc="upper left", frameon=False, fontsize=6.2)
        d.text(0.02, 0.02, "Failures and no-call rows retained; caller deadline unmeasured", transform=d.transAxes, fontsize=5.9, color=GREY)
    else:
        unavailable(d, "No recorded complete conditional costs")
    return export(fig, out, "p0_recovery_and_cost_4panels"), {"finite_state_distribution_count": distribution_count,
            "unknown_states_not_plotted_as_zero": unknown_states, "matched_finite_pair_count": len(valid_pairs),
            "cost_rows_with_finite_values": sum(len(series) for series in cost_series)}


def scope_diagnostic(graph_rows, request_rows, validation, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.32), layout="constrained")
    fig.suptitle("P0: actual capped domains, fixed-base blocking and resource structure", fontsize=10)
    a, b, c, d = axes.ravel()
    configs = config_order(graph_rows)
    labels = [f"{view[1:]}\n{gap}" for view, gap in configs]
    panel(a, "(a) Scope clipping and fixed-base exclusion", "Station count / ground gap (s)", "Actual scope mean fraction (%)")
    x = np.arange(len(configs))
    for offset, key, label, color in ((-0.17, "cap_truncated_scope_fraction", "Cap256 truncated scopes", VIEW_COLORS["R8"]),
                                       (0.17, "mean_fixed_base_blocked_fraction", "Candidate sphere blocked by B", VIEW_COLORS["R12"])):
        means = []
        for config_i, (view, gap) in enumerate(configs):
            values = [number(row, key) for row in graph_rows if identity(row.get("graph_id"))[1:] == (view, gap)]
            values = [100 * value for value in values if value is not None]
            means.append(float(np.mean(values)) if values else np.nan)
            for source_i, value in enumerate(values):
                a.scatter(config_i + offset + (source_i - (len(values) - 1) / 2) * 0.045, value,
                          color=INK, s=9, alpha=0.65, marker=SOURCE_MARKERS[source_i % len(SOURCE_MARKERS)], zorder=3)
        a.bar(x + offset, means, width=0.3, color=color, alpha=0.45, label=label, zorder=2)
    a.set_xticks(x, labels); a.set_ylim(0, 105)
    if configs:
        a.legend(loc="upper left", frameon=False, fontsize=6.2)
    else:
        unavailable(a, "No completed scope summaries")
    unique = {}
    for row in request_rows:
        if row.get("scope_sha256"):
            unique.setdefault((row.get("graph_id"), row["scope_sha256"]), row)
    scopes = list(unique.values())
    panel(b, "(b) Coupling inside actual recovery domains", "Actual retained recovery vertices", "Cross-owner / local conflict edges (%)")
    scatter_count = 0
    for view in ("R8", "R12"):
        for gap in (170, 340, 680):
            rows = [row for row in scopes if identity(row.get("graph_id"))[1:] == (view, gap)
                    and number(row, "local_vertices") is not None and number(row, "local_edges") is not None
                    and number(row, "local_cross_owner_edges") is not None]
            if rows:
                scatter_count += len(rows)
                b.scatter([number(row, "local_vertices") for row in rows],
                          [100 * number(row, "local_cross_owner_edges") / max(1, number(row, "local_edges")) for row in rows],
                          s=12, color=VIEW_COLORS[view], marker=GAP_MARKERS[gap], alpha=0.42, edgecolors="none")
    b.axvline(256, color=GREY, linestyle=":", linewidth=0.7); b.set_xlim(left=0); b.set_ylim(0, 105)
    if scatter_count:
        b.legend(handles=[Line2D([0], [0], marker="o", color=VIEW_COLORS[v], linestyle="", label=v, markersize=4) for v in ("R8", "R12")]
                 + [Line2D([0], [0], marker=GAP_MARKERS[g], color=GREY, linestyle="", label=f"g={g}", markersize=4) for g in (170, 340, 680)],
                 loc="upper left", ncol=2, frameon=False, fontsize=6.2)
        b.text(0.99, 0.01, f"{scatter_count} unique scopes; not physical sources", transform=b.transAxes, ha="right", fontsize=5.9, color=GREY)
    else:
        unavailable(b, "Local coupling fields unavailable")
    panel(c, "(c) Full-graph temporal conflict types", "Station count / ground gap (s)", "Within-graph edge share (%)")
    raw_graphs = validation.get("graphs", {}) if validation else {}
    bottom = np.zeros(len(configs))
    valid_type_count = 0
    for key, label in (("ground_only_edges", "Ground only"), ("satellite_only_edges", "Satellite only"), ("double_resource_edges", "Both")):
        means = []
        for view, gap in configs:
            values = [100 * float(raw[key]) / max(1, int(raw["edges"])) for graph_id, raw in raw_graphs.items()
                      if graph_id in {row.get("graph_id") for row in graph_rows} and identity(graph_id)[1:] == (view, gap) and key in raw]
            means.append(float(np.mean(values)) if values else np.nan)
            if key == "double_resource_edges":
                valid_type_count += len(values)
        c.bar(x, means, width=0.66, bottom=bottom, color=RESOURCE_COLORS[key], label=label, zorder=2)
        bottom += np.nan_to_num(means, nan=0)
    c.set_xticks(x, labels); c.set_ylim(0, 110)
    if valid_type_count:
        c.legend(loc="upper right", ncol=3, frameon=False, fontsize=6.0, handlelength=0.8, columnspacing=0.6)
        both = [int(raw.get("double_resource_edges", 0)) for graph_id, raw in raw_graphs.items() if graph_id in {row.get("graph_id") for row in graph_rows}]
        if both and not any(both):
            c.text(0.02, 0.04, "Both-resource edges = 0; no artificial cliques", transform=c.transAxes, fontsize=6.1, color=INK)
    else:
        unavailable(c, "Resource-type validation unavailable\nUnknown is not a zero")
    panel(d, "(d) Candidate count before and after cap", "Base-compatible candidates before cap", "Actual retained recovery vertices")
    domain_count = 0
    maximum = 256.0
    for view in ("R8", "R12"):
        rows = [row for row in scopes if identity(row.get("graph_id"))[1] == view
                and number(row, "eligible_before_cap") is not None and number(row, "local_vertices") is not None]
        if rows:
            domain_count += len(rows)
            before = [number(row, "eligible_before_cap") for row in rows]
            after = [number(row, "local_vertices") for row in rows]
            maximum = max(maximum, max(before))
            d.scatter(before, after, s=12, color=VIEW_COLORS[view], marker="o" if view == "R8" else "s", alpha=0.42, edgecolors="none", label=view)
    d.plot([0, 256], [0, 256], color=GREY, linestyle=":", linewidth=0.8, label="No clipping")
    d.axhline(256, color=GREY, linestyle="--", linewidth=0.7)
    d.set_xlim(0, maximum * 1.04); d.set_ylim(0, 270)
    if domain_count:
        d.legend(loc="upper left", frameon=False, fontsize=6.2)
        d.text(0.99, 0.03, "Cap256 is shared; clipping is not proof of useful lost sets", transform=d.transAxes, ha="right", fontsize=5.7, color=GREY)
    else:
        unavailable(d, "Actual capped-domain fields unavailable")
    return export(fig, out, "p0_scope_and_resource_4panels"), {"unique_scope_count": len(scopes),
            "scopes_with_local_coupling_fields": scatter_count, "scopes_with_cap_fields": domain_count,
            "resource_type_validated_graphs": valid_type_count}


def run(root, analysis_dir, graph_validation):
    out = root / "reports" / "P0_FIGURES"
    out.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_dir / "analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError("Run analyze_p0.py on the selected completed P0 collection before drawing")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8-sig"))
    graph_rows = read_csv(analysis_dir / "graph_acceptance.csv")
    state_rows = read_csv(analysis_dir / "state_summary.csv")
    request_rows = read_csv(analysis_dir / "all_requests.csv")
    budget_pairs = read_csv(analysis_dir / "paired_200_1000ms_results.csv")
    validation = json.loads(graph_validation.read_text(encoding="utf-8-sig")) if graph_validation.exists() else {}
    family = style()
    files1, counts1 = main_diagnostic(graph_rows, state_rows, request_rows, budget_pairs, out)
    files2, counts2 = scope_diagnostic(graph_rows, request_rows, validation, out)
    inputs = [analysis_path] + [analysis_dir / name for name in ("graph_acceptance.csv", "state_summary.csv", "all_requests.csv", "paired_200_1000ms_results.csv")]
    if graph_validation.exists():
        inputs.append(graph_validation)
    manifest = {"status": "DRAWN_FROM_REAL_P0_ANALYSIS", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "physical_source_group_count": analysis.get("physical_source_group_count"),
                "derived_graph_count": len(graph_rows), "font_family": family,
                "all_zero_values_retained": True, "unavailable_values_not_converted_to_zero": True,
                "plot_spread_is_not_a_confidence_interval": True, "caller_deadline_test_performed": False,
                "learning_advantage_established": False, "main_diagnostic_counts": counts1, "scope_diagnostic_counts": counts2,
                "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs if path.exists()],
                "outputs": [{"path": str(path), "sha256": sha256(path)} for path in files1 + files2],
                "generator_sha256": sha256(Path(__file__))}
    (out / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = """# P0 诊断图口径

两张独立四小图均同时导出矢量 PDF 与 PNG，字体、配色、线宽和版心一致。

第一张：G1 按实际物理来源、站数与 gap 分列；30%虚线只是开发投入参考。净增量是每个实际控制状态所有有效有限探针中的最好已观测增量，用于机会诊断，不是学习策略结果。零增量保留；无有效有限结果的状态列为未知，不转成零。200→1000ms 是同scope/warm/seed/repeat真实配对；浅线保留全部配对，粗线为站网中位数，带为四分位范围，不是置信区间。成本是完整条件调用实际返回成本，失败和no-call行保留；native slice不是caller deadline。

第二张：截断与完整base遮挡读取实际作用域统计；耦合与cap散点按graph+scope去重。资源边类型来自graph_validation.json的完整原图，明确与局部scope分开，不把同资源全天成员伪造成团。双资源边为零时标记零，不制造结构；缺失信息明确显示未测。

状态、请求与同源配置图不是额外独立物理来源。各来源图保留原点；不画虚构置信区间或图学习优势。symlog用于同时保留零与不同量级，坐标明确标记。数据、脚本和导出图的SHA256见FIGURE_MANIFEST.json。
"""
    (out / "README_ZH.md").write_text(notes, encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "figures": len(files1 + files2), "out": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--graph-validation", type=Path)
    arguments = parser.parse_args()
    dataset = arguments.dataset_root.resolve()
    run(dataset, arguments.analysis_dir.resolve() if arguments.analysis_dir else dataset / "reports" / "p0_analysis",
        arguments.graph_validation.resolve() if arguments.graph_validation else dataset / "reports" / "graph_validation.json")
