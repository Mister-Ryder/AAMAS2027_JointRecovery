"""Compact, source-backed conditional opportunity panels; no learned results.

Reads the completed original label03 and its existing independent audit.
Only NumPy/Matplotlib and the standard library are used. No fitting, graph
generation, native execution, scientific-model imports, or parameter search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

COLORS = dict(menu="#0072B2", resource="#D55E00", teal="#008A69",
              purple="#6950A1", dark="#233342", gray="#667585")
WORKPOINTS = ("slice0.01", "slice0.05", "slice0.2")
TOPOLOGY_MARKERS = dict(erdos="o", components="s", bipartite="^")
FILES = ("opportunity_workpoints", "opportunity_scenarios", "opportunity_slack")
CAPTION = r"""\textbf{Executed conditional recovery opportunity, not learned-policy performance.}
All 72 training and 24 validation graphs retain their original weight, with
equal menu/resource mass and a fixed mean of two actual snapshots.
(a) Best single admitted extra per native workpoint versus graph-averaged
available-request cost; solid/dotted curves divide by the original incumbent
or the current snapshot's known feasible best, respectively. The latter is a
scale diagnostic, not a change to the primary metric. No oracle-policy cost
or matched-total-budget speedup is implied.
(b) All 72 menu graphs by topology and density; coupling is fixed at 0.04,
so differences are descriptive. (c) Graph-level finite native extra versus
observable clique-partition headroom, both normalized by current known best;
each snapshot takes maxima across available alternatives before graph
averaging. The diagonal marks equality, not an optimum certificate.
Axes use a linear neighborhood of zero and logarithmic spacing beyond it.
These cloned single-call outcomes do not bound adaptive multi-call policy
quality or include its complete prediction and caller costs."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clique_upper(nodes, adjacency, weights, supplied):
    """Unscaled observable cover/partition recipe, independently reconstructed.

    Same deterministic weight/degree/ID cover order and largest-factor-first
    disjoint partition as frozen v4_factors_fast / v4_model_fast. This is
    mathematical observable headroom, not a stored neural tensor or optimum.
    """
    local = {v: i for i, v in enumerate(nodes)}
    neighbors = [frozenset(local[u] for u in adjacency[v] if u in local) for v in nodes]
    order = sorted(range(len(nodes)), key=lambda i: (-float(weights[nodes[i]]),
                                                    -len(neighbors[i]), i))
    rank = {v: i for i, v in enumerate(order)}
    ranked = [sum(1 << rank[v] for v in row) for row in neighbors]
    masks = [sum(1 << v for v in row) for row in neighbors]
    covered = [0] * len(nodes)
    factors = set()
    for seed in order:
        allowed = ranked[seed]
        if not allowed:
            continue
        clique = [seed]
        while allowed:
            bit = allowed & -allowed
            vertex = order[bit.bit_length() - 1]
            clique.append(vertex)
            allowed &= ranked[vertex]
        factor = tuple(sorted(clique))
        if factor in factors:
            continue
        factors.add(factor)
        mask = sum(1 << v for v in factor)
        for vertex in factor:
            covered[vertex] |= mask ^ (1 << vertex)
    for vertex, mask in enumerate(masks):
        uncovered = mask & ~covered[vertex] & ~((1 << (vertex + 1)) - 1)
        while uncovered:
            bit = uncovered & -uncovered
            factors.add((vertex, bit.bit_length() - 1))
            uncovered ^= bit
    for raw in supplied:
        factor = tuple(sorted(local[v] for v in raw if v in local))
        if len(factor) >= 2:
            factors.add(factor)
    left = set(range(len(nodes)))
    upper = 0.0
    for factor in sorted(factors, key=lambda f: (-len(f), f)):
        if not all(v in neighbors[u] for i, u in enumerate(factor) for v in factor[i + 1:]):
            raise ValueError("Observable factor must be a genuine original conflict clique")
        block = left.intersection(factor)
        if block:
            upper += max(float(weights[nodes[v]]) for v in block)
            left.difference_update(block)
    return upper + math.fsum(float(weights[nodes[v]]) for v in sorted(left))


def load_data(collection, audit_path):
    audit = read(audit_path)
    if audit["status"] != "PASS_complete96_bound_files_memberships_conditional_opportunity_not_learning":
        raise ValueError("Use the existing completed96 audit, not partial labels")
    records = audit["graph_records"]
    if len(records) != 96 or {g["case_id"] for g in records} != {f"case_{i:03d}" for i in range(96)}:
        raise ValueError("Original complete96 graph coverage required")
    graphs = []
    status_counts = {}
    for rec in records:
        spec = rec["spec"]
        graph = dict(case_id=rec["case_id"], spec=spec, weight=rec["graph_weight"], states=[])
        if rec["declared_failed"]:
            graph["states"] = [dict(global_extra=0., current_extra=0., slack=None, native_extra=0.,
                                    wp={k: dict(primary=0., current=0., cost=0.) for k in WORKPOINTS})] * 2
            graph.update(global_extra=0., current_extra=0., slack=None, native_extra=0.)
            graphs.append(graph)
            continue
        folder = collection / "cases" / rec["case_id"]
        labels = read(folder / "labels.json")
        with np.load(folder / "observable.npz", allow_pickle=False) as npz:
            weights, edges = npz["weights"], npz["edges"]
            flat, ptr = npz["cliques_flat"], npz["cliques_ptr"]
        adjacency = [set() for _ in weights]
        for u, v in edges:
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
        supplied = [flat[ptr[i]:ptr[i + 1]].tolist() for i in range(len(ptr) - 1)]
        upper_cache = {}
        for group in labels["groups"]:
            state = group["state"]
            original = max(1., state["initial_value"])
            current = max(1., state["initial_value"] + state["best_gain"])
            slacks, native = [], []
            wp = {}
            for workpoint in WORKPOINTS:
                rows = [r for r in group["alternatives"] if r["workpoint"] == workpoint]
                extra = max((max(0., r["admitted_gain"] - state["best_gain"]) for r in rows), default=0.)
                available = [r for r in rows if r["available"]]
                cost = float(np.mean([r["elapsed_seconds"] or 0. for r in available])) if available else 0.
                wp[workpoint] = dict(primary=extra / original, current=extra / current, cost=cost)
                for row in available:
                    status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
                    nodes = tuple(row["scope_nodes"])
                    if nodes not in upper_cache:
                        upper_cache[nodes] = clique_upper(nodes, adjacency, weights, supplied)
                    upper, lower = upper_cache[nodes], row["previous_warm_value"]
                    if lower > upper + 1e-7 * max(1., abs(upper)):
                        raise ValueError("Actual feasible warm exceeds observable bound")
                    slacks.append(max(0., upper - lower) / current)
                    delta = 0.
                    if row["raw_supervision_valid"] and row["on_time"]:
                        if row["recovery_value"] > upper + 1e-7 * max(1., abs(upper)):
                            raise ValueError("Actual feasible return exceeds observable bound")
                        delta = max(0., row["recovery_value"] - lower)
                    native.append(delta / current)
            graph["states"].append(dict(global_extra=max(v["primary"] for v in wp.values()),
                                         current_extra=max(v["current"] for v in wp.values()),
                                         slack=max(slacks, default=0.), native_extra=max(native, default=0.), wp=wp))
        if len(graph["states"]) != 2:
            raise ValueError("Keep both original actual snapshots")
        for name in ("global_extra", "current_extra", "slack", "native_extra"):
            graph[name] = float(np.mean([s[name] for s in graph["states"]]))
        if graph["native_extra"] > graph["slack"] + 1e-10:
            raise ValueError("Finite feasible extra must not exceed observable headroom")
        graphs.append(graph)
    # One existing-audit summary cross-check. No new collection audit or gate.
    for split in ("training", "validation"):
        rows = [g for g in graphs if g["spec"]["split"] == split]
        actual = sum(g["weight"] * g["global_extra"] for g in rows)
        expected = audit["statistics"]["splits"][split]["mean_graph_state_best_admitted_extra_relative"]
        if not math.isclose(actual, expected, abs_tol=1e-11, rel_tol=1e-10):
            raise ValueError("Graph/domain-weighted plot quantity differs from completed audit")
    return graphs, audit, status_counts


def aggregate(graphs, domain, split, workpoint, field):
    rows = [g for g in graphs if g["spec"]["domain"] == domain and g["spec"]["split"] == split]
    mass = sum(g["weight"] for g in rows)
    return sum(g["weight"] * np.mean([s["wp"][workpoint][field] for s in g["states"]]) for g in rows) / mass


def setup_style():
    font_manager.findfont("Arial", fallback_to_default=False)
    matplotlib.rcParams.update({"font.family": "Arial", "font.size": 7.4,
        "axes.labelsize": 7.4, "axes.titlesize": 8., "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2, "legend.fontsize": 7.4, "text.color": COLORS["dark"],
        "axes.labelcolor": COLORS["dark"], "axes.edgecolor": COLORS["gray"],
        "xtick.color": COLORS["dark"], "ytick.color": COLORS["dark"],
        "axes.linewidth": .55, "lines.linewidth": .85, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})


def prepare_axis(ax, title):
    ax.set_title(title, loc="left", pad=3, fontweight="bold")
    ax.tick_params(length=2., width=.55, pad=1.8)
    ax.grid(axis="y", color="#E4E8EB", linewidth=.45)
    ax.set_axisbelow(True)


def panel_workpoints(ax, graphs):
    prepare_axis(ax, "(a) Finite workpoints")
    for domain in ("menu", "resource"):
        for split in ("training", "validation"):
            xs = [1000 * aggregate(graphs, domain, split, w, "cost") for w in WORKPOINTS]
            for field, style in (("primary", "-"), ("current", ":")):
                ys = [100 * aggregate(graphs, domain, split, w, field) for w in WORKPOINTS]
                ax.plot(xs, ys, style, color=COLORS[domain], alpha=.8 if split == "validation" else .45,
                        marker="o", markersize=2.6, markerfacecolor=COLORS[domain] if split == "validation" else "white",
                        markeredgewidth=.6, zorder=3 if split == "validation" else 2)
    ax.set(xlim=(0, 222), xlabel="Mean available request cost (ms)", ylabel="Conditional extra (%)")
    ax.set_yscale("symlog", linthresh=.5, linscale=.5)
    ax.set_ylim(-.12, 65)
    ax.yaxis.set_major_locator(FixedLocator([0, 1, 5, 20, 60]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xticks([0, 50, 100, 200])
    ax.annotate("39.98%", xy=(205, 39.9814), xytext=(-5, 5), textcoords="offset points",
                ha="right", color=COLORS["menu"], fontsize=7.2)
    ax.annotate("7.18%", xy=(205, 7.1780), xytext=(-4, 3), textcoords="offset points",
                ha="right", color=COLORS["menu"], fontsize=7.2)
    ax.annotate("1.01%", xy=(193, 1.0062), xytext=(-2, 5), textcoords="offset points",
                ha="right", color=COLORS["resource"], fontsize=7.2)


def panel_scenarios(ax, graphs):
    prepare_axis(ax, "(b) Menu structure")
    offsets = dict(erdos=-.013, components=0., bipartite=.013)
    for g in graphs:
        if g["spec"]["domain"] != "menu":
            continue
        spec = g["spec"]
        jitter = (int(g["case_id"].split("_")[1]) % 7 - 3) * .0012
        x = spec["density"] + offsets[spec["topology"]] + jitter
        filled = spec["split"] == "validation"
        ax.scatter(x, 100 * g["global_extra"], s=10 if filled else 8,
                   marker=TOPOLOGY_MARKERS[spec["topology"]], linewidths=.6,
                   facecolors=COLORS["menu"] if filled else "white",
                   edgecolors=COLORS["menu"], alpha=.85 if filled else .45, zorder=3 if filled else 2)
    ax.set(xlim=(.03, .50), xlabel="Conflict-density setting", ylabel="Extra / original incumbent (%)")
    ax.set_yscale("symlog", linthresh=10, linscale=.7)
    ax.set_ylim(0, 180)
    ax.set_xticks([.08, .20, .45], ["0.08", "0.20", "0.45"])
    ax.yaxis.set_major_locator(FixedLocator([0, 20, 40, 80, 160]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.text(.98, .035, "All 72 menu graphs\ncoupling = 0.04 (fixed)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.2)


def panel_slack(ax, graphs):
    prepare_axis(ax, "(c) Bound vs. actual recovery")
    for g in graphs:
        if g["slack"] is None:
            continue
        filled = g["spec"]["split"] == "validation"
        color = COLORS[g["spec"]["domain"]]
        ax.scatter(100 * g["slack"], 100 * g["native_extra"], s=11 if filled else 8,
                   linewidths=.55, facecolors=color if filled else "white", edgecolors=color,
                   alpha=.85 if filled else .5, zorder=3 if filled else 2)
    maximum = max(100 * g["slack"] for g in graphs if g["slack"] is not None)
    ylim = max(100 * g["native_extra"] for g in graphs) * 1.45
    ax.plot([0, maximum * 1.2], [0, maximum * 1.2], "--", color=COLORS["gray"], linewidth=.65)
    ax.set_xscale("symlog", linthresh=.2, linscale=.5)
    ax.set_yscale("symlog", linthresh=.2, linscale=.5)
    ax.set(xlim=(0, maximum * 1.2), ylim=(0, ylim), xlabel="Upper headroom / current best (%)",
           ylabel="Native extra / current best (%)")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator([0, .5, 2, 10, 50, 200]))
        axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.text(.02, .965, "96 graphs, two states", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.2)


def legend_handles():
    return [Line2D([], [], color=COLORS["menu"], label="Menu"),
            Line2D([], [], color=COLORS["resource"], label="Resource"),
            Line2D([], [], color=COLORS["gray"], linestyle="-", label="Original incumbent"),
            Line2D([], [], color=COLORS["gray"], linestyle=":", label="Current known best"),
            Line2D([], [], color=COLORS["gray"], marker="o", linestyle="", markerfacecolor="white", markersize=3.5, label="Train"),
            Line2D([], [], color=COLORS["gray"], marker="o", linestyle="", markersize=3.5, label="Validation"),
            Line2D([], [], color=COLORS["gray"], marker="o", linestyle="", markersize=3.5, label="ER"),
            Line2D([], [], color=COLORS["gray"], marker="s", linestyle="", markersize=3.5, label="Components"),
            Line2D([], [], color=COLORS["gray"], marker="^", linestyle="", markersize=3.5, label="Bipartite")]


def save(fig, out, stem, description):
    for suffix in ("pdf", "svg", "png"):
        metadata = {"Title": stem, "Creator": "Source-backed conditional opportunity plot"}
        if suffix == "svg":
            metadata["Description"] = description
        elif suffix == "pdf":
            metadata["Subject"] = description
        else:
            metadata["Description"] = description
        fig.savefig(out / (stem + "." + suffix), dpi=420, metadata=metadata)


def write_handoff(out, graphs, audit_path, status_counts):
    source = "labels.json groups/state/alternatives and original observable.npz; existing complete96 audit"
    rows = ["| Graph | Split | Domain | Global/original % | Global/current % | Headroom/current % | Native/current % |",
            "|---|---|---|---:|---:|---:|---:|"]
    for g in graphs:
        values = ["N/A" if g[k] is None else f"{100 * g[k]:.8g}" for k in
                  ("global_extra", "current_extra", "slack", "native_extra")]
        rows.append("| " + " | ".join([g["case_id"], g["spec"]["split"], g["spec"]["domain"], *values]) + " |")
    qa = """# Conditional opportunity panels: source and render QA

## Evidence and use

All original72 training +24 validation graphs and both actual snapshots are retained.
Quality keeps the original graph/domain weights and zero outcomes; queries are not
independent observations. Panel(b) explicitly shows every menu graph (72), not an
outcome-selected subset; resource graphs appear in (a)/(c). No model/partial fit or
confirmation data was opened. No new solver, training, data generation or gate ran.

Primary uses original-incumbent reward. Dotted curves/current-best scatter are
explicit scale/structure diagnostics and do not replace primary. Finite native
extra uses actual returned recovery minus that request's actual W. Headroom uses
the deterministic observable clique-cover/disjoint-partition recipe reconstructed
from original R, edges, weights and supplied resource cliques; it is not a model
output or an optimum label. Each state takes maxima, then the two states average;
scatter points are graphs. No cloned outcomes are added as a feasible policy.

Panel(a) cost is per-graph/state mean over available requests at that workpoint,
including empty-scope/retained-W procedures. It is not cost of evaluating every
candidate or full policy/caller runtime. Train and validation remain separate.
The existing-audit equal-domain means were checked once at plot export. No
additional dataset/source audit, fixture suite or scientific-run repetition.

## Visual production

Arial; 7.4pt labels/7.2pt ticks and factual annotations; native panels2.2×1.75in.
Colors: menu#0072B2 / resource#D55E00; dark#233342 / gray#667585.
One shared legend; fill distinguishes split, marker distinguishes menu topology,
solid/dotted distinguishes the two workpoint denominators. White backgrounds,
quiet grids, actual zeros, linear neighborhoods of zero/symlog beyond. No
interpolation, invented confidence band, significance star, or causal annotation.
Separate PDF/SVG/420dpiPNG; the supplied float preserves approx native width.
Preview includes all three panels and the same shared legend. Render QA pending.

## Source pointers

Audit: %s
Per graph: bulk_03/out/cases/<Graph>/labels.json groups[0/1] plus observable.npz.
Graph/spec/domain and fixed weights: audit.graph_records. Global uses
max(admitted_gain−state.best_gain,0). Original denom=max(1,state.initial_value);
current denom=max(1,state.initial_value+state.best_gain). Cost=elapsed_seconds
on available records. Native uses recovery_value−previous_warm_value only for
raw_supervision_valid and on_time. Workpoints retain .01/.05/.2 seconds.
Backend status totals (all available alternative calls): %s

## Graph marks (all96)

%s
""" % (audit_path.as_posix(), json.dumps(status_counts, sort_keys=True), "\n".join(rows))
    (out / "opportunity_QA.md").write_text(qa, encoding="utf-8")
    tex = "\\begin{figure*}[t]\n\\centering\n"
    tex += "\\includegraphics[width=.326\\textwidth]{v4_figures/residual_results/" + FILES[0] + ".pdf}\\hfill\n"
    tex += "\\includegraphics[width=.326\\textwidth]{v4_figures/residual_results/" + FILES[1] + ".pdf}\\hfill\n"
    tex += "\\includegraphics[width=.326\\textwidth]{v4_figures/residual_results/" + FILES[2] + ".pdf}\n"
    tex += "\\includegraphics[width=.97\\textwidth]{v4_figures/residual_results/opportunity_legend.pdf}\n"
    tex += "\\caption{" + CAPTION + "}\n\\label{fig:conditional-recovery-opportunity}\n\\end{figure*}\n"
    (out / "figure_float.tex").write_text(tex, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graphs, audit, status_counts = load_data(args.collection.resolve(), args.audit.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()
    description = json.dumps(dict(audit=args.audit.as_posix(), collection=args.collection.as_posix(), panels=FILES,
        audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(), graphs=len(graphs),
        graph_weighted=True, states=2, learned_results=False), sort_keys=True)
    panels = (panel_workpoints, panel_scenarios, panel_slack)
    for stem, draw in zip(FILES, panels):
        fig, ax = plt.subplots(figsize=(2.2, 1.75))
        fig.subplots_adjust(left=.22, right=.985, bottom=.23, top=.85)
        draw(ax, graphs)
        save(fig, args.out, stem, description)
        plt.close(fig)
    legend = plt.figure(figsize=(6.6, .43))
    legend.legend(handles=legend_handles(), loc="center", frameon=False, ncol=5,
                  handlelength=1.45, handletextpad=.4, columnspacing=1.)
    save(legend, args.out, "opportunity_legend", description)
    plt.close(legend)
    preview = plt.figure(figsize=(6.85, 2.23))
    for i, draw in enumerate(panels):
        ax = preview.add_axes([.066 + .33 * i, .32, .245, .485])
        draw(ax, graphs)
    preview.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(.5, .005),
                   ncol=5, frameon=False, handlelength=1.45, handletextpad=.4, columnspacing=1.)
    save(preview, args.out, "opportunity_preview", description)
    plt.close(preview)
    write_handoff(args.out, graphs, args.audit, status_counts)
    print(json.dumps(dict(status="exported_complete96_conditional_not_learning", graphs=len(graphs),
        panels=3, outputs=str(args.out), validation_opportunity=audit["statistics"]["splits"]["validation"]["mean_graph_state_best_admitted_extra_relative"],
        slack_ranges={d: [min(g["slack"] for g in graphs if g["spec"]["domain"] == d and g["slack"] is not None),
                          max(g["slack"] for g in graphs if g["spec"]["domain"] == d and g["slack"] is not None)] for d in ("menu", "resource")}), indent=2))


if __name__ == "__main__":
    main()
