"""Read real P1 fit/replay/caller receipts and make an experiment report.

No training, native execution, label rewriting or favourable-scenario selection
is performed. Missing stages/runs remain WAITING. Cold-clone fixed-call replay,
pooled-workpoint replay and paid, actual deadline execution stay separate.
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
from jointrecovery_method_names import annotate_method_fields, method_display, mapping_markdown, METHOD_MAPPING

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

POLICIES = ("Capacity-seed17", "Capacity-seed29", "CheapSummary-seed17", "CheapSummary-seed29", "P1", "Greedy")
FAMILIES = ("Capacity", "CheapSummary", "P1", "Greedy")
COLORS = {"Capacity": "#2F6B9A", "CheapSummary": "#CA8C28", "P1": "#687B64", "Greedy": "#39434D"}
MARKERS = {"Capacity": "o", "CheapSummary": "s", "P1": "^", "Greedy": "D"}
PHASE_ORDER = {"DEVELOPMENT": 0, "CALIBRATION": 0, "VALIDATION": 1, "TEST": 2, "UNKNOWN": -1}
EXPECTED_SOURCES = {"DEVELOPMENT": {"r004", "r005"}, "CALIBRATION": {"r004", "r005"},
                    "VALIDATION": {"r006", "r007"}, "TEST": {"r008", "r009", "r010", "r011"}}
INK, GREY, GRID = "#26313B", "#7F8991", "#E0E5E9"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical_policy(value):
    return str(value).replace("ResidualCapacity", "Capacity").replace("ResidualCheapSummary", "CheapSummary")


def family(value):
    return canonical_policy(value).split("-seed")[0]


def seed(value):
    match = re.search(r"-seed(\d+)$", str(value))
    return int(match.group(1)) if match else None


def source_name(value):
    match = re.search(r"r(\d+)", str(value))
    return "r%03d" % int(match.group(1)) if match else "r%03d" % int(value)


def phase_name(sources, calibration=False):
    sources = {source_name(s) for s in sources}
    if calibration:
        return "CALIBRATION"
    if sources and sources <= EXPECTED_SOURCES["DEVELOPMENT"]:
        return "DEVELOPMENT"
    if sources and sources <= EXPECTED_SOURCES["VALIDATION"]:
        return "VALIDATION"
    if sources and sources <= EXPECTED_SOURCES["TEST"]:
        return "TEST"
    return "UNKNOWN"


def write_csv(path, rows):
    if not rows:
        path.write_text("status\nWAITING_FOR_ACTUAL_RESULTS\n", encoding="utf-8")
        return
    rows = [annotate_method_fields(row) for row in rows]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def hierarchical(rows, metric, expected_sources=None, six_graphs=True):
    """Equal states within graph, equal graphs within source, equal sources.

    Missing numeric values invalidate the corresponding mean. They are never
    silently removed or replaced by zero. Driver-declared failed delivery zeros
    are already explicit actual outcomes in the input, not inferred here.
    """
    expected = sorted(expected_sources or {r["source"] for r in rows})
    source_values = []
    for source in expected:
        selected = [r for r in rows if r["source"] == source]
        graphs = sorted({r["graph_id"] for r in selected})
        graph_values = []
        missing = 0
        for graph in graphs:
            values = [r.get(metric) for r in selected if r["graph_id"] == graph]
            valid = [float(v) for v in values if finite(v)]
            missing += len(values) - len(valid)
            graph_values.append(float(np.mean(valid)) if valid and len(valid) == len(values) else None)
        complete = bool(graphs and (not six_graphs or len(graphs) == 6) and missing == 0
                        and all(v is not None for v in graph_values))
        source_values.append({"source": source, "graph_count": len(graphs), "record_count": len(selected),
                              "missing_numeric_rows": missing, "mean": float(np.mean(graph_values)) if complete else None})
    overall = float(np.mean([v["mean"] for v in source_values])) if source_values and all(v["mean"] is not None for v in source_values) else None
    return overall, source_values


def fit_inputs(fit_root, inputs):
    rows, history, status = [], [], "WAITING_FOR_FIT"
    if fit_root is None or not fit_root.exists():
        return rows, history, {"status": status, "fit_root": str(fit_root) if fit_root else None}
    protocol_path, completion_path = fit_root / "protocol.json", fit_root / "completion.json"
    protocol = load(protocol_path) if protocol_path.exists() else {}
    completion = load(completion_path) if completion_path.exists() else {}
    inputs.extend(p for p in (protocol_path, completion_path) if p.exists())
    expected = {(variant, s) for variant in ("ResidualCapacity", "ResidualCheapSummary") for s in (17, 29)}
    actual = {(v["variant"], v["seed"]) for v in completion.get("fits", [])}
    complete = completion.get("status") == "ACTUAL_P1_FIT_COMPLETE" and actual == expected
    if complete and protocol_path.exists() and completion.get("protocol_sha256") != sha(protocol_path):
        raise ValueError("Fit completion and frozen protocol hashes differ")
    if complete:
        status = "ACTUAL_FOUR_FITS_COMPLETE"
    by_fit = {(v["variant"], v["seed"]): v for v in completion.get("fits", [])}
    for variant, s in sorted(expected):
        model_dir = fit_root / f"{variant}-seed{s}"
        receipt = by_fit.get((variant, s))
        history_path = model_dir / "history.json"
        records = load(history_path) if history_path.exists() else []
        if history_path.exists():
            inputs.append(history_path)
        checkpoint = model_dir / "final.pt"
        rows.append({"policy": canonical_policy(f"{variant}-seed{s}"), "status": "ACTUAL_FIT_COMPLETE" if receipt else "WAITING",
                     "final_epoch": records[-1].get("epoch") if records else None,
                     "checkpoint_path": str(checkpoint), "checkpoint_in_local_mirror": checkpoint.exists(),
                     "checkpoint_recorded_sha256": receipt.get("sha256") if receipt else None,
                     "parameter_counts": receipt.get("parameter_counts") if receipt else None,
                     "fit_sources": protocol.get("fit_sources"), "development_sources": protocol.get("development_train_family_holdout"),
                     "test_sources_used_during_fit": completion.get("P2_validation_or_test_used") if complete else None})
        for record in records:
            dev = record.get("development", {})
            replay_rows = [dict(r, source=source_name(r["source"])) for r in dev.get("fixed_call_rows", [])
                           if r.get("request_menu") == "native_200ms" and r.get("fixed_call_limit") == 1]
            regret, source_values = hierarchical(replay_rows, "conditional_regret_seconds", {"r004", "r005"}) if replay_rows else (None, [])
            history.append({"policy": canonical_policy(f"{variant}-seed{s}"), "epoch": record.get("epoch"),
                            "objective_specific_development_loss": dev.get("mean_state_loss"),
                            "matched_200ms_one_call_development_regret_seconds": regret,
                            "regret_source_values": source_values, "optimizer_steps": record.get("optimizer_steps"),
                            "train_state_components": record.get("train_mean_state_components"),
                            "parameter_l2_change": record.get("parameter_l2_change")})
    return rows, history, {"status": status, "fit_root": str(fit_root), "protocol_sha256": sha(protocol_path) if protocol_path.exists() else None,
                           "fit_sources": protocol.get("fit_sources"), "development_sources": protocol.get("development_train_family_holdout")}


def allocation_inputs(roots, fit_status, inputs):
    all_rows, summaries, status_rows = [], [], []
    seen = set()
    for out in roots:
        summary_path, result_path = out / "summary.json", out / "fixed_call_results.json"
        if not summary_path.exists() or not result_path.exists():
            status_rows.append({"path": str(out), "status": "WAITING_FOR_FIXED_CALL_RESULTS"})
            continue
        summary = load(summary_path)
        if summary.get("status") != "FIXED_CALL_COLD_CLONE_LABEL_REPLAY_COMPLETE":
            status_rows.append({"path": str(out), "status": "WAITING_FOR_COMPLETED_FIXED_CALL_REPLAY"})
            continue
        if fit_status.get("protocol_sha256") and summary.get("fit_protocol_sha256") != fit_status["protocol_sha256"]:
            raise ValueError("Fixed-call allocation does not bind the selected fit protocol")
        expected = {source_name(s) for s in summary["sources"]}
        phase = phase_name(expected)
        records = load(result_path)
        inputs.extend([summary_path, result_path])
        for record in records:
            row = dict(record, policy=canonical_policy(record["policy"]), source=source_name(record["source"]),
                       phase=phase, allocation_output=str(out), actual_online_timing_evidence=False,
                       menu_semantics="POOLED_WORKPOINTS_UNMATCHED_NATIVE_COST" if record["request_menu"] == "pooled_workpoints" else "MATCHED_SINGLE_NATIVE_WORKPOINT")
            key = (row["example"], row["policy"], row["request_menu"], row["fixed_call_limit"])
            if key in seen:
                raise ValueError("Duplicate fixed-call examples across selected output directories")
            seen.add(key); all_rows.append(row)
        status_rows.append({"path": str(out), "status": "ACTUAL_COLD_CLONE_REPLAY_COMPLETE", "phase": phase,
                            "physical_sources": sorted(expected), "rows": len(records)})
    for phase in sorted({r["phase"] for r in all_rows}, key=lambda p: PHASE_ORDER.get(p, -1)):
        selected_phase = [r for r in all_rows if r["phase"] == phase]
        expected = {r["source"] for r in selected_phase}
        for menu in sorted({r["request_menu"] for r in selected_phase}):
            for calls in (1, 2, 4):
                for policy in POLICIES:
                    rows = [r for r in selected_phase if r["policy"] == policy and r["request_menu"] == menu and r["fixed_call_limit"] == calls]
                    gain, source_values = hierarchical(rows, "beyond_snapshot_gain_seconds", expected)
                    regret, _ = hierarchical(rows, "conditional_regret_seconds", expected)
                    actual_calls, _ = hierarchical(rows, "actually_selected_unique_requests", expected)
                    summaries.append({"phase": phase, "policy": policy, "request_menu": menu, "fixed_call_limit": calls,
                                      "status": "ACTUAL" if gain is not None and regret is not None else "WAITING_INCOMPLETE_GROUP",
                                      "physical_source_count": len(expected), "phase_source_coverage_complete": expected == EXPECTED_SOURCES.get(phase, set()),
                                      "source_equal_gain_seconds": gain, "source_equal_conditional_regret_seconds": regret,
                                      "source_equal_selected_call_count": actual_calls, "source_values": source_values,
                                      "not_real_deadline_execution": True,
                                      "menu_semantics": "POOLED_UNMATCHED_COST" if menu == "pooled_workpoints" else "MATCHED_SINGLE_WORKPOINT"})
    return all_rows, summaries, status_rows


def completed_call_counts(report):
    events = report.get("events", [])
    observed = sum(event.get("kind") == "candidate" and event.get("source") == "actual_native_request"
                   and event.get("actual_request", {}).get("diagnostics", {}).get("native_called") is True for event in events)
    finished = [event for event in events if event.get("kind") == "finished"]
    if finished:
        calls = finished[-1].get("actual_calls", [])
        count = sum(call.get("diagnostics", {}).get("native_called", call.get("native_called")) is True for call in calls)
        return count, False, observed
    return observed, True, observed


def benchmark_inputs(roots, fit_status, inputs):
    raw_rows, statuses, seen = [], [], set()
    for out in roots:
        protocol_path = out / "driver_protocol.json"
        if not protocol_path.exists():
            statuses.append({"path": str(out), "status": "WAITING_FOR_BENCHMARK_PROTOCOL"})
            continue
        protocol = load(protocol_path)
        inputs.append(protocol_path)
        if fit_status.get("protocol_sha256") and protocol.get("fit", {}).get("fit_protocol_sha256") != fit_status["protocol_sha256"]:
            raise ValueError("Actual benchmark does not bind the selected fit protocol")
        sources = {source_name(s) for s in protocol["sources"]}
        phase = phase_name(sources, calibration=protocol["stage"] == "calibrate")
        summary_path = out / "summary.json"
        if summary_path.exists():
            inputs.append(summary_path)
        summary = load(summary_path) if summary_path.exists() else {}
        complete = summary.get("status") == "ACTUAL_SERIAL_MATRIX_COMPLETE"
        run_rows_path = out / "actual_run_rows.json"
        existing = {row["run_id"]: row for row in load(run_rows_path)} if run_rows_path.exists() else {}
        if run_rows_path.exists():
            inputs.append(run_rows_path)
        completed = 0
        for cell in protocol.get("jobs", []):
            key = (phase, cell["graph_id"], cell["policy_label"], cell["budget_id"])
            if key in seen:
                raise ValueError("Duplicate actual benchmark cells; select one frozen receipt set")
            seen.add(key)
            receipt_path = out / "receipts" / f"{cell['run_id']}.json"
            receipt = load(receipt_path) if receipt_path.exists() else existing.get(cell["run_id"])
            base = {"phase": phase, "policy": canonical_policy(cell["policy_label"]), "source": source_name(cell["source"]),
                    "graph_id": cell["graph_id"], "budget_id": cell["budget_id"], "deadline_seconds": cell["deadline_seconds"],
                    "run_id": cell["run_id"], "benchmark_output": str(out), "benchmark_matrix_complete": complete}
            if receipt is None:
                raw_rows.append(dict(base, status="WAITING_FOR_ACTUAL_RUN_RECEIPT", actual_delivered_gain_seconds=None,
                                     strict_delivered_gain_seconds=None, strict_gain_beyond_paid_prefix_seconds=None,
                                     valid_on_time_return_indicator=None, observed_completed_native_calls=None,
                                     external_caller_observed_return_seconds=None, actual_gain_beyond_paid_shared_prefix_seconds=None))
                continue
            completed += 1
            if receipt_path.exists():
                inputs.append(receipt_path)
            report_path = out / "runs" / cell["run_id"] / "actual_policy.json"
            report = load(report_path) if report_path.exists() else None
            if report is not None:
                inputs.append(report_path)
            usable = receipt.get("report_usable_for_calibration") is True
            actual = receipt.get("actual_gain_seconds")
            prefix_extra = receipt.get("actual_gain_beyond_paid_shared_prefix_seconds")
            strict = receipt.get("strict_delivered_gain_including_failed_runs_seconds")
            on_time = usable and receipt.get("missed_return_sample") is False
            strict_extra = float(prefix_extra) if on_time and finite(prefix_extra) else (0.0 if finite(strict) and strict == 0 else None)
            calls, lower_bound, completed_observed = completed_call_counts(report) if report is not None else (None, None, None)
            raw_rows.append(dict(base, status="ACTUAL_RECEIPT" if usable else "ACTUAL_FAILED_OR_UNKNOWN_RECEIPT",
                failure_or_unknown_outcome=receipt.get("failure_or_unknown_outcome"), report_state=receipt.get("report_state"),
                report_usable=usable, raw_actual_gain_seconds=actual, paid_shared_prefix_gain_seconds=receipt.get("paid_shared_prefix_gain_seconds"),
                actual_gain_beyond_paid_shared_prefix_seconds=prefix_extra, strict_delivered_gain_seconds=strict,
                strict_gain_beyond_paid_prefix_seconds=strict_extra, strict_zero_from_executed_failure=not usable and strict == 0,
                valid_on_time_return_indicator=float(on_time), missed_return_sample=receipt.get("missed_return_sample"),
                external_caller_observed_return_seconds=receipt.get("external_caller_observed_return_seconds"),
                whole_cli_wall_seconds_including_resident_setup=receipt.get("whole_cli_wall_seconds_including_resident_setup"),
                observed_completed_native_calls=calls, native_call_count_is_lower_bound=lower_bound,
                completed_native_candidate_events_observed_by_caller=completed_observed,
                completed_detail_missing_in_local_mirror=report is None, stop_reason=receipt.get("stop_reason"),
                actual_policy_path=str(report_path), receipt_path=str(receipt_path),
                native_workpoints_ms=protocol.get("native_workpoints_ms"), resident_setup_outside_D=True))
        statuses.append({"path": str(out), "phase": phase, "status": "ACTUAL_MATRIX_COMPLETE" if complete and completed == len(protocol.get("jobs", [])) else "WAITING_FOR_COMPLETE_ACTUAL_MATRIX",
                         "expected_cells": len(protocol.get("jobs", [])), "actual_receipts": completed, "physical_sources": sorted(sources),
                         "frozen_budgets": protocol.get("budgets"), "native_workpoints_ms": protocol.get("native_workpoints_ms")})
    aggregate = []
    for phase in sorted({r["phase"] for r in raw_rows}, key=lambda p: PHASE_ORDER.get(p, -1)):
        phase_rows = [r for r in raw_rows if r["phase"] == phase]
        expected = {r["source"] for r in phase_rows}
        for budget in sorted({r["budget_id"] for r in phase_rows}):
            deadlines = {r["deadline_seconds"] for r in phase_rows if r["budget_id"] == budget}
            if len(deadlines) != 1:
                raise ValueError("Same actual phase/budget id has different frozen deadlines")
            for policy in POLICIES:
                rows = [r for r in phase_rows if r["budget_id"] == budget and r["policy"] == policy]
                result = {"phase": phase, "budget_id": budget, "deadline_seconds": next(iter(deadlines)), "policy": policy,
                          "physical_source_count": len(expected), "phase_source_coverage_complete": expected == EXPECTED_SOURCES.get(phase, set()),
                          "native_calls_have_unfinished_lower_bounds": any(r.get("native_call_count_is_lower_bound") is True for r in rows),
                          "failed_or_unknown_runs": sum(r.get("failure_or_unknown_outcome") is True for r in rows),
                          "pending_receipts": sum(r["status"].startswith("WAITING") for r in rows)}
                for metric in ("strict_delivered_gain_seconds", "strict_gain_beyond_paid_prefix_seconds", "valid_on_time_return_indicator",
                               "observed_completed_native_calls", "external_caller_observed_return_seconds",
                               "actual_gain_beyond_paid_shared_prefix_seconds", "paid_shared_prefix_gain_seconds"):
                    result[metric], values = hierarchical(rows, metric, expected)
                    result[f"{metric}_source_values"] = values
                result["status"] = "ACTUAL" if (result["strict_delivered_gain_seconds"] is not None and not result["pending_receipts"]
                                                 and all(row["benchmark_matrix_complete"] for row in rows)) else "WAITING_INCOMPLETE_GROUP"
                aggregate.append(result)
    return raw_rows, aggregate, statuses


def family_values(rows, metric):
    result = []
    for name in FAMILIES:
        selected = [r for r in rows if family(r["policy"]) == name]
        expected = 2 if name in ("Capacity", "CheapSummary") else 1
        values = [float(r[metric]) for r in selected if finite(r.get(metric))]
        result.append({"family": name, "mean": float(np.mean(values)) if len(values) == expected else None,
                       "seed_min": min(values) if len(values) == expected else None, "seed_max": max(values) if len(values) == expected else None,
                       "fit_seed_count": len(values), "seed_range_is_not_confidence_interval": True})
    return result


def plot_style():
    font = font_manager.findfont(font_manager.FontProperties(family=["Arial", "DejaVu Sans"]), fallback_to_default=True)
    name = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.family": name, "font.size": 7.6, "axes.titlesize": 8.5, "axes.labelsize": 7.6,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.titleweight": "semibold",
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GREY, "axes.linewidth": .65,
                         "grid.color": GRID, "grid.linewidth": .5, "lines.linewidth": 1.2, "lines.markersize": 3.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white"})
    return name


def panel(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", pad=5); ax.set_xlabel(xlabel, labelpad=3); ax.set_ylabel(ylabel, labelpad=3)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y"); ax.set_axisbelow(True)


def waiting(ax):
    ax.text(.5, .5, "WAITING\nActual results unavailable", ha="center", va="center", transform=ax.transAxes, color=GREY)
    ax.set_xticks([]); ax.set_yticks([])


def line_families(ax, rows, x_key, metric, percentage=False):
    xs = sorted({r[x_key] for r in rows})
    drawn = False
    for name in FAMILIES:
        values = []
        for x in xs:
            group = [r for r in rows if r[x_key] == x]
            item = next(v for v in family_values(group, metric) if v["family"] == name)
            factor = 100 if percentage else 1
            values.append((x, item["mean"] * factor if item["mean"] is not None else np.nan,
                           item["seed_min"] * factor if item["seed_min"] is not None else np.nan,
                           item["seed_max"] * factor if item["seed_max"] is not None else np.nan))
        if values and any(math.isfinite(v[1]) for v in values):
            x, y, low, high = map(np.asarray, zip(*values))
            ax.plot(x, y, color=COLORS[name], marker=MARKERS[name], linestyle="--" if name == "Greedy" else "-", label=method_display(name, english=True))
            if name in ("Capacity", "CheapSummary"):
                ax.fill_between(x, low, high, color=COLORS[name], alpha=.13)
            drawn = True
    if drawn:
        ax.legend(frameon=False, ncol=1, loc="best", handlelength=1.4, columnspacing=.8, fontsize=6.)
    else:
        waiting(ax)
    return drawn


def export(fig, out, name):
    fig.text(.989, .994, "❋", ha="right", va="top", color=GREY, fontsize=7, fontfamily="DejaVu Sans")
    files = []
    for extension in ("pdf", "png"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=.035); files.append(path)
    plt.close(fig)
    return files


def figures(history, replay, actual, replay_phase, actual_phase, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.4), layout="constrained")
    fig.suptitle(f"JointRecovery static diagnostic: cold-clone replay ({replay_phase or 'WAITING'})", fontsize=10)
    a, b, c, d = axes.ravel()
    for ax, menu, label in ((a, "native_200ms", "(a) Matched 200 ms native workpoint"),
                             (b, "native_1000ms", "(b) Matched 1000 ms native workpoint"),
                             (c, "pooled_workpoints", "(c) Pooled workpoints; unequal native costs")):
        panel(ax, label, "Maximum selected calls", "Extra linked seconds beyond snapshot ↑")
        rows = [r for r in replay if r["phase"] == replay_phase and r["request_menu"] == menu and r["status"] == "ACTUAL"]
        line_families(ax, rows, "fixed_call_limit", "source_equal_gain_seconds")
        if rows:
            ax.set_xticks([1, 2, 4]); ax.set_ylim(bottom=0)
    panel(d, "(d) Development ranking diagnostic", "Training epoch", "200 ms / 1-call conditional regret (s) ↓")
    history_drawn = False
    for policy in POLICIES[:4]:
        rows = sorted([r for r in history if r["policy"] == policy and finite(r.get("matched_200ms_one_call_development_regret_seconds"))], key=lambda r: r["epoch"])
        if rows:
            d.plot([r["epoch"] for r in rows], [r["matched_200ms_one_call_development_regret_seconds"] for r in rows],
                   color=COLORS[family(policy)], linestyle="-" if seed(policy) == 17 else "--", label=method_display(policy, english=True))
            history_drawn = True
    if history_drawn:
        d.legend(frameon=False, fontsize=6.0, ncol=1); d.set_ylim(bottom=0)
    else:
        waiting(d)
    files = export(fig, out, "p1_fixed_call_and_fit_4panels")
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.35), layout="constrained")
    fig.suptitle(f"JointRecovery static max4: paid-deadline diagnostic ({actual_phase or 'WAITING'})", fontsize=10)
    definitions = (("(a) Strict delivered increment", "strict_delivered_gain_seconds", "Strict on-time linked seconds ↑", False),
                   ("(b) Strict gain beyond paid common prefix", "strict_gain_beyond_paid_prefix_seconds", "On-time extra beyond prefix (s) ↑", False),
                   ("(c) Valid actual on-time return", "valid_on_time_return_indicator", "Valid on-time return rate (%) ↑", True),
                   ("(d) Actual completed native-call evidence", "observed_completed_native_calls", "Observed completed calls (count)", False))
    for ax, (title, metric, ylabel, percentage) in zip(axes.ravel(), definitions):
        panel(ax, title, "Frozen complete deadline D (s)", ylabel)
        rows = [r for r in actual if r["phase"] == actual_phase and r["status"] == "ACTUAL"]
        line_families(ax, rows, "deadline_seconds", metric, percentage)
        if rows:
            ax.set_ylim(bottom=0)
            ax.set_xticks(sorted({r["deadline_seconds"] for r in rows}))
            if percentage:
                ax.set_ylim(0, 105)
            if metric == "observed_completed_native_calls":
                ax.text(.02, .02, "Unfinished-worker totals are lower bounds; see table", transform=ax.transAxes, color=GREY, fontsize=5.8)
    files.extend(export(fig, out, "p1_actual_deadline_4panels"))
    return files


def formatted(value, decimals=6):
    return f"{value:.{decimals}f}" if finite(value) else "WAITING"


def write_report(out, fit_rows, fit_status, replay, actual, allocation_status, benchmark_status, replay_phase, actual_phase, manifest):
    text = ["# JointRecovery：静态 max4 与固定调用内部诊断", "", "**本文完整框架方法为 JointRecovery（JR，FullCapacity）；本目录 Capacity 为其静态 max4 内部诊断。P1 是 Independent-replacement+CHILS 对照，不是本文主方法。** 详见 README_METHODS_ZH.md。充裕求解质量主结果另见 reports/JOINTRECOVERY_QUALITY_RESULTS。", "", "本文件是实验报告，不是论文正文。它只读取真实结果，不执行训练、native搜索或样本筛选；尚无结果的阶段明确为WAITING，不补零，也不输出优势结论。", "",
            "三种证据分别报告：固定单workpoint的冷快照标签回放；含不同native成本的pooled回放；重新支付动作、scope、共同warm、推理、搜索、完整验证和caller返回的实际完整D执行。前两种不是完整时间优势证据。", "",
            "按同一物理来源内等图、每图内等实际状态，最后对来源等权。两个拟合种子逐项保留；图中的浅带是两seed的[min,max]范围，不是置信区间。配置图、控制状态及两个seed不是新的物理来源。", "",
            "## 拟合与来源", "", f"状态：{fit_status['status']}。拟合来源r000–r003；开发来源r004–r005；验证r006–r007；测试r008–r011。", "",
            "| 模型 | 状态 | 实际最终epoch | checkpoint已镜像 |", "|---|---|---:|---|"]
    if not fit_rows:
        text.append("| JointRecovery / JR-CheapSummary 两seed（静态诊断） | WAITING | WAITING | WAITING |")
    for row in fit_rows:
        text.append(f"| {method_display(row['policy'])} | {row['status']} | {row['final_epoch'] if row['final_epoch'] is not None else 'WAITING'} | {row['checkpoint_in_local_mirror']} |")
    text.extend(["", "Greedy在当前冻结实现中用q+w(W)对同一native请求菜单排序，仍允许共同native调用；它不是完全不调用native的Greedy-only策略。共同paid prefix收益另列，不能归给学习。P1用q+结构P1代理排序。", "", "## 固定调用回放：单workpoint先看，pooled单列", ""])
    if not replay:
        text.append("WAITING：尚无实际完成的fixed_call_results/summary。")
    for p in sorted({r['phase'] for r in replay}, key=lambda v: PHASE_ORDER.get(v, -1)):
        for menu in sorted({r['request_menu'] for r in replay if r['phase'] == p}, key=lambda v: (v == 'pooled_workpoints', v)):
            text.extend([f"### {p} / {menu}", "", "| 方法 | 1call增秒↑ | 2call增秒↑ | 4call增秒↑ | 1call观察池后悔值↓ | 物理来源数 |", "|---|---:|---:|---:|---:|---:|"])
            for policy in POLICIES:
                rows = [r for r in replay if r['phase'] == p and r['request_menu'] == menu and r['policy'] == policy]
                by_call = {r['fixed_call_limit']: r for r in rows}
                gains = [formatted(by_call.get(k, {}).get('source_equal_gain_seconds')) for k in (1, 2, 4)]
                regret = formatted(by_call.get(1, {}).get('source_equal_conditional_regret_seconds'))
                count = max((r['physical_source_count'] for r in rows), default=0)
                text.append(f"| {method_display(policy)} | {' | '.join(gains)} | {regret} | {count if count else 'WAITING'} |")
            if menu == 'pooled_workpoints':
                text.append("\n该菜单可以选择不同native时长，固定调用数不等于匹配计算成本；此表只描述冷快照观察标签上的选择，不支持实际完整D优势。")
    text.extend(["", "## 实际完整D与caller返回", ""])
    if not actual:
        text.append("WAITING：尚无真实benchmark driver receipt。")
    for p in sorted({r['phase'] for r in actual}, key=lambda v: PHASE_ORDER.get(v, -1)):
        for budget in sorted({r['budget_id'] for r in actual if r['phase'] == p}, key=lambda b: min(r['deadline_seconds'] for r in actual if r['phase'] == p and r['budget_id'] == b)):
            rows = [r for r in actual if r['phase'] == p and r['budget_id'] == budget]
            deadline = rows[0]['deadline_seconds']
            text.extend([f"### {p} / {budget} / D={deadline:.6f}s", "", "| 方法 | strict按时增秒↑ | strict共同prefix之外增秒↑ | 有效按时返回率↑ | 实际已完成调用 | caller成本(s)↓ | 物理来源数 | 执行失败/未知 | 待执行receipt |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for policy in POLICIES:
                row = next((r for r in rows if r['policy'] == policy), {})
                rate = row.get('valid_on_time_return_indicator')
                count = formatted(row.get('observed_completed_native_calls'), 3)
                if row.get('native_calls_have_unfinished_lower_bounds'):
                    count = '≥' + count
                text.append(f"| {method_display(policy)} | {formatted(row.get('strict_delivered_gain_seconds'))} | {formatted(row.get('strict_gain_beyond_paid_prefix_seconds'))} | {formatted(100 * rate, 2) + '%' if finite(rate) else 'WAITING'} | {count} | {formatted(row.get('external_caller_observed_return_seconds'))} | {row.get('physical_source_count', 'WAITING')} | {row.get('failed_or_unknown_runs', 'WAITING')} | {row.get('pending_receipts', 'WAITING')} |")
    text.extend(["", "strict增秒采用driver冻结口径：已实际执行且收据明确失败/未交付的运行有显式strict=0；原始gain、成本缺测仍保留未知。未启动运行或缺运行receipt是WAITING，不填0。有效按时返回率要求有效完整收据且caller实际返回不超过D，执行失败不计成功。",
                 "", "完整caller曲线中的strict共同prefix之外净收益，要求实际按时交付；内核内部已有收益或迟到收益不能冒充caller收益。部署驻留图/模型与原始全图初态是协议声明的setup，整体CLI wall另存，不能重复添加或冒充D内成本。未收到finished时，已观察native完成调用仅是总调用数的下界，表中以≥明确标记。", "",
                 "## 覆盖、等待项与可追溯数据", "", f"回放图显示阶段为{replay_phase or 'WAITING'}，完整D图为{actual_phase or 'WAITING'}；分别按TEST→VALIDATION→DEVELOPMENT顺序选择已完整的实际证据，不依优势挑选。未完成组仍保留CSV与WAITING标签；部分来源覆盖不是完整阶段确认。", ""])
    for item in allocation_status + benchmark_status:
        text.append(f"- {item.get('phase', 'UNKNOWN')} / {item['status']} / {item['path']}")
    if not allocation_status and not benchmark_status:
        text.append("- WAITING：未找到实际P1分配或完整D阶段输出。")
    text.extend(["", "可追溯文件：fit_history.csv、fit_receipts.csv、fixed_call_all_rows.csv、fixed_call_source_equal.csv、actual_deadline_all_receipts.csv、actual_deadline_source_equal.csv、seed_ranges.csv。所有输入和图输出哈希见P1_REPORT_MANIFEST.json。报告没有自动宣称模型优势；实际差异必须结合全部来源、强廉价对照和完整成本解释。", ""])
    (out / "P1_RESULTS_ZH.md").write_text("\n".join(text), encoding="utf-8")
    (out / "README_METHODS_ZH.md").write_text(mapping_markdown("short"), encoding="utf-8")


def discover(root, explicit_fit=False):
    fits, allocations, benchmarks = [], [], []
    execution = root / "execution"
    if execution.exists():
        for path in execution.rglob("completion.json"):
            if load(path).get("status") == "ACTUAL_P1_FIT_COMPLETE":
                fits.append(path.parent)
        for path in execution.rglob("summary.json"):
            status = load(path).get("status")
            if status == "FIXED_CALL_COLD_CLONE_LABEL_REPLAY_COMPLETE":
                allocations.append(path.parent)
        benchmarks = [path.parent for path in execution.rglob("driver_protocol.json")]
    if len(fits) > 1 and not explicit_fit:
        raise ValueError("Multiple sealed P1 fits; select --fit-root explicitly")
    return (fits[0] if fits else None), sorted(set(allocations)), sorted(set(benchmarks))


def run(root, fit_root, allocation_roots, benchmark_roots, plot_phase):
    out = root / "reports" / "P1_RESULTS"
    out.mkdir(parents=True, exist_ok=True)
    inputs = []
    fit_rows, history, fit_status = fit_inputs(fit_root, inputs)
    replay_rows, replay, allocation_status = allocation_inputs(allocation_roots, fit_status, inputs)
    actual_rows, actual, benchmark_status = benchmark_inputs(benchmark_roots, fit_status, inputs)
    replay_phases = {r['phase'] for r in replay if r.get('status') == 'ACTUAL' and r.get('phase_source_coverage_complete')}
    actual_phases = {r['phase'] for r in actual if r.get('status') == 'ACTUAL' and r.get('phase_source_coverage_complete') and r['phase'] != 'CALIBRATION'}
    replay_phase = plot_phase or (max(replay_phases, key=lambda p: PHASE_ORDER.get(p, -1)) if replay_phases else None)
    actual_phase = plot_phase or (max(actual_phases, key=lambda p: PHASE_ORDER.get(p, -1)) if actual_phases else None)
    family_rows = []
    for kind, data, grouping, metrics in (
        ("FIXED_CALL_REPLAY", replay, ("phase", "request_menu", "fixed_call_limit"), ("source_equal_gain_seconds", "source_equal_conditional_regret_seconds")),
        ("ACTUAL_DEADLINE", actual, ("phase", "budget_id", "deadline_seconds"), ("strict_delivered_gain_seconds", "strict_gain_beyond_paid_prefix_seconds", "valid_on_time_return_indicator", "observed_completed_native_calls"))):
        keys = sorted({tuple(row[x] for x in grouping) for row in data})
        for key in keys:
            rows = [r for r in data if tuple(r[x] for x in grouping) == key]
            for metric in metrics:
                family_rows.extend(dict(kind=kind, metric=metric, **dict(zip(grouping, key)), **values) for values in family_values(rows, metric))
    font = plot_style()
    figure_files = figures(history, replay, actual, replay_phase, actual_phase, out)
    has_outcome_results = any(row.get('status') == 'ACTUAL' for row in replay + actual)
    report_status = ("ACTUAL_P1_REPORT" if has_outcome_results else
                     "ACTUAL_FIT_COMPLETE_RESULTS_WAITING" if fit_status['status'] == 'ACTUAL_FOUR_FITS_COMPLETE' else
                     "WAITING_FOR_ACTUAL_P1_RESULTS")
    manifest = {"status": report_status,
                "proposed_method_display": "JointRecovery (JR)", "static_diagnostic_raw_key": "Capacity",
                "report_role": "INTERNAL_STATIC_MAX4_AND_FIXED_CALL_DIAGNOSTIC", "method_display_mapping": METHOD_MAPPING,
                "generated_utc": datetime.now(timezone.utc).isoformat(), "fit": fit_status,
                "allocation_outputs": allocation_status, "actual_benchmarks": benchmark_status,
                "replay_plot_phase": replay_phase, "actual_deadline_plot_phase": actual_phase,
                "font_family": font, "seed_range_is_not_confidence_interval": True, "missing_runs_not_imputed_as_zero": True,
                "driver_explicit_executed_failed_delivery_zero_retained": True, "learning_advantage_automatically_claimed": False,
                "inputs": [{"path": str(path), "sha256": sha(path)} for path in sorted(set(inputs))],
                "figure_outputs": [{"path": str(path), "sha256": sha(path)} for path in figure_files],
                "generator_sha256": sha(Path(__file__))}
    write_csv(out / "fit_receipts.csv", fit_rows); write_csv(out / "fit_history.csv", history)
    write_csv(out / "fixed_call_all_rows.csv", replay_rows); write_csv(out / "fixed_call_source_equal.csv", replay)
    write_csv(out / "actual_deadline_all_receipts.csv", actual_rows); write_csv(out / "actual_deadline_source_equal.csv", actual)
    write_csv(out / "seed_ranges.csv", family_rows)
    (out / "P1_REPORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(out, fit_rows, fit_status, replay, actual, allocation_status, benchmark_status, replay_phase, actual_phase, manifest)
    print(json.dumps({"status": manifest['status'], "fit_status": fit_status['status'], "fixed_replay_rows": len(replay_rows),
                      "declared_run_rows": len(actual_rows), "actual_received_run_rows": sum(not row['status'].startswith('WAITING') for row in actual_rows),
                      "replay_plot_phase": replay_phase, "actual_deadline_plot_phase": actual_phase,
                      "report": str(out / 'P1_RESULTS_ZH.md')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--fit-root', type=Path)
    parser.add_argument('--allocation-out', type=Path, action='append')
    parser.add_argument('--benchmark-out', type=Path, action='append')
    parser.add_argument('--plot-phase', choices=['DEVELOPMENT', 'VALIDATION', 'TEST'])
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    discovered_fit, discovered_allocations, discovered_benchmarks = discover(dataset, explicit_fit=bool(args.fit_root))
    run(dataset, args.fit_root.resolve() if args.fit_root else discovered_fit,
        [path.resolve() for path in args.allocation_out] if args.allocation_out else discovered_allocations,
        [path.resolve() for path in args.benchmark_out] if args.benchmark_out else discovered_benchmarks, args.plot_phase)
