"""Describe frozen P1 cold-clone labels without running any executor or learner.

Each snapshot is parsed once for recorded scalar outcomes and execution identity.
No graph is loaded, no feasibility check is rerun, and no large evidence file is
rehash-verified. Verified fallback/empty-scope results are retained as labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BUDGETS = (10, 50, 200, 1000)
TOLERANCE_SECONDS = 1e-7
PAIRS = tuple(itertools.combinations(BUDGETS, 2))
ADJACENT_PAIRS = ((10, 50), (50, 200), (200, 1000))
COST_FIELDS = (
    "complete_conditional_return_seconds", "preparation_seconds",
    "process_launch_seconds", "native_process_seconds", "parse_seconds",
    "validation_and_rescore_seconds", "extra_validation_and_rescore_seconds",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return dict(count=0, mean=None, median=None, p95=None, minimum=None, maximum=None)
    return dict(count=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), minimum=min(values), maximum=max(values))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, tuple)) else value
                             for key, value in row.items()})


def percentage(count, denominator):
    return count / denominator if denominator else None


def signed_direction(value):
    return 1 if value > TOLERANCE_SECONDS else -1 if value < -TOLERANCE_SECONDS else 0


def recorded_label(row):
    """Match P1's available verified completion/fallback outcome, not its solver-only subset."""
    return (row.get("available_for_allocation") is True and row.get("complete_membership_valid") is True
            and finite(row.get("q_plus_recovery_seconds")) is not None
            and finite(row.get("recovery_value_seconds")) is not None
            and finite(row.get("complete_value_seconds")) is not None)


def collect(root):
    graph_metadata, rows, inputs, snapshot_files = {}, [], [], []
    expected_graphs = sorted(root.glob("JR-DUAL-r*-R*-g*"))
    for directory in expected_graphs:
        summary_path, protocol_path = directory / "summary.json", directory / "protocol.json"
        summary, protocol = read(summary_path), read(protocol_path)
        metadata = summary["graph"]
        match = re.fullmatch(r"JR-DUAL-(r\d+)-(R\d+)-g(\d+)", metadata["graph_id"])
        if match is None:
            raise ValueError("Unexpected graph identity: " + metadata["graph_id"])
        source, station_view, gap = match.group(1), match.group(2), int(match.group(3))
        config = f"{station_view}-g{gap:04d}"
        graph_metadata[metadata["graph_id"]] = dict(
            source=source, source_group=metadata["source_group"], graph_id=metadata["graph_id"],
            station_view=station_view, gap_seconds=gap, config=config, split=metadata["split"],
            role="fit" if int(source[1:]) < 4 else "development_and_calibration", status=summary["status"],
            original_native_workpoint_order_ms=protocol.get("additional_native_budget_order_ms", protocol.get("budgets_ms")),
            states_declared=summary["actual_unique_controller_states"],
            rows_declared=summary["actual_request_rows"], repeats_declared=protocol.get("repeats", 2),
            native_seed=protocol.get("native_seed"), native_binary_sha256=protocol.get("native_binary_sha256"),
        )
        for file in (summary_path, protocol_path):
            content = file.read_bytes()
            inputs.append(dict(path=str(file), bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        for path in sorted(directory.glob("snapshot_*.json")):
            group = read(path)
            snapshot = group["snapshot"]
            best = float(snapshot["best_gain_seconds"])
            original = float(snapshot["original_value_seconds"])
            scoped = {scope["action_index"]: scope for scope in group["scopes"]}
            snapshot_files.append(dict(path=str(path), bytes=path.stat().st_size,
                                       graph_id=metadata["graph_id"], snapshot_id=group["snapshot_id"],
                                       controller_state_identity=group["controller_state_identity"],
                                       rehash_or_feasibility_recheck_performed=False))
            for raw in group["requests"]:
                diagnostics = raw.get("diagnostics", {})
                scope = scoped[raw["action_index"]]
                value = finite(raw.get("q_plus_recovery_seconds"))
                observed = recorded_label(raw)
                extra = max(0., value - best) if observed else None
                warm_hash = signature(sorted(raw.get("actual_warm_members", [])))
                identity = dict(snapshot_sha256=raw.get("snapshot_sha256"),
                                controller_state_identity=group["controller_state_identity"],
                                scope_sha256=raw.get("scope_sha256"), warm_sha256=warm_hash,
                                native_seed=diagnostics.get("seed", protocol.get("native_seed")),
                                native_binary_sha256=diagnostics.get("binary_sha256", protocol.get("native_binary_sha256")),
                                cpu_affinity=diagnostics.get("inherited_cpu_affinity"))
                # Empty/no-call workpoints have no native affinity metadata. The
                # physical frozen context remains identical and is matched alone.
                context_identity = {key: identity[key] for key in
                                    ("snapshot_sha256", "controller_state_identity", "scope_sha256", "warm_sha256")}
                row = dict(**{key: graph_metadata[metadata["graph_id"]][key] for key in
                              ("source", "source_group", "graph_id", "station_view", "gap_seconds", "config", "split", "role")},
                           snapshot_id=group["snapshot_id"], state_kind=group["state_kind"],
                           action_index=int(raw["action_index"]), workpoint_ms=int(raw["workpoint_ms"]), repeat=int(raw["repeat"]),
                           snapshot_best_gain_seconds=best, original_value_seconds=original,
                           q_seconds=finite(raw.get("q_seconds")), L_seconds=finite(raw.get("L_seconds")),
                           U_seconds=finite(raw.get("U_seconds")), P1_recovery_seconds=finite(raw.get("P1_recovery_seconds")),
                           recovery_value_seconds=finite(raw.get("recovery_value_seconds")),
                           q_plus_recovery_seconds=value, complete_value_seconds=finite(raw.get("complete_value_seconds")),
                           local_delta=finite(raw.get("local_delta")), actual_extra_above_snapshot_seconds=extra,
                           actual_extra_global_fraction=extra / original if extra is not None else None,
                           available_for_allocation=raw.get("available_for_allocation"),
                           complete_membership_valid=raw.get("complete_membership_valid"),
                           actual_finite_supervision_valid=raw.get("actual_finite_supervision_valid"),
                           failed=raw.get("failed"), native_called=diagnostics.get("native_called"),
                           executor_status=diagnostics.get("status"),
                           saved_zero_extra_gain_flag=raw.get("zero_extra_gain"),
                           zero_above_snapshot=extra <= TOLERANCE_SECONDS if extra is not None else None,
                           recorded_verified_completion_label=observed,
                           fixed_context_sha256=signature(context_identity),
                           fixed_context=context_identity,
                           native_seed=identity["native_seed"], native_binary_sha256=identity["native_binary_sha256"],
                           cpu_affinity=identity["cpu_affinity"],
                           local_vertices=scope.get("local_vertices"), cap_truncated=scope.get("cap_truncated"),
                           snapshot_path=str(path))
                for field in COST_FIELDS:
                    row[field] = finite(raw.get(field) if field in raw else diagnostics.get(field))
                rows.append(row)
            del group
    keys = [(r["graph_id"], r["snapshot_id"], r["action_index"], r["workpoint_ms"], r["repeat"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate label row in selected collection")
    return graph_metadata, rows, inputs, snapshot_files


def label_means(rows, metadata):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["graph_id"], row["snapshot_id"], row["action_index"], row["workpoint_ms"])].append(row)
    result = []
    for key, repeats in sorted(grouped.items()):
        repeats.sort(key=lambda row: row["repeat"])
        sample = repeats[0]
        all_valid = all(row["recorded_verified_completion_label"] for row in repeats)
        expected = metadata[key[0]]["repeats_declared"]
        same_context = len({row["fixed_context_sha256"] for row in repeats}) == 1
        complete = all_valid and same_context and len(repeats) == expected
        values = [row["q_plus_recovery_seconds"] for row in repeats] if complete else []
        signed = mean(values)
        above = max(0., signed - sample["snapshot_best_gain_seconds"]) if signed is not None else None
        row = {field: sample[field] for field in (
            "source", "source_group", "graph_id", "station_view", "gap_seconds", "config", "split", "role",
            "snapshot_id", "state_kind", "action_index", "workpoint_ms", "snapshot_best_gain_seconds", "original_value_seconds",
            "q_seconds", "L_seconds", "U_seconds", "P1_recovery_seconds", "fixed_context_sha256", "snapshot_path")}
        row.update(expected_repeats=expected, recorded_repeats=len(repeats), verified_completed_repeats=sum(r["recorded_verified_completion_label"] for r in repeats),
                   label_status="AVAILABLE_REPEAT_MEAN" if complete else "UNAVAILABLE_OR_INCOMPLETE_REPEAT_MEAN",
                   repeated_context_matches=same_context, available_repeat_mean=complete,
                   mean_signed_gain_seconds=signed, mean_recovery_value_seconds=mean([r["recovery_value_seconds"] for r in repeats]) if complete else None,
                   incumbent_preserving_repeat_mean_extra_seconds=above,
                   mean_per_repeat_clipped_extra_seconds=mean([r["actual_extra_above_snapshot_seconds"] for r in repeats]) if complete else None,
                   extra_global_fraction=above / sample["original_value_seconds"] if above is not None else None,
                   positive_extra_repeats=sum(r["actual_extra_above_snapshot_seconds"] is not None and r["actual_extra_above_snapshot_seconds"] > TOLERANCE_SECONDS for r in repeats),
                   failed_repeats=sum(r["failed"] is True for r in repeats),
                   native_called_repeats=sum(r["native_called"] is True for r in repeats),
                   native_finite_valid_repeats=sum(r["actual_finite_supervision_valid"] is True and r["failed"] is False for r in repeats),
                   empty_scope_repeats=sum(r["executor_status"] == "empty_scope_no_native_call" for r in repeats),
                   already_spent_repeats=sum(r["executor_status"] == "already_spent_in_actual_history" for r in repeats),
                   zero_extra_repeat_mean=above <= TOLERANCE_SECONDS if above is not None else None,
                   repeat_signed_gain_range_seconds=max(values) - min(values) if values else None,
                   repeat_ids=[r["repeat"] for r in repeats],
                   repeat_gains=[r["q_plus_recovery_seconds"] for r in repeats],
                   repeat_native_seeds=[r["native_seed"] for r in repeats],
                   repeat_native_binaries=[r["native_binary_sha256"] for r in repeats],
                   repeat_affinities=[r["cpu_affinity"] for r in repeats],
                   repeat_native_called=[r["native_called"] for r in repeats])
        result.append(row)
    return result


def paired_actions(labels):
    grouped = defaultdict(dict)
    for row in labels:
        grouped[(row["graph_id"], row["snapshot_id"], row["action_index"])][row["workpoint_ms"]] = row
    pairs = []
    for key, workpoints in sorted(grouped.items()):
        for low, high in PAIRS:
            a, b = workpoints.get(low), workpoints.get(high)
            exemplar = a or b
            if exemplar is None:
                continue
            reasons = []
            if a is None or b is None:
                reasons.append("MISSING_WORKPOINT")
            elif not a["available_repeat_mean"] or not b["available_repeat_mean"]:
                reasons.append("UNAVAILABLE_OR_INCOMPLETE_LABEL")
            elif a["fixed_context_sha256"] != b["fixed_context_sha256"]:
                reasons.append("DIFFERENT_ACTUAL_SNAPSHOT_SCOPE_OR_WARM")
            else:
                if a["repeat_ids"] != b["repeat_ids"]:
                    reasons.append("DIFFERENT_REPEAT_IDS")
                if a["repeat_native_seeds"] != b["repeat_native_seeds"] or a["repeat_native_binaries"] != b["repeat_native_binaries"]:
                    reasons.append("DIFFERENT_NATIVE_SEED_OR_BINARY")
                # Compare affinity only where both sides actually called native.
                if any(x and y and aa != bb for x, y, aa, bb in zip(a["repeat_native_called"], b["repeat_native_called"], a["repeat_affinities"], b["repeat_affinities"])):
                    reasons.append("DIFFERENT_NATIVE_AFFINITY")
            available = not reasons
            differences = [y - x for x, y in zip(a["repeat_gains"], b["repeat_gains"])] if available else []
            difference = mean(differences)
            extra_difference = (b["incumbent_preserving_repeat_mean_extra_seconds"] - a["incumbent_preserving_repeat_mean_extra_seconds"]) if available else None
            directions = [signed_direction(value) for value in differences]
            pairs.append(dict(**{field: exemplar[field] for field in ("source", "source_group", "graph_id", "config", "station_view", "gap_seconds", "snapshot_id", "state_kind", "action_index")},
                              low_workpoint_ms=low, high_workpoint_ms=high, adjacent_pair=(low, high) in ADJACENT_PAIRS,
                              pairing_status="PAIRED_REPEATED_COMPLETED_LABELS" if available else ";".join(reasons),
                              comparable=available, low_signed_mean_seconds=a["mean_signed_gain_seconds"] if a else None,
                              high_signed_mean_seconds=b["mean_signed_gain_seconds"] if b else None,
                              signed_mean_difference_seconds=difference,
                              incumbent_preserving_extra_difference_seconds=extra_difference,
                              mean_completed_value_changed=abs(difference) > TOLERANCE_SECONDS if difference is not None else None,
                              mean_delivered_extra_changed=abs(extra_difference) > TOLERANCE_SECONDS if extra_difference is not None else None,
                              both_repeats_same_nonzero_direction=(len(set(directions)) == 1 and directions[0] != 0) if directions else None,
                              repeat_differences_seconds=differences,
                              low_failed_repeats=a["failed_repeats"] if a else None, high_failed_repeats=b["failed_repeats"] if b else None,
                              fixed_context_sha256=exemplar["fixed_context_sha256"]))
    return pairs


def paired_states(labels, action_pairs):
    grouped = defaultdict(dict)
    for row in labels:
        grouped[(row["graph_id"], row["snapshot_id"])][(row["action_index"], row["workpoint_ms"])] = row
    paired = defaultdict(list)
    for row in action_pairs:
        paired[(row["graph_id"], row["snapshot_id"], row["low_workpoint_ms"], row["high_workpoint_ms"])].append(row)
    result = []
    for (graph_id, state), all_labels in sorted(grouped.items()):
        for low, high in PAIRS:
            candidate_pairs = paired[(graph_id, state, low, high)]
            valid = [pair for pair in candidate_pairs if pair["comparable"]]
            if not candidate_pairs:
                continue
            exemplar = candidate_pairs[0]
            actions = sorted(pair["action_index"] for pair in valid)
            low_scores = {a: all_labels[(a, low)]["mean_signed_gain_seconds"] for a in actions}
            high_scores = {a: all_labels[(a, high)]["mean_signed_gain_seconds"] for a in actions}
            low_extra = {a: all_labels[(a, low)]["incumbent_preserving_repeat_mean_extra_seconds"] for a in actions}
            high_extra = {a: all_labels[(a, high)]["incumbent_preserving_repeat_mean_extra_seconds"] for a in actions}
            raw_reversals = delivered_reversals = delivered_relation_changes = 0
            for a, b in itertools.combinations(actions, 2):
                dr1, dr2 = signed_direction(low_scores[a] - low_scores[b]), signed_direction(high_scores[a] - high_scores[b])
                dd1, dd2 = signed_direction(low_extra[a] - low_extra[b]), signed_direction(high_extra[a] - high_extra[b])
                raw_reversals += dr1 * dr2 < 0
                delivered_reversals += dd1 * dd2 < 0
                delivered_relation_changes += dd1 != dd2
            low_best = max(low_extra.values()) if low_extra else None
            high_best = max(high_extra.values()) if high_extra else None
            low_top = [a for a in actions if low_best - low_extra[a] <= TOLERANCE_SECONDS] if low_best is not None else []
            high_top = [a for a in actions if high_best - high_extra[a] <= TOLERANCE_SECONDS] if high_best is not None else []
            result.append(dict(**{field: exemplar[field] for field in ("source", "source_group", "graph_id", "config", "station_view", "gap_seconds", "snapshot_id", "state_kind")},
                               low_workpoint_ms=low, high_workpoint_ms=high, adjacent_pair=(low, high) in ADJACENT_PAIRS,
                               declared_action_pairs=len(candidate_pairs), comparable_action_pairs=len(actions),
                               unpaired_action_pairs=len(candidate_pairs) - len(actions),
                               state_has_comparable_labels=bool(actions), state_has_comparable_rankings=len(actions) >= 2,
                               completed_value_changed_actions=sum(p["mean_completed_value_changed"] is True for p in valid),
                               delivered_extra_changed_actions=sum(p["mean_delivered_extra_changed"] is True for p in valid),
                               both_repeats_same_direction_actions=sum(p["both_repeats_same_nonzero_direction"] is True for p in valid),
                               raw_complete_value_strict_reversal_pairs=raw_reversals,
                               incumbent_preserving_strict_reversal_pairs=delivered_reversals,
                               incumbent_preserving_relation_change_pairs=delivered_relation_changes,
                               any_strict_delivered_ranking_reversal=delivered_reversals > 0 if len(actions) >= 2 else None,
                               any_delivered_relation_change=delivered_relation_changes > 0 if len(actions) >= 2 else None,
                               delivered_top_set_changed=set(low_top) != set(high_top) if actions else None,
                               delivered_top_sets_disjoint=not set(low_top).intersection(high_top) if actions else None,
                               low_top_action_indices=low_top, high_top_action_indices=high_top,
                               low_empirical_best_extra_seconds=low_best, high_empirical_best_extra_seconds=high_best,
                               best_extra_difference_seconds=high_best - low_best if actions else None,
                               comparable_action_indices=actions))
    return result


def state_workpoints(labels):
    grouped = defaultdict(list)
    for row in labels:
        grouped[(row["graph_id"], row["snapshot_id"], row["workpoint_ms"])].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        sample = values[0]
        valid = [r for r in values if r["available_repeat_mean"]]
        best = max((r["incumbent_preserving_repeat_mean_extra_seconds"] for r in valid), default=None)
        result.append(dict(**{field: sample[field] for field in ("source", "source_group", "graph_id", "config", "station_view", "gap_seconds", "snapshot_id", "state_kind", "workpoint_ms")},
                           action_workpoints=len(values), available_label_means=len(valid),
                           unavailable_label_means=len(values) - len(valid),
                           empirical_best_extra_seconds=best,
                           extra_opportunity_observed=best > TOLERANCE_SECONDS if best is not None else None,
                           best_extra_global_fraction=best / sample["original_value_seconds"] if best is not None else None,
                           positive_extra_action_means=sum(r["incumbent_preserving_repeat_mean_extra_seconds"] > TOLERANCE_SECONDS for r in valid),
                           original_value_seconds=sample["original_value_seconds"],
                           snapshot_best_gain_seconds=sample["snapshot_best_gain_seconds"],
                           any_recorded_failed_repeat=any(r["failed_repeats"] for r in values)))
    return result


def source_config_tables(rows, labels, states, pairs):
    requests, requests_labels, request_states = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        requests[(row["source"], row["config"], row["workpoint_ms"])].append(row)
    for row in labels:
        requests_labels[(row["source"], row["config"], row["workpoint_ms"])].append(row)
    for row in states:
        request_states[(row["source"], row["config"], row["workpoint_ms"])].append(row)
    cells = []
    for key, observed in sorted(requests.items()):
        l, s = requests_labels[key], request_states[key]
        valid = [r for r in l if r["available_repeat_mean"]]
        states_available = [r for r in s if r["extra_opportunity_observed"] is not None]
        opportunities = sum(r["extra_opportunity_observed"] is True for r in s)
        cell = dict(source=key[0], source_group=observed[0]["source_group"], config=key[1],
                    station_view=observed[0]["station_view"], gap_seconds=observed[0]["gap_seconds"], workpoint_ms=key[2],
                    states=len(s), states_with_available_labels=len(states_available),
                    states_with_extra_opportunity=opportunities,
                    observed_extra_opportunity_fraction=percentage(opportunities, len(states_available)),
                    mean_state_empirical_best_extra_seconds=mean([r["empirical_best_extra_seconds"] for r in s]),
                    median_state_empirical_best_extra_seconds=stats([r["empirical_best_extra_seconds"] for r in s])["median"],
                    maximum_state_empirical_best_extra_seconds=stats([r["empirical_best_extra_seconds"] for r in s])["maximum"],
                    mean_state_best_extra_global_fraction=mean([r["best_extra_global_fraction"] for r in s]),
                    request_repeat_rows=len(observed), request_mean_labels=len(l), available_repeat_mean_labels=len(valid),
                    unavailable_repeat_mean_labels=len(l) - len(valid),
                    verified_complete_repeat_rows=sum(r["recorded_verified_completion_label"] for r in observed),
                    zero_extra_repeat_rows=sum(r["zero_above_snapshot"] is True for r in observed),
                    zero_extra_repeat_mean_labels=sum(r["zero_extra_repeat_mean"] is True for r in valid),
                    failed_repeat_rows=sum(r["failed"] is True for r in observed),
                    unavailable_for_allocation_repeat_rows=sum(r["available_for_allocation"] is not True for r in observed),
                    empty_scope_repeat_rows=sum(r["executor_status"] == "empty_scope_no_native_call" for r in observed),
                    already_spent_repeat_rows=sum(r["executor_status"] == "already_spent_in_actual_history" for r in observed),
                    native_called_repeat_rows=sum(r["native_called"] is True for r in observed),
                    native_finite_valid_repeat_rows=sum(r["actual_finite_supervision_valid"] is True and r["failed"] is False for r in observed))
        for field in COST_FIELDS:
            distribution = stats([r[field] for r in observed])
            for statistic, value in distribution.items():
                cell[field.removesuffix("_seconds") + "_" + statistic + ("" if statistic == "count" else "_seconds")] = value
        native_cost = stats([r["complete_conditional_return_seconds"] for r in observed if r["native_called"] is True])
        for statistic, value in native_cost.items():
            cell["native_called_complete_conditional_" + statistic + ("" if statistic == "count" else "_seconds")] = value
        cells.append(cell)
    source_rows = []
    numeric_fields = ["observed_extra_opportunity_fraction", "mean_state_empirical_best_extra_seconds", "mean_state_best_extra_global_fraction"]
    for source in sorted({c["source"] for c in cells}):
        for budget in BUDGETS:
            selected = [c for c in cells if c["source"] == source and c["workpoint_ms"] == budget]
            if not selected:
                continue
            source_row = dict(source=source, source_group=selected[0]["source_group"], workpoint_ms=budget,
                              configs=len(selected), states=sum(r["states"] for r in selected),
                              failed_repeat_rows=sum(r["failed_repeat_rows"] for r in selected),
                              unavailable_repeat_mean_labels=sum(r["unavailable_repeat_mean_labels"] for r in selected))
            source_row.update({"config_equal_" + field: mean([r[field] for r in selected]) for field in numeric_fields})
            source_rows.append(source_row)
    config_rows = []
    for config in sorted({c["config"] for c in cells}):
        for budget in BUDGETS:
            selected = [c for c in cells if c["config"] == config and c["workpoint_ms"] == budget]
            if not selected:
                continue
            entry = dict(config=config, workpoint_ms=budget, physical_sources=len(selected),
                         states=sum(r["states"] for r in selected))
            entry.update({"source_equal_" + field: mean([r[field] for r in selected]) for field in numeric_fields})
            config_rows.append(entry)
    pair_cells = []
    grouped_pairs = defaultdict(list)
    for row in pairs:
        grouped_pairs[(row["source"], row["config"], row["low_workpoint_ms"], row["high_workpoint_ms"])].append(row)
    for key, values in sorted(grouped_pairs.items()):
        comparable = [r for r in values if r["state_has_comparable_labels"]]
        rankings = [r for r in values if r["state_has_comparable_rankings"]]
        entry = dict(source=key[0], source_group=values[0]["source_group"], config=key[1],
                     low_workpoint_ms=key[2], high_workpoint_ms=key[3], adjacent_pair=(key[2], key[3]) in ADJACENT_PAIRS,
                     states=len(values), comparable_states=len(comparable), comparable_ranking_states=len(rankings),
                     states_with_any_completed_value_change=sum(r["completed_value_changed_actions"] > 0 for r in comparable),
                     states_with_any_delivered_extra_change=sum(r["delivered_extra_changed_actions"] > 0 for r in comparable),
                     states_with_repeat_same_direction_change=sum(r["both_repeats_same_direction_actions"] > 0 for r in comparable),
                     states_with_strict_delivered_ranking_reversal=sum(r["any_strict_delivered_ranking_reversal"] is True for r in rankings),
                     states_with_delivered_relation_change=sum(r["any_delivered_relation_change"] is True for r in rankings),
                     states_with_top_set_change=sum(r["delivered_top_set_changed"] is True for r in comparable),
                     states_with_disjoint_top_sets=sum(r["delivered_top_sets_disjoint"] is True for r in comparable),
                     strict_delivered_reversal_fraction=percentage(sum(r["any_strict_delivered_ranking_reversal"] is True for r in rankings), len(rankings)),
                     completed_value_change_fraction=percentage(sum(r["completed_value_changed_actions"] > 0 for r in comparable), len(comparable)),
                     delivered_extra_change_fraction=percentage(sum(r["delivered_extra_changed_actions"] > 0 for r in comparable), len(comparable)),
                     mean_best_extra_difference_seconds=mean([r["best_extra_difference_seconds"] for r in comparable]),
                     maximum_absolute_best_extra_difference_seconds=stats([abs(r["best_extra_difference_seconds"]) for r in comparable])["maximum"])
        pair_cells.append(entry)
    return cells, source_rows, config_rows, pair_cells


def pretty(value, digits=3):
    return "不可用" if value is None else f"{value:.{digits}f}"


def report(destination, metadata, rows, labels, states, state_pairs, cells, source_rows, pair_cells, inputs, snapshots, collection):
    global_workpoints = []
    for budget in BUDGETS:
        source_values = [r for r in source_rows if r["workpoint_ms"] == budget]
        request_values = [r for r in rows if r["workpoint_ms"] == budget]
        state_values = [r for r in states if r["workpoint_ms"] == budget]
        entry = dict(workpoint_ms=budget, physical_sources=len(source_values), states=len(state_values),
                     states_with_observed_extra_opportunity=sum(r["extra_opportunity_observed"] is True for r in state_values),
                     source_equal_opportunity_fraction=mean([r["config_equal_observed_extra_opportunity_fraction"] for r in source_values]),
                     source_equal_mean_best_extra_seconds=mean([r["config_equal_mean_state_empirical_best_extra_seconds"] for r in source_values]),
                     source_equal_mean_best_extra_global_fraction=mean([r["config_equal_mean_state_best_extra_global_fraction"] for r in source_values]),
                     all_recorded_conditional_cost_seconds=stats([r["complete_conditional_return_seconds"] for r in request_values]),
                     native_called_conditional_cost_seconds=stats([r["complete_conditional_return_seconds"] for r in request_values if r["native_called"] is True]),
                     failed_rows=sum(r["failed"] is True for r in request_values),
                     zero_above_snapshot_rows=sum(r["zero_above_snapshot"] is True for r in request_values),
                     unavailable_for_allocation_rows=sum(r["available_for_allocation"] is not True for r in request_values),
                     statuses=dict(Counter(r["executor_status"] for r in request_values)))
        global_workpoints.append(entry)
    global_pairs = []
    for low, high in PAIRS:
        values = [r for r in state_pairs if r["low_workpoint_ms"] == low and r["high_workpoint_ms"] == high]
        ranks = [r for r in values if r["state_has_comparable_rankings"]]
        cell_values = [r for r in pair_cells if r["low_workpoint_ms"] == low and r["high_workpoint_ms"] == high]
        source_rates = []
        for source in sorted({r["source"] for r in cell_values}):
            own = [r["strict_delivered_reversal_fraction"] for r in cell_values if r["source"] == source]
            source_rates.append(mean(own))
        global_pairs.append(dict(low_workpoint_ms=low, high_workpoint_ms=high, adjacent_pair=(low, high) in ADJACENT_PAIRS,
                                 comparable_states=sum(r["state_has_comparable_labels"] for r in values), comparable_ranking_states=len(ranks),
                                 states_with_completed_value_change=sum(r["completed_value_changed_actions"] > 0 for r in values),
                                 states_with_delivered_extra_change=sum(r["delivered_extra_changed_actions"] > 0 for r in values),
                                 states_with_both_repeats_same_direction_change=sum(r["both_repeats_same_direction_actions"] > 0 for r in values),
                                 states_with_strict_delivered_ranking_reversal=sum(r["any_strict_delivered_ranking_reversal"] is True for r in ranks),
                                 states_with_delivered_relation_change=sum(r["any_delivered_relation_change"] is True for r in ranks),
                                 states_with_top_set_change=sum(r["delivered_top_set_changed"] is True for r in values),
                                 states_with_disjoint_top_sets=sum(r["delivered_top_sets_disjoint"] is True for r in values),
                                 source_equal_strict_delivered_reversal_fraction=mean(source_rates),
                                 mean_paired_state_best_extra_difference_seconds=mean([r["best_extra_difference_seconds"] for r in values])))
    sources = sorted({r["source"] for r in rows})
    state_count = len({(r["graph_id"], r["snapshot_id"]) for r in rows})
    summary = dict(status="ACTUAL_FROZEN_P1_LABEL_DESCRIPTION_NO_LEARNING_EVALUATION", created_utc=datetime.now(timezone.utc).isoformat(),
                   collection=str(collection), physical_sources=len(sources), source_ids=sources, derived_configs=len(metadata),
                   controller_states=state_count, request_repeat_rows=len(rows), action_workpoint_repeat_mean_labels=len(labels),
                   workpoints_ms=list(BUDGETS), expected_repeats=2, tolerance_seconds=TOLERANCE_SECONDS,
                   hierarchy="state means within configuration, configuration means within physical source, sources equal weight",
                   cold_clone_label="Available recorded valid complete schedule return, including verified fallback/empty scope. Mean q+w(T) over both repeats, then preserve snapshot incumbent.",
                   paired_identity="Same actual snapshot/scope/warm, both repeats; same native seed/binary/affinity where native called. No new geometry, native or feasibility check.",
                   zero_flag_disambiguation="Saved zero_extra_gain concerns local recovery; zero_above_snapshot here concerns realizable marginal gain beyond the current held schedule.",
                   actual_full_deadline_performance_measured=False, learning_advantage_established=False,
                   workpoint_results=global_workpoints, pair_results=global_pairs,
                   input_small_files_sha256=inputs, snapshots_read_once=snapshots,
                   collection_input_binding_claimed_not_reverified=True,
                   graph_or_membership_or_large_hash_rechecks_performed=False)
    (destination / "analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# P1 四档 cold-clone 标签特征", "",
             f"本报告覆盖 {len(sources)} 个物理母源（{', '.join(sources)}）、{len(metadata)} 张派生配置图、{state_count} 个实际控制状态、{len(rows):,} 条重复请求记录。每个物理源包含 R8/R12 × 地面间隔 170/340/680 秒六种配置；同源配置和状态不是独立物理样本。r000–r003 是拟合来源，r004/r005 是开发与成本校准来源。未使用 r006 及之后的验证/测试来源。", "",
             "本次只描述已经冻结的返回标签。没有训练模型、再次调用 native、重新构图或重做可行性复核，也没有根据结果选择配置。每个 snapshot 只解析一次，使用它已经记录的完成验证标志和实际 warm 身份；collection 中原有冻结绑定保留但不重新读取大文件验哈希。", "",
             "## 标签和配对口径", "",
             "标签沿用 P1 的可交付 cold-clone 完成口径：请求必须可分配、完整返回成员已验证、实际完成值有限；经过验证的 fallback、空恢复域和零收益均保留。native 的 actual_finite_supervision_valid 另列，不能把它为 false 的空域或保留结果当成标签缺失。已在历史中花过的请求为 unavailable，不作为新请求重复分配。两个重复都完整才取均值，缺失或不可用重复不填零。", "",
             "排序先比较两重复平均后的 q+w(T)。可交付边际收益按 max(0, mean[q+w(T)]−当前 snapshot 已保留收益) 计算，与固定调用回放一致；另外保存 mean[max(0, 单重复增量)]，两者不可混用。snapshot 中已付费的共同三贪心及历史结果都在基准中，报告增量来自它们之外。初始状态、历史状态分别保留。", "",
             "同动作跨档配对固定实际 snapshot、scope、warm、重复编号及 native 种子/二进制；两侧调用 native 时还匹配 CPU affinity。只有这些冻结条件一致的两个完成标签才可比较。数值变化容差是 1e−7 秒，用于消除浮点噪声，不是统计显著性阈值。200/1000 毫秒先完成，50/10 毫秒随后在冻结 cold clone 上追加；所见变化是实际完成结果差异，不承诺工作点与收益严格单调。", "",
             "## 共同贪心和当前 incumbent 之外的机会", "",
             "表中的最佳增量是每个实际状态在该档的已观测候选请求中取最大值，然后状态→配置→物理源等权平均。它是经验性标签上沿，既不是认证最优值，也不是部署策略收益。箭头仅表明数值方向。", "",
             "| native 工作点 | 有可交付额外机会的状态 / 全部状态 ↑ | 源等权机会比例 ↑ | 源等权最佳增量秒 ↑ | 对完整调度价值的平均增量比例 ↑ |", "|---:|---:|---:|---:|---:|"]
    for entry in global_workpoints:
        lines.append(f"| {entry['workpoint_ms']} ms | {entry['states_with_observed_extra_opportunity']} / {entry['states']} | {pretty(None if entry['source_equal_opportunity_fraction'] is None else 100 * entry['source_equal_opportunity_fraction'], 1)}% | {pretty(entry['source_equal_mean_best_extra_seconds'])} | {pretty(None if entry['source_equal_mean_best_extra_global_fraction'] is None else 100 * entry['source_equal_mean_best_extra_global_fraction'], 4)}% |")
    lines += ["", "完整调度百分比的分母为各状态原始完整调度价值；它不是 local_delta，也不能把本表的秒增量称为百分比改善。`source_config_workpoint.csv` 给出全部六源、六配置、四档，不只展示有利子集。", "",
              "## 短时间档是否增加排序或收益变化", "",
              "下表将三个相邻档和 10→1000 毫秒端点比较列出。‘完成值变化’可以发生在仍不超过 incumbent 的请求上；‘额外收益变化’指保留 incumbent 后的边际变化；‘真正反转’要求同一对动作在两档分别严格优于对方。并列最优集合变化单独列出，不能替代严格反转。", "",
              "| 配对工作点 | 可比较状态 | 完成值变化状态 | 额外收益变化状态 | 两重复同方向变化状态 | 可交付排序严格反转状态 | 最优并列集合变化状态 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for entry in global_pairs:
        if not entry["adjacent_pair"] and (entry["low_workpoint_ms"], entry["high_workpoint_ms"]) != (10, 1000):
            continue
        lines.append(f"| {entry['low_workpoint_ms']}→{entry['high_workpoint_ms']} ms | {entry['comparable_states']} | {entry['states_with_completed_value_change']} | {entry['states_with_delivered_extra_change']} | {entry['states_with_both_repeats_same_direction_change']} | {entry['states_with_strict_delivered_ranking_reversal']} / {entry['comparable_ranking_states']} | {entry['states_with_top_set_change']} |")
    endpoint = next((e for e in global_pairs if e["low_workpoint_ms"] == 10 and e["high_workpoint_ms"] == 1000), None)
    if endpoint:
        if endpoint["states_with_strict_delivered_ranking_reversal"]:
            lines += ["", f"10→1000 毫秒确实观察到 {endpoint['states_with_strict_delivered_ranking_reversal']} 个状态存在可交付收益排序的严格反转；短档使标签包含了实际请求选择差异。但最优请求集合只在 {endpoint['states_with_top_set_change']} 个状态改变，其余状态的最佳请求集合保持一致。因而部分排序信号与最优选择普遍稳定同时存在。发生率和差异幅度须结合逐源/逐配置表判断，不能由‘存在反转’推出 Capacity 优于 CheapSummary，或推出真实 D 下的策略优越性。"]
        else:
            lines += ["", "10→1000 毫秒未观察到可交付收益排序的严格反转。即使存在完成值或并列集合变化，也不能据此宣称短档已经产生有力的策略选择信号；仍需实际基线与完整 D 结果判断。"]
    lines += ["", "`paired_state_workpoints.csv` 和 `paired_action_workpoints.csv` 保留全部六种跨档组合及未能配对的原因，包含变差和不变结果；`source_config_workpoint_pairs.csv` 保留每个物理源/配置的发生率。没有用相关状态数量构造置信区间。", "",
              "## 四档实际条件返回成本", "",
              "下表第一组包含所有实际记录的条件调用，包括空域、已花请求和失败；第二组只描述确实调用 native 的记录，便于区分无调用零/低成本混合。它们均是从请求准备、导出、启动、搜索、解析、验证到该条件调用返回的实测成本，不包括此前共同贪心/历史已付费成本，也不是 resident-model/allocator 的完整实际 D 性能。native 工作点超时不能直接算调用者最终交付迟到。", "",
              "| 工作点 | 全部条件返回中位 ms ↓ | 全部条件返回 P95 ms ↓ | 实际 native 调用返回中位 ms ↓ | 实际 native 调用返回 P95 ms ↓ | 失败行 / 不可分配行 |", "|---:|---:|---:|---:|---:|---:|"]
    for entry in global_workpoints:
        all_cost, called = entry["all_recorded_conditional_cost_seconds"], entry["native_called_conditional_cost_seconds"]
        ms = lambda value: pretty(1000 * value if value is not None else None, 2)
        lines.append(f"| {entry['workpoint_ms']} ms | {ms(all_cost['median'])} | {ms(all_cost['p95'])} | {ms(called['median'])} | {ms(called['p95'])} | {entry['failed_rows']} / {entry['unavailable_for_allocation_rows']} |")
    lines += ["", "这些分位数是记录级描述分布，不是跨物理来源的估计区间。准备/启动/native/解析/两次验证等分项均在 `source_config_workpoint.csv`。native 搜索时间不等于条件返回成本，条件返回成本也不等于完整调度决策的最终交付 D。", "",
              "## 零、失败和待回答问题", "",
              f"全部 {len(rows):,} 行中：{sum(r['failed'] is True for r in rows):,} 行记录 failed；{sum(r['zero_above_snapshot'] is True for r in rows):,} 行完整可交付结果未超过当前 incumbent；{sum(r['available_for_allocation'] is not True for r in rows):,} 行不可重新分配；{sum(r['executor_status'] == 'empty_scope_no_native_call' for r in rows):,} 行为空恢复域。以上类别有重叠，不能相加当成总数。原始 `zero_extra_gain` 是局部恢复的标志，与本报告的‘不超过完整当前 incumbent’不是同一指标。", "",
              "训练和实际 D 尚未在本报告中评价。因此没有模型排名、学习优势、在线成本下降或论文录用结论。这里可回答数据是否存在可学习的完成结果差异；它不能替代 Capacity/CheapSummary/P1/Greedy 对照、两种子范围、独立物理源验证及严格完整返回成本评估。", "",
              "## 文件索引", "",
              "- `source_config_workpoint.csv`：源 × 配置 × 四档，包括机会、零/失败/空域以及完整分项成本。",
              "- `source_workpoint_equal_config.csv` / `config_workpoint_equal_source.csv`：明确层级的等权表。",
              "- `request_recorded_scalars.csv`：全部请求标量与已记录验证/失败标志、原 snapshot 路径。",
              "- `action_workpoint_repeat_means.csv`：两重复均值及不可用原因计数。",
              "- `state_workpoint_opportunities.csv`：经验候选上沿；不作为部署策略结果。",
              "- `paired_action_workpoints.csv` / `paired_state_workpoints.csv` / `source_config_workpoint_pairs.csv`：同实际上下文跨档配对。",
              "- `analysis.json`：口径、全局摘要和轻量输入来源绑定。",
              "- `label_characteristics_4panels.pdf/.png`：单张统一风格诊断图，若生成则作为实验报告材料。", ""]
    (destination / "P1_LABEL_CHARACTERISTICS_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def figures(destination, summary, rows, action_pairs, pair_cells, source_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 7.3,
                         "axes.titlesize": 8.1, "axes.labelsize": 7.4, "xtick.labelsize": 6.9,
                         "ytick.labelsize": 6.9, "axes.edgecolor": "#333333", "axes.linewidth": .6,
                         "grid.color": "#E3E5E8", "grid.linewidth": .45, "pdf.fonttype": 42})
    blue, gold, olive, pink = "#276B9B", "#B38422", "#6B7750", "#BA6D86"
    colors = (blue, gold, olive, pink)
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.65), layout="constrained")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", zorder=0)
    a, b, c, d = axes.flat
    for source in summary["source_ids"]:
        own = sorted((r for r in source_rows if r["source"] == source), key=lambda r: r["workpoint_ms"])
        a.plot([r["workpoint_ms"] for r in own], [r["config_equal_mean_state_empirical_best_extra_seconds"] for r in own], color="#AAB4BD", alpha=.65, linewidth=.7, marker="o", markersize=2.3)
    a.plot(BUDGETS, [r["source_equal_mean_best_extra_seconds"] for r in summary["workpoint_results"]], color=blue, linewidth=1.5, marker="o", markersize=3, label="Equal-source mean")
    a.set_xscale("log"); a.set_xticks(BUDGETS, [str(x) for x in BUDGETS]); a.set_ylim(bottom=0)
    a.set_title("(a) Observed opportunity above incumbent", loc="left", fontweight="bold")
    a.set_xlabel("Native workpoint (ms)"); a.set_ylabel("Best extra contact time (s) ↑")
    a.legend(frameon=False, fontsize=6.5, loc="best")
    a.text(.02, .96, "Thin traces: six physical sources", transform=a.transAxes, va="top", fontsize=6.3, color="#555555")

    for pair, color, style in zip(ADJACENT_PAIRS, colors, ("-", "--", ":")):
        values = sorted(r["signed_mean_difference_seconds"] for r in action_pairs if r["comparable"] and (r["low_workpoint_ms"], r["high_workpoint_ms"]) == pair)
        if values:
            b.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=color, linestyle=style, linewidth=1.2, label=f"{pair[0]}→{pair[1]} ms")
    b.axvline(0, color="#444444", linewidth=.65)
    b.set_xscale("symlog", linthresh=.01); b.set_ylim(0, 1); b.yaxis.set_major_formatter(PercentFormatter(1))
    b.set_title("(b) Same-request paired completion changes", loc="left", fontweight="bold")
    b.set_xlabel("Higher − lower workpoint gain (s)"); b.set_ylabel("Recorded paired labels (%)")
    b.legend(frameon=False, fontsize=6.3, loc="lower right")

    plot_pairs = ADJACENT_PAIRS + ((10, 1000),)
    x = np.arange(len(plot_pairs))
    mean_rates = []
    for i, pair in enumerate(plot_pairs):
        rates = []
        for source in summary["source_ids"]:
            own = [r["strict_delivered_reversal_fraction"] for r in pair_cells if r["source"] == source and (r["low_workpoint_ms"], r["high_workpoint_ms"]) == pair]
            rate = mean(own)
            if rate is not None:
                rates.append(rate); c.scatter(i, rate, color="#AAB4BD", s=10, linewidths=.3, edgecolors="#666666", zorder=3)
        mean_rates.append(mean(rates))
    c.plot(x, mean_rates, color=blue, marker="D", markersize=3.2, linewidth=1.3)
    c.set_xticks(x, [f"{l}→{h}" for l, h in plot_pairs]); c.set_ylim(bottom=0)
    c.yaxis.set_major_formatter(PercentFormatter(1)); c.set_title("(c) Strict delivered-ranking reversals", loc="left", fontweight="bold")
    c.set_xlabel("Paired native workpoints (ms)"); c.set_ylabel("States with strict reversal (%)")
    c.text(.02, .96, "Dots: physical sources; line: equal-source mean", transform=c.transAxes, va="top", fontsize=6.1, color="#555555")

    for budget, color, style in zip(BUDGETS, colors, ("-", "--", "-.", ":")):
        values = sorted(1000 * r["complete_conditional_return_seconds"] for r in rows if r["workpoint_ms"] == budget and r["complete_conditional_return_seconds"] is not None)
        if values:
            d.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=color, linestyle=style, linewidth=1.2, label=f"{budget} ms")
    d.set_xscale("symlog", linthresh=1); d.set_ylim(0, 1); d.yaxis.set_major_formatter(PercentFormatter(1))
    d.set_title("(d) Complete conditional return cost", loc="left", fontweight="bold")
    d.set_xlabel("Observed return cost (ms) ↓"); d.set_ylabel("All recorded requests (%)")
    d.legend(frameon=False, fontsize=6.3, loc="lower right", ncol=2)
    fig.suptitle("Frozen cold-clone labels: 6 physical sources · 36 configurations · 432 states", x=.01, ha="left", fontsize=9, fontweight="bold")
    fig.text(.985, .992, "❋", ha="right", va="top", color=blue, fontsize=12)
    for extension in ("pdf", "png"):
        fig.savefig(destination / f"label_characteristics_4panels.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--labels-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", action="store_true", help="Optional standalone diagnostic; the standard light report creates no figures")
    parser.add_argument("--no-figures", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.labels_root or args.dataset_root / "execution" / "p1_all_train_labels"
    destination = args.output or args.dataset_root / "reports" / "P1_LABEL_CHARACTERISTICS"
    destination.mkdir(parents=True, exist_ok=True)
    metadata, rows, inputs, snapshots = collect(root)
    labels = label_means(rows, metadata)
    action_pairs = paired_actions(labels)
    state_pairs = paired_states(labels, action_pairs)
    states = state_workpoints(labels)
    cells, source_rows, config_rows, pair_cells = source_config_tables(rows, labels, states, state_pairs)
    tables = {
        "source_config_workpoint.csv": cells, "source_workpoint_equal_config.csv": source_rows,
        "config_workpoint_equal_source.csv": config_rows, "request_recorded_scalars.csv": rows,
        "action_workpoint_repeat_means.csv": labels, "state_workpoint_opportunities.csv": states,
        "paired_action_workpoints.csv": action_pairs, "paired_state_workpoints.csv": state_pairs,
        "source_config_workpoint_pairs.csv": pair_cells,
    }
    for name, records in tables.items():
        write_csv(destination / name, records)
    summary = report(destination, metadata, rows, labels, states, state_pairs, cells, source_rows, pair_cells, inputs, snapshots, root)
    if args.figures and not args.no_figures:
        figures(destination, summary, rows, action_pairs, pair_cells, source_rows)
    print(json.dumps({"status": summary["status"], "sources": summary["physical_sources"], "graphs": summary["derived_configs"],
                      "states": summary["controller_states"], "rows": len(rows), "report": str(destination / "P1_LABEL_CHARACTERISTICS_ZH.md"),
                      "workpoints": summary["workpoint_results"], "paired_results": summary["pair_results"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
