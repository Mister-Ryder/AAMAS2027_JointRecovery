"""Build immutable, resource-typed MWIS instances from complete STK accesses.

Only complete visibility windows are vertices. Resource memberships are NOT
all-day cliques: conflicts depend on the original interval and switching gap.
The 1 us ticks are an explicit solver serialization, not physical accuracy.
Edge construction retains the exported decimal endpoints (up to 9 places).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Iterable

import numpy as np

NS = 1_000_000_000
US = 1_000_000
INT64_MAX = np.iinfo(np.int64).max
REQUIRED_COLUMNS = {
    "source_group", "geometry_id", "replicate_id", "epoch_utc", "contact_id",
    "pass_id", "satellite_id", "site_id", "antenna_id", "start_rel_seconds",
    "end_rel_seconds", "duration_seconds", "start_utc", "end_utc",
    "boundary_crossing", "access_settings_hash",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decimal_ns(value: str | int | float | Decimal) -> int:
    scaled = Decimal(str(value)) * NS
    if scaled != scaled.to_integral_value():
        raise ValueError("Endpoint exceeds the supported 9 decimal export places")
    result = int(scaled)
    if abs(result) > INT64_MAX:
        raise OverflowError("Decimal endpoint exceeds int64 serialization")
    return result


def microsecond_ticks(value: str | int | float | Decimal) -> int:
    result = int((Decimal(str(value)) * US).to_integral_value(rounding=ROUND_HALF_EVEN))
    if abs(result) > INT64_MAX:
        raise OverflowError("Microsecond serialization exceeds int64")
    return result


def parse_utc(text: str) -> datetime:
    """Accept ISO views or the STK UTCG day-month-year report convention."""
    text = text.strip()
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        clean = text.removesuffix(" UTCG").strip()
        if "." in clean:
            main, fraction = clean.rsplit(".", 1)
            result = datetime.strptime(main, "%d %b %Y %H:%M:%S")
            result = result.replace(microsecond=int((fraction + "000000")[:6]))
        else:
            result = datetime.strptime(clean, "%d %b %Y %H:%M:%S")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def read_contacts(path: Path, scene: dict, stations: list[dict], horizon_seconds: int):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        absent = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f"Missing CSV columns: {sorted(absent)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No full visibility contacts in {path}")
    satellite_ids = [x["satellite_id"] for x in scene["satellites"]]
    satellite_map = {x: i for i, x in enumerate(satellite_ids)}
    station_map = {x["site_id"]: x for x in stations}
    if len(satellite_map) != len(satellite_ids):
        raise ValueError("Duplicate satellite ID in the scene definition")
    ids, physical_keys = set(), set()
    start_ns, end_ns, start_ticks, end_ticks, weights, owners = [], [], [], [], [], []
    duration_discrepancies = []
    endpoint_tick_errors_ns = []
    utc_max_error_seconds = 0.0
    utc_display_digits = []
    utc_display_max_allowed_error_seconds = 0.0
    epoch = datetime.fromisoformat(scene["epoch_utc"].replace("Z", "+00:00"))
    for row in rows:
        if row["contact_id"] in ids:
            raise ValueError(f"Repeated contact_id {row['contact_id']}")
        ids.add(row["contact_id"])
        if row["source_group"] != scene["source_group"]:
            raise ValueError("Mixed source groups in mother contact library")
        if row["replicate_id"] != scene["replicate_id"] or row["geometry_id"] != scene["geometry_id"]:
            raise ValueError("CSV realization/geometry disagrees with parameters")
        if row["epoch_utc"] != scene["epoch_utc"]:
            raise ValueError("CSV epoch disagrees with independent scene epoch")
        if row["satellite_id"] not in satellite_map:
            raise ValueError("Unknown satellite; owner must be an actual satellite")
        if row["site_id"] not in station_map:
            raise ValueError("Unknown ground site")
        if row["antenna_id"] != station_map[row["site_id"]]["antenna_id"]:
            raise ValueError("Antenna mapping differs from the capacity-one station")
        if row["boundary_crossing"].strip().lower() not in {"0", "false", "no", "none"}:
            raise ValueError("Cross-boundary window included in full-contact CSV")
        start = Decimal(row["start_rel_seconds"])
        end = Decimal(row["end_rel_seconds"])
        duration = end - start
        if not (Decimal(0) <= start < end <= Decimal(horizon_seconds)):
            raise ValueError(f"Invalid/full-horizon window: {row['contact_id']}")
        reported_duration = Decimal(row["duration_seconds"])
        discrepancy = abs(reported_duration - duration)
        if discrepancy > Decimal("0.000000002"):
            raise ValueError(f"Duration is not raw endpoint difference: {row['contact_id']}")
        duration_discrepancies.append(float(discrepancy))
        s_ns, e_ns = decimal_ns(start), decimal_ns(end)
        s_us, e_us = microsecond_ticks(start), microsecond_ticks(end)
        if e_us <= s_us:
            raise ValueError("A positive complete window vanishes at microsecond serialization")
        key = (row["satellite_id"], row["antenna_id"], s_ns, e_ns)
        if key in physical_keys:
            raise ValueError("Duplicate physical visibility interval (duplicate reward risk)")
        physical_keys.add(key)
        start_ns.append(s_ns); end_ns.append(e_ns)
        start_ticks.append(s_us); end_ticks.append(e_us)
        endpoint_tick_errors_ns.extend([abs(s_us * 1000 - s_ns), abs(e_us * 1000 - e_ns)])
        weights.append(float(duration)); owners.append(satellite_map[row["satellite_id"]])
        # UTC text is an auxiliary human view; Python datetime preserves microseconds.
        for field, value in (("start_utc", start), ("end_utc", end)):
            moment = parse_utc(row[field])
            fractional = re.search(r":\d{2}\.(\d+)", row[field])
            digits = len(fractional.group(1)) if fractional else 0
            utc_display_digits.append(digits)
            # STK ConvertDate's human UTCG may round to milliseconds even when
            # the authoritative EpSec export carries nine fractional digits.
            # The datetime parser itself truncates beyond microseconds.
            allowed_error = 0.5 * 10.0 ** (-digits) + 0.000001
            utc_display_max_allowed_error_seconds = max(utc_display_max_allowed_error_seconds, allowed_error)
            error = abs((moment - epoch).total_seconds() - float(value))
            utc_max_error_seconds = max(utc_max_error_seconds, error)
            if error > allowed_error:
                raise ValueError("UTC human view and authoritative relative seconds differ beyond display precision")
        if not row["access_settings_hash"]:
            raise ValueError("Access settings hash missing")
    arrays = {
        "start_ns": np.asarray(start_ns, dtype=np.int64),
        "end_ns": np.asarray(end_ns, dtype=np.int64),
        "start_ticks": np.asarray(start_ticks, dtype=np.int64),
        "end_ticks": np.asarray(end_ticks, dtype=np.int64),
        "weights": np.asarray(weights, dtype=np.float64),
        "owner": np.asarray(owners, dtype=np.int32),
    }
    report = {
        "full_contact_count": len(rows), "duplicate_contact_ids": 0,
        "duplicate_physical_intervals": 0, "negative_or_zero_duration": 0,
        "crossing_windows_in_full_contact_csv": 0,
        "duration_endpoint_max_difference_seconds": max(duration_discrepancies),
        "utc_relative_max_difference_seconds": utc_max_error_seconds,
        "utc_human_view_fractional_digits_observed": sorted(set(utc_display_digits)),
        "utc_display_max_allowed_difference_seconds": utc_display_max_allowed_error_seconds,
        "utc_comparison_uses_actual_display_precision": True,
        "relative_decimal_seconds_are_authoritative_for_edges_and_reward": True,
        "microsecond_endpoint_max_rounding_error_ns": max(endpoint_tick_errors_ns),
        "int64_tick_overflow": False,
        "satellite_count_in_parameter_mapping": len(satellite_ids),
        "observed_satellite_count": len(set(owners)),
        "microsecond_tick_note": "Serialization only; not physical accuracy. Edges retain 9-digit original decimal endpoints.",
        "within_same_satellite_site_actual_overlap": count_pair_overlaps(rows, arrays),
    }
    if report["within_same_satellite_site_actual_overlap"]:
        raise ValueError("One STK satellite/site access stream contains overlapping intervals")
    return rows, arrays, report


def count_pair_overlaps(rows: list[dict], arrays: dict) -> int:
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row["satellite_id"], row["site_id"])].append(i)
    count = 0
    for indices in groups.values():
        indices.sort(key=lambda x: (int(arrays["start_ns"][x]), rows[x]["contact_id"]))
        maximum_end = -1
        for i in indices:
            count += int(int(arrays["start_ns"][i]) < maximum_end)
            maximum_end = max(maximum_end, int(arrays["end_ns"][i]))
    return count


def grouped_edges(start_ns: np.ndarray, end_ns: np.ndarray, groups: Iterable[str | int], gap_seconds: int):
    """Exact [s,e) plus gap conflict, including containment and identical starts."""
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    left, right = [], []
    gap_ns = decimal_ns(gap_seconds)
    near_tick_boundary = 0
    equality_compatible = 0
    n = len(start_ns)
    for indices in members.values():
        indices.sort(key=lambda i: (int(start_ns[i]), i))
        positions = np.asarray(indices, dtype=np.int64)
        starts = start_ns[positions]
        for local_i, u in enumerate(indices):
            threshold = int(end_ns[u]) + gap_ns
            stop = int(np.searchsorted(starts, threshold, side="left"))
            lo = local_i + 1
            if stop > lo:
                candidates = positions[lo:stop]
                left.extend(np.minimum(u, candidates).tolist())
                right.extend(np.maximum(u, candidates).tolist())
            # Count cases whose microsecond serialization could alter strict inequality.
            near_lo = max(lo, int(np.searchsorted(starts, threshold - 1000, side="left")))
            near_hi = int(np.searchsorted(starts, threshold + 1000, side="right"))
            near_tick_boundary += max(0, near_hi - near_lo)
            equal_hi = int(np.searchsorted(starts, threshold, side="right"))
            equality_compatible += max(0, equal_hi - max(lo, stop))
    codes = np.asarray(left, dtype=np.int64) * n + np.asarray(right, dtype=np.int64)
    return codes, {"near_microsecond_boundary_pairs": near_tick_boundary,
                   "exact_gap_equality_compatible_pairs": equality_compatible}


def build_edges(start_ns, end_ns, antennas, owners, ground_gap_seconds, satellite_gap_seconds):
    n = len(start_ns)
    if n and n > int(np.sqrt(INT64_MAX)):
        raise OverflowError("Packed edge encoding exceeds int64")
    ground, ground_report = grouped_edges(start_ns, end_ns, antennas, ground_gap_seconds)
    satellite, satellite_report = grouped_edges(start_ns, end_ns, owners, satellite_gap_seconds)
    codes = np.concatenate((ground, satellite))
    types = np.concatenate((np.ones(len(ground), np.uint8), np.full(len(satellite), 2, np.uint8)))
    if len(codes):
        order = np.argsort(codes, kind="stable")
        codes, types = codes[order], types[order]
        first = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
        edge_types = np.bitwise_or.reduceat(types, first)
        codes = codes[first]
        edges = np.column_stack((codes // n, codes % n)).astype(np.int64, copy=False)
    else:
        edge_types = np.empty(0, dtype=np.uint8)
        edges = np.empty((0, 2), dtype=np.int64)
    return edges, edge_types, codes, {"ground": ground_report, "satellite": satellite_report}


def resource_memberships(rows, owners, stations, satellite_ids):
    antenna_ids = [s["antenna_id"] for s in stations]
    ant_map = {x: i for i, x in enumerate(antenna_ids)}
    n_ground = len(antenna_ids)
    ground = np.asarray([ant_map[x["antenna_id"]] for x in rows], dtype=np.int32)
    vertex_factors = np.column_stack((ground, owners + n_ground)).astype(np.int32)
    n_factors = n_ground + len(satellite_ids)
    vertex_list = [[] for _ in range(n_factors)]
    for vertex, pair in enumerate(vertex_factors):
        vertex_list[int(pair[0])].append(vertex); vertex_list[int(pair[1])].append(vertex)
    lengths = np.asarray([len(x) for x in vertex_list], dtype=np.int64)
    indptr = np.r_[np.int64(0), np.cumsum(lengths)]
    vertices = np.asarray([v for group in vertex_list for v in group], dtype=np.int64)
    return {
        "vertex_factors": vertex_factors,
        "factor_indptr": indptr, "factor_vertices": vertices,
        "factor_ids": np.asarray(antenna_ids + satellite_ids),
        "factor_kind": np.asarray(["ground_antenna"] * n_ground + ["satellite_channel"] * len(satellite_ids)),
        "factor_capacity": np.ones(n_factors, dtype=np.int8),
        "factor_semantics": np.asarray("Temporal resource memberships, not static cliques; conflicts are only the edges."),
    }


def greedy_schedule_check(rows, arrays, ground_gap, satellite_gap, horizon, active_stations):
    """One ordinary duration-greedy feasible witness; never a learning claim."""
    calendars = defaultdict(list)
    chosen = []
    for vertex in sorted(range(len(rows)), key=lambda i: (-arrays["weights"][i], rows[i]["contact_id"])):
        start, end = int(arrays["start_ns"][vertex]), int(arrays["end_ns"][vertex])
        resources = (("ground:" + rows[vertex]["antenna_id"], decimal_ns(ground_gap)),
                     ("sat:" + rows[vertex]["satellite_id"], decimal_ns(satellite_gap)))
        feasible = True
        insertions = []
        for resource, gap_ns in resources:
            calendar = calendars[resource]
            position = bisect.bisect_left(calendar, (start, end, vertex))
            if position and start < calendar[position - 1][1] + gap_ns:
                feasible = False; break
            if position < len(calendar) and calendar[position][0] < end + gap_ns:
                feasible = False; break
            insertions.append((calendar, position))
        if feasible:
            for calendar, position in insertions:
                calendar.insert(position, (start, end, vertex))
            chosen.append(vertex)
    total = float(np.sum(arrays["weights"][chosen], dtype=np.float64))
    bound = min(len(active_stations), len(set(arrays["owner"]))) * horizon
    if total > bound + 0.000001:
        raise AssertionError("Capacity-one schedule violates total duration sanity bound")
    selected = np.asarray(chosen, dtype=np.int64)
    mask = np.zeros(len(rows), dtype=np.bool_)
    mask[selected] = True
    if np.any(mask[arrays["edge_u"]] & mask[arrays["edge_v"]]):
        raise AssertionError("Resource-calendar witness is not independent in the conflict graph")
    return {"duration_greedy_feasible_count": len(chosen),
            "duration_greedy_total_seconds": total,
            "capacity_total_duration_upper_bound_seconds": bound,
            "capacity_bound_valid": True,
            "duration_greedy_independent_set_valid": True}, selected


def build_view(rows, mother_arrays, selected_sites, stations, satellite_ids, scene, config, horizon):
    selected = np.asarray([i for i, row in enumerate(rows) if row["site_id"] in selected_sites], dtype=np.int64)
    view_rows = [rows[i] for i in selected]
    arrays = {key: value[selected].copy() for key, value in mother_arrays.items()}
    active_stations = [x for x in stations if x["site_id"] in selected_sites]
    edges, edge_types, codes, boundary = build_edges(
        arrays["start_ns"], arrays["end_ns"], [x["antenna_id"] for x in view_rows], arrays["owner"],
        config["ground_gap_seconds"], config["satellite_gap_seconds"])
    degrees = np.bincount(edges.ravel(), minlength=len(view_rows)).astype(np.int64)
    arrays.update({"edges": edges, "edge_u": edges[:, 0], "edge_v": edges[:, 1],
                   "edge_types": edge_types, "degrees": degrees, "agents": arrays["owner"],
                   "mother_contact_index": selected})
    for column in REQUIRED_COLUMNS - {"start_rel_seconds", "end_rel_seconds", "duration_seconds"}:
        arrays[column] = np.asarray([row[column] for row in view_rows])
    arrays.update({"start_seconds": np.asarray([float(row["start_rel_seconds"]) for row in view_rows], dtype=np.float64),
                   "end_seconds": np.asarray([float(row["end_rel_seconds"]) for row in view_rows], dtype=np.float64),
                   "start_decimal_seconds": np.asarray([row["start_rel_seconds"] for row in view_rows]),
                   "end_decimal_seconds": np.asarray([row["end_rel_seconds"] for row in view_rows]),
                   "duration_decimal_seconds": np.asarray([str(Decimal(row["end_rel_seconds"]) - Decimal(row["start_rel_seconds"])) for row in view_rows]),
                   "satellite_id_mapping": np.asarray(satellite_ids),
                   "source_group": np.asarray(scene["source_group"]), "split": np.asarray(scene["split"]),
                   "geometry_id": np.asarray(scene["geometry_id"]), "replicate_id": np.asarray(scene["replicate_id"]),
                   "ground_gap_seconds": np.asarray(config["ground_gap_seconds"], dtype=np.int64),
                   "satellite_gap_seconds": np.asarray(config["satellite_gap_seconds"], dtype=np.int64),
                   "planning_horizon_seconds": np.asarray(horizon, dtype=np.int64),
                   "solver_tick_seconds": np.asarray(0.000001, dtype=np.float64),
                   "edge_endpoint_serialization_seconds": np.asarray(0.000000001, dtype=np.float64)})
    arrays.update(resource_memberships(view_rows, arrays["owner"], active_stations, satellite_ids))
    arrays["factor_gap_seconds"] = np.r_[np.full(len(active_stations), config["ground_gap_seconds"], dtype=np.int64),
                                           np.full(len(satellite_ids), config["satellite_gap_seconds"], dtype=np.int64)]
    cross_owner = arrays["owner"][edges[:, 0]] != arrays["owner"][edges[:, 1]]
    # Independently interpret the saved edges and check resource-bit meanings.
    u, v = edges[:, 0], edges[:, 1]
    first = np.where(arrays["start_ns"][u] <= arrays["start_ns"][v], u, v)
    later = np.where(first == u, v, u)
    ground_same = arrays["vertex_factors"][u, 0] == arrays["vertex_factors"][v, 0]
    satellite_same = arrays["owner"][u] == arrays["owner"][v]
    ground_expected = ground_same & (arrays["start_ns"][later] < arrays["end_ns"][first] + decimal_ns(config["ground_gap_seconds"]))
    satellite_expected = satellite_same & (arrays["start_ns"][later] < arrays["end_ns"][first] + decimal_ns(config["satellite_gap_seconds"]))
    if np.any(edge_types != ground_expected.astype(np.uint8) + 2 * satellite_expected.astype(np.uint8)):
        raise AssertionError("Saved conflict resource types violate original interval rule")
    if np.any(edge_types == 0) or np.any(u >= v) or len(np.unique(codes)) != len(codes):
        raise AssertionError("Invalid, self, duplicate or unordered edge in conflict graph")
    stats = {
        "vertices": len(view_rows), "edges": len(edges),
        "ground_edges_including_double": int(np.count_nonzero(edge_types & 1)),
        "satellite_edges_including_double": int(np.count_nonzero(edge_types & 2)),
        "ground_only_edges": int(np.count_nonzero(edge_types == 1)),
        "satellite_only_edges": int(np.count_nonzero(edge_types == 2)),
        "double_resource_edges": int(np.count_nonzero(edge_types == 3)),
        "cross_agent_edges": int(np.count_nonzero(cross_owner)),
        "cross_agent_edge_fraction": float(np.mean(cross_owner)) if len(edges) else 0.0,
        "mean_degree": float(np.mean(degrees)), "median_degree": float(np.median(degrees)),
        "max_degree": int(np.max(degrees)), "degree_p95": float(np.percentile(degrees, 95)),
        "mean_contact_duration_seconds": float(np.mean(arrays["weights"])),
        "min_contact_duration_seconds": float(np.min(arrays["weights"])),
        "max_contact_duration_seconds": float(np.max(arrays["weights"])),
        "raw_contact_duration_sum_seconds": float(np.sum(arrays["weights"])),
        "satellite_mapping_count": len(satellite_ids), "station_count": len(active_stations),
        "boundary_strict_inequality": boundary,
        "owner_contract": "Satellite identity; not a time partition or all-day capacity-one clique.",
        "edge_resource_types_valid": True,
        "duplicate_or_self_edges": 0,
        "R8_unchanged_subset": bool(np.array_equal(arrays["weights"], mother_arrays["weights"][selected])
                                   and np.array_equal(arrays["start_ns"], mother_arrays["start_ns"][selected])
                                   and np.array_equal(arrays["end_ns"], mother_arrays["end_ns"][selected])),
    }
    schedule_report, selected_witness = greedy_schedule_check(view_rows, arrays, config["ground_gap_seconds"],
                                                             config["satellite_gap_seconds"], horizon, active_stations)
    stats.update(schedule_report)
    arrays["duration_greedy_witness_vertices"] = selected_witness
    return arrays, codes, stats


def run(root: Path, replicate_ids: list[str]):
    parameter_path = root / "protocol" / "JointRecovery_parameters.json"
    parameters = json.loads(parameter_path.read_text(encoding="utf-8-sig"))
    parameter_hash = sha256(parameter_path)
    scene_map = {x["replicate_id"]: x for x in parameters["scene_definitions"]}
    horizon = int(parameters["common_settings"]["planning_horizon_hours"] * 3600)
    graph_dir, reports_dir = root / "graphs", root / "reports"
    graph_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "graph_validation.json"
    summary, report = [], {"status": "PASSED", "parameters_sha256": parameter_hash,
                           "graph_contract_version": "complete-stk-duration-v1",
                           "physical_accuracy_note": "9-digit export and microsecond solver ticks do not establish physical time accuracy.",
                           "learning_advantage_evaluated": False, "mothers": {}, "graphs": {}, "gap_monotonicity": []}
    if report_path.exists():
        prior_report = json.loads(report_path.read_text(encoding="utf-8"))
        if prior_report["parameters_sha256"] != parameter_hash:
            raise ValueError("Existing graph report uses different parameters; choose a fresh dataset directory")
        report = prior_report
    for replicate in replicate_ids:
        scene = scene_map[replicate]
        source_csv = root / "raw_geometry" / scene["scene_id"] / "contacts.csv"
        rows, mother_arrays, source_validation = read_contacts(source_csv, scene, parameters["stations"], horizon)
        csv_hash = sha256(source_csv)
        source_validation["contacts_sha256"] = csv_hash
        report["mothers"][scene["scene_id"]] = source_validation
        report["gap_monotonicity"] = [x for x in report["gap_monotonicity"] if not x["smaller_gap_graph"].startswith(scene["scene_id"] + "-")]
        satellite_ids = [x["satellite_id"] for x in scene["satellites"]]
        for view_name in parameters["priorities"]["P0"]["station_views"]:
            previous_codes = None
            previous_graph = None
            for config in sorted(parameters["graph_configurations"], key=lambda x: x["ground_gap_seconds"]):
                graph_id = f"{scene['scene_id']}-{view_name}-{config['config_id']}"
                arrays, codes, stats = build_view(rows, mother_arrays, set(parameters["station_views"][view_name]),
                                                 parameters["stations"], satellite_ids, scene, config, horizon)
                arrays["graph_id"] = np.asarray(graph_id)
                arrays["parameters_sha256"] = np.asarray(parameter_hash)
                arrays["contacts_sha256"] = np.asarray(csv_hash)
                output = graph_dir / f"{graph_id}.npz"
                np.savez_compressed(output, **arrays)
                stats.update({"graph_id": graph_id, "source_group": scene["source_group"], "split": scene["split"],
                              "replicate_id": replicate, "station_view": view_name,
                              "ground_gap_seconds": config["ground_gap_seconds"], "satellite_gap_seconds": config["satellite_gap_seconds"],
                              "contacts_sha256": csv_hash, "parameters_sha256": parameter_hash, "npz_sha256": sha256(output)})
                report["graphs"][graph_id] = stats
                (graph_dir / f"{graph_id}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if previous_codes is not None:
                    indices = np.searchsorted(codes, previous_codes)
                    passed = bool(np.all(indices < len(codes)) and np.array_equal(codes[indices], previous_codes))
                    report["gap_monotonicity"].append({"smaller_gap_graph": previous_graph, "larger_gap_graph": graph_id,
                                                        "subset": passed, "additional_edges": int(len(codes) - len(previous_codes))})
                    if not passed:
                        raise AssertionError("Increasing ground switching gap removed a conflict edge")
                previous_codes, previous_graph = codes, graph_id
                print(json.dumps({"graph_id": graph_id, "vertices": stats["vertices"], "edges": stats["edges"],
                                  "mean_degree": stats["mean_degree"]}), flush=True)
    summary = [{key: value for key, value in report["graphs"][graph_id].items() if not isinstance(value, (dict, list))}
               for graph_id in sorted(report["graphs"])]
    report["graph_count"] = len(summary)
    report["source_group_count"] = len(report["mothers"])
    report["source_group_leakage"] = False
    report["validation_completed_utc"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (reports_dir / "graph_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print(json.dumps({"status": "PASSED", "graphs": len(summary), "report": str(report_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replicates", nargs="+", default=["r000", "r001"])
    args = parser.parse_args()
    run(args.dataset_root.resolve(), args.replicates)
