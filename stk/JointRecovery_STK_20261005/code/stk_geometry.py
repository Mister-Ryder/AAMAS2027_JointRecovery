"""Generate the fixed JointRecovery P0 geometry in an owned STK 11 session.

The implementation uses the installed STK 11 COM interfaces, not stk12.
No previous contact CSV is consumed. Importing this module never starts STK.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

TYPELIB = "{D6A1725B-89FF-43A4-995B-7F055549F4EB}"
HELP_XML = Path(r"C:\Program Files\AGI\STK 11\Help\ObjectModel\xml\Interop.AGI.STKObjects.xml")
EARTH_CB = Path(r"C:\Program Files\AGI\STK 11\STKData\CentralBodies\Earth\Earth.cb")
CONTACT_FIELDS = ["source_group", "geometry_id", "replicate_id", "epoch_utc", "contact_id",
                  "pass_id", "satellite_id", "site_id", "antenna_id", "start_rel_seconds",
                  "end_rel_seconds", "duration_seconds", "start_utc", "end_utc",
                  "boundary_crossing", "access_settings_hash"]
RAW_FIELDS = CONTACT_FIELDS + ["padding_boundary_clipped"]
_TYPELIB_MODULE = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def utc_datetime(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def utcg(value):
    if isinstance(value, str):
        value = utc_datetime(value)
    # Use English month names independently of the operating system locale.
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{value.day:02d} {months[value.month-1]} {value.year} {value:%H:%M:%S}.000000"


def decimal_endpoint(value):
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite Access endpoint")
    return format(float(value), ".9f")


def classify_interval(start, stop, horizon):
    """Classify unmodified intervals; no epsilon admits an outside endpoint."""
    if not (math.isfinite(start) and math.isfinite(stop) and start < stop):
        raise ValueError("Access interval must have finite increasing endpoints")
    if start >= 0.0 and stop <= horizon:
        return "none"
    if stop <= 0.0:
        return "outside_before"
    if start >= horizon:
        return "outside_after"
    if start < 0.0 and stop > horizon:
        return "both"
    return "left" if start < 0.0 else "right"


def validate_parameters(parameters, replicates):
    settings = parameters["common_settings"]
    required = {"propagator": "J2Perturbation", "initial_state_coordinate_system": "J2000",
                "earth_model": "WGS84", "planning_horizon_hours": 72,
                "propagation_padding_hours_each_side": 1, "minimum_elevation_deg": 15,
                "site_altitude_m_ellipsoid": 0, "satellite_capacity": 1,
                "antennas_per_site": 1, "access_max_step_seconds": 30,
                "access_time_convergence_seconds": 0.001, "access_precise_event_times": True,
                "light_time_delay": False}
    for key, expected in required.items():
        if settings.get(key) != expected:
            raise ValueError(f"P0 contract mismatch: {key}={settings.get(key)!r}, expected {expected!r}")
    stations = parameters["stations"]
    if len(stations) != 12 or len({s["site_id"] for s in stations}) != 12:
        raise ValueError("P0 requires the fixed 12 unique sites")
    exact_coordinates = {(lat, lon) for lat in (28, 32) for lon in (88, 94, 100, 106, 112, 118)}
    if {(s["latitude_deg"], s["longitude_deg_east"]) for s in stations} != exact_coordinates:
        raise ValueError("Site coordinates differ from the R12 protocol")
    scenes = {s["replicate_id"]: s for s in parameters["scene_definitions"]}
    selected = []
    for replicate in replicates:
        scene = scenes[replicate]
        sats = scene["satellites"]
        if len(sats) != 168 or len({s["satellite_id"] for s in sats}) != 168:
            raise ValueError(f"{replicate}: expected 168 unique satellite element records")
        if (utc_datetime(scene["horizon_stop_utc"]) - utc_datetime(scene["epoch_utc"])).total_seconds() != 259200:
            raise ValueError(f"{replicate}: planning interval is not 72 hours")
        for sat in sats:
            for key in ("semi_major_axis_km", "eccentricity", "inclination_deg", "argument_of_perigee_deg", "raan_deg", "mean_anomaly_deg"):
                if not math.isfinite(float(sat[key])):
                    raise ValueError(f"Nonfinite orbit element: {sat['satellite_id']}/{key}")
        selected.append(scene)
    return selected


def _cast(obj, interface):
    # STK 11 provides automation implementations as _IAg..., while IAg...
    # are vtable-only interfaces. This also handles STKUtil-returned bases.
    global _TYPELIB_MODULE
    import win32com.client.gencache as cache
    if _TYPELIB_MODULE is None:
        _TYPELIB_MODULE = cache.EnsureModule(TYPELIB, 0, 1, 0, bForDemand=False)
    name = "_" + interface
    iid = _TYPELIB_MODULE.NamesToIIDMap.get(name)
    if iid is None:
        raise RuntimeError(f"The installed STK 11 typelib does not contain {interface}")
    wrapper = getattr(cache.GetModuleForCLSID(iid), name)
    return wrapper(getattr(obj, "_oleobj_", obj))


def connect(root, command):
    result = root.ExecuteCommand(command)
    return [str(result.Item(i)) for i in range(result.Count)]


@contextmanager
def owned_stk11():
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    application = None
    try:
        application = win32com.client.DispatchEx("STK11.Application")
        application.Visible = False
        application.UserControl = False
        root = application.Personality2
        version = connect(root, "GetSTKVersion / Details")
        if not version or not any("v11." in value for value in version):
            raise RuntimeError(f"STK 11 required; actual version is {version}")
        yield root, version
    finally:
        if application is not None:
            try:
                application.Quit()
            except Exception as exc:
                print(f"Owned STK instance cleanup warning: {exc}", file=sys.stderr, flush=True)
        application = None
        pythoncom.CoUninitialize()


def constraints_snapshot(obj):
    result = []
    for i in range(obj.AccessConstraints.Count):
        constraint = _cast(obj.AccessConstraints.Item(i), "IAgAccessConstraint")
        result.append({"type": int(constraint.ConstraintType), "name": str(constraint.ConstraintName)})
    return result


def keep_constraints(obj, allowed):
    collection = obj.AccessConstraints
    # Enumerate first because removal mutates the collection indices.
    for constraint in constraints_snapshot(obj):
        if constraint["type"] not in allowed:
            collection.RemoveConstraint(constraint["type"])
    # 26=LOS, 14=elevation, verified in the installed STK11 typelib.
    for kind in sorted(allowed):
        if not collection.IsConstraintActive(kind):
            if not collection.IsConstraintSupported(kind):
                raise RuntimeError(f"Required constraint {kind} unsupported for {obj.InstanceName}")
            collection.AddConstraint(kind)


def orbit_snapshot(satellite):
    prop_type = int(satellite.PropagatorType)
    if prop_type != 1:
        raise RuntimeError(f"Expected J2Perturbation type 1, found {prop_type}")
    propagator = _cast(satellite.Propagator, "IAgVePropagatorJ2Perturbation")
    state = _cast(propagator.InitialState.Representation.ConvertTo(1), "IAgOrbitStateClassical")
    state.SizeShapeType = 4  # eSizeShapeSemimajorAxis
    state.LocationType = 2   # eLocationMeanAnomaly
    state.Orientation.AscNodeType = 1  # eAscNodeRAAN
    shape = _cast(state.SizeShape, "IAgClassicalSizeShapeSemimajorAxis")
    location = _cast(state.Location, "IAgClassicalLocationMeanAnomaly")
    ascnode = _cast(state.Orientation.AscNode, "IAgOrientationAscNodeRAAN")
    return {"propagator_type": prop_type, "propagator": "J2Perturbation",
            "propagation_start_utcg": str(propagator.StartTime), "propagation_stop_utcg": str(propagator.StopTime),
            "propagation_step_seconds": float(propagator.Step), "propagation_frame_enum": int(propagator.InitialState.PropagationFrame),
            "initial_state_coordinate_system_enum": int(state.CoordinateSystemType),
            "initial_state_epoch_utcg": str(state.Epoch), "central_body": str(state.CentralBodyName),
            "semi_major_axis_km": float(shape.SemiMajorAxis), "eccentricity": float(shape.Eccentricity),
            "inclination_deg": float(state.Orientation.Inclination), "argument_of_perigee_deg": float(state.Orientation.ArgOfPerigee),
            "raan_deg": float(ascnode.Value), "mean_anomaly_deg": float(location.Value),
            "constraints": constraints_snapshot(satellite)}


def build_scene(root, parameters, specification, output):
    settings = parameters["common_settings"]
    epoch = utc_datetime(specification["epoch_utc"])
    stop = utc_datetime(specification["horizon_stop_utc"])
    padding = timedelta(hours=settings["propagation_padding_hours_each_side"])
    start_padded, stop_padded = epoch - padding, stop + padding
    root.NewScenario(specification["scene_id"])
    units = root.UnitPreferences
    units.SetCurrentUnit("DateFormat", "UTCG")
    units.SetCurrentUnit("DistanceUnit", "km")
    units.SetCurrentUnit("AngleUnit", "deg")
    units.SetCurrentUnit("TimeUnit", "sec")
    scenario = root.CurrentScenario
    scenario.SetTimePeriod(utcg(start_padded), utcg(stop_padded))
    scenario.Epoch = utcg(epoch)
    commands = []
    def send(command):
        commands.append({"command": command, "result": connect(root, command)})
    send("Units_Set * Connect Distance km")
    actual_sats = []
    for i, satellite_spec in enumerate(specification["satellites"], 1):
        name = satellite_spec["satellite_id"]
        send(f"New / */Satellite {name}")
        elements = [satellite_spec[key] for key in ("semi_major_axis_km", "eccentricity", "inclination_deg", "argument_of_perigee_deg", "raan_deg", "mean_anomaly_deg")]
        values = " ".join(format(float(value), ".17g") for value in elements)
        send(f'SetState */Satellite/{name} Classical J2Perturbation "{utcg(start_padded)}" "{utcg(stop_padded)}" 30 J2000 "{utcg(epoch)}" {values}')
        satellite = _cast(root.GetObjectFromPath(f"Satellite/{name}"), "IAgSatellite")
        keep_constraints(satellite, {26})
        actual = orbit_snapshot(satellite)
        for key in ("semi_major_axis_km", "eccentricity", "inclination_deg", "argument_of_perigee_deg"):
            if abs(actual[key] - float(satellite_spec[key])) > 1e-7:
                raise RuntimeError(f"Applied orbit element differs: {name}, {key}")
        for key in ("raan_deg", "mean_anomaly_deg"):
            angular_error = (actual[key] - float(satellite_spec[key]) + 180.0) % 360.0 - 180.0
            if abs(angular_error) > 1e-7:
                raise RuntimeError(f"Applied angular element differs: {name}, {key}")
        actual_sats.append({"satellite_id": name, "requested": satellite_spec, "actual": actual})
        if i == 1 or i % 28 == 0:
            print(f"{specification['replicate_id']} propagated {i}/168", flush=True)
    actual_sites = []
    for station_spec in parameters["stations"]:
        name = station_spec["site_id"]
        send(f"New / */Facility {name}")
        facility = _cast(root.GetObjectFromPath(f"Facility/{name}"), "IAgFacility")
        facility.UseTerrain = False
        facility.AltRef = 2  # eWGS84, ellipsoid altitude (not mean sea level).
        facility.HeightAboveGround = 0.0
        facility.Position.AssignGeodetic(float(station_spec["latitude_deg"]), float(station_spec["longitude_deg_east"]), 0.0)
        keep_constraints(facility, {14, 26})
        elevation = _cast(facility.AccessConstraints.GetActiveConstraint(14), "IAgAccessCnstrMinMax")
        elevation.EnableMin = True
        elevation.EnableMax = False
        elevation.Min = float(settings["minimum_elevation_deg"])
        actual_sites.append({**station_spec, "actual_use_terrain": bool(facility.UseTerrain),
                             "actual_altitude_reference_enum": int(facility.AltRef), "actual_height_above_ground_km": float(facility.HeightAboveGround),
                             "actual_min_elevation_deg": float(elevation.Min), "elevation_min_enabled": bool(elevation.EnableMin),
                             "elevation_max_enabled": bool(elevation.EnableMax), "active_constraints": constraints_snapshot(facility)})
    write_json(output / "actual_orbit_elements.json", actual_sats)
    write_json(output / "actual_station_settings.json", actual_sites)
    write_json(output / "connect_commands.json", commands)
    return {"name": str(scenario.InstanceName), "epoch_utcg": str(scenario.Epoch),
            "analysis_start_utcg": str(scenario.StartTime), "analysis_stop_utcg": str(scenario.StopTime),
            "planning_start_utc": specification["epoch_utc"], "planning_stop_utc": specification["horizon_stop_utc"],
            "propagation_padding_seconds_each_side": padding.total_seconds(),
            "propagation_step_seconds": 30, "orbit_element_count": len(actual_sats), "site_count": len(actual_sites)}


def configure_access(access, settings, max_step):
    advanced = access.Advanced
    advanced.UsePreciseEventTimes = bool(settings["access_precise_event_times"])
    advanced.UseFixedTimeStep = False
    advanced.MaxTimeStep = float(max_step)
    advanced.TimeConvergence = float(settings["access_time_convergence_seconds"])
    advanced.EnableLightTimeDelay = bool(settings["light_time_delay"])
    actual = {"UsePreciseEventTimes": bool(advanced.UsePreciseEventTimes),
              "UseFixedTimeStep": bool(advanced.UseFixedTimeStep), "MaxTimeStep": float(advanced.MaxTimeStep),
              "MinTimeStep": float(advanced.MinTimeStep), "TimeConvergence": float(advanced.TimeConvergence),
              "EnableLightTimeDelay": bool(advanced.EnableLightTimeDelay), "AbsoluteTolerance": float(advanced.AbsoluteTolerance),
              "RelativeTolerance": float(advanced.RelativeTolerance), "AberrationType": int(advanced.AberrationType)}
    for key, expected in (("UsePreciseEventTimes", True), ("UseFixedTimeStep", False), ("EnableLightTimeDelay", False)):
        if actual[key] != expected:
            raise RuntimeError(f"Access property was not applied: {key}")
    if actual["MaxTimeStep"] != max_step or actual["TimeConvergence"] != settings["access_time_convergence_seconds"]:
        raise RuntimeError("Access numeric settings were not applied")
    return actual


def extract_pair(access, padded_start, padded_stop):
    access.ComputeAccess()
    provider = access.DataProviders.GetDataPrvIntervalFromPath("Access Data")
    report = provider.Exec(str(padded_start), str(padded_stop))
    starts, stops = [], []
    if report.DataSets.Count:
        starts = list(report.DataSets.GetDataSetByName("Start Time").GetValues())
        stops = list(report.DataSets.GetDataSetByName("Stop Time").GetValues())
    if len(starts) != len(stops):
        raise RuntimeError("Unequal raw Access endpoint array lengths")
    intervals = sorted(zip(map(float, starts), map(float, stops)))
    previous_stop = -math.inf
    for start, stop in intervals:
        if not (math.isfinite(start) and math.isfinite(stop) and padded_start - 1e-7 <= start < stop <= padded_stop + 1e-7):
            raise RuntimeError(f"Invalid padded Access interval: {start}, {stop}")
        if start < previous_stop:
            raise RuntimeError("Overlapping or duplicate Access intervals returned for a pair")
        previous_stop = stop
    return intervals, {"provider": "Access Data", "date_unit": "EpSec", "start_values": starts, "stop_values": stops,
                       "original_value_python_types": sorted({type(value).__name__ for value in starts + stops})}


def fixed_spot_check_pairs(specification, stations):
    # Chosen from indices before Access execution, independent of opportunity or
    # algorithm results. Both shells, both latitudes and east/west edges appear.
    satellites = specification["satellites"]
    return {(satellites[si]["satellite_id"], stations[gi]["site_id"])
            for si, gi in ((0, 0), (0, 11), (41, 2), (41, 9), (83, 5), (83, 6),
                           (84, 0), (84, 11), (125, 2), (125, 9), (167, 5), (167, 6))}


def compare_intervals(baseline, tighter, tolerance=0.005):
    count_match = len(baseline) == len(tighter)
    max_error = max((max(abs(a-c), abs(b-d)) for (a, b), (c, d) in zip(baseline, tighter)), default=0.0) if count_match else None
    return {"baseline_interval_count": len(baseline), "spot_interval_count": len(tighter),
            "interval_count_match": count_match, "maximum_endpoint_difference_seconds": max_error,
            "comparison_tolerance_seconds": tolerance, "within_comparison_tolerance": bool(count_match and max_error <= tolerance)}


def save_scene_ascii_bridge(root, output, specification):
    """Bridge STK 11's non-Unicode save path without changing the data location.

    The complete native scene directory is copied into the requested dataset;
    a second ASCII copy is reloaded to check relocation and dependencies.
    Only newly created, verified private temporary directories are removed.
    """
    destination = output / "scene"
    destination.mkdir(exist_ok=True)
    if any(destination.rglob("*")):
        raise RuntimeError(f"Refusing to overwrite existing native scene files: {destination}")
    temporary_parent = Path(tempfile.gettempdir()).resolve()
    if not str(temporary_parent).isascii():
        raise RuntimeError("STK save bridge requires an ASCII system temporary directory")
    with tempfile.TemporaryDirectory(prefix="JR_STK_Save_", dir=temporary_parent) as temporary_name:
        temporary = Path(temporary_name).resolve()
        if temporary.parent != temporary_parent or not temporary.name.startswith("JR_STK_Save_"):
            raise RuntimeError("Temporary path validation failed")
        source = temporary / "native"
        relocated = temporary / "relocated"
        source.mkdir()
        scene_name = specification["scene_id"] + ".sc"
        root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
        root.SaveScenarioAs(str(source / scene_name))
        native_files = [p for p in source.rglob("*") if p.is_file()]
        if not (source / scene_name).is_file() or not native_files:
            raise RuntimeError("STK returned from save without the native scenario file")
        needles = {str(temporary).encode().lower(), temporary.as_posix().encode().lower(),
                   str(source).encode().lower(), source.as_posix().encode().lower()}
        absolute_temp_references = []
        for path in native_files:
            data = path.read_bytes().lower()
            if any(needle in data for needle in needles):
                absolute_temp_references.append(path.relative_to(source).as_posix())
        if absolute_temp_references:
            raise RuntimeError(f"Native scene contains nonportable absolute temporary references: {absolute_temp_references}")
        shutil.copytree(source, destination, dirs_exist_ok=True)
        # Reload from a distinct path populated from the delivered Chinese-path
        # files, not from the original save directory. STK receives ASCII only.
        shutil.copytree(destination, relocated)
        root.CloseScenario()
        root.LoadScenario(str(relocated / scene_name))
        root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
        scene = root.CurrentScenario
        observed = {"scene_name": str(scene.InstanceName), "child_count": int(scene.Children.Count),
                    "epoch_utcg": str(scene.Epoch), "analysis_start_utcg": str(scene.StartTime),
                    "analysis_stop_utcg": str(scene.StopTime)}
        if observed["scene_name"] != specification["scene_id"] or observed["child_count"] != 180:
            raise RuntimeError(f"Relocated native scene is incomplete: {observed}")
        root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")
        epoch_value = float(root.ConversionUtility.ConvertDate("UTCG", "EpSec", utcg(specification["epoch_utc"])))
        if abs(epoch_value) > 1e-8:
            raise RuntimeError("Relocated scenario epoch changed")
        root.CloseScenario()
        file_receipts = [{"path": p.relative_to(destination).as_posix(), "size_bytes": p.stat().st_size, "sha256": digest(p)}
                         for p in sorted(destination.rglob("*")) if p.is_file()]
        return {"mechanism": "STK11 saved to an owned ASCII temporary directory; all native scene files copied to requested dataset directory",
                "reason": "STK11 SaveScenarioAs rejected the requested Unicode directory; geometry and constraints unchanged",
                "absolute_temporary_references": absolute_temp_references,
                "relocation_reload_validation": observed, "validation_from_delivered_copy": True,
                "temporary_directory_retained": False, "native_file_receipts": file_receipts}


def recover_scene(parameters_path, parameters, specification, output_root):
    """Recover a save-only failure without recomputing the 2016 access pairs."""
    output = output_root / "raw_geometry" / specification["scene_id"]
    previous = json.loads((output / "manifest.json").read_text(encoding="utf-8-sig"))
    if previous.get("status") != "failed" or "Cannot write to output directory" not in previous.get("error", ""):
        raise RuntimeError("Scene recovery is restricted to the observed STK save-path failure")
    if previous["parameters_sha256"] != digest(parameters_path):
        raise RuntimeError("Original Access execution used different input parameters")
    protected_files = ["contacts.csv", "boundary_contacts.csv", "access_raw.csv", "raw_access_reports.jsonl",
                       "actual_orbit_elements.json", "actual_station_settings.json", "connect_commands.json", "scenario_summary.json"]
    before_hashes = {name: digest(output / name) for name in protected_files}
    manifest = dict(previous)
    manifest["status"] = "save_recovery_running"
    manifest["original_geometry_generator_sha256"] = previous["generator_sha256"]
    manifest["save_recovery_generator_sha256"] = digest(__file__)
    manifest["original_save_error"] = previous["error"]
    manifest["save_recovery_started_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    settings = parameters["common_settings"]
    horizon = settings["planning_horizon_hours"] * 3600.0
    padding = settings["propagation_padding_hours_each_side"] * 3600.0
    padded_start, padded_stop = -padding, horizon + padding
    original_reports, pair_records, variants = {}, [], {}
    counts = {"raw_padded": 0, "full_contained": 0, "crossing": 0, "outside_planning": 0, "padding_boundary_clipped": 0}
    with (output / "raw_access_reports.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            report = json.loads(line)
            key = report["satellite_id"], report["site_id"]
            if key in original_reports:
                raise RuntimeError("Duplicate original Access report pair")
            if canonical_hash(report["settings"]) != report["settings_hash"]:
                raise RuntimeError("Original Access settings hash mismatch")
            original_reports[key] = report
            variants[report["settings_hash"]] = report["settings"]
            starts, stops = report["start_values"], report["stop_values"]
            if len(starts) != len(stops):
                raise RuntimeError("Original raw report contains unequal endpoint arrays")
            pair_count = {"satellite_id": key[0], "site_id": key[1], "raw_intervals": len(starts), "contained": 0, "crossing": 0, "access_settings_hash": report["settings_hash"]}
            for a, b in zip(map(float, starts), map(float, stops)):
                category = classify_interval(a, b, horizon)
                counts["raw_padded"] += 1
                counts["padding_boundary_clipped"] += int(abs(a - padded_start) < 1e-7 or abs(b - padded_stop) < 1e-7)
                if category == "none":
                    counts["full_contained"] += 1; pair_count["contained"] += 1
                elif category in {"left", "right", "both"}:
                    counts["crossing"] += 1; pair_count["crossing"] += 1
                else:
                    counts["outside_planning"] += 1
            pair_records.append(pair_count)
    if len(original_reports) != 2016 or counts != previous["counts_partial"]:
        raise RuntimeError("Original reports are not the complete 2016-pair failed-save execution")
    metadata = output / ("save_recovery_metadata_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    metadata.mkdir(exist_ok=False)
    spot_records = []
    try:
        with owned_stk11() as (root, version):
            summary = build_scene(root, parameters, specification, metadata)
            new_orbits = json.loads((metadata / "actual_orbit_elements.json").read_text(encoding="utf-8"))
            old_orbits = json.loads((output / "actual_orbit_elements.json").read_text(encoding="utf-8"))
            new_stations = json.loads((metadata / "actual_station_settings.json").read_text(encoding="utf-8"))
            old_stations = json.loads((output / "actual_station_settings.json").read_text(encoding="utf-8"))
            if canonical_hash(new_orbits) != canonical_hash(old_orbits) or canonical_hash(new_stations) != canonical_hash(old_stations):
                raise RuntimeError("Rebuilt scene actual orbit/constraint settings differ from the original Access execution")
            root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")
            for sat_id, site_id in sorted(fixed_spot_check_pairs(specification, parameters["stations"])):
                satellite = root.GetObjectFromPath(f"Satellite/{sat_id}")
                facility = root.GetObjectFromPath(f"Facility/{site_id}")
                access = satellite.GetAccessToObject(facility)
                original = original_reports[sat_id, site_id]
                actual_settings = configure_access(access, settings, settings["access_max_step_seconds"])
                if actual_settings != original["settings"]:
                    raise RuntimeError("Recovered scene Access settings differ from frozen original settings")
                baseline, _ = extract_pair(access, padded_start, padded_stop)
                original_intervals = sorted(zip(map(float, original["start_values"]), map(float, original["stop_values"])))
                reproduction = compare_intervals(original_intervals, baseline, tolerance=1e-7)
                if not reproduction["within_comparison_tolerance"]:
                    raise RuntimeError(f"Recovered scene does not reproduce the original 30-second report for {sat_id}/{site_id}")
                tight_settings = configure_access(access, settings, settings["spot_check_max_step_seconds"])
                tight_intervals, tight_report = extract_pair(access, padded_start, padded_stop)
                spot_records.append({"satellite_id": sat_id, "site_id": site_id, "baseline_settings_hash": original["settings_hash"],
                                     "recovery_30s_vs_original": reproduction, "tight_settings": tight_settings,
                                     "tight_report": tight_report, "comparison": compare_intervals(original_intervals, tight_intervals)})
                configure_access(access, settings, settings["access_max_step_seconds"])
                access.ComputeAccess()
                print(f"{specification['replicate_id']} save recovery fixed pair {len(spot_records)}/12", flush=True)
            write_json(output / "pair_counts.json", pair_records)
            write_json(output / "access_settings_actual.json", variants)
            write_json(output / "spot_check_15s.json", {"selection_rule": "12 fixed index pairs; recomputed during save recovery because the original pre-save metadata was not persisted",
                       "pairs": spot_records, "all_counts_match": all(s["comparison"]["interval_count_match"] for s in spot_records),
                       "all_within_5ms": all(s["comparison"]["within_comparison_tolerance"] for s in spot_records),
                       "all_original_30s_reproduced_within_100ns": True})
            bridge = save_scene_ascii_bridge(root, output, specification)
            write_json(output / "scene_save_bridge.json", bridge)
            after_hashes = {name: digest(output / name) for name in protected_files}
            if before_hashes != after_hashes:
                raise RuntimeError("Save recovery unexpectedly changed a protected original data file")
            manifest.update({"status": "success", "stk_version": version, "scenario": summary, "counts": counts,
                             "pair_count": len(original_reports), "spot_check_pair_count": len(spot_records),
                             "spot_check_all_counts_match": all(s["comparison"]["interval_count_match"] for s in spot_records),
                             "spot_check_all_within_5ms": all(s["comparison"]["within_comparison_tolerance"] for s in spot_records),
                             "earth_definition_path": str(EARTH_CB), "earth_definition_sha256": digest(EARTH_CB),
                             "earth_shape": "WGS84 installed central-body definition; no central-body overrides requested",
                             "access_settings_hashes": sorted(variants), "save_recovery_elapsed_seconds": time.monotonic() - started,
                             "scene_save_bridge": bridge, "protected_original_files_unchanged": before_hashes,
                             "recovery_actual_orbits_and_stations_match_original": True,
                             "recovery_access_pairs_recomputed": len(spot_records), "original_access_pairs_not_recomputed": 2004,
                             "utc_display_precision_note": "STK UTCG display uses milliseconds; relative endpoints retain 9 decimals and are authoritative",
                             "limitations": ["Printed fractional digits are serialization precision, not physical accuracy.",
                                             "Padded-boundary intervals are marked and excluded from the planning dataset.",
                                             "STK11 native save/load uses ASCII technical bridge; final scene and all dependencies reside in the requested data directory."]})
        manifest.pop("error", None)
        manifest.pop("counts_partial", None)
        manifest["outputs"] = [{"path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest(path)}
                               for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"]
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": "success", "recovered_scene": specification["scene_id"], "original_csv_and_raw_reports_unchanged": True, "counts": counts}), flush=True)
        return manifest
    except Exception as exc:
        manifest.update({"status": "save_recovery_failed", "error": str(exc), "save_recovery_elapsed_seconds": time.monotonic() - started})
        write_json(output / "manifest.json", manifest)
        raise


def generate_scene(parameters_path, parameters, specification, output_root):
    output = output_root / "raw_geometry" / specification["scene_id"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "scene_id": specification["scene_id"], "source_group": specification["source_group"],
                "replicate_id": specification["replicate_id"], "geometry_id": specification["geometry_id"], "split": specification["split"],
                "parameters_sha256": digest(parameters_path), "generator_sha256": digest(__file__),
                "scientific_contract": "new STK geometry; entire visibility contacts; reward=end-start; no slicing, min-duration filter, reweighting or copied historical contacts",
                "time_precision_note": "CSV uses 9 fractional decimal digits (at most 0.5 ns serialization error); raw_access_reports.jsonl retains original provider double values. Graph reward is the exact frozen CSV endpoint difference. This is not a claim of nanosecond physical accuracy.",
                "input_history": "none; only this frozen parameter JSON is consumed"}
    write_json(output / "manifest.json", manifest)
    settings = parameters["common_settings"]
    horizon = settings["planning_horizon_hours"] * 3600.0
    padding = settings["propagation_padding_hours_each_side"] * 3600.0
    padded_start, padded_stop = -padding, horizon + padding
    counts = {"raw_padded": 0, "full_contained": 0, "crossing": 0, "outside_planning": 0, "padding_boundary_clipped": 0}
    pair_records, spot_records, setting_variants = [], [], {}
    try:
        if not EARTH_CB.is_file():
            raise RuntimeError("Installed Earth central-body definition not found")
        earth_text = EARTH_CB.read_text(encoding="utf-8", errors="replace")
        if "WGS84" not in earth_text or "6.378137E6" not in earth_text:
            raise RuntimeError("Installed Earth definition does not match the required WGS84 ellipsoid")
        (output / "earth_definition_used.txt").write_text(earth_text, encoding="utf-8")
        with owned_stk11() as (root, version):
            summary = build_scene(root, parameters, specification, output)
            write_json(output / "scenario_summary.json", summary)
            root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")
            check_epoch = float(root.ConversionUtility.ConvertDate("UTCG", "EpSec", utcg(specification["epoch_utc"])))
            if abs(check_epoch) > 1e-8:
                raise RuntimeError(f"Scenario epoch is not planning start: {check_epoch}")
            selected_spots = fixed_spot_check_pairs(specification, parameters["stations"])
            with (output / "contacts.csv").open("w", newline="", encoding="utf-8-sig") as full_stream, \
                 (output / "boundary_contacts.csv").open("w", newline="", encoding="utf-8-sig") as boundary_stream, \
                 (output / "access_raw.csv").open("w", newline="", encoding="utf-8-sig") as raw_stream, \
                 (output / "raw_access_reports.jsonl").open("w", encoding="utf-8") as reports:
                full_writer = csv.DictWriter(full_stream, fieldnames=CONTACT_FIELDS)
                boundary_writer = csv.DictWriter(boundary_stream, fieldnames=RAW_FIELDS)
                raw_writer = csv.DictWriter(raw_stream, fieldnames=RAW_FIELDS)
                full_writer.writeheader(); boundary_writer.writeheader(); raw_writer.writeheader()
                pair_index = 0
                for station in parameters["stations"]:
                    facility = root.GetObjectFromPath(f"Facility/{station['site_id']}")
                    for satellite_spec in specification["satellites"]:
                        pair_index += 1
                        sat_id, site_id = satellite_spec["satellite_id"], station["site_id"]
                        satellite = root.GetObjectFromPath(f"Satellite/{sat_id}")
                        access = satellite.GetAccessToObject(facility)
                        actual_settings = configure_access(access, settings, settings["access_max_step_seconds"])
                        access_hash = canonical_hash(actual_settings)
                        setting_variants[access_hash] = actual_settings
                        intervals, report = extract_pair(access, padded_start, padded_stop)
                        reports.write(json.dumps({"satellite_id": sat_id, "site_id": site_id, "settings": actual_settings, "settings_hash": access_hash, **report}, allow_nan=False) + "\n")
                        pair_counts = {"satellite_id": sat_id, "site_id": site_id, "raw_intervals": len(intervals), "contained": 0, "crossing": 0, "access_settings_hash": access_hash}
                        for pass_index, (start, stop) in enumerate(intervals, 1):
                            classification = classify_interval(start, stop, horizon)
                            clipped = bool(abs(start - padded_start) < 1e-7 or abs(stop - padded_stop) < 1e-7)
                            start_text, stop_text = decimal_endpoint(start), decimal_endpoint(stop)
                            row = {key: specification[key] for key in ("source_group", "geometry_id", "replicate_id", "epoch_utc")}
                            row.update({"contact_id": f"{specification['scene_id']}:{site_id}:{sat_id}:P{pass_index:04d}",
                                        "pass_id": f"{site_id}:{sat_id}:P{pass_index:04d}", "satellite_id": sat_id, "site_id": site_id,
                                        "antenna_id": station["antenna_id"], "start_rel_seconds": start_text, "end_rel_seconds": stop_text,
                                        "duration_seconds": format(Decimal(stop_text) - Decimal(start_text), "f"),
                                        "start_utc": str(root.ConversionUtility.ConvertDate("EpSec", "UTCG", repr(start))),
                                        "end_utc": str(root.ConversionUtility.ConvertDate("EpSec", "UTCG", repr(stop))),
                                        "boundary_crossing": classification, "access_settings_hash": access_hash, "padding_boundary_clipped": clipped})
                            raw_writer.writerow(row)
                            counts["raw_padded"] += 1
                            counts["padding_boundary_clipped"] += int(clipped)
                            if classification == "none":
                                full_writer.writerow({key: row[key] for key in CONTACT_FIELDS})
                                counts["full_contained"] += 1
                                pair_counts["contained"] += 1
                            elif classification in {"left", "right", "both"}:
                                boundary_writer.writerow(row)
                                counts["crossing"] += 1
                                pair_counts["crossing"] += 1
                            else:
                                counts["outside_planning"] += 1
                        pair_records.append(pair_counts)
                        if (sat_id, site_id) in selected_spots:
                            tight_settings = configure_access(access, settings, settings["spot_check_max_step_seconds"])
                            tight_intervals, tight_report = extract_pair(access, padded_start, padded_stop)
                            spot_records.append({"satellite_id": sat_id, "site_id": site_id, "baseline_settings_hash": access_hash,
                                                 "tight_settings": tight_settings, "tight_report": tight_report,
                                                 "comparison": compare_intervals(intervals, tight_intervals)})
                            # Restore the primary frozen settings before saving the scene.
                            configure_access(access, settings, settings["access_max_step_seconds"])
                            access.ComputeAccess()
                        if pair_index == 1 or pair_index % 84 == 0:
                            print(f"{specification['replicate_id']} Access {pair_index}/2016; contained {counts['full_contained']}", flush=True)
                            full_stream.flush(); boundary_stream.flush(); raw_stream.flush(); reports.flush()
                            write_json(output / "progress.json", {"pair_count": pair_index, "counts": counts, "elapsed_seconds": time.monotonic() - started})
            write_json(output / "pair_counts.json", pair_records)
            write_json(output / "access_settings_actual.json", setting_variants)
            write_json(output / "spot_check_15s.json", {"selection_rule": "12 fixed index pairs, chosen before access execution; both shells and station-grid boundaries",
                       "pairs": spot_records, "all_counts_match": all(s["comparison"]["interval_count_match"] for s in spot_records),
                       "all_within_5ms": all(s["comparison"]["within_comparison_tolerance"] for s in spot_records)})
            if counts["full_contained"] == 0:
                raise RuntimeError("No full contacts returned; cannot label an empty dataset successful")
            bridge = save_scene_ascii_bridge(root, output, specification)
            write_json(output / "scene_save_bridge.json", bridge)
            manifest.update({"status": "success", "stk_version": version, "scenario": summary, "counts": counts,
                             "pair_count": len(pair_records), "spot_check_pair_count": len(spot_records),
                             "spot_check_all_counts_match": all(s["comparison"]["interval_count_match"] for s in spot_records),
                             "spot_check_all_within_5ms": all(s["comparison"]["within_comparison_tolerance"] for s in spot_records),
                             "earth_definition_path": str(EARTH_CB), "earth_definition_sha256": digest(EARTH_CB),
                             "earth_shape": "WGS84 installed central-body definition; no central-body overrides requested",
                             "constraints": "satellite LOS only; facility LOS and 15-degree minimum elevation only; terrain disabled; no sensor/RF/weather constraints",
                             "access_settings_hashes": sorted(setting_variants), "elapsed_seconds": time.monotonic() - started,
                             "scene_save_bridge": bridge,
                             "utc_display_precision_note": "STK UTCG display uses milliseconds; relative endpoints retain 9 decimals and are authoritative",
                             "limitations": ["Precise-event tolerance is 1 ms; printed fractional digits are serialization precision, not physical accuracy.",
                                             "Padded-boundary intervals can be clipped by the propagation horizon; those are marked and never admitted to the planning dataset.",
                                             "Installed WGS84 shape definition is preserved; J2 gravity/rotation defaults are retained and serialized in the editable scene."]})
        manifest["outputs"] = [{"path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest(path)}
                               for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"]
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": "success", "scene_id": specification["scene_id"], "output": str(output), "counts": counts}, ensure_ascii=False), flush=True)
        return manifest
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc), "elapsed_seconds": time.monotonic() - started, "counts_partial": counts})
        write_json(output / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replicates", nargs="+", default=["r000", "r001"])
    parser.add_argument("--validate-only", action="store_true", help="Check fixed input contracts without starting STK")
    parser.add_argument("--recover-scene", help="Recover this replicate's save-only failure; preserve its original contact files")
    args = parser.parse_args(argv)
    parameter_path = args.parameters.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    parameters = json.loads(parameter_path.read_text(encoding="utf-8-sig"))
    selected = validate_parameters(parameters, [args.recover_scene] if args.recover_scene else args.replicates)
    if args.validate_only:
        print(json.dumps({"status": "input_contract_valid", "scene_ids": [s["scene_id"] for s in selected],
                          "satellites_per_scene": [len(s["satellites"]) for s in selected], "sites": len(parameters["stations"]),
                          "parameters_sha256": digest(parameter_path), "will_start_com": False}))
        return 0
    for specification in selected:
        if args.recover_scene:
            recover_scene(parameter_path, parameters, specification, output_root)
        else:
            generate_scene(parameter_path, parameters, specification, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
