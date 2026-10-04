"""Read-only replay of the entire original-release exposed public40 grid.

Independent JSON/CSV/NumPy/byte checks; no Torch, upstream adapter, model,
graph generation, solver invocation or remote access. Core result replay and
external declared-byte provenance are separate conclusions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from math import fsum, isfinite
from pathlib import Path, PurePosixPath
import re
import tarfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/autodl-tmp/AAMAS1979_JointRecovery_v01"
RUNNER_SHA = "99e84929ca2ae0a2d260f65c1db92c7fb7e9d58195eb52af5069420c53c45682"
BUILD_SHA = "2dc9a4f5f10d2dfd5afc3526b761b0b597d9d1bedae4ccc5dc4ad314a36dfe9d"
PUBLIC_SHA = "411d462effc0d9a92b5da6d1cfe458115f8e31643ebd759c1aff7aa438cf8824"
CO_PROTOCOL_SHA = "ab62819d757170030b5cfd02769c40d4e3ca3cd236ebb466f38c3f50b2463216"
CATALOG_SHA = "629674a376f5574345b8297d77ed04e0318fa5c06cf2e38a1a9c3a774271a715"
CAPSULE = "0a05961ea2b928e17b18d514b87895bef0b9dd2af7e5cdb63fb2492e9f9ee06f"
NATIVE = ("CHILS-p1", "CHILS-p16-c16", "M2WIS", "M2WIS+s", "Struction-fast", "Struction-strong", "WeightedBR")
NEURAL = ("DIFUSCO-SAT-50x1", "DIFUSCO-SAT-50x4", "COExpander-SAT-CM-1x1")
SEEDS = (17, 29, 43)
INITIAL_SEEDS = {17: 101, 29: 202, 43: 303}
SECONDS = (.5, 2.)
FAMILIES = {"DIMACS": 16, "SATLIB-UF": 18, "SATLIB-CBS": 6}
EXECUTION = "primary_original_optimized_release"
SOURCES = ("experiments/v4_unit_public_baselines.py", "experiments/v4_unit_baseline_build.py",
    "experiments/v4_baseline_build.py", "experiments/v4_baseline_setup.py", "experiments/v4_published_weighted.py",
    "experiments/v4_weighted_br_guard.py", "experiments/v4_weighted_br_compat.py",
    "experiments/v3_published_baselines.py", "experiments/v3_coexpander_baseline.py",
    "src/joint_recovery/v4_public_data.py", "src/joint_recovery/__init__.py", "src/joint_recovery/core.py")
COMMITS = {"CHILS": "515952724cd3dcc6c4365a340ecf0f1da782119a",
    "KaMIS": "2e4b3861b8063f05e8520434b97ce59e21ca49e7",
    "DIFUSCO": "65eb3b7bde76097c0d3438ce743fa6769baf9ce5",
    "COExpander": "f77926950b8a7f20239eed748a4d8de9b14ea5ea"}
SAT_SHA = "360144ce5de7bc9dc972f0de1dc077f8c9b1abd2a7b800b833dfa32ebc5febd9"
CO_SHA = "3f5250daf3fcda477bb3ee9f048b7eef40b996813b5e66cec73cabf98dcb0f51"
CO_TREE_SHA = "61a0ac091c944f7a778b794eab89e90d527127803a04452546a07fc0adce309a"
DIFUSCO_FILES = {
    "pl_mis_model.py": "1ce4822e8c592c54afa5ba6a232931dfd1f554dff3d95b270e977b8a56efcc86",
    "pl_meta_model.py": "86790faf73a19ecb23aa02983d1298ddfbfeccc2c354431f5331c953a19d256a",
    "models/gnn_encoder.py": "2d78e25f625dcb798e0def733e90e146d86b87a72e34958f2769f0bde6c4a4b9",
    "models/nn.py": "d9547fb22b421a16dc079cb82a9d98630100b5e1b2a710d343f4d3f90b8661f3",
    "utils/diffusion_schedulers.py": "1f8c79c6425de814f3f9f9ab6760a42d363779f8c05f019f54a8bd807cef0c25",
    "utils/mis_utils.py": "c823ed0efca759945865060566d1040b28ed40d148cdef17f8de02f287213c24",
    "co_datasets/mis_dataset.py": "7db088412a445ec1ce94325e0fea76bbfb5b12a1a8d168a38e6cd520f5d8c85f",
    "utils/lr_schedulers.py": "550e5556ce2c96466a743f481bfaa00ac444742d2cdb9a43debe576b8b888238"}
CO_SETTINGS = dict(task="MIS", model="COExpanderCMModel", checkpoint_family="SATLIB", sparse_factor=1,
    hidden_dim=256, block_layers=[2, 4, 4, 2], inference_steps=1, determinate_steps=1, sampling_num=1,
    batch_size=1, beam_size=-1, use_rlsa=False)
CITATIONS = {
    "CHILS": ("grossmann2025chils", "https://doi.org/10.4230/LIPIcs.SEA.2025.22"),
    "M2WIS": ("grossmann2024mwis", "https://doi.org/10.7155/jgaa.v28i1.2997"),
    "Struction": ("gellner2021struction", "https://doi.org/10.1137/1.9781611976472.10"),
    "WeightedBR": ("lamm2019exact", "https://doi.org/10.1137/1.9781611975499.12"),
    "DIFUSCO": ("sun2023difusco", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html"),
    "COExpander": ("ma2025coexpander", "https://proceedings.mlr.press/v267/ma25r.html")}
BINARY_KEYS = {method: "chils" if method.startswith("CHILS") else "mmwis" if method.startswith("M2WIS")
               else "struction" if method.startswith("Struction") else "weighted_br" for method in NATIVE}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=invalid)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hex_sha(value):
    require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None, "Explicit lowercase SHA256 required")
    return value


def finite(value, message):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) and value >= 0, message)
    return float(value)


def exact(value, message):
    require(isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value) is not None, message)
    return int(value)


def child(root, relative):
    root = Path(root).resolve(); name = PurePosixPath(str(relative).replace("\\", "/"))
    require(not name.is_absolute() and name.parts and ".." not in name.parts, "Unsafe declared relative path")
    path = root.joinpath(*name.parts)
    require(root in path.resolve().parents and not path.is_symlink(), "Evidence path outside original directory")
    return path


class Paths:
    """Explicit path-prefix mapping only; never discover a file by basename."""
    def __init__(self, records=()):
        self.records = list(records)
        if not any(row["remote"].rstrip("/") == REMOTE for row in self.records):
            self.records.append(dict(remote=REMOTE, local=str(ROOT)))
        names = [row["remote"].rstrip("/") for row in self.records]
        require(len(names) == len(set(names)), "Duplicate remote path mapping")
        self.records.sort(key=lambda row: -len(row["remote"]))
        for row in self.records:
            require(PurePosixPath(row["remote"]).is_absolute() and ".." not in PurePosixPath(row["remote"]).parts,
                    "Explicit original absolute path prefix required")
    def resolve(self, original):
        require(isinstance(original, str) and ".." not in PurePosixPath(original).parts, "Unsafe original evidence path")
        for row in self.records:
            prefix = row["remote"].rstrip("/")
            if original == prefix:
                return Path(row["local"]).resolve()
            if original.startswith(prefix + "/"):
                return child(row["local"], original[len(prefix)+1:])
        raise FileNotFoundError("No explicit local mapping for original path: " + original)
    def bound(self, original, expected):
        path = self.resolve(original)
        require(path.is_file() and sha(path) == hex_sha(expected), "Required original dependency absent/hash differs: " + original)
        return path


def transport_check(collection, receipt_path, receipt_sha256, archive_path=None):
    require(sha(receipt_path) == hex_sha(receipt_sha256), "Caller-pinned fetch receipt changed")
    receipt = read(receipt_path); root = Path(receipt_path).resolve().parent
    require(receipt["capsule"] == CAPSULE and receipt["job"] == "v4_unit_public_original_02", "Wrong fixed root/capsule/job transport")
    archive = Path(archive_path) if archive_path else root / "TRANSPORT.tar.gz"
    require(sha(archive) == hex_sha(receipt["transport_archive_sha256"]), "Original transport archive bytes changed")
    bindings = {}; archived = set()
    for item in receipt["fetched"]:
        name = item["path"]
        require(name not in bindings and isinstance(item["bytes"], int) and not isinstance(item["bytes"], bool), "Duplicate/malformed fetched artifact")
        path = child(root, name)
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == hex_sha(item["sha256"]), "Fetched bytes/hash differ: " + name)
        bindings[name] = item
    with tarfile.open(archive, "r|gz") as stream:
        for item in stream:
            if item.isdir():
                continue
            require(item.isfile(), "Nonregular artifact in transport archive")
            name = PurePosixPath(item.name).as_posix()
            while name.startswith("./"):
                name = name[2:]
            require(name in bindings and name not in archived, "Undeclared/duplicate original transport member: " + name)
            digest = hashlib.sha256(); handle = stream.extractfile(item)
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
            require(item.size == bindings[name]["bytes"] and digest.hexdigest() == bindings[name]["sha256"], "Archived/extracted bytes differ: " + name)
            archived.add(name)
    require(set(bindings) == archived, "Fetched manifest does not cover the complete original transport archive")
    prefix = Path(collection).resolve().relative_to(root).as_posix() + "/"
    return {name[len(prefix):]: row for name, row in bindings.items() if name.startswith(prefix)}, receipt


def checked_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def mask(values, graph, message):
    values = np.asarray(values)
    require(values.shape == (graph["n"],) and values.dtype.kind in "biuf" and np.isfinite(values).all()
            and np.isin(values, (0, 1)).all(), message + ": exact binary membership required")
    result = values.astype(np.bool_)
    require(not np.any(result[graph["u"]] & result[graph["v"]]), message + ": original-edge conflict")
    return result, int(np.count_nonzero(result))


def original_graph(path, entry):
    require(sha(path) == entry["sha256"], "Original public NPZ bytes changed")
    arrays = checked_npz(path)
    fields = {"weights", "agents", "edge_u", "edge_v"}
    if entry["family"] in ("SATLIB-UF", "SATLIB-CBS"):
        fields.update(("clause_ids", "literals"))
    require(set(arrays) == fields, "Original public graph NPZ schema changed")
    w, owners, u, v = (arrays[k] for k in ("weights", "agents", "edge_u", "edge_v"))
    n = len(w)
    require(w.shape == (n,) and w.dtype.kind in "iuf" and np.isfinite(w).all() and np.all(w == 1)
            and 1 <= n <= 3195, "Original exact unit graph required; no heterogeneous coercion")
    require(owners.shape == (n,) and owners.dtype.kind in "iu", "Original owner vector changed")
    if "clause_ids" in arrays:
        require(all(arrays[key].shape == (n,) and arrays[key].dtype.kind in "iu" for key in ("clause_ids", "literals")), "Original SAT annotation vectors changed")
    require(u.ndim == v.ndim == 1 and u.shape == v.shape and u.dtype.kind in "iu" and v.dtype.kind in "iu"
            and np.all(u < v) and np.all(v < n), "Original edge endpoints malformed")
    keys = u.astype(np.int64)*n + v
    require(np.all(u >= 0) and np.all(keys[1:] > keys[:-1]) and entry["n"] == n and entry["m"] == len(u), "Original exact graph dimensions/order changed")
    content = hashlib.sha256(w.astype("<f8").tobytes() + owners.astype("<i8").tobytes() +
                             np.column_stack((u, v)).astype("<i8").tobytes()).hexdigest()
    require(content == entry["graph_content_sha256"], "Original semantic graph hash differs")
    return dict(n=n, m=len(u), weights=w, u=u.astype(np.int64), v=v.astype(np.int64))


def graph_inputs(protocol, paths):
    prior = read(paths.bound(protocol["original_public_protocol"]["path"], PUBLIC_SHA))
    co = read(paths.bound(protocol["original_co_protocol"]["path"], CO_PROTOCOL_SHA))
    require(protocol["original_public_protocol"]["sha256"] == PUBLIC_SHA and protocol["original_co_protocol"]["sha256"] == CO_PROTOCOL_SHA,
            "Original two public context receipts changed")
    catalog = read(paths.bound(co["public_manifest_path"], CATALOG_SHA))
    require(catalog["graph_count"] == 40 and catalog["preparation_complete"] is True and catalog["evaluation_outcomes_read"] == 0,
            "Original complete public catalog required")
    entries = protocol["graph_manifest"]
    require(len(entries) == 40 and len({g["graph_id"] for g in entries}) == 40
            and Counter(g["family"] for g in entries) == Counter(FAMILIES), "Entire fixed16/18/6 public40 cohort required")
    require([g["graph_id"] for g in entries] == [g["instance_id"] for g in prior["graph_manifest"]], "Original graph registry replaced/reordered")
    co_map = {g["instance"]: g for g in co["graphs"]}; catalog_map = {g["instance_id"]: g for g in catalog["entries"]}
    result = {}
    for entry, old in zip(entries, prior["graph_manifest"]):
        gid = entry["graph_id"]; neural = co_map[gid]; declared = catalog_map[gid]
        for key in ("family", "n", "m", "graph_content_sha256"):
            require(entry[key] == old[key] == neural[key] == declared[key], "Original public graph context differs: " + key)
        require(entry["sha256"] == old["graph_npz_sha256"] == neural["graph_npz_sha256"] == declared["graph_npz_sha256"], "Original graph byte identities differ")
        graph = original_graph(paths.bound(entry["path"], entry["sha256"]), entry)
        require(set(entry["initials"]) == {"101", "202", "303"}, "All three original shared initials required")
        initials = {}
        for seed in SEEDS:
            initial = entry["initials"][str(INITIAL_SEEDS[seed])]; prior_initial = neural["initials"][str(INITIAL_SEEDS[seed])]
            require(initial["sha256"] == prior_initial["sha256"] and initial["objective"] == prior_initial["objective"], "Original shared initial receipt changed")
            archive = checked_npz(paths.bound(initial["path"], initial["sha256"]))
            require("selected" in archive, "Missing original initial membership")
            selected, value = mask(archive["selected"], graph, "Original initial")
            require(value == initial["objective"], "Original initial objective differs")
            initials[seed] = (selected, value)
        result[gid] = dict(entry=entry, graph=graph, initials=initials)
    return result


def source_snapshots(protocol, collection):
    records = protocol["capsule"]["files"]
    require(protocol["capsule"]["capsule"] == CAPSULE == sha_json(records)
            and len({r["path"] for r in records}) == len(records), "Original scientific capsule differs")
    source = protocol["source_sha256"]
    require(set(source) == set(SOURCES) and source[SOURCES[0]] == RUNNER_SHA, "Original complete active wrapper source map differs")
    manifest = {row["path"]: row for row in records}
    for name, expected in source.items():
        path = child(collection, "source_snapshot/" + name)
        require(name in manifest and manifest[name]["sha256"] == expected and path.is_file()
                and sha(path) == expected and path.stat().st_size == manifest[name]["bytes"], "Required frozen active source snapshot differs: " + name)
    return len(source)


def citation(method):
    return CITATIONS[next(prefix for prefix in CITATIONS if method.startswith(prefix))]


def official_receipts(protocol, paths):
    require(protocol["native_build"]["sha256"] == BUILD_SHA and protocol["native_execution_mode"] == EXECUTION, "Primary original optimized build required")
    build_path = paths.bound(protocol["native_build"]["path"], BUILD_SHA); build = read(build_path)
    require(build["status"].startswith("BUILD_PASS") and build["numeric_port"] is False and set(build["binaries"]) == {"chils", "weighted_br", "mmwis", "struction"}, "Original release declarations changed")
    for name in ("CHILS", "KaMIS"):
        require(build["source_closure"][name]["commit"] == COMMITS[name], "Original native source commit differs")
    guard_path = paths.bound(protocol["native_guard"]["path"], protocol["native_guard"]["sha256"])
    guard = read(guard_path); guard_protocol = read(guard_path.parent / "protocol.json"); complete = read(guard_path.parent / "completion.json")
    require(guard["status"] == "PASS" and guard["rows"] == 63 and guard["native_build_sha256"] == BUILD_SHA
            and guard["guard_source_sha256"] == RUNNER_SHA and guard["native_execution_mode"] == EXECUTION
            and guard["performance_comparison"] is False and guard["public_graphs_opened"] is False
            and all(guard[k] is True for k in ("all_original_graph_feasible", "all_small_fixture_optima_matched", "no_runtime_errors_observed")), "Original finite native guard declaration differs")
    require(guard["protocol_sha256"] == sha(guard_path.parent / "protocol.json") and complete["guard_sha256"] == sha(guard_path)
            and complete["status"] == "complete_original_unit_native_guards" and complete["rows"] == 63
            and guard_protocol["source_sha256"] == RUNNER_SHA and guard_protocol["native_build_sha256"] == BUILD_SHA,
            "Native guard receipts no longer close")
    fixtures = {
        "unit-path7": (7, [(i, i+1) for i in range(6)], 4),
        "unit-two-triangles": (6, [(0,1), (0,2), (1,2), (3,4), (3,5), (4,5)], 2),
        "unit-cycle9": (9, [(i, i+1) for i in range(8)] + [(0,8)], 4)}
    expected = {(g, m, s) for g in fixtures for m in NATIVE for s in SEEDS}; keys = set()
    for row in guard["records"]:
        key = (row["graph_id"], row["method"], row["seed"])
        require(key in expected and key not in keys, "Native63 guard coverage differs"); keys.add(key)
        n, edges, optimum = fixtures[key[0]]; graph = dict(n=n, u=np.asarray([u for u,v in edges]), v=np.asarray([v for u,v in edges]))
        folder = guard_path.parent / "native" / ("%s_%s_%s" % key)
        output = folder / "returned.txt"
        require(sha(output) == row["raw_output_sha256"] and sha(folder/"stdout.txt") == row["stdout_sha256"]
                and sha(folder/"stderr.txt") == row["stderr_sha256"], "Actual guard raw/log bytes differ")
        selected, value = text_membership(output, graph, key[1])
        require(row["status"] == "RETURNED_VALID" and row["returned_feasible"] is True and row["exact_optimum_matched"] is True
                and row["returncode"] == 0 and row["watchdog_timeout"] is False and row["ubsan_runtime_failed"] is False
                and value == optimum == exact(row["raw_value_exact"], "Guard exact raw") == exact(row["exhaustive_optimum_exact"], "Guard optimum")
                and row["binary_sha256"] == build["binaries"][BINARY_KEYS[key[1]]]["sha256"], "Actual finite native guard return differs")
    require(keys == expected and len(guard["records"]) == 63, "Native guard grid incomplete")
    configs, sources = protocol["neural_configs"], protocol["neural_source_closure"]
    require(set(configs) == set(sources) == set(NEURAL), "All three fixed released neural workpoints required")
    for method in NEURAL:
        config, closure = configs[method], sources[method]; difusco = method.startswith("DIFUSCO")
        require(config["device"] == "cuda" and config["steps"] == (50 if difusco else 1)
                and config["sequential_samples"] == (4 if method.endswith("50x4") else 1) and config["parallel_samples"] == 1,
                "Released fixed sampling recipe changed")
        require(closure["commit"] == COMMITS["DIFUSCO" if difusco else "COExpander"] and closure["checkpoint_sha256"] == (SAT_SHA if difusco else CO_SHA), "Official released neural identity changed")
        require(closure["source_sha256_lf"] == DIFUSCO_FILES if difusco else len(closure["source_sha256_lf"]) == 69 and sha_json(closure["source_sha256_lf"]) == CO_TREE_SHA, "Official active neural source map changed")
        hex_sha(closure["python_sha256"])
    return build


def text_membership(path, graph, method):
    text = Path(path).read_text(encoding="ascii"); rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        require(re.fullmatch(r"[+-]?[0-9]+", line.strip()) is not None, "Malformed original native membership row")
        rows.append(int(line.strip()))
    if method.startswith("CHILS"):
        require(len(rows) == len(set(rows)) and all(1 <= v <= graph["n"] for v in rows), "Native original vertex IDs invalid")
        values = np.zeros(graph["n"], dtype=np.bool_); values[np.asarray(rows, dtype=np.int64)-1] = True
    else:
        require(len(rows) == graph["n"] and all(value in (0,1) for value in rows), "Native original binary vector invalid")
        values = np.asarray(rows, dtype=np.int64)
    return mask(values, graph, "Native text return")


def native_command(method, binary, folder, seconds, seed):
    graph = str(PurePosixPath(folder)/"native.graph"); output = str(PurePosixPath(folder)/"returned.txt")
    if method.startswith("CHILS"):
        return [binary, "-g", graph, "-p", "1" if method == "CHILS-p1" else "16", "-c",
                "16" if method == "CHILS-p16-c16" else "1", "-r", str(seed), "-t", str(seconds), "-o", output,
                "-i", str(PurePosixPath(folder)/"initial.ids")]
    args = [binary, graph, "--weight_source=file", "--seed="+str(seed), "--time_limit="+str(seconds), "--output="+output]
    if method.startswith("M2WIS"):
        args.append("--config=" + ("mmwiss" if method.endswith("+s") else "mmwis"))
    if method == "Struction-strong":
        args.append("--cyclicStrong")
    return args


def metis_bytes(graph):
    adjacent = [[] for _ in range(graph["n"])]
    for u,v in zip(graph["u"],graph["v"]):
        adjacent[int(u)].append(int(v)+1); adjacent[int(v)].append(int(u)+1)
    lines = ["%d %d 10" % (graph["n"],graph["m"])]
    lines.extend(" ".join(map(str,[1]+sorted(neighbors))) for neighbors in adjacent)
    return ("\n".join(lines)+"\n").encode("ascii")


def metis_check(path, graph):
    require(Path(path).read_bytes() == metis_bytes(graph), "Native export changed original unit edges/rewards")


def row_key(row):
    require(isinstance(row["seed"], int) and not isinstance(row["seed"], bool) and row["seed"] in SEEDS, "Exact declared native/neural seed required")
    return row["graph_id"], row["method"], row["seed"], row["search_seconds_requested"]


def unit_name(key):
    gid, method, seed, seconds = key
    return "%s_%s_seed%d" % (gid, method, seed) + ("_t%s" % seconds if seconds is not None else "")


def expected_grid(entries):
    return {(g["graph_id"],m,s,t) for g in entries for m in NATIVE for s in SEEDS for t in SECONDS} | {
        (g["graph_id"],m,s,None) for g in entries for m in NEURAL for s in SEEDS}


def worker_setup(collection, method, protocol):
    folder = child(collection, "workers/"+method); setup_path = folder/"setup.json"
    if not setup_path.is_file():
        failure = child(collection, "workers/"+method+"_startup_failure.json")
        require(failure.is_file() and isinstance(read(failure)["error"], str), "Resident neither setup nor explicit startup failure")
        return dict(status="startup_failed", model_setup_receipt_sha256=None, cold_parent_setup_seconds=None)
    record = read(setup_path); ready = record["ready"]; setup = ready["setup"]
    require(ready["status"] == "ready" and setup["checkpoint_sha256"] == protocol["neural_source_closure"][method]["checkpoint_sha256"], "Resident official setup receipt differs")
    config = protocol["neural_configs"][method]
    argv = record["command"]
    require(len(argv) == 12 and argv[1].endswith("/experiments/v4_unit_public_baselines.py") and argv[5].endswith("/workers/"+method)
            and argv == [config["python"], argv[1], "--mode", "neural-worker", "--out", argv[5],
                         "--method",method,"--checkout",config["checkout"],"--checkpoint",config["checkpoint"]],
            "Resident setup command recipe changed")
    if method.startswith("DIFUSCO"):
        require(setup["source_commit"] == COMMITS["DIFUSCO"] and setup["verified_source_sha256_lf"] == DIFUSCO_FILES
                and setup["device"] == "cuda" and setup["diffusion_type"] == "categorical" and setup["inference_steps"] == 50
                and setup["sequential_samples"] == config["sequential_samples"] and setup["parallel_samples"] == 1, "Released DIFUSCO loaded setup differs")
    else:
        require(setup["official_commit"] == COMMITS["COExpander"] and setup["source_tree_sha256_lf"] == CO_TREE_SHA
                and setup["source_sha256_lf"] == protocol["neural_source_closure"][method]["source_sha256_lf"]
                and setup["settings"] == CO_SETTINGS and setup["official_strict_state_load"] is True
                and setup["warm_start_supported"] is False and setup["device"] == "cuda", "Released COExpander loaded setup differs")
    return dict(status="loaded_official_fixed_workpoint", model_setup_receipt_sha256=sha(setup_path),
        cold_parent_setup_seconds=finite(record["cold_parent_setup_seconds"], "Once-only parent resident setup"),
        worker_setup_seconds=finite(ready["setup_wall_seconds"], "Once-only worker setup"), setup=setup)


def replay_unit(row, item, collection, build, protocol, setup):
    key = row_key(row); gid, method, seed, seconds = key; name = unit_name(key)
    graph = item["graph"]; initial, initial_value = item["initials"][seed]
    require(row["family"] == item["entry"]["family"] and row["n"] == graph["n"] and row["m"] == graph["m"]
            and row["initial_seed"] == INITIAL_SEEDS[seed] and exact(row["initial_value_exact"], "Initial exact cardinality") == initial_value,
            "Original row graph/shared-initial context changed")
    path = child(collection, row["membership_path"])
    require(row["membership_path"] == "memberships/"+name+".npz" and sha(path) == row["membership_sha256"], "Committed membership identity changed")
    archive = checked_npz(path)
    require({"initial", "accepted"} <= set(archive) <= {"initial", "accepted", "raw"}, "Committed initial/accepted/raw schema changed")
    replay_initial, _ = mask(archive["initial"], graph, "Committed initial")
    require(np.array_equal(replay_initial, initial), "A method borrowed a different original shared initial")
    accepted, accepted_value = mask(archive["accepted"], graph, "Accepted return")
    raw, raw_value = mask(archive["raw"], graph, "Stored raw return") if "raw" in archive else (None, None)
    success = row["status"] == "RETURNED_VALID"
    require(row["accepted_feasible"] is True and exact(row["accepted_value_exact"], "Accepted exact") == accepted_value
            and exact(row["gain_exact"], "Exact gain") == accepted_value-initial_value and accepted_value >= initial_value,
            "Original accepted cardinality/gain differs")
    expected = raw if success and raw is not None and raw_value > initial_value else initial
    require(np.array_equal(accepted, expected), "Failed/nonimproving output bypassed original incumbent fallback")
    if success:
        require(raw is not None and row["returned_membership"] is True and row["returned_feasible"] is True
                and exact(row["raw_value_exact"], "Exact raw") == raw_value, "Successful actual raw return differs")
    else:
        require(row["status"] in ("FAILED_PROCESS", "NO_RETURNED_MEMBERSHIP", "FAILED_ADAPTER", "FAILED_NEURAL"), "Unknown unsupported/status cannot disappear from grid")
        require(row.get("raw_value_exact") is None or raw_value == exact(row["raw_value_exact"], "Failed diagnostic raw"), "Failed raw diagnostic differs")
        require(row["status"] == "NO_RETURNED_MEMBERSHIP" or isinstance(row.get("reason", ""), str), "Explicit failure reason must be serializable")
    cold = resident = worker = None; within = None; physical_raw_value = None; physical_raw_invalid = False
    if method in NATIVE:
        require(seconds in SECONDS, "Undeclared native soft-search setting")
        if row["status"] != "FAILED_ADAPTER":
            require(row["native_execution_mode"] == EXECUTION and row["instrumented_original_source"] is False
                    and row["instrumentation_overhead"] is False and row["weight_scale"] == 1
                    and row["matched_repair_budget"] is False and row["native_warmstart"] == method.startswith("CHILS")
                    and row["configured_threads"] == (16 if method == "CHILS-p16-c16" else 1)
                    and row["same_CPU_primary"] == (method == "CHILS-p1") and row["citation"] == citation(method)[1], "Native original release/configuration claim changed")
            require(all(isinstance(row[field],bool) for field in ("returned_membership","returned_feasible","watchdog_timeout","ubsan_runtime_failed","within_requested_search_time"))
                    and row["returned_membership"] == row["returned_feasible"] == (raw is not None)
                    and isinstance(row["returncode"],int) and not isinstance(row["returncode"],bool), "Exact native return/failure flags required")
            folder = child(collection, "native/"+name); binary = build["binaries"][BINARY_KEYS[method]]
            require(row["binary_sha256"] == binary["sha256"] and isinstance(row["command"], list), "Native binary declaration differs")
            argv = row["command"]; remote_folder = str(PurePosixPath(argv[2] if method.startswith("CHILS") else argv[1]).parent)
            require(argv == native_command(method, binary["path"], remote_folder, seconds, seed), "Native original command changed")
            require(hashlib.sha256(metis_bytes(graph)).hexdigest() == row["metis_sha256"] and sha(folder/"stdout.txt") == row["stdout_sha256"]
                    and sha(folder/"stderr.txt") == row["stderr_sha256"], "Native input/log hashes differ")
            if (folder/"native.graph").is_file():
                metis_check(folder/"native.graph", graph)
            if method.startswith("CHILS"):
                selected, _ = text_membership(folder/"initial.ids", graph, method)
                require(np.array_equal(selected, initial), "Native warmstart differs from original common initial")
            stderr = (folder/"stderr.txt").read_bytes()
            require(row["ubsan_runtime_failed"] == (b"runtime error:" in stderr), "Actual runtime failure ledger differs")
            failed = row["watchdog_timeout"] or row["returncode"] != 0 or row["ubsan_runtime_failed"]
            output = folder/"returned.txt"; parsed = None
            if output.is_file():
                require(sha(output) == row["raw_output_sha256"], "Native raw output byte hash changed")
                try:
                    parsed, parsed_value = text_membership(output, graph, method)
                    physical_raw_value = parsed_value
                except ValueError:
                    failed = True; physical_raw_invalid = True
            require((parsed is None) == (raw is None) and (raw is None or np.array_equal(parsed, raw)), "Native text and committed raw mask differ")
            require(row["status"] == ("FAILED_PROCESS" if failed else "RETURNED_VALID" if raw is not None else "NO_RETURNED_MEMBERSHIP"), "Native return/failure status differs")
            cold = finite(row["cold_wall_seconds"], "Native cold parent wall")
            require(cold == row["adapter_seconds"] and row["within_requested_search_time"] == (cold <= seconds)
                    and row["ontime_value_exact"] == (str(accepted_value) if cold <= seconds else None), "Native cold/soft-search timing semantics changed")
            finite(row["process_seconds"], "Native subprocess cost"); within = cold <= seconds
    else:
        require(method in NEURAL and seconds is None and row["weight_scale"] == 1 and row["configured_threads"] == 1
                and row["native_warmstart"] is False and row["matched_repair_budget"] is False
                and row["neural_workpoint"] == method and row["cold_wall_seconds"] is None and row["ontime_value_exact"] is None,
                "Fixed resident workpoint obtained invented timing/initial semantics")
        resident = finite(row["resident_parent_seconds"], "Actual resident parent query cost")
        folder = child(collection, "neural/"+name)
        if (folder/"raw.npz").is_file():
            try:
                returned = checked_npz(folder/"raw.npz")
                require(set(returned) == {"selected"}, "Official worker output schema changed")
                selected, physical_raw_value = mask(returned["selected"], graph, "Physical resident raw return")
                require(raw is None or np.array_equal(selected,raw), "Physical and committed raw mask differ")
            except Exception:
                physical_raw_invalid = True
                require(not success and raw is None, "Successful/committed raw is not its physical artifact")
        if success:
            require(setup["status"] == "loaded_official_fixed_workpoint" and row["model_setup_receipt_sha256"] == setup["model_setup_receipt_sha256"], "Neural query uses another setup")
            arrays = checked_npz(folder/"input.npz")
            require(set(arrays) == {"weights", "edge_u", "edge_v"} and np.array_equal(arrays["weights"], graph["weights"])
                    and np.array_equal(arrays["edge_u"], graph["u"]) and np.array_equal(arrays["edge_v"], graph["v"]), "Neural resident got a changed original graph")
            response = row["worker_response"]
            require(response["status"] == "ok" and PurePosixPath(response["output"]).as_posix().endswith("/neural/"+name+"/raw.npz")
                    and response["details_path"] == response["output"]+".json"
                    and sha(folder/"raw.npz") == response["output_sha256"] and sha(folder/"raw.npz.json") == response["details_sha256"], "Resident returned artifact bytes differ")
            returned = checked_npz(folder/"raw.npz"); require(set(returned) == {"selected"}, "Official worker output schema changed")
            selected, value = mask(returned["selected"], graph, "Official resident return")
            details = read(folder/"raw.npz.json")
            require(np.array_equal(selected, raw) and value == raw_value == details["raw_value"] and details == row["upstream_details"], "Resident output/details/committed mask differ")
            require(details["seed"] == seed and details.get("native_warmstart", False) is False, "Neural detail seed/initial contract changed")
            require(details["accepted_value"] == raw_value, "Official from-scratch worker retained an undisclosed warm start")
            if method.startswith("COExpander"):
                require(details["native_warm_start"] is False and details["initial_value"] == 0
                        and details["original_network_called"] is True and details["original_solver"] == "COExpanderMISSolver.solve"
                        and details["vertices"] == graph["n"] and details["undirected_edges"] == graph["m"]
                        and details["settings"] == CO_SETTINGS, "Official COExpander query procedure changed")
            for field, expected_value in setup["setup"].items():
                if field in details:
                    require(details[field] == expected_value, "Neural returned setup differs: " + field)
            worker = finite(response["worker_load_solve_write_seconds"], "Resident worker input/inference/persistence cost")
    return dict(graph_id=gid, family=row["family"], method=method, seed=seed, soft_seconds=seconds, status=row["status"],
        initial_value=initial_value, accepted_value=accepted_value, feasible_raw_value=raw_value,
        successful_raw_value=raw_value if success else None, gain=accepted_value-initial_value,
        relative_gain=float((accepted_value-initial_value)/max(1,initial_value)), cold_wall_seconds=cold,
        resident_parent_seconds=resident, worker_load_solve_write_seconds=worker,
        physical_raw_artifact_value=physical_raw_value, physical_raw_artifact_invalid=physical_raw_invalid,
        within_soft_search_wall=within, failure_or_no_return=not success, fallback_to_initial=np.array_equal(accepted, initial))


def provenance(protocol, build, paths):
    records = []
    def check(original, expected, kind, normalized=False, size=None):
        status = "missing_local_declared_bytes"; actual = None
        try:
            path = paths.resolve(original)
            if path.is_file():
                actual = hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest() if normalized else sha(path)
                status = "PASS_declared_bytes" if actual == expected and (size is None or path.stat().st_size == size) else "FAIL_declared_bytes"
        except (FileNotFoundError, OSError):
            pass
        records.append(dict(kind=kind, original_path=original, declared_sha256=expected, actual_sha256=actual, status=status))
    for key, binary in build["binaries"].items():
        check(binary["path"], binary["sha256"], "native_binary_"+key, size=binary["bytes"])
    source_root = str(PurePosixPath(protocol["native_build"]["path"]).parent/"sources")
    for repo, closure in build["source_closure"].items():
        for name, expected in closure["tracked_file_sha256"].items():
            check(str(PurePosixPath(source_root)/repo/name), expected, "native_tracked_source_"+repo)
    seen = set()
    for method, config in protocol["neural_configs"].items():
        closure = protocol["neural_source_closure"][method]
        declarations = [(config["checkpoint"], closure["checkpoint_sha256"], "released_neural_checkpoint", False),
                        (config["python"], closure["python_sha256"], "declared_python_executable", False)]
        base = PurePosixPath(config["checkout"])/"difusco" if method.startswith("DIFUSCO") else PurePosixPath(config["checkout"])
        declarations += [(str(base/name), expected, "official_neural_active_source", True) for name, expected in closure["source_sha256_lf"].items()]
        for original, expected, kind, normalized in declarations:
            if (original,expected,kind) not in seen:
                seen.add((original,expected,kind)); check(original,expected,kind,normalized)
    counts = Counter(row["status"] for row in records)
    return dict(status="PASS_all_declared_external_bytes" if counts["PASS_declared_bytes"] == len(records)
                else "INCOMPLETE_or_mismatched_external_byte_provenance_not_environment_certificate",
        counts=dict(counts), declarations=records, dependency_scope="Only predeclared source/checkpoint/binary/python hashes; no Torch/CUDA/OS dependency expansion or environment certification")


def average_complete(values):
    return fsum(values)/len(values) if values and all(value is not None for value in values) else None


def aggregate(records, entries, setups):
    require(len(records) == 2040 and {row_key(dict(graph_id=r["graph_id"],method=r["method"],seed=r["seed"],search_seconds_requested=r["soft_seconds"])) for r in records} == expected_grid(entries), "Full independent2040 denominator required")
    configs = [(m,t) for m in NATIVE for t in SECONDS] + [(m,None) for m in NEURAL]
    tables = []; graph_rows = []
    metrics = ("accepted_value", "successful_raw_value", "feasible_raw_value", "relative_gain", "cold_wall_seconds", "resident_parent_seconds", "worker_load_solve_write_seconds")
    for method, seconds in configs:
        summaries = []
        for entry in entries:
            rows = [r for r in records if r["graph_id"] == entry["graph_id"] and r["method"] == method and r["soft_seconds"] == seconds]
            require(len(rows) == 3 and {r["seed"] for r in rows} == set(SEEDS), "Every graph/config must retain all three seeds")
            graph = dict(graph_id=entry["graph_id"], family=entry["family"], method=method, soft_seconds=seconds,
                status_counts=dict(Counter(r["status"] for r in rows)), failure_or_no_return=sum(r["failure_or_no_return"] for r in rows),
                raw_available_seeds=sum(r["successful_raw_value"] is not None for r in rows),
                diagnostic_raw_available_seeds=sum(r["feasible_raw_value"] is not None for r in rows),
                fallbacks=sum(r["fallback_to_initial"] for r in rows), seeds=list(SEEDS),
                cold_wall_within_requested_soft_search=sum(r.get("within_soft_search_wall") is True for r in rows),
                cold_wall_soft_search_unknown=sum(method in NATIVE and r.get("within_soft_search_wall") is None for r in rows))
            graph.update({metric: average_complete([r[metric] for r in rows]) for metric in metrics})
            summaries.append(graph); graph_rows.append(graph)
        families = {}
        for family, count in FAMILIES.items():
            group = [r for r in summaries if r["family"] == family]
            require(len(group) == count, "Original family denominator changed")
            families[family] = dict(original_graphs=count, seeds_per_graph=3,
                failure_or_no_return=sum(r["failure_or_no_return"] for r in group),
                raw_available_seeds=sum(r["raw_available_seeds"] for r in group),
                diagnostic_raw_available_seeds=sum(r["diagnostic_raw_available_seeds"] for r in group),
                raw_incomplete_graphs=sum(r["successful_raw_value"] is None for r in group),
                fallbacks=sum(r["fallbacks"] for r in group),
                cold_wall_within_requested_soft_search=sum(r["cold_wall_within_requested_soft_search"] for r in group),
                cold_wall_soft_search_unknown=sum(r["cold_wall_soft_search_unknown"] for r in group),
                **{metric: average_complete([r[metric] for r in group]) for metric in metrics})
        overall = {metric: average_complete([families[family][metric] for family in FAMILIES]) for metric in metrics}
        overall.update(original_graphs=40, equal_family_weights={family: 1/3 for family in FAMILIES},
            failure_or_no_return=sum(r["failure_or_no_return"] for r in summaries), raw_available_seeds=sum(r["raw_available_seeds"] for r in summaries),
            diagnostic_raw_available_seeds=sum(r["diagnostic_raw_available_seeds"] for r in summaries),
            raw_incomplete_graphs=sum(r["successful_raw_value"] is None for r in summaries), fallbacks=sum(r["fallbacks"] for r in summaries),
            cold_wall_within_requested_soft_search=sum(r["cold_wall_within_requested_soft_search"] for r in summaries),
            cold_wall_soft_search_unknown=sum(r["cold_wall_soft_search_unknown"] for r in summaries))
        bibkey, link = citation(method)
        tables.append(dict(method=method, soft_seconds=seconds, bibtex_key=bibkey, published_reference=link,
            resource_configuration="16 CPU threads; separate multicore workpoint" if method == "CHILS-p16-c16" else "one CPU" if method in NATIVE else "one resident CUDA worker; fixed released SAT workpoint",
            timing_scope="native cold parent wall vs requested soft search" if method in NATIVE else "resident parent query; once-only model setup separately",
            once_only_setup_parent_seconds=setups.get(method,{}).get("cold_parent_setup_seconds"),
            once_only_setup_worker_seconds=setups.get(method,{}).get("worker_setup_seconds"),
            once_only_resident_setup=setups.get(method), families=families, overall_equal_family=overall))
    require(len(tables) == 17, "All14 native soft settings + three neural workpoints required")
    return dict(tables=tables, graph_summaries=graph_rows, independent_unit="original graph; three seeds averaged within graph; equal families overall",
        missing_raw_handling="Raw mean is null whenever an original seed/graph has no successful feasible raw return; never impute zero or reweight successful subset",
        accepted_handling="All original graph/seed slots retained with original incumbent fallback on failed/nonimproving return",
        matched_total_deadline_or_SOTA_claim=False, held_out_confirmation=False, checkpoint_overlap="unverified", resource_scopes_mixed_into_speedup=False)


def csv_check(path, unit_rows):
    with Path(path).open(encoding="utf8", newline="") as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    require(len(rows) == 2040, "Summary CSV missing declared rows")
    expected = {}
    for row in unit_rows:
        key = unit_name(row_key(row)); expected[key] = row
    seen = set()
    for record in rows:
        seconds = None if record["search_seconds_requested"] == "" else float(record["search_seconds_requested"])
        key = unit_name((record["graph_id"], record["method"], int(record["seed"]), seconds))
        require(key in expected and key not in seen, "Summary CSV substituted/repeated an original row"); seen.add(key)
        source = expected[key]
        for field in fields:
            value = source.get(field)
            text = "" if value is None else json.dumps(value,sort_keys=True) if isinstance(value,(dict,list,tuple)) else str(value)
            require(record[field] == text, "Summary CSV/unit JSON differs: " + field)
    require(seen == set(expected), "Summary CSV coverage differs")


def analyze(collection, protocol_sha256, completion_sha256, fetch_receipt, fetch_sha256, paths, archive=None):
    collection = Path(collection).resolve()
    # Read no partial baseline results before full completion, pinned grid and transport.
    require(sha(collection/"protocol.json") == hex_sha(protocol_sha256) and sha(collection/"completion.json") == hex_sha(completion_sha256), "Caller-pinned full protocol/completion changed")
    protocol = read(collection/"protocol.json"); complete = read(collection/"completion.json")
    require(complete["status"] == "complete_exposed40_published_baselines_diagnostic_only" and complete["rows"] == complete["expected_rows"] == 2040
            and complete["protocol_sha256"] == protocol_sha256 and complete["new_confirmation_inputs_opened"] is False, "Entire completed2040 diagnostic registry required before result reading")
    require(protocol["native_methods"] == list(NATIVE) and protocol["neural_methods"] == list(NEURAL) and protocol["seeds"] == list(SEEDS)
            and protocol["initial_seed_map"] == {str(k):v for k,v in INITIAL_SEEDS.items()} and protocol["native_search_seconds"] == list(SECONDS)
            and protocol["expected_rows"] == 2040 and protocol["expected_native_rows"] == 1680 and protocol["expected_neural_rows"] == 360
            and protocol["families"] == FAMILIES and protocol["no_download"] is True and protocol["no_fit_tune_or_selection"] is True,
            "Fixed original public configurations/counts changed")
    bindings, receipt = transport_check(collection, fetch_receipt, fetch_sha256, archive)
    require(all(name in bindings for name in ("protocol.json","completion.json","summaries.csv","opening.json","freeze_completion.json","native_phase_completion.json")), "Complete frozen/opened/finished phase artifacts missing from original transport")
    require(sha(collection/"summaries.csv") == complete["summary_sha256"], "Full summary bytes changed")
    freeze = read(collection/"freeze_completion.json"); opening = read(collection/"opening.json"); native_done = read(collection/"native_phase_completion.json")
    require(freeze["status"] == "complete_exposed40_preparation_no_solver_run" and freeze["expected_rows"] == 2040 and freeze["protocol_sha256"] == protocol_sha256
            and opening["protocol_sha256"] == protocol_sha256 and opening["new_confirmation_inputs_opened"] is False and opening["no_fit_tune"] is True
            and native_done["status"] == "complete_native_coverage" and native_done["rows"] == native_done["expected_rows"] == 1680, "Frozen original phases do not close")
    source_count = source_snapshots(protocol, collection); inputs = graph_inputs(protocol, paths); build = official_receipts(protocol, paths)
    setups = {method: worker_setup(collection,method,protocol) for method in NEURAL}
    expected = expected_grid(protocol["graph_manifest"]); unit_files = {name for name in bindings if name.startswith("units/") and name.endswith(".json")}
    require(unit_files == {"units/"+unit_name(key)+".json" for key in expected}, "Exact2040 original per-unit artifacts required")
    original_rows = []; records = []
    for key in sorted(expected, key=lambda k: (k[0],k[1],k[2],-1 if k[3] is None else k[3])):
        row = read(collection/"units"/(unit_name(key)+".json")); require(row_key(row) == key, "Original unit name/context differs")
        require(row["membership_path"] in bindings, "Committed membership absent from full transport")
        original_rows.append(row); records.append(replay_unit(row,inputs[key[0]],collection,build,protocol,setups.get(key[1])))
    csv_check(collection/"summaries.csv",original_rows)
    successes = sum(row["status"] == "RETURNED_VALID" for row in records)
    require(complete["successful_returned_rows"] == successes and complete["failures_or_unsupported_rows"] == 2040-successes, "Complete success/failure ledger differs")
    return dict(status="PASS_complete2040_original_inputs_memberships_objectives_status_configuration_replay",
        protocol_sha256=protocol_sha256,completion_sha256=completion_sha256,fetch_receipt_sha256=fetch_sha256,
        transport_archive_sha256=receipt["transport_archive_sha256"],analysis_source_sha256=sha(__file__),frozen_active_sources_checked=source_count,
        original_graphs=40, original_initials=120, original_rows=2040, status_counts=dict(Counter(r["status"] for r in records)),
        records=records, statistics=aggregate(records,protocol["graph_manifest"],setups), external_declared_byte_provenance=provenance(protocol,build,paths),
        limitations="Exposed public40 diagnostic; released-checkpoint overlap unverified. Native soft stops and neural fixed sampling/resident timing are different scopes, not matched total D. Declared external byte provenance is separate from result replay and does not certify full runtime environment.")


def table_csv(path, tables):
    rows = []
    for config in tables:
        for family, values in list(config["families"].items()) + [("overall_equal_family",config["overall_equal_family"])]:
            row = {key:config[key] for key in ("method","soft_seconds","bibtex_key","published_reference","resource_configuration","timing_scope",
                                              "once_only_setup_parent_seconds","once_only_setup_worker_seconds")}
            row.update(family=family, **{key:value for key,value in values.items() if not isinstance(value,dict)})
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w",encoding="utf8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection",required=True); parser.add_argument("--protocol-sha256",required=True)
    parser.add_argument("--completion-sha256",required=True); parser.add_argument("--fetch-receipt",required=True)
    parser.add_argument("--fetch-sha256",required=True); parser.add_argument("--archive")
    parser.add_argument("--path-map"); parser.add_argument("--path-map-sha256"); parser.add_argument("--out",required=True)
    args = parser.parse_args(); out = Path(args.out).resolve(); collection = Path(args.collection).resolve()
    require(not out.exists() and out != collection and collection not in out.parents, "Fresh output outside original collection required")
    mapping = []
    if args.path_map:
        require(args.path_map_sha256 and sha(args.path_map) == hex_sha(args.path_map_sha256), "Explicit path-map byte pin required")
        mapping = read(args.path_map)
    record = analyze(collection,args.protocol_sha256,args.completion_sha256,args.fetch_receipt,args.fetch_sha256,Paths(mapping),args.archive)
    out.mkdir(parents=True); (out/"audit.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf8")
    table_csv(out/"published_original_table.csv",record["statistics"]["tables"])
    print(json.dumps({key:record[key] for key in ("status","original_graphs","original_initials","original_rows","status_counts")},allow_nan=False))
    print(json.dumps({"external_declared_byte_provenance":record["external_declared_byte_provenance"]["status"]}))


if __name__ == "__main__":
    main()
