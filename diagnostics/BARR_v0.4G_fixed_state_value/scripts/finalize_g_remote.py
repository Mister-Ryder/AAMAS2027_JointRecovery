"""Archive immutable G fixed-state evidence after native searches finish naturally."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import zipfile

import g_protocol
from g_protocol import dump_new, guard_root, now, read, sha


def finalize(root, analysis, allow_failed=False):
    root, analysis = Path(root).resolve(), Path(analysis).resolve()
    if analysis == root or root not in analysis.parents:
        raise ValueError("G archive analysis must stay inside the isolated G root")
    scanner = getattr(g_protocol, "live_native_processes", None) or getattr(g_protocol, "live_native_processes", None)
    if scanner is None:
        raise RuntimeError("G read-only native-process completion scanner is unavailable")
    if scanner(root):
        raise RuntimeError("G native searches are still running; archive waits for natural completion")
    summary_path, index_path = analysis / "summary.json", analysis / "analysis_manifest.json"
    summary = read(summary_path) if summary_path.exists() else None
    index = read(index_path) if index_path.exists() else None
    audit_pass = bool(summary and index and summary.get("overall_audit_pass") is True and index.get("overall_audit_pass") is True)
    if not audit_pass and not allow_failed:
        raise RuntimeError("G independent complete-frame analysis did not pass; failure archival requires --allow-failed")
    if index:
        for name, digest in index["files"].items():
            path = (analysis / name).resolve()
            if analysis not in path.parents or not path.is_file() or sha(path) != digest:
                raise RuntimeError("G analysis manifest binding failed: " + name)
    archive, manifest, receipt = [root / name for name in ["evidence_bundle_v04g.zip", "evidence_manifest_v04g.json", "finalization_v04g_completion.json"]]
    if any(path.exists() for path in [archive, manifest, receipt]):
        raise FileExistsError("G archive evidence already exists; never replace it")
    wall_begin, cpu_begin = time.perf_counter(), time.process_time()
    setup = read(root / "setup_receipt.json") if (root / "setup_receipt.json").is_file() else {}
    frozen_files = setup.get("frozen_files", {})
    ledger = set(frozen_files)
    missing_frozen, changed_frozen = [], []
    for name, expected in frozen_files.items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or path == root or root not in path.parents:
            raise RuntimeError("G archive frozen ledger path escaped its root: " + name)
        if not path.is_file() or path.is_symlink():
            missing_frozen.append(name)
        elif sha(path) != expected:
            changed_frozen.append(name)
    if audit_pass and (missing_frozen or changed_frozen):
        raise RuntimeError("G frozen evidence changed after its passing analysis")
    excluded = {archive.name, manifest.name, receipt.name, "finalization_stdout.log", "finalization_stderr.log",
                "archive_stdout.log", "archive_stderr.log", "failure_archive_stdout.log", "failure_archive_stderr.log"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        name = relative.as_posix()
        if path.name in excluded:
            continue
        if name not in ledger and (any(part in {".git", "__pycache__"} for part in relative.parts) or path.suffix in {".tmp", ".part", ".pyc"}):
            continue
        if name not in ledger and "build" in relative.parts and path.name not in {"barr_state_capture", "barr_state_replay", "barr_state_tests", "barr_state_oracle", "flags.make"}:
            continue
        if root not in path.resolve().parents:
            raise RuntimeError("G evidence path escaped its root")
        files.append((path, relative.as_posix()))
    hashes = {name: sha(path) for path, name in files}
    if ledger - set(hashes) != set(missing_frozen):
        raise RuntimeError("G archive omitted an existing explicitly frozen ledger artifact")
    if not hashes:
        raise RuntimeError("Refuse to archive an empty G evidence root")
    dump_new(manifest, hashes)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as package:
        for path, name in files:
            package.write(path, name)
        package.write(manifest, manifest.name)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or len(package.namelist()) != len(hashes) + 1 or len(set(package.namelist())) != len(hashes) + 1:
            raise RuntimeError("G archive CRC or member count failed")
        for name, expected in hashes.items():
            digest = hashlib.sha256()
            with package.open(name) as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise RuntimeError("G archived member differs from its manifest: " + name)
    for path, name in files:
        if sha(path) != hashes[name]:
            raise RuntimeError("G evidence changed during archive: " + name)
    proof = dict(complete=True, archived_utc=now(), quality_audit_pass=audit_pass,
                 archive=str(archive), archive_sha256=sha(archive), bytes=archive.stat().st_size,
                 members=len(hashes) + 1, manifest_sha256=sha(manifest),
                 analysis_manifest_sha256=sha(index_path) if index_path.exists() else None,
                 performance_positions=0, fixed_state_positions=summary.get("actual_fork_positions", 0) if summary else 0,
                 capture_positions=summary.get("capture_positions", 0) if summary else 0, failure_evidence_preserved=not audit_pass,
                 frozen_ledger_files=len(ledger), missing_frozen_files=missing_frozen, changed_frozen_files=changed_frozen,
                 no_native_program_termination=True, no_performance_reruns=True, no_source_or_result_mutation=True,
                 experiment_scope="fixed known-action state continuation, not a conventional performance matrix",
                 server_shutdown=False, archive_wall_seconds=time.perf_counter() - wall_begin,
                 archive_cpu_seconds=time.process_time() - cpu_begin)
    dump_new(receipt, proof)
    print(json.dumps(proof), flush=True)
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--analysis", default="analysis_g_final")
    parser.add_argument("--allow-failed", action="store_true")
    args = parser.parse_args()
    root = guard_root(args.root)
    os.sched_setaffinity(0, {11})
    finalize(root, (root / args.analysis).resolve(), args.allow_failed)


if __name__ == "__main__":
    main()
