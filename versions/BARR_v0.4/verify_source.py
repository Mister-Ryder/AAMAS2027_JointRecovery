"""Verify the immutable research source; no compilation or search."""
from pathlib import Path, PurePosixPath
import hashlib
import json

ROOT = Path(__file__).resolve().parent
METADATA_SHA256 = "414fab5f0a1d40170d386a48b0a931148a69509c64992191d4d8f0cf9017f5d8"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    meta_path = ROOT / "FINAL_VERSION.json"
    if digest(meta_path) != METADATA_SHA256:
        raise SystemExit("FAIL: original FINAL_VERSION.json changed")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(metadata["source_files"]) != 50:
        raise SystemExit("FAIL: expected 50 frozen source files")
    failures = []
    for relative, expected in metadata["source_files"].items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit("FAIL: unsafe manifest path")
        actual = ROOT / metadata["source_directory"] / relative
        if not actual.is_file() or digest(actual) != expected:
            failures.append(relative)
    entry = metadata["entrypoint"]
    entry_path = ROOT / entry["path"]
    if not entry_path.is_file() or digest(entry_path) != entry["sha256"]:
        failures.append(entry["path"])
    if failures:
        raise SystemExit("FAIL: " + ", ".join(failures))
    print("PASS: original metadata, all 50 frozen source files and original entrypoint match SHA256")

if __name__ == "__main__":
    main()
