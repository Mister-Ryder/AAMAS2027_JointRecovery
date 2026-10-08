#!/usr/bin/env python3
"""Verify delivery files; generated build/run files do not alter this check."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
    failures=[]
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            failures.append(name)
    print(json.dumps({'status':'FAIL' if failures else 'PASS','files':len(manifest['files']),'failures':failures},indent=2))
    raise SystemExit(bool(failures))
