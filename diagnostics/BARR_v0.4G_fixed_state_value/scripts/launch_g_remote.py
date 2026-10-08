"""Explicit once-only detached launch after fresh source/protocol freeze."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
sys.dont_write_bytecode = True
from g_protocol import PYTHON, dump_new, guard_root, now, sha
from run_g_remote import setup_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = guard_root(args.root)
    if any((root / name).exists() for name in ['launch_g_claim.json', 'launch_g_fixed_state.json', 'chain_started.json', 'registration_capture.json', 'results/capture']):
        raise FileExistsError('G launch already claimed/started; no automatic rerun')
    setup_audit(root)
    if not Path(PYTHON).is_file():
        raise ValueError('Registered absolute Python interpreter is missing')
    os.sched_setaffinity(0, {11})
    argv = [PYTHON, '-B', str(root / 'scripts/run_g_chain_remote.py'), '--root', str(root)]
    dump_new(root / 'launch_g_claim.json', {'created_utc': now(), 'command': argv,
                                         'setup_sha256': sha(root / 'setup_receipt.json'),
                                         'freeze_sha256': sha(root / 'protocol/source_and_protocol_freeze.json')})
    log = root / 'controller_logs/g_chain.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    with log.open('xb') as stream:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=str(root), env=env, start_new_session=True)
    receipt = {'schema': 'barr_v04G_detached_launch_v1', 'created_utc': now(), 'pid': process.pid,
               'command': argv, 'log': log.relative_to(root).as_posix(), 'detached': True,
               'population': 4, 'threads': 1, 'capture_positions': 8, 'maximum_fork_positions': 240,
               'fresh_future_seconds': 360, 'maximum_workers': 12, 'no_native_termination_or_rerun': True}
    dump_new(root / 'launch_g_fixed_state.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
