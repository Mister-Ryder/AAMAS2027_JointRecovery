"""Once-only natural capture -> audit -> fork -> audit -> archive chain."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
from g_protocol import PYTHON, dump_new, guard_root, now, sha


def step(root, name, argv):
    log = root / 'chain_logs'
    log.mkdir(exist_ok=True)
    start = now()
    with (log / (name + '_stdout.log')).open('x', encoding='utf-8') as stdout, (log / (name + '_stderr.log')).open('x', encoding='utf-8') as stderr:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
    receipt = {'step': name, 'started_utc': start, 'completed_utc': now(), 'command': argv,
               'returncode': result.returncode, 'stdout_sha256': sha(log / (name + '_stdout.log')),
               'stderr_sha256': sha(log / (name + '_stderr.log'))}
    dump_new(log / (name + '_completion.json'), receipt)
    if result.returncode:
        raise RuntimeError(name + ' failed; no automatic native rerun')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = guard_root(args.root)
    if not Path(PYTHON).is_file():
        raise ValueError('Registered absolute server Python interpreter is unavailable')
    if shutil.disk_usage(root).free < 1 << 30:
        raise RuntimeError('Insufficient space for isolated fixed-state evidence; old archives are retained')
    os.sched_setaffinity(0, {11})
    dump_new(root / 'chain_started.json', {'pid': os.getpid(), 'started_utc': now(), 'python': PYTHON,
                                         'script_sha256': sha(Path(__file__)), 'no_automatic_rerun': True})
    completed = []
    try:
        for name, script, options in [
                ('capture', 'run_g_remote.py', ['--phase', 'capture']),
                ('capture_audit', 'analyze_g_remote.py', ['--captures-only']),
                ('fork', 'run_g_remote.py', ['--phase', 'fork']),
                ('analysis', 'analyze_g_remote.py', []),
                ('archive', 'finalize_g_remote.py', [])]:
            completed.append(step(root, name, [PYTHON, '-B', str(root / 'scripts' / script), '--root', str(root)] + options))
        dump_new(root / 'chain_completion.json', {'complete': True, 'completed_utc': now(), 'steps': completed,
                                                 'native_shutdown_or_termination': False})
        return 0
    except Exception as exc:
        dump_new(root / 'chain_failure.json', {'complete': False, 'created_utc': now(), 'steps': completed,
                                              'error': type(exc).__name__ + ': ' + str(exc), 'traceback': traceback.format_exc(),
                                              'no_native_rerun': True, 'no_native_termination': True})
        # Preserve a failed root only after its processes have naturally ended.
        # Archiving failure is reported, never repaired by rerunning searches.
        try:
            step(root, 'failure_archive', [PYTHON, '-B', str(root / 'scripts/finalize_g_remote.py'), '--root', str(root), '--allow-failed'])
        except Exception as archive_exc:
            dump_new(root / 'failure_archive_failure.json', {'created_utc': now(), 'error': str(archive_exc)})
        return 2


if __name__ == '__main__':
    sys.exit(main())
