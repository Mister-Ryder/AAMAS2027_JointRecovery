"""Capture/fork coordinator.  No timeouts, kills, automatic reruns or exclusions.

Each CPU owns serial whole-state chains; each future-stream pair uses the same
core.  All native binaries run in child processes, so child CPU is counted once.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import multiprocessing as mp
import os
try:
    import resource
except ImportError:  # Protocol/command fixtures also run on the Windows host.
    resource = None
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from g_protocol import (ARMS, CAPTURE_BINARY, CPUS, GRAPHS, REPLAY_BINARY, STREAMS, THRESHOLDS,
                        canonical_sha, dump_new, guard_root, inside, now, number, read, sha,
                        update, validate_design, validate_plans, verify_ledger)
from g_state_audit import capture_audit, replay_audit


def setup_audit(root):
    setup = read(root / 'setup_receipt.json')
    freeze = read(root / 'protocol/source_and_protocol_freeze.json')
    if setup.get('status') != 'G_BUILD_READY' or setup.get('eligible_cpus') != CPUS or setup.get('population') != 4 or setup.get('threads') != 1 or setup.get('native_seconds') != 360:
        raise ValueError('Isolated G source/resource setup is not READY')
    validate_design(read(root / 'fixed_state_design.json'))
    capture, fork = read(root / 'protocol/capture_config.json'), read(root / 'protocol/fork_plan.json')
    validate_plans(capture, fork)
    for key, path in [('setup_sha256', root / 'setup_receipt.json'), ('design_sha256', root / 'fixed_state_design.json'),
                      ('capture_config_sha256', root / 'protocol/capture_config.json'), ('fork_plan_sha256', root / 'protocol/fork_plan.json')]:
        if freeze.get(key) != sha(path):
            raise ValueError('Source/protocol freeze binding changed: ' + key)
    ledger = verify_ledger(root, setup['frozen_files'])
    if not ledger['pass_all']:
        raise ValueError('Frozen source/input/script identity changed: ' + repr(ledger['errors']))
    if 'files_sha256' in freeze:
        checked = verify_ledger(root, freeze['files_sha256'])
        if not checked['pass_all']:
            raise ValueError('Final freeze file inventory changed: ' + repr(checked['errors']))
    for relative in [CAPTURE_BINARY, REPLAY_BINARY]:
        if sha(inside(root, relative)) != setup['binaries'][relative]:
            raise ValueError('Capture/replay native binary identity changed')
    for name in setup['cpu_calibration_paths']:
        if name not in setup['frozen_files']:
            raise ValueError('Actual CPU calibration evidence must be in frozen ledger')
        calibration = read(inside(root, name))
        if calibration.get('real_quota_cores') != 12 or calibration.get('concurrent_workers') != 12 or calibration.get('solver_calls') != 0 or calibration.get('all_eligible') is not True or calibration.get('eligible_cpus') != CPUS:
            raise ValueError('Actual concurrent twelve-core CPU qualification did not pass')
        if sorted(row['cpu'] for row in calibration['rows']) != CPUS or any(row['cpu_wall_ratio'] < .95 or row.get('eligible') is not True for row in calibration['rows']):
            raise ValueError('A required CPU failed its fresh calibration')
    return setup, capture, fork


def command(root, task, cell):
    common = ['--input', str(root / 'inputs/native' / (task['graph'] + '.barr')),
              '--output', str(cell / 'native_result.json'), '--seconds', '360']
    if task['phase'] == 'capture':
        return [str(root / CAPTURE_BINARY)] + common + ['--state-dir', str(cell / 'snapshots'), '--seed', '101']
    if task['phase'] != 'fork' or task['arm'] not in ARMS or task['future_seed'] not in STREAMS:
        raise ValueError('Unregistered replay command identity')
    return [str(root / REPLAY_BINARY)] + common + ['--snapshot', str(inside(root, task['snapshot_path'])),
            '--action', task['arm'], '--future-seed', str(task['future_seed'])]


def validate_native_cost(raw, process_wall, process_cpu):
    native_wall = number(raw['native_seconds'], 'native_seconds', 359.99)
    native_cpu = number(raw['cpu_seconds'], 'cpu_seconds')
    if native_wall > process_wall + .1 or native_cpu > process_cpu + .1:
        raise ValueError('Native budget cost exceeds enclosing child process cost; synthetic clocks are not performance evidence')
    return native_wall, native_cpu


def execute(root, task, setup):
    if resource is None:
        raise RuntimeError('Native execution requires the registered Linux server')
    cell = root / 'results' / task['phase'] / task['cell_id']
    cell.mkdir(parents=True, exist_ok=False)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    row = dict(task, status='STARTED', started_utc=now(), population=4, threads=1,
               future_budget_seconds=360 if task['phase'] == 'fork' else None,
               native_output=(cell / 'native_result.json').relative_to(root).as_posix())
    dump_new(cell / 'task.json', task)
    try:
        binary_name = CAPTURE_BINARY if task['phase'] == 'capture' else REPLAY_BINARY
        binary = root / binary_name
        definition = next(g for g in setup['graphs'] if g['id'] == task['graph'])
        native_input = root / 'inputs/native' / (task['graph'] + '.barr')
        if sha(binary) != setup['binaries'][binary_name] or sha(native_input) != definition['native_sha256']:
            raise ValueError('Native binary/input changed before execution')
        if sorted(os.sched_getaffinity(0)) != [task['cpu']]:
            raise ValueError('Native worker lost its registered single-core affinity')
        row.update(binary_sha256=sha(binary), native_input_sha256=sha(native_input), affinity=[task['cpu']])
        argv = command(root, task, cell)
        row['command'] = argv
        dump_new(cell / 'command.json', argv)
        env = dict(os.environ)
        for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
            env[name] = '1'
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_start = time.perf_counter()
        row['native_process_started_utc'] = now()
        with (cell / 'native_stdout.log').open('x', encoding='utf-8') as stdout, (cell / 'native_stderr.log').open('x', encoding='utf-8') as stderr:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env)
            row['native_pid'] = process.pid
            dump_new(cell / 'native_started.json', {'pid': process.pid, 'utc': row['native_process_started_utc'], 'command': argv})
            returncode = process.wait()  # Intentional unbounded natural completion.
        row['native_process_completed_utc'] = now()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        process_wall = time.perf_counter() - child_start
        process_cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        row.update(returncode=returncode, native_process_wall_seconds=process_wall,
                   native_process_cpu_seconds=process_cpu, max_rss_kb=after.ru_maxrss,
                   rss_scope='worker lifetime child peak, not isolated cell peak')
        if returncode:
            raise RuntimeError('Native returned failure code ' + str(returncode))
        raw = read(cell / 'native_result.json')
        native_wall, native_cpu = validate_native_cost(raw, process_wall, process_cpu)
        row.update(native_seconds=native_wall, native_cpu_seconds=native_cpu,
                   native_process_outside_budget_wall_seconds=max(0., process_wall - native_wall),
                   native_process_outside_budget_cpu_seconds=max(0., process_cpu - native_cpu),
                   cpu_wall_ratio=native_cpu / native_wall, min_cpu_ratio=.95,
                   resource_qualified=native_cpu / native_wall >= .95,
                   resource_flags=[] if native_cpu / native_wall >= .95 else ['native_cpu_wall_ratio_below_registered_threshold'])
        audit_start_wall, audit_start_cpu = time.perf_counter(), time.process_time()
        audited = capture_audit(root, row, setup) if task['phase'] == 'capture' else replay_audit(root, row, setup)
        row.update(status='COMPLETE', independent_audit=audited,
                   validation_wall_seconds=time.perf_counter() - audit_start_wall,
                   validation_cpu_seconds=time.process_time() - audit_start_cpu,
                   native_output_sha256=sha(cell / 'native_result.json'))
        if sha(binary) != row['binary_sha256'] or sha(native_input) != row['native_input_sha256']:
            raise ValueError('Native source/input changed during a cell')
    except Exception as exc:
        row.update(status='NATIVE_OR_AUDIT_ERROR', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    row.update(completed_utc=now(), controller_cpu_seconds=time.process_time() - start_cpu,
               total_cell_seconds=time.perf_counter() - start_wall)
    row['controller_only_cpu_seconds'] = row['controller_cpu_seconds']
    row['total_cpu_seconds'] = row['controller_cpu_seconds'] + row.get('native_process_cpu_seconds', 0.)
    dump_new(cell / 'result.json', row)
    return row


def core_chain(root_string, cpu, tasks, setup):
    root = Path(root_string)
    os.sched_setaffinity(0, {cpu})
    for task in tasks:
        execute(root, task, setup)
    return {'cpu': cpu, 'positions': len(tasks)}


def census(root, phase, tasks):
    rows = []
    for task in tasks:
        path = root / 'results' / phase / task['cell_id'] / 'result.json'
        if path.is_file():
            rows.append(read(path))
    return rows


def run_phase(root, phase, config, tasks, setup):
    result_root = root / 'results' / phase
    registration_path = root / ('registration_' + phase + '.json')
    completion_path = root / ('completion_' + phase + '.json')
    if result_root.exists() or registration_path.exists() or completion_path.exists():
        raise FileExistsError('Fresh phase only: no overwrite or automatic solver rerun')
    inherited = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(CPUS + [11]))
    expanded = sorted(os.sched_getaffinity(0))
    if not set(task['cpu'] for task in tasks) <= set(expanded):
        raise ValueError('Registered calibrated CPU is unavailable')
    groups = defaultdict(list)
    for task in tasks:
        groups[task['cpu']].append(task)
    if len(groups) > 12 or any(cpu not in CPUS for cpu in groups):
        raise ValueError('Real 12-core quota/CPU affinity plan exceeded')
    result_root.mkdir(parents=True, exist_ok=False)
    registration = {'schema': 'barr_v04G_phase_registration_v1', 'phase': phase, 'created_utc': now(),
                    'positions': len(tasks), 'config': config, 'tasks': tasks,
                    'config_canonical_sha256': canonical_sha(config), 'tasks_canonical_sha256': canonical_sha(tasks),
                    'setup_sha256': sha(root / 'setup_receipt.json'),
                    'freeze_sha256': sha(root / 'protocol/source_and_protocol_freeze.json'),
                    'script_sha256': sha(Path(__file__)), 'inherited_affinity': inherited,
                    'expanded_affinity': expanded, 'cpus': sorted(groups), 'real_cpu_quota': 12,
                    'failure_policy': 'all positions retained; no automatic rerun/kill/resource exclusion'}
    dump_new(registration_path, registration)
    os.sched_setaffinity(0, {11})
    worker_errors = []
    if tasks:
        # spawn avoids inheriting native/global RNG state from the coordinator.
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=mp.get_context('spawn')) as pool:
            futures = {pool.submit(core_chain, str(root), cpu, group, setup): cpu for cpu, group in groups.items()}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    worker_errors.append({'cpu': futures[future], 'error': type(exc).__name__ + ': ' + str(exc)})
                update(root / ('progress_' + phase + '.json'), {'utc': now(), 'expected_positions': len(tasks),
                                                              'positions': len(census(root, phase, tasks)), 'worker_errors': worker_errors})
    rows = census(root, phase, tasks)
    journal = result_root / 'runs.jsonl'
    with journal.open('x', encoding='utf-8') as stream:
        import json
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    proof = {'phase': phase, 'complete': len(rows) == len(tasks) and not worker_errors,
             'completed_utc': now(), 'expected_positions': len(tasks), 'positions': len(rows),
             'statuses': dict(Counter(row['status'] for row in rows)),
             'errors': sum(row['status'] != 'COMPLETE' for row in rows), 'worker_errors': worker_errors,
             'resource_unqualified': sum(row.get('resource_qualified') is False for row in rows),
             'registration_sha256': sha(registration_path), 'runs_sha256': sha(journal)}
    dump_new(completion_path, proof)
    return proof


def capture_tasks(config):
    return [dict(cell, phase='capture', arm='capture', seconds=360) for cell in config['cells']]


def freeze_fork_tasks(root, plan):
    completion = read(root / 'completion_capture.json')
    if not completion.get('complete') or completion.get('errors') or completion.get('worker_errors'):
        raise ValueError('Capture frame has failures; retain evidence and do not create fork outcomes')
    audit_receipt = read(root / 'capture_audit/capture_audit_completion.json')
    if audit_receipt.get('complete') is not True:
        raise ValueError('Independent complete snapshot audit must pass before replay')
    snapshot_index_path = root / 'capture_audit/snapshot_index.json'
    if audit_receipt['snapshot_index_sha256'] != sha(snapshot_index_path):
        raise ValueError('Snapshot audit index identity changed')
    snapshots = read(snapshot_index_path)
    indexed = {(s['graph'], s['threshold_seconds']): s for s in snapshots['available_states']}
    tasks, available, missing = [], [], []
    for state in plan['states']:
        snapshot = indexed.get((state['graph'], state['threshold_seconds']))
        if snapshot is None:
            missing.append(dict(state, status='MISSING_CAPTURE_SLOT', unavailable_native_positions=10))
            continue
        available.append(dict(state, snapshot=snapshot))
        for pair in state['pairs']:
            for arm in pair['arm_order']:
                task = dict(phase='fork', cell_id=pair['pair_id'] + '_' + arm, state_id=state['state_id'],
                            state_index=state['state_index'], graph=state['graph'], threshold_seconds=state['threshold_seconds'],
                            cpu=state['cpu'], arm=arm, future_seed=pair['future_seed'], pair_id=pair['pair_id'],
                            replicate_index=pair['replicate_index'], arm_order=pair['arm_order'], seconds=360,
                            snapshot_path=snapshot['snapshot_path'], snapshot_sha256=snapshot['snapshot_sha256'],
                            metadata_path=snapshot['metadata_path'], metadata_sha256=snapshot['metadata_sha256'])
                tasks.append(task)
    actual = {'schema': 'barr_v04G_actual_fork_frame_v1', 'phase': 'fork', 'created_utc': now(),
              'prospective_plan_sha256': sha(root / 'protocol/fork_plan.json'),
              'capture_completion_sha256': sha(root / 'completion_capture.json'),
              'capture_audit_receipt_sha256': sha(root / 'capture_audit/capture_audit_completion.json'),
              'snapshot_index_sha256': sha(snapshot_index_path), 'available_states': available,
              'missing_states': missing, 'available_positions': len(tasks), 'maximum_positions': 240,
              'unavailable_positions': 10 * len(missing), 'tasks': tasks,
              'future_outcomes_observed_before_actual_frame': 0,
              'scope': 'subset solely from prespecified missing capture slots; no outcome-based state/seed selection'}
    dump_new(root / 'protocol/fork_config.json', actual)
    return actual, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--phase', choices=['capture', 'fork'], required=True)
    args = parser.parse_args()
    root = guard_root(args.root)
    setup, capture, fork = setup_audit(root)
    if args.phase == 'capture':
        config, tasks = capture, capture_tasks(capture)
    else:
        config, tasks = freeze_fork_tasks(root, fork)
    proof = run_phase(root, args.phase, config, tasks, setup)
    print(__import__('json').dumps(proof), flush=True)
    return 0 if proof['complete'] and not proof['errors'] else 2


if __name__ == '__main__':
    sys.exit(main())
