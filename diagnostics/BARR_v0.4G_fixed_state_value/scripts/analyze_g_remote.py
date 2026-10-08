"""Independent complete-frame state/value analysis; no solver calls or reruns."""
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import argparse
import json
import math
import os
import statistics
import sys
import time

sys.dont_write_bytecode = True
from g_protocol import (CPUS, STREAMS, canonical_sha, dump_new, guard_root, now, read, sha,
                        validate_plans, verify_ledger)
from g_state_audit import capture_audit, replay_audit


def phase_rows(root, phase, setup, audit=True):
    registration_path = root / ('registration_' + phase + '.json')
    completion_path = root / ('completion_' + phase + '.json')
    registration, completion = read(registration_path), read(completion_path)
    tasks = registration['tasks']
    errors, rows, audits = [], [], []
    if registration['tasks_canonical_sha256'] != canonical_sha(tasks) or registration['config_canonical_sha256'] != canonical_sha(registration['config']):
        errors.append('Registered task/config canonical identity mismatch')
    if completion['registration_sha256'] != sha(registration_path) or completion['runs_sha256'] != sha(root / 'results' / phase / 'runs.jsonl'):
        errors.append('Completion registration/journal binding changed')
    if registration['setup_sha256'] != sha(root / 'setup_receipt.json') or registration['freeze_sha256'] != sha(root / 'protocol/source_and_protocol_freeze.json'):
        errors.append('Phase used different setup/source freeze')
    with (root / 'results' / phase / 'runs.jsonl').open(encoding='utf-8') as stream:
        journal = [json.loads(line) for line in stream if line.strip()]
    if len({row['cell_id'] for row in journal}) != len(journal):
        errors.append('Duplicate phase journal identity')
    journal_map = {row['cell_id']: row for row in journal}
    expected_ids = {task['cell_id'] for task in tasks}
    if set(journal_map) != expected_ids:
        errors.append('All registered positions, including failures, must have a journal row')
    disk_ids = {path.parent.name for path in (root / 'results' / phase).glob('*/result.json')}
    if disk_ids != set(journal_map):
        errors.append('Cell result set differs from registered/journal frame')
    for task in tasks:
        row = journal_map.get(task['cell_id'])
        if row is None:
            rows.append(dict(task, status='MISSING_NATIVE_RESULT'))
            continue
        rows.append(row)
        try:
            if read(root / 'results' / phase / row['cell_id'] / 'result.json') != row:
                raise ValueError('Raw per-cell result differs from journal')
            for key, expected in task.items():
                if row.get(key) != expected:
                    raise ValueError('Registered cell identity changed: ' + key)
            if row['status'] != 'COMPLETE':
                raise ValueError('Retained native/audit failure: ' + str(row.get('error')))
            if row['native_output_sha256'] != sha(root / row['native_output']):
                raise ValueError('Native JSON changed after execution audit')
            if audit:
                checked = capture_audit(root, row, setup) if phase == 'capture' else replay_audit(root, row, setup)
                if checked != row['independent_audit']:
                    raise ValueError('Recomputed original integer/state audit differs from saved audit')
                audits.append(dict(cell_id=row['cell_id'], verified=True, audit=checked))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append(row['cell_id'] + ': ' + str(exc))
            audits.append(dict(cell_id=row['cell_id'], verified=False, error=str(exc)))
    if completion.get('expected_positions') != len(tasks) or completion.get('positions') != len(rows) or not completion.get('complete') or completion.get('errors') or completion.get('worker_errors'):
        errors.append('Retained completion frame contains missing/failed positions')
    return {'phase': phase, 'rows': rows, 'audits': audits, 'errors': errors,
            'registration': registration, 'completion': completion,
            'complete_audited_frame': not errors}


def audit_captures(root):
    setup = read(root / 'setup_receipt.json')
    frame = phase_rows(root, 'capture', setup)
    output = root / 'capture_audit'
    output.mkdir(parents=True, exist_ok=False)
    index = {'schema': 'barr_v04G_independent_snapshot_index_v1', 'available_states': [], 'missing_slots': [],
             'capture_registration_sha256': sha(root / 'registration_capture.json'),
             'capture_completion_sha256': sha(root / 'completion_capture.json')}
    for row in frame['rows']:
        if row.get('status') != 'COMPLETE':
            continue
        verified = row['independent_audit']
        index['available_states'].extend(verified['snapshots'])
        index['missing_slots'].extend({'graph': row['graph'], 'threshold_seconds': threshold,
                                       'status': 'MISSING_CAPTURE_SLOT'} for threshold in verified['missing_thresholds'])
    if len(index['available_states']) + len(index['missing_slots']) != 24:
        frame['errors'].append('Full 24-slot captured/missing table is incomplete')
    dump_new(output / 'capture_frame_audit.json', frame)
    dump_new(output / 'snapshot_index.json', index)
    proof = {'complete': not frame['errors'], 'created_utc': now(), 'capture_positions': len(frame['rows']),
             'available_states': len(index['available_states']), 'missing_slots': len(index['missing_slots']),
             'snapshot_index_sha256': sha(output / 'snapshot_index.json'),
             'capture_frame_audit_sha256': sha(output / 'capture_frame_audit.json'), 'errors': frame['errors'],
             'scope': 'independent raw original graph/timeline/witness + full native serialization identity; no future outcomes read'}
    dump_new(output / 'capture_audit_completion.json', proof)
    return proof


def cost_audit(rows):
    intervals, events, errors = defaultdict(list), [], []
    for row in rows:
        if row.get('native_process_started_utc') and row.get('native_process_completed_utc'):
            start, end = [datetime.fromisoformat(row[key].replace('Z', '+00:00')) for key in ['native_process_started_utc', 'native_process_completed_utc']]
            intervals[row['cpu']].append((start, end, row['cell_id']))
            events.extend([(start, 1, row['cell_id']), (end, -1, row['cell_id'])])
            if end < start:
                errors.append('Negative native interval: ' + row['cell_id'])
        if row.get('status') != 'COMPLETE':
            continue
        try:
            cpu, wall = row['native_cpu_seconds'], row['native_seconds']
            expected_ratio = cpu / wall
            if not math.isclose(row['cpu_wall_ratio'], expected_ratio, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('Native CPU/wall differs')
            if row['resource_qualified'] != (expected_ratio >= .95) or row['resource_flags'] != ([] if expected_ratio >= .95 else ['native_cpu_wall_ratio_below_registered_threshold']):
                raise ValueError('CPU qualification/flags differ')
            if row['controller_only_cpu_seconds'] != row['controller_cpu_seconds'] or not math.isclose(row['total_cpu_seconds'], row['controller_cpu_seconds'] + row['native_process_cpu_seconds'], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('Native child CPU counted twice or omitted')
            if row['cpu'] not in CPUS or row['affinity'] != [row['cpu']] or row['population'] != 4 or row['threads'] != 1 or wall < 359.99:
                raise ValueError('Single-core/population/fresh360 identity differs')
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            errors.append(row['cell_id'] + ': ' + str(exc))
    for cpu, periods in intervals.items():
        frontier = None
        for start, end, cell in sorted(periods):
            if frontier is not None and start < frontier:
                errors.append('Overlapping native programs on same CPU%d: %s' % (cpu, cell))
            frontier = end if frontier is None else max(frontier, end)
    live = maximum = 0
    for instant, delta, cell in sorted(events):
        live += delta; maximum = max(maximum, live)
    if maximum > 12:
        errors.append('Observed native concurrency exceeds real twelve-core quota')
    metrics = ['native_seconds', 'native_cpu_seconds', 'native_process_wall_seconds', 'native_process_cpu_seconds',
               'native_process_outside_budget_wall_seconds', 'native_process_outside_budget_cpu_seconds',
               'controller_only_cpu_seconds', 'total_cpu_seconds', 'total_cell_seconds', 'validation_wall_seconds']
    groups = {}
    for phase, arm in sorted({(r['phase'], r['arm']) for r in rows}):
        available = [r for r in rows if (r['phase'], r['arm']) == (phase, arm)]
        groups[phase + '/' + arm] = {'positions': len(available), 'statuses': dict(Counter(r.get('status') for r in available))}
        for metric in metrics:
            values = [r[metric] for r in available if metric in r]
            groups[phase + '/' + arm][metric] = {'recorded_positions': len(values), 'sum': sum(values), 'mean': statistics.mean(values) if values else None,
                                               'min': min(values) if values else None, 'max': max(values) if values else None}
    return {'pass_all': not errors, 'errors': errors, 'maximum_observed_native_concurrency': maximum,
            'resource_flagged': sum(r.get('resource_qualified') is False for r in rows), 'by_phase_arm': groups,
            'accounting': 'All G natives are external children: controller CPU + complete child CPU once; native budget CPU is a subset of child CPU; restore/load/receipt overhead separate',
            'resource_policy': 'flags retained descriptively; no automatic quality exclusion'}


def descriptive(deltas, critical):
    if not deltas:
        return None
    exact = sum(Fraction(value) for value in deltas) / len(deltas)
    values = [float(value) / 1e6 for value in deltas]
    sd = statistics.stdev(values) if len(values) > 1 else None
    se = sd / math.sqrt(len(values)) if sd is not None else None
    mean = float(exact) / 1e6
    interval = [mean - critical * se, mean + critical * se] if se is not None else None
    label = 'estimated_positive' if interval and interval[0] > 0 else 'estimated_negative' if interval and interval[1] < 0 else 'undetermined'
    return {'n': len(values), 'deltas_ticks': deltas, 'deltas_target_seconds': values,
            'mean_delta_ticks_exact': {'numerator': exact.numerator, 'denominator': exact.denominator},
            'mean_delta_target_seconds': mean, 'sd_target_seconds': sd, 'se_target_seconds': se,
            'range_target_seconds': [min(values), max(values)], 'model_dependent_t95_interval_target_seconds': interval,
            'descriptive_label': label, 'wins_ties_losses': [sum(v > 0 for v in deltas), sum(v == 0 for v in deltas), sum(v < 0 for v in deltas)],
            'interval_scope': 'small-n descriptive model-dependent t interval, not proof or iid physical replication'}


def paired_states(plan, actual, rows):
    indexed = {row['cell_id']: row for row in rows}
    available = {state['state_id'] for state in actual['available_states']}
    output, errors = [], []
    for state in plan['states']:
        item = {k: state[k] for k in ['state_id', 'state_index', 'graph', 'threshold_seconds', 'cpu']}
        item['pairs'] = []
        deltas = []
        for pair in state['pairs']:
            branches = {arm: indexed.get(pair['pair_id'] + '_' + arm) for arm in ['recover', 'continue']}
            p = dict(pair, available=state['state_id'] in available, complete=False)
            if p['available'] and all(branches.values()) and all(row.get('status') == 'COMPLETE' for row in branches.values()):
                a, b = branches['recover'], branches['continue']
                try:
                    x, y = a['independent_audit'], b['independent_audit']
                    if a['cpu'] != b['cpu'] or a['snapshot_sha256'] != b['snapshot_sha256'] or x['ready_state_sha256'] != y['ready_state_sha256'] or x['archive_before_ticks'] != y['archive_before_ticks']:
                        raise ValueError('Both arms did not start from identical complete prepared state/RNG/archive on same CPU')
                    first, second = [branches[arm] for arm in pair['arm_order']]
                    if first['native_process_completed_utc'] > second['native_process_started_utc']:
                        raise ValueError('Same-core pair did not obey serial prespecified order')
                    delta = x['archive_terminal_ticks'] - y['archive_terminal_ticks']
                    relative = Fraction(100 * delta, y['archive_terminal_ticks'])
                    p.update(complete=True, recover_terminal_ticks=x['archive_terminal_ticks'], continue_terminal_ticks=y['archive_terminal_ticks'],
                             frozen_archive_ticks=x['archive_before_ticks'], delta_ticks=delta, delta_target_seconds=delta / 1e6,
                             relative_percent=float(relative), relative_percent_exact={'numerator': relative.numerator, 'denominator': relative.denominator},
                             recover_gain_over_frozen_ticks=x['future_archive_gain_ticks'], continue_gain_over_frozen_ticks=y['future_archive_gain_ticks'],
                             direct_action_archive_gain_ticks=x['direct_action_archive_gain_ticks'],
                             resource_flags={'recover': a['resource_flags'], 'continue': b['resource_flags']},
                             action_seconds={arm: branches[arm]['independent_audit']['action_seconds'] for arm in branches},
                             continuation_seconds={arm: branches[arm]['independent_audit']['continuation_seconds'] for arm in branches},
                             population_diversity={arm: branches[arm]['independent_audit']['populations'] for arm in branches})
                    deltas.append(delta)
                except (KeyError, ValueError, ZeroDivisionError) as exc:
                    errors.append(pair['pair_id'] + ': ' + str(exc)); p['error'] = str(exc)
            elif p['available']:
                p['error'] = 'Retained unavailable or failed fork position; no completed-subset quality mean'
                errors.append(pair['pair_id'] + ': ' + p['error'])
            item['pairs'].append(p)
        item['status'] = 'MISSING_CAPTURE_SLOT' if state['state_id'] not in available else 'COMPLETE' if len(deltas) == 5 else 'INCOMPLETE_FORK_FRAME'
        item['five_stream_descriptive'] = descriptive(deltas, 2.7764451051977987) if len(deltas) == 5 else None
        item['exact_captured_rng_stream_zero'] = item['pairs'][0] if item['pairs'][0].get('complete') else None
        rekey = [p['delta_ticks'] for p in item['pairs'][1:] if p.get('complete')]
        item['four_rekey_stream_descriptive'] = descriptive(rekey, 3.182446305284263) if len(rekey) == 4 else None
        item['inference_scope'] = 'one captured structural/controller state; stream0 exact RNG; four controlled rekeys; enriched nonindependent states, not dataset prevalence'
        output.append(item)
    return output, errors


def analyze(root, output):
    output = Path(output).resolve()
    if output.exists() or Path(root).resolve() not in output.parents:
        raise FileExistsError('Independent G analysis requires a fresh internal output directory')
    output.mkdir(parents=True, exist_ok=False)
    begin_wall, begin_cpu = time.perf_counter(), time.process_time()
    setup = read(root / 'setup_receipt.json')
    capture, fork = read(root / 'protocol/capture_config.json'), read(root / 'protocol/fork_plan.json')
    validate_plans(capture, fork)
    actual = read(root / 'protocol/fork_config.json')
    cap, future = phase_rows(root, 'capture', setup), phase_rows(root, 'fork', setup)
    states, pairing_errors = paired_states(fork, actual, future['rows'])
    sources = verify_ledger(root, setup['frozen_files'])
    resources = cost_audit(cap['rows'] + future['rows'])
    errors = cap['errors'] + future['errors'] + pairing_errors + sources['errors'] + resources['errors']
    if actual['prospective_plan_sha256'] != sha(root / 'protocol/fork_plan.json') or actual['available_positions'] != len(future['rows']) or len(actual['available_states']) + len(actual['missing_states']) != 24:
        errors.append('Actual fork frame does not preserve the complete prospective available/missing table')
    if errors:
        for state in states:
            state['five_stream_descriptive'] = None
            state['four_rekey_stream_descriptive'] = None
            state['exact_captured_rng_stream_zero'] = None
            state['estimate_unavailable_reason'] = 'Complete-frame/source/state/resource audit failed; raw retained differences are for investigation only'
    missing = sum(s['status'] == 'MISSING_CAPTURE_SLOT' for s in states)
    summary = {'schema': 'barr_v04G_known_action_state_value_analysis_v1', 'created_utc': now(),
               'overall_audit_pass': not errors, 'errors': errors, 'capture_positions': len(cap['rows']),
               'prospective_states': 24, 'available_states': 24 - missing, 'missing_states': missing,
               'prospective_fork_pairs': 120, 'prospective_fork_positions': 240,
               'actual_fork_positions': len(future['rows']), 'unavailable_positions_missing_capture': 10 * missing,
               'native_failures': sum(r.get('status') != 'COMPLETE' for r in cap['rows'] + future['rows']),
               'resource_flagged': resources['resource_flagged'],
               'state_label_counts': dict(Counter(s['five_stream_descriptive']['descriptive_label'] if s['five_stream_descriptive'] else 'AUDIT_INVALID' if errors else s['status'] for s in states)),
               'prospective_pair_order_counts': dict(Counter('>'.join(p['arm_order']) for s in fork['states'] for p in s['pairs'])),
               'actual_available_pair_order_counts': dict(Counter('>'.join(p['arm_order']) for s in states if s['status'] != 'MISSING_CAPTURE_SLOT' for p in s['pairs'])),
               'states': states, 'resources': resources, 'sources': sources,
               'budget_definition': 'Each arm receives fresh360 wall seconds beginning before action/skip and common pending feedback, distinct from capture remaining allowance; native restore/decode/rekey and final receipt I/O outside allowance are recorded as process overhead',
               'estimand': 'conditional already-known pair execution value including opportunity cost and downstream archive/diversity feedback; capture/scout discovery sunk cost separately measured and not identified as total discovery net value',
               'scope': 'strict-double-negative positive-opportunity enriched time-threshold states on eight reused configurations of one physical mother source; no positive-state prevalence/recall or independent-dataset claim',
               'uncertainty': 'Stream0 exact captured RNG shown separately; four deterministic rekeys separate; five-stream mean/SD/range/t4 interval are exploratory model-dependent summaries, not five iid copies of the exact captured RNG state',
               'failure_policy': 'all missing, failed, negative and resource-flagged positions retained; no automatic rerun or completed-subset state mean',
               'analysis_wall_seconds': time.perf_counter() - begin_wall, 'analysis_cpu_seconds': time.process_time() - begin_cpu}
    artifacts = {'summary.json': summary, 'capture_frame_audit.json': cap, 'fork_frame_audit.json': future,
                 'state_pair_values.json': states, 'source_audit.json': sources, 'resource_audit.json': resources}
    for name, value in artifacts.items():
        dump_new(output / name, value)
    manifest = {'overall_audit_pass': not errors, 'created_utc': now(),
                'files': {name: sha(output / name) for name in artifacts}}
    dump_new(output / 'analysis_manifest.json', manifest)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--captures-only', action='store_true')
    parser.add_argument('--output', default='analysis_g_final')
    args = parser.parse_args()
    root = guard_root(args.root)
    os.sched_setaffinity(0, {11})
    proof = audit_captures(root) if args.captures_only else analyze(root, root / args.output)
    print(json.dumps(proof if args.captures_only else {k: proof[k] for k in ['overall_audit_pass', 'available_states', 'missing_states', 'actual_fork_positions', 'state_label_counts', 'errors']}), flush=True)
    return 0 if proof.get('complete', proof.get('overall_audit_pass')) else 2


if __name__ == '__main__':
    sys.exit(main())
