"""Frozen fixed-state known-action experiment.  Python 3.8 compatible.

This module never runs a search.  The prospective slot/stream/order matrix is
created before capture; absent states remain explicit unavailable positions.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os

ROOT_NAME = 'barr_v04_fixed_state_20261008_002'
PYTHON = '/root/miniconda3/bin/python'
CPUS = [5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
GRAPHS = ['g0340', 'g0680', 'g1200', 'g1800', 'gW0340_gE1200_s0150',
          'gW0680_gE1200_s0150', 'gW1200_gE0340_s0150', 'gW1200_gE0680_s0150']
THRESHOLDS = [60, 120, 180]
STREAMS = [0, 1901, 1907, 1913, 1931]
ARMS = ['recover', 'continue']
CAPTURE_BINARY = 'source/BARR_v0.4G/build/barr_state_capture'
REPLAY_BINARY = 'source/BARR_v0.4G/build/barr_state_replay'
SLOT_RULE = 'earliest_eligible_unfilled_threshold; one_snapshot_per_epoch; no_replacement_seed'
FUTURE_POLICY = 'pulse_only_local_fusion; no_further_joint_scout_or_kernel'
NATIVE_CONFIG = {'seed': 101, 'kernel_nodes': 128, 'population': 4, 'threads': 1,
                 'local_iterations': 64, 'challengers': 12, 'proposals': 6, 'execute_top': 2,
                 'max_kernel_vertices': 4096, 'max_gnn_vertices': 4096, 'max_rounds': -1,
                 'event_stale': 8, 'explore_every': 8, 'gate_trials': 8,
                 'pair_max_seeds': 0, 'pair_fusion_every': 4, 'seconds': 360,
                 'local_seconds': .025, 'fusion_seconds': .10, 'kernel_seconds': .03,
                 'gate_fraction': .05, 'gate_cooldown': .5, 'gate_warmup': 30,
                 'pair_slice': .5, 'mode': 'pair', 'rank': 'heuristic',
                 'recovery_backend': 'hybrid', 'gate': 'value', 'pair_policy': 'fusion-refine',
                 'pair_component': 'full', 'decompose': True,
                 'factor_width': 10, 'factor_boundary': 10, 'factor_entries': 262144}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def update(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def integer(value, label):
    if type(value) is not int:
        raise ValueError(label + ' must be a JSON integer')
    return value


def number(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(label + ' must be finite and at least ' + str(minimum))
    return float(value)


def inside(root, relative, must_exist=True):
    root = Path(root).resolve()
    value = Path(relative)
    path = (root / value).resolve()
    if value.is_absolute() or '..' in value.parts or path == root or root not in path.parents:
        raise ValueError('Artifact escaped its isolated root: ' + str(relative))
    if must_exist and (not path.is_file() or path.is_symlink()):
        raise ValueError('Missing or symbolic artifact: ' + str(relative))
    return path


def guard_root(root):
    requested = Path(root)
    resolved = requested.resolve()
    if not requested.is_absolute() or requested.name != ROOT_NAME or resolved.name != ROOT_NAME or requested.is_symlink():
        raise ValueError('Use the exact isolated absolute G root')
    return resolved


def validate_design(design):
    if design.get('schema') != 'barr_v04G_fixed_state_value_design_v1' or design.get('graphs') != GRAPHS:
        raise ValueError('Upper-scope G design/schema/eight-configuration frame changed')
    capture, fork, allocation = [design[k] for k in ['capture', 'fork', 'allocation']]
    if (capture.get('seed'), capture.get('native_seconds'), capture.get('first_strict_joint_after_thresholds_seconds'),
            capture.get('max_states_per_graph'), capture.get('one_state_per_epoch'), capture.get('negative_singletons')) != (101, 360, THRESHOLDS, 3, True, 'both strictly negative'):
        raise ValueError('Capture slot/strict-joint/360-second policy changed')
    if fork.get('future_wall_seconds_per_arm') != 360 or fork.get('replicate_seeds') != STREAMS:
        raise ValueError('Fresh future wall budget/streams changed')
    if fork.get('arms') != ['execute_pair_then_continue', 'continue_only']:
        raise ValueError('Only the two known-action continuation arms are allowed')
    if allocation.get('native_cpus') != CPUS or allocation.get('threads') != 1 or allocation.get('population') != 4 or allocation.get('coordinator_cpu') != 11 or allocation.get('real_cpu_quota') != 12:
        raise ValueError('Population, thread, resource quota or affinity frame changed')
    if allocation.get('maximum_fork_pairs') != 120 or allocation.get('maximum_fork_native_cells') != 240 or allocation.get('maximum_capture_native_cells') != 8:
        raise ValueError('Prospective intent-to-treat frame changed')
    return True


def plans(design):
    validate_design(design)
    capture = {'schema': 'barr_v04G_capture_plan_v1', 'phase': 'capture', 'seed': 101,
               'seconds': 360, 'population': 4, 'threads': 1, 'cpus': CPUS[:8],
               'coordinator_cpu': 11, 'slot_rule': SLOT_RULE, 'thresholds': THRESHOLDS,
               'cells': [{'cell_id': 'capture_%02d' % i, 'graph': graph, 'cpu': CPUS[i], 'seed': 101}
                         for i, graph in enumerate(GRAPHS)]}
    states = []
    for graph_index, graph in enumerate(GRAPHS):
        for slot_index, threshold in enumerate(THRESHOLDS):
            index = graph_index * 3 + slot_index
            # Two potential whole-state chains per core, opposite first order.
            pairs = []
            for repeat, future_seed in enumerate(STREAMS):
                order = ARMS[:] if (index // 12 + repeat) % 2 == 0 else list(reversed(ARMS))
                pairs.append({'pair_id': 'state_%02d_stream_%04d' % (index, future_seed),
                              'future_seed': future_seed, 'replicate_index': repeat, 'arm_order': order})
            states.append({'state_id': 'state_%02d' % index, 'state_index': index, 'graph': graph,
                           'threshold_seconds': threshold, 'cpu': CPUS[index % 12], 'pairs': pairs})
    fork = {'schema': 'barr_v04G_prospective_fork_plan_v1', 'phase': 'fork', 'seconds': 360,
            'population': 4, 'threads': 1, 'cpus': CPUS[:], 'coordinator_cpu': 11,
            'future_policy': FUTURE_POLICY, 'replicate_seeds': STREAMS[:], 'states': states,
            'maximum_states': 24, 'maximum_pairs': 120, 'maximum_positions': 240,
            'budget_scope': 'fresh_future360_includes_action_feedback_and_timed_logging; capture_scout_sunk_separate',
            'missing_policy': 'retain_all_missing_slots_and_their_ten_unavailable_positions; no_substitution',
            'statistics': {'all_five': 'descriptive paired mean/SD/range/t4 95% model-dependent interval',
                           'exact_stream_zero': 'separate exact captured RNG continuation',
                           'rekey_streams': 'four prespecified alternative future RNG streams summarized separately',
                           'label': 'positive if interval lower>0; negative if upper<0; otherwise undetermined; incomplete unavailable',
                           'scope': 'conditional known-action future value; no scout-discovery value or enriched prevalence'}}
    validate_plans(capture, fork)
    return capture, fork


def validate_plans(capture, fork):
    expected_capture = [(g, CPUS[i], 101) for i, g in enumerate(GRAPHS)]
    if capture.get('phase') != 'capture' or capture.get('seconds') != 360 or capture.get('slot_rule') != SLOT_RULE or capture.get('thresholds') != THRESHOLDS:
        raise ValueError('Capture budget/slot rule differs from frozen design')
    if [(c['graph'], c['cpu'], c['seed']) for c in capture['cells']] != expected_capture:
        raise ValueError('Capture graph/seed/core frame differs')
    if len(set(c['cell_id'] for c in capture['cells'])) != 8:
        raise ValueError('Capture identities are not unique')
    if fork.get('phase') != 'fork' or fork.get('seconds') != 360 or fork.get('replicate_seeds') != STREAMS or fork.get('cpus') != CPUS or fork.get('future_policy') != FUTURE_POLICY:
        raise ValueError('Fork resource/budget/stream/policy differs')
    states = fork['states']
    if len(states) != 24 or Counter(s['cpu'] for s in states) != Counter({cpu: 2 for cpu in CPUS}):
        raise ValueError('All 24 prospective slots require two whole-state blocks on each CPU')
    identities, orders = set(), Counter()
    for index, state in enumerate(states):
        if (state['state_id'], state['state_index'], state['graph'], state['threshold_seconds'], state['cpu']) != ('state_%02d' % index, index, GRAPHS[index // 3], THRESHOLDS[index % 3], CPUS[index % 12]):
            raise ValueError('Prospective state identity/allocation changed')
        if [p['future_seed'] for p in state['pairs']] != STREAMS or len(state['pairs']) != 5:
            raise ValueError('Each state requires all five fixed future streams')
        for repeat, pair in enumerate(state['pairs']):
            expected = ARMS[:] if (index // 12 + repeat) % 2 == 0 else list(reversed(ARMS))
            if pair['arm_order'] != expected or pair['pair_id'] in identities:
                raise ValueError('Paired future stream identity/order changed')
            identities.add(pair['pair_id']); orders[tuple(pair['arm_order'])] += 1
    if orders != Counter({('recover', 'continue'): 60, ('continue', 'recover'): 60}):
        raise ValueError('Maximum prospective pair order is not balanced')
    return True


def write_protocol_configs(root, setup):
    """Called by fresh-root preparation before its final setup/freeze receipts."""
    root = Path(root).resolve()
    design_path = root / 'fixed_state_design.json'
    capture, fork = plans(read(design_path))
    if sorted(g['id'] for g in setup['graphs']) != sorted(GRAPHS):
        raise ValueError('Copied setup does not contain the complete frozen graph frame')
    if setup.get('population') != 4 or setup.get('threads') != 1 or setup.get('native_seconds') != 360 or setup.get('eligible_cpus') != CPUS:
        raise ValueError('Setup population/thread/budget/calibrated CPU qualification differs')
    for relative in [CAPTURE_BINARY, REPLAY_BINARY]:
        if sha(inside(root, relative)) != setup['binaries'][relative]:
            raise ValueError('New capture/replay binary identity is unbound: ' + relative)
    common = {'design_sha256': sha(design_path), 'binaries': dict(setup['binaries']),
              'input_hashes': {g['id']: {k: g[k] for k in ['original_sha256', 'normalized_sha256', 'native_sha256']} for g in setup['graphs']},
              'cpu_calibration_paths': setup['cpu_calibration_paths'], 'min_cpu_ratio': .95}
    capture.update(common); fork.update(common)
    paths = {'capture_config_sha256': root / 'protocol/capture_config.json',
             'fork_plan_sha256': root / 'protocol/fork_plan.json'}
    dump_new(paths['capture_config_sha256'], capture)
    dump_new(paths['fork_plan_sha256'], fork)
    return {key: sha(path) for key, path in paths.items()}


def verify_ledger(root, ledger):
    if not isinstance(ledger, dict) or not ledger:
        raise ValueError('A complete nonempty frozen artifact ledger is required')
    errors = []
    for name, expected in ledger.items():
        try:
            if len(expected) != 64 or sha(inside(root, name)) != expected:
                errors.append('Changed frozen artifact: ' + name)
        except (ValueError, OSError, TypeError) as exc:
            errors.append(str(exc))
    return {'pass_all': not errors, 'files': len(ledger), 'errors': errors}


def live_native_processes(root):
    prefix = str(Path(root).resolve()) + '/source/'
    found = []
    if not Path('/proc').exists():
        return [{'reason': 'Linux process inventory unavailable'}]
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            if (item / 'stat').read_text().split(')', 1)[1].split()[0] == 'Z':
                continue
            if prefix in (item / 'maps').read_text(errors='replace'):
                found.append({'pid': int(item.name), 'command': (item / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')})
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            found.append({'pid': int(item.name), 'reason': 'process inspection denied'})
    return found
