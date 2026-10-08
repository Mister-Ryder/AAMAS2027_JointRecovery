"""Independent, read-only audits of capture snapshots and replay JSON mirrors."""
from pathlib import Path
import math

from g_protocol import NATIVE_CONFIG, THRESHOLDS, STREAMS, integer, number, read, sha
from g_audit_helpers import graph_data, checked_solution, population_audit, strict_pair_audit, check_diversity


def artifact(root, value):
    root = Path(root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError('Snapshot/native artifact escaped root or is missing: ' + str(value))
    return path


def solution(record):
    """Native solution mirrors use integer ticks, never float objective coercion."""
    if not isinstance(record, dict):
        raise ValueError('Solution mirror is not an object')
    value = {'selected': record['selected'], 'ticks': record['ticks']}
    if 'raw' in record:
        value['raw'] = record['raw']
    return value


def population(data, records):
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError('Complete four-member population mirror is required')
    for record in records:
        checked_solution(data, solution(record))
    return population_audit(data, [item['selected'] for item in records])


def snapshot_audit(root, graph, entry, setup, fixture_expected_config=None):
    data = graph_data(root, graph, setup)
    blob = artifact(root, entry['snapshot_path'])
    meta_path = artifact(root, entry['metadata_path'])
    meta = read(meta_path)
    if meta.get('schema') != 'barr_fixed_state_meta_v1':
        raise ValueError('Unknown native full-state metadata schema')
    if meta['config'] != (NATIVE_CONFIG if fixture_expected_config is None else fixture_expected_config):
        raise ValueError('Snapshot configuration differs from the complete frozen E parameter definition')
    if sha(blob) != entry['snapshot_sha256'] or meta['snapshot_sha256'] != sha(blob) or entry.get('roundtrip_equal') is not True:
        raise ValueError('Snapshot identity or native complete round-trip validation failed')
    frozen = next(g for g in setup['graphs'] if g['id'] == graph)
    if meta['graph_sha256'] != frozen['native_sha256'] or meta['n'] != frozen['n'] or meta['m'] != frozen['m']:
        raise ValueError('Snapshot graph identity differs from frozen original input')
    members = population(data, meta['population'])
    archive = checked_solution(data, solution(meta['archive']))
    if archive['objective_ticks'] < frozen['initial_ticks']:
        raise ValueError('Snapshot archive regressed below the common initial archive')
    target = solution(meta['target'])
    target_audit = checked_solution(data, target)
    if meta.get('used_fusion') is True:
        fused = solution(meta['fused_state'])
        checked_solution(data, fused)
        if fused != target:
            raise ValueError('Captured post-fusion target differs from saved fused instance')
    elif not any(solution(item) == target for item in meta['population']):
        raise ValueError('Non-fused target is absent from the frozen population')
    action = meta['action']
    proof = strict_pair_audit(data, target['selected'], action['outside'], action['blockers'])
    if action['unary_gains_ticks'] != proof['unary_gain_ticks'] or integer(action['pair_gain_ticks'], 'pair gain') != proof['pair_gain_ticks']:
        raise ValueError('Native witness differs from original integer blocker proof')
    # Full local/search/controller inventory is emitted by native serialization;
    # independent Python verifies feasible mirrors, blob identity and inventory
    # presence; exact private-state replay is checked by native roundtrip tests.
    internals = list(meta['population']) + ([meta['fused_state']] if meta['used_fusion'] else [])
    for index, item in enumerate(internals):
        for key in ['private_inventory', 'private_sha256', 'rng_state', 'rng_sha256']:
            if key not in item or not item[key]:
                raise ValueError('Missing full private/RNG state inventory: member%d/%s' % (index, key))
        required = {'s', 'blocks', 'blocked_weight', 'queue', 'queued', 'undo', 'recording', 'protected_vertex'}
        if set(item['private_inventory']) != required or item['private_inventory']['s'] != frozen['n'] or any(item['private_inventory'][key] != frozen['n'] for key in ['blocks', 'blocked_weight', 'queued']):
            raise ValueError('Full local mutable inventory is incomplete or inconsistent')
        import hashlib
        if hashlib.sha256(item['rng_state'].encode()).hexdigest() != item['rng_sha256']:
            raise ValueError('Member RNG textual state hash mismatch')
    if not isinstance(meta.get('controller'), dict) or not meta['controller']:
        raise ValueError('Full controller/timing/cache state inventory is missing')
    required = {'round', 'epoch', 'stale', 'last_event_seconds', 'capture_elapsed_seconds', 'capture_remaining_seconds',
                'pending_prefix_seconds', 'prefix_normalized', 'escalation_seconds', 'threshold_seconds', 'scout_seconds',
                'allstate_sha256', 'non_rng_sha256', 'audit_rng_state', 'audit_rng_sha256', 'certified_masks', 'fused_certificate'}
    if set(meta['controller']) != required or meta['controller']['allstate_sha256'] != sha(blob) or meta['controller']['epoch'] != entry['epoch'] or meta['controller']['threshold_seconds'] != entry['threshold_seconds'] or meta['controller']['capture_elapsed_seconds'] != entry['capture_elapsed_seconds']:
        raise ValueError('Controller inventory/snapshot identity/slot metadata differs')
    if meta['controller']['prefix_normalized'] is not True or meta['controller']['escalation_seconds'] < meta['controller']['pending_prefix_seconds']:
        raise ValueError('Captured pending common cost prefix was not normalized exactly once')
    if hashlib.sha256(meta['controller']['audit_rng_state'].encode()).hexdigest() != meta['controller']['audit_rng_sha256']:
        raise ValueError('Captured audit RNG textual state hash differs')
    target_index = integer(meta['feedback_target_index'], 'snapshot feedback target index')
    if not 0 <= target_index < 4 or meta['target'] != (meta['fused_state'] if meta['used_fusion'] else meta['population'][target_index]):
        raise ValueError('Frozen exact intervention target differs from pending fused/member private state')
    caches = meta['controller']['certified_masks']
    if not isinstance(caches, list) or len(caches) != 4:
        raise ValueError('Frozen per-member certificate cache inventory is incomplete')
    for cache in caches + [meta['controller']['fused_certificate']]:
        checked = __import__('g_audit_helpers').membership_audit(data, cache)
        if not checked['feasible']:
            raise ValueError('Frozen cached certificate mask violates original resources')
    return {'verified': True, 'graph': graph, 'threshold_seconds': entry['threshold_seconds'],
            'capture_elapsed_seconds': entry['capture_elapsed_seconds'], 'epoch': entry['epoch'],
            'snapshot_path': blob.relative_to(Path(root).resolve()).as_posix(),
            'metadata_path': meta_path.relative_to(Path(root).resolve()).as_posix(),
            'snapshot_sha256': sha(blob), 'metadata_sha256': sha(meta_path),
            'archive_ticks': archive['objective_ticks'], 'target_ticks': target_audit['objective_ticks'],
            'population': members, 'witness': proof, 'used_fusion': bool(meta['used_fusion']),
            'private_state_validation_scope': 'full native roundtrip plus frozen inventory; independent original masks/objectives/edges/timelines/witness'}


def capture_audit(root, row, setup):
    native = read(artifact(root, row['native_output']))
    if native.get('schema') != 'barr_fixed_capture_v1':
        raise ValueError('Unknown capture output schema')
    if native['config'] != NATIVE_CONFIG:
        raise ValueError('Capture did not use the full frozen E configuration')
    data = graph_data(root, row['graph'], setup)
    frozen = next(g for g in setup['graphs'] if g['id'] == row['graph'])
    if native['seed'] != 101 or native['thresholds'] != THRESHOLDS or native['graph_sha256'] != frozen['native_sha256'] or native['n'] != frozen['n'] or native['m'] != frozen['m']:
        raise ValueError('Capture graph/seed/threshold identity differs from frozen protocol')
    terminal = checked_solution(data, {'selected': native['result']['selected'], 'ticks': native['result']['tick_value'], 'raw': native['result']['original_value']})
    if terminal['objective_ticks'] < frozen['initial_ticks']:
        raise ValueError('Capture terminal archive regressed below common initial')
    snapshots = native['snapshots']
    if not isinstance(snapshots, list) or len(snapshots) > 3:
        raise ValueError('Capture exceeded three prospective threshold slots')
    thresholds = [integer(entry['threshold_seconds'], 'threshold') for entry in snapshots]
    missing = native['missing_thresholds']
    if any(type(t) is not int for t in missing) or len(set(thresholds + missing)) != 3 or sorted(thresholds + missing) != THRESHOLDS:
        raise ValueError('Available plus missing slots do not equal the frozen complete slot table')
    if thresholds != THRESHOLDS[:len(thresholds)] or missing != THRESHOLDS[len(thresholds):]:
        raise ValueError('Capture did not fill earliest eligible unfilled thresholds in order')
    epochs, elapsed, audits = set(), -1, []
    for entry in snapshots:
        epoch = integer(entry['epoch'], 'snapshot epoch')
        current = number(entry['capture_elapsed_seconds'], 'capture elapsed')
        if epoch in epochs or current < entry['threshold_seconds'] or current < elapsed or current > 360.1:
            raise ValueError('Capture reused an epoch or violated threshold/time ordering')
        epochs.add(epoch); elapsed = current
        audits.append(snapshot_audit(root, row['graph'], entry, setup))
    return {'verified': True, 'snapshots': audits, 'missing_thresholds': missing, 'terminal_archive': terminal,
            'capture_snapshot_io_seconds': sum(number(entry['capture_io_seconds'], 'capture snapshot I/O') for entry in snapshots),
            'captured_sunk_scout_seconds': [entry['scout_seconds'] for entry in snapshots],
            'slots': [{'threshold_seconds': t, 'available': t in thresholds} for t in THRESHOLDS],
            'sampling_scope': 'strict positive double-negative known-action opportunities enriched after frozen thresholds; missing retained'}


def replay_audit(root, row, setup, fixture_expected_config=None):
    native = read(artifact(root, row['native_output']))
    if native.get('schema') != 'barr_fixed_replay_v1':
        raise ValueError('Unknown replay output schema')
    if native['config'] != (NATIVE_CONFIG if fixture_expected_config is None else fixture_expected_config) or native['budget_seconds'] != 360:
        raise ValueError('Replay configuration/fresh future allowance differs from frozen policy')
    if native['action'] != row['arm'] or native['future_seed'] != row['future_seed'] or native['future_seed'] not in STREAMS:
        raise ValueError('Replay action or prespecified future RNG stream changed')
    if native['snapshot_sha256'] != row['snapshot_sha256'] or sha(artifact(root, row['snapshot_path'])) != row['snapshot_sha256']:
        raise ValueError('Replay did not use the exact registered full snapshot')
    if native.get('roundtrip_equal') is not True or not isinstance(native['ready_state_sha256'], str) or len(native['ready_state_sha256']) != 64:
        raise ValueError('Replay full-state restore/readiness validation failed')
    data = graph_data(root, row['graph'], setup)
    meta = read(artifact(root, row['metadata_path']))
    if 'metadata_sha256' in row and sha(artifact(root, row['metadata_path'])) != row['metadata_sha256']:
        raise ValueError('Frozen snapshot metadata identity changed before replay audit')
    if native['graph_sha256'] != meta['graph_sha256'] or native['ready_non_rng_sha256'] != meta['controller']['non_rng_sha256']:
        raise ValueError('Replay rekey changed graph/structural/controller state')
    if row['future_seed'] == 0 and native['ready_state_sha256'] != meta['controller']['allstate_sha256']:
        raise ValueError('Stream0 did not restore the exact captured complete RNG state')
    if row['future_seed'] != 0 and native['ready_state_sha256'] == meta['controller']['allstate_sha256']:
        raise ValueError('A nonzero controlled future stream was not rekeyed')
    if native.get('io_paths_restored') is not False:
        raise ValueError('Replay must not reuse capture I/O paths')
    archives, populations = {}, {}
    for stage in ['before', 'after_action', 'terminal']:
        archives[stage] = checked_solution(data, solution(native['archive_' + stage]))
        populations[stage] = population(data, native['population_' + stage])
        check_diversity(native['diversity_' + stage], populations[stage])
    if solution(native['archive_before']) != solution(meta['archive']):
        raise ValueError('Fork archive_before differs from common captured archive')
    if [solution(item) for item in native['population_before']] != [solution(item) for item in meta['population']]:
        raise ValueError('Fork population_before differs from captured structural state')
    before = archives['before']['objective_ticks']
    after = archives['after_action']['objective_ticks']
    terminal = archives['terminal']['objective_ticks']
    if not before <= after <= terminal:
        raise ValueError('Archive monotonicity violated during action/continuation')
    witness = strict_pair_audit(data, meta['target']['selected'], meta['action']['outside'], meta['action']['blockers'])
    if native.get('after_action_includes_pending_feedback') is not True:
        raise ValueError('Initial-stage mirror must include the common pending-fusion feedback')
    target_index = integer(native['feedback_target_index'], 'feedback target index')
    if not 0 <= target_index < 4 or target_index != meta['feedback_target_index']:
        raise ValueError('Replay changed the frozen pending-feedback destination')
    expected_population = [solution(item) for item in meta['population']]
    expected_target = solution(meta['target']) if row['arm'] == 'continue' else {'selected': witness['after_selected'], 'ticks': witness['after_ticks']}
    if meta['used_fusion'] or row['arm'] == 'recover':
        if expected_target['ticks'] >= expected_population[target_index]['ticks']:
            expected_population[target_index] = expected_target
    if [solution(item) for item in native['population_after_action']] != expected_population:
        raise ValueError('Initial-stage population differs from known action and common pending feedback')
    if any(native['target_after_action'][key] != expected_target[key] for key in ['selected', 'ticks']):
        raise ValueError('Initial-stage intervention target mirror differs from known pair/skip')
    checked_solution(data, solution(native['target_after_action']))
    expected_feedback = meta['used_fusion'] and expected_target['ticks'] >= meta['population'][target_index]['ticks']
    if native['feedback_applied'] != expected_feedback:
        raise ValueError('Common pending-feedback application flag differs from frozen >= policy')
    if row['arm'] == 'continue' and solution(native['archive_after_action']) != solution(native['archive_before']):
        raise ValueError('Control branch changed the already-refreshed frozen archive before continuation')
    if row['arm'] == 'recover' and after != max(before, witness['after_ticks']):
        raise ValueError('Recover branch archive refresh differs from known pair only')
    stats = native['future_stats']
    for key in ['joint_scout_calls', 'kernel_calls', 'joint_recovery_calls']:
        if integer(stats[key], key) != 0:
            raise ValueError('A fork branch invoked prohibited future joint/Kernel work')
    for key in ['native_seconds', 'cpu_seconds', 'action_seconds', 'continuation_seconds']:
        number(native[key], key)
    if (fixture_expected_config is None and native['native_seconds'] < 359.99) or native['action_seconds'] + native['continuation_seconds'] > native['native_seconds'] + .05:
        raise ValueError('Fresh future budget or native stage timing is invalid')
    return {'verified': True, 'ready_state_sha256': native['ready_state_sha256'],
            'ready_non_rng_sha256': native['ready_non_rng_sha256'],
            'snapshot_sha256': native['snapshot_sha256'], 'archive_before_ticks': before,
            'archive_after_action_ticks': after, 'archive_terminal_ticks': terminal,
            'direct_action_archive_gain_ticks': after - before, 'future_archive_gain_ticks': terminal - before,
            'witness': witness, 'populations': populations, 'future_stats': stats,
            'action_seconds': native['action_seconds'], 'continuation_seconds': native['continuation_seconds'],
            'capture_elapsed_seconds': native['capture_elapsed_seconds'],
            'capture_sunk_scout_seconds': native['capture_sunk_scout_seconds'],
            'capture_sunk_prefix_seconds': native['capture_sunk_prefix_seconds'],
            'pair_apply_seconds': native['pair_apply_seconds'], 'feedback_seconds': native['feedback_seconds']}
