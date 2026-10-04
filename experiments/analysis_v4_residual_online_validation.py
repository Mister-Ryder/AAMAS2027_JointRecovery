"""Independent complete-grid residual validation replay (stdlib and NumPy).

No fitting, Torch, shared scientific controller, solver, generator or network
is imported. Shared executed greedy improvements and subsequent native extra
are audited separately. This is development validation, never confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict, deque
import csv
import hashlib
import io
import json
from math import fsum, isclose, isfinite
from pathlib import Path, PurePosixPath
import pickle
import re
import struct
import tarfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('ResidualCapacity', 'ResidualFreeOccupancy', 'ResidualNoAux',
            'ResidualNoWarmMembership', 'ResidualCheapSummary')
CLASSICS = ('Immediate', 'Upper', 'P1', 'Lower', 'RoundRobin', 'AnytimeGreedyPortfolio')
SEEDS = (17, 29, 43)
DEADLINES = (.139, .556, 2.221)
POINTS = {'slice0.01': (.01, .035), 'slice0.05': (.05, .075), 'slice0.2': (.2, .214)}
CALIBRATION_SHA256 = '495dab54ad7ca75e8a543338eb71da1049e20afdd35ad18825669d37d54c5bec'
CHILS_SHA256 = '19610c03f334c6267f94543ad3053d792cba56e9ae211fceb6c36f21750c88a0'
MATERIALITY = .005


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: '+key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf8'), parse_constant=invalid, object_pairs_hook=pairs)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finite(value, message, minimum=None):
    require(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
            and isfinite(float(value)) and (minimum is None or value >= minimum), message)
    return float(value)


def equal(actual, expected, message, native=False):
    if native:
        require(isinstance(actual, (int, np.integer)) and not isinstance(actual, (bool, np.bool_))
                and int(actual) == int(expected), message)
    else:
        require(isclose(finite(actual, message), finite(expected, message), rel_tol=1e-12, abs_tol=1e-9), message)


def flag(record, key):
    require(isinstance(record[key], bool), 'Exact boolean required: ' + key)
    return record[key]


def ids(values, n, message):
    require(isinstance(values, (list, tuple, np.ndarray)), message)
    require(not isinstance(values, np.ndarray) or values.ndim == 1, message)
    values = tuple(values)
    require(all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values), message)
    result = tuple(map(int, values))
    require(len(result) == len(set(result)) and all(0 <= v < n for v in result), message)
    return result


def objective(values, graph):
    weights = graph['weights']
    return sum(int(weights[v]) for v in sorted(values)) if graph['native'] else fsum(float(weights[v]) for v in sorted(values))


def membership(values, graph, message='Original membership', subset=None, base=()):
    chosen = frozenset(ids(values, len(graph['weights']), message))
    base = frozenset(base)
    require(subset is None or chosen <= frozenset(subset), message + ': outside scope')
    require(not chosen & base, message + ': overlaps base')
    full = chosen | base
    require(not any(graph['adjacency'][v] & full for v in full), message + ': original-edge conflict')
    return chosen, objective(chosen, graph)


def grouped(values, ptr):
    values, ptr = np.asarray(values), np.asarray(ptr)
    require(values.ndim == 1 and values.dtype.kind in 'iu' and ptr.ndim == 1 and ptr.dtype.kind in 'iu'
            and len(ptr) and ptr[0] == 0 and ptr[-1] == len(values) and np.all(ptr <= len(values))
            and np.all(ptr[1:] >= ptr[:-1]), 'Exact monotonic membership pointers required')
    return [values[int(a):int(b)] for a, b in zip(ptr, ptr[1:])]


def bound_file(root, relative, binding):
    root = Path(root).resolve(); relative = PurePosixPath(relative)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts and '\\' not in str(relative), 'Unsafe bound file path')
    path = root.joinpath(*relative.parts)
    require(path.is_file() and not path.is_symlink() and root in path.resolve().parents, 'Bound file absent/redirected: ' + str(relative))
    require(re.fullmatch('[0-9a-f]{64}', binding['sha256']) is not None and sha(path) == binding['sha256'], 'Bound file SHA differs: ' + str(relative))
    if 'bytes' in binding:
        require(isinstance(binding['bytes'], int) and not isinstance(binding['bytes'], bool)
                and path.stat().st_size == binding['bytes'], 'Bound file size differs')
    return path


def graph_from_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {'weights', 'agents', 'edges', 'selected', 'cliques_flat', 'cliques_ptr'}, 'Original observable NPZ schema differs')
        arrays = {key: archive[key].copy() for key in archive.files}
    w, owners, edges = (arrays[key] for key in ('weights', 'agents', 'edges'))
    require(w.ndim == 1 and w.dtype.kind in 'iuf' and np.isfinite(w).all() and (w >= 0).all(), 'Original nonnegative finite rewards required')
    require(owners.shape == w.shape and owners.dtype.kind in 'iu' and (owners >= 0).all(), 'Original exact agent IDs required')
    require(edges.ndim == 2 and edges.shape[1:] == (2,) and edges.dtype.kind in 'iu', 'Original exact edge pairs required')
    pairs = [tuple(map(int, row)) for row in edges]
    require(pairs == sorted(set(pairs)) and all(0 <= u < v < len(w) for u, v in pairs), 'Original canonical edges required')
    adjacent = [set() for _ in w]
    for u, v in pairs:
        adjacent[u].add(v); adjacent[v].add(u)
    graph = dict(weights=w, agents=owners, adjacency=tuple(map(frozenset, adjacent)), native=w.dtype.kind in 'iu')
    original, initial = membership(arrays['selected'], graph)
    cliques = []
    for raw in grouped(arrays['cliques_flat'], arrays['cliques_ptr']):
        clique = ids(raw, len(w), 'Original resource clique')
        require(all(v in adjacent[u] for i, u in enumerate(clique) for v in clique[i+1:]), 'Original factor is not a conflict clique')
        cliques.append(clique)
    graph.update(original=original, initial=initial, cliques=tuple(cliques))
    identity = sha_json(dict(schema='v4_native_graph_v1', rewards_dtype=str(w.dtype), weights=[x.item() for x in w], agents=[int(x) for x in owners], edges=pairs))
    return graph, identity


def specifications():
    cells = [dict(domain='menu', replacements=n, menus=8, density=d, coupling=.04, topology=t)
             for n in (64, 256) for t in ('erdos', 'components', 'bipartite') for d in (.08, .20, .45)]
    cells += [dict(domain='resource', satellites=8, grounds=g, passes=p, alternatives=6) for p in (48, 96) for g in (2, 4, 8)]
    return [dict(cell, split=split, seed=start+repeat*24+i)
            for split, start, repeats in (('training', 20420000, 3), ('validation', 20430000, 1))
            for repeat in range(repeats) for i, cell in enumerate(cells)]


def check_source(manifest, source_root, expected_manifest_sha=None):
    files = manifest['files']; names = [row['path'] for row in files]
    require(names == sorted(set(names)), 'Unique sorted source closure required')
    if expected_manifest_sha is not None:
        path = Path(source_root)/'source_manifest.json'
        require(sha(path) == expected_manifest_sha, 'Original immutable source manifest bytes differ')
        require(read(path)['files'] == files, 'Original source manifest file registry differs')
    for row in files:
        bound_file(source_root, row['path'], row)
    declared = {row['path']: row['sha256'] for row in files}
    if 'capsule' in manifest:
        require(manifest['capsule'] == sha_json(files), 'Scientific capsule identity differs')
    for origin in manifest.get('execution_origins', ()):
        require(declared.get(origin['path']) == origin['sha256'], 'Recorded scientific import origin differs')
    return declared


def check_collection(root, protocol_sha, completion_sha, source_root):
    root = Path(root).resolve()
    require(sha(root/'protocol.json') == protocol_sha and sha(root/'completion.json') == completion_sha, 'Online-bound new collection differs')
    p, c = read(root/'protocol.json'), read(root/'completion.json')
    require(p['status'] == 'frozen_actual_fresh_label_queue' and p['complete_specs'] == specifications()
            and p['splits'] == ['training', 'validation'] and p['state_weight'] == .5
            and p.get('executed_common_greedy_prefix') is True and p.get('fixed_scope_cap') == 256
            and p['model_fits_started'] == p['confirmation_graphs_generated'] == 0, 'New original collection registry/protocol differs')
    require(c['status'] == 'complete_fresh_label_coverage_not_fitting' and c['expected_cases'] == 96
            and len(c['cases']) == 96 and c['protocol_sha256'] == protocol_sha
            and c['model_fits_started'] == c['confirmation_graphs_generated'] == 0
            and c['includes_failed_and_no_startable_cases'] is True, 'Complete new96 collection required')
    require(read(root/'progress.json')['records'] == c['cases'], 'Complete collection progress differs')
    source = check_source(p['source_capsule'], source_root, p['pre_collection_gate']['source_manifest_sha256'])
    for origin in p['execution_origins']:
        require(source.get(origin['path']) == origin['sha256'], 'Actual original collection import origin differs')
    barrier = p['pre_collection_gate']; gate_root = root/barrier['directory']
    require(barrier['directory'] == 'pre_collection_gate' and sha(gate_root/'completion.json') == barrier['completion_sha256'], 'Original precollection source gate binding differs')
    gate = read(gate_root/'completion.json')
    require(gate['status'] == barrier['status'] == 'complete_residual_pre_collection_source_runtime_gate_not_quality'
            and gate['tests_run'] == barrier['tests_run'] == 73 and gate['files'] == barrier['files']
            and gate['source_manifest_sha256'] == barrier['source_manifest_sha256'], 'Original precollection gate metadata differs')
    require(set(gate['files']) == {'protocol.json', 'guards.txt', 'guards.json', 'source_origins.json'}, 'Exact original precollection gate artifacts required')
    for name, binding in gate['files'].items():
        bound_file(gate_root, name, binding)
    guards = read(gate_root/'guards.json')
    require(guards['expected_tests'] == guards['tests_run'] == 73
            and all(guards[k] == 0 for k in ('failures', 'errors', 'skipped', 'unexpected_successes', 'expected_failures'))
            and guards['log_sha256'] == sha(gate_root/'guards.txt'), 'Original precollection guards contain a failure/skip')
    for origin in read(gate_root/'source_origins.json')['origins']:
        require(source.get(origin['path']) == origin['sha256'], 'Original precollection gate actual import origin differs')
    require(p['calibration_sha256'] == CALIBRATION_SHA256 and p['calibration']['caps'] == [64, 256, 1024], 'Parent calibration metadata differs')
    graphs, files = {}, {}
    for i, (entry, spec) in enumerate(zip(c['cases'], specifications())):
        cid = 'case_%03d' % i
        require(entry['case_id'] == cid and entry['spec'] == spec, 'Original collection graph replaced')
        require({'spec.json', 'labels.json'} <= set(entry['files']) <= {'spec.json', 'labels.json', 'observable.npz'}, 'Original collection file registry differs')
        folder = root/'cases'/cid
        paths = {name: bound_file(folder, name, binding) for name, binding in entry['files'].items()}
        require(read(paths['spec.json']) == spec, 'Original case specification differs')
        labels = read(paths['labels.json'])
        require(labels['spec'] == spec and labels['status'] == entry['status'], 'Original labels binding differs')
        if 'observable.npz' in paths:
            graph, identity = graph_from_npz(paths['observable.npz'])
            require(identity == entry['graph_sha256'], 'Original graph identity differs')
        else:
            require(entry['status'] == 'fresh_declared_graph_collection_failed', 'Missing original graph is not a declared failed slot')
            graph = None
        graphs[cid] = (entry, graph); files[cid] = entry['files']
    return p, graphs, files


class _NumericDtype:
    """Primitive NumPy scalar metadata only; arrays/objects stay unsupported."""
    formats = {'f8': 'd', 'f4': 'f', 'i8': 'q', 'i4': 'i',
               'u8': 'Q', 'u4': 'I', 'b1': '?'}

    def __init__(self, code, align=False, copy=False):
        require(code in self.formats and isinstance(align, bool) and isinstance(copy, bool),
                'Numeric checkpoint scalar dtype required')
        self.code, self.byteorder = code, '='

    def __setstate__(self, state):
        require(isinstance(state, tuple) and len(state) == 8 and state[0] == 3
                and state[1] in ('<', '>', '=', '|') and state[2:5] == (None, None, None)
                and state[5:] == (-1, -1, 0), 'Primitive checkpoint dtype state required')
        self.byteorder = state[1]


def _numeric_scalar(dtype, payload):
    require(isinstance(dtype, _NumericDtype) and isinstance(payload, bytes),
            'Numeric checkpoint scalar payload required')
    fmt = ('=' if dtype.byteorder == '|' else dtype.byteorder) + dtype.formats[dtype.code]
    require(len(payload) == struct.calcsize(fmt), 'Checkpoint scalar byte length differs')
    value = struct.unpack(fmt, payload)[0]
    require(isfinite(value), 'Finite checkpoint scalar metadata required')
    return value


def _latin1_bytes(value, encoding, errors='strict'):
    require(isinstance(value, str) and encoding in ('latin1', 'latin-1') and errors == 'strict',
            'Only protocol2 latin1 checkpoint bytes supported')
    return value.encode('latin1')


class _MetadataOnly(pickle.Unpickler):
    """Restricted own-checkpoint metadata reader; tensors remain opaque stubs."""
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if module == 'torch' and name.endswith('Storage'):
            return ('opaque_storage_type', name)
        if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2'):
            return lambda *args: ('opaque_tensor', args[1:4])
        if (module, name) == ('numpy', 'dtype'):
            return _NumericDtype
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == 'scalar':
            return _numeric_scalar
        if (module, name) == ('_codecs', 'encode'):
            return _latin1_bytes
        raise ValueError('Unexpected checkpoint pickle global: %s.%s' % (module, name))

    def persistent_load(self, key):
        require(isinstance(key, tuple) and key[0] == 'storage', 'Unexpected checkpoint persistent object')
        return ('opaque_storage', key[2:])


def checkpoint_metadata(path):
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        require(len(candidates) == 1, 'One own checkpoint metadata stream required')
        payload = archive.read(candidates[0])
    record = _MetadataOnly(io.BytesIO(payload)).load()
    require(isinstance(record, dict), 'Own checkpoint metadata dictionary required')
    return record


def check_fits(root, online, collection_files, source_root):
    root = Path(root).resolve()
    require(sha(root/'protocol.json') == online['fit_protocol_sha256'] and sha(root/'completion.json') == online['fit_completion_sha256'], 'Online-bound full fit bytes differ')
    p, c = read(root/'protocol.json'), read(root/'completion.json')
    require(p['status'] == 'frozen_fixed_fit_suite_before_validation_selection' and p['variants'] == list(VARIANTS)
            and p['fit_seeds'] == list(SEEDS) and p['epochs'] == 40 and p['all_seeds_retained'] is True
            and p['no_online_validation'] is True and p['confirmation_graphs_generated'] == 0, 'Fixed all15 fit registry differs')
    require(c['status'] == 'complete_all_predeclared_fits_not_policy_evidence' and c['protocol_sha256'] == online['fit_protocol_sha256']
            and c['all_fit_seeds_retained'] is True and c['no_online_validation'] is True
            and c['confirmation_graphs_generated'] == 0, 'Complete all15 fit receipt required')
    bindings = p['bindings']
    for key, field in (('collection_protocol_sha256', 'collector_protocol_sha256'), ('collection_completion_sha256', 'collector_completion_sha256')):
        require(bindings[key] == online[field], 'Fit/new label provenance differs')
    require(bindings['calibration_sha256'] == CALIBRATION_SHA256, 'Fit calibration differs')
    require(bindings['training_case_files'] == {cid: value for cid, value in collection_files.items() if int(cid[-3:]) < 72}
            and bindings['validation_case_files'] == {cid: value for cid, value in collection_files.items() if int(cid[-3:]) >= 72}, 'All96 fit/collection file bindings differ')
    check_source(bindings['fit_source_capsule'], source_root, bindings['fit_source_manifest_sha256'])
    expected = {(v, s) for v in VARIANTS for s in SEEDS}
    entries = {(row['variant'], row['fit_seed']): row for row in c['fits']}
    records = {(row['variant'], row['fit_seed']): row for row in online['checkpoint_records']}
    require(len(entries) == len(c['fits']) == len(records) == len(online['checkpoint_records']) == 15
            and set(entries) == set(records) == expected, 'Exactly all15 bound fit/checkpoint pairs required')
    for pair, entry in entries.items():
        v, seed = pair; folder = root/(v+'_seed%d' % seed); record = records[pair]
        require(set(entry['files']) == {'selected.pt', 'epoch40.pt', 'history.json'}, 'Exactly three declared fit artifacts required')
        paths = {name: bound_file(folder, name, b) for name, b in entry['files'].items()}
        require(sha(folder/'completion.json') == entry['completion_sha256'] == record['completion_sha256'], 'Child fit completion differs')
        child, history = read(folder/'completion.json'), read(paths['history.json'])
        expected_config = dict(variant=v, fit_seed=seed, epochs=40, batch_graphs=4, learning_rate=.001,
                               auxiliary_weight=.05, ranking_weight=1., temperature=.2, hidden=32, layers=2)
        require(child['status'] == 'complete_fixed_fit_not_online_policy_validation' and child['files'] == entry['files']
                and child['bindings'] == bindings and child['variant'] == v and child['fit_seed'] == seed
                and child['all40_epochs_retained'] is True and history['bindings'] == bindings
                and history['config'] == expected_config and history['fit_seed'] == seed, 'Child fit registry differs')
        epochs = history['epochs']
        require(len(epochs) == 40 and [e['epoch'] for e in epochs] == list(range(1, 41)), 'Every40 epoch is required')
        values = [finite(e['validation']['graph_domain_balanced_regret'], 'Finite complete validation regret', 0) for e in epochs]
        selected = min(range(40), key=lambda i: (values[i], i))+1
        require(selected == entry['selected_epoch'] == history['selected_epoch'] == record['selected_epoch'], 'First-minimum selected epoch differs')
        require(record['checkpoint_relative'] == v+'_seed%d/selected.pt' % seed
                and record['checkpoint_sha256'] == entry['files']['selected.pt']['sha256']
                and record['history_sha256'] == entry['files']['history.json']['sha256'], 'Online selected checkpoint differs')
        for name, epoch in (('selected.pt', selected), ('epoch40.pt', 40)):
            metadata = checkpoint_metadata(paths[name])
            require(metadata['schema'] == 'v4_fixed_scope_joint_residual_fit_v1' and metadata['variant'] == v
                    and metadata['fit_seed'] == seed and metadata['epoch'] == epoch and metadata['bindings'] == bindings
                    and metadata['config'] == history['config'] and metadata['no_online_validation'] is True
                    and metadata['no_confirmation'] is True, 'Checkpoint independent metadata differs')
            equal(metadata['selection_metric_value'], values[epoch-1], 'Checkpoint epoch metric differs')
    return 15


def coordination_cells(graph, seed):
    """Independent observable C/E menu replay; no solver or graph construction."""
    S, adjacent, n = graph['original'], graph['adjacency'], len(graph['weights'])
    rng = np.random.default_rng(seed)
    available = [v for v in range(n) if v not in S]
    ranked = sorted(available, key=lambda v: (-(objective((v,), graph)-objective(adjacent[v] & S, graph)), v))
    leading = set(ranked[:32]); remaining = [v for v in available if v not in leading]
    rng.shuffle(remaining); pool = ranked[:32]+remaining[:32]
    commits = []
    for v in pool:
        if not adjacent[v] & frozenset(commits):
            commits.append(v)
        if len(commits) >= 8:
            break
    anchors = sorted(S); rng.shuffle(anchors); result = []; seen = {}
    for target_c in (0, 2, 4, 8):
        for target_e in (0, 16, 64):
            if target_c == target_e == 0:
                continue
            cell = dict(target_commitments=target_c, target_extra_releases=target_e)
            if len(commits) < target_c:
                result.append((None, dict(cell, status='insufficient_compatible_commitments'))); continue
            C = tuple(commits[:target_c]); mandatory = frozenset(v for c in C for v in adjacent[c] & S)
            E = set(); queue = deque(sorted(mandatory) or anchors[:1]); queued = set(queue)
            while len(E) < target_e:
                if not queue:
                    remaining_anchors = [v for v in anchors if v not in queued]
                    if not remaining_anchors:
                        break
                    queue.append(remaining_anchors[0]); queued.add(remaining_anchors[0])
                v = queue.popleft()
                if v not in mandatory:
                    E.add(v)
                adjacent_selected = set()
                for u in adjacent[v]-S:
                    adjacent_selected.update(adjacent[u] & S)
                next_ids = sorted(adjacent_selected-queued); rng.shuffle(next_ids)
                queue.extend(next_ids); queued.update(next_ids)
            if not C and not E:
                result.append((None, dict(cell, status='no_incumbent_releases_available'))); continue
            identity = (tuple(sorted(C)), tuple(sorted(E)))
            if len({int(graph['agents'][v]) for v in mandatory | E | frozenset(C)}) < 2:
                result.append((None, dict(cell, status='single_partition_only'))); continue
            cell.update(actual_commitments=len(C), actual_extra_releases=len(E), mandatory_displacements=len(mandatory))
            if identity in seen:
                result.append((None, dict(cell, status='actual_action_alias', alias_cell=seen[identity]))); continue
            seen[identity] = len(result)
            result.append((dict(inserts=list(identity[0]), releases=list(identity[1])), dict(cell, status='prepared')))
    require(len(result) == 11, 'Independent eleven-cell menu invariant')
    return result


def scope_replay(record, action, index, graph):
    S = graph['original']; adjacent = graph['adjacency']; n = len(graph['weights'])
    C = frozenset(ids(action['inserts'], n, 'Commitments')); E = frozenset(ids(action['releases'], n, 'Extra releases'))
    require(not C & S and E <= S, 'Original C/E membership differs')
    membership(tuple(C), graph, 'Compatible commitments')
    D = frozenset(v for c in C for v in adjacent[c] & S) | E; B = (S-D) | C
    membership(tuple(B), graph, 'Fixed committed base')
    candidates = set(D)
    for v in D:
        candidates.update(adjacent[v])
    eligible = {v for v in candidates if v not in B and not adjacent[v] & B}
    known = eligible & S
    ordered = tuple(sorted(known))+tuple(sorted(eligible-known, key=lambda v: (-graph['weights'][v].item(), v)))
    R = ordered[:256]; pool = frozenset(R)
    resources = [tuple(v for v in factor if v in pool) for factor in graph['cliques'] if pool.intersection(factor)]
    resources = tuple(factor for factor in resources if len(factor) > 1)
    require(record['action_index'] == index and record['cap'] == 256 and tuple(record['replacements']) == R
            and frozenset(record['base']) == B and frozenset(record['displaced']) == D
            and tuple(record['inserts']) == tuple(sorted(C)) and record['eligible_before_cap'] == len(eligible)
            and tuple(map(tuple, record['resource_cliques'])) == resources, 'Original fixedR256 scope/order/base differs')
    if 'releases' in record:
        require(tuple(record['releases']) == tuple(sorted(E)) and tuple(map(tuple, record['action_identity'])) == (tuple(sorted(C)), tuple(sorted(E))), 'Scope release/action identity differs')
    q = objective(C, graph)-objective(D, graph)
    equal(record['immediate_gain'], q, 'Exact signed q differs', graph['native'])
    return dict(action_index=index, cap=256, base=B, displaced=D, replacements=R, resource_cliques=resources, immediate_gain=q)


def executed_warm(scope, graph):
    pool = frozenset(scope['replacements']); adjacent = graph['adjacency']; weights = graph['weights']
    floor, _ = membership(tuple(graph['original'] & pool), graph, 'Original warm floor', pool, scope['base'])
    candidates = [floor]; degree = {v: len(adjacent[v] & pool) for v in pool}
    for exponent in (0., .5, 1.):
        order = sorted(pool, key=lambda v: (-weights[v].item() if exponent == 0. else
                        -float(weights[v])/(1+degree[v])**exponent, -weights[v].item(), v))
        selected = set()
        for v in order:
            if not adjacent[v] & selected:
                selected.add(v)
        checked, _ = membership(tuple(selected), graph, 'Actually executed greedy warm', pool, scope['base'])
        candidates.append(checked)
    selected = max(candidates, key=lambda candidate: (objective(candidate, graph), tuple(sorted(candidate))))
    return selected, tuple(candidates)


def clique_upper(scope, graph):
    """Observable greedy clique cover + disjoint partition; no optimality claim."""
    nodes = scope['replacements']; local = {v: i for i, v in enumerate(nodes)}
    neighbors = [frozenset(local[u] for u in graph['adjacency'][v] if u in local) for v in nodes]
    order = sorted(range(len(nodes)), key=lambda i: (-graph['weights'][nodes[i]].item(), -len(neighbors[i]), i))
    factors = set(); covered = set()
    edges = {(i, j) for i, row in enumerate(neighbors) for j in row if i < j}
    for seed in order:
        if not neighbors[seed]:
            continue
        clique = [seed]; allowed = set(neighbors[seed])
        for vertex in order:
            if vertex in allowed:
                clique.append(vertex); allowed.intersection_update(neighbors[vertex])
        factor = tuple(sorted(clique)); factors.add(factor)
        covered.update((u, v) for i, u in enumerate(factor) for v in factor[i+1:])
    factors.update(edges-covered)
    factors.update(tuple(sorted(local[v] for v in factor)) for factor in scope['resource_cliques'])
    left = set(range(len(nodes))); upper = 0
    for factor in sorted(factors, key=lambda group: (-len(group), group)):
        require(all(v in neighbors[u] for i, u in enumerate(factor) for v in factor[i+1:]), 'Inferred upper bound factor is not a clique')
        block = left.intersection(factor)
        if block:
            vertex = max(block, key=lambda v: graph['weights'][nodes[v]])
            upper += objective((nodes[vertex],), graph); left.difference_update(block)
    upper += objective(tuple(nodes[i] for i in left), graph)
    return upper


def absolute_boundary(row, trace):
    start = finite(trace['started_timestamp'], 'Actual outer start timestamp')
    D = finite(row['deadline_seconds'], 'Original total D', 0)
    deadline = finite(trace['deadline_timestamp'], 'Absolute outer deadline')
    require(deadline == start+D, 'Actual absolute deadline differs from original start+D')
    caller = finite(row['actual_interface_receipt_timestamp'], 'Actual caller receipt timestamp')
    final = finite(row['final_validation_ready_timestamp'], 'Actual full-validation timestamp')
    require(start <= final <= caller, 'Actual caller precedes paid final validation')
    require(row['actual_interface_receipt_seconds'] == caller-start and row['outer_return_seconds'] == caller-start
            and row['final_validation_ready_seconds'] == final-start, 'Exact original elapsed subtraction differs')
    require(flag(row, 'outer_deadline_miss') == (caller >= deadline), 'Absolute caller boundary differs')
    validated = flag(row, 'final_selected_valid') and final < deadline
    require(flag(row, 'validation_ready_before_D') == validated, 'Absolute final validation boundary differs')
    timely_return = validated and caller < deadline
    require(flag(row, 'final_return_validated_by_D') == timely_return, 'Actual caller/final validation strict returned boundary differs')
    return start, deadline, caller, timely_return


def prefix_decomposition(row, admissions, original):
    timely = [a for a in admissions if a['on_time']]
    prefix = [a for a in timely if a['kind'] == 'common_prefix']
    last = prefix[-1] if prefix else dict(selected=original, gain=0)
    gain = last['gain']; certified = timely[-1] if timely else dict(selected=original, gain=0)
    equal(row['common_prefix_certified_gain'], gain, 'Actual timely common prefix gain differs')
    equal(row['certified_gain'], certified['gain'], 'Last timely certified gain differs')
    equal(row['certified_gain_above_prefix'], max(0, certified['gain']-gain), 'Source extra above actual prefix differs')
    equal(row['returned_prefix_component'], min(gain, row['final_return_gain']), 'Strict returned prefix component differs')
    equal(row['returned_gain_above_prefix'], max(0, row['final_return_gain']-gain), 'Strict returned extra above actual prefix differs')
    equal(row['returned_prefix_component']+row['returned_gain_above_prefix'], row['final_return_gain'], 'Strict returned gain decomposition fails')
    return frozenset(last['selected']), frozenset(certified['selected'])


def check_fetch(root, receipt_path, expected_sha, archive=None):
    receipt_path = Path(receipt_path).resolve(); base = receipt_path.parent
    require(sha(receipt_path) == expected_sha, 'Caller-pinned transport receipt differs')
    receipt = read(receipt_path)
    require(re.fullmatch(r'v4_[A-Za-z0-9_-]+', receipt['job']) is not None
            and re.fullmatch('[0-9a-f]{64}', receipt['capsule']) is not None
            and receipt['scientific_result_modified'] is False, 'Exact original transport job/capsule identity required')
    files = receipt['fetched']; names = [f['path'] for f in files]
    require(len(names) == len(set(names)), 'Duplicate fetch member')
    for item in files:
        bound_file(base, item['path'], item)
    require(root.parent == base and root.name == 'out', 'Run must be exact fetched out directory')
    required = {'out/protocol.json', 'out/completion.json', 'launch.json'}
    require(required <= set(names), 'Transport omitted hard run bindings')
    launch = read(base/'launch.json')
    require(launch['capsule'] == receipt['capsule'], 'Actual launch/capsule differs')
    archive = Path(archive) if archive else base/'TRANSPORT.tar.gz'
    require(sha(archive) == receipt['transport_archive_sha256'], 'Transport archive bytes differ')
    expected = {item['path']: item for item in files}; seen = set()
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            name = member.name; rel = PurePosixPath(name)
            require(member.isfile() and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name
                    and name == rel.as_posix() and name in expected and name not in seen, 'Unsafe/unknown transport member')
            payload = bundle.extractfile(member).read(); item = expected[name]
            require(len(payload) == item['bytes'] and hashlib.sha256(payload).hexdigest() == item['sha256'], 'Original archive member differs')
            seen.add(name)
    require(seen == set(expected), 'Incomplete transport archive')
    return receipt


def replay_unit(receipt, masks, case, graph):
    counters = Counter(units=1)
    require(finite(receipt['graph_weight'], 'Fixed graph mass', 0) == case['graph_weight']
            and receipt['domain'] == case['spec']['domain'], 'Actual case mass/domain differs')
    if graph is None:
        require(receipt['status'] == 'declared_input_unavailable' and receipt['trace'] is None
                and receipt['source_receipt'] is None and receipt['initial_value'] is None
                and receipt['final_return_gain'] == receipt['certified_gain'] == 0
                and receipt['final_return_relative_gain'] == receipt['certified_relative_gain'] == 0
                and receipt['final_return_validated_by_D'] is False, 'Unavailable graph obtained nonzero quality credit')
        require(all(len(masks[key]) == 0 for key in ('original_selected', 'final_return_selected', 'certified_selected', 'common_prefix_selected', 'source_selected')), 'Unavailable input fabricated memberships')
        counters['unavailable'] += 1
        return counters
    original, initial, native = graph['original'], graph['initial'], graph['native']
    require(receipt['status'] in ('complete_unit', 'unit_failed_retained'), 'Actual unit status differs')
    equal(receipt['initial_value'], initial, 'Original objective differs', native)
    require(membership(masks['original_selected'], graph)[0] == original, 'Original incumbent differs')
    checked = {}
    for name in ('final_return_selected', 'certified_selected', 'common_prefix_selected', 'source_selected'):
        checked[name] = membership(masks[name], graph, name)
        counters['full_memberships'] += 1
        if name in receipt:
            require(checked[name][0] == frozenset(ids(receipt[name], len(graph['weights']), name)), 'JSON/NPZ membership differs')
    for name, gain_key in (('final_return_selected', 'final_return_gain'), ('certified_selected', 'certified_gain'), ('common_prefix_selected', 'common_prefix_certified_gain')):
        equal(receipt[gain_key], checked[name][1]-initial, 'Original membership objective gain differs', native)
        require(receipt[gain_key] >= 0, 'Incumbent was not preserved')
    for prefix in ('final_return', 'certified'):
        equal(receipt[prefix+'_relative_gain'], receipt[prefix+'_gain']/max(1, initial), 'Original objective normalization differs')
    trace = receipt['trace']
    require(trace is not None, 'Available graph requires actual decision clock ledger')
    start, deadline, caller, timely_return = absolute_boundary(receipt, trace)
    source = receipt['source_receipt']
    if timely_return:
        require(source is not None and checked['final_return_selected'][0] == checked['source_selected'][0], 'Strict timely membership differs from actual source')
    else:
        require(checked['final_return_selected'][0] == original and receipt['final_return_gain'] == 0, 'Late/failed actual caller received quality credit')
        counters['strict_zero_return'] += 1
    if receipt['status'] == 'unit_failed_retained':
        require(not timely_return and receipt['final_return_gain'] == 0, 'Failed unit gained primary credit')
        counters['failed'] += 1
    counters['caller_late'] += caller >= deadline
    admission_masks = grouped(masks['admission_selected'], masks['admission_ptr'])
    admissions = trace['admissions']
    require(len(admission_masks) == len(admissions), 'Complete admission membership grid differs')
    previous_ready, previous_gain = start, 0
    for values, admission in zip(admission_masks, admissions):
        selected, value = membership(values, graph, 'Paid admission')
        require(selected == frozenset(ids(admission['selected'], len(graph['weights']), 'Admission JSON')), 'Admission membership differs')
        equal(admission['gain'], value-initial, 'Admission original objective differs', native)
        require(admission['gain'] > previous_gain, 'Admission is not an actual strict incumbent improvement')
        ready = finite(admission['ready_timestamp'], 'Admission absolute readiness')
        require(previous_ready <= ready <= caller and admission['ready_elapsed'] == ready-start, 'Actual monotonic admission timestamp differs')
        require(flag(admission, 'on_time') == (ready < deadline), 'Strict absolute source-available boundary differs')
        require(admission['kind'] in ('common_prefix', 'native_search'), 'Reserved methods cannot introduce unshared extra cheap candidates')
        previous_ready, previous_gain = ready, admission['gain']; counters['admissions'] += 1
        counters['late_admissions'] += not admission['on_time']
    prefix, certificate = prefix_decomposition(receipt, admissions, original)
    require(prefix == checked['common_prefix_selected'][0] and certificate == checked['certified_selected'][0], 'Actual last timely prefix/certificate differs')
    require(flag(receipt, 'source_improvement_certified_by_D') == any(a['on_time'] for a in admissions)
            and flag(receipt, 'source_certified_available_by_D') is True, 'Source available diagnostic differs')
    expected_cells = coordination_cells(graph, case['spec']['seed']+1000)
    coverage = [[action, cell] for action, cell in expected_cells]
    if trace['coverage']:
        require(trace['coverage'] == coverage, 'Original observable eleven-cell proposal menu differs')
    else:
        require(not trace['scope_order'], 'Scopes exist before original proposals')
    actions = [action for action, _ in expected_cells if action is not None]
    scopes = {}
    require(len(trace['scope_order']) == len(trace['full_scope_records']), 'Complete scope C/E record grid differs')
    scope_groups = {suffix: grouped(masks['scope_'+suffix], masks['scope_'+suffix+'_ptr']) for suffix in ('R', 'B', 'D', 'C', 'E')}
    require(all(len(groups) == len(trace['scope_order']) for groups in scope_groups.values()), 'All original scope arrays required')
    warms = {}
    for position, (record, full) in enumerate(zip(trace['scope_order'], trace['full_scope_records'])):
        index = record['action_index']
        require(isinstance(index, int) and not isinstance(index, bool) and index == position and index < len(actions), 'Scope preparation must be the actual action prefix')
        require(all(full[name] == value for name, value in record.items()), 'Full C/E scope record differs')
        scope = scope_replay(full, actions[index], index, graph); scopes[(index, 256)] = scope
        for suffix, field in (('R', 'replacements'), ('B', 'base'), ('D', 'displaced'), ('C', 'inserts'), ('E', 'releases')):
            values = scope_groups[suffix][position]
            expected = record[field] if field in record else full[field]
            actual = ids(values, len(graph['weights']), 'Saved scope '+suffix)
            require(tuple(actual) == tuple(expected) if suffix == 'R' else frozenset(actual) == frozenset(expected), 'Saved original scope differs: '+suffix)
        warm, candidates = executed_warm(scope, graph); upper = clique_upper(scope, graph)
        require(objective(warm, graph) <= (upper if native else upper+1e-9), 'Observed warm exceeds observable clique partition bound')
        warms[index] = dict(executed=warm, lower=objective(warm, graph), upper=upper, candidates=candidates)
        counters['executed_greedy_replays'] += 3
        counters['original_warm_floors'] += 1
    attempts = [] if source is None else source['attempts']
    warm_masks = grouped(masks['attempt_warm'], masks['attempt_warm_ptr'])
    returned_masks = grouped(masks['attempt_returned'], masks['attempt_returned_ptr'])
    require(len(attempts) == len(warm_masks) == len(returned_masks), 'Every actual native warm/return array required')
    raw_rows = trace['raw_attempts']
    counters['actual_native_calls'] = len(raw_rows)
    require(len(raw_rows) >= len(attempts), 'Actual native raw execution ledger incomplete')
    if source is None:
        require(checked['source_selected'][0] == original, 'Failed source fabricated committed membership')
        require(all(row['source_attempt_valid'] is False and row['actual_signed_gain'] is None for row in raw_rows), 'Failed source fabricated admitted attempt metadata')
        counters['uncommitted_raw_calls'] += len(raw_rows)
        return counters
    require(checked['source_selected'][0] == frozenset(ids(source['selected'], len(graph['weights']), 'Actual source selected')), 'Source NPZ/JSON differs')
    equal(source['incumbent_value'], initial, 'Source original incumbent differs', native)
    equal(source['on_time_gain'], checked['source_selected'][1]-initial, 'Source selected original gain differs')
    require(source['cheap_candidates_observed'] == source['cheap_candidates_admitted'] == 0, 'Reserved Lower must not reexecute a separate greedy candidate portfolio')
    require(source['proposals'] in (0, len(actions)), 'Actual proposal count differs')
    if source['proposals']:
        require(source['action_identities'] == [[a['inserts'], a['releases']] for a in actions], 'Controller original C/E action identities differ')
    require(source['declared_requests'] == source['proposals']*3 and source['duplicate_scope_requests_skipped'] == 0, 'Fixed one-scope/three workpoint request declaration differs')
    observations = source['known_warm_observations']; current_warm = {i: frozenset() for i in warms}
    expected_admissions = []; best, best_gain = original, 0
    require(len(observations) <= len(scopes) and len(observations) >= max(0, len(scopes)-1), 'Executed shared warm preparation coverage differs')
    for i, observation in enumerate(observations):
        scope = scopes[(i, 256)]; actual = warms[i]['executed']
        require(observation['action_index'] == i and observation['cap'] == 256
                and observation['action_identity'] == [actions[i]['inserts'], actions[i]['releases']]
                and frozenset(ids(observation['recovered'], len(graph['weights']), 'Actual shared W')) == actual, 'Actual shared warm is not the declared executed portfolio')
        gain = max(0, objective(scope['base'] | actual, graph)-initial)
        equal(observation['gain_if_valid'], gain, 'Actual L/common full objective differs')
        finite(observation['validated_sample_seconds'], 'Source relative common readiness', 0)
        if flag(observation, 'admitted_on_time'):
            current_warm[i] = actual
            if gain > best_gain:
                best, best_gain = scope['base'] | actual, gain
                expected_admissions.append(('common_prefix', best, best_gain))
    require(source['prepared_requests'] == 3*sum(flag(observation, 'admitted_on_time') for observation in observations), 'Actual prepared request prefix differs from paid warm completion')
    require(len(raw_rows) == len(attempts), 'Completed source must retain every attempted native call')
    seen_keys = set()
    for attempt, raw, warm_values, returned_values in zip(attempts, raw_rows, warm_masks, returned_masks):
        key = tuple(attempt['key']); require(len(key) == 3 and key not in seen_keys and key[:2] in scopes and key[2] in POINTS, 'Invalid/duplicate/spent native request')
        seen_keys.add(key); scope = scopes[key[:2]]; index = key[0]
        amount, cost = POINTS[key[2]]
        require(attempt['native_budget_kind'] == 'seconds' and attempt['native_budget_amount'] == amount and attempt['expected_seconds'] == cost, 'Actual shared native workpoint differs')
        require(finite(attempt['remaining_before_call'], 'Actual remaining native budget', 0) >= cost
                and finite(attempt['elapsed_seconds'], 'Actual spent native elapsed', 0) >= 0, 'Native request was not startable/spent')
        warm, value = membership(warm_values, graph, 'Actual native warm', scope['replacements'], scope['base'])
        require(warm == frozenset(ids(attempt['warm_start'], len(graph['weights']), 'Actual native warm JSON')) == current_warm[index], 'Native warm leaked future/unexecuted recovery')
        require(all(raw['scope'][name] == item for name, item in trace['scope_order'][index].items())
                and raw['scope']['releases'] == actions[index]['releases'] and raw['workpoint']['name'] == key[2]
                and frozenset(raw['warm_start']) == warm and raw['remaining_before'] == attempt['remaining_before_call'], 'Raw call/current scope/workpoint/warm differs')
        require(start <= finite(raw['started'], 'Raw actual call start') <= finite(raw['backend_returned'], 'Raw backend return') <= caller, 'Actual raw backend monotonic timing differs')
        valid = flag(attempt, 'valid'); admitted = flag(attempt, 'admitted_on_time')
        require(flag(raw, 'source_attempt_valid') == valid, 'Raw/committed native validity differs')
        if not valid:
            require(not admitted and attempt['recovered'] is None and len(returned_values) == 0
                    and raw['actual_signed_gain'] is None and raw['q_plus_recovery_gain'] is None, 'Invalid native output gained certified recovery')
            counters['invalid_native'] += 1; continue
        recovered, recovery_value = membership(returned_values, graph, 'Actual valid native return', scope['replacements'], scope['base'])
        require(recovered == frozenset(ids(attempt['recovered'], len(graph['weights']), 'Actual native return JSON')), 'Native NPZ/JSON return differs')
        raw_return = raw['raw_return']
        require(raw_return is not None and recovered == frozenset(ids(raw_return['recovered'], len(graph['weights']), 'Raw native return'))
                and attempt['diagnostics']['binary_sha256'] == raw_return['diagnostics']['binary_sha256'] == CHILS_SHA256, 'Actual native binary/raw membership differs')
        signed = objective(scope['base'] | recovered, graph)-initial
        q_r = scope['immediate_gain']+recovery_value
        equal(signed, q_r, 'Exact original signed q+joint-recovery decomposition', native)
        equal(attempt['gain_if_valid'], max(0, signed), 'Actual original full repaired gain differs')
        equal(raw['actual_signed_gain'], signed, 'Original signed raw gain differs', native)
        equal(raw['q_plus_recovery_gain'], q_r, 'Raw signed q+recovery differs', native)
        equal(raw['signed_decomposition_difference'], 0, 'Signed decomposition difference differs')
        require(recovery_value <= (warms[index]['upper'] if native else warms[index]['upper']+1e-9), 'Actual native recovery exceeds observable upper bound')
        counters['valid_native'] += 1
        counters['native_late_source_flag'] += not admitted
        counters['native_above_actual_warm'] += recovery_value > warms[index]['lower']
        counters['native_zero_or_negative_signed'] += signed <= 0
        if admitted:
            counters['source_admitted_native'] += 1
            if recovery_value > objective(current_warm[index], graph):
                current_warm[index] = recovered
            gain = max(0, signed)
            if gain > best_gain:
                best, best_gain = scope['base'] | recovered, gain
                expected_admissions.append(('native_search', best, best_gain))
    require(len(attempts) <= (0 if receipt['method'] == 'AnytimeGreedyPortfolio' else 8), 'Native query cap changed')
    require({tuple(key) for key in source['spent_requests']} == seen_keys and len(source['spent_requests']) == len(seen_keys), 'All failed/late native requests remain spent exactly once')
    require(len(expected_admissions) == len(admissions), 'Source actual improvement observer coverage differs')
    for expected, admission in zip(expected_admissions, admissions):
        kind, selected, gain = expected
        require(admission['kind'] == kind and frozenset(admission['selected']) == selected, 'Actual improvement stage/membership differs')
        equal(admission['gain'], gain, 'Actual source improvement gain differs')
    require(checked['source_selected'][0] == best, 'Source output is not actual executed incumbent trajectory')
    if receipt['method'] == 'AnytimeGreedyPortfolio':
        require(not attempts and receipt['certified_gain_above_prefix'] == receipt['returned_gain_above_prefix'] == 0, 'Shared greedy-only comparator obtained native extra credit')
    return counters


def replay(root, collection, fits, protocol_sha, completion_sha, fetch_receipt, fetch_sha, source_root=ROOT, archive=None,
           collection_source_root=None, fit_source_root=None, chils=None):
    root, collection, fits, source_root = map(lambda x: Path(x).resolve(), (root, collection, fits, source_root))
    require(sha(root/'protocol.json') == protocol_sha and sha(root/'completion.json') == completion_sha, 'Caller-pinned complete online bytes differ')
    protocol, complete = read(root/'protocol.json'), read(root/'completion.json')
    require(complete['status'] == 'complete_validation24_actual_online_not_confirmation' and complete['completed'] is True
            and complete['completed_units'] == complete['expected_units'] == 1512
            and complete['no_confirmation_graphs'] is True, 'Only the completed original1512 matrix may be analyzed')
    require(complete['protocol_sha256'] == protocol_sha, 'Online completion/protocol binding differs')
    transfer = check_fetch(root, fetch_receipt, fetch_sha, archive)
    for key, name in (('opened_sha256', 'opened.json'), ('progress_sha256', 'progress.jsonl'), ('classical_selection_sha256', 'classical_selection.json')):
        require(sha(root/name) == complete[key], 'Completed online dependency differs: '+name)
    require(protocol['status'] == 'frozen_validation24_actual_online_before_execution'
            and protocol['expected_units'] == 1512 and protocol['variants'] == list(VARIANTS)
            and protocol['classics'] == list(CLASSICS) and protocol['fit_seeds'] == list(SEEDS)
            and protocol['deadlines'] == list(DEADLINES) and protocol['caps'] == [256]
            and protocol['fixed_scope_cap'] == 256 and protocol['executed_common_greedy_prefix'] is True
            and protocol['greedy_only_queries'] == 0 and protocol['lower_uses_already_paid_warm'] is True
            and protocol['max_queries'] == 8 and protocol['no_confirmation_graphs'] is True
            and protocol['no_resume_retry_or_replacement'] is True and protocol['no_teacher_input'] is True,
            'Frozen fixedscope/residual/sharedprefix/complete policy registry differs')
    require(protocol['chils_sha256'] == CHILS_SHA256 and protocol['chils_population'] == protocol['chils_threads'] == 1
            and protocol['chils_seed'] == 17 and protocol['calibration_sha256'] == CALIBRATION_SHA256, 'Actual same native kernel/cost metadata differs')
    sources = check_source(protocol['source_capsule'], source_root, protocol['source_capsule']['manifest_sha256'])
    required = {'experiments/v4_residual_online_validation.py', 'experiments/v4_residual_online_validation_execute.py',
                'experiments/v4_residual_fit.py', 'experiments/v4_residual_training_data.py', 'experiments/v4_residual_common.py',
                'experiments/v4_neighborhoods.py', 'src/joint_recovery/v4_residual_model.py',
                'src/joint_recovery/v4_residual_controller_coordination.py', 'src/joint_recovery/v4_budgeted_recovery.py',
                'tests/test_v4_residual_online_validation.py', 'tests/test_v4_residual_online_validation_execute.py'}
    require(required <= set(sources), 'Required owned online source/test closure missing')
    opened = read(root/'opened.json'); freeze = read(root/'freeze_completion.json')
    require(opened['protocol_sha256'] == protocol_sha and opened['expected_units'] == 1512
            and freeze['protocol_sha256'] == protocol_sha and freeze['expected_units'] == 1512
            and opened['source_manifest_sha256'] == freeze['source_manifest_sha256'] == protocol['source_capsule']['manifest_sha256'], 'Exactly one bound freeze/opened closure differs')
    guards, barrier = read(root/'one_shot_guards.json'), read(root/'one_shot_pre_run.json')
    require(guards['expected_tests'] == guards['tests_run'] == 16 and guards['successful'] is True
            and all(guards[key] == 0 for key in ('failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes'))
            and guards['frozen_protocol_sha256'] == protocol_sha and sha(root/'one_shot_guards.txt') == guards['log_sha256']
            and barrier['frozen_protocol_sha256'] == protocol_sha and barrier['guards_sha256'] == sha(root/'one_shot_guards.json')
            and barrier['planned_units'] == 1512 and barrier['no_protocol_mutation'] is True, 'Original one-shot pre-run guard barrier differs')
    declared_origins = protocol['source_capsule']['execution_origins']+barrier['source_receipt']['launcher_and_guard_origins']
    for origin in declared_origins:
        require(sources.get(origin['path']) == origin['sha256'], 'Original online/launcher/guard execution origin differs')
    require(required <= {record['path'] for record in declared_origins}, 'Required actual scientific and launch origins missing')
    _, originals, case_files = check_collection(collection, protocol['collector_protocol_sha256'], protocol['collector_completion_sha256'], collection_source_root or source_root)
    fit_count = check_fits(fits, protocol, case_files, fit_source_root or source_root)
    if chils is not None:
        require(sha(chils) == CHILS_SHA256, 'Independent local declared native binary bytes differ')
    cases = protocol['case_registry']; by_case = {}
    require(len(cases) == 24, 'All original validation graph slots required')
    for index, case in enumerate(cases, 72):
        cid = 'case_%03d' % index; entry, graph = originals[cid]
        require(case['case_id'] == cid and case['spec'] == entry['spec'] and case['original_status'] == entry['status']
                and case['file_bindings'] == entry['files'], 'Online original validation input registry differs')
        equal(case['graph_weight'], .5/(18 if case['spec']['domain'] == 'menu' else 6), 'Original fixed equal-domain mass differs')
        by_case[cid] = (case, graph)
    require(Counter(c['spec']['domain'] for c in cases) == Counter(menu=18, resource=6), 'All18/6 original domain graph slots required')
    methods = [(v, seed) for v in VARIANTS for seed in SEEDS]+[(c, None) for c in CLASSICS]
    plan = [dict(unit_index=i, case_id=cid, deadline_seconds=D, method=method, fit_seed=seed)
            for i, (cid, D, method, seed) in enumerate((cid, D, method, seed) for cid in by_case for D in DEADLINES for method, seed in methods)]
    require(protocol['units'] == plan, 'Exact original1512 unit plan/order differs')
    rows, artifacts = complete['rows'], complete['artifacts']
    require(len(rows) == len(artifacts) == 1512, 'Every declared row/artifact slot required')
    artifact_map = {row['unit_index']: row for row in artifacts}
    require(len(artifact_map) == 1512 and set(artifact_map) == set(range(1512)), 'Complete unique artifact index registry required')
    progress = [json.loads(line) for line in (root/'progress.jsonl').read_text(encoding='utf8').splitlines()]
    require(len(progress) == 1512, 'All unit progress records required')
    checks = Counter(); seen = set()
    expected_masks = {'original_selected', 'certified_selected', 'final_return_selected', 'common_prefix_selected', 'source_selected',
                      'admission_selected', 'admission_ptr', 'attempt_warm', 'attempt_warm_ptr', 'attempt_returned', 'attempt_returned_ptr'}
    expected_masks.update('scope_'+suffix+tail for suffix in ('R', 'B', 'D', 'C', 'E') for tail in ('', '_ptr'))
    for slot, row, event in zip(plan, rows, progress):
        require(all(row[key] == value for key, value in slot.items()), 'Actual unit differs from its original frozen slot')
        index = slot['unit_index']; key = (row['case_id'], row['method'], row['fit_seed'], row['deadline_seconds'])
        require(key not in seen, 'Duplicate actual original unit'); seen.add(key)
        files = artifact_map[index]['files']; require(set(files) == {'receipt.json', 'membership.npz'}, 'Exactly the full actual unit artifact pair required')
        folder = root/'units'/('unit_%04d' % index)
        paths = {name: bound_file(folder, name, value) for name, value in files.items()}
        receipt = read(paths['receipt.json'])
        require(all(receipt[name] == value for name, value in row.items()) and receipt['membership_npz_sha256'] == files['membership.npz']['sha256'], 'Complete scalar/full actual receipt or membership binding differs')
        require(event['completed_units'] == index+1 and event['expected_units'] == 1512 and event['artifact_sha256'] == files
                and event['unit_status'] == row['status'] and all(event[k] == row[k] for k in ('case_id', 'method', 'fit_seed', 'deadline_seconds')), 'Actual progress/grid artifact binding differs')
        with np.load(paths['membership.npz'], allow_pickle=False) as archive_npz:
            require(set(archive_npz.files) == expected_masks, 'Actual complete membership NPZ schema differs')
            masks = {name: archive_npz[name].copy() for name in archive_npz.files}
        case, graph = by_case[row['case_id']]
        unit_checks = replay_unit(receipt, masks, case, graph)
        checks.update(unit_checks)
        rows[index] = dict(row, independent_counts=dict(unit_checks))
    require(complete['unit_failures'] == sum(r['status'] != 'complete_unit' for r in rows), 'Complete original failure count differs')
    checks.update(original_graphs=24, retained_fit_checkpoints=fit_count, source_files=len(sources), fetched_files=len(transfer['fetched']))
    checks['native_declared_binary_local_bytes_checked'] = int(chils is not None)
    return protocol, rows, dict(checks)


def bootstrap_graphs(graph_rows, comparator, samples=10000, seed=1979):
    """Paired stratified ORIGINAL graph bootstrap, after all fit seeds average."""
    selected = [row for row in graph_rows if row['deadline_seconds'] == .556]
    require(len(selected) == 24*11, 'Exactly264 original graph-policy means required')
    by = defaultdict(dict)
    for row in selected:
        require(row['method'] not in by[row['case_id']], 'Duplicate graph-policy bootstrap cell')
        by[row['case_id']][row['method']] = row
    require(len(by) == 24 and all(set(cell) == set(VARIANTS+CLASSICS) for cell in by.values()), 'Bootstrap requires all24 graph means and11 policies')
    require(all(len({row['domain'] for row in cell.values()}) == 1 for cell in by.values()), 'Original graph domain changed across policies')
    require(samples >= 100 and isinstance(samples, int), 'Explicit finite graph bootstrap count required')
    rng = np.random.default_rng(seed); draws_fixed = np.zeros(samples); draws_best = np.zeros(samples)
    observed = {}; domain_counts = {'menu': 18, 'resource': 6}
    for domain, count in domain_counts.items():
        cells = [by[cid] for cid in sorted(by) if by[cid][VARIANTS[0]]['domain'] == domain]
        require(len(cells) == count, 'Original bootstrap18/6 graph domain coverage required')
        capacity = np.asarray([cell[VARIANTS[0]]['strict_return_relative_gain'] for cell in cells])
        classic = np.asarray([[cell[c]['strict_return_relative_gain'] for c in CLASSICS] for cell in cells])
        draw = rng.integers(0, count, size=(samples, count))
        domain_capacity = capacity[draw].mean(axis=1); domain_classic = classic[draw].mean(axis=1)
        draws_fixed += .5*(domain_capacity-domain_classic[:, CLASSICS.index(comparator)])
        observed[domain] = dict(capacity=domain_capacity, classics=domain_classic)
    pooled_capacity = .5*(observed['menu']['capacity']+observed['resource']['capacity'])
    pooled_classic = .5*(observed['menu']['classics']+observed['resource']['classics'])
    draws_best = pooled_capacity-pooled_classic.max(axis=1)
    return dict(unit='original graph, three fitting seeds averaged first; domains independently resampled with fixed half mass',
                bootstrap_samples=samples, bootstrap_seed=seed, paired_selected_comparator=comparator,
                paired_selected_comparator_95_interval=np.quantile(draws_fixed, [.025, .975]).tolist(),
                reselected_complete_six_classics_95_interval=np.quantile(draws_best, [.025, .975]).tolist(),
                validation_selection_is_development_evidence=True, independent_confirmatory_interval=False,
                native_requests_and_fit_seeds_not_independent_samples=True)


def aggregate(protocol, rows, bootstrap_samples=10000):
    cells = defaultdict(list)
    for row in rows:
        cells[(row['case_id'], row['method'], row['deadline_seconds'])].append(row)
    cases = protocol['case_registry']; graph_rows = []
    metrics = ('strict_return_relative_gain', 'source_available_relative_gain', 'common_prefix_relative_gain',
               'source_extra_above_prefix_relative_gain', 'returned_prefix_component_relative_gain', 'returned_extra_above_prefix_relative_gain',
               'recorded_deadline_miss_rate', 'failed_rate', 'unavailable_rate', 'mean_actual_native_calls',
               'mean_source_admitted_native_calls', 'mean_valid_native_returns', 'mean_native_returns_above_actual_warm')
    for case in cases:
        cid = case['case_id']
        for D in DEADLINES:
            for method in VARIANTS+CLASSICS:
                units = cells[(cid, method, D)]
                require(len(units) == (3 if method in VARIANTS else 1)
                        and {r['fit_seed'] for r in units} == (set(SEEDS) if method in VARIANTS else {None}), 'All fitting seeds/original graph cells required')
                values = defaultdict(list); latency = []
                for r in units:
                    denominator = max(1, r['initial_value']) if r['initial_value'] is not None else 1
                    values['strict_return_relative_gain'].append(r['final_return_relative_gain'])
                    values['source_available_relative_gain'].append(r['certified_relative_gain'])
                    for metric, field in (('common_prefix_relative_gain', 'common_prefix_certified_gain'),
                            ('source_extra_above_prefix_relative_gain', 'certified_gain_above_prefix'),
                            ('returned_prefix_component_relative_gain', 'returned_prefix_component'),
                            ('returned_extra_above_prefix_relative_gain', 'returned_gain_above_prefix')):
                        values[metric].append(r.get(field, 0)/denominator)
                    values['recorded_deadline_miss_rate'].append(bool(r.get('outer_deadline_miss', False)))
                    values['failed_rate'].append(r['status'] != 'complete_unit')
                    values['unavailable_rate'].append(r['status'] == 'declared_input_unavailable')
                    for metric, counter in (('mean_actual_native_calls', 'actual_native_calls'),
                            ('mean_source_admitted_native_calls', 'source_admitted_native'),
                            ('mean_valid_native_returns', 'valid_native'),
                            ('mean_native_returns_above_actual_warm', 'native_above_actual_warm')):
                        values[metric].append(r.get('independent_counts', {}).get(counter, 0))
                    if r.get('actual_interface_receipt_seconds') is not None:
                        latency.append(r['actual_interface_receipt_seconds'])
                graph_rows.append(dict(case_id=cid, domain=case['spec']['domain'], graph_seed=case['spec']['seed'],
                    family=case['spec'].get('topology', 'resource'), graph_weight=case['graph_weight'],
                    method=method, deadline_seconds=D, retained_units=len(units), failed_units=sum(values['failed_rate']),
                    **{name: fsum(values[name])/len(units) for name in metrics},
                    actual_caller_latency_mean_seconds=fsum(latency)/len(latency) if latency else None,
                    actual_caller_latency_complete=len(latency) == len(units)))
    tables = []
    for D in DEADLINES:
        for method in VARIANTS+CLASSICS:
            selected = [r for r in graph_rows if r['method'] == method and r['deadline_seconds'] == D]
            require(len(selected) == 24, 'Original graph denominator cannot change')
            row = dict(method=method, deadline_seconds=D, original_graphs=24,
                       retained_units=sum(r['retained_units'] for r in selected), failed_units=sum(r['failed_units'] for r in selected))
            row.update({name: fsum(r['graph_weight']*r[name] for r in selected) for name in metrics})
            observed_mass = fsum(r['graph_weight'] for r in selected if r['actual_caller_latency_mean_seconds'] is not None)
            row['observed_latency_graph_mass'] = observed_mass
            row['mean_actual_caller_latency_seconds'] = fsum(r['graph_weight']*r['actual_caller_latency_mean_seconds'] for r in selected) if all(r['actual_caller_latency_complete'] for r in selected) else None
            row['observed_actual_caller_latency_mean_seconds'] = fsum(r['graph_weight']*r['actual_caller_latency_mean_seconds'] for r in selected if r['actual_caller_latency_mean_seconds'] is not None)/observed_mass if observed_mass else None
            for domain, count in (('menu', 18), ('resource', 6)):
                row[domain+'_strict_return_relative_gain'] = fsum(r['strict_return_relative_gain'] for r in selected if r['domain'] == domain)/count
                row[domain+'_returned_extra_above_prefix_relative_gain'] = fsum(r['returned_extra_above_prefix_relative_gain'] for r in selected if r['domain'] == domain)/count
            tables.append(row)
    scores = {r['method']: r['strict_return_relative_gain'] for r in tables if r['deadline_seconds'] == .556 and r['method'] in CLASSICS}
    comparator = max(CLASSICS, key=lambda name: (scores[name], -CLASSICS.index(name)))
    capacity = next(r for r in tables if r['method'] == VARIANTS[0] and r['deadline_seconds'] == .556)
    difference = capacity['strict_return_relative_gain']-scores[comparator]
    bootstrap = bootstrap_graphs(graph_rows, comparator, bootstrap_samples)
    return dict(rows=tables, graph_rows=graph_rows, complete_classical_scores=scores, frozen_classical_recomputed=comparator,
        primary_mean_difference=difference, mean_materiality_threshold=MATERIALITY,
        development_mean_materiality_pass=difference >= MATERIALITY,
        development_paired_lower95_positive=bootstrap['paired_selected_comparator_95_interval'][0] > 0,
        bootstrap=bootstrap, confirmation_pending=True, confirmation_or_acceptance_established=False,
        common_prefix_is_not_learned_native_extra=True, graph_weights_redistributed=False)


def csv_file(path, records):
    with Path(path).open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'collection', 'fit', 'protocol-sha256', 'completion-sha256', 'fetch-receipt', 'fetch-sha256', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--source-root', required=True); parser.add_argument('--archive')
    parser.add_argument('--collection-source-root'); parser.add_argument('--fit-source-root'); parser.add_argument('--chils')
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    args = parser.parse_args(); out = Path(args.out).resolve()
    require(not out.exists(), 'Independent analysis output must be entirely fresh')
    for input_root in (args.run, args.collection, args.fit):
        sealed = Path(input_root).resolve()
        require(out != sealed and sealed not in out.parents, 'Audit output cannot modify sealed run/collection/fits')
    protocol, rows, checks = replay(args.run, args.collection, args.fit, args.protocol_sha256, args.completion_sha256,
                                  args.fetch_receipt, args.fetch_sha256, args.source_root, args.archive,
                                  args.collection_source_root, args.fit_source_root, args.chils)
    statistics = aggregate(protocol, rows, args.bootstrap_samples)
    selection = read(Path(args.run)/'classical_selection.json')
    require(selection['selected'] == statistics['frozen_classical_recomputed'], 'Complete six-classic frozen comparison differs')
    for name, score in statistics['complete_classical_scores'].items():
        equal(selection['strict_return_scores'][name], score, 'Frozen complete classical score differs')
    record = dict(status='PASS_completed1512_independent_residual_membership_prefix_caller_replay', checks=checks,
        run_completion_sha256=args.completion_sha256, analysis_source_sha256=sha(__file__), statistics=statistics,
        timing_limitations='Actual outer caller, final validation and improving source observer use recorded absolute timestamps. Nonimproving source known-warm/native admissions have only source relative readiness/flags; they are NOT independently certified against an absolute source internal deadline. No physical hard-deadline/host isolation claim.',
        checkpoint_limitations='Own selected/epoch40 metadata and all declared bytes are bound without Torch; tensor storages are opaque and are not independently checked for numerical finiteness.',
        native_binary_provenance='Local declared binary SHA checked' if args.chils else 'Declared native binary SHA checked in protocol/valid attempt receipts; native file bytes not supplied to this audit',
        evidence_boundary='Complete original validation24 development evidence only; no confirmation or conference acceptance claim.')
    out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf8')
    csv_file(out/'residual_online_table.csv', statistics['rows']); csv_file(out/'residual_online_graph_means.csv', statistics['graph_rows'])
    print(json.dumps(dict(status=record['status'], checks=checks, primary_mean_difference=statistics['primary_mean_difference'],
                         comparator=statistics['frozen_classical_recomputed']), allow_nan=False))


if __name__ == '__main__':
    main()
