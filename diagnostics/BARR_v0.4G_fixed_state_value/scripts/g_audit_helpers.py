"""Independent original integer/complete-edge/resource audit for G frozen states."""
from collections import defaultdict
import numpy as np
from g_protocol import integer as _int, number, read, sha, inside

def _integer_array(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(label + ' requires integer dtype and exact shape')
    return array

def membership_audit(data, selected):
    """Exact objective, complete original graph and maximum-frontier timelines."""
    if not isinstance(selected, list) or any(type(vertex) is not int for vertex in selected):
        raise ValueError('Membership must be a JSON list of integers without bool/float coercion')
    weights = np.asarray(data['weight_ticks'])
    n = len(weights)
    _integer_array(weights, (n,), 'weight_ticks')
    if len(set(selected)) != len(selected) or any(vertex < 0 or vertex >= n for vertex in selected):
        raise ValueError('Membership has duplicate or out-of-range IDs')
    u = np.asarray(data['edge_u'])
    v = np.asarray(data['edge_v'])
    _integer_array(u, (len(u),), 'edge_u')
    _integer_array(v, u.shape, 'edge_v')
    if len(u) and (min(int(u.min()), int(v.min())) < 0 or max(int(u.max()), int(v.max())) >= n):
        raise ValueError('Original graph has an invalid edge endpoint')
    starts = _integer_array(data['start_ticks'], (n,), 'start_ticks')
    ends = _integer_array(data['end_ticks'], (n,), 'end_ticks')
    ground_gaps = _integer_array(data['ground_gap_by_node_ticks'], (n,), 'ground_gap_by_node_ticks')
    satellite_gap = np.asarray(data['satellite_gap_ticks'])
    if satellite_gap.shape != () or not np.issubdtype(satellite_gap.dtype, np.integer):
        raise ValueError('satellite_gap_ticks must be an integer scalar')
    satellite_gap = int(satellite_gap)
    if satellite_gap < 0 or np.any(ground_gaps < 0) or np.any(ends < starts):
        raise ValueError('Physical interval ends/gaps are invalid')
    for key in ('satellite_id', 'antenna_id'):
        if np.asarray(data[key]).shape != (n,):
            raise ValueError(key + ' must contain one resource identity per vertex')
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(selected, dtype=np.int64)] = True
    graph_violations = int(np.count_nonzero(mask[u] & mask[v]))
    timelines = {}
    for resource, gaps in [('satellite_id', satellite_gap), ('antenna_id', ground_gaps)]:
        groups = defaultdict(list)
        for vertex in selected:
            groups[str(data[resource][vertex])].append(vertex)
        violations, examples = 0, []
        for label, members in groups.items():
            members.sort(key=lambda vertex: (int(starts[vertex]), int(ends[vertex]), vertex))
            frontier, frontier_vertex = None, None
            for vertex in members:
                start = int(starts[vertex])
                if frontier is not None and start < frontier:
                    violations += 1
                    if len(examples) < 5:
                        examples.append({'resource': label, 'earlier': frontier_vertex, 'later': vertex,
                                         'later_start_ticks': start, 'earlier_available_ticks': frontier})
                gap = gaps if resource == 'satellite_id' else int(gaps[vertex])
                available = int(ends[vertex]) + gap
                if frontier is None or available > frontier:
                    frontier, frontier_vertex = available, vertex
        timelines[resource] = {'violations': violations, 'examples': examples,
                               'count_semantics': 'later intervals conflicting with an earlier maximum availability frontier'}
    objective = sum(int(weights[vertex]) for vertex in selected)
    return {'selected_count': len(selected), 'objective_ticks': objective,
            'edge_violations': graph_violations, 'satellite_timeline': timelines['satellite_id'],
            'antenna_timeline': timelines['antenna_id'],
            'feasible': graph_violations == 0 and all(item['violations'] == 0 for item in timelines.values())}


_CACHE = {}


def graph_data(root, graph, setup):
    from pathlib import Path
    root = Path(root).resolve()
    definition = next(g for g in setup['graphs'] if g['id'] == graph)
    key = (str(root), graph, definition['original_sha256'], definition['normalized_sha256'])
    if key not in _CACHE:
        original = inside(root, 'inputs/original/' + graph + '.npz')
        normalized = inside(root, 'inputs/normalized/' + graph + '.npz')
        if sha(original) != definition['original_sha256'] or sha(normalized) != definition['normalized_sha256']:
            raise ValueError('Frozen original/normalized graph hash changed')
        with np.load(original, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        with np.load(normalized, allow_pickle=False) as archive:
            initial = archive['initial_mask']
            if not np.array_equal(data['weight_ticks'], archive['weight_ticks']):
                raise ValueError('Original and normalized integer weights differ')
        if initial.shape != (len(data['weight_ticks']),) or not np.all((initial == 0) | (initial == 1)):
            raise ValueError('Frozen common initial mask is invalid')
        audit = membership_audit(data, np.flatnonzero(initial).tolist())
        if not audit['feasible'] or audit['objective_ticks'] != definition['initial_ticks']:
            raise ValueError('Frozen initial solution failed exact original feasibility/objective')
        if len(data['weight_ticks']) != definition['n'] or len(data['edge_u']) != definition['m']:
            raise ValueError('Frozen original graph dimensions changed')
        native_path = inside(root, 'inputs/native/' + graph + '.barr')
        if sha(native_path) != definition['native_sha256']:
            raise ValueError('Frozen native graph identity changed')
        raw_weights = []
        with native_path.open(encoding='utf-8') as stream:
            if stream.readline().split() != ['BARR1', str(definition['n']), str(definition['m'])]:
                raise ValueError('Native BARR1 graph dimensions differ')
            for index, weight in enumerate(data['weight_ticks']):
                fields = stream.readline().split()
                if len(fields) != 3 or int(fields[0]) != int(weight):
                    raise ValueError('Native vertex integer weight differs from original NPZ')
                value = float(fields[1])
                if not np.isfinite(value) or abs(value - int(weight) / 1e6) > 1e-9:
                    raise ValueError('Native raw weight differs from original microsecond ticks')
                raw_weights.append(value)
        data['_native_raw_weights'] = raw_weights
        _CACHE[key] = data
    return _CACHE[key]


def population_audit(data, population):
    if not isinstance(population, list) or len(population) != 4:
        raise ValueError('A complete four-member population mirror is required')
    audits = [membership_audit(data, mask) for mask in population]
    if not all(item['feasible'] for item in audits):
        raise ValueError('A frozen population member violates original graph/resources')
    n = len(data['weight_ticks'])
    weights = data['weight_ticks']
    total_weight = sum(int(v) for v in weights)
    masks = [set(mask) for mask in population]
    differences = [masks[i] ^ masks[j] for i in range(4) for j in range(i + 1, 4)]
    hamming = [len(diff) / float(n) for diff in differences]
    weighted = [sum(int(weights[v]) for v in diff) / float(total_weight) if total_weight else 0 for diff in differences]
    return {'members': audits, 'distinct_masks': len({tuple(sorted(mask)) for mask in masks}),
            'normalized_pairwise_hamming': sum(hamming) / 6,
            'normalized_weighted_pairwise_hamming': sum(weighted) / 6,
            'pairwise_hamming': hamming, 'pairwise_weighted_hamming': weighted}


def strict_pair_audit(data, selected, pair, blockers=None):
    """Independently recover exact blocker unions and both negative unary gains."""
    if not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair) or pair[0] == pair[1]:
        raise ValueError('Known action requires exactly two distinct integer outsiders')
    base = membership_audit(data, selected)
    if not base['feasible']:
        raise ValueError('Frozen intervention target is infeasible')
    n = len(data['weight_ticks'])
    if any(v < 0 or v >= n or v in selected for v in pair):
        raise ValueError('Known pair is not outside the frozen intervention target')
    chosen = set(selected)
    u, v = data['edge_u'], data['edge_v']
    adjacent = []
    for outsider in pair:
        neighbors = set(int(x) for x in v[u == outsider]) | set(int(x) for x in u[v == outsider])
        if pair[1 - len(adjacent)] in neighbors:
            raise ValueError('Known outsider pair has an internal original graph edge')
        adjacent.append(neighbors & chosen)
    union = adjacent[0] | adjacent[1]
    if blockers is not None:
        if isinstance(blockers, list) and len(blockers) == 2 and all(isinstance(x, list) for x in blockers):
            for index in range(2):
                if any(type(x) is not int for x in blockers[index]) or len(set(blockers[index])) != len(blockers[index]) or set(blockers[index]) != adjacent[index]:
                    raise ValueError('Saved per-outsider blockers differ from original graph adjacency')
        elif not isinstance(blockers, list) or any(type(v) is not int for v in blockers) or len(set(blockers)) != len(blockers) or set(blockers) != union:
            raise ValueError('Saved blocker set differs from original graph blocker union')
    weights = data['weight_ticks']
    unary = [int(weights[pair[i]]) - sum(int(weights[x]) for x in adjacent[i]) for i in range(2)]
    gain = sum(int(weights[x]) for x in pair) - sum(int(weights[x]) for x in union)
    if not all(x < 0 for x in unary) or gain <= 0:
        raise ValueError('Captured action is not a strict double-negative positive joint witness')
    after = sorted(chosen - union | set(pair))
    after_audit = membership_audit(data, after)
    if not after_audit['feasible'] or after_audit['objective_ticks'] - base['objective_ticks'] != gain:
        raise ValueError('Applying known pair fails independent objective/resource audit')
    return {'before_ticks': base['objective_ticks'], 'unary_gain_ticks': unary,
            'pair_gain_ticks': gain, 'blockers': sorted(union), 'after_selected': after,
            'after_ticks': after_audit['objective_ticks'], 'feasible': True}


def checked_solution(data, record):
    if not isinstance(record, dict):
        raise ValueError('Objective/membership mirror must be an object')
    audit = membership_audit(data, record['selected'])
    if not audit['feasible'] or audit['objective_ticks'] != _int(record['ticks'], 'solution ticks'):
        raise ValueError('Raw original integer objective/full-edge/resource mismatch')
    if 'raw' in record:
        import math
        expected = math.fsum(data['_native_raw_weights'][vertex] for vertex in record['selected'])
        value = number(record['raw'], 'native raw objective')
        if abs(value - expected) > 1e-8:
            raise ValueError('Native raw archive objective differs from original native weight sum')
        audit.update(raw_objective_target_seconds=expected, raw_comparison_absolute_tolerance=1e-8)
    return audit


def check_diversity(saved, calculated):
    import math
    for name in ['distinct_masks', 'normalized_pairwise_hamming', 'normalized_weighted_pairwise_hamming']:
        if name not in saved or isinstance(saved[name], bool) or not math.isclose(saved[name], calculated[name], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('Native diversity differs from independently calculated population: ' + name)
    return True


