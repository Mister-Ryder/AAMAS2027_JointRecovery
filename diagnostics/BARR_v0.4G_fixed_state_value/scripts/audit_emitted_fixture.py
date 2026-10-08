"""Consume actual C++ PRIVATE fixed-clock emitter artifacts, never performance.

The synthetic fixture has no physical resources, so unique artificial labels
exercise the same timeline audit without implying real-data resource evidence.
Production configuration/native-duration rejection is checked separately.
"""
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import numpy as np

from g_protocol import NATIVE_CONFIG, dump_new, now, read, sha
from g_state_audit import replay_audit, snapshot_audit, capture_audit
from run_g_remote import validate_native_cost


def fixture_graph(path):
    with Path(path).open(encoding='utf-8') as stream:
        magic, n, m = stream.readline().split(); n, m = int(n), int(m)
        if magic != 'BARR1': raise ValueError('Unknown emitted fixture graph')
        weights, raw = [], []
        for _ in range(n):
            tick, value, owner = stream.readline().split(); weights.append(int(tick)); raw.append(float(value))
        edges = [tuple(map(int, stream.readline().split())) for _ in range(m)]
        initial = list(map(int, stream.readline().split()))
        if len(initial) != initial[0] + 1 or stream.read().strip(): raise ValueError('Truncated or extra synthetic graph input')
    data = {'weight_ticks': np.array(weights, dtype=np.int64), 'edge_u': np.array([u for u,v in edges], dtype=np.uint32),
            'edge_v': np.array([v for u,v in edges], dtype=np.uint32), 'start_ticks': np.zeros(n, dtype=np.int64),
            'end_ticks': np.ones(n, dtype=np.int64), 'satellite_id': np.array(['s%d'%i for i in range(n)]),
            'antenna_id': np.array(['a%d'%i for i in range(n)]), 'satellite_gap_ticks': np.array(0, dtype=np.int64),
            'ground_gap_by_node_ticks': np.zeros(n, dtype=np.int64), '_native_raw_weights': raw}
    return data, initial[1:]


def audit(directory):
    directory = Path(directory).resolve()
    data, initial = fixture_graph(directory / 'fixture.barr')
    meta = read(directory / 'fixture.meta.json')
    manifest = read(directory / 'fixture_manifest.json')
    if manifest.get('schema') != 'barr_fixed_state_fixture_v1' or manifest.get('synthetic_test_clock') is not True or manifest.get('performance_evidence') is not False:
        raise ValueError('Actual C++ emitter must explicitly identify synthetic non-performance evidence')
    expected_config = NATIVE_CONFIG
    if meta['config'] != expected_config:
        raise ValueError('Actual C++ synthetic configuration differs from its declared fixed-clock fixture')
    setup = {'graphs': [{'id': 'fixture', 'n': len(data['weight_ticks']), 'm': len(data['edge_u']),
                         'native_sha256': sha(directory / 'fixture.barr'),
                         'initial_ticks': sum(int(data['weight_ticks'][v]) for v in initial)}]}
    entry = {'snapshot_path': 'fixture.bin', 'metadata_path': 'fixture.meta.json', 'snapshot_sha256': sha(directory / 'fixture.bin'),
             'threshold_seconds': meta['controller']['threshold_seconds'], 'capture_elapsed_seconds': meta['controller']['capture_elapsed_seconds'],
             'epoch': meta['controller']['epoch'], 'roundtrip_equal': True}
    with patch('g_state_audit.graph_data', return_value=data):
        snapshot = snapshot_audit(directory, 'fixture', entry, setup, fixture_expected_config=expected_config)
        captures = capture_audit(directory, {'native_output': 'fixture.capture.json', 'graph': 'fixture'}, setup)
        arms = {}
        production_rejections = {}
        for arm in ['recover', 'continue']:
            row = dict(graph='fixture', arm=arm, future_seed=0, native_output=arm+'.json',
                       snapshot_path='fixture.bin', metadata_path='fixture.meta.json', snapshot_sha256=entry['snapshot_sha256'])
            arms[arm] = replay_audit(directory, row, setup, fixture_expected_config=expected_config)
            try:
                validate_native_cost(read(directory / (arm + '.json')), process_wall=1., process_cpu=1.)
            except ValueError as exc:
                production_rejections[arm] = str(exc)
            else:
                raise ValueError('Synthetic PRIVATE-clock branch was incorrectly accepted as production evidence')
            short = dict(read(directory / (arm + '.json')), native_seconds=359.)
            try:
                validate_native_cost(short, process_wall=360., process_cpu=360.)
            except ValueError:
                production_rejections[arm + '_short_budget'] = 'short native budget rejected'
            else:
                raise ValueError('Short branch was incorrectly accepted as production evidence')
        if arms['recover']['ready_state_sha256'] != arms['continue']['ready_state_sha256']:
            raise ValueError('Actual C++ paired readiness state is not identical')
    return {'schema': 'barr_v04G_actual_cpp_python_fixture_audit_v1', 'pass_all': True, 'created_utc': now(),
            'performance_results': False, 'native_quality_calls': 0, 'private_testclock_only': True,
            'snapshot': snapshot, 'capture': captures, 'arms': arms, 'production_rejections': production_rejections,
            'files_sha256': {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()},
            'scope': 'actual C++ emitted schema/graph/strict pair/private inventory/RNG SHA/common feedback/archive/diversity concordance; synthetic unique timeline resources, no six-minute quality claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    proof = audit(args.fixture)
    dump_new(args.output, proof)
    print(json.dumps({key: proof[key] for key in ['pass_all', 'performance_results', 'private_testclock_only', 'native_quality_calls']}), flush=True)
