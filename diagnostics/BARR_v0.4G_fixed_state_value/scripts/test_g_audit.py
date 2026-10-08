"""Offline original-graph, full-state mirror and feedback fixtures."""
from copy import deepcopy
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from g_protocol import NATIVE_CONFIG, read, sha, dump_new
from g_audit_helpers import membership_audit, strict_pair_audit, population_audit, check_diversity
from g_state_audit import capture_audit, replay_audit, snapshot_audit


def graph():
    return {'weight_ticks': np.array([3, 3, 5, 1, 1, 1], dtype=np.int64),
            'edge_u': np.array([0, 1], dtype=np.uint32), 'edge_v': np.array([2, 2], dtype=np.uint32),
            'start_ticks': np.zeros(6, dtype=np.int64), 'end_ticks': np.ones(6, dtype=np.int64),
            'satellite_id': np.array(['a', 'b', 'c', 'd', 'e', 'f']),
            'antenna_id': np.array(['u', 'v', 'w', 'x', 'y', 'z']),
            'ground_gap_by_node_ticks': np.zeros(6, dtype=np.int64),
            'satellite_gap_ticks': np.array(0, dtype=np.int64),
            '_native_raw_weights': [3e-6, 3e-6, 5e-6, 1e-6, 1e-6, 1e-6]}


def sol(selected, ticks, raw=False):
    record = {'selected': selected, 'ticks': ticks}
    if raw: record['raw'] = ticks / 1e6
    return record


class AuditTests(unittest.TestCase):
    def test_17_bool_vertex_rejected(self):
        with self.assertRaises(ValueError): membership_audit(graph(), [True])

    def test_18_exact_large_integer_sum(self):
        data = graph(); data['weight_ticks'][2] = (1 << 54) + 3
        self.assertEqual(membership_audit(data, [2])['objective_ticks'], (1 << 54) + 3)

    def test_19_complete_edge_rejection(self):
        self.assertFalse(membership_audit(graph(), [0, 2])['feasible'])

    def test_20_maximum_frontier_nested_interval(self):
        data = graph(); data['satellite_id'][:] = 'a'
        data['start_ticks'] = np.array([0, 1, 3, 100, 100, 100], dtype=np.int64)
        data['end_ticks'] = np.array([10, 2, 4, 101, 101, 101], dtype=np.int64)
        self.assertFalse(membership_audit(data, [0, 1, 2])['feasible'])
        self.assertEqual(membership_audit(data, [0, 1, 2])['satellite_timeline']['violations'], 2)

    def test_21_gap_equality_allowed(self):
        data = graph(); data['satellite_id'][:2] = 'a'
        data['start_ticks'][1] = 1
        self.assertTrue(membership_audit(data, [0, 1])['feasible'])

    def test_22_strict_shared_blocker_joint(self):
        item = strict_pair_audit(graph(), [2], [0, 1], [[2], [2]])
        self.assertEqual(item['unary_gain_ticks'], [-2, -2])
        self.assertEqual(item['pair_gain_ticks'], 1)
        self.assertEqual(item['after_selected'], [0, 1])

    def test_23_wrong_per_side_blocker_rejected(self):
        with self.assertRaises(ValueError): strict_pair_audit(graph(), [2], [0, 1], [[2], []])

    def test_24_nonnegative_singleton_rejected(self):
        data = graph(); data['weight_ticks'][0] = 5
        with self.assertRaises(ValueError): strict_pair_audit(data, [2], [0, 1])

    def test_25_all_six_population_pairs(self):
        item = population_audit(graph(), [[2], [2], [0, 1], [0, 1]])
        self.assertEqual(item['distinct_masks'], 2)
        self.assertAlmostEqual(item['normalized_pairwise_hamming'], 1/3)
        self.assertAlmostEqual(item['normalized_weighted_pairwise_hamming'], (4*11)/(6*14))
        check_diversity(item, item)
        with self.assertRaises(ValueError): check_diversity(dict(item, distinct_masks=3), item)

    def test_26_late_first_event_fills_only_earliest_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = {'schema': 'barr_fixed_capture_v1', 'config': NATIVE_CONFIG, 'seed': 101, 'thresholds': [60,120,180], 'n': 6, 'm': 2, 'graph_sha256': 'g'*64, 'result': {'selected': [2], 'tick_value': 5, 'original_value': 5e-6}, 'snapshots': [{'threshold_seconds': 60, 'capture_elapsed_seconds': 181, 'epoch': 8, 'capture_io_seconds': .01, 'scout_seconds': .02}], 'missing_thresholds': [120, 180]}
            dump_new(root / 'capture.json', native)
            with patch('g_state_audit.snapshot_audit', return_value={}), patch('g_state_audit.graph_data', return_value=graph()):
                checked = capture_audit(root, {'native_output': 'capture.json', 'graph': 'g0340'}, {'graphs':[{'id':'g0340','n':6,'m':2,'initial_ticks':5,'native_sha256':'g'*64}]})
            self.assertEqual(checked['missing_thresholds'], [120, 180])

    def test_27_same_epoch_multiple_slots_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = {'schema': 'barr_fixed_capture_v1', 'config': NATIVE_CONFIG, 'seed': 101, 'thresholds': [60,120,180], 'n': 6, 'm': 2, 'graph_sha256': 'g'*64, 'result': {'selected': [2], 'tick_value': 5, 'original_value': 5e-6}, 'snapshots': [{'threshold_seconds': t, 'capture_elapsed_seconds': 181, 'epoch': 8} for t in [60, 120]], 'missing_thresholds': [180]}
            dump_new(root / 'capture.json', native)
            with patch('g_state_audit.snapshot_audit', return_value={}), patch('g_state_audit.graph_data', return_value=graph()):
                with self.assertRaises(ValueError): capture_audit(root, {'native_output': 'capture.json', 'graph': 'g0340'}, {'graphs':[{'id':'g0340','n':6,'m':2,'initial_ticks':5,'native_sha256':'g'*64}]})

    def test_28_continue_common_fused_feedback(self):
        self.replay_case('continue', fused=True)

    def test_29_recover_known_pair_only(self):
        self.replay_case('recover', fused=False)

    def test_30_rekey_non_rng_invariant(self):
        self.replay_case('continue', fused=False, seed=1901)

    def test_31_prohibited_future_kernel_rejected(self):
        self.replay_case('continue', fused=False, illegal_kernel=True)

    def replay_case(self, action, fused, seed=0, illegal_kernel=False):
        data = graph()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); blob = root / 'snapshot.bin'; blob.write_bytes(b'offline-full-state-fixture')
            digest = sha(blob); before = [sol([2], 5) for _ in range(4)]
            if fused: before[0] = sol([], 0)
            meta = {'graph_sha256': 'g'*64, 'archive': sol([2], 5, True), 'population': before,
                    'target': sol([2], 5), 'action': {'outside': [0, 1], 'blockers': [[2], [2]]},
                    'used_fusion': fused, 'feedback_target_index': 0,
                    'controller': {'allstate_sha256': digest, 'non_rng_sha256': 'n'*64}}
            dump_new(root / 'snapshot.meta.json', meta)
            after = deepcopy(before)
            target_after = sol([0, 1], 6) if action == 'recover' else sol([2], 5)
            if fused or action == 'recover': after[0] = target_after
            archive_after = sol([0, 1], 6, True) if action == 'recover' else sol([2], 5, True)
            raw = {'schema': 'barr_fixed_replay_v1', 'config': deepcopy(NATIVE_CONFIG), 'budget_seconds': 360,
                   'snapshot_sha256': digest, 'graph_sha256': 'g'*64, 'action': action, 'future_seed': seed,
                   'roundtrip_equal': True, 'io_paths_restored': False, 'ready_state_sha256': digest if seed == 0 else 'r'*64,
                   'ready_non_rng_sha256': 'n'*64, 'feedback_target_index': 0,
                   'after_action_includes_pending_feedback': True, 'feedback_applied': fused,
                   'target_after_action': dict(target_after, raw=target_after['ticks']/1e6),
                   'archive_before': sol([2], 5, True), 'archive_after_action': archive_after,
                   'archive_terminal': archive_after, 'population_before': before,
                   'population_after_action': after, 'population_terminal': after,
                   'future_stats': {'joint_scout_calls': 0, 'kernel_calls': 1 if illegal_kernel else 0, 'joint_recovery_calls': 0},
                   'native_seconds': 360., 'cpu_seconds': 359., 'action_seconds': .01, 'continuation_seconds': 359.99,
                   'capture_elapsed_seconds': 60.1, 'capture_sunk_scout_seconds': .02, 'capture_sunk_prefix_seconds': .03,
                   'pair_apply_seconds': .005 if action == 'recover' else 0, 'feedback_seconds': .001}
            for stage, pop in [('before', before), ('after_action', after), ('terminal', after)]:
                raw['diversity_' + stage] = population_audit(data, [p['selected'] for p in pop])
            dump_new(root / 'replay.json', raw)
            row = {'native_output': 'replay.json', 'graph': 'g0340', 'arm': action, 'future_seed': seed,
                   'snapshot_path': 'snapshot.bin', 'snapshot_sha256': digest, 'metadata_path': 'snapshot.meta.json'}
            with patch('g_state_audit.graph_data', return_value=data):
                if illegal_kernel:
                    with self.assertRaises(ValueError): replay_audit(root, row, {})
                else:
                    checked = replay_audit(root, row, {})
                    self.assertTrue(checked['verified'])
                    self.assertEqual(checked['direct_action_archive_gain_ticks'], 1 if action == 'recover' else 0)


if __name__ == '__main__': unittest.main()
