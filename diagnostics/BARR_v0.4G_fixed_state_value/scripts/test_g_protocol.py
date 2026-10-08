"""Offline protocol/arithmetic fixtures; these are not solver experiments."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from g_protocol import (ARMS, CPUS, GRAPHS, NATIVE_CONFIG, ROOT_NAME, STREAMS, canonical_sha,
                        dump_new, inside, plans, read, validate_design, validate_plans, verify_ledger, sha)
from analyze_g_remote import descriptive, paired_states, cost_audit
from run_g_remote import command, capture_tasks


def design():
    return read(Path(__file__).resolve().parents[1] / 'fixed_state_design.json')


class ProtocolTests(unittest.TestCase):
    def test_01_upper_scope_exact(self):
        self.assertTrue(validate_design(design()))

    def test_02_slot_matrix_and_order(self):
        capture, fork = plans(design())
        self.assertEqual(len(capture['cells']), 8)
        self.assertEqual(len(fork['states']), 24)
        orders = Counter(tuple(p['arm_order']) for state in fork['states'] for p in state['pairs'])
        self.assertEqual(orders, Counter({('recover', 'continue'): 60, ('continue', 'recover'): 60}))
        for cpu in CPUS:
            states = [s for s in fork['states'] if s['cpu'] == cpu]
            self.assertEqual(len(states), 2)
            self.assertEqual(Counter(tuple(p['arm_order']) for s in states for p in s['pairs']), Counter({('recover', 'continue'): 5, ('continue', 'recover'): 5}))

    def test_03_budgets_cannot_shorten(self):
        capture, fork = plans(design())
        capture['seconds'] = 10
        with self.assertRaises(ValueError): validate_plans(capture, fork)

    def test_04_streams_cannot_substitute(self):
        capture, fork = plans(design())
        fork['states'][0]['pairs'][0]['future_seed'] = 1999
        with self.assertRaises(ValueError): validate_plans(capture, fork)

    def test_05_same_core_orders_cannot_adapt(self):
        capture, fork = plans(design())
        fork['states'][12]['pairs'][0]['arm_order'] = ARMS[:]
        with self.assertRaises(ValueError): validate_plans(capture, fork)

    def test_06_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): inside(Path(directory), '../foreign', False)
            with self.assertRaises(ValueError): inside(Path(directory), directory, False)

    def test_07_frozen_ledger_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'proof.txt'; path.write_text('frozen')
            ledger = {'proof.txt': sha(path)}
            self.assertTrue(verify_ledger(root, ledger)['pass_all'])
            path.write_text('changed')
            self.assertFalse(verify_ledger(root, ledger)['pass_all'])

    def test_08_new_receipt_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'; dump_new(path, {'a': 1})
            with self.assertRaises(FileExistsError): dump_new(path, {'a': 2})

    def test_09_native_commands_fresh360(self):
        root = Path('/isolated')
        tasks = capture_tasks(plans(design())[0])
        argv = command(root, tasks[0], root / 'cell')
        self.assertEqual(argv[argv.index('--seconds') + 1], '360')
        self.assertEqual(argv[argv.index('--seed') + 1], '101')
        self.assertNotIn('--future-seed', argv)

    def test_10_exact_rational_and_negative_retained(self):
        item = descriptive([1, 2, -8, 4, 5], 2.7764451051977987)
        self.assertEqual(item['mean_delta_ticks_exact'], {'numerator': 4, 'denominator': 5})
        self.assertEqual(item['wins_ties_losses'], [4, 0, 1])
        self.assertEqual(item['descriptive_label'], 'undetermined')

    def test_11_all_negative_is_estimated_not_proof(self):
        item = descriptive([-4, -4, -4, -4, -4], 2.7764451051977987)
        self.assertEqual(item['descriptive_label'], 'estimated_negative')

    def test_12_missing_slots_and_failed_pairs_no_mean(self):
        fork = plans(design())[1]
        states, errors = paired_states(fork, {'available_states': [fork['states'][0]]}, [])
        self.assertEqual(states[0]['status'], 'INCOMPLETE_FORK_FRAME')
        self.assertIsNone(states[0]['five_stream_descriptive'])
        self.assertEqual(sum(s['status'] == 'MISSING_CAPTURE_SLOT' for s in states), 23)
        self.assertEqual(len(errors), 5)

    def test_13_child_cpu_not_double_counted(self):
        row = dict(phase='fork', arm='recover', cpu=5, cell_id='x', status='COMPLETE',
                   native_cpu_seconds=359, native_seconds=360, cpu_wall_ratio=359/360,
                   native_process_cpu_seconds=359.5, controller_cpu_seconds=.5,
                   controller_only_cpu_seconds=.5, total_cpu_seconds=360,
                   resource_qualified=True, resource_flags=[], affinity=[5], population=4, threads=1)
        checked = cost_audit([row]); self.assertTrue(checked['pass_all'])
        row['total_cpu_seconds'] += 359
        self.assertFalse(cost_audit([row])['pass_all'])

    def test_14_resource_flags_remain_quality_descriptive(self):
        row = dict(phase='fork', arm='continue', cpu=5, cell_id='x', status='COMPLETE',
                   native_cpu_seconds=180, native_seconds=360, cpu_wall_ratio=.5,
                   native_process_cpu_seconds=180.5, controller_cpu_seconds=.5,
                   controller_only_cpu_seconds=.5, total_cpu_seconds=181,
                   resource_qualified=False, resource_flags=['native_cpu_wall_ratio_below_registered_threshold'],
                   affinity=[5], population=4, threads=1)
        checked = cost_audit([row]); self.assertTrue(checked['pass_all']); self.assertEqual(checked['resource_flagged'], 1)

    def test_15_prespecified_missing_not_rebalanced(self):
        fork = plans(design())[1]
        available = [fork['states'][0]]
        actual_orders = Counter(tuple(p['arm_order']) for s in available for p in s['pairs'])
        self.assertEqual(actual_orders, Counter({('recover', 'continue'): 3, ('continue', 'recover'): 2}))

    def test_16_complete_config_cooldown_E(self):
        self.assertEqual(NATIVE_CONFIG['gate_cooldown'], .5)
        self.assertEqual(NATIVE_CONFIG['pair_component'], 'full')
        self.assertEqual(NATIVE_CONFIG['population'], 4)


if __name__ == '__main__': unittest.main()
