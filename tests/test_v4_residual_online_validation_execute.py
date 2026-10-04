"""Two finite one-shot API mocks; no corpus, solver, model or real online run."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import v4_residual_online_validation_execute as launch


def arguments(root):
    return SimpleNamespace(out=str(root/'out'),collection_root=str(root/'collection'),fit_root=str(root/'fits'),
        protocol_sha256='a'*64,completion_sha256='b'*64,fit_protocol_sha256='c'*64,
        fit_completion_sha256='d'*64,source_manifest=str(root/'manifest.json'),
        source_manifest_sha256='e'*64,calibration=str(root/'calibration.json'),chils=str(root/'pinnedbinary'),device='cuda')


def mock_freeze(args):
    out=Path(args.out);out.mkdir()
    launch.online.write_json(out/'protocol.json',dict(fixed='mock_plan_no_production_graphs'))
    launch.online.write_json(out/'freeze_completion.json',dict(status='complete_online_validation_freeze_not_execution',
        expected_units=1512,protocol_sha256=launch.online.digest(out/'protocol.json'),
        source_manifest_sha256=args.source_manifest_sha256))


def mock_guards(success):
    def guards(suite,out,protocol_sha):
        record=dict(expected_tests=16,tests_run=16,successful=success,failures=0 if success else 1,
            errors=0,skipped=0,expected_failures=0,unexpected_successes=0,frozen_protocol_sha256=protocol_sha)
        launch.online.write_json(out/'one_shot_guards.json',record)
        return record
    return guards


class OneShotTests(unittest.TestCase):
    def test_guard_failure_retains_new_freeze_and_never_runs_or_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=arguments(Path(temporary))
            with patch.object(launch,'prepare_sources',return_value=(None,{})), \
                    patch.object(launch.online,'freeze',side_effect=mock_freeze) as freeze, \
                    patch.object(launch,'run_guard_suite',side_effect=mock_guards(False)), \
                    patch.object(launch.online,'run') as run:
                with self.assertRaises(RuntimeError):launch.execute(args)
                self.assertEqual(freeze.call_count,1);run.assert_not_called()
                out=Path(args.out);before=(out/'protocol.json').read_bytes()
                self.assertFalse(json.loads((out/'failure.json').read_text())['completed'])
                with self.assertRaises(FileExistsError):launch.execute(args)
                self.assertEqual(freeze.call_count,1);run.assert_not_called()
                self.assertEqual((out/'protocol.json').read_bytes(),before)
            for changes in (dict(skipped=1),dict(errors=1),dict(expected_failures=1),dict(unexpected_successes=1),dict(tests_run=15)):
                bad=dict(expected_tests=16,tests_run=16,successful=True,failures=0,errors=0,skipped=0,
                    expected_failures=0,unexpected_successes=0);bad.update(changes)
                with self.assertRaises(RuntimeError):launch.require_guards(bad)

    def test_single_process_binds_actual_generated_sha_and_keeps_all_input_args(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=arguments(Path(temporary));original=deepcopy(vars(args));calls=[]
            def freeze(value):calls.append(('freeze',vars(value).copy()));return mock_freeze(value)
            def run(value):calls.append(('run',vars(value).copy()));return 'mock_only_no_policy_execution'
            with patch.object(launch,'prepare_sources',return_value=(None,{'fixture':'source_only'})), \
                    patch.object(launch.online,'freeze',side_effect=freeze), \
                    patch.object(launch,'run_guard_suite',side_effect=mock_guards(True)), \
                    patch.object(launch.online,'validate_source_closure',return_value={}), \
                    patch.object(launch.online,'run',side_effect=run):
                self.assertEqual(launch.execute(args),'mock_only_no_policy_execution')
            self.assertEqual([step[0] for step in calls],['freeze','run'])
            self.assertEqual(calls[0][1]['stage'],'freeze');self.assertEqual(calls[1][1]['stage'],'run')
            self.assertIsNone(calls[0][1]['online_protocol_sha256'])
            self.assertEqual(calls[1][1]['online_protocol_sha256'],launch.online.digest(Path(args.out)/'protocol.json'))
            self.assertEqual(vars(args),original)
            for key,value in original.items():self.assertEqual(calls[0][1][key],value);self.assertEqual(calls[1][1][key],value)
            receipt=json.loads((Path(args.out)/'one_shot_pre_run.json').read_text())
            self.assertEqual(receipt['frozen_protocol_sha256'],calls[1][1]['online_protocol_sha256'])
            self.assertTrue(receipt['no_protocol_mutation']);self.assertTrue(receipt['no_fitting'])
            self.assertFalse((Path(args.out)/'failure.json').exists())
            self.assertEqual(len(launch.REQUIRED_SOURCES),33)


if __name__=='__main__':unittest.main()
