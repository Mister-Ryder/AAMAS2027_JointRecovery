"""One-shot launch: immutable online freeze, fixed16 guards, then actual run.

No fitting, confirmation generation or optional policy warmup is provided.
All model/data/native-kernel arguments remain explicit. The frozen online
protocol is never edited by this wrapper; guards write separate receipts.
"""
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    # Canonical name makes the script's actual source visible to the existing
    # scientific import-origin validator, without loading a duplicate module.
    sys.modules['experiments.v4_residual_online_validation_execute']=sys.modules[__name__]

from experiments import v4_residual_online_validation as online
from experiments import v4_residual_fit_gate as fit_gate

MAIN_SHA256='80750a6959e4d36e0c463fde6d1a2aba1e2baa929aa5cf59905ac6f36f309d68'
MAIN_TEST_SHA256='d6d7511137e3dcb40690ffeab7996700d86ff0ae27487ab34a69e958d27b0841'
GUARD_PATTERN='test_v4_residual_online_validation.py'
WRAPPER_PATTERN='test_v4_residual_online_validation_execute.py'
EXPECTED_GUARDS=16
REQUIRED_SOURCES=tuple(sorted(set(fit_gate.REQUIRED_PATHS)|
    set(online.MODULE_PATHS.values())|{'tests/test_v4_residual_online_validation.py',
    'experiments/v4_residual_online_validation_execute.py',
    'tests/test_v4_residual_online_validation_execute.py'}))


def prepare_sources(args):
    """Bind the complete version-specific33 source union before freezing.

    Test modules are imported for exact origin checks. Only the sixteen
    already frozen online tests are later executed in the actual launch.
    The two wrapper mocks are preparation checks, not an extra policy warmup.
    """
    if (online.digest(ROOT/'experiments/v4_residual_online_validation.py')!=MAIN_SHA256 or
            online.digest(ROOT/'tests'/GUARD_PATTERN)!=MAIN_TEST_SHA256):
        raise ValueError('The existing online source/test bytes must remain frozen')
    suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern=GUARD_PATTERN)
    wrapper_suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern=WRAPPER_PATTERN)
    if suite.countTestCases()!=EXPECTED_GUARDS or wrapper_suite.countTestCases()!=2:
        raise ValueError('Exact predeclared sixteen guards and two wrapper mocks required')
    fit_gate.source_gate(args.source_manifest,args.source_manifest_sha256)
    source=online.validate_source_closure(ROOT,args.source_manifest,args.source_manifest_sha256)
    by_path={row['path']:row['sha256'] for row in source['files']}
    if len(REQUIRED_SOURCES)!=33 or not set(REQUIRED_SOURCES)<=set(by_path):
        raise ValueError('Complete33 residual source union is required')
    origins=[]
    for name,relative in (
            ('experiments.v4_residual_online_validation_execute','experiments/v4_residual_online_validation_execute.py'),
            ('test_v4_residual_online_validation','tests/test_v4_residual_online_validation.py'),
            ('test_v4_residual_online_validation_execute','tests/test_v4_residual_online_validation_execute.py')):
        module=sys.modules.get(name);file=getattr(module,'__file__',None)
        if file is None or Path(file).resolve()!=ROOT/relative or online.digest(file)!=by_path[relative]:
            raise ValueError('Actual launcher/guard source origin mismatch: '+name)
        origins.append(dict(module=name,path=relative,sha256=by_path[relative]))
    return suite,dict(required_sources=REQUIRED_SOURCES,source_manifest_sha256=args.source_manifest_sha256,
        launcher_and_guard_origins=origins,all_declared_sources_hashed=True,
        unused_original_collector_sources_are_hash_bound_not_executed=True)


def require_guards(record):
    if (record.get('tests_run')!=EXPECTED_GUARDS or record.get('expected_tests')!=EXPECTED_GUARDS or
            record.get('successful') is not True or any(record.get(name)!=0 for name in (
                'failures','errors','skipped','expected_failures','unexpected_successes'))):
        raise RuntimeError('The frozen sixteen guards must all pass with no skip/failure')


def run_guard_suite(suite,out,protocol_sha256):
    stream=StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (out/'one_shot_guards.txt').open('x',encoding='utf-8') as handle:handle.write(stream.getvalue())
    record=dict(status='finite_online_guards_not_production_policy',expected_tests=EXPECTED_GUARDS,tests_run=result.testsRun,
        successful=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),expected_failures=len(result.expectedFailures),
        unexpected_successes=len(result.unexpectedSuccesses),
        frozen_protocol_sha256=protocol_sha256,log_sha256=online.digest(out/'one_shot_guards.txt'),
        no_dataset_or_solver_in_test_fixtures=True,no_fitting=True,no_production_policy_warmup=True)
    online.write_json(out/'one_shot_guards.json',record)
    return record


def bind_protocol(out,source_manifest_sha256):
    protocol_sha=online.digest(out/'protocol.json');closed=online.fit._read_json(out/'freeze_completion.json')
    if (closed.get('status')!='complete_online_validation_freeze_not_execution' or
            closed.get('expected_units')!=1512 or closed.get('protocol_sha256')!=protocol_sha or
            closed.get('source_manifest_sha256')!=source_manifest_sha256):
        raise ValueError('The newly generated online freeze receipt is not completely bound')
    return protocol_sha


def execute(args):
    """Exactly one freeze and one run; failures are retained without retry."""
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError('One-shot output/attempt must be entirely new')
    for input_path in (args.collection_root,args.fit_root):
        input_root=Path(input_path).resolve()
        if out==input_root or input_root in out.parents:
            raise ValueError('One-shot output cannot mutate the sealed collection/fits')
    try:
        suite,source_receipt=prepare_sources(args)
        freeze_args=argparse.Namespace(**vars(args));freeze_args.stage='freeze';freeze_args.online_protocol_sha256=None
        online.freeze(freeze_args)
        protocol_sha=bind_protocol(out,args.source_manifest_sha256)
        guards=run_guard_suite(suite,out,protocol_sha);require_guards(guards)
        if bind_protocol(out,args.source_manifest_sha256)!=protocol_sha:
            raise ValueError('The guards must never change the frozen online protocol')
        online.validate_source_closure(ROOT,args.source_manifest,args.source_manifest_sha256)
        online.write_json(out/'one_shot_pre_run.json',dict(status='complete_one_shot_guard_barrier_before_online_run',
            frozen_protocol_sha256=protocol_sha,guards_sha256=online.digest(out/'one_shot_guards.json'),
            source_receipt=source_receipt,planned_units=1512,no_protocol_mutation=True,
            no_optional_model_warmup=True,no_fitting=True,no_confirmation_generation=True))
        run_args=argparse.Namespace(**vars(args));run_args.stage='run';run_args.online_protocol_sha256=protocol_sha
        return online.run(run_args)
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True)
        if not any((out/name).exists() for name in ('completion.json','failure.json')):
            online.write_json(out/'failure.json',dict(status='failed_one_shot_online_validation_no_retry',completed=False,
                exception_type=type(error).__name__,message=str(error),no_fitting=True,no_confirmation_generation=True))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--collection-root',required=True);parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--completion-sha256',required=True);parser.add_argument('--fit-root',required=True)
    parser.add_argument('--fit-protocol-sha256',required=True);parser.add_argument('--fit-completion-sha256',required=True)
    parser.add_argument('--source-manifest',required=True);parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--calibration',required=True);parser.add_argument('--chils',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    return execute(parser.parse_args())


if __name__=='__main__':main()
