"""After four sealed fits, run all declared development comparisons only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import p0_recovery_probes as p0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset-root','runtime-root','fit-root','p0-root','out','chils','chils-source'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh followup output required')
    out.mkdir(parents=True,exist_ok=True)
    fit=Path(args.fit_root).resolve(); started=time.monotonic()
    launch=json.loads((fit.parent/'launcher.json').read_text())
    status={'status':'WAITING_FOR_FOUR_SEALED_FITS','fit_root':str(fit),
            'sources':[4,5],'native_workpoints_ms':[10,50,200,1000],
            'no_validation_or_test_evaluation':True}
    p0.write_json(out/'progress.json',status)
    while not (fit/'completion.json').is_file():
        proc=Path('/proc',str(launch['pid']),'stat')
        if not proc.exists() or proc.read_text().split()[2]=='Z':
            raise RuntimeError('Fit stopped before sealed completion; original fit output preserved')
        if time.monotonic()-started>7200:raise TimeoutError('Fit wait timed out; original job preserved')
        time.sleep(15)
    completion=json.loads((fit/'completion.json').read_text())
    if completion['status']!='ACTUAL_P1_FIT_COMPLETE' or len(completion['fits'])!=4:
        raise ValueError('Four declared fits are not complete')
    code=Path(__file__).resolve().parent
    status['status']='RUNNING_ALL_DEVELOPMENT_FIXED_CALL_COMPARISONS';p0.write_json(out/'progress.json',status)
    print(json.dumps({'stage':status['status']}),flush=True)
    subprocess.run([sys.executable,str(code/'p1_fit_and_allocate.py'),'allocate',
        '--runtime-root',args.runtime_root,'--graphs-dir',str(Path(args.dataset_root)/'graphs'),
        '--p0-root',args.p0_root,'--fit-root',str(fit),'--sources','4','5',
        '--out',str(out/'fixed_calls'),'--device','cpu'],check=True)
    status['status']='RUNNING_72_SERIAL_ACTUAL_CALLER_CALIBRATION_CELLS';p0.write_json(out/'progress.json',status)
    print(json.dumps({'stage':status['status']}),flush=True)
    subprocess.run([sys.executable,str(code/'p1_benchmark_driver.py'),'calibrate',
        '--dataset-root',args.dataset_root,'--runtime-root',args.runtime_root,'--fit-root',str(fit),
        '--actual-cli',str(code/'p1_fit_and_allocate.py'),'--chils',args.chils,'--chils-source',args.chils_source,
        '--budgets-ms','10','50','200','1000','--sources','4','5','--out',str(out/'calibration')],check=True)
    budget=json.loads((out/'calibration/frozen_budgets.json').read_text())
    if budget['status']!='BUDGETS_FROZEN_FROM_DEVELOPMENT_ACTUAL_CALLERS':
        raise RuntimeError('Actual calibration did not produce frozen budgets')
    status.update(status='ALL_CORE_P1_DEVELOPMENT_COMPARISONS_COMPLETE',
                  fit_completion_sha256=p0.sha_file(fit/'completion.json'),
                  fixed_call_summary_sha256=p0.sha_file(out/'fixed_calls/summary.json'),
                  frozen_budgets_sha256=p0.sha_file(out/'calibration/frozen_budgets.json'))
    p0.write_json(out/'completion.json',status);p0.write_json(out/'progress.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
