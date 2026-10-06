"""Finish predeclared TRAIN source preparation and real probe collection only.

No fitting, model selection, test generation, or result-dependent filtering.
Reads STK success manifests; each fixed source produces all six fixed graphs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import cloud_runner as cloud


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-seconds',type=int,default=3600)
    args=parser.parse_args()
    root=cloud.ROOT
    status_path=root/'execution/TRAIN_SOURCE_PIPELINE.json'
    if status_path.exists():raise FileExistsError('Pipeline status exists; do not duplicate native jobs')
    status={'status':'RUNNING','sources':{},'no_training_or_test_selection':True,
            'probe_budgets_ms':[200,1000,50,10],'repeats':2,'target_states_per_graph':12,
            'short_probe_basis':'P0_SHORT_PROBE_GATE.json: actual r000 net opportunity established'}
    started=time.monotonic()
    def save():
        status['elapsed_seconds']=time.monotonic()-started
        pending=status_path.with_suffix('.new')
        pending.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
        pending.replace(status_path)
    client=cloud.connect()
    try:
        save()
        while True:
            for source in range(2,6):
                name='r%03d'%source
                if name in status['sources']:continue
                manifest=root/'raw_geometry'/('JR-DUAL-'+name)/'manifest.json'
                if not manifest.exists():continue
                record=json.loads(manifest.read_text(encoding='utf-8'))
                if record['status']=='failed':raise RuntimeError('STK source failed: '+name)
                if record['status']!='success':continue
                print(json.dumps({'source':name,'stage':'BUILD_ALL_SIX_GRAPHS'}),flush=True)
                subprocess.run([sys.executable,str(root/'code/build_graphs.py'),'--dataset-root',str(root),
                                '--replicates',name],check=True)
                graph_ids=['JR-DUAL-'+name+'-'+network+'-g%04d'%gap
                           for network in ('R12','R8') for gap in (170,340,680)]
                cloud.stage_graphs(client,graph_ids)
                jobs=[]
                for index,graph_id in enumerate(graph_ids):
                    job='p1_labels_'+name+'_%d'%index
                    cloud.launch(client,SimpleNamespace(job=job,graphs=[graph_id],states=12,repeats=2,
                                 budgets_ms=[200,1000,50,10],cpu_index=8+6*(source-2)+index))
                    jobs.append(job)
                status['sources'][name]={'geometry_manifest_sha256':cloud.digest(manifest),
                                        'graph_ids':graph_ids,'jobs':jobs,'stage':'PROBES_RUNNING'}
                save()
            if status['sources']:
                jobs=[job for info in status['sources'].values() for job in info['jobs']]
                code='''import pathlib,json
root=pathlib.Path(ROOT); rows=[]
for job in JOBS:
    directory=root/"runs"/job
    launch=json.loads((directory/"launcher.json").read_text())
    proc=pathlib.Path("/proc",str(launch["pid"]),"stat")
    active=proc.exists() and proc.read_text().split()[2]!="Z"
    complete=(directory/"out/batch_summary.json").is_file()
    rows.append({"job":job,"active":active,"complete":complete,
      "error_tail":(directory/"stdout.log").read_text(errors="replace")[-1500:] if not active and not complete else None})
print(json.dumps(rows))
'''
                rows=cloud.run(client,'ROOT='+repr(cloud.REMOTE)+'\nJOBS='+repr(jobs)+'\n'+code)
                lookup={row['job']:row for row in rows}
                for name,info in status['sources'].items():
                    failures=[lookup[job] for job in info['jobs'] if not lookup[job]['active'] and not lookup[job]['complete']]
                    if failures:
                        status['failed_jobs']=failures
                        raise RuntimeError('Actual probe job failed for '+name)
                    if all(lookup[job]['complete'] for job in info['jobs']):
                        if info['stage']!='COMPLETE':print(json.dumps({'source':name,'stage':'PROBES_COMPLETE'}),flush=True)
                        info['stage']='COMPLETE'
                save()
            if len(status['sources'])==4 and all(info['stage']=='COMPLETE' for info in status['sources'].values()):
                status['status']='ALL_FOUR_FIXED_TRAIN_SOURCES_AND_24_PROBE_GRAPHS_COMPLETE'
                save()
                subprocess.run([sys.executable,str(root/'code/dataset_catalog.py'),'--dataset-root',str(root)],check=True)
                print(json.dumps({'status':status['status'],'elapsed_seconds':status['elapsed_seconds']}),flush=True)
                break
            if time.monotonic()-started>args.timeout_seconds:raise TimeoutError('Source pipeline timeout; existing jobs are preserved')
            time.sleep(30)
    except Exception as exc:
        status.update(status='STOPPED_WITH_ERROR_EXISTING_DATA_PRESERVED',error=str(exc));save()
        raise
    finally:client.close()


if __name__=='__main__':main()
