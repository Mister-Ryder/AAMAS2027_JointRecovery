"""Build all predeclared validation/test graphs without executing any policy."""
from __future__ import annotations
import json
import subprocess
import sys
import time
import cloud_runner as cloud


def main():
    root=cloud.ROOT; path=root/'execution/HOLDOUT_DATA_PIPELINE.json'
    if path.exists():raise FileExistsError('Holdout pipeline already exists')
    status={'status':'RUNNING_GEOMETRY_AND_GRAPH_PREPARATION_ONLY','sources':{},
            'validation_sources':[6,7],'test_sources':[8,9,10,11],
            'algorithm_or_policy_evaluation_performed':False,
            'source_selection':'all predeclared sources; no result-based selection'}
    started=time.monotonic()
    def save():
        status['elapsed_seconds']=time.monotonic()-started
        pending=path.with_suffix('.new');pending.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8');pending.replace(path)
    client=cloud.connect()
    try:
        save()
        while len(status['sources'])<6:
            for source in range(6,12):
                name='r%03d'%source
                if name in status['sources']:continue
                manifest=root/'raw_geometry'/('JR-DUAL-'+name)/'manifest.json'
                if not manifest.exists():continue
                geometry=json.loads(manifest.read_text(encoding='utf-8'))
                if geometry['status']=='failed':raise RuntimeError('STK source failed: '+name)
                if geometry['status']!='success':continue
                print(json.dumps({'source':name,'stage':'BUILD_FIXED_HOLDOUT_GRAPHS_NO_POLICY_EVALUATION'}),flush=True)
                subprocess.run([sys.executable,str(root/'code/build_graphs.py'),'--dataset-root',str(root),'--replicates',name],check=True)
                graph_ids=['JR-DUAL-'+name+'-'+network+'-g%04d'%gap for network in ('R12','R8') for gap in (170,340,680)]
                cloud.stage_graphs(client,graph_ids)
                status['sources'][name]={'geometry_manifest_sha256':cloud.digest(manifest),'graph_ids':graph_ids,'status':'DATA_READY_NOT_EVALUATED'}
                save()
                subprocess.run([sys.executable,str(root/'code/dataset_catalog.py'),'--dataset-root',str(root)],check=True)
            if time.monotonic()-started>3600:raise TimeoutError('Holdout generation timeout; existing sources preserved')
            if len(status['sources'])<6:time.sleep(30)
        status['status']='ALL_SIX_FIXED_HOLDOUT_SOURCES_AND_36_GRAPHS_READY_NOT_EVALUATED';save()
        print(json.dumps({'status':status['status']}),flush=True)
    except Exception as exc:
        status.update(status='STOPPED_WITH_ERROR_EXISTING_DATA_PRESERVED',error=str(exc));save();raise
    finally:client.close()


if __name__=='__main__':main()
