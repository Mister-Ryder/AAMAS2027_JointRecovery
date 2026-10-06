"""Isolated cloud staging and execution for this new STK source family."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import posixpath
import shlex
import stat
import tarfile
import time
import paramiko

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(os.environ.get('JR_RUNTIME_ROOT', str(ROOT / 'runtime')))
REMOTE = '/root/autodl-tmp/AAMAS1979_JointRecovery_STK_20261005'
PYTHON = '/root/miniconda3/bin/python'
CHILS = '/root/autodl-tmp/AAMAS1979_JointRecovery_v01/tools/CHILS-v3/CHILS'
CHILS_SOURCE = '/root/autodl-tmp/AAMAS1979_JointRecovery_v01/tools/CHILS-v3'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):h.update(block)
    return h.hexdigest()

def connect():
    client=paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(os.environ['JR_SSH_HOST'],port=int(os.environ.get('JR_SSH_PORT','22')),
      username=os.environ.get('JR_SSH_USER','root'),key_filename=os.environ['JR_SSH_KEY'],
      allow_agent=False,look_for_keys=False,timeout=20,banner_timeout=30,auth_timeout=30)
    return client

def run(client,code):
    stdin,stdout,stderr=client.exec_command(PYTHON+' -',timeout=120)
    stdin.write(code);stdin.flush();stdin.channel.shutdown_write()
    text=stdout.read().decode('utf-8'); errors=stderr.read().decode('utf-8')
    if stdout.channel.recv_exit_status():raise RuntimeError(errors or text)
    if errors:print(errors)
    return json.loads(text)

def mkdir(sftp,path):
    try:sftp.stat(path)
    except FileNotFoundError:
        mkdir(sftp,posixpath.dirname(path));sftp.mkdir(path)

def source_files():
    result=[]
    for folder in ('src/joint_recovery','experiments'):
        for p in sorted((LEGACY/folder).rglob('*.py')):
            if '__pycache__' not in p.parts:
                result.append((p,'runtime/'+p.relative_to(LEGACY).as_posix()))
    for p in sorted((ROOT/'code').glob('*.py')):
        if p.name!='cloud_runner.py':result.append((p,'code/'+p.name))
    return result

def stage(client,skip_graphs=False):
    files=source_files()
    manifest={'files':[{'path':name,'sha256':digest(p),'bytes':p.stat().st_size} for p,name in files]}
    capsule_id=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    local_work=ROOT/'execution';local_work.mkdir(exist_ok=True)
    package=local_work/(capsule_id+'.tar.gz')
    with tarfile.open(package,'w:gz') as archive:
        for p,name in files:archive.add(p,arcname=name)
    remote_capsule=REMOTE+'/capsules/'+capsule_id
    with client.open_sftp() as sftp:
        mkdir(sftp,REMOTE+'/transfers');mkdir(sftp,REMOTE+'/graphs');mkdir(sftp,REMOTE+'/protocol')
        sftp.put(str(package),REMOTE+'/transfers/'+package.name)
        if not skip_graphs:
            for p in sorted((ROOT/'graphs').glob('*')):
                if p.is_file():sftp.put(str(p),REMOTE+'/graphs/'+p.name)
        sftp.put(str(ROOT/'protocol/JointRecovery_parameters.json'),REMOTE+'/protocol/JointRecovery_parameters.json')
    remote_code='''import pathlib,tarfile,hashlib,json
root=pathlib.Path(ROOT); capsule=pathlib.Path(CAPSULE); package=pathlib.Path(PACKAGE)
assert str(root)=="/root/autodl-tmp/AAMAS1979_JointRecovery_STK_20261005"
assert capsule.parent==root/"capsules"
if not capsule.exists():
    capsule.mkdir(parents=True)
    with tarfile.open(package) as archive:
        for item in archive.getmembers():
            p=(capsule/item.name).resolve()
            assert str(p).startswith(str(capsule)+"/") and item.isfile()
        archive.extractall(capsule)
for record in MANIFEST["files"]:
    p=capsule/record["path"]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"]
(capsule/"source_manifest.json").write_text(json.dumps(MANIFEST,indent=2)+"\\n")
package.unlink()
print(json.dumps({"capsule":str(capsule),"files":len(MANIFEST["files"]),"source_manifest_sha256":hashlib.sha256((capsule/"source_manifest.json").read_bytes()).hexdigest()}))
'''
    prefix='ROOT='+repr(REMOTE)+'\nCAPSULE='+repr(remote_capsule)+'\nPACKAGE='+repr(REMOTE+'/transfers/'+package.name)+'\nMANIFEST='+repr(manifest)+'\n'
    receipt=run(client,prefix+remote_code)
    receipt.update(remote_root=REMOTE,capsule_id=capsule_id,source_manifest=manifest)
    (local_work/'CLOUD_STAGE.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_manifest'}))

def fetch(client,job):
    remote=REMOTE+'/runs/'+job
    local=ROOT/'execution'/job
    local.mkdir(parents=True,exist_ok=True)
    count=0
    with client.open_sftp() as sftp:
        def walk(source,dest):
            nonlocal count
            for entry in sftp.listdir_attr(source):
                if entry.filename in ('.','..') or '/' in entry.filename or '\\' in entry.filename:raise ValueError('Bad remote filename')
                rp=source+'/'+entry.filename;lp=dest/entry.filename
                if stat.S_ISDIR(entry.st_mode):lp.mkdir(exist_ok=True);walk(rp,lp)
                elif stat.S_ISREG(entry.st_mode):sftp.get(rp,str(lp));count+=1
        walk(remote,local)
    print(json.dumps({'job':job,'downloaded_files':count,'local':str(local)}))

def fetch_archive(client,job,collection=False):
    if not job or not all(c.isalnum() or c in '_-' for c in job):raise ValueError('Safe job ID required')
    code='''import pathlib,json,tarfile,hashlib
root=pathlib.Path(ROOT); source=root/STORAGE/JOB
assert root==pathlib.Path("/root/autodl-tmp/AAMAS1979_JointRecovery_STK_20261005")
assert source.is_dir() and source.parent==root/STORAGE
if STORAGE=="runs":
    launch=json.loads((source/"launcher.json").read_text())
    proc=pathlib.Path("/proc",str(launch["pid"]),"stat")
    assert not proc.exists() or proc.read_text().split()[2]=="Z","Job is still running"
else:assert (source/"collection.json").is_file()
archive=root/"transfers"/(JOB+".results.tar.gz")
archive.parent.mkdir(exist_ok=True)
with tarfile.open(archive,"w:gz",compresslevel=3) as tar:
    for p in sorted(source.rglob("*")):
        assert not p.is_symlink()
        if p.is_file():tar.add(p,arcname=JOB+"/"+p.relative_to(source).as_posix(),recursive=False)
print(json.dumps({"archive":str(archive),"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"bytes":archive.stat().st_size}))
'''
    info=run(client,'ROOT='+repr(REMOTE)+'\nJOB='+repr(job)+'\nSTORAGE='+repr('collections' if collection else 'runs')+'\n'+code)
    archive=ROOT/'execution'/(job+'.results.tar.gz')
    with client.open_sftp() as sftp:sftp.get(info['archive'],str(archive))
    if digest(archive)!=info['sha256']:raise ValueError('Downloaded result archive differs')
    destination=(ROOT/'execution').resolve()
    with tarfile.open(archive,'r:gz') as tar:
        for item in tar.getmembers():
            resolved=(destination/item.name).resolve()
            if not str(resolved).startswith(str(destination/ job)+os.sep) or not item.isfile():
                raise ValueError('Invalid result member: '+item.name)
        tar.extractall(destination,filter='data')
    receipt=dict(job=job,archive_sha256=info['sha256'],archive_bytes=info['bytes'],local=str(destination/job))
    (destination/('FETCH_'+job+'.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    # Remove only the verified task-owned transfer archive, never result files.
    with client.open_sftp() as sftp:sftp.remove(info['archive'])
    archive.unlink()
    print(json.dumps(receipt))

def stage_graphs(client,names):
    if not names or any(not all(c.isalnum() or c in '_-' for c in name) for name in names):
        raise ValueError('Explicit safe graph IDs required')
    records=[]
    with client.open_sftp() as sftp:
        mkdir(sftp,REMOTE+'/graphs')
        for name in names:
            for suffix in ('.npz','.json'):
                path=ROOT/'graphs'/(name+suffix)
                if not path.is_file():raise FileNotFoundError(str(path))
                remote=REMOTE+'/graphs/'+path.name
                expected=digest(path)
                try:
                    with sftp.open(remote,'rb') as stream:
                        existing=hashlib.sha256(stream.read()).hexdigest()
                    if existing!=expected:raise ValueError('Frozen remote graph differs: '+path.name)
                    uploaded=False
                except FileNotFoundError:
                    sftp.put(str(path),remote+'.incoming')
                    sftp.rename(remote+'.incoming',remote)
                    uploaded=True
                records.append({'file':path.name,'sha256':expected,'uploaded':uploaded})
    receipt={'graph_ids':names,'files':records,'remote_root':REMOTE}
    (ROOT/'execution'/('GRAPH_STAGE_'+names[0]+'.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'graphs':len(names),'uploaded_files':sum(r['uploaded'] for r in records)}))

def stage_protocols(client,names):
    if not names or any(Path(name).name!=name for name in names):raise ValueError('Explicit protocol filenames required')
    records=[]
    with client.open_sftp() as sftp:
        mkdir(sftp,REMOTE+'/protocol')
        for name in names:
            path=ROOT/'protocol'/name
            if not path.is_file():raise FileNotFoundError(str(path))
            expected=digest(path); remote=REMOTE+'/protocol/'+name
            try:
                with sftp.open(remote,'rb') as stream:actual=hashlib.sha256(stream.read()).hexdigest()
                if actual!=expected:raise ValueError('Frozen remote protocol differs: '+name)
                uploaded=False
            except FileNotFoundError:
                sftp.put(str(path),remote+'.incoming');sftp.rename(remote+'.incoming',remote);uploaded=True
            records.append({'file':name,'sha256':expected,'uploaded':uploaded})
    print(json.dumps({'protocols':records}))

def launch(client,args):
    staged=json.loads((ROOT/'execution/CLOUD_STAGE.json').read_text(encoding='utf-8'))
    capsule=staged['capsule']
    graphs=args.graphs
    if not graphs or any(not all(c.isalnum() or c in '_-' for c in name) for name in graphs):
        raise ValueError('Safe graph IDs required')
    job=args.job
    if not job or not all(c.isalnum() or c in '_-' for c in job):raise ValueError('Safe job ID required')
    command=[PYTHON,capsule+'/code/p0_recovery_probes.py','--runtime-root',capsule+'/runtime',
       '--chils',CHILS,'--chils-source',CHILS_SOURCE,'--out',REMOTE+'/runs/'+job+'/out',
       '--states',str(args.states),'--repeats',str(args.repeats),'--seed','17','--budgets-ms']
    command += [str(x) for x in args.budgets_ms]
    for graph in graphs:command += ['--graph',REMOTE+'/graphs/'+graph+'.npz']
    code='''import json,pathlib,subprocess,os,time
root=pathlib.Path(ROOT); directory=root/"runs"/JOB
assert str(root)=="/root/autodl-tmp/AAMAS1979_JointRecovery_STK_20261005"
directory.mkdir(parents=True,exist_ok=False)
available=sorted(os.sched_getaffinity(0)); cpu=available[CPU_INDEX%len(available)]
environment=os.environ.copy()
environment.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONUNBUFFERED="1")
with (directory/"stdout.log").open("wb") as log:
    process=subprocess.Popen(COMMAND,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
      env=environment,cwd=directory,start_new_session=True,close_fds=True,
      preexec_fn=lambda:os.sched_setaffinity(0,{cpu}))
receipt={"pid":process.pid,"job":JOB,"cpu_affinity":[cpu],"command":COMMAND,"capsule":CAPSULE,
  "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"native_threads":1,"native_seed":17}
(directory/"launcher.json").write_text(json.dumps(receipt,indent=2)+"\\n")
print(json.dumps(receipt))
'''
    prefix='ROOT='+repr(REMOTE)+'\nJOB='+repr(job)+'\nCOMMAND='+repr(command)+'\nCAPSULE='+repr(capsule)+'\nCPU_INDEX='+repr(args.cpu_index)+'\n'
    receipt=run(client,prefix+code)
    (ROOT/'execution'/('LAUNCH_'+job+'.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))

def status(client):
    code='''import pathlib,json,os,shutil
root=pathlib.Path(ROOT); rows=[]
for p in sorted((root/"runs").glob("*")) if (root/"runs").exists() else []:
    row={"job":p.name}
    for name in ("progress.json","completion.json","batch_summary.json","failure.json"):
        q=p/"out"/name
        if q.exists():row[name]=json.loads(q.read_text())
    if (p/"launcher.json").exists():
        launch=json.loads((p/"launcher.json").read_text());row["pid"]=launch["pid"];row["running"]=pathlib.Path("/proc",str(launch["pid"])).exists()
        if row["running"]:
            try:row["running"]=(pathlib.Path("/proc",str(launch["pid"]),"stat").read_text().split()[2]!="Z")
            except FileNotFoundError:row["running"]=False
    row["complete_states"]=len(list((p/"out").rglob("snapshot_*.json")))
    if (p/"stdout.log").exists():row["log_tail"]=(p/"stdout.log").read_text(errors="replace")[-3500:]
    rows.append(row)
print(json.dumps({"jobs":rows,"free_bytes":shutil.disk_usage(root).free}))
'''
    result=run(client,'ROOT='+repr(REMOTE)+'\n'+code)
    (ROOT/'execution/CLOUD_STATUS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    # Whole detailed state is saved locally; concise status is returned to chat.
    text=[]
    for row in result['jobs']:
        text.append({'job':row['job'],'pid':row.get('pid'),'running':row.get('running'),
                     'complete_states':row.get('complete_states'),'complete':'completion.json' in row or 'batch_summary.json' in row,
                     'log_tail':row.get('log_tail','')[-1500:],'failure':row.get('failure.json')})
    print(json.dumps({'jobs':text,'free_bytes':result['free_bytes']}))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['stage','stage-graphs','stage-protocols','status','fetch','fetch-archive','fetch-collection','launch'])
    parser.add_argument('--job')
    parser.add_argument('--graphs',nargs='+')
    parser.add_argument('--files',nargs='+')
    parser.add_argument('--skip-graphs',action='store_true')
    parser.add_argument('--states',type=int,default=12)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--budgets-ms',nargs='+',type=int,default=[200,1000])
    parser.add_argument('--cpu-index',type=int,default=0)
    args=parser.parse_args()
    client=connect()
    try:
        if args.operation=='stage':stage(client,args.skip_graphs)
        elif args.operation=='stage-graphs':stage_graphs(client,args.graphs)
        elif args.operation=='stage-protocols':stage_protocols(client,args.files)
        elif args.operation=='status':status(client)
        elif args.operation=='launch':launch(client,args)
        elif args.operation=='fetch-archive':fetch_archive(client,args.job)
        elif args.operation=='fetch-collection':fetch_archive(client,args.job,collection=True)
        else:
            if not args.job or not all(c.isalnum() or c in '_-' for c in args.job):raise ValueError('Safe job name required')
            fetch(client,args.job)
    finally:client.close()

if __name__=='__main__':main()
