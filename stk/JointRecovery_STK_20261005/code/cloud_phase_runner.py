"""Launch sealed research stages and index explicit completed P0 collections."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import cloud_runner as cloud


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("A safe stage name is required")
    return value


def launch(client, args):
    job = safe_name(args.job)
    script = Path(args.script).name
    if script != args.script or not re.fullmatch(r"[A-Za-z0-9_]+\.py", script):
        raise ValueError("Only a staged Python script may be launched")
    staged = json.loads((cloud.ROOT / "execution/CLOUD_STAGE.json").read_text(encoding="utf-8"))
    capsule = staged["capsule"]
    arguments = json.loads(Path(args.args_json).read_text(encoding="utf-8"))
    if not isinstance(arguments, list) or any(not isinstance(x, str) for x in arguments):
        raise ValueError("Stage arguments must be a JSON array of strings")
    replacements = {"{CAPSULE}": capsule, "{RUNTIME}": capsule + "/runtime",
                    "{OUT}": cloud.REMOTE + "/runs/" + job + "/out",
                    "{REMOTE}": cloud.REMOTE, "{CHILS}": cloud.CHILS,
                    "{CHILS_SOURCE}": cloud.CHILS_SOURCE}
    for index, argument in enumerate(arguments):
        for marker, value in replacements.items():
            argument = argument.replace(marker, value)
        arguments[index] = argument
    command = [cloud.PYTHON, capsule + "/code/" + script] + arguments
    remote_code = '''import os,pathlib,subprocess,time,json
root=pathlib.Path(ROOT); directory=root/"runs"/JOB
assert root==pathlib.Path("/root/autodl-tmp/AAMAS1979_JointRecovery_STK_20261005")
assert pathlib.Path(COMMAND[1]).is_file()
directory.mkdir(parents=True,exist_ok=False)
available=sorted(os.sched_getaffinity(0))
cpus={available[(CPU_INDEX+i)%len(available)] for i in range(CPU_COUNT)}
environment=os.environ.copy()
environment.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONUNBUFFERED="1")
with (directory/"stdout.log").open("wb") as log:
    process=subprocess.Popen(COMMAND,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
       cwd=directory,env=environment,start_new_session=True,close_fds=True,
       preexec_fn=lambda:os.sched_setaffinity(0,cpus))
receipt={"job":JOB,"pid":process.pid,"capsule":CAPSULE,"command":COMMAND,
         "cpu_affinity":sorted(cpus),"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
(directory/"launcher.json").write_text(json.dumps(receipt,indent=2)+"\\n")
print(json.dumps(receipt))
'''
    prefix = "ROOT=" + repr(cloud.REMOTE) + "\nJOB=" + repr(job) + "\nCOMMAND=" + repr(command)
    prefix += "\nCAPSULE=" + repr(capsule) + "\nCPU_INDEX=" + repr(args.cpu_index)
    prefix += "\nCPU_COUNT=" + repr(args.cpu_count) + "\n"
    receipt = cloud.run(client, prefix + remote_code)
    (cloud.ROOT / "execution" / ("LAUNCH_" + job + ".json")).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


def collect(client, args):
    name = safe_name(args.name)
    jobs = [safe_name(job) for job in args.jobs]
    code = '''import pathlib,json,os,hashlib
root=pathlib.Path(ROOT); output=root/"collections"/NAME
assert root==pathlib.Path("/root/autodl-tmp/AAMAS1979_JointRecovery_STK_20261005")
assert not output.exists()
graphs={}; records=[]
for job in JOBS:
    source=root/"runs"/job/"out"
    assert (source/"batch_summary.json").is_file(),"P0 batch incomplete: "+job
    for summary_path in sorted(source.glob("*/summary.json")):
        summary=json.loads(summary_path.read_text())
        assert summary["status"]=="ACTUAL_P0_PROBES_COMPLETE_NO_TRAINING"
        graph_id=summary["graph"]["graph_id"]
        assert graph_id not in graphs,"Duplicate graph in declared collection: "+graph_id
        graphs[graph_id]=summary_path.parent
if not graphs:raise ValueError("Empty completed collection")
output.mkdir(parents=True)
for graph_id,source in sorted(graphs.items()):
    destination=output/graph_id; destination.mkdir()
    for p in sorted(source.iterdir()):
        if p.is_file():
            os.link(p,destination/p.name)
            records.append({"graph_id":graph_id,"name":p.name,"source":str(p),
                            "sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
manifest={"name":NAME,"jobs":JOBS,"graph_count":len(graphs),"files":records,
          "immutable_inputs":True,"selection_by_declared_job_not_outcome":True}
(output/"collection.json").write_text(json.dumps(manifest,indent=2)+"\\n")
print(json.dumps({"name":NAME,"path":str(output),"graphs":len(graphs),"files":len(records)}))
'''
    prefix = "ROOT=" + repr(cloud.REMOTE) + "\nNAME=" + repr(name) + "\nJOBS=" + repr(jobs) + "\n"
    receipt = cloud.run(client, prefix + code)
    (cloud.ROOT / "execution" / ("COLLECTION_" + name + ".json")).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    launcher = sub.add_parser("launch")
    launcher.add_argument("--job", required=True)
    launcher.add_argument("--script", required=True)
    launcher.add_argument("--args-json", required=True)
    launcher.add_argument("--cpu-index", type=int, default=4)
    launcher.add_argument("--cpu-count", type=int, default=1)
    collection = sub.add_parser("collect")
    collection.add_argument("--name", required=True)
    collection.add_argument("--jobs", nargs="+", required=True)
    args = parser.parse_args()
    if args.operation == "launch" and not 1 <= args.cpu_count <= 16:
        parser.error("cpu-count must be between 1 and 16")
    client = cloud.connect()
    try:
        if args.operation == "launch":launch(client,args)
        else:collect(client,args)
    finally:client.close()


if __name__ == "__main__":main()
