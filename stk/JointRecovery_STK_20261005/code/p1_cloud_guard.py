"""One isolated P1 contract check using existing cloud torch; no experiments."""
from pathlib import Path
import json
import cloud_runner as cloud
import p0_recovery_probes as p0


def main():
    code_root = Path(__file__).resolve().parent
    staged = json.loads((cloud.ROOT / "execution/CLOUD_STAGE.json").read_text(encoding="utf-8"))
    names = ("p1_fit_and_allocate.py", "p1_actual_policy.py", "p1_contract_test.py")
    closure = {name: p0.sha_file(code_root / name) for name in names}
    suffix = p0.sha_json(closure)[:12]
    destination = cloud.REMOTE + "/guards/p1_input_units_" + suffix
    client = cloud.connect()
    try:
        with client.open_sftp() as sftp:
            cloud.mkdir(sftp, destination)
            for name in names:
                sftp.put(str(code_root / name), destination + "/" + name)
        prefix = "DIRECTORY=" + repr(destination) + "\nCAPSULE=" + repr(staged["capsule"]) + "\n"
        result = cloud.run(client, prefix + '''import subprocess,os,json,time
environment=os.environ.copy()
environment.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONPATH=CAPSULE+"/code")
started=time.perf_counter()
process=subprocess.run(["/root/miniconda3/bin/python",DIRECTORY+"/p1_contract_test.py","--runtime-root",CAPSULE+"/runtime"],
  capture_output=True,text=True,timeout=120,env=environment)
print(json.dumps({"returncode":process.returncode,"stdout":process.stdout,"stderr":process.stderr,
 "elapsed_seconds":time.perf_counter()-started,"fit_executed":False,"native_executed":False,
 "real_dataset_read":False,"guard_directory":DIRECTORY}))
''')
    finally:
        client.close()
    result["code_sha256"] = closure
    result["runtime_capsule"] = staged["capsule"]
    p0.write_json(cloud.ROOT / "protocol/P1_INPUT_UNIT_GUARD.json", result)
    print(json.dumps(result, ensure_ascii=False))
    if result["returncode"]:
        raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
