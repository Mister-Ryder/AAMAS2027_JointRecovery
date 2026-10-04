"""Frozen fresh conditional labels, not fitting or on-time policy evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

try:
    from . import v3_domains as _fresh_domains
    from .v4_residual_training_data import (fresh_split_specs,construct_fresh_case,split_graph_weight,
        collect_label_groups,CollectionConfig,graph_identity)
    from .v4_backends import chils_backend
except ImportError:
    from experiments import v3_domains as _fresh_domains
    from experiments.v4_residual_training_data import (fresh_split_specs,construct_fresh_case,split_graph_weight,
        collect_label_groups,CollectionConfig,graph_identity)
    from experiments.v4_backends import chils_backend
from joint_recovery.v4_budgeted_recovery import Workpoint

CALIBRATION_SHA256="495dab54ad7ca75e8a543338eb71da1049e20afdd35ad18825669d37d54c5bec"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def validated_calibration(path):
    if sha(path)!=CALIBRATION_SHA256:raise ValueError("Calibration source bytes differ from the frozen receipt")
    calibration=json.loads(Path(path).read_text())
    if calibration["status"]!="frozen_before_V4_fitting_and_fresh_validation":raise ValueError("Frozen calibration required")
    if calibration["total_deadlines_seconds"]!=[.139,.556,2.221] or calibration["caps"]!=[64,256,1024]:
        raise ValueError("Frozen primary deadline/cap grid changed")
    if calibration["primary_deadline_seconds"]!=.556:raise ValueError("Frozen primary deadline changed")
    points=tuple(Workpoint(**{k:p[k] for k in ("name","kind","amount","expected_seconds")})
                 for p in calibration["workpoints"])
    if [p.expected_seconds for p in points]!=[.035,.075,.214]:raise ValueError("Shared cost calibration changed")
    return calibration,CollectionConfig(calibration["primary_deadline_seconds"],points,sha(path),1)


def wait_for_runtime(path):
    gate=Path(path)
    while not gate.exists():
        if (gate.parent/"failure.json").exists():raise ValueError("Previous isolated runtime failed")
        time.sleep(10.)
    if json.loads(gate.read_text())["status"]!="complete_factorized_runtime_gate_not_quality":
        raise ValueError("The factorized native runtime gate did not complete")


def imported_source_origins(root,manifest,required):
    by_path={r["path"]:r["sha256"] for r in manifest["files"]}
    stems={Path(p).stem for p in required if p.startswith("experiments/")}
    origins=[]
    for name,module in tuple(sys.modules.items()):
        source=getattr(module,"__file__",None)
        if not source:continue
        scientific=(name=="__main__" or name.startswith("joint_recovery") or name.split(".")[-1] in stems)
        try:relative=Path(source).resolve().relative_to(root).as_posix()
        except ValueError:
            if scientific:raise ValueError("Scientific module imported outside the immutable capsule")
            continue
        if relative.endswith(".py"):
            actual=sha(source)
            if by_path.get(relative)!=actual:raise ValueError("Imported source outside the verified capsule")
            origins.append(dict(module=name,path=relative,sha256=actual))
    if not required<={r["path"] for r in origins}:raise ValueError("Required sources not actually imported")
    return origins


def save_observable(path,graph,selected,cliques=()):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    edges=np.asarray(sorted((u,v) for u in range(graph.n) for v in graph.adjacency[u] if u<v),dtype=np.int64).reshape(-1,2)
    offsets=np.cumsum([0]+[len(c) for c in cliques],dtype=np.int64)
    members=np.asarray([v for c in cliques for v in c],dtype=np.int64)
    np.savez_compressed(path,weights=graph.weights,agents=graph.agents,edges=edges,
        selected=np.asarray(sorted(selected),dtype=np.int64),cliques_flat=members,cliques_ptr=offsets)


def collect_case(spec,case_id,out,config,backend):
    folder=out/"cases"/case_id;folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/"spec.json",spec);started=time.perf_counter()
    graph_id=None
    try:
        graph,selected,metadata=construct_fresh_case(spec);graph_id=graph_identity(graph)
        dataset_setup_seconds=time.perf_counter()-started
        save_observable(folder/"observable.npz",graph,selected)
        labels=collect_label_groups(graph,selected,seed=spec["seed"]+1000,config=config,
            backend=backend,graph_weight=split_graph_weight(spec))
        if labels["graph_sha256"]!=graph_id:raise ValueError("Collection graph identity changed")
        labels.update(spec=spec,source_metadata=metadata,graph_construction_seconds=dataset_setup_seconds,
                      dataset_construction_outside_decision_timer=True)
    except Exception as error:
        labels=dict(status="fresh_declared_graph_collection_failed",spec=spec,graph_sha256=graph_id,
            groups=[],graph_loss_weight=split_graph_weight(spec),included_in_coverage=True,
            failure=dict(type=type(error).__name__,message=str(error)),confirmation_graphs_generated=0,
            incomplete_case_labels_not_certified=True,learning_advantage_established=False)
    write_json(folder/"labels.json",labels)
    files={name:dict(bytes=(folder/name).stat().st_size,sha256=sha(folder/name))
           for name in ("spec.json","observable.npz","labels.json") if (folder/name).exists()}
    return dict(case_id=case_id,spec=spec,status=labels["status"],graph_sha256=graph_id,files=files,
        actual_history_calls=labels.get("history_actual_calls",0),
        actual_alternative_calls=labels.get("alternative_actual_calls",0),
        total_collection_seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True);parser.add_argument("--calibration",required=True)
    parser.add_argument("--splits",nargs="+",choices=("training","validation"),default=["training","validation"])
    parser.add_argument("--chils",default="/root/autodl-tmp/AAMAS1979_JointRecovery_v01/tools/CHILS-v3/CHILS")
    parser.add_argument("--wait-for-completion")
    args=parser.parse_args();out=Path(args.out).resolve()
    if out.exists():raise FileExistsError("A fresh label output is required")
    if len(set(args.splits))!=len(args.splits):raise ValueError("Distinct complete split names required")
    if args.wait_for_completion:wait_for_runtime(args.wait_for_completion)
    calibration,config=validated_calibration(args.calibration)
    root=Path(__file__).resolve().parents[1];manifest=json.loads((root/"source_manifest.json").read_text())
    required={"experiments/v4_residual_collect_training.py","experiments/v4_residual_training_data.py","experiments/v4_neighborhoods.py",
        "experiments/v4_backends.py","experiments/v3_domains.py","experiments/v3_pilot.py",
        "experiments/v3_solvers.py","experiments/v3_published_baselines.py","src/joint_recovery/__init__.py",
        "src/joint_recovery/core.py","src/joint_recovery/efficient_core.py","src/joint_recovery/generators.py",
        "src/joint_recovery/v4_budgeted_recovery.py","src/joint_recovery/v4_factors.py",
        "experiments/v4_residual_common.py"}
    names={r["path"] for r in manifest["files"]}
    if not required<=names or len(names)!=len(manifest["files"]):raise ValueError("Complete scientific source closure required")
    for record in manifest["files"]:
        if sha(root/record["path"])!=record["sha256"]:raise ValueError("Immutable source capsule changed")
    origins=imported_source_origins(root,manifest,required)
    # A version-specific, actual-runtime finite barrier precedes every declared
    # graph. This is not a solver/learning result or a paid policy warmup.
    from experiments import v4_residual_fit_gate as gate
    pre_collection_gate=gate.run_pre_collection_gate(out/'pre_collection_gate',
        root/'source_manifest.json',sha(root/'source_manifest.json'))
    pre_collection_binding=dict(directory='pre_collection_gate',
        completion_sha256=sha(out/'pre_collection_gate'/'completion.json'),
        status=pre_collection_gate['status'],tests_run=pre_collection_gate['tests_run'],
        files=pre_collection_gate['files'],source_manifest_sha256=pre_collection_gate['source_manifest_sha256'])
    # The warm executor is verified before any fresh graph or outcome exists.
    backend=chils_backend(args.chils)
    specs=[spec for split in args.splits for spec in fresh_split_specs(split)]
    write_json(out/"protocol.json",dict(status="frozen_actual_fresh_label_queue",source_capsule=manifest,
        complete_specs=specs,splits=args.splits,calibration=calibration,calibration_sha256=sha(args.calibration),
        execution_origins=origins,pre_collection_gate=pre_collection_binding,
        labels_schema="case/spec.json+labels.json+observable.npz(weights,agents,edges,selected,cliques_flat,cliques_ptr)",
        state_weight=.5,executed_common_greedy_prefix=True,fixed_scope_cap=256,graph_weight="equal domains then equal graphs within domain",
        backend_reset_each_alternative=True,peer_outcomes_never_update_peers=True,
        serial_single_native_thread=True,model_fits_started=0,confirmation_graphs_generated=0,
        no_policy_or_learning_advantage_claim=True,
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform())))
    started=time.perf_counter();records=[]
    for index,spec in enumerate(specs):
        record=collect_case(spec,"case_%03d"%index,out,config,backend);records.append(record)
        write_json(out/"progress.json",dict(expected=len(specs),completed=len(records),records=records))
        print(json.dumps({k:v for k,v in record.items() if k!="files"}),flush=True)
    write_json(out/"completion.json",dict(status="complete_fresh_label_coverage_not_fitting",cases=records,
        expected_cases=len(specs),protocol_sha256=sha(out/"protocol.json"),
        elapsed_seconds=time.perf_counter()-started,model_fits_started=0,confirmation_graphs_generated=0,
        includes_failed_and_no_startable_cases=True))


if __name__=="__main__":main()
