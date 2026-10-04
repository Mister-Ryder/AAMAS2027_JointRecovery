"""FixedR256 residual-label replay and CUDA fit-compatibility gate, NOT fitting.

Import/--help only describe this entry. run_gate explicitly requires completed
caller-pinned labels, an immutable source capsule and Py3.8/Torch1.11/CUDA.
It runs 73 existing finite guards, independently replays every declared graph,
then performs at most2 states x5 untrained CUDA forward/loss/backward checks.
There is no optimizer, solver, graph generation, policy trial or confirmation.
run_pre_collection_gate verifies runtime/source/finite guards only; it never
loads labels or invokes a CUDA forward. Graph generation in finite fixtures
means explicit tiny test graphs only, never the declared research generators.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import io
import json
from math import isfinite
from pathlib import Path
import platform
import sys
import time
import traceback
import unittest

import numpy as np
import torch

try:
    from . import v4_residual_fit as fit
except ImportError:
    from experiments import v4_residual_fit as fit


TEST_COUNTS = {"test_v4_residual_fit.py":25,"test_v4_residual_collect_training.py":7,"test_v4_residual_training_data.py":17,
    "test_v4_residual_common.py":10,"test_v4_residual_model.py":14}
COLLECTION_PATHS = ("experiments/v3_domains.py","experiments/v3_pilot.py","experiments/v3_published_baselines.py",
    "experiments/v3_solvers.py","experiments/v4_backends.py","experiments/v4_residual_collect_training.py",
    "experiments/v4_neighborhoods.py","experiments/v4_residual_training_data.py","src/joint_recovery/__init__.py",
    "src/joint_recovery/core.py","src/joint_recovery/efficient_core.py","src/joint_recovery/generators.py",
    "src/joint_recovery/v4_budgeted_recovery.py","src/joint_recovery/v4_factors.py",
    "tests/test_v4_residual_collect_training.py","tests/test_v4_residual_training_data.py")
REQUIRED_PATHS = tuple(sorted(set(COLLECTION_PATHS)|set(fit.FIT_SOURCE_PATHS)|
    {"experiments/v4_residual_fit_gate.py","tests/test_v4_residual_fit.py",
     "tests/test_v4_residual_common.py","tests/test_v4_residual_model.py",
     "src/joint_recovery/v4_residual_controller_coordination.py",
     "src/joint_recovery/v4_controller_coordination.py"}))
CALIBRATION_PATH = "docs/research_v4/CALIBRATION_FROZEN.json"
CAPACITY_VARIANTS = frozenset(("ResidualCapacity", "ResidualNoAux", "ResidualNoWarmMembership"))


def write_json(path,value):
    with Path(path).open("x",encoding="utf-8") as handle:
        handle.write(json.dumps(value,indent=2,allow_nan=False)+"\n")


def runtime_gate():
    version=str(torch.__version__)
    if sys.version_info[:2]!=(3,8) or version.split("+")[0]!="1.11.0" or not torch.cuda.is_available():
        raise RuntimeError("This actual gate requires Python3.8/PyTorch1.11.0/CUDA; no CPU fallback")
    return dict(python=platform.python_version(),torch=version,numpy=np.__version__,cuda_available=True,
        cuda_runtime=torch.version.cuda,device=torch.cuda.get_device_name(0),
        scope="compatibility_not_quality_cost_or_host_exclusivity")


def source_gate(path,expected_sha256):
    manifest=fit.validate_fit_sources(path,expected_sha256)
    paths={r["path"] for r in manifest["files"]}
    if not set(REQUIRED_PATHS)<=paths:
        raise ValueError("Residual capsule must retain collection/common/controller/model/tests/gate closure")
    # The finite queue/fit tests read this exact transport asset. It is not
    # Python source and is not added to the scientific import count.
    root=Path(__file__).resolve().parents[1]
    calibration=root/CALIBRATION_PATH
    if (not calibration.is_file() or calibration.is_symlink() or
            fit.sha256_file(calibration)!=fit.CALIBRATION_SHA256):
        raise ValueError("Exact frozen calibration JSON is missing from the gate capsule")
    # The common guard reads and pins the historical coordination file. Import
    # both controller versions so the actual-origin receipt covers their bytes
    # as well as the files that the finite guards read.
    importlib.import_module("joint_recovery.v4_controller_coordination")
    importlib.import_module("joint_recovery.v4_residual_controller_coordination")
    return manifest


def actual_origins(manifest):
    """Check real scientific modules, excluding external library inventories."""
    root=Path(__file__).resolve().parents[1];declared={r["path"]:r for r in manifest["files"]}
    stems={Path(p).stem for p in REQUIRED_PATHS if p.startswith(("experiments/","tests/"))};origins=[]
    for name,module in tuple(sys.modules.items()):
        file=getattr(module,"__file__",None)
        if not file:continue
        scientific=(name=="__main__" or name.startswith("joint_recovery") or name.split(".")[-1] in stems)
        try:relative=Path(file).resolve().relative_to(root).as_posix()
        except ValueError:
            if scientific:raise ValueError("Scientific import outside actual immutable gate capsule")
            continue
        if not (relative.startswith("experiments/") or relative.startswith("src/joint_recovery/") or relative.startswith("tests/")):
            continue
        if relative.endswith(".py"):
            row=declared.get(relative)
            if row is None or fit.sha256_file(file)!=row["sha256"]:
                raise ValueError("Imported scientific byte source missing from gate capsule")
            origins.append(dict(module=name,path=relative,sha256=row["sha256"]))
    if not set(REQUIRED_PATHS)<={r["path"] for r in origins}:
        raise ValueError("Full gate scientific closure was not actually imported")
    return origins


def finite_guards(out):
    root=Path(__file__).resolve().parents[1];all_suite=unittest.TestSuite();counts={}
    for pattern,expected in TEST_COUNTS.items():
        suite=unittest.TestLoader().discover(str(root/"tests"),pattern=pattern)
        counts[pattern]=suite.countTestCases()
        if counts[pattern]!=expected:raise ValueError("Frozen finite test coverage count differs: "+pattern)
        all_suite.addTests(suite)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(all_suite)
    log=stream.getvalue();(out/"guards.txt").write_text(log,encoding="utf-8")
    record=dict(expected_tests=sum(TEST_COUNTS.values()),tests_run=result.testsRun,by_source=counts,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        unexpected_successes=len(result.unexpectedSuccesses),expected_failures=len(result.expectedFailures),
        log_sha256=fit.sha256_file(out/"guards.txt"),scope="finite_mocks_not_actual_collection_or_training")
    write_json(out/"guards.json",record)
    if (not result.wasSuccessful() or result.testsRun!=sum(TEST_COUNTS.values()) or result.skipped or
            result.expectedFailures or result.unexpectedSuccesses):
        raise RuntimeError("Actual-runtime73 finite guards did not all pass without skips")
    return record


def _gate_files(out,names):
    return {name:dict(sha256=fit.sha256_file(out/name),bytes=(out/name).stat().st_size)
            for name in names}


def run_pre_collection_gate(out,manifest,manifestSHA):
    """Explicit runtime/source/73-mock guard gate before any research graph.

    The caller may bind this completion in its subsequent collection protocol.
    This API never loads a collection, packs a research state, runs a CUDA
    forward/backward, invokes a solver or creates an optimizer.
    """
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        runtime=runtime_gate();source=source_gate(manifest,manifestSHA)
        write_json(out/"protocol.json",dict(status="frozen_residual_pre_collection_source_runtime_gate",
            source_manifest_sha256=manifestSHA,source_capsule=source,runtime=runtime,
            calibration_sha256=fit.CALIBRATION_SHA256,finite_test_counts=TEST_COUNTS,
            labels_loaded=0,research_graphs_generated=0,actual_solver_calls=0,
            cuda_forward_calls=0,cuda_backward_calls=0,optimizer_steps=0))
        guards=finite_guards(out)
        origins=actual_origins(source);write_json(out/"source_origins.json",dict(origins=origins))
        source_gate(manifest,manifestSHA)
        files=_gate_files(out,("protocol.json","guards.txt","guards.json","source_origins.json"))
        result=dict(status="complete_residual_pre_collection_source_runtime_gate_not_quality",files=files,
            source_manifest_sha256=manifestSHA,calibration_sha256=fit.CALIBRATION_SHA256,
            scientific_source_paths=list(REQUIRED_PATHS),tests_run=guards["tests_run"],
            elapsed_seconds=time.perf_counter()-started,labels_loaded=0,
            research_graphs_generated=0,confirmation_graphs_generated=0,actual_solver_calls=0,
            cuda_forward_calls=0,cuda_backward_calls=0,optimizer_steps=0,model_fits_started=0,
            scope="73 finite explicit/mock fixtures + actual runtime/source compatibility; no corpus/quality evidence")
        write_json(out/"completion.json",result)
        return result
    except Exception as error:
        write_json(out/"failure.json",dict(status="failed_residual_pre_collection_source_runtime_gate",
            type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            research_graphs_generated=0,labels_loaded=0,actual_solver_calls=0,
            cuda_forward_calls=0,cuda_backward_calls=0,optimizer_steps=0,no_quality_claim=True))
        raise


def resource_source_replay(case,labels):
    """Saved contact physics/edge identity only; never generate another graph."""
    if case.spec["domain"]!="resource" or not case.states:return None
    meta=labels["source_metadata"];contacts=meta["contacts"];graph=case.graph
    if (meta["kind"]!="synthetic_visibility_window_contacts" or meta["seed"]!=case.spec["seed"] or
            meta["satellites"]!=case.spec["satellites"] or meta["grounds"]!=case.spec["grounds"] or len(contacts)!=graph.n):
        raise ValueError("Resource physics metadata is not the declared original source")
    for v,contact in enumerate(contacts):
        if (contact["satellite"]!=int(graph.agents[v]) or
                contact["start"]<contact["visibility_start"]-1e-8 or
                contact["end"]>contact["visibility_end"]+1e-8):
            raise ValueError("Original resource contacts/owner/visibility differ")
        fit._equal(float(graph.weights[v]),contact["duration"]*contact["priority"],"original duration reward")
    for u,one in enumerate(contacts):
        for v in range(u+1,len(contacts)):
            two=contacts[v];satellite=one["satellite"]==two["satellite"];ground=one["ground"]==two["ground"]
            setup=max(meta["satellite_setup"] if satellite else 0.,meta["ground_setup"] if ground else 0.)
            conflict=(satellite or ground) and not (one["end"]+setup<=two["start"] or two["end"]+setup<=one["start"])
            if conflict!=(v in graph.adjacency[u]):raise ValueError("Original resource edges differ from saved interval physics")
    return dict(saved_contact_count=len(contacts),original_edges_match_saved_physics=True,
        claim="synthetic_interval_resources_not_real_satellite_learning_gain")


def supervision_summary(dataset):
    """All graph slots/rows, graph-first opportunities; never select by gain."""
    domains={domain:dict(graph_slots=0,failed_graph_slots=0,groups_present=0,available_states=0,
        raw_valid_states=0,positive_marginal_states=0,unequal_ranking_states=0,request_rows=0,available_rows=0,
        executed_rows=0,spent_rows=0,valid_membership_returns=0,raw_valid_rows=0,on_time_rows=0,
        late_valid_rows=0,failure_rows=0,not_startable_rows=0,already_spent_rows=0,
        raw_returns_below_actual_warm=0,raw_returns_above_actual_warm=0,
        on_time_returns_above_actual_warm=0,fixed_weight_opportunity_mass=0.) for domain in ("menu","resource")}
    cases=[];root=Path(dataset.root)
    for case in dataset.cases:
        stats=domains[case.spec["domain"]];stats["graph_slots"]+=1
        labels=fit._read_json(root/"cases"/case.case_id/"labels.json")
        if fit.sha256_file(root/"cases"/case.case_id/"labels.json")!=case.file_bindings["labels.json"]["sha256"]:
            raise ValueError("Label bytes changed after independent replay")
        if not case.states:
            stats["failed_graph_slots"]+=1
            cases.append(dict(case_id=case.case_id,spec=case.spec,status=case.status,
                graph_weight=case.graph_weight,supervision_available=False,states=[],
                physical_source_replay=None));continue
        physics=resource_source_replay(case,labels);group_rows=[]
        for group in labels["groups"]:
            state=group["state"];rows=group["alternatives"];stats["groups_present"]+=1
            available=[r for r in rows if r["available"]];raw=[r for r in rows if r["raw_supervision_valid"]]
            marginal=[max(0,r["admitted_gain"]-state["best_gain"]) for r in available]
            maximum=max(marginal,default=0);pairs=sum(a>b for a in marginal for b in marginal)
            extra=[r["recovery_value"]-r["previous_warm_value"] for r in raw]
            timely_extra=[r["recovery_value"]-r["previous_warm_value"] for r in raw if r["on_time"]]
            stats["available_states"]+=bool(available);stats["raw_valid_states"]+=bool(raw)
            stats["positive_marginal_states"]+=maximum>0;stats["unequal_ranking_states"]+=pairs>0
            stats["request_rows"]+=len(rows);stats["available_rows"]+=len(available)
            for row in rows:
                for field,counter in (("launched","executed_rows"),("spent","spent_rows"),
                        ("membership_valid","valid_membership_returns"),("raw_supervision_valid","raw_valid_rows"),("on_time","on_time_rows")):
                    stats[counter]+=bool(row[field])
                stats["late_valid_rows"]+=row["raw_supervision_valid"] and not row["on_time"]
                stats["failure_rows"]+=row["failure"] is not None
                stats["not_startable_rows"]+=row["status"].startswith("not_startable")
                stats["already_spent_rows"]+=row["status"]=="already_spent_in_history"
            stats["raw_returns_below_actual_warm"]+=sum(v<0 for v in extra)
            stats["raw_returns_above_actual_warm"]+=sum(v>0 for v in extra)
            stats["on_time_returns_above_actual_warm"]+=sum(v>0 for v in timely_extra)
            relative=float(maximum/max(1.,state["initial_value"]))
            stats["fixed_weight_opportunity_mass"]+=case.graph_weight*.5*relative
            group_rows.append(dict(state_name=state["name"],state_sha256=group["state_sha256"],
                state_weight=.5,available_requests=len(available),raw_supervised_requests=len(raw),
                maximum_raw_recovery_above_actual_warm=max(extra,default=None),
                maximum_on_time_recovery_above_actual_warm=max(timely_extra,default=None),
                positive_admitted_marginal_opportunity_native=maximum,
                relative_conditional_opportunity=relative,unequal_admitted_marginal_pairs=pairs))
        cases.append(dict(case_id=case.case_id,spec=case.spec,status=case.status,graph_weight=case.graph_weight,
            supervision_available=True,states=group_rows,physical_source_replay=physics))
    for stats in domains.values():stats["expected_half_states"]=2*stats["graph_slots"]
    return dict(split=dataset.split,domains=domains,cases=cases,graph_count=len(dataset.cases),
        equal_domain_fixed_weight_label_opportunity_mass=sum(d["fixed_weight_opportunity_mass"] for d in domains.values()),
        scope="conditional_executor_label_opportunity_not_learned_or_full_policy_regret",
        missing_supervision_is_not_observed_zero_regret=True,all_slots_retained=True)


def first_observable_states(training):
    chosen=[];missing=[]
    for domain in ("menu","resource"):
        found=None
        for case in training.cases:
            if case.spec["domain"]!=domain:continue
            for index,state in enumerate(case.states):
                if state.views:found=(case,index,state);break
            if found is not None:break
        if found is None:missing.append(domain)
        else:chosen.append(found)
    return tuple(chosen),tuple(missing)


def parameter_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        cpu=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(cpu.dtype).encode());digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def model_detail_checks(batch,details,model,variant):
    """Fractional representation checks, not an integer recovery certificate."""
    from joint_recovery import v4_residual_model as residual
    lower,upper=batch["lower"],batch["upper"]
    recovery=details["recovery"]
    if (not bool(torch.isfinite(recovery).all()) or
            bool((recovery<lower-1e-6).any()) or bool((recovery>upper+1e-6).any())):
        raise ValueError("Residual finite recovery prediction is outside its checked L/U interval")
    maximum_load=None;occupancy=details.get("occupancy")
    if occupancy is not None:
        if not bool(((occupancy>=0)&(occupancy<=1)).all()):
            raise ValueError("Fractional occupancy outside [0,1]")
        nodes,factors=batch["incidence"]
        loads=occupancy.new_zeros(batch["factor_count"]).index_add(0,factors,occupancy[nodes])
        maximum_load=float(loads.max().item()) if len(loads) else 0.
        if variant in CAPACITY_VARIANTS and maximum_load>1.+1e-5:
            raise ValueError("Projected fractional capacity load exceeds1")
        # The matched free-occupancy control intentionally permits loads>1.
        # Record its load without certifying it as projected or feasible.
    if variant=="ResidualNoWarmMembership":
        units=batch["units"]
        if (model.use_warm_membership is not False or units is None or
                bool(units["x"][:,3].any()) or bool(units["warm"].any()) or
                bool(batch["context"][:,residual.WARM_FRACTION_INDEX].any())):
            raise ValueError("NoWarmMembership leaked warm IDs/fraction")
        if not torch.equal(batch["context"][:,residual.WARM_VALUE_INDEX],lower):
            raise ValueError("NoWarmMembership removed the common semantic lower bound L")
    return dict(maximum_fractional_factor_load=maximum_load,
        projected_capacity_checked=variant in CAPACITY_VARIANTS,
        free_occupancy_load_is_diagnostic=variant=="ResidualFreeOccupancy",
        bound_interval_checked=True,
        no_warm_membership_retains_actual_L=variant=="ResidualNoWarmMembership")


def cuda_state_checks(training):
    """First available input order only; at most10 backward calls, zero steps."""
    chosen,missing=first_observable_states(training);records=[]
    for case,index,state in chosen:
        for variant in fit.VARIANTS:
            config=fit.FitConfig(variant,17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
            model=fit.build_model(config).to("cuda");torch.cuda.synchronize()
            before=parameter_digest(model)
            # Physically remove future labels from the objects passed to the
            # packer. Known views/context remain actual prior observations.
            observable_case=replace(case,states=())
            observable_state=replace(state,rows=())
            batch=fit.pack_state(observable_case,observable_state,model,config,"cuda")
            details=model(batch,return_details=True)
            targets=fit.state_targets(case,state,batch,"cuda")
            loss=fit.state_loss(details,targets,batch,auxiliary_weight=config.effective_auxiliary_weight,
                ranking_weight=config.ranking_weight,temperature=config.temperature)
            if any(not bool(torch.isfinite(value)) for key,value in loss.items() if torch.is_tensor(value)):
                raise ValueError("Non-finite actual CUDA state loss")
            for key,value in details.items():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise ValueError("Non-finite actual CUDA model detail: "+key)
            representation=model_detail_checks(batch,details,model,variant)
            loss["total"].backward();torch.cuda.synchronize()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not grads or any(not bool(torch.isfinite(value).all()) for value in grads):
                raise ValueError("Actual CUDA backward has missing/non-finite gradient tensors")
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError("Non-finite actual CUDA parameters")
            after=parameter_digest(model)
            if after!=before:raise ValueError("Compatibility gate changed parameters without authorization")
            records.append(dict(case_id=case.case_id,domain=case.spec["domain"],group_index=index,
                state_name=state.record["name"],state_sha256=state.state_sha256,variant=variant,untrained_seed=17,
                available_requests=len(state.views),raw_supervised_requests=sum(r["raw_supervision_valid"] for r in state.rows),
                request_local_nodes=batch.get("node_count",int(batch["ptr"][-1].item())),
                loss={key:float(value.detach().cpu().item()) for key,value in loss.items() if torch.is_tensor(value)},
                finite_gradient_tensors=len(grads),nonzero_gradient_tensors=sum(bool(g.any()) for g in grads),
                absent_gradient_tensors=sum(p.grad is None for p in model.parameters()),
                missing_gradients_scope="inactive_heads_or_unavailable_aux_labels; not a gradient-collapse conclusion",
                representation_checks=representation,
                capacity_scope="capacity variants only; free occupancy load is diagnostic; no integer schedule or deadline guarantee",
                teacher_separation="labels stripped before pack/forward; returned targets introduced only after forward",
                parameter_sha256_before=before,parameter_sha256_after=after,optimizer_steps=0))
            del loss,targets,details,batch,model
    return dict(selection="training registry first available menu state then first available resource state; no outcome criterion",
        selected_states=[dict(case_id=c.case_id,domain=c.spec["domain"],group_index=i,state_sha256=s.state_sha256) for c,i,s in chosen],
        missing_available_domains=missing,checks=records,maximum_states=2,maximum_backward_calls=10,
        optimizer_steps=0,no_quality_claim=True)


def run_gate(collection_root,out,*,expected_protocol_sha256,expected_completion_sha256,
        source_manifest_path,expected_source_manifest_sha256):
    collection_root=Path(collection_root).resolve();out=Path(out).resolve()
    if out==collection_root or collection_root in out.parents:
        raise ValueError("Gate scratch output must not mutate the sealed collection")
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    try:
        runtime=runtime_gate();manifest=source_gate(source_manifest_path,expected_source_manifest_sha256)
        write_json(out/"protocol.json",dict(status="frozen_residual_label_fit_gate_not_training",
            collection_protocol_sha256=expected_protocol_sha256,collection_completion_sha256=expected_completion_sha256,
            calibration_sha256=fit.CALIBRATION_SHA256,source_manifest_sha256=expected_source_manifest_sha256,
            source_capsule=manifest,runtime=runtime,finite_test_counts=TEST_COUNTS,
            state_selection="first available training menu/resource, at most2; no gain selection",
            maximum_untrained_backward_calls=10,optimizer_steps=0,confirmation_graphs_generated=0))
        guards=finite_guards(out)
        training=fit.load_dataset(collection_root,"training",expected_protocol_sha256=expected_protocol_sha256,
            expected_completion_sha256=expected_completion_sha256)
        validation=fit.load_dataset(collection_root,"validation",expected_protocol_sha256=expected_protocol_sha256,
            expected_completion_sha256=expected_completion_sha256)
        supervision=dict(training=supervision_summary(training),validation=supervision_summary(validation),
            all96_declared_slots_replayed=True,graph_split_before_states_queries=True)
        write_json(out/"replay.json",supervision)
        cuda=cuda_state_checks(training);write_json(out/"cuda.json",cuda)
        origins=actual_origins(manifest);write_json(out/"source_origins.json",dict(origins=origins))
        source_gate(source_manifest_path,expected_source_manifest_sha256)
        if (fit.sha256_file(collection_root/"protocol.json")!=expected_protocol_sha256 or
                fit.sha256_file(collection_root/"completion.json")!=expected_completion_sha256):
            raise ValueError("Sealed collection receipts changed during gate")
        names=("protocol.json","guards.txt","guards.json","replay.json","cuda.json","source_origins.json")
        files={name:dict(sha256=fit.sha256_file(out/name),bytes=(out/name).stat().st_size) for name in names}
        result=dict(status="complete_residual_label_fit_compatibility_gate_not_quality",files=files,
            collection_protocol_sha256=expected_protocol_sha256,collection_completion_sha256=expected_completion_sha256,
            source_manifest_sha256=expected_source_manifest_sha256,tests_run=guards["tests_run"],
            graph_slots_replayed=96,cuda_backward_calls=len(cuda["checks"]),missing_available_domains=cuda["missing_available_domains"],
            elapsed_seconds=time.perf_counter()-started,model_fits_started=0,optimizer_steps=0,
            actual_solver_calls=0,fresh_graphs_generated=0,confirmation_graphs_generated=0,
            training_disposition="compatibility_only; parent must inspect full supervision/opportunity before authorizing fit",
            learning_advantage_established=False,no_online_policy_or_speed_claim=True)
        write_json(out/"completion.json",result);print(json.dumps(result),flush=True)
        return result
    except Exception as error:
        failure=dict(status="failed_residual_label_fit_compatibility_gate",type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),model_fits_started=0,optimizer_steps=0,no_quality_claim=True)
        write_json(out/"failure.json",failure)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--protocol-sha256",required=True);parser.add_argument("--completion-sha256",required=True)
    parser.add_argument("--source-manifest",required=True);parser.add_argument("--source-manifest-sha256",required=True)
    args=parser.parse_args()
    run_gate(args.collection_root,args.out,expected_protocol_sha256=args.protocol_sha256,
        expected_completion_sha256=args.completion_sha256,source_manifest_path=args.source_manifest,
        expected_source_manifest_sha256=args.source_manifest_sha256)


if __name__=="__main__":main()
