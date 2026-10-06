"""Explicit P1 fit/fixed-call replay, only after a signed P0 opportunity gate.

Importing the module fits nothing. Exactly Capacity/CheapSummary h32-L2 and
seeds 17/29 are implemented. Public or P2 test source rows are never fit inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np

import p0_recovery_probes as p0

SCHEMA = "joint_recovery_stk_p1_v1"
VARIANTS = ("ResidualCapacity", "ResidualCheapSummary")
FIT_SEEDS = (17, 29)
FIT_SOURCES = tuple(range(4))
DEVELOPMENT_SOURCES = (4, 5)
FIXED_CALLS = (1, 2, 4)
MODEL_FILES = ("src/joint_recovery/v4_residual_model.py", "src/joint_recovery/v4_factorized_model.py",
               "src/joint_recovery/v4_model.py", "src/joint_recovery/v4_model_fast.py",
               "src/joint_recovery/v4_factors_fast.py")


def source_number(name):
    match = re.search(r"r(\d{3})(?:$|[^0-9])", str(name))
    if not match:
        raise ValueError("Explicit physical source r000...r011 required: " + str(name))
    return int(match.group(1))


def load_model_api(runtime):
    import torch
    from joint_recovery.v4_budgeted_recovery import ControllerState, Request, RequestView, Workpoint
    from joint_recovery.v4_residual_model import (
        build_residual_model, pack_residual_v4, pack_residual_summary)
    from joint_recovery.v4_model_fast import FastStaticPackingCache
    from experiments.v4_neighborhoods import CoordinationAction
    return dict(torch=torch, ControllerState=ControllerState, Request=Request, RequestView=RequestView,
                Workpoint=Workpoint, Action=CoordinationAction, build=build_residual_model,
                pack=pack_residual_v4, summary_pack=pack_residual_summary, Cache=FastStaticPackingCache,
                model_source_sha256={key: p0.sha_file(Path(runtime) / key) for key in MODEL_FILES})


@dataclass
class Example:
    key: str
    source: int
    graph_id: str
    graph: object
    group: dict
    scopes: dict
    views: tuple
    rows: tuple
    context: object
    binding: dict


def graph_paths(root):
    result = {}
    for path in sorted(Path(root).resolve().rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            graph_id = str(p0.scalar(data, "graph_id", path.stem))
        if graph_id in result:
            raise ValueError("Duplicate graph_id in graph inputs: " + graph_id)
        result[graph_id] = path
    return result


def rebuild_example(graph, group, metadata, binding, api, model_api):
    snapshot = group["snapshot"]
    original = frozenset(snapshot["original"])
    best = frozenset(snapshot["best_selected"])
    if not api["feasible"](graph, original) or not api["feasible"](graph, best):
        raise ValueError("P1 snapshot original/best is infeasible")
    original_value = api["objective"](graph, original)
    best_gain = max(0.0, api["objective"](graph, best) - original_value)
    if not math.isclose(best_gain, snapshot["best_gain_seconds"], rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError("P1 snapshot g* differs from actual complete membership")
    cache = api["Cache"](graph, original, ())
    scopes = {}
    for descriptor in group["scopes"]:
        index = descriptor["action_index"]
        action = model_api["Action"](tuple(descriptor["inserts"]), tuple(descriptor["releases"]))
        scope = cache.scope(index, action, 256)
        for field in ("base", "displaced", "replacements"):
            if tuple(sorted(getattr(scope, field))) != tuple(sorted(descriptor[field])):
                raise ValueError("P1 exact observable scope replay differs: " + field)
        if tuple(scope.replacements) != tuple(descriptor["replacements"]):
            raise ValueError("P1 ordered R differs from native execution input")
        if not math.isclose(scope.immediate_gain, descriptor["q_seconds"], rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError("P1 known q differs from original rewards")
        scopes[index] = scope
    repeats = {}
    for row in group["requests"]:
        if not row.get("available_for_allocation", True):
            continue
        key = (row["action_index"], row["workpoint_ms"])
        scope = scopes[key[0]]
        actual_warm = frozenset(snapshot["warm_by_action"][str(key[0])])
        if actual_warm != frozenset(row["actual_warm_members"]):
            raise ValueError("A P1 label is not conditioned on the saved actual warm")
        returned = api["check"](graph, scope, row["returned_recovery_members"])
        signed = scope.immediate_gain + api["objective"](graph, returned)
        if not math.isclose(signed, row["q_plus_recovery_seconds"], rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError("Returned label q+w(T) differs from original membership rescore")
        if frozenset(row["complete_feasible_members"]) != scope.base | returned:
            raise ValueError("P1 full returned membership differs from B union T")
        repeats.setdefault(key, []).append(row)
    rows, views = [], []
    for (index, budget), actual_rows in sorted(repeats.items()):
        scope = scopes[index]
        warm = frozenset(snapshot["warm_by_action"][str(index)])
        warm = api["check"](graph, scope, warm)
        point = model_api["Workpoint"]("native-%dms" % budget, "seconds", budget / 1000., max(.001, budget / 1000.))
        views.append(model_api["RequestView"](model_api["Request"](scope, point), warm))
        gains = [scope.immediate_gain + api["objective"](graph, r["returned_recovery_members"]) for r in actual_rows]
        memberships = [frozenset(r["returned_recovery_members"]) for r in actual_rows]
        rows.append(dict(action_index=index, workpoint_ms=budget, signed_gain_seconds=float(np.mean(gains)),
                         mean_net_marginal_seconds=float(np.mean([max(0.0, g - best_gain) for g in gains])),
                         recovery_value_seconds=float(np.mean([api["objective"](graph, m) for m in memberships])),
                         occupancy_mean=[float(np.mean([v in membership for membership in memberships]))
                                         for v in scope.replacements],
                         repeats=len(actual_rows), failed_repeats=sum(r["failed"] for r in actual_rows),
                         verified_fallback_is_observed_executor_outcome=True,
                         complete_memberships=[sorted(scope.base | m) for m in memberships]))
    # Remaining-time conditioning has no P0 supervision. Freeze this input at
    # zero for fit AND deployment; real time only controls the outer allocator.
    context = model_api["ControllerState"](original_value, best_gain, 0., tuple(
        (int(a), int(cap), "native-%dms" % int(b)) for a, cap, b in snapshot["spent_requests"]))
    key = metadata["graph_id"] + ":" + group["snapshot_id"]
    return Example(key, source_number(metadata["source_group"]), metadata["graph_id"], graph,
                   group, scopes, tuple(views), tuple(rows), context, binding)


def load_examples(graphs_root, p0_root, api, model_api, allowed_sources):
    paths = graph_paths(graphs_root)
    examples, seen_graphs = [], set()
    for summary_path in sorted(Path(p0_root).resolve().rglob("summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "ACTUAL_P0_PROBES_COMPLETE_NO_TRAINING":
            continue
        graph_id = summary["graph"]["graph_id"]
        source = source_number(summary["graph"]["source_group"])
        if source not in allowed_sources:
            continue
        if graph_id in seen_graphs:
            raise ValueError("Repeated P0 graph output; supply one sealed collection root")
        seen_graphs.add(graph_id)
        graph, metadata = p0.load_graph(paths[graph_id], api)
        if metadata["npz_sha256"] != summary["graph"]["npz_sha256"]:
            raise ValueError("P0 graph hash differs from current new STK graph")
        for group_path in sorted(summary_path.parent.glob("snapshot_*.json")):
            group = json.loads(group_path.read_text(encoding="utf-8"))
            binding = dict(graph_npz=str(paths[graph_id]), graph_npz_sha256=metadata["npz_sha256"],
                           snapshot_file=str(group_path), snapshot_sha256=p0.sha_file(group_path),
                           p0_summary_sha256=p0.sha_file(summary_path))
            examples.append(rebuild_example(graph, group, metadata, binding, api, model_api))
    return examples


def require_training_sources(examples):
    for source in FIT_SOURCES + DEVELOPMENT_SOURCES:
        graph_ids = sorted({e.graph_id for e in examples if e.source == source})
        if len(graph_ids) != 6:
            raise ValueError("P1 requires all six R8/R12 x 170/340/680 graphs for r%03d, got %d" % (source, len(graph_ids)))
    if any(e.source not in FIT_SOURCES + DEVELOPMENT_SOURCES for e in examples):
        raise ValueError("P2 confirmation sources cannot enter fit/development")


def freeze_scale(examples):
    values, records = [], []
    for source in FIT_SOURCES:
        candidates = sorted((e for e in examples if e.source == source), key=lambda e: e.graph_id)
        chosen = next(e for e in candidates if "R12-g0170" in e.graph_id)
        median = float(np.median(chosen.graph.weights))
        values.append(median)
        records.append(dict(source="r%03d" % source, graph_id=chosen.graph_id,
                            full_contact_duration_median_seconds=median, graph_sha256=chosen.binding["graph_npz_sha256"]))
    scale = max(1., float(np.median(values)))
    return dict(scale_seconds=scale, rule="median of per-fit-source R12-g0170 full-contact-duration medians",
                fit_sources=records, uses_development_or_validation_or_test_labels=False,
                q_L_U_features_and_signed_targets_divided_by_same_scale=True,
                predictions_multiplied_back_to_original_seconds=True,
                remaining_time_head_feature_frozen_zero_for_no_unavailable_supervision=True)


def observable_batch(example, variant, scale, device, model_api):
    if not example.views:
        return None
    summary = variant == "ResidualCheapSummary"
    cache = model_api["Cache"](summary_only=summary)
    packer = model_api["summary_pack"] if summary else model_api["pack"]
    return packer(example.graph, example.views, example.context, scale=scale, device=device, static_cache=cache)


def targets(example, scale, device, model_api):
    torch = model_api["torch"]
    return dict(raw=torch.tensor([r["signed_gain_seconds"] / scale for r in example.rows], dtype=torch.float32, device=device),
                marginal=torch.tensor([r["mean_net_marginal_seconds"] / scale for r in example.rows], dtype=torch.float32, device=device),
                occupancy=torch.tensor([v for r in example.rows for v in r["occupancy_mean"]], dtype=torch.float32, device=device))


def state_loss(details, target, batch, graph_model, model_api):
    torch = model_api["torch"]
    F = torch.nn.functional
    raw = details["raw_gain"]
    zero = raw.sum() * 0.
    regression = F.smooth_l1_loss(raw, target["raw"])
    desired = target["marginal"][:, None] > target["marginal"][None, :] + 1e-6
    ranking = F.softplus(-(raw[:, None] - raw[None, :])[desired] / .2).mean() if bool(desired.any()) else zero
    auxiliary = zero
    if graph_model and len(target["occupancy"]):
        assignment = batch["node_request"]
        node_loss = F.binary_cross_entropy_with_logits(details["logits"], target["occupancy"], reduction="none")
        sums = raw.new_zeros(len(raw)).index_add(0, assignment, node_loss)
        counts = raw.new_zeros(len(raw)).index_add(0, assignment, torch.ones_like(node_loss))
        covered = counts > 0
        if bool(covered.any()):
            auxiliary = (sums[covered] / counts[covered]).mean()
    return dict(total=regression + ranking + .05 * auxiliary,
                regression=regression, ranking=ranking, auxiliary=auxiliary,
                unequal_pairs=int(desired.sum().item()))


def tensor_tree_to(value, device):
    if hasattr(value, "to") and hasattr(value, "dtype"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: tensor_tree_to(v, device) for key, v in value.items()}
    return value


def seed_all(seed, model_api):
    random.seed(seed)
    np.random.seed(seed)
    torch = model_api["torch"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fixed_call_outcomes(example, predictions, policy):
    predictions = np.asarray(predictions, dtype=float)
    if len(predictions) != len(example.rows) or not np.isfinite(predictions).all():
        raise ValueError("Finite prediction per unique available request required")
    base = example.context.best_gain
    actual = np.asarray([row["signed_gain_seconds"] for row in example.rows])
    result = []
    menus = [("pooled_workpoints", tuple(range(len(example.rows))))]
    menus += [("native_%dms" % budget, tuple(i for i, row in enumerate(example.rows) if row["workpoint_ms"] == budget))
              for budget in sorted({row["workpoint_ms"] for row in example.rows})]
    for menu, indices in menus:
        order = sorted(indices, key=lambda i: (-predictions[i], example.rows[i]["action_index"],
                                              -example.rows[i]["workpoint_ms"]))
        oracle = max(0., max([base] + [float(actual[i]) for i in indices]) - base)
        for calls in FIXED_CALLS:
            selected = order[:calls]
            gain = max([base] + [float(actual[index]) for index in selected])
            result.append(dict(example=example.key, graph_id=example.graph_id, source="r%03d" % example.source,
                           request_menu=menu,
                           policy=policy, fixed_call_limit=calls, actually_selected_unique_requests=len(selected),
                           selected_keys=[[example.rows[i]["action_index"], example.rows[i]["workpoint_ms"]] for i in selected],
                           snapshot_best_gain_seconds=base, selected_gain_seconds=gain,
                           beyond_snapshot_gain_seconds=max(0., gain - base), oracle_single_request_opportunity_seconds=oracle,
                           conditional_regret_seconds=oracle - max(0., gain - base),
                           observed_clone_repeats_averaged=True,
                           scope="fixed_call_cold_clone_label_selection_not_online_or_timing_evidence"))
    return result


def predict(model, example, variant, scale, device, model_api):
    batch = observable_batch(example, variant, scale, device, model_api)
    if batch is None:
        return np.zeros(0), None
    torch = model_api["torch"]
    with torch.no_grad():
        details = model(batch, return_details=True)
        prediction = (details["raw_gain"] * scale).detach().cpu().numpy()
    return prediction, batch


def evaluate_model(model, examples, variant, scale, device, model_api):
    losses, records = [], []
    model.eval()
    torch = model_api["torch"]
    with torch.no_grad():
        for example in examples:
            batch = observable_batch(example, variant, scale, device, model_api)
            if batch is None:
                continue
            details = model(batch, return_details=True)
            components = state_loss(details, targets(example, scale, device, model_api), batch,
                                    variant == "ResidualCapacity", model_api)
            losses.append(float(components["total"].cpu()))
            raw = (details["raw_gain"] * scale).detach().cpu().numpy()
            records.extend(fixed_call_outcomes(example, raw, variant))
    return dict(mean_state_loss=float(np.mean(losses)) if losses else None, fixed_call_rows=records)


def gradient_distribution(model):
    norms, fractions, names = [], [], {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            names[name] = dict(gradient_present=False)
            continue
        grad = parameter.grad.detach()
        norm = float(grad.norm().cpu())
        fraction = float((grad != 0).float().mean().cpu())
        names[name] = dict(gradient_present=True, norm=norm, nonzero_fraction=fraction)
        norms.append(norm)
        fractions.append(fraction)
    return dict(global_norm=math.sqrt(math.fsum(n * n for n in norms)),
                parameter_gradient_norm_quantiles=np.quantile(norms, (0., .25, .5, .75, 1.)).tolist() if norms else [],
                mean_nonzero_gradient_fraction=float(np.mean(fractions)) if fractions else 0., by_parameter=names)


def require_gate(path, examples):
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    if gate.get("allow_p1_execution") is not True or not gate.get("reason"):
        raise ValueError("Explicit root P0 opportunity decision required before fitting")
    positives = sum(any(r["mean_net_marginal_seconds"] > 1e-7 for r in e.rows) for e in examples
                    if e.source in (0, 1))
    if positives == 0:
        raise ValueError("No observed P0 r000/r001 net opportunity; do not train blindly")
    return dict(gate_file_sha256=p0.sha_file(path), gate=gate,
                r000_r001_observed_positive_states=positives,
                negative_and_zero_states_retained=True)


def fit(args, api, model_api):
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Fit requires a fresh output directory")
    examples = load_examples(args.graphs_dir, args.p0_root, api, model_api, FIT_SOURCES + DEVELOPMENT_SOURCES)
    require_training_sources(examples)
    gate = require_gate(args.p0_gate, examples)
    scale_receipt = freeze_scale(examples)
    scale = scale_receipt["scale_seconds"]
    training = [e for e in examples if e.source in FIT_SOURCES]
    development = [e for e in examples if e.source in DEVELOPMENT_SOURCES]
    out.mkdir(parents=True, exist_ok=True)
    conditional_costs = {}
    for example in training:
        for row in example.group["requests"]:
            if row.get("available_for_allocation", True):
                conditional_costs.setdefault(row["workpoint_ms"], []).append(row["complete_conditional_return_seconds"])
    protocol = dict(schema=SCHEMA, status="P1_FROZEN_BEFORE_FIT", variants=VARIANTS, fit_seeds=FIT_SEEDS,
                    fit_sources=FIT_SOURCES, development_train_family_holdout=DEVELOPMENT_SOURCES,
                    P2_validation_sources=(6, 7), P2_test_sources=(8, 9, 10, 11),
                    normalization=scale_receipt, epochs=args.epochs, batch_graphs=4, hidden=32, graph_layers=2,
                    optimizer="Adam", learning_rate=.001, auxiliary_weight=.05, ranking_weight=1., temperature=.2,
                    source_group_split=True, feature_outcome_information=False,
                    conditional_cost_p95_seconds={str(b): float(np.quantile(costs, .95))
                                                  for b, costs in conditional_costs.items()},
                    p0_gate=gate, code_sha256=p0.sha_file(__file__), reference_sha256=api["reference_sha256"],
                    model_source_sha256=model_api["model_source_sha256"],
                    input_bindings={e.key: e.binding for e in examples},
                    final_epoch_checkpoint_no_development_checkpoint_selection=True,
                    source_balancing="equal physical sources / equal graphs per source / equal actual states per graph")
    p0.write_json(out / "protocol.json", protocol)
    torch = model_api["torch"]
    p0.write_json(out / "environment.json", dict(torch_version=torch.__version__, device=args.device))
    # Observable features and targets are physically separate files. Serialized
    # inputs only contain graph/scope/warm/workpoint; labels never enter packing.
    feature_index = []
    for index, example in enumerate(examples):
        name = "e%04d" % index
        directory = out / "features"
        directory.mkdir(exist_ok=True)
        for variant in VARIANTS:
            batch = observable_batch(example, variant, scale, "cpu", model_api)
            torch.save(batch, directory / (name + "." + variant + ".observable.pt"))
        torch.save(targets(example, scale, "cpu", model_api), directory / (name + ".targets.pt"))
        feature_index.append(dict(example=example.key, source=example.source, stem=name,
                                  actual_warm_by_action=example.group["snapshot"]["warm_by_action"],
                                  ordered_scopes=[dict(action_index=v.request.scope.action_index,
                                                      R=list(v.request.scope.replacements), base=sorted(v.request.scope.base),
                                                      warm=sorted(v.warm_start), workpoint_seconds=v.request.workpoint.amount)
                                                  for v in example.views], original_graph=example.binding))
    p0.write_json(out / "feature_index.json", feature_index)
    stems = {record["example"]: record["stem"] for record in feature_index}
    weights = {}
    for example in training:
        source_graphs = {e.graph_id for e in training if e.source == example.source}
        graph_states = sum(e.graph_id == example.graph_id for e in training)
        weights[example.key] = 1. / len(FIT_SOURCES) / len(source_graphs) / graph_states
    fit_results = []
    for variant in VARIANTS:
        for seed in FIT_SEEDS:
            seed_all(seed, model_api)
            model = model_api["build"](variant).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            directory = out / (variant + "-seed%d" % seed)
            directory.mkdir()
            initial_parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            initial = evaluate_model(model, development, variant, scale, args.device, model_api)
            history = [dict(epoch=0, development=initial, no_optimizer_steps=True)]
            started = time.perf_counter()
            for epoch in range(1, args.epochs + 1):
                model.train()
                graph_ids = sorted({e.graph_id for e in training})
                order = np.random.default_rng(seed + epoch).permutation(len(graph_ids))
                losses, grad_records = [], []
                # Match the old minimal optimizer regime: four whole graphs
                # per step. Mean states within each graph, then mean graphs.
                for start in range(0, len(order), 4):
                    selected_graphs = {graph_ids[int(i)] for i in order[start:start + 4]}
                    current = [e for e in training if e.graph_id in selected_graphs]
                    batch_weight = math.fsum(weights[e.key] for e in current)
                    optimizer.zero_grad()
                    for example in current:
                        stem = stems[example.key]
                        batch = torch.load(out / "features" / (stem + "." + variant + ".observable.pt"), map_location="cpu")
                        target = torch.load(out / "features" / (stem + ".targets.pt"), map_location="cpu")
                        if batch is None:
                            continue
                        batch = tensor_tree_to(batch, args.device)
                        target = tensor_tree_to(target, args.device)
                        details = model(batch, return_details=True)
                        components = state_loss(details, target, batch, variant == "ResidualCapacity", model_api)
                        (components["total"] * weights[example.key] / batch_weight).backward()
                        losses.append({key: float(components[key].detach().cpu()) for key in
                                       ("total", "regression", "ranking", "auxiliary")})
                    grad_records.append(gradient_distribution(model))
                    optimizer.step()
                evaluation = evaluate_model(model, development, variant, scale, args.device, model_api)
                drift = math.sqrt(math.fsum(float((p.detach().cpu() - initial_parameters[name]).square().sum())
                                            for name, p in model.named_parameters()))
                record = dict(epoch=epoch, optimizer_steps=epoch * math.ceil(len(graph_ids) / 4),
                              train_mean_state_components={key: float(np.mean([row[key] for row in losses]))
                                                           for key in ("total", "regression", "ranking", "auxiliary")},
                              gradient_distribution_per_optimizer_step=grad_records, parameter_l2_change=drift,
                              development=evaluation, elapsed_seconds=time.perf_counter() - started)
                history.append(record)
                p0.write_json(directory / "history.json", history)
                print(json.dumps(dict(variant=variant, seed=seed, epoch=epoch,
                                      development_loss=evaluation["mean_state_loss"])), flush=True)
            checkpoint = dict(schema=SCHEMA, variant=variant, fit_seed=seed, epoch=args.epochs,
                              model_state_dict=model.state_dict(), scale_seconds=scale,
                              protocol_sha256=p0.sha_file(out / "protocol.json"), normalization=scale_receipt,
                              remaining_time_feature_frozen_zero=True)
            torch.save(checkpoint, directory / "final.pt")
            fit_results.append(dict(variant=variant, seed=seed, final_checkpoint=str(directory / "final.pt"),
                                    sha256=p0.sha_file(directory / "final.pt"), parameter_counts=model.parameter_counts()))
    p0.write_json(out / "completion.json", dict(schema=SCHEMA, status="ACTUAL_P1_FIT_COMPLETE",
                  fits=fit_results, protocol_sha256=p0.sha_file(out / "protocol.json"),
                  P2_validation_or_test_used=False, online_advantage_established=False))


def allocation(args, api, model_api):
    fit_root = Path(args.fit_root).resolve()
    protocol = json.loads((fit_root / "protocol.json").read_text(encoding="utf-8"))
    if protocol["model_source_sha256"] != model_api["model_source_sha256"]:
        raise ValueError("Current model runtime differs from the frozen fit")
    sources = tuple(args.sources)
    if not sources or any(v not in (4, 5, 6, 7, 8, 9, 10, 11) for v in sources):
        raise ValueError("Allocation evaluation must use development/validation/test sources only")
    examples = load_examples(args.graphs_dir, args.p0_root, api, model_api, sources)
    for source in sources:
        if len({e.graph_id for e in examples if e.source == source}) != 6:
            raise ValueError("Fixed-call evaluation requires all six declared graphs for r%03d" % source)
    scale = protocol["normalization"]["scale_seconds"]
    rows, input_predictions = [], []
    for variant in VARIANTS:
        for seed in FIT_SEEDS:
            checkpoint_path = fit_root / (variant + "-seed%d" % seed) / "final.pt"
            checkpoint = model_api["torch"].load(checkpoint_path, map_location=args.device)
            if checkpoint["protocol_sha256"] != p0.sha_file(fit_root / "protocol.json"):
                raise ValueError("Checkpoint does not bind the current frozen P1 fit protocol")
            model = model_api["build"](variant).to(args.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            for example in examples:
                predictions, _ = predict(model, example, variant, scale, args.device, model_api)
                rows.extend(fixed_call_outcomes(example, predictions, variant + "-seed%d" % seed))
                input_predictions.append(dict(example=example.key, policy=variant + "-seed%d" % seed,
                                              keys=[[r["action_index"], r["workpoint_ms"]] for r in example.rows],
                                              predicted_signed_gain_seconds=predictions.tolist()))
    for example in examples:
        greedy, analytic = [], []
        descriptors = {d["action_index"]: d for d in example.group["scopes"]}
        for view, row in zip(example.views, example.rows):
            scope = view.request.scope
            greedy.append(scope.immediate_gain + api["objective"](example.graph, view.warm_start))
            analytic.append(scope.immediate_gain + descriptors[row["action_index"]]["P1_recovery_seconds"])
        rows.extend(fixed_call_outcomes(example, greedy, "Greedy"))
        rows.extend(fixed_call_outcomes(example, analytic, "P1"))
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Allocation output must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    p0.write_json(out / "predictions.json", input_predictions)
    p0.write_json(out / "fixed_call_results.json", rows)
    policies = sorted({row["policy"] for row in rows})
    summary = []
    # Hierarchical averaging: source first, then source-level mean. Additional
    # states/configurations never create extra independent physical sources.
    for policy in policies:
      for menu in sorted({r["request_menu"] for r in rows}):
        for calls in FIXED_CALLS:
            source_values = []
            for source in sources:
                graph_values = []
                for graph_id in sorted({e.graph_id for e in examples if e.source == source}):
                    selected = [r for r in rows if r["policy"] == policy and r["request_menu"] == menu and
                                r["fixed_call_limit"] == calls and r["graph_id"] == graph_id]
                    if selected:
                        graph_values.append(float(np.mean([r["beyond_snapshot_gain_seconds"] for r in selected])))
                if graph_values:
                    source_values.append(dict(source="r%03d" % source, mean_gain_seconds=float(np.mean(graph_values))))
            summary.append(dict(policy=policy, request_menu=menu, fixed_call_limit=calls, source_values=source_values,
                                mean_gain_seconds=float(np.mean([v["mean_gain_seconds"] for v in source_values]))
                                if source_values else None))
    p0.write_json(out / "summary.json", dict(schema=SCHEMA, status="FIXED_CALL_COLD_CLONE_LABEL_REPLAY_COMPLETE",
                  sources=sources, physical_sources=len(sources), examples=len(examples), results=summary,
                  fit_protocol_sha256=p0.sha_file(fit_root / "protocol.json"),
                  no_native_calls_during_replay=True, not_actual_online_or_timing_advantage=True,
                  complete_input_bindings={e.key: e.binding for e in examples}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "allocate", "actual-policy"))
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--graphs-dir")
    parser.add_argument("--p0-root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--p0-gate")
    parser.add_argument("--fit-root")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--sources", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--graph")
    parser.add_argument("--chils")
    parser.add_argument("--chils-source")
    parser.add_argument("--policy", choices=("ResidualCapacity", "ResidualCheapSummary", "P1", "Greedy"))
    parser.add_argument("--fit-seed", type=int, choices=FIT_SEEDS, default=17)
    parser.add_argument("--action-seed", type=int, default=17)
    parser.add_argument("--max-calls", type=int, choices=FIXED_CALLS, default=4)
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("--budgets-ms", type=int, nargs="+", default=[200, 1000])
    args = parser.parse_args()
    if args.stage == "fit" and (not args.p0_gate or not args.graphs_dir or not args.p0_root or args.epochs != 40):
        parser.error("fit needs gate, complete graphs/P0 roots and predeclared 40 epochs")
    if args.stage == "allocate" and (not args.fit_root or not args.graphs_dir or not args.p0_root):
        parser.error("allocate needs fit, graphs and P0 roots")
    if args.stage == "actual-policy" and (not args.fit_root or not args.graph or not args.chils or not args.chils_source or
                                          not args.policy or not args.deadline_seconds or args.device != "cpu"):
        parser.error("actual-policy needs graph/native/policy/positive deadline and resident CPU inference")
    return args


def main():
    args = parse_args()
    api = p0.load_runtime(args.runtime_root)
    model_api = load_model_api(args.runtime_root)
    if args.stage == "fit":
        fit(args, api, model_api)
    elif args.stage == "allocate":
        allocation(args, api, model_api)
    else:
        from p1_actual_policy import actual_cli
        actual_cli(args, api, model_api)


if __name__ == "__main__":
    main()
