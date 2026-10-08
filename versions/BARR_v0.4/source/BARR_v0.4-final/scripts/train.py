#!/usr/bin/env python3
"""Train action ranking on bounded marginal gain and measured cost, NOT membership.

Complete counterfactual groups only. Source-equal sampling, source-disjoint train
and validation. TEST labels and overlapping graph hashes are rejected. A layers=0
MLP control uses exactly the same features, targets and training selection.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from barr_model import RankingNet, native_score
from barr_io import atomic_json, sha256


def read_groups(path: Path, permitted: set[str]):
    groups = []; excluded = 0
    for line in path.read_text().splitlines():
        g = json.loads(line)
        if str(g.get("split", "")).upper() not in permitted:
            raise ValueError("training/validation file contains forbidden or unknown split")
        if not g.get("source_group") or not g.get("graph_sha256"):
            raise ValueError("missing physical source or graph identity")
        if not g.get("complete") or len(g.get("actions", [])) < 2:
            excluded += 1; continue
        for action in g["actions"]:
            if action["gain_ticks"] < 0 or action["solver_seconds"] <= 0 or not math.isfinite(action["solver_seconds"]):
                raise ValueError("invalid bounded outcome label")
        groups.append(g)
    if not groups:
        raise ValueError("no complete multi-action groups")
    return groups, excluded


def group_loss(net, group):
    predictions = torch.stack([net.action(a) for a in group["actions"]])
    target = torch.tensor([[math.log1p(a["gain_ticks"] / a["weight_scale"]), math.log(max(a["solver_seconds"], 1e-6))]
                           for a in group["actions"]], dtype=predictions.dtype)
    regression = F.smooth_l1_loss(predictions[:, 0], target[:, 0]) + .05 * F.smooth_l1_loss(predictions[:, 1], target[:, 1])
    # Pairwise comparisons use actual gain-per-second under the SAME bounded executor.
    utilities = torch.tensor([a["gain_ticks"] / max(a["solver_seconds"], 1e-6) for a in group["actions"]], dtype=predictions.dtype)
    pairs = torch.triu(utilities[:, None] != utilities[None, :], diagonal=1)
    logutility = torch.log(torch.expm1(predictions[:, 0].clamp(0, 20)) + 1e-6) - predictions[:, 1].clamp(-16, 10)
    # Different kernels can have different weight normalization.
    logutility = logutility + torch.log(torch.tensor([a["weight_scale"] for a in group["actions"]], dtype=predictions.dtype))
    difference = (logutility[:, None] - logutility[None, :])[pairs]
    sign = torch.sign((utilities[:, None] - utilities[None, :])[pairs])
    ranking = F.softplus(-sign * difference).mean() if len(difference) else predictions.sum() * 0
    return regression + .02 * ranking


def validate(net, groups):
    by_source = defaultdict(list)
    with torch.no_grad():
        for g in groups:
            preds = [net.action(a) for a in g["actions"]]
            chosen = max(range(len(preds)), key=lambda i: native_score(preds[i], g["actions"][i]["weight_scale"]))
            actual = [a["gain_ticks"] / max(a["solver_seconds"], 1e-6) for a in g["actions"]]
            oracle = max(actual)
            regret = (oracle - actual[chosen]) / max(1., oracle)
            loss = float(group_loss(net, g))
            by_source[g["source_group"]].append((regret, loss))
    regrets = [sum(x[0] for x in rows) / len(rows) for rows in by_source.values()]
    losses = [sum(x[1] for x in rows) / len(rows) for rows in by_source.values()]
    return {"source_equal_normalized_regret": sum(regrets) / len(regrets), "source_equal_loss": sum(losses) / len(losses)}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    a = p.parse_args()
    if a.epochs < 1 or a.steps_per_epoch < 1:
        p.error("epochs and steps must be positive")
    torch.set_num_threads(1); torch.manual_seed(a.seed); random.seed(a.seed)
    train, excluded_t = read_groups(a.train, {"TRAIN"})
    val, excluded_v = read_groups(a.validation, {"VAL", "DEV"})
    contracts = {json.dumps(g.get("executor_contract"), sort_keys=True) for g in train + val}
    if len(contracts) != 1 or "null" in contracts:
        raise ValueError("labels lack a single frozen v0.3 executor contract; recollect for this backend")
    contract = train[0]["executor_contract"]
    train_sources = {g["source_group"] for g in train}; val_sources = {g["source_group"] for g in val}
    if train_sources & val_sources or {g["graph_sha256"] for g in train} & {g["graph_sha256"] for g in val}:
        raise ValueError("source/graph leakage between training and validation")
    by_source = defaultdict(list)
    for g in train:
        by_source[g["source_group"]].append(g)
    sources = sorted(by_source)
    net = RankingNet(a.hidden, a.layers).double()
    # Avoid an initially negative gain head producing all-zero clipped deployment ranks.
    with torch.no_grad():
        net.out.bias[0] = .1; net.out.bias[1] = -7
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.learning_rate)
    history = []; best_key = (float("inf"), float("inf")); best_state = None
    for epoch in range(a.epochs):
        net.train(); train_loss = 0
        for _ in range(a.steps_per_epoch):
            group = random.choice(by_source[random.choice(sources)])
            optimizer.zero_grad(); loss = group_loss(net, group)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite learning loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step(); train_loss += float(loss.detach())
        net.eval(); metric = validate(net, val)
        key = (metric["source_equal_normalized_regret"], metric["source_equal_loss"])
        if key < best_key:
            best_key = key; best_state = copy.deepcopy(net.state_dict()); best_epoch = epoch
        history.append({"epoch": epoch, "train_loss": train_loss / a.steps_per_epoch, **metric})
        print(json.dumps(history[-1]), flush=True)
    net.load_state_dict(best_state); a.out.parent.mkdir(parents=True, exist_ok=True); net.export(a.out)
    positives = sum(act["gain_ticks"] > 0 for g in train for act in g["actions"])
    total = sum(len(g["actions"]) for g in train)
    atomic_json(a.out.with_suffix(a.out.suffix + ".json"), {"schema": "barr_training_v1", "model_sha256": sha256(a.out), "executor_contract": contract,
        "train_sha256": sha256(a.train), "validation_sha256": sha256(a.validation), "train_sources": sorted(train_sources),
        "validation_sources": sorted(val_sources), "excluded_incomplete_train": excluded_t, "excluded_incomplete_val": excluded_v,
        "parameter_count": sum(p.numel() for p in net.parameters()), "positive_gain_labels": positives, "total_labels": total, "degenerate_all_zero_labels": positives == 0,
        "best_epoch": best_epoch, "history": history, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "real_STK_performance_validated": False, "target": "observed bounded gain beyond exact fusion and measured native action cost",
        "not_a_proof_of_GNN_value": True})
    if positives == 0:
        print("WARNING: all training gains are zero; this model cannot support useful marginal-gain learning claims.", file=sys.stderr)
    print(a.out)
