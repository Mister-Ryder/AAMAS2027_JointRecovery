"""Action-conditioned, compatibility-capacity recovery value models.

This module never calls the execution/search kernel when constructing features.
It predicts recovery occupancies, projects them against an observable conflict
clique cover, and uses their weighted sum as the ONLY recovery-value readout.
The fractional prediction is not an executable independent set. The unchanged
deterministic kernel remains responsible for the final feasible improvement.

Designed for Python 3.8 / PyTorch 1.11 as well as the local CPU runtime.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import ACTION, DISPLACED, REPLACEMENT, INCOMPATIBILITY


def conflict_clique_cover(adjacency, weights=None, mode="cliques"):
    """Cover every edge with valid cliques without solving an optimization task.

    One greedy clique is grown from each nonisolated vertex, using descending
    (weight, degree) and vertex position for deterministic ties. Remaining edges
    are added as two-vertex factors. The result is at most |V| + |E| factors.
    The cover can depend on vertex position at exact ordering ties; capacity
    validity does not. No maximal-clique enumeration or label information is
    used. ``edges`` uses precisely the edge factors; ``none`` uses no factors.
    """
    if mode not in ("none", "edges", "cliques"):
        raise ValueError("projection mode must be none, edges, or cliques")
    n = len(adjacency)
    neighbors = [frozenset(int(j) for j in row) for row in adjacency]
    if any(i in row or any(j < 0 or j >= n for j in row)
           for i, row in enumerate(neighbors)):
        raise ValueError("invalid local conflict adjacency")
    if any(i not in neighbors[j] for i, row in enumerate(neighbors) for j in row):
        raise ValueError("local conflict adjacency must be symmetric")
    edge_set = {(i, j) for i, row in enumerate(neighbors) for j in row if i < j}
    if mode == "none":
        return ()
    if mode == "edges":
        return tuple(tuple(pair) for pair in sorted(edge_set))
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)):
        raise ValueError("clique ordering weights must be finite and size n")
    ordering = sorted(range(n), key=lambda i: (-float(w[i]), -len(neighbors[i]), i))
    factors = set()
    covered = set()
    for seed in ordering:
        if not neighbors[seed]:
            continue
        clique = [seed]
        allowed = set(neighbors[seed])
        for v in ordering:
            if v in allowed:
                clique.append(v)
                allowed.intersection_update(neighbors[v])
        factor = tuple(sorted(clique))
        if len(factor) >= 2:
            factors.add(factor)
            covered.update((u, v) for index, u in enumerate(factor)
                           for v in factor[index + 1:])
    factors.update(tuple(pair) for pair in edge_set - covered)
    return tuple(sorted(factors))


def _vector(values, count, default=None):
    if values is None:
        values = default
    a = np.asarray(values, dtype=np.float32)
    if a.ndim == 0:
        return np.full(count, float(a), dtype=np.float32)
    if a.shape != (count,):
        raise ValueError("scalar or one value per response required")
    return a


def pack_v3(graph, selected, responses: Sequence, budget=64, scale=None,
            device="cpu", projection="cliques", drop_replacement_edges=False):
    """Pack one state's observable responses into a dense tensor batch.

    ``scale`` must equal the target normalizer; by default max(1, graph mean
    weight), matching the original study. Scalar/per-action budgets are both
    accepted. Response nodes and edge relations retain their original roles.
    Original-degree normalization is replaced by log1p(degree)/log(65), which
    does not shrink the same local degree when the parent graph becomes larger.
    Agent membership is a contextual grouping, not a feasibility constraint.
    """
    del selected  # membership is already encoded by build_response; no outcome.
    count = len(responses)
    if not count:
        raise ValueError("at least one action response required")
    if scale is None:
        scale = max(1.0, float(np.mean(graph.weights)))
    scales = _vector(scale, count)
    budgets = _vector(budget, count)
    if (np.any(scales <= 0) or not np.all(np.isfinite(scales))
            or np.any(budgets < 0) or not np.all(np.isfinite(budgets))):
        raise ValueError("positive scales and nonnegative finite budgets required")
    nmax = max(len(r.node_ids) for r in responses)
    records, fmax, amax = [], 1, 1
    for r in responses:
        ids = np.asarray(r.node_ids, dtype=np.int64)
        rpositions = np.flatnonzero(r.roles == REPLACEMENT)
        rindex = {int(position): j for j, position in enumerate(rpositions)}
        conflicts = [set() for _ in rpositions]
        for (u, v), et in zip(r.edge_index.T, r.edge_type):
            if et == INCOMPATIBILITY and int(u) in rindex and int(v) in rindex:
                conflicts[rindex[int(u)]].add(rindex[int(v)])
        # Dropping R-R edges also drops the compatibility factors. Keeping a
        # cover built from a removed relation would invalidate that ablation.
        factors = conflict_clique_cover(
            conflicts, graph.weights[ids[rpositions]],
            "none" if drop_replacement_edges else projection)
        ownership = graph.agents[ids]
        distinct_agents = np.unique(ownership)
        fmax, amax = max(fmax, len(factors)), max(amax, len(distinct_agents))
        records.append((ids, rpositions, factors, ownership, distinct_agents))
    x = np.zeros((count, nmax, 10), np.float32)
    mask = np.zeros((count, nmax), np.float32)
    weights = np.zeros((count, nmax), np.float32)
    replacement = np.zeros((count, nmax), np.float32)
    edges = np.zeros((count, 4, nmax, nmax), np.float32)
    cliques = np.zeros((count, fmax, nmax), np.float32)
    agents = np.zeros((count, amax, nmax), np.float32)
    immediate = np.zeros(count, np.float32)
    diagnostics = []
    for i, (r, record) in enumerate(zip(responses, records)):
        ids, rp, factors, ownership, distinct_agents = record
        n = len(ids)
        x[i, :n] = r.x
        x[i, :n, 5] = np.log1p([len(graph.adjacency[int(v)]) for v in ids]) / np.log(65.)
        mask[i, :n] = 1.
        weights[i, :n] = graph.weights[ids] / scales[i]
        replacement[i, rp] = 1.
        immediate[i] = float(np.sum(weights[i, :n][r.roles == ACTION])
                             - np.sum(weights[i, :n][r.roles == DISPLACED]))
        for (u, v), et in zip(r.edge_index.T, r.edge_type):
            if drop_replacement_edges and r.roles[u] == REPLACEMENT and r.roles[v] == REPLACEMENT:
                continue
            edges[i, int(et), int(v), int(u)] = 1.
        for j, factor in enumerate(factors):
            cliques[i, j, rp[list(factor)]] = 1.
        for j, owner in enumerate(distinct_agents):
            agents[i, j, :n] = ownership == owner
        diagnostics.append(dict(nodes=n, replacements=len(rp), factors=len(factors),
                                factor_memberships=sum(map(len, factors)),
                                max_clique=max(map(len, factors), default=0),
                                agents=len(distinct_agents)))
    # Conditioning is continuous so unseen budgets remain valid inputs; this
    # gives no extrapolation guarantee. Count normalization is action-local.
    rcounts = replacement.sum(axis=1)
    budget_features = np.stack((np.log1p(budgets) / np.log(1025.),
                               np.log1p(budgets / np.maximum(1., rcounts)) / np.log(1025.),
                               np.log1p(rcounts) / np.log(65.)), axis=1).astype(np.float32)
    arrays = dict(x=x, mask=mask, weights=weights, replacement=replacement,
                  edges=edges, cliques=cliques, agents=agents,
                  immediate=immediate, budget_features=budget_features)
    result = {name: torch.as_tensor(array, device=device) for name, array in arrays.items()}
    result["diagnostics"] = diagnostics
    return result


def capacity_project(probabilities, cliques):
    """Simultaneous capacity contraction; every provided clique sums to <=1.

    d_i = max(1, max_{Q contains i} sum_{j in Q} p_j), z_i = p_i/d_i.
    All p must lie in [0,1]; uncovered isolated vertices need no contraction.
    Every constraint uses the same pre-projection probabilities. Division by
    the largest incident load ensures all overlapping factors remain valid.
    """
    loads = torch.bmm(cliques, probabilities.unsqueeze(-1)).squeeze(-1)
    denominator = (loads.unsqueeze(-1) * cliques).amax(dim=1).clamp_min(1.)
    return probabilities / denominator


def merge_v3_batches(batches):
    """Concatenate already packed states, preserving independent graph blocks.

    Padding is observable only through masks. State-group sizes for the loss
    remain the caller's responsibility; actions never exchange graph messages.
    """
    if not batches:
        raise ValueError("at least one packed state required")
    nmax = max(b["x"].shape[1] for b in batches)
    fmax = max(b["cliques"].shape[1] for b in batches)
    amax = max(b["agents"].shape[1] for b in batches)
    items = {name: [] for name in batches[0] if name != "diagnostics"}
    diagnostics = []
    for batch in batches:
        n, f, a = batch["x"].shape[1], batch["cliques"].shape[1], batch["agents"].shape[1]
        items["x"].append(F.pad(batch["x"], (0, 0, 0, nmax - n)))
        for name in ("mask", "weights", "replacement"):
            items[name].append(F.pad(batch[name], (0, nmax - n)))
        items["edges"].append(F.pad(batch["edges"], (0, nmax - n, 0, nmax - n)))
        items["cliques"].append(F.pad(batch["cliques"], (0, nmax - n, 0, fmax - f)))
        items["agents"].append(F.pad(batch["agents"], (0, nmax - n, 0, amax - a)))
        for name in ("immediate", "budget_features"):
            items[name].append(batch[name])
        diagnostics.extend(batch["diagnostics"])
    result = {name: torch.cat(values, dim=0) for name, values in items.items()}
    result["diagnostics"] = diagnostics
    return result


class CapacityRecoveryNet(nn.Module):
    """Learned fractional recovery occupancy with a constrained value readout.

    There is deliberately no unconstrained action-score residual. ``projection``
    is controlled by pack_v3, allowing matched-weight structural ablations.
    Agent relay shares context within owners but does not assign agents rewards.
    """
    def __init__(self, hidden=48, layers=3, use_budget=True, use_agent_relay=True,
                 readout="occupancy"):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.use_budget, self.use_agent_relay = use_budget, use_agent_relay
        if readout not in ("occupancy", "unconstrained"):
            raise ValueError("readout must be occupancy or unconstrained")
        self.readout = readout
        self.embed = nn.Linear(13, hidden)
        self.self_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.relation_layers = nn.ModuleList(
            nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(4))
            for _ in range(layers))
        self.agent_layers = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(layers))
        self.occupancy_head = nn.Sequential(nn.Linear(3 * hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.occupancy_head[-1].bias.data.fill_(1.)
        # The generic readout is a control that shares every encoder layer and
        # optional mask objective. Both heads exist in all variants so encoder
        # initialization is matched; report active as well as stored parameters.
        self.free_head = nn.Sequential(nn.Linear(4 * hidden + 7, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch, return_details=False):
        x, mask = batch["x"], batch["mask"]
        conditioning = batch["budget_features"]
        if not self.use_budget:
            # Keep replacement-count information identical. Only the two
            # channels involving B are removed by the no-budget ablation.
            conditioning = torch.cat((torch.zeros_like(conditioning[:, :2]),
                                      conditioning[:, 2:]), dim=1)
        repeated = conditioning.unsqueeze(1).expand(-1, x.shape[1], -1)
        h = torch.relu(self.embed(torch.cat((x, repeated), dim=-1))) * mask.unsqueeze(-1)
        typed = batch["edges"]
        degree = typed.sum(dim=(1, 3)).unsqueeze(-1).clamp_min(1.)
        agent_membership = batch["agents"]
        agent_size = agent_membership.sum(dim=-1, keepdim=True).clamp_min(1.)
        for self_layer, relation_layers, agent_layer in zip(self.self_layers, self.relation_layers, self.agent_layers):
            messages = torch.zeros_like(h)
            for t, relation_layer in enumerate(relation_layers):
                messages = messages + torch.bmm(typed[:, t], relation_layer(h))
            messages = messages / degree
            if self.use_agent_relay:
                owned_mean = torch.bmm(agent_membership, h) / agent_size
                messages = messages + torch.bmm(agent_membership.transpose(1, 2), agent_layer(owned_mean))
            h = torch.relu(self_layer(h) + messages) * mask.unsqueeze(-1)
        contexts = []
        for role in (ACTION, DISPLACED):
            indicator = x[:, :, role:role + 1] * mask.unsqueeze(-1)
            pooled = (h * indicator).sum(dim=1) / indicator.sum(dim=1).clamp_min(1.)
            contexts.append(pooled.unsqueeze(1).expand(-1, h.shape[1], -1))
        logits = self.occupancy_head(torch.cat([h] + contexts + [repeated], dim=-1)).squeeze(-1)
        proposed = torch.sigmoid(logits) * batch["replacement"]
        occupancy = capacity_project(proposed, batch["cliques"])
        recovery = (occupancy * batch["weights"]).sum(dim=1)
        raw_gain = batch["immediate"] + recovery
        if self.readout == "unconstrained":
            pools, counts = [], []
            for role in range(4):
                indicator = x[:, :, role:role + 1] * mask.unsqueeze(-1)
                size = indicator.sum(dim=1)
                pools.append((h * indicator).sum(dim=1) / size.clamp_min(1.))
                counts.append(torch.log1p(size))
            raw_gain = self.free_head(torch.cat(pools + counts + [conditioning], dim=1)).squeeze(-1)
        scores = raw_gain.clamp_min(0.)
        if return_details:
            return dict(scores=scores, raw_gain=raw_gain, recovery=recovery,
                        occupancy=occupancy, proposed=proposed, logits=logits)
        return scores

    def parameter_counts(self):
        """Parameters on the deployed scoring path, alongside checkpoint total."""
        stored = sum(p.numel() for p in self.parameters())
        inactive = self.free_head if self.readout == "occupancy" else self.occupancy_head
        active = stored - sum(p.numel() for p in inactive.parameters())
        if not self.use_agent_relay:
            active -= sum(p.numel() for p in self.agent_layers.parameters())
        return dict(stored=stored, active_scoring=active)


def occupancy_labels(responses, outcomes, nmax=None, device="cpu"):
    """Offline auxiliary labels, never inputs to pack_v3 or deployment.

    Uses raw_selected before acceptance: a rejected update still has a feasible
    realized recovery set. Different feasible sets with equal value need not
    agree, hence this auxiliary objective is weighted lightly or ablated.
    """
    if len(responses) != len(outcomes):
        raise ValueError("one executed outcome per response required")
    nmax = nmax or max(len(r.node_ids) for r in responses)
    labels = np.zeros((len(responses), nmax), np.float32)
    for i, (response, outcome) in enumerate(zip(responses, outcomes)):
        for j, node in enumerate(response.node_ids):
            labels[i, j] = float(response.roles[j] == REPLACEMENT and node in outcome.raw_selected)
    return torch.as_tensor(labels, device=device)


def capacity_loss(details, gains, group_sizes, batch, occupancy_target=None,
                  raw_gains=None, auxiliary_weight=.05):
    """Same-state realized preferences, calibrated value, optional node labels."""
    scores = details["scores"]
    gains = torch.as_tensor(gains, dtype=scores.dtype, device=scores.device)
    # Regression before clamp supplies a useful gradient on rejected actions.
    target = gains if raw_gains is None else torch.as_tensor(raw_gains, dtype=scores.dtype, device=scores.device)
    loss = .25 * F.smooth_l1_loss(details["raw_gain"], target)
    groups = torch.repeat_interleave(torch.arange(len(group_sizes), device=scores.device),
                                    torch.tensor(group_sizes, device=scores.device))
    if len(groups) != len(scores):
        raise ValueError("group sizes must cover the packed action count")
    difference = gains[:, None] - gains[None, :]
    ordered = (difference > 1e-6) & (groups[:, None] == groups[None, :])
    if ordered.any():
        predicted = details["raw_gain"][:, None] - details["raw_gain"][None, :]
        loss = loss + F.softplus(-predicted[ordered]).mean()
    if occupancy_target is not None and auxiliary_weight:
        auxiliary = F.binary_cross_entropy_with_logits(details["logits"], occupancy_target,
                                                       reduction="none")
        replacement = batch["replacement"]
        loss = loss + auxiliary_weight * (auxiliary * replacement).sum() / replacement.sum().clamp_min(1.)
    return loss
