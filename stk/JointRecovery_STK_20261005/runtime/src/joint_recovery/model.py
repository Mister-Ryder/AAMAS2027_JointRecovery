"""Typed action-response graph encoder, without outcome-derived inputs."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def pack(responses, device="cpu", drop_replacement_edges=False):
    """Pack disconnected response graphs; no communication between actions."""
    xs, edges, types, graph_ids = [], [], [], []
    offset = 0
    for i, response in enumerate(responses):
        x = np.asarray(response.x, dtype=np.float32).copy()
        ei = np.asarray(response.edge_index, dtype=np.int64)
        et = np.asarray(response.edge_type, dtype=np.int64)
        if drop_replacement_edges and ei.shape[1]:
            # Remove replacement incompatibility and its SAME_AGENT mirror.
            # Keep node summaries identical: this tests edge traversal beyond
            # aggregate degree information, not absence of all compatibility data.
            rr = (x[ei[0], 3] > .5) & (x[ei[1], 3] > .5)
            keep = ~(rr & ((et == 0) | (et == 3)))
            ei, et = ei[:, keep], et[keep]
        xs.append(x)
        edges.append(ei + offset)
        types.append(et)
        graph_ids.append(np.full(len(x), i, dtype=np.int64))
        offset += len(x)
    return (torch.tensor(np.concatenate(xs), device=device),
            torch.tensor(np.concatenate(edges, axis=1), device=device),
            torch.tensor(np.concatenate(types), device=device),
            torch.tensor(np.concatenate(graph_ids), device=device), len(responses))


class RecoveryGNN(nn.Module):
    """Three relation-specific message layers and separate role-pooled readout."""
    def __init__(self, hidden=48, layers=3, node_dim=10, edge_types=4):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.self_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.edge_layers = nn.ModuleList(nn.ModuleList(nn.Linear(hidden, hidden, bias=False)
                                            for _ in range(edge_types)) for _ in range(layers))
        self.readout = nn.Sequential(nn.Linear(hidden * 4 + 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch):
        x, edges, types, gids, count = batch
        h = torch.relu(self.embed(x))
        for self_layer, edge_layers in zip(self.self_layers, self.edge_layers):
            agg = torch.zeros_like(h)
            degree = torch.zeros(len(h), 1, device=h.device)
            for t, edge_layer in enumerate(edge_layers):
                mask = types == t
                source, dest = edges[:, mask]
                if source.numel():
                    agg.index_add_(0, dest, edge_layer(h[source]))
                    degree.index_add_(0, dest, torch.ones(len(source), 1, device=h.device))
            h = torch.relu(self_layer(h) + agg / degree.clamp_min(1))
        pools, sizes = [], []
        for role in range(4):
            indicator = x[:, role:role + 1]
            pooled = torch.zeros(count, h.shape[1], device=h.device)
            size = torch.zeros(count, 1, device=h.device)
            pooled.index_add_(0, gids, h * indicator)
            size.index_add_(0, gids, indicator)
            pools.append(pooled / size.clamp_min(1))
            sizes.append(torch.log1p(size))
        return self.readout(torch.cat(pools + sizes, dim=1)).squeeze(1)


class PooledMLP(nn.Module):
    """Matched node inputs, no edge traversal; compact flat-information control."""
    def __init__(self, hidden=48, node_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.readout = nn.Sequential(nn.Linear(hidden * 4 + 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch):
        x, edges, types, gids, count = batch
        h = self.encoder(x)
        pools, sizes = [], []
        for role in range(4):
            indicator = x[:, role:role + 1]
            pooled = torch.zeros(count, h.shape[1], device=h.device)
            size = torch.zeros(count, 1, device=h.device)
            pooled.index_add_(0, gids, h * indicator)
            size.index_add_(0, gids, indicator)
            pools.append(pooled / size.clamp_min(1))
            sizes.append(torch.log1p(size))
        return self.readout(torch.cat(pools + sizes, dim=1)).squeeze(1)


def ranking_loss(scores, gains, group_sizes):
    """Pairwise supervised ordering of different feasible outcomes, plus regression."""
    loss = .25 * torch.mean((scores - gains) ** 2)
    groups = torch.repeat_interleave(torch.arange(len(group_sizes), device=scores.device),
                                    torch.tensor(group_sizes, device=scores.device))
    diff = gains[:, None] - gains[None, :]
    mask = (diff > 1e-6) & (groups[:, None] == groups[None, :])
    if mask.any():
        delta = scores[:, None] - scores[None, :]
        loss = loss + torch.nn.functional.softplus(-delta[mask]).mean()
    return loss
