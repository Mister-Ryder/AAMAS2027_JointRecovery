"""Budget-monotone compatibility-constrained recovery value candidate.

No encoder or recovery endpoint depends on execution budget. Only a positive-
slope scalar mixture gate receives log(1+B). The endpoints are capacity-
feasible occupancies sorted by their weighted values; their convex mixture
therefore remains capacity feasible and has nondecreasing recovery value.
This structural consistency does not imply accurate finite-budget prediction.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .core import ACTION, DISPLACED
from .v3_capacity import CapacityRecoveryNet, capacity_project


def monotone_occupancy_mix(first, second, weights, log_budget,
                           slope_raw, intercept):
    """Capacity-preserving, value-monotone interpolation of two fixed endpoints.

    Endpoint order can change with the graph/action, but NEVER with B. Individual
    node occupancies need not grow with B: a better recovery may replace a
    previous choice with another compatible combination. log_budget is monotone
    in B and positively scaled, e.g. log1p(B)/log(1025).
    """
    first_value = (first * weights).sum(dim=1)
    second_value = (second * weights).sum(dim=1)
    first_is_low = (first_value <= second_value).unsqueeze(-1)
    low = torch.where(first_is_low, first, second)
    high = torch.where(first_is_low, second, first)
    slope = F.softplus(slope_raw)
    gate = torch.sigmoid(slope * log_budget + intercept)
    occupancy = low + gate.unsqueeze(-1) * (high - low)
    return dict(occupancy=occupancy, low=low, high=high, gate=gate,
                slope=slope, low_value=(low * weights).sum(dim=1),
                high_value=(high * weights).sum(dim=1))


class MonotoneBudgetRecoveryNet(CapacityRecoveryNet):
    """Independent seventh variant; checkpoint/loss API matches v3 capacity.

    The original six variants are untouched. Their typed/owner encoder design
    is reused here, but the two B-derived input channels are always zero. The
    third channel, log replacement count, remains observable graph context.
    The first occupancy head is inherited; a second endpoint head and a gate
    head are independent additions. The unused generic head remains stored so
    parameter_counts reports deployment path and total checkpoint separately.
    """
    def __init__(self, hidden=48, layers=3, use_agent_relay=True):
        super().__init__(hidden=hidden, layers=layers, use_budget=False,
                         use_agent_relay=use_agent_relay, readout="occupancy")
        self.second_endpoint_head = nn.Sequential(
            nn.Linear(3 * hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.second_endpoint_head[-1].bias.data.fill_(2.)
        self.gate_head = nn.Sequential(nn.Linear(4 * hidden + 5, hidden),
                                       nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, batch, return_details=False):
        x, mask = batch["x"], batch["mask"]
        # Do not forward either budget-derived channel into the encoder,
        # endpoint heads, owner relay, or gate-parameter head. This is required
        # for the monotonicity proof, rather than a data-dependent heuristic.
        conditioning = torch.cat((torch.zeros_like(batch["budget_features"][:, :2]),
                                  batch["budget_features"][:, 2:]), dim=1)
        repeated = conditioning.unsqueeze(1).expand(-1, x.shape[1], -1)
        h = torch.relu(self.embed(torch.cat((x, repeated), dim=-1))) * mask.unsqueeze(-1)
        typed = batch["edges"]
        degree = typed.sum(dim=(1, 3)).unsqueeze(-1).clamp_min(1.)
        owners = batch["agents"]
        owner_size = owners.sum(dim=-1, keepdim=True).clamp_min(1.)
        for self_layer, relations, owner_layer in zip(
                self.self_layers, self.relation_layers, self.agent_layers):
            messages = torch.zeros_like(h)
            for t, relation in enumerate(relations):
                messages = messages + torch.bmm(typed[:, t], relation(h))
            messages = messages / degree
            if self.use_agent_relay:
                owned_mean = torch.bmm(owners, h) / owner_size
                messages = messages + torch.bmm(owners.transpose(1, 2), owner_layer(owned_mean))
            h = torch.relu(self_layer(h) + messages) * mask.unsqueeze(-1)
        pools, counts = [], []
        for role in range(4):
            indicator = x[:, :, role:role + 1] * mask.unsqueeze(-1)
            size = indicator.sum(dim=1)
            pools.append((h * indicator).sum(dim=1) / size.clamp_min(1.))
            counts.append(torch.log1p(size))
        action_context = pools[ACTION].unsqueeze(1).expand(-1, h.shape[1], -1)
        displaced_context = pools[DISPLACED].unsqueeze(1).expand(-1, h.shape[1], -1)
        endpoint_features = torch.cat((h, action_context, displaced_context, repeated), dim=-1)
        first_logits = self.occupancy_head(endpoint_features).squeeze(-1)
        second_logits = self.second_endpoint_head(endpoint_features).squeeze(-1)
        first = capacity_project(torch.sigmoid(first_logits) * batch["replacement"], batch["cliques"])
        second = capacity_project(torch.sigmoid(second_logits) * batch["replacement"], batch["cliques"])
        gate_parameters = self.gate_head(torch.cat(pools + counts + [conditioning[:, 2:]], dim=1))
        mixed = monotone_occupancy_mix(first, second, batch["weights"],
                                       batch["budget_features"][:, 0],
                                       gate_parameters[:, 0], gate_parameters[:, 1])
        occupancy = mixed["occupancy"]
        recovery = (occupancy * batch["weights"]).sum(dim=1)
        raw_gain = batch["immediate"] + recovery
        scores = raw_gain.clamp_min(0.)
        if return_details:
            # The common auxiliary BCE interface supervises the actual mixed
            # occupancy, not a B-independent endpoint. It never becomes a
            # feature. clamp avoids infinities on padded/nonreplacement nodes.
            bounded = occupancy.clamp(1e-6, 1. - 1e-6)
            logits = torch.log(bounded) - torch.log1p(-bounded)
            return dict(scores=scores, raw_gain=raw_gain, recovery=recovery,
                        occupancy=occupancy, proposed=occupancy, logits=logits,
                        first_endpoint=first, second_endpoint=second,
                        low_endpoint=mixed["low"], high_endpoint=mixed["high"],
                        low_value=mixed["low_value"], high_value=mixed["high_value"],
                        budget_gate=mixed["gate"], budget_slope=mixed["slope"])
        return scores

    def parameter_counts(self):
        counts = super().parameter_counts()
        # Parent dynamically counts self.parameters(), including new heads,
        # and removes precisely the unused free action readout.
        return counts
