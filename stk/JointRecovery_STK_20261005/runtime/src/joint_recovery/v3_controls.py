"""Independent information/readout controls for joint-recovery experiments.

No teacher, optimizer, solver, or recovery mask is called during feature
construction. Existing frozen v3_capacity and v3_budget modules are unchanged.
The controls distinguish (i) local recovery from P/D provenance, and (ii)
clique information from the capacity-constrained readout using that information.
"""
from __future__ import annotations

import torch
from torch import nn

from .core import ACTION, DISPLACED, REPLACEMENT
from .v3_capacity import CapacityRecoveryNet, capacity_project


CLIQUE_STATISTIC_NAMES = (
    "log_factor_count", "max_clique_size", "mean_clique_size",
    "std_clique_size", "mean_factor_memberships", "max_factor_memberships",
    "retained_weight_fraction_p1", "log_replacement_weight",
    "log_analytic_recovery_p1", "log_pair_conflict_count",
    "high_order_factor_fraction", "known_immediate_gain",
)


def local_recovery_batch(batch, minimal_features=True, keep_owner_relations=False):
    """Remove all P/D nodes/messages; retain known q, R count and budget.

    The default recomputes node features ONLY from normalized R weights and
    the induced R conflict graph: role R, local reward, log local degree and
    local normalized degree. Original graph degree, parent maximum reward,
    P/D-dependent response-size normalization, and outside-owner statistics
    are removed. Thus this is a deliberately smaller-information control.
    ``minimal_features=False`` is a separate context-masking control retaining
    the original observable R node summaries, and should be named distinctly.
    """
    replacement = batch["replacement"]
    pair_mask = replacement.unsqueeze(1) * replacement.unsqueeze(2)
    edges = batch["edges"] * pair_mask.unsqueeze(1)
    edges = edges.clone()
    edges[:, 1:3] = 0.  # P-D and D-R dependency relations are not R relations.
    if not keep_owner_relations:
        edges[:, 3] = 0.
    weights = batch["weights"] * replacement
    x = batch["x"] * replacement.unsqueeze(-1)
    if minimal_features:
        x = torch.zeros_like(x)
        x[:, :, REPLACEMENT] = replacement
        local_max = weights.amax(dim=1, keepdim=True).clamp_min(1e-12)
        x[:, :, 4] = weights / local_max
        degree = edges[:, 0].sum(dim=2)
        x[:, :, 5] = torch.log1p(degree) / torch.log(torch.tensor(65., device=x.device))
        if keep_owner_relations:
            x[:, :, 8] = edges[:, 3].sum(dim=2) / degree.clamp_min(1.)
        x[:, :, 9] = degree / (replacement.sum(dim=1, keepdim=True) - 1.).clamp_min(1.)
    owners = batch["agents"] * replacement.unsqueeze(1)
    result = dict(batch)
    result.update(x=x, mask=replacement, weights=weights, edges=edges,
                  agents=owners, cliques=torch.zeros_like(batch["cliques"]))
    result["diagnostics"] = [dict(nodes=int(replacement[i].sum().item()),
                                  replacements=int(replacement[i].sum().item()),
                                  factors=0, factor_memberships=0, max_clique=0,
                                  agents=int((owners[i].sum(dim=1) > 0).sum().item()))
                             for i in range(len(replacement))]
    return result


def analytic_clique_details(batch):
    """p=1 clique contraction: no learned parameter, search, or output label.

    A fractional analytic score, NOT a certified upper or lower bound on
    integer recovery: uneven clique weights can underpredict, and odd cycles
    can overpredict. Candidate/factor preparation cost still must be charged.
    """
    proposed = batch["replacement"]
    occupancy = capacity_project(proposed, batch["cliques"])
    recovery = (occupancy * batch["weights"]).sum(dim=1)
    raw = batch["immediate"] + recovery
    return dict(scores=raw.clamp_min(0.), raw_gain=raw, recovery=recovery,
                occupancy=occupancy, proposed=proposed)


def clique_statistics(batch):
    """Scalar summaries derived from the SAME observable clique family.

    These give a free scalar model the high-order witness information supplied
    to the capacity layer. They are a compressed summary, not an assertion of
    equivalence to the complete incidence tensor on all possible instances.
    Normalized raw rewards and q are exposed, matching the known-value inputs
    used by the constrained model, rather than hiding them in reward/max tags.
    """
    factors = batch["cliques"]
    sizes = factors.sum(dim=2)
    valid = (sizes > 0).to(sizes.dtype)
    factor_count = valid.sum(dim=1)
    denominator = factor_count.clamp_min(1.)
    mean_size = sizes.sum(dim=1) / denominator
    variance = ((sizes - mean_size.unsqueeze(-1)) ** 2 * valid).sum(dim=1) / denominator
    memberships = factors.sum(dim=1) * batch["replacement"]
    count = batch["replacement"].sum(dim=1)
    mean_memberships = memberships.sum(dim=1) / count.clamp_min(1.)
    total_weight = (batch["weights"] * batch["replacement"]).sum(dim=1)
    analytic = analytic_clique_details(batch)["recovery"]
    rr = batch["edges"][:, 0] * batch["replacement"].unsqueeze(1) * batch["replacement"].unsqueeze(2)
    pair_count = .5 * rr.sum(dim=(1, 2))
    return torch.stack((torch.log1p(factor_count), sizes.amax(dim=1), mean_size,
                        torch.sqrt(variance.clamp_min(0.)), mean_memberships,
                        memberships.amax(dim=1), analytic / total_weight.clamp_min(1e-12),
                        torch.log1p(total_weight), torch.log1p(analytic.clamp_min(0.)),
                        torch.log1p(pair_count), (sizes >= 3).to(sizes.dtype).sum(dim=1) / denominator,
                        batch["immediate"]), dim=1)


def _encode(model, batch):
    """Exact shared encoder equations; source is independent of frozen model."""
    x, mask = batch["x"], batch["mask"]
    conditioning = batch["budget_features"]
    if not model.use_budget:
        conditioning = torch.cat((torch.zeros_like(conditioning[:, :2]), conditioning[:, 2:]), dim=1)
    repeated = conditioning.unsqueeze(1).expand(-1, x.shape[1], -1)
    h = torch.relu(model.embed(torch.cat((x, repeated), dim=-1))) * mask.unsqueeze(-1)
    degree = batch["edges"].sum(dim=(1, 3)).unsqueeze(-1).clamp_min(1.)
    owner_size = batch["agents"].sum(dim=-1, keepdim=True).clamp_min(1.)
    for self_layer, relations, owner_layer in zip(model.self_layers, model.relation_layers, model.agent_layers):
        messages = torch.zeros_like(h)
        for relation_type, relation in enumerate(relations):
            messages = messages + torch.bmm(batch["edges"][:, relation_type], relation(h))
        messages = messages / degree
        if model.use_agent_relay:
            owned_mean = torch.bmm(batch["agents"], h) / owner_size
            messages = messages + torch.bmm(batch["agents"].transpose(1, 2), owner_layer(owned_mean))
        h = torch.relu(self_layer(h) + messages) * mask.unsqueeze(-1)
    pools, sizes = [], []
    for role in range(4):
        indicator = x[:, :, role:role + 1] * mask.unsqueeze(-1)
        size = indicator.sum(dim=1)
        pools.append((h * indicator).sum(dim=1) / size.clamp_min(1.))
        sizes.append(torch.log1p(size))
    free_features = torch.cat(pools + sizes + [conditioning], dim=1)
    return h, repeated, pools, free_features


def _node_auxiliary(model, batch, h, repeated, pools):
    action = pools[ACTION].unsqueeze(1).expand(-1, h.shape[1], -1)
    displaced = pools[DISPLACED].unsqueeze(1).expand(-1, h.shape[1], -1)
    logits = model.occupancy_head(torch.cat((h, action, displaced, repeated), dim=-1)).squeeze(-1)
    proposed = torch.sigmoid(logits) * batch["replacement"]
    return logits, proposed


class LocalRecoveryGNN(CapacityRecoveryNet):
    """R-only graph encoder, free recovery fraction, and known q readout.

    Default lacks owner/provenance relations because neither changes the local
    recovery objective once R is fixed. The positive free scalar fraction times
    sum R weights is NOT a compatibility-constrained occupancy readout. Optional
    mask BCE trains the common encoder/head but masks remain training-only.
    """
    def __init__(self, hidden=48, layers=3, use_budget=True, use_agent_relay=False,
                 minimal_features=True, keep_owner_relations=False):
        super().__init__(hidden, layers, use_budget, use_agent_relay, readout="unconstrained")
        self.minimal_features = minimal_features
        self.keep_owner_relations = keep_owner_relations

    def forward(self, batch, return_details=False):
        local = local_recovery_batch(batch, self.minimal_features, self.keep_owner_relations)
        h, repeated, pools, features = _encode(self, local)
        fraction_logit = self.free_head(features).squeeze(-1)
        fraction = torch.sigmoid(fraction_logit)
        total_weight = local["weights"].sum(dim=1)
        recovery = fraction * total_weight
        raw = batch["immediate"] + recovery
        if not return_details:
            return raw.clamp_min(0.)
        logits, proposed = _node_auxiliary(self, local, h, repeated, pools)
        return dict(scores=raw.clamp_min(0.), raw_gain=raw, recovery=recovery,
                    occupancy=proposed, proposed=proposed, logits=logits,
                    recovery_fraction=fraction)

    def parameter_counts(self):
        result = super().parameter_counts()
        # Only the incompatibility relation is on the default deployment path.
        absent = (1, 2) if self.keep_owner_relations else (1, 2, 3)
        result["active_scoring"] -= sum(p.numel() for relations in self.relation_layers
                                         for index in absent for p in relations[index].parameters())
        # Three role pools and role counts are identically zero after masking.
        result["active_scoring"] -= self.hidden * (3 * self.hidden + 3)
        if self.minimal_features:
            empty_input_columns = 5 if self.keep_owner_relations else 6
            result["active_scoring"] -= empty_input_columns * self.hidden
        result["input"] = "minimal induced R graph" if self.minimal_features else "R-only messages; original R summaries"
        return result


class FreeGNNCliqueStats(CapacityRecoveryNet):
    """Shared full-response encoder + unrestricted clique-statistic bypass.

    Same typed/owner layers as the frozen free and capacity architectures. A
    12-statistic MLP adds unconstrained recovery credit; no capacity restriction
    is imposed on its final scalar. known_immediate=True injects q explicitly
    so the control does not need to relearn its exact additive arithmetic.
    """
    def __init__(self, hidden=48, layers=3, use_budget=True, use_agent_relay=True,
                 known_immediate=True):
        super().__init__(hidden, layers, use_budget, use_agent_relay, readout="unconstrained")
        self.known_immediate = known_immediate
        self.clique_stat_head = nn.Sequential(nn.Linear(len(CLIQUE_STATISTIC_NAMES), hidden),
                                               nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch, return_details=False):
        h, repeated, pools, features = _encode(self, batch)
        base = self.free_head(features).squeeze(-1)
        stats = clique_statistics(batch)
        credit = base + self.clique_stat_head(stats).squeeze(-1)
        raw = credit + batch["immediate"] if self.known_immediate else credit
        if not return_details:
            return raw.clamp_min(0.)
        logits, proposed = _node_auxiliary(self, batch, h, repeated, pools)
        return dict(scores=raw.clamp_min(0.), raw_gain=raw, recovery=raw - batch["immediate"],
                    occupancy=proposed, proposed=proposed, logits=logits,
                    clique_statistics=stats)

    def parameter_counts(self):
        result = super().parameter_counts()
        result["clique_statistics"] = len(CLIQUE_STATISTIC_NAMES)
        result["known_immediate"] = self.known_immediate
        return result


class AnalyticCliqueP1(nn.Module):
    """Zero-parameter observable selector; do not send to an optimizer."""
    def forward(self, batch, return_details=False):
        details = analytic_clique_details(batch)
        return details if return_details else details["scores"]

    def parameter_counts(self):
        return dict(stored=0, active_scoring=0, input="observable R weights and clique factors; known q")
