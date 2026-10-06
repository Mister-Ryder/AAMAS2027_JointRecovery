"""Final readout/sufficiency controls using identical observable recovery input.

No frozen implementation is changed. LocalRClique isolates P/D context under
the original capacity readout. CapacityWithScalarBypass nests the exact full
capacity model in a model with an unrestricted residual, retaining its entire
clique incidence, occupancy pathway and training-only mask head.
"""
from __future__ import annotations

import torch

from .v3_capacity import CapacityRecoveryNet, capacity_project
from .v3_controls import _encode, _node_auxiliary, local_recovery_batch


READOUT_CONFIG_NAMES = {
    "LocalRClique": "newcontrol-local-cap",
    "CapacityWithScalarBypass": "newcontrol-bypass",
}


def local_clique_batch(batch):
    """Exactly default LocalRecoveryGNN features/messages, original R cliques.

    Only the constrained readout requires reinstating the factor tensor. R
    order, rewards, known q, scale, and all budget/count channels are untouched.
    The factor tensor comes from pack_v3, never from a solver or outcome.
    """
    local = local_recovery_batch(batch, minimal_features=True, keep_owner_relations=False)
    local["cliques"] = batch["cliques"]
    sizes = batch["cliques"].sum(dim=2)
    valid = sizes > 0
    for i, diagnostic in enumerate(local["diagnostics"]):
        diagnostic.update(factors=int(valid[i].sum().item()),
                          factor_memberships=int(sizes[i].sum().item()),
                          max_clique=int(sizes[i].max().item()))
    return local


class LocalRClique(CapacityRecoveryNet):
    """Minimal R-only encoder + unchanged weighted occupancy/clique readout."""
    readout_config_name = READOUT_CONFIG_NAMES["LocalRClique"]

    def __init__(self, hidden=48, layers=3, use_budget=True, use_agent_relay=False):
        if use_agent_relay:
            raise ValueError("LocalRClique strictly disables owner relay to match default LocalR")
        super().__init__(hidden, layers, use_budget=use_budget,
                         use_agent_relay=False, readout="occupancy")

    def forward(self, batch, return_details=False):
        return super().forward(local_clique_batch(batch), return_details=return_details)

    def parameter_counts(self):
        result = super().parameter_counts()
        # Default LocalR features have six identically zero input columns;
        # P/D role mean contexts are identically zero, as are typed 1/2/3 edges.
        result["active_scoring"] -= 6 * self.hidden
        result["active_scoring"] -= 2 * self.hidden * self.hidden
        result["active_scoring"] -= sum(p.numel() for relations in self.relation_layers
                                         for t in (1, 2, 3) for p in relations[t].parameters())
        result["input"] = "minimal induced R graph; same observable R clique incidence; known q"
        result["readout_config"] = self.readout_config_name
        return result


class CapacityWithScalarBypass(CapacityRecoveryNet):
    """Original full capacity pathway plus its existing unrestricted free head.

    The residual uses original role means/counts and budget channels. All full
    clique information remains on the original occupancy pathway. The original
    no-bypass model is the nested special case with final residual layer zero.
    Initializing with the same seed gives a bitwise-identical original pathway
    and score on the same device; no extra parameters are stored, but the free
    head is now active. Zeroing the residual does not consume random numbers.
    """
    readout_config_name = READOUT_CONFIG_NAMES["CapacityWithScalarBypass"]

    def __init__(self, hidden=48, layers=3, use_budget=True, use_agent_relay=True):
        super().__init__(hidden, layers, use_budget=use_budget,
                         use_agent_relay=use_agent_relay, readout="occupancy")
        self.reset_scalar_bypass()

    def reset_scalar_bypass(self):
        """Use only at initialization/conversion, never after loading trained BP."""
        with torch.no_grad():
            self.free_head[-1].weight.zero_()
            self.free_head[-1].bias.zero_()

    def forward(self, batch, return_details=False):
        # _encode uses the exact original encoder equations and pools; no
        # fused linear maps or altered accumulation order is introduced.
        h, repeated, pools, features = _encode(self, batch)
        logits, proposed = _node_auxiliary(self, batch, h, repeated, pools)
        occupancy = capacity_project(proposed, batch["cliques"])
        recovery = (occupancy * batch["weights"]).sum(dim=1)
        capacity_raw = batch["immediate"] + recovery
        residual = self.free_head(features).squeeze(-1)
        raw = capacity_raw + residual
        if not return_details:
            return raw.clamp_min(0.)
        # recovery is deliberately the capacity-path recovery, not raw-q;
        # expose the residual separately to keep auxiliary semantics explicit.
        return dict(scores=raw.clamp_min(0.), raw_gain=raw, recovery=recovery,
                    occupancy=occupancy, proposed=proposed, logits=logits,
                    capacity_raw_gain=capacity_raw, scalar_residual=residual)

    def parameter_counts(self):
        stored = sum(p.numel() for p in self.parameters())
        free = sum(p.numel() for p in self.free_head.parameters())
        active = stored
        if not self.use_agent_relay:
            active -= sum(p.numel() for p in self.agent_layers.parameters())
        return dict(stored=stored, active_scoring=active,
                    scalar_bypass_parameters=free,
                    input="original full response and original clique incidence",
                    readout_config=self.readout_config_name)
