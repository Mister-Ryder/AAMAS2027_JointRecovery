"""Reserved fixed-scope, warm-anchored finite recovery prediction.

Source-only implementation until a new executed-label protocol is frozen.
The actual feasible warm reward is a semantic floor, not an oracle outcome.
Clique-partition bounds and floating-point predictions are not integer
recovery certificates. Every packing/cache/transfer operation belongs inside
the caller's measured decision boundary.
"""
from __future__ import annotations

from math import isfinite

import numpy as np
import torch
from torch import nn

from . import v4_factorized_model as factored
from . import v4_model as original
from .v4_budgeted_recovery import PriorityEvaluation, _check_recovery, objective
from .v4_model_fast import FastStaticPackingCache


VARIANTS = ("ResidualCapacity", "ResidualFreeOccupancy", "ResidualNoAux",
            "ResidualNoWarmMembership", "ResidualCheapSummary")
RESIDUAL_BIAS = -4.
FIXED_CAP = 256
BOUND_ABSOLUTE_TOLERANCE = 1e-8
BOUND_RELATIVE_TOLERANCE = 1e-10
UPPER_SUMMARY_INDEX = original.SUMMARY_FEATURES.index("partition_upper_bound")
WARM_VALUE_INDEX = factored.HEAD_FEATURES.index("normalized_actual_warm_value")
WARM_FRACTION_INDEX = factored.HEAD_FEATURES.index("actual_warm_fraction")


def _bounds(graph, views, scale, cache):
    """Host-side original-edge/warm/bound checks, with no execution targets."""
    lower = []; upper = []; adjustment = []
    for view in views:
        scope = view.request.scope
        warm = _check_recovery(graph, scope, view.warm_start)
        lo = float(objective(graph, warm)) / scale
        stats = cache.get(graph, scope, scale)[2]
        hi = float(stats[UPPER_SUMMARY_INDEX])
        tolerance = BOUND_ABSOLUTE_TOLERANCE + BOUND_RELATIVE_TOLERANCE * max(1., abs(lo), abs(hi))
        if not isfinite(lo) or not isfinite(hi) or lo < 0 or hi < 0 or lo > hi + tolerance:
            raise ValueError("Actual feasible warm reward is outside the observable clique-partition bound")
        # Only declared floating summation noise may close a negative gap.
        checked_hi = max(lo, hi)
        lower.append(lo); upper.append(checked_hi); adjustment.append(checked_hi - hi)
    return lower, upper, adjustment


def _fixed_views(views):
    views = tuple(views)
    if not views or any(view.request.scope.cap != FIXED_CAP or
                        len(view.request.scope.replacements) > FIXED_CAP for view in views):
        raise ValueError("The reserved alternative requires nonempty request coverage with fixed cap256")
    return views


def _attach_bounds(batch, graph, views, scale, cache, use_warm_membership, summary_only):
    lower, upper, adjustment = _bounds(graph, views, scale, cache)
    device = batch["context"].device
    batch["lower"] = torch.as_tensor(np.asarray(lower, np.float32), device=device)
    batch["upper"] = torch.as_tensor(np.asarray(upper, np.float32), device=device)
    batch["bound_roundoff_adjustment"] = tuple(adjustment)
    batch["summary_only"] = summary_only
    batch["use_warm_membership"] = bool(use_warm_membership)
    batch.setdefault("node_ids", tuple(tuple(view.request.scope.replacements) for view in views))
    if not use_warm_membership:
        batch["context"] = batch["context"].clone()
        batch["context"][:, WARM_FRACTION_INDEX] = 0.
        if not summary_only and batch["units"] is not None:
            batch["units"] = dict(batch["units"])
            batch["units"]["x"] = batch["units"]["x"].clone()
            batch["units"]["x"][:, 3] = 0.
            batch["units"]["warm"] = torch.zeros_like(batch["units"]["warm"])
    return batch


def pack_residual_v4(graph, views, state, scale=None, device="cpu", static_cache=None,
                     embedding_cache=None, use_warm_membership=True):
    """Same q-independent structural packing plus actual L and observable U.

    NoWarmMembership removes warm IDs/fraction, but keeps its actual reward L,
    both as a semantic floor and a scalar head input. Native warm starts are
    unchanged. Actual warm memberships remain conservative cache-version keys.
    """
    views = _fixed_views(views)
    scale = original.normalization_scale(graph) if scale is None else float(scale)
    cache = FastStaticPackingCache() if static_cache is None else static_cache
    batch = factored.pack_factorized_v4(graph, views, state, scale=scale, device=device,
        static_cache=cache, embedding_cache=embedding_cache, use_warm_start=True)
    return _attach_bounds(batch, graph, views, scale, cache, use_warm_membership, False)


def pack_residual_summary(graph, views, state, scale=None, device="cpu", static_cache=None,
                          use_warm_membership=True):
    """Summary control pays only for the observables it actually uses."""
    views = _fixed_views(views)
    scale = original.normalization_scale(graph) if scale is None else float(scale)
    cache = FastStaticPackingCache(summary_only=True) if static_cache is None else static_cache
    batch = factored.pack_factorized_summary(graph, views, state, scale=scale, device=device,
        static_cache=cache, use_warm_start=True)
    return _attach_bounds(batch, graph, views, scale, cache, use_warm_membership, True)


def _checked_interval(batch):
    lower, upper = batch["lower"], batch["upper"]
    if (lower.ndim != 1 or upper.shape != lower.shape or len(lower) != batch["request_count"]
            or not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(upper).all())
            or bool((lower < 0).any()) or bool((upper < lower).any())):
        raise ValueError("One finite nonnegative checked L/U interval per request is required")
    return lower, upper, (upper - lower).clamp_min(0.)


class ResidualRecoveryValueNet(factored.FactorizedRecoveryValueNet):
    """Matched capacity/free-occupancy h32/L2 bounded residual graph heads."""
    summary_only = False

    def __init__(self, hidden=32, layers=2, readout="capacity", use_warm_membership=True):
        # Both graph controls instantiate inherited heads in the same order.
        super().__init__(hidden=hidden, layers=layers, readout=readout, use_warm_start=True)
        self.use_warm_membership = bool(use_warm_membership)
        # The old free scalar head is unused: both controls use the new bounded
        # residual scalar head, and both use their matched occupancy logits.
        del self.free_head
        width = 2 * hidden + len(factored.HEAD_FEATURES) + len(original.SUMMARY_FEATURES) + 1
        self.residual_head = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.residual_head[2].bias, RESIDUAL_BIAS)

    def encode_units(self, units):
        if not self.use_warm_membership:
            units = dict(units); units["x"] = units["x"].clone()
            units["x"][:, 3] = 0.
        return super().encode_units(units)

    def forward(self, batch, return_details=False, embedding_cache=None):
        lower, upper, gap = _checked_interval(batch)
        h, pooled, weighted = self._representations(batch, embedding_cache)
        context = batch["context"]
        if not self.use_warm_membership:
            context = context.clone(); context[:, WARM_FRACTION_INDEX] = 0.
        assignment = batch["node_request"]; unit = batch["request_unit"]
        expanded = h[batch["unit_node"]]
        node_features = torch.cat((expanded, pooled[unit][assignment], context[assignment],
                                   batch["summary"][assignment]), dim=1)
        logits = self.occupancy_head(node_features).squeeze(-1)
        proposed = torch.sigmoid(logits)
        occupancy = original.capacity_project_sparse(proposed, batch) if self.readout == "capacity" else proposed
        occupancy_value = original._sum(occupancy * batch["weights"], assignment, batch["request_count"])
        normalized_occupancy = occupancy_value / upper.clamp_min(1e-12)
        features = torch.cat((pooled[unit], weighted[unit], context, batch["summary"],
                              normalized_occupancy[:, None]), dim=1)
        fraction = torch.sigmoid(self.residual_head(features).squeeze(-1))
        recovery = lower + gap * fraction
        raw = batch["immediate"] + recovery
        details = dict(scores=raw.clamp_min(0.), raw_gain=raw, recovery=recovery,
            lower=lower, upper=upper, residual_fraction=fraction, logits=logits,
            proposed=proposed, occupancy=occupancy, capacity_recovery=occupancy_value,
            normalized_occupancy_value=normalized_occupancy,
            encoded_units=len(batch["new_unit_indices"]))
        return details if return_details else details["scores"]

    def parameter_counts(self):
        count = sum(parameter.numel() for parameter in self.parameters())
        return dict(stored=count, active_scoring=count)


class ResidualSummaryValueNet(nn.Module):
    """No graph encoder: same actual warm floor, U, budget context and bias."""
    summary_only = True
    use_warm_start = True

    def __init__(self, hidden=32, use_warm_membership=True):
        super().__init__()
        if hidden < 1:
            raise ValueError("Positive summary head width required")
        self.hidden = hidden; self.use_warm_membership = bool(use_warm_membership)
        width = len(factored.HEAD_FEATURES) + len(original.SUMMARY_FEATURES)
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.head[2].bias, RESIDUAL_BIAS)

    def forward(self, batch, return_details=False):
        lower, upper, gap = _checked_interval(batch)
        context = batch["context"]
        if not self.use_warm_membership:
            context = context.clone(); context[:, WARM_FRACTION_INDEX] = 0.
        fraction = torch.sigmoid(self.head(torch.cat((context, batch["summary"]), dim=1)).squeeze(-1))
        recovery = lower + gap * fraction
        raw = batch["immediate"] + recovery
        details = dict(scores=raw.clamp_min(0.), raw_gain=raw, recovery=recovery,
                       lower=lower, upper=upper, residual_fraction=fraction, logits=None)
        return details if return_details else details["scores"]

    def parameter_counts(self):
        count = sum(parameter.numel() for parameter in self.parameters())
        return dict(stored=count, active_scoring=count)


def build_residual_model(variant):
    """Closed production family: five names, graph encoder always h32/L2."""
    if variant not in VARIANTS:
        raise ValueError("Unknown reserved residual variant")
    if variant == "ResidualCheapSummary":
        return ResidualSummaryValueNet(hidden=32)
    return ResidualRecoveryValueNet(hidden=32, layers=2,
        readout="free" if variant == "ResidualFreeOccupancy" else "capacity",
        use_warm_membership=variant != "ResidualNoWarmMembership")


def priority_callback_residual(model, scale=None, device="cpu"):
    """Paid scope/bounds/tensors/inference/cache lookups and native CPU output."""
    static = FastStaticPackingCache(summary_only=model.summary_only)
    embeddings = factored.DecisionEmbeddingCache()

    def evaluate(graph, views, state):
        if not state.spent_requests:
            static.reset(); embeddings.reset()
        embeddings.bind(graph, model)
        if model.summary_only:
            batch = pack_residual_summary(graph, views, state, scale=scale, device=device,
                static_cache=static, use_warm_membership=model.use_warm_membership)
        else:
            batch = pack_residual_v4(graph, views, state, scale=scale, device=device,
                static_cache=static, embedding_cache=embeddings,
                use_warm_membership=model.use_warm_membership)
        with torch.no_grad():
            prediction = model(batch) if model.summary_only else model(batch, embedding_cache=embeddings)
            native = prediction * batch["scale"]
        return PriorityEvaluation(tuple(float(value) for value in native.detach().cpu().tolist()),
            semantics="q-offset finite recovery inside common actual-warm/clique interval; point priority")

    return evaluate


def merge_residual_batches(batches):
    """Optional uncached offline merge; preserve all bound/request identities."""
    batches = tuple(batches)
    if not batches or len({batch["summary_only"] for batch in batches}) != 1 or len({
            batch["use_warm_membership"] for batch in batches}) != 1:
        raise ValueError("Nonempty consistently packed residual batches required")
    if not batches[0]["summary_only"]:
        result = factored.merge_factorized_batches(batches)
    else:
        names = ("context", "summary", "immediate", "best_gain", "scale")
        result = {name: torch.cat([batch[name] for batch in batches], dim=0) for name in names}
        ptr = []; offset = 0
        for batch in batches:
            ptr.append((batch["ptr"] if not ptr else batch["ptr"][1:]) + offset)
            offset += int(batch["ptr"][-1])
        result.update(ptr=torch.cat(ptr), request_count=sum(batch["request_count"] for batch in batches),
            node_ids=tuple(ids for batch in batches for ids in batch["node_ids"]),
            request_keys=tuple(keys for batch in batches for keys in batch.get("request_keys", ())),
            head_feature_names=factored.HEAD_FEATURES)
    result["lower"] = torch.cat([batch["lower"] for batch in batches])
    result["upper"] = torch.cat([batch["upper"] for batch in batches])
    result["bound_roundoff_adjustment"] = tuple(x for batch in batches for x in batch["bound_roundoff_adjustment"])
    result["summary_only"] = batches[0]["summary_only"]
    result["use_warm_membership"] = batches[0]["use_warm_membership"]
    return result


execution_targets = original.execution_targets
recovery_value_loss = original.recovery_value_loss
merge_targets = original.merge_v4_targets
