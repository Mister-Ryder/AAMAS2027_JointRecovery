"""V4 source-only fitting API over sealed, actual warm-aligned labels.

Importing this module never loads a dataset, fits a model, runs a solver or
constructs confirmation graphs. Offline conditional label regret is not an
online full-deadline policy result. The explicit fit API requires bound inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import copy
import hashlib
import json
from math import isfinite
from numbers import Integral
from pathlib import Path
import random
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from joint_recovery.core import Graph, is_feasible
from joint_recovery.v4_budgeted_recovery import (
    NativeIntegerGraph, ControllerState, Request, RequestView, Workpoint,
    _check_recovery, objective)
from joint_recovery.v4_model import normalization_scale
try:
    from . import v4_residual_training_data as collection
    from .v4_neighborhoods import coordination_cells
    from .v4_residual_common import ExecutedWarmScopeCache as NeighborhoodScopeCache
except ImportError:
    from experiments import v4_residual_training_data as collection
    from experiments.v4_neighborhoods import coordination_cells
    from experiments.v4_residual_common import ExecutedWarmScopeCache as NeighborhoodScopeCache


VARIANTS = ("ResidualCapacity", "ResidualFreeOccupancy", "ResidualNoAux",
            "ResidualNoWarmMembership", "ResidualCheapSummary")
DIAGNOSTIC_VARIANT = "ReservedRepeatForbidden"
FIT_SEEDS = (17, 29, 43)
GROUP_NAMES = ("initial_executed_greedy", "round_robin_actual_history")
CALIBRATION_SHA256 = "495dab54ad7ca75e8a543338eb71da1049e20afdd35ad18825669d37d54c5bec"
MODEL_SHA256 = "d6c974b3e57b1880bbb0d24c6dceca8f343d303c47e74736f3864c8da054a750"
TIME_CLOSURE_TOLERANCE_SECONDS = 1e-6
FIT_SOURCE_PATHS = ("experiments/v4_residual_fit.py","experiments/v4_residual_training_data.py","experiments/v4_neighborhoods.py",
    "experiments/v4_residual_common.py","src/joint_recovery/v4_residual_model.py",
    "src/joint_recovery/__init__.py","src/joint_recovery/core.py","src/joint_recovery/v4_budgeted_recovery.py",
    "src/joint_recovery/v4_factors.py","src/joint_recovery/v4_factors_fast.py","src/joint_recovery/v4_model.py",
    "src/joint_recovery/v4_model_fast.py","src/joint_recovery/v4_factorized_model.py")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _finite(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("Finite numeric %s required" % name)
    if minimum is not None and value < minimum:
        raise ValueError("%s is below its declared minimum" % name)
    return value


def _equal(actual, expected, name, native=False):
    if native:
        if isinstance(actual, bool) or not isinstance(actual, Integral) or actual != expected:
            raise ValueError("Exact native %s differs" % name)
    elif not np.isclose(_finite(actual, name), expected, rtol=1e-12, atol=1e-9):
        raise ValueError("%s differs from original graph replay" % name)


def _ids(values, graph, name, sorted_required=False):
    if not isinstance(values, (list, tuple)):
        raise ValueError("Explicit membership array required: " + name)
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0 or v >= graph.n for v in values):
        raise ValueError("Exact in-range integer IDs required: " + name)
    result = tuple(map(int, values))
    if len(result) != len(set(result)) or (sorted_required and result != tuple(sorted(result))):
        raise ValueError("Distinct canonical membership required: " + name)
    return result


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError("Explicit boolean required: " + name)
    return value


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def constant(value): raise ValueError("Non-finite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=constant)


@dataclass(frozen=True)
class FitConfig:
    variant: str
    fit_seed: int
    epochs: int = 40
    batch_graphs: int = 4
    learning_rate: float = .001
    auxiliary_weight: float = .05
    ranking_weight: float = 1.
    temperature: float = .2
    hidden: int = 32
    layers: int = 2

    def __post_init__(self):
        if self.variant not in VARIANTS or self.fit_seed not in FIT_SEEDS:
            raise ValueError("Predeclared variant and fit seed required")
        if (self.epochs, self.batch_graphs, self.learning_rate, self.ranking_weight,
                self.temperature, self.hidden, self.layers) != (40, 4, .001, 1., .2, 32, 2):
            raise ValueError("The first fitting source freezes 40 epochs/4 graphs/h32-L2/Adam1e-3/rank1/temp.2")
        if self.auxiliary_weight != .05:
            raise ValueError("The source protocol fixes mask coefficient .05; NoAux only disables BCE")

    @property
    def effective_auxiliary_weight(self):
        return 0. if self.variant in ("ResidualNoAux", "ResidualCheapSummary") else self.auxiliary_weight


def state_loss(details, targets, batch, *, auxiliary_weight=.05, ranking_weight=1., temperature=.2):
    """One state's means, with no graph weight or hidden request-count weight.

    All packed rows must be available. Invalid executions remain ranking rows
    with admitted gain zero; late valid outputs retain signed/mask regression.
    Prediction differences are UNCLAMPED signed values. Desired pairs compare
    actual positive admitted gain above the known current best. Every unequal
    unordered pair contributes exactly once. Empty-state weight remains zero.
    """
    raw = details["raw_gain"]
    if raw.ndim != 1 or len(raw) != batch["request_count"]:
        raise ValueError("One raw signed prediction per available request required")
    if temperature <= 0 or auxiliary_weight < 0 or ranking_weight < 0:
        raise ValueError("Positive temperature/nonnegative loss coefficients required")
    if not bool(torch.isfinite(raw).all()): raise ValueError("Non-finite model prediction")
    valid = targets["valid"]
    if valid.dtype != torch.bool or valid.shape != raw.shape:
        raise ValueError("One explicit raw-supervision validity flag per request required")
    if any(targets[k].shape != raw.shape for k in ("raw_gain", "gains")):
        raise ValueError("Signed/admitted targets must match request order")
    if not bool(torch.isfinite(targets["raw_gain"]).all()) or not bool(torch.isfinite(targets["gains"]).all()):
        raise ValueError("Finite normalized targets required")
    if batch["best_gain"].shape != raw.shape or not bool(torch.isfinite(batch["best_gain"]).all()):
        raise ValueError("Explicit finite known-best context required")
    zero = raw.sum()*0.
    regression = F.smooth_l1_loss(raw[valid], targets["raw_gain"][valid]) if bool(valid.any()) else zero
    actual_marginal = (targets["gains"]-batch["best_gain"]).clamp_min(0.)
    desired = actual_marginal[:,None] > actual_marginal[None,:]
    ranking = (F.softplus(-(raw[:,None]-raw[None,:])[desired]/temperature).mean()
               if bool(desired.any()) else zero)
    auxiliary = zero
    if auxiliary_weight:
        logits = details.get("logits")
        if logits is None: raise ValueError("Mask-supervised variant requires occupancy logits")
        assignment = batch["node_request"]
        if assignment.dtype != torch.int64 or logits.ndim != 1 or len(logits) != len(assignment):
            raise ValueError("One occupancy logit per request-local node required")
        if len(assignment) and (int(assignment.min()) < 0 or int(assignment.max()) >= len(raw)):
            raise ValueError("Node/request incidence is outside this state")
        mask = targets["recovery_mask"]
        if mask.shape != logits.shape or not bool(torch.isfinite(logits).all()) or not bool(((mask==0)|(mask==1)).all()):
            raise ValueError("Finite logits and exact returned binary mask required")
        if len(assignment):
            losses = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
            sums = raw.new_zeros(len(raw)).index_add(0, assignment, losses)
            counts = raw.new_zeros(len(raw)).index_add(0, assignment, torch.ones_like(losses))
            mask_valid = valid & (counts > 0)
            if bool(mask_valid.any()): auxiliary = (sums[mask_valid]/counts[mask_valid]).mean()
    total = regression + ranking_weight*ranking + auxiliary_weight*auxiliary
    return dict(total=total, regression=regression, ranking=ranking, auxiliary=auxiliary,
                unequal_pairs=int(desired.sum().item()), valid_raw=int(valid.sum().item()))


def weighted_graph_loss(state_components, graph_weight):
    """Two fixed half states; graph/domain weight applied ONCE after means."""
    if len(state_components) != 2 or not isfinite(graph_weight) or graph_weight <= 0:
        raise ValueError("Two half-weighted states and a fixed positive graph weight required")
    return {key: graph_weight*.5*(state_components[0][key]+state_components[1][key])
            for key in ("total", "regression", "ranking", "auxiliary")}


def zero_state(model):
    """No available requests: zero loss with a graph-safe autograd connection."""
    zero = next(model.parameters()).sum()*0.
    return dict(total=zero, regression=zero, ranking=zero, auxiliary=zero, unequal_pairs=0, valid_raw=0)


def offline_state_metrics(raw, admitted, best_gain, expected_seconds, incumbent_value):
    """Conditional one-query allocation from sealed labels; NOT policy quality."""
    raw, admitted = np.asarray(raw, float), np.asarray(admitted, float)
    costs = np.asarray(expected_seconds, float)
    if (raw.ndim != 1 or admitted.shape != raw.shape or costs.shape != raw.shape or
            not np.isfinite(raw).all() or not np.isfinite(admitted).all() or
            not np.isfinite(costs).all() or np.any(costs <= 0)):
        raise ValueError("Finite native predictions/admitted labels and shared positive costs required")
    _finite(best_gain, "known best", 0); _finite(incumbent_value, "incumbent", 0)
    desired = np.maximum(admitted-best_gain, 0.)
    priorities = np.maximum(raw-best_gain, 0.)/costs
    index = int(np.argmax(priorities)) if len(raw) and priorities.max() > 0 else None
    selected = best_gain if index is None else max(best_gain, admitted[index])
    oracle = max(best_gain, float(admitted.max()) if len(admitted) else best_gain)
    differences = desired[:,None]-desired[None,:]
    pairs = differences > 0
    prediction = raw[:,None]-raw[None,:]
    correct = float(np.mean((prediction[pairs]>0)+.5*(prediction[pairs]==0))) if pairs.any() else None
    return dict(relative_conditional_allocation_regret=(oracle-selected)/max(1., incumbent_value),
        native_conditional_regret=oracle-selected, oracle_marginal=oracle-best_gain,
        selected_index=index, pair_accuracy=correct, unequal_pairs=int(pairs.sum()),
        available_requests=len(raw), scope="offline_same_state_labels_not_full_policy")


@dataclass(frozen=True)
class LoadedState:
    record: dict
    context: ControllerState
    views: tuple
    rows: tuple
    state_sha256: str


@dataclass(frozen=True)
class LoadedCase:
    case_id: str
    spec: dict
    graph: object
    selected: frozenset
    cliques: tuple
    states: tuple
    graph_weight: float
    status: str
    file_bindings: dict


def graph_from_npz(path):
    """Exact collector schema; no object arrays, ID coercion or dtype loss."""
    with np.load(path, allow_pickle=False) as data:
        names = {"weights", "agents", "edges", "selected", "cliques_flat", "cliques_ptr"}
        if set(data.files) != names: raise ValueError("Observable NPZ schema differs")
        arrays = {key: np.array(data[key], copy=True) for key in names}
    weights, agents, edges = (arrays[k] for k in ("weights", "agents", "edges"))
    if (weights.ndim != 1 or weights.dtype.kind not in "iuf" or
            not np.isfinite(weights).all() or np.any(weights <= 0) or
            agents.shape != weights.shape or agents.dtype.kind not in "iu" or
            edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu"):
        raise ValueError("Native finite weights and exact agent/edge arrays required")
    n = len(weights); adjacency = [set() for _ in range(n)]
    canonical = []
    for u, v in edges.tolist():
        if not 0 <= u < v < n: raise ValueError("Edges must be canonical in-range undirected pairs")
        if v in adjacency[u]: raise ValueError("Duplicate original edge")
        adjacency[u].add(v); adjacency[v].add(u); canonical.append((u,v))
    if canonical != sorted(canonical): raise ValueError("Original edges must retain canonical ordering")
    cls = NativeIntegerGraph if weights.dtype.kind in "iu" else Graph
    graph = cls(weights, agents, tuple(map(frozenset, adjacency)), "sealed-fresh-training")
    selected_array = arrays["selected"]
    if selected_array.ndim != 1 or selected_array.dtype.kind not in "iu":
        raise ValueError("Native original membership vector required")
    selected = frozenset(_ids(selected_array.tolist(), graph, "original", sorted_required=True))
    if not is_feasible(graph, selected): raise ValueError("Original incumbent is infeasible")
    flat, ptr = arrays["cliques_flat"], arrays["cliques_ptr"]
    if (flat.ndim != 1 or ptr.ndim != 1 or flat.dtype.kind not in "iu" or ptr.dtype.kind not in "iu"
            or not len(ptr) or ptr[0] != 0 or ptr[-1] != len(flat) or np.any(np.diff(ptr.astype(np.int64)) < 0)):
        raise ValueError("Exact supplied-clique offsets required")
    cliques = tuple(_ids(flat[int(a):int(b)].tolist(), graph, "resource clique", sorted_required=True)
                    for a,b in zip(ptr[:-1], ptr[1:]))
    if len(set(cliques)) != len(cliques) or any(len(q)<2 for q in cliques):
        raise ValueError("Distinct nontrivial genuine supplied cliques required")
    for factor in cliques:
        if any(v not in graph.adjacency[u] for i,u in enumerate(factor) for v in factor[i+1:]):
            raise ValueError("Resource cover contains a non-clique")
    return graph, selected, cliques


def _state(record, graph, selected, config):
    if set(record) != {"name", "original", "best_selected", "initial_value", "best_gain",
                       "warm_by_action", "spent_requests", "prefix_elapsed_seconds", "remaining_seconds"}:
        raise ValueError("Snapshot record schema differs")
    if frozenset(_ids(record["original"], graph, "snapshot original", True)) != selected:
        raise ValueError("State belongs to a different original incumbent")
    native = graph.weights.dtype.kind in "iu"
    initial = objective(graph, selected)
    _equal(record["initial_value"], initial, "initial value", native)
    best = frozenset(_ids(record["best_selected"], graph, "snapshot best", True))
    if not is_feasible(graph, best): raise ValueError("Snapshot incumbent is not materialized feasible membership")
    _equal(record["best_gain"], max(0,objective(graph,best)-initial), "known best gain", native)
    elapsed = _finite(record["prefix_elapsed_seconds"], "paid prefix elapsed", 0)
    remaining = _finite(record["remaining_seconds"], "remaining context", 0)
    # deadline-now and D-(now-started) can differ after IEEE subtraction at
    # large absolute uptime. This bounded replay tolerance never admits a row.
    expected_remaining = max(0.,config.deadline_seconds-elapsed)
    if abs(remaining-expected_remaining) > TIME_CLOSURE_TOLERANCE_SECONDS:
        raise ValueError("Remaining time closure exceeds the bounded 1e-6 second tolerance")
    warm = {}
    for item in record["warm_by_action"]:
        index = item["action_index"]
        if isinstance(index,bool) or not isinstance(index,Integral) or index in warm:
            raise ValueError("Exact unique warm action indices required")
        warm[int(index)] = frozenset(_ids(item["recovered"], graph, "known warm", True))
    spent = []
    for key in record["spent_requests"]:
        if (not isinstance(key,list) or len(key)!=3 or any(isinstance(k,bool) or not isinstance(k,Integral) for k in key[:2])
                or not isinstance(key[2],str)):
            raise ValueError("Canonical spent request keys required")
        spent.append(tuple(key))
    if spent != sorted(set(spent)): raise ValueError("Spent ledger contains duplicate/noncanonical keys")
    context = ControllerState(initial, record["best_gain"], remaining, tuple(spent))
    return context, best, warm


def _row(row, request, graph, original, warm, context, config, *, state_sha=None, history=False):
    scope = request.scope; native = graph.weights.dtype.kind in "iu"
    for key in ("action_index","cap"):
        if isinstance(row.get(key),bool) or not isinstance(row.get(key),Integral):
            raise ValueError("Request identity must use exact integer fields")
    if (not isinstance(row.get("request_key"),list) or len(row["request_key"])!=3 or
            any(isinstance(k,bool) or not isinstance(k,Integral) for k in row["request_key"][:2]) or
            not isinstance(row["request_key"][2],str)):
        raise ValueError("Exact request key required")
    for key in ("scope_nodes","base","displaced","action_inserts","action_extra_releases","actual_warm_start"):
        _ids(row[key],graph,key,sorted_required=key!="scope_nodes")
    fields = dict(request_key=list(request.key), action_index=scope.action_index, cap=scope.cap,
        workpoint=request.workpoint.name, native_seconds=request.workpoint.amount,
        expected_seconds=request.workpoint.expected_seconds, scope_nodes=list(scope.replacements),
        base=sorted(scope.base), displaced=sorted(scope.displaced), action_inserts=list(scope.inserts),
        action_extra_releases=sorted(scope.displaced-frozenset(
            v for c in scope.inserts for v in graph.adjacency[c]&original)),
        actual_warm_start=sorted(warm & frozenset(scope.replacements)))
    for key, expected in fields.items():
        if row.get(key) != expected: raise ValueError("Request replay differs: " + key)
    _equal(row["immediate_gain"], scope.immediate_gain, "fixed q", native)
    if state_sha is not None and row.get("state_sha256") != state_sha:
        raise ValueError("Alternative no longer clones the common snapshot")
    _equal(row["context_best_gain"], context.best_gain, "row current best", native)
    _equal(row["context_remaining_seconds"], context.remaining_seconds, "row remaining")
    if row["context_spent_requests"] != [list(k) for k in context.spent_requests]:
        raise ValueError("Alternative spent context differs")
    launched, spent, available, valid, raw_valid, timely = (_boolean(row[k], k) for k in (
        "launched", "spent", "available", "membership_valid", "raw_supervision_valid", "on_time"))
    if spent != launched or (launched and not available): raise ValueError("Executed/failure calls must be available and spent")
    failure = row["failure"]
    if failure is not None and (not isinstance(failure,dict) or not isinstance(failure.get("type"),str)):
        raise ValueError("Malformed failure ledger")
    if raw_valid != (valid and failure is None): raise ValueError("Raw target validity differs from full executor receipt")
    if launched and not valid and failure is None: raise ValueError("Invalid executed membership needs an explicit failure")
    if valid:
        actual = frozenset(_ids(row["returned_recovery"], graph, "actual return", True))
        _check_recovery(graph,scope,actual)
        signed = objective(graph,scope.base|actual)-context.incumbent_value
        _equal(signed, scope.immediate_gain+objective(graph,actual), "q+recovery identity", native)
        for key, expected in (("signed_return_delta",signed),("returned_gain",max(0,signed)),
                ("candidate_value",objective(graph,scope.base|actual)),("recovery_value",objective(graph,actual)),
                ("previous_warm_value",objective(graph,warm))):
            _equal(row[key],expected,key,native)
    else:
        actual = None
        if row["returned_recovery"] is not None or row["signed_return_delta"] is not None or row["returned_gain"] != 0:
            raise ValueError("Failed membership cannot become a raw execution target")
    if row["elapsed_seconds"] is None:
        if timely or raw_valid: raise ValueError("Missing validation timestamp cannot admit or supervise an output")
    else:
        elapsed = _finite(row["elapsed_seconds"], "executor plus validation elapsed", 0)
        remaining = _finite(row.get("actual_remaining_before_call",context.remaining_seconds), "actual remaining", 0)
        if remaining > context.remaining_seconds + 1e-9: raise ValueError("Request obtained future budget")
        if launched and (request.workpoint.amount>remaining or request.workpoint.expected_seconds>remaining):
            raise ValueError("Actual executor was launched outside frozen startability")
        if not history: _equal(remaining,context.remaining_seconds,"cloned request remaining")
        started=_finite(row['call_started_timestamp'],'actual call start timestamp')
        ready=_finite(row['validation_ready_timestamp'],'actual validation timestamp',started)
        if elapsed!=ready-started:
            raise ValueError('Saved elapsed does not use the exact actual ready/start timestamps')
        absolute=row['absolute_history_deadline']
        if history:
            absolute=_finite(absolute,'actual absolute history deadline')
        elif absolute is not None:
            raise ValueError('A cloned alternative cannot obtain a history absolute deadline')
        if timely != (failure is None and ready-started < remaining and
                (absolute is None or ready<absolute)):
            raise ValueError("Strict actual validation-ready deadline admission differs")
    if not launched:
        expected_available = (request.key not in context.spent_requests and context.remaining_seconds>0 and
            request.workpoint.amount<=context.remaining_seconds and request.workpoint.expected_seconds<=context.remaining_seconds)
        # A missing before-launch timestamp is an available, unexecuted failure.
        if available and row["status"] != "timestamp_before_launch_failed":
            raise ValueError("Available request silently dropped without execution")
        if not history and available != expected_available: raise ValueError("Startability differs from shared calibrated costs")
        if timely or valid or spent: raise ValueError("Unexecuted call contains a future output")
    if not isinstance(row["diagnostics"],dict): raise ValueError("Explicit serializable executor diagnostics required")
    if raw_valid and row["diagnostics"].get("binary_sha256") != collection.CHILS_SHA256:
        raise ValueError("Pinned actual CHILS metadata missing from a raw supervised output")
    _equal(row["admitted_gain"],row["returned_gain"] if timely else 0,"admitted gain",native)
    return actual


def replay_case(graph, selected, cliques, spec, labels, config):
    """Independent original-edge/base/warm/state/label replay; no solver call."""
    weight = collection.split_graph_weight(spec)
    if (labels.get("executed_common_greedy_prefix") is not True or labels.get("fixed_scope_cap") != 256 or labels["status"] != "actual_finite_labels_not_learning_or_policy_evidence" or
            labels["graph_sha256"] != collection.graph_identity(graph) or
            labels["calibration_sha256"] != config.calibration_sha256 or
            labels["deadline_seconds"] != config.deadline_seconds or
            labels["executor_sha256"] != collection.CHILS_SHA256 or
            labels["executor_receipt_required"] is not True or labels["dry_guard_mocks_only"] is not False or
            labels["history_max_queries"] != 1 or labels["cells_expected"] != 11 or
            labels["confirmation_graphs_generated"] != 0 or labels["learning_advantage_established"] is not False):
        raise ValueError("Labels are not bound production finite-label records")
    native = graph.weights.dtype.kind in "iu"; initial = objective(graph,selected)
    preparation_started=_finite(labels['preparation_started_timestamp'],'actual preparation start')
    preparation_deadline=_finite(labels['preparation_deadline_timestamp'],'actual preparation deadline',preparation_started)
    if preparation_deadline!=preparation_started+config.deadline_seconds:
        raise ValueError('Preparation deadline does not use the actual absolute start plus D')
    cells = coordination_cells(graph,selected,seed=spec["seed"]+1000)
    cache = NeighborhoodScopeCache(graph,selected,cliques)
    if len(labels["coverage"]) != 11: raise ValueError("Fixed eleven-cell coverage is incomplete")
    scopes = []; requests = []; warm = {}; seen = {}; scope_status = {}
    for index, ((action, expected), actual) in enumerate(zip(cells,labels["coverage"])):
        for key,value in dict(expected,cell_index=index).items():
            if actual.get(key) != value: raise ValueError("Observable fixed action coverage differs: "+key)
        if action is None: continue
        if actual["inserts"] != list(action.inserts) or actual["releases"] != list(action.releases):
            raise ValueError("Action C/E identity changed")
        warm[index] = frozenset(); seen[index] = {}; described = actual["scopes"]
        if len(described) != 1: raise ValueError("FixedR256 coverage differs")
        for cap, item in zip(config.caps,described):
            scope = cache.scope(index,action,cap)
            if scope.replacements in seen[index]:
                if item != dict(cap=cap,status="actual_scope_alias",alias_cap=seen[index][scope.replacements]):
                    raise ValueError("Actual R alias does not use its lowest cap")
                continue
            seen[index][scope.replacements] = cap
            if (item["cap"] != cap or item["nodes"] != list(scope.replacements) or
                    item["base"] != sorted(scope.base) or item["displaced"] != sorted(scope.displaced) or
                    item["status"] not in ("prepared","prepared_after_deadline")):
                raise ValueError("Fixed base or actual nested R differs from original graph")
            scopes.append(scope); requests.extend(Request(scope,p) for p in config.workpoints)
            scope_status[(index,cap)]=item["status"]
    known = labels["known_warm_observations"]
    if len(known) != len(scopes): raise ValueError("Known warm observation coverage differs")
    best, best_value, last = selected, initial, 0.
    for scope, row in zip(scopes,known):
        actual = frozenset(_ids(row["recovered"],graph,"paid executed common greedy warm",True))
        if (row["action_index"],row["cap"],actual) != (scope.action_index,scope.cap,cache.initial_known_warm(scope)):
            raise ValueError("Common warm differs from graph-only executed preparatory recovery")
        _check_recovery(graph,scope,actual); value = objective(graph,scope.base|actual)
        _equal(row["signed_delta"],value-initial,"known-warm signed objective",native)
        ready = _finite(row["validation_ready_elapsed"],"known-warm ready elapsed",last); last = ready
        absolute_ready=_finite(row['validation_ready_timestamp'],'actual known-warm validation timestamp',preparation_started)
        if ready!=absolute_ready-preparation_started:
            raise ValueError('Known-warm elapsed does not use actual absolute timestamps')
        if _boolean(row["on_time"],"known-warm admission") != (absolute_ready<preparation_deadline):
            raise ValueError("Known warm is admitted after its deadline")
        if scope_status[(scope.action_index,scope.cap)] != ("prepared" if row["on_time"] else "prepared_after_deadline"):
            raise ValueError("Scope preparation admission differs from actual known-warm timestamp")
        if row["on_time"]:
            if objective(graph,actual)>objective(graph,warm[scope.action_index]): warm[scope.action_index]=actual
            if value>best_value: best,best_value=scope.base|actual,value
    groups = labels["groups"]
    if len(groups)!=2: raise ValueError("Both fixed half-state groups are required")
    first = groups[0]["state"]
    context0,best0,warm0 = _state(first,graph,selected,config)
    if (first["name"]!=GROUP_NAMES[0] or first["prefix_elapsed_seconds"]<last or context0.spent_requests or
            best0!=best or warm0!=warm): raise ValueError("Initial state differs from paid actual known warm")
    history = labels["history"]; ordered=[r for r in requests if r.scope.cap==256 and r.workpoint.amount==.05]
    if len(history)!=len(ordered): raise ValueError("Planned fixed RR history coverage differs")
    spent=[]; launched=0; history_remaining_ceiling=context0.remaining_seconds
    for request,row in zip(ordered,history):
        if row["request_key"]!=list(request.key): raise ValueError("History order changed by outcome")
        if launched:
            if row.get("status")!="outside_fixed_history_prefix" or any(row[k] for k in ("launched","spent","available","on_time")):
                raise ValueError("More than one actual history request or fabricated tail")
            continue
        remaining=_finite(row["context_remaining_seconds"],"history context remaining",0)
        if remaining>context0.remaining_seconds+1e-9: raise ValueError("History has an independent extra budget")
        context=ControllerState(initial,max(0,best_value-initial),remaining,tuple(spent))
        if row.get('absolute_history_deadline')!=preparation_deadline:
            raise ValueError('History deadline is not the actual preparation deadline')
        actual=_row(row,request,graph,selected,warm[request.scope.action_index],context,config,history=True)
        if row["launched"]: launched+=1; spent.append(request.key)
        if row["launched"] and row["elapsed_seconds"] is not None:
            history_remaining_ceiling=max(0.,row["actual_remaining_before_call"]-row["elapsed_seconds"])
        if row["on_time"]:
            if objective(graph,actual)>objective(graph,warm[request.scope.action_index]): warm[request.scope.action_index]=actual
            value=objective(graph,request.scope.base|actual)
            if value>best_value: best,best_value=request.scope.base|actual,value
    second=groups[1]["state"]; context1,best1,warm1=_state(second,graph,selected,config)
    if (second["name"]!=GROUP_NAMES[1] or second["prefix_elapsed_seconds"]<first["prefix_elapsed_seconds"] or
            context1.remaining_seconds>history_remaining_ceiling+1e-9 or
            context1.spent_requests!=tuple(sorted(spent)) or best1!=best or warm1!=warm or
            labels["history_actual_calls"]!=launched): raise ValueError("History snapshot is not derived from actual valid timely execution")
    loaded=[]; calls=0
    for group,context,known_warm in ((groups[0],context0,warm0),(groups[1],context1,warm1)):
        digest=sha256_json(group["state"])
        if group["state_sha256"]!=digest: raise ValueError("Snapshot SHA binding differs")
        if group["state_loss_weight"]!=.5: raise ValueError("Both states must retain fixed half weight")
        _equal(group["graph_loss_weight"],weight,"fixed graph/domain weight")
        rows=group["alternatives"]
        if len(rows)!=len(requests): raise ValueError("Counterfactual alternative coverage differs")
        for request,row in zip(requests,rows):
            _row(row,request,graph,selected,known_warm[request.scope.action_index],context,config,state_sha=digest)
            calls+=row["launched"]
        # Weight metadata is checked, then never multiplied a second time.
        expected_rows=[dict(row) for row in rows]; collection._loss_weights(expected_rows,weight)
        for row,expected in zip(rows,expected_rows):
            for key in ("regression_request_weight","auxiliary_request_weight","ranking_request_weight"):
                _equal(row[key],expected[key],key)
        indices=[i for i,row in enumerate(rows) if row["available"]]
        views=tuple(RequestView(requests[i],known_warm[requests[i].scope.action_index]&frozenset(requests[i].scope.replacements)) for i in indices)
        loaded.append(LoadedState(group["state"],context,views,tuple(rows[i] for i in indices),digest))
    if labels["alternative_actual_calls"]!=calls: raise ValueError("Actual label call count differs from spent ledger")
    return tuple(loaded),weight


@dataclass(frozen=True)
class BoundDataset:
    split: str
    cases: tuple
    protocol_sha256: str
    completion_sha256: str
    root: str
    source_capsule: dict


def _bound_file(root, relative, binding):
    root = Path(root).resolve(); relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts: raise ValueError("Artifact path escapes its declared root")
    target = root/relative
    if target.is_symlink() or root not in target.resolve().parents or not target.is_file():
        raise ValueError("Artifact is missing, redirected or outside its root")
    if set(binding) != {"sha256","bytes"} or isinstance(binding["bytes"],bool) or not isinstance(binding["bytes"],int):
        raise ValueError("Exact file byte/hash receipt required")
    if target.stat().st_size != binding["bytes"] or sha256_file(target) != binding["sha256"]:
        raise ValueError("Sealed artifact bytes differ")
    return target


def _check_source_capsule(manifest):
    root = Path(__file__).resolve().parents[1]
    required = {"experiments/v4_residual_collect_training.py","experiments/v4_residual_training_data.py","experiments/v4_neighborhoods.py",
        "experiments/v4_backends.py","experiments/v3_domains.py","experiments/v3_pilot.py",
        "experiments/v3_solvers.py","experiments/v3_published_baselines.py","src/joint_recovery/__init__.py",
        "src/joint_recovery/core.py","src/joint_recovery/efficient_core.py","src/joint_recovery/generators.py",
        "src/joint_recovery/v4_budgeted_recovery.py","src/joint_recovery/v4_factors.py","experiments/v4_residual_common.py"}
    records = manifest["files"]; paths = [r["path"] for r in records]
    if len(paths)!=len(set(paths)) or not required<=set(paths): raise ValueError("Collection scientific source closure is incomplete")
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or root not in (root/relative).resolve().parents:
            raise ValueError("Collection source path escapes its immutable capsule")
        if sha256_file(root/relative)!=record["sha256"]: raise ValueError("Current replay source differs from sealed collection source")
    # Helper/cache/graph utilities must actually have been imported here, not
    # merely hash similarly named files elsewhere on PYTHONPATH.
    import sys
    for module,path in ((collection,"experiments/v4_residual_training_data.py"),
            (sys.modules[NeighborhoodScopeCache.__module__],"experiments/v4_residual_common.py"),
            (sys.modules["joint_recovery.core"],"src/joint_recovery/core.py"),
            (sys.modules["joint_recovery.v4_budgeted_recovery"],"src/joint_recovery/v4_budgeted_recovery.py"),
            (sys.modules["joint_recovery.v4_factors"],"src/joint_recovery/v4_factors.py")):
        if Path(module.__file__).resolve()!=root/path:
            raise ValueError("Scientific replay import escaped the bound source root")


def _collection_config(protocol):
    calibration = protocol["calibration"]
    if (protocol.get("executed_common_greedy_prefix") is not True or protocol.get("fixed_scope_cap") != 256 or
            protocol["calibration_sha256"]!=CALIBRATION_SHA256 or
            calibration["status"]!="frozen_before_V4_fitting_and_fresh_validation" or
            calibration["primary_deadline_seconds"]!=.556 or
            calibration["total_deadlines_seconds"]!=[.139,.556,2.221] or
            calibration["caps"]!=[64,256,1024] or calibration["history_max_queries"]!=1):
        raise ValueError("Fresh labels use a different frozen cost/deadline protocol")
    points = tuple(Workpoint(**{key:p[key] for key in ("name","kind","amount","expected_seconds")})
                   for p in calibration["workpoints"])
    if (tuple(p.name for p in points)!=tuple("slice%g"%s for s in collection.NATIVE_SLICES) or
            tuple(p.expected_seconds for p in points)!=(.035,.075,.214)):
        raise ValueError("Shared native-point calibration differs")
    return collection.CollectionConfig(.556,points,CALIBRATION_SHA256)


def load_dataset(root, split, *, expected_protocol_sha256, expected_completion_sha256):
    """Load only complete72/24 bound registry; failed graph slots retain weight.

    The source capsule and all96 case bytes are checked for either split. This
    never generates a graph, replaces a failure or interprets confirmation.
    """
    if split not in collection.SPLITS: raise ValueError("Only frozen training/validation labels are allowed")
    root = Path(root).resolve(); protocol_path=root/"protocol.json"; complete_path=root/"completion.json"
    if (sha256_file(protocol_path)!=expected_protocol_sha256 or
            sha256_file(complete_path)!=expected_completion_sha256):
        raise ValueError("Caller-pinned protocol/completion receipt differs")
    protocol,complete = _read_json(protocol_path),_read_json(complete_path)
    expected=list(collection.fresh_split_specs("training"))+list(collection.fresh_split_specs("validation"))
    if (protocol["status"]!="frozen_actual_fresh_label_queue" or protocol["splits"]!=["training","validation"] or
            protocol["complete_specs"]!=expected or protocol["state_weight"]!=.5 or
            protocol["backend_reset_each_alternative"] is not True or
            protocol["peer_outcomes_never_update_peers"] is not True or
            protocol["serial_single_native_thread"] is not True or protocol["model_fits_started"]!=0 or
            protocol["confirmation_graphs_generated"]!=0): raise ValueError("Complete fixed fresh collection protocol differs")
    if (complete["status"]!="complete_fresh_label_coverage_not_fitting" or
            complete["expected_cases"]!=96 or len(complete["cases"])!=96 or
            complete["protocol_sha256"]!=expected_protocol_sha256 or complete["model_fits_started"]!=0 or
            complete["confirmation_graphs_generated"]!=0 or complete["includes_failed_and_no_startable_cases"] is not True):
        raise ValueError("Incomplete/replaced fresh graph registry cannot fit")
    _check_source_capsule(protocol["source_capsule"]); config=_collection_config(protocol)
    loaded=[]
    for index,(spec,entry) in enumerate(zip(expected,complete["cases"])):
        case_id="case_%03d"%index
        if entry["case_id"]!=case_id or entry["spec"]!=spec: raise ValueError("Declared graph order or split was changed")
        files=entry["files"]
        if not {"spec.json","labels.json"}<=set(files) or not set(files)<={"spec.json","labels.json","observable.npz"}:
            raise ValueError("Case artifact schema differs")
        folder=root/"cases"/case_id
        if (folder.is_symlink() or (root/"cases").is_symlink() or root not in folder.resolve().parents):
            raise ValueError("Case directory is redirected outside the collection root")
        paths={name:_bound_file(folder,name,binding) for name,binding in files.items()}
        saved_spec,labels=_read_json(paths["spec.json"]),_read_json(paths["labels.json"])
        if saved_spec!=spec or labels["spec"]!=spec or labels["status"]!=entry["status"]:
            raise ValueError("Case records do not bind the declared graph")
        weight=collection.split_graph_weight(spec)
        if labels["status"]=="fresh_declared_graph_collection_failed":
            if (labels["groups"]!=[] or labels["included_in_coverage"] is not True or
                    labels["incomplete_case_labels_not_certified"] is not True or
                    labels["learning_advantage_established"] is not False or
                    labels["confirmation_graphs_generated"]!=0): raise ValueError("Failed graph slot was fabricated or removed")
            _equal(labels["graph_loss_weight"],weight,"failed graph fixed weight")
            if labels["graph_sha256"]!=entry["graph_sha256"]: raise ValueError("Failed graph source binding differs")
            if "observable.npz" in paths:
                graph,selected,cliques=graph_from_npz(paths["observable.npz"])
                if collection.graph_identity(graph)!=entry["graph_sha256"]: raise ValueError("Failed graph observable bytes changed identity")
            else: graph=None; selected=frozenset(); cliques=()
            states=()
        else:
            if "observable.npz" not in paths: raise ValueError("A successful case requires original observables")
            graph,selected,cliques=graph_from_npz(paths["observable.npz"])
            if collection.graph_identity(graph)!=entry["graph_sha256"]: raise ValueError("Manifest graph identity differs")
            states,weight=replay_case(graph,selected,cliques,spec,labels,config)
            if (entry["actual_history_calls"]!=labels["history_actual_calls"] or
                    entry["actual_alternative_calls"]!=labels["alternative_actual_calls"]):
                raise ValueError("Completion actual-call coverage differs")
        if spec["split"]==split:
            loaded.append(LoadedCase(case_id,spec,graph,selected,cliques,states,weight,entry["status"],files))
    if len(loaded)!=(72 if split=="training" else 24): raise ValueError("Fixed graph split is incomplete")
    return BoundDataset(split,tuple(loaded),expected_protocol_sha256,expected_completion_sha256,str(root),protocol["source_capsule"])


def build_model(config):
    """Matched predeclared architectures; this constructs only an untrained net."""
    from joint_recovery import v4_residual_model as residual
    if sha256_file(residual.__file__)!=MODEL_SHA256: raise ValueError("Reserved residual model bytes changed after source freeze")
    return residual.build_residual_model(config.variant)


def pack_state(case, state, model, config, device="cpu",static_cache=None):
    """Encoder sees only graph, reconstructed views, known context and factors."""
    from joint_recovery import v4_residual_model as residual
    from joint_recovery.v4_model_fast import FastStaticPackingCache
    if not state.views: raise ValueError("An unavailable state is a zero loss; it has no invented model batch")
    scale=normalization_scale(case.graph); warm=config.variant!="ResidualNoWarmMembership"
    cache=static_cache if static_cache is not None else FastStaticPackingCache(summary_only=config.variant=="ResidualCheapSummary")
    if config.variant=="ResidualCheapSummary":
        return residual.pack_residual_summary(case.graph,state.views,state.context,scale=scale,device=device,
            static_cache=cache,use_warm_membership=warm)
    return residual.pack_residual_v4(case.graph,state.views,state.context,scale=scale,device=device,
        static_cache=cache,embedding_cache=None,use_warm_membership=warm)


def state_targets(case,state,batch,device="cpu"):
    """Actual returned labels enter loss only, after observable packing."""
    scale=normalization_scale(case.graph); raw=[]; admitted=[]; valid=[]; masks=[]
    for view,row in zip(state.views,state.rows):
        actual=frozenset(row["returned_recovery"] or ()) if row["raw_supervision_valid"] else frozenset()
        raw.append(float(row["signed_return_delta"])/scale if row["raw_supervision_valid"] else 0.)
        admitted.append(float(row["admitted_gain"])/scale); valid.append(row["raw_supervision_valid"])
        masks.extend(float(v in actual) for v in view.request.scope.replacements)
    result=dict(raw_gain=torch.tensor(raw,dtype=torch.float32,device=device),
        gains=torch.tensor(admitted,dtype=torch.float32,device=device),
        valid=torch.tensor(valid,dtype=torch.bool,device=device),
        recovery_mask=torch.tensor(masks,dtype=torch.float32,device=device))
    if len(raw)!=batch["request_count"] or batch["request_keys"]!=tuple(v.request.key for v in state.views):
        raise ValueError("Packed request order differs from actual execution targets")
    return result


def graph_loss(case,model,config,device="cpu",static_cache=None):
    if not case.states: return weighted_graph_loss((zero_state(model),zero_state(model)),case.graph_weight)
    components=[]
    for state in case.states:
        if not state.views: components.append(zero_state(model)); continue
        batch=pack_state(case,state,model,config,device,static_cache); details=model(batch,return_details=True)
        targets=state_targets(case,state,batch,device)
        components.append(state_loss(details,targets,batch,auxiliary_weight=config.effective_auxiliary_weight,
            ranking_weight=config.ranking_weight,temperature=config.temperature))
    return weighted_graph_loss(components,case.graph_weight)


def evaluate_offline(dataset,model,config,device="cpu",static_caches=None):
    """All graph slots retain fixed weights; no solver/online validation call."""
    previous=model.training; model.eval(); rows=[]
    try:
        with torch.no_grad():
            for case in dataset.cases:
                metrics=[];components=[];cache=None if static_caches is None else static_caches[case.case_id]
                for state in case.states:
                    if state.views:
                        batch=pack_state(case,state,model,config,device,cache); details=model(batch,return_details=True)
                        targets=state_targets(case,state,batch,device)
                        components.append(state_loss(details,targets,batch,auxiliary_weight=config.effective_auxiliary_weight,
                            ranking_weight=config.ranking_weight,temperature=config.temperature))
                        raw=details["raw_gain"].detach().cpu().numpy()*normalization_scale(case.graph)
                        metric=offline_state_metrics(raw,[row["admitted_gain"] for row in state.rows],
                            state.context.best_gain,[v.request.workpoint.expected_seconds for v in state.views],state.context.incumbent_value)
                    else:
                        components.append(zero_state(model))
                        metric=offline_state_metrics([],[],state.context.best_gain,[],state.context.incumbent_value)
                    metrics.append(metric)
                # Failed graph slots have unavailable conditional supervision,
                # retain zero weighted contribution and explicit coverage.
                regret=sum(m["relative_conditional_allocation_regret"] for m in metrics)*.5
                accuracy=sum(m["pair_accuracy"] or 0. for m in metrics)*.5
                if not components: components=[zero_state(model),zero_state(model)]
                loss=weighted_graph_loss(components,case.graph_weight)
                rows.append(dict(case_id=case.case_id,domain=case.spec["domain"],status=case.status,
                    graph_weight=case.graph_weight,loss={k:float(v.item()) for k,v in loss.items()},
                    relative_conditional_allocation_regret=regret,pair_accuracy=accuracy,states=metrics,
                    pair_accuracy_defined_states=sum(m["pair_accuracy"] is not None for m in metrics),
                    available_state_count=sum(bool(s.views) for s in case.states),
                    supervision_available=bool(case.states)))
    finally: model.train(previous)
    rank_mass=sum(row["graph_weight"]*row["pair_accuracy"] for row in rows)
    rank_weight=sum(row["graph_weight"]*.5*row["pair_accuracy_defined_states"] for row in rows)
    return dict(scope="offline_conditional_labels_not_full_policy",graph_rows=rows,
        loss={key:sum(row["loss"][key] for row in rows) for key in ("total","regression","ranking","auxiliary")},
        graph_domain_balanced_regret=sum(row["graph_weight"]*row["relative_conditional_allocation_regret"] for row in rows),
        pair_accuracy_weighted_mass=rank_mass,pair_accuracy_defined_weight=rank_weight,
        graph_domain_balanced_pair_accuracy=rank_mass/rank_weight if rank_weight else None,
        graph_count=len(rows),failed_graph_slots=sum(not row["supervision_available"] for row in rows),
        unavailable_state_count=2*len(rows)-sum(row["available_state_count"] for row in rows),
        no_online_validation=True)


def validate_fit_sources(path, expected_sha256):
    """Full byte closure plus actual origins for the fitting scientific imports."""
    if sha256_file(path)!=expected_sha256: raise ValueError("Caller-pinned fitting source manifest differs")
    manifest=_read_json(path); root=Path(__file__).resolve().parents[1]
    records=manifest["files"];names=[r["path"] for r in records]
    if len(names)!=len(set(names)) or not set(FIT_SOURCE_PATHS)<=set(names):
        raise ValueError("Fitting scientific source closure is incomplete")
    for row in records:
        relative=Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or root not in (root/relative).resolve().parents:
            raise ValueError("Fitting source escapes immutable root")
        if sha256_file(root/relative)!=row["sha256"]: raise ValueError("Fitting source bytes changed")
    import importlib
    import sys
    direct=((sys.modules[__name__],"experiments/v4_residual_fit.py"),
        (collection,"experiments/v4_residual_training_data.py"),
        (sys.modules[NeighborhoodScopeCache.__module__],"experiments/v4_residual_common.py"))
    imported=tuple((importlib.import_module(module_name),relative) for module_name,relative in (
            ("joint_recovery","src/joint_recovery/__init__.py"),
            ("joint_recovery.core","src/joint_recovery/core.py"),
            ("joint_recovery.v4_budgeted_recovery","src/joint_recovery/v4_budgeted_recovery.py"),
            ("joint_recovery.v4_factors","src/joint_recovery/v4_factors.py"),
            ("joint_recovery.v4_factors_fast","src/joint_recovery/v4_factors_fast.py"),
            ("joint_recovery.v4_model","src/joint_recovery/v4_model.py"),
            ("joint_recovery.v4_model_fast","src/joint_recovery/v4_model_fast.py"),
            ("joint_recovery.v4_factorized_model","src/joint_recovery/v4_factorized_model.py"),
            ("joint_recovery.v4_residual_model","src/joint_recovery/v4_residual_model.py")))
    for module,relative in direct+imported:
        if Path(module.__file__).resolve()!=root/relative: raise ValueError("Actual fitting import escaped bound root")
    return manifest


def checkpoint_record(model,config,epoch,metric,bindings):
    if isinstance(epoch,bool) or not isinstance(epoch,Integral) or not 1<=epoch<=40:
        raise ValueError("Checkpoint epoch is outside fixed fitting source")
    _finite(metric,"conditional validation regret",0)
    return dict(schema="v4_fixed_scope_joint_residual_fit_v1",variant=config.variant,fit_seed=config.fit_seed,
        epoch=int(epoch),config=dict(config.__dict__),selection_metric="graph_domain_balanced_offline_conditional_allocation_regret",
        selection_metric_value=metric,selection_rule="first_minimum_among_all40_epochs_all24_validation_graphs",
        model_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
        bindings=copy.deepcopy(bindings),loss_scope="state means; graph*.5 once; full-registry graph-batch sampling",
        no_online_validation=True,no_confirmation=True,no_learning_advantage_claim=True)


def load_bound_checkpoint(path,expected_sha256,config,bindings,device="cpu"):
    """Load only caller-pinned trusted own checkpoints; no unbound pickle input."""
    if sha256_file(path)!=expected_sha256: raise ValueError("Checkpoint bytes differ from caller-pinned receipt")
    # These files are locally generated trusted fit artifacts, not arbitrary
    # downloads. Omit weights_only: PyTorch1.11 does not provide that argument.
    record=torch.load(path,map_location="cpu")
    if (record["schema"]!="v4_fixed_scope_joint_residual_fit_v1" or record["variant"]!=config.variant or
            record["fit_seed"]!=config.fit_seed or record["config"]!=dict(config.__dict__) or
            record["bindings"]!=bindings or record["no_online_validation"] is not True or
            record["no_confirmation"] is not True or record["no_learning_advantage_claim"] is not True):
        raise ValueError("Checkpoint configuration/source/data binding differs")
    if not 1<=record["epoch"]<=40 or not isfinite(record["selection_metric_value"]):
        raise ValueError("Checkpoint epoch/selection metadata is invalid")
    model=build_model(config);model.load_state_dict(record["model_state"],strict=True);model.to(device)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("Checkpoint has non-finite parameters")
    return model,record


def first_minimum_epoch(metrics):
    if not metrics or any(not isfinite(value) or value<0 for value in metrics):
        raise ValueError("Finite nonnegative complete conditional regrets required")
    return min(range(len(metrics)),key=lambda i:(metrics[i],i))+1


def _write_json(path,value):
    path=Path(path)
    with path.open("x",encoding="utf-8") as handle:
        handle.write(json.dumps(value,indent=2,allow_nan=False)+"\n")


def epoch_progress(out,config,epoch,regret,loss,elapsed_seconds):
    """Replace one NONFINAL latest-epoch log; completion.json certifies finish.

    This observable update is not a checkpoint, a target or policy outcome.
    Even epoch40 is nonfinal until the independent completion receipt exists.
    """
    if isinstance(epoch,bool) or not isinstance(epoch,Integral) or not 1<=epoch<=40:
        raise ValueError("Progress epoch is outside the frozen loop")
    _finite(regret,"offline progress regret",0);_finite(elapsed_seconds,"progress elapsed",0)
    if set(loss)!={"total","regression","ranking","auxiliary"}:
        raise ValueError("Progress needs the four already-computed loss components")
    for key,value in loss.items():_finite(value,"progress "+key,0)
    record=dict(status="fit_epoch_update_nonfinal",is_final=False,
        semantics="replace_latest_epoch_update; completion.json alone records finished fit",
        variant=config.variant,fit_seed=config.fit_seed,epoch=int(epoch),expected_epochs=40,
        offline_conditional_regret=regret,training_weighted_loss=dict(loss),elapsed_seconds=elapsed_seconds,
        scope="offline_conditional_labels_not_full_policy")
    path=Path(out)/"progress.json";temporary=path.with_name("progress.json.tmp")
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(record,separators=(",",":"),allow_nan=False),flush=True)
    return record


def fit_one(training,validation,config,out,*,bindings,device="cpu"):
    """Explicit fitting API, never invoked by finite source guards.

    Graph batches sample the complete72 registry. Multiplying batch weighted
    sums by N/batch_size is the unbiased graph sampling factor; it never
    divides by a realized batch's weights or available requests. Failed/empty
    states remain zero fixed-weight contributions. Static CPU records may be
    retained as offline dataset preparation; no gradient embedding is cached.
    This amortization is NOT used as an online deadline/speed claim.
    """
    if (training.split!="training" or validation.split!="validation" or
            len(training.cases)!=72 or len(validation.cases)!=24 or
            training.protocol_sha256!=validation.protocol_sha256 or
            training.completion_sha256!=validation.completion_sha256):
        raise ValueError("Complete bound graph splits required")
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    random.seed(config.fit_seed);np.random.seed(config.fit_seed);torch.manual_seed(config.fit_seed)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available(): raise ValueError("Requested CUDA runtime is unavailable")
        torch.cuda.manual_seed_all(config.fit_seed)
    model=build_model(config).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    from joint_recovery.v4_model_fast import FastStaticPackingCache
    caches={case.case_id:FastStaticPackingCache(summary_only=config.variant=="ResidualCheapSummary")
            for case in training.cases+validation.cases}
    rng=np.random.RandomState(config.fit_seed);history=[];selected=None;selected_metric=float("inf");started=time.perf_counter()
    for epoch in range(1,41):
        model.train();order=rng.permutation(72);training_loss={k:0. for k in ("total","regression","ranking","auxiliary")}
        for offset in range(0,72,4):
            indices=order[offset:offset+4];optimizer.zero_grad()
            components=[graph_loss(training.cases[int(i)],model,config,device,caches[training.cases[int(i)].case_id]) for i in indices]
            summed={key:sum(row[key] for row in components) for key in training_loss}
            loss=summed["total"]*(72./len(indices))
            if not bool(torch.isfinite(loss)): raise ValueError("Non-finite weighted fitting loss")
            loss.backward();optimizer.step()
            for key in training_loss: training_loss[key]+=float(summed[key].detach().cpu().item())
        validation_metrics=evaluate_offline(validation,model,config,device,caches)
        metric=validation_metrics["graph_domain_balanced_regret"]
        if metric<selected_metric:
            selected_metric=metric;selected=checkpoint_record(model,config,epoch,metric,bindings)
        history.append(dict(epoch=epoch,training_weighted_loss=training_loss,
            training_loss_scope="online_optimization_epoch_sum_with_changing_parameters",
            validation=validation_metrics))
        epoch_progress(out,config,epoch,metric,training_loss,time.perf_counter()-started)
    if selected is None or selected["epoch"]!=first_minimum_epoch([row["validation"]["graph_domain_balanced_regret"] for row in history]):
        raise ValueError("Checkpoint selection violates first-minimum rule")
    final=checkpoint_record(model,config,40,history[-1]["validation"]["graph_domain_balanced_regret"],bindings)
    torch.save(selected,out/"selected.pt");torch.save(final,out/"epoch40.pt")
    _write_json(out/"history.json",dict(config=dict(config.__dict__),bindings=bindings,epochs=history,
        selected_epoch=selected["epoch"],fit_seed=config.fit_seed,elapsed_seconds=time.perf_counter()-started,
        curves_scope="all40_epochs_all24_graphs_offline_conditional_labels",no_online_validation=True))
    files={name:dict(sha256=sha256_file(out/name),bytes=(out/name).stat().st_size)
           for name in ("selected.pt","epoch40.pt","history.json")}
    _write_json(out/"completion.json",dict(status="complete_fixed_fit_not_online_policy_validation",
        variant=config.variant,fit_seed=config.fit_seed,selected_epoch=selected["epoch"],files=files,
        bindings=bindings,all40_epochs_retained=True,no_learning_advantage_claim=True))
    return dict(variant=config.variant,fit_seed=config.fit_seed,selected_epoch=selected["epoch"],
        files=files,completion_sha256=sha256_file(out/"completion.json"))


def fit_suite(collection_root,out,*,expected_protocol_sha256,expected_completion_sha256,
        fit_manifest_path,expected_fit_manifest_sha256,device="cpu",include_repeat_capacity=False):
    """Explicit all5x3 API; optional repeat diagnostic frozen before outcomes."""
    if include_repeat_capacity is not False: raise ValueError("Only the fixed five reserved variants are permitted")
    manifest=validate_fit_sources(fit_manifest_path,expected_fit_manifest_sha256)
    training=load_dataset(collection_root,"training",expected_protocol_sha256=expected_protocol_sha256,
        expected_completion_sha256=expected_completion_sha256)
    validation=load_dataset(collection_root,"validation",expected_protocol_sha256=expected_protocol_sha256,
        expected_completion_sha256=expected_completion_sha256)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    variants=VARIANTS+((DIAGNOSTIC_VARIANT,) if include_repeat_capacity else ())
    bindings=dict(collection_protocol_sha256=expected_protocol_sha256,collection_completion_sha256=expected_completion_sha256,
        calibration_sha256=CALIBRATION_SHA256,fit_source_manifest_sha256=expected_fit_manifest_sha256,
        fit_source_capsule=manifest,training_case_files={c.case_id:c.file_bindings for c in training.cases},
        validation_case_files={c.case_id:c.file_bindings for c in validation.cases})
    _write_json(out/"protocol.json",dict(status="frozen_fixed_fit_suite_before_validation_selection",
        variants=variants,fit_seeds=FIT_SEEDS,epochs=40,batch_graphs=4,optimizer="Adam1e-3",
        bindings=bindings,include_repeat_capacity=include_repeat_capacity,
        selection="firstminimum_all24_graph_domain_offline_conditional_regret",all_seeds_retained=True,
        no_lucky_fit_selection=True,no_online_validation=True,confirmation_graphs_generated=0,
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=str(device))))
    fits=[]
    for variant in variants:
        for seed in FIT_SEEDS:
            fits.append(fit_one(training,validation,FitConfig(variant,seed),out/(variant+"_seed%d"%seed),bindings=bindings,device=device))
    validate_fit_sources(fit_manifest_path,expected_fit_manifest_sha256)
    if (sha256_file(Path(collection_root)/"protocol.json")!=expected_protocol_sha256 or
            sha256_file(Path(collection_root)/"completion.json")!=expected_completion_sha256):
        raise ValueError("Collection binding changed during fitting")
    _write_json(out/"completion.json",dict(status="complete_all_predeclared_fits_not_policy_evidence",fits=fits,
        protocol_sha256=sha256_file(out/"protocol.json"),all_fit_seeds_retained=True,no_learning_advantage_claim=True,
        no_online_validation=True,confirmation_graphs_generated=0))
    return fits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--protocol-sha256",required=True);parser.add_argument("--completion-sha256",required=True)
    parser.add_argument("--source-manifest",required=True);parser.add_argument("--source-manifest-sha256",required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--include-repeat-capacity",action="store_true")
    args=parser.parse_args()
    fit_suite(args.collection_root,args.out,expected_protocol_sha256=args.protocol_sha256,
        expected_completion_sha256=args.completion_sha256,fit_manifest_path=args.source_manifest,
        expected_fit_manifest_sha256=args.source_manifest_sha256,device=args.device,
        include_repeat_capacity=args.include_repeat_capacity)


if __name__=="__main__": main()
