"""Sparse, observable-only finite recovery value prototypes (V4).

All predictions are in graph-normalized gain units; ``priority_callback``
converts them back to native objective units. Auxiliary memberships enter only
``execution_targets``/the loss, never ``pack_v4``. Fractional capacities are
not feasible integer schedules. No deployment or model fitting happens here.
Uses index_add and an exact segmented maximum fallback for PyTorch 1.11.
"""
from __future__ import annotations

from math import fsum, isfinite, log1p
from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .v4_factors import observable_factors
from .v4_budgeted_recovery import PriorityEvaluation, _check_recovery, objective


NODE_FEATURES = ("normalized_weight", "log_local_degree", "local_degree_fraction",
                 "actual_warm_membership", "log_factor_memberships", "log_max_factor_size")
CONTEXT_FEATURES = ("normalized_q", "log_R", "kind_seconds", "kind_nodes", "kind_iterations",
                    "log_native_amount", "log_expected_request_seconds", "log_remaining_seconds",
                    "remaining_over_expected", "normalized_warm_value", "warm_fraction")
SUMMARY_FEATURES = ("normalized_R_weight", "max_normalized_weight", "mean_normalized_weight",
                    "std_normalized_weight", "log_edges", "edge_density", "log_factors",
                    "max_factor_size", "mean_factor_size", "mean_memberships",
                    "high_order_factor_fraction", "normalized_p1", "partition_upper_bound")


def normalization_scale(graph):
    weights = np.asarray(graph.weights, dtype=np.float64)
    if weights.shape != (graph.n,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Finite nonnegative graph rewards are required")
    return max(1., float(fsum(map(float, weights)) / max(1, graph.n)))


class StaticPackingCache:
    """Paid outcome-free scope preparation, bounded to one decision callback.

    No encoder outputs, predictions, warm masks or execution outcomes are
    cached. Keep the actual graph object alive; the caller must not mutate its
    weights/edges during a decision. Different bases/displacements never alias.
    """
    def __init__(self): self.graph=None; self.records={}
    def reset(self): self.graph=None; self.records.clear()
    def get(self,graph,scope,scale):
        if self.graph is not graph:
            self.reset(); self.graph=graph
        key=(scope.action_index,scope.inserts,scope.base,scope.displaced,
             scope.replacements,scope.resource_cliques,scope.immediate_gain,scale)
        if key not in self.records:
            neighbors,factors=observable_factors(graph,scope)
            weights=np.asarray(graph.weights[list(scope.replacements)],np.float64)/scale
            stats,memberships,maximum=_summary(weights,neighbors,factors)
            n=len(weights)
            x=np.zeros((n,len(NODE_FEATURES)),np.float64)
            x[:,0]=weights; degree=np.asarray(list(map(len,neighbors)),np.float64)
            x[:,1]=np.log1p(degree); x[:,2]=degree/max(1,n-1)
            x[:,4]=np.log1p(memberships); x[:,5]=np.log1p(maximum)
            pairs=[(i,j) for i,row in enumerate(neighbors) for j in sorted(row)]
            edges=np.asarray(pairs,np.int64).reshape(-1,2).T.copy()
            memberships=[(v,f) for f,q in enumerate(factors) for v in q]
            incidence=np.asarray(memberships,np.int64).reshape(-1,2).T.copy()
            by_node=np.asarray(sorted(memberships),np.int64).reshape(-1,2).T.copy()
            self.records[key]=(factors,weights,stats,x,edges,incidence,by_node)
        return self.records[key]


def _summary(weights, neighbors, factors):
    n = len(weights); edges = sum(map(len, neighbors)) // 2
    memberships = np.zeros(n, np.float64); maximum = np.ones(n, np.float64)
    for factor in factors:
        memberships[list(factor)] += 1.; maximum[list(factor)] = np.maximum(maximum[list(factor)],len(factor))
    p1 = float(fsum(float(w/d) for w,d in zip(weights,maximum)))
    remaining = set(range(n)); upper = 0.
    for factor in sorted(factors,key=lambda q:(-len(q),q)):
        block = remaining.intersection(factor)
        if block:
            upper += max(float(weights[v]) for v in block); remaining.difference_update(block)
    upper += fsum(float(weights[v]) for v in sorted(remaining))
    sizes = tuple(map(len,factors))
    stats = (float(weights.sum()), float(weights.max()) if n else 0.,
        float(weights.mean()) if n else 0., float(weights.std()) if n else 0., log1p(edges),
        2.*edges/max(1,n*(n-1)), log1p(len(factors)), float(max(sizes,default=0)),
        float(np.mean(sizes)) if sizes else 0., float(memberships.mean()) if n else 0.,
        sum(size>=3 for size in sizes)/max(1,len(sizes)), p1, upper)
    return stats,memberships,maximum


def _context(graph,scope,workpoint,state,known,scale):
    n=len(scope.replacements); q=float(scope.immediate_gain)/scale
    if not isfinite(q): raise ValueError("Finite known immediate gain required")
    return (q,log1p(n),float(workpoint.kind=="seconds"),float(workpoint.kind=="nodes"),
        float(workpoint.kind=="iterations"),log1p(workpoint.amount),log1p(workpoint.expected_seconds),
        log1p(state.remaining_seconds),log1p(state.remaining_seconds/workpoint.expected_seconds),
        float(objective(graph,known))/scale,len(known)/max(1,n))


def pack_v4(graph, views, state, scale=None, device="cpu",static_cache=None):
    """Flatten independent R graphs; no padded R-by-R/factor-by-R tensors.

    Inputs are only graph/scope/native workpoint/current actual warm starts and
    remaining wall time. Current best gain is retained for the ranking loss,
    not used as extra graph information by the encoder. No outcome parameter
    is accepted. Request order is preserved, including empty scopes.
    """
    views = tuple(views)
    if not views:
        raise ValueError("At least one observable request required")
    scale = normalization_scale(graph) if scale is None else float(scale)
    if not isfinite(scale) or scale <= 0 or not isfinite(state.remaining_seconds) or state.remaining_seconds < 0:
        raise ValueError("Positive finite scale and nonnegative remaining seconds required")
    static_cache=StaticPackingCache() if static_cache is None else static_cache
    x=[]; weights=[]; warm=[]; contexts=[]; summaries=[]; node_request=[]
    edges=[]; incidence=[]; by_nodes=[]; factor_request=[]; ptr=[0]; factor_count=0
    nodes_metadata=[]; factor_metadata=[]
    for request_index,view in enumerate(views):
        scope,workpoint = view.request.scope,view.request.workpoint
        factors,w,stats,static_x,static_edges,static_incidence,static_by_node=static_cache.get(graph,scope,scale)
        known = _check_recovery(graph,scope,view.warm_start)
        n=len(w); offset=ptr[-1]; current_warm=np.asarray([v in known for v in scope.replacements],np.float64)
        contexts.append(_context(graph,scope,workpoint,state,known,scale))
        summaries.append(stats)
        dynamic_x=static_x.copy(); dynamic_x[:,3]=current_warm
        x.append(dynamic_x); weights.append(w); warm.append(current_warm)
        node_request.append(np.full(n,request_index,np.int64)); edges.append(static_edges+offset)
        delta=np.asarray([[offset],[factor_count]],np.int64)
        incidence.append(static_incidence+delta); by_nodes.append(static_by_node+delta)
        factor_request.append(np.full(len(factors),request_index,np.int64))
        ptr.append(offset+n); factor_count += len(factors)
        nodes_metadata.append(tuple(scope.replacements)); factor_metadata.append(factors)
    arrays={"x":np.concatenate(x,axis=0).astype(np.float32),
        "weights":np.concatenate(weights).astype(np.float32),"warm":np.concatenate(warm).astype(np.float32),
        "context":np.asarray(contexts,np.float32),"summary":np.asarray(summaries,np.float32),
        "node_request":np.concatenate(node_request),"edge_index":np.concatenate(edges,axis=1),
        "incidence":np.concatenate(incidence,axis=1),"incidence_by_node":np.concatenate(by_nodes,axis=1),
        "factor_request":np.concatenate(factor_request),"ptr":np.asarray(ptr,np.int64),
        "immediate":np.asarray([v[0] for v in contexts],np.float32),
        "best_gain":np.full(len(views),float(state.best_gain)/scale,np.float32),
        "scale":np.full(len(views),scale,np.float32)}
    if any(not np.all(np.isfinite(a)) for a in arrays.values()):
        raise ValueError("All observable packed features must be finite")
    result={name:torch.as_tensor(a,device=device) for name,a in arrays.items()}
    result.update(request_count=len(views),node_count=ptr[-1],factor_count=factor_count,
        node_ids=tuple(nodes_metadata),factors=tuple(factor_metadata),
        request_keys=tuple(v.request.key for v in views),feature_names=dict(
            node=NODE_FEATURES,context=CONTEXT_FEATURES,summary=SUMMARY_FEATURES))
    return result


def pack_summary_v4(graph,views,state,scale=None,device="cpu",static_cache=None):
    """Same analytic context/summary without unused node/GPU incidence packing.

    The summary control pays for its scope/factor/statistic construction and
    warm validation but transfers only the actual MLP inputs. It is not charged
    for a graph encoder representation it never consumes.
    """
    views=tuple(views)
    if not views: raise ValueError("At least one observable request required")
    scale=normalization_scale(graph) if scale is None else float(scale)
    if not isfinite(scale) or scale<=0 or not isfinite(state.remaining_seconds) or state.remaining_seconds<0:
        raise ValueError("Positive finite scale and nonnegative remaining seconds required")
    cache=StaticPackingCache() if static_cache is None else static_cache
    contexts=[]; summaries=[]
    for view in views:
        scope,workpoint=view.request.scope,view.request.workpoint
        _,_,stats,_,_,_,_=cache.get(graph,scope,scale)
        known=_check_recovery(graph,scope,view.warm_start)
        contexts.append(_context(graph,scope,workpoint,state,known,scale)); summaries.append(stats)
    arrays=dict(context=np.asarray(contexts,np.float32),summary=np.asarray(summaries,np.float32),
        immediate=np.asarray([c[0] for c in contexts],np.float32),
        best_gain=np.full(len(views),float(state.best_gain)/scale,np.float32),
        scale=np.full(len(views),scale,np.float32))
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Finite summary inputs required")
    result={name:torch.as_tensor(value,device=device) for name,value in arrays.items()}
    result.update(request_count=len(views),request_keys=tuple(v.request.key for v in views),
        feature_names=dict(context=CONTEXT_FEATURES,summary=SUMMARY_FEATURES))
    return result


def merge_v4_batches(batches):
    """Offline training concatenation with disjoint node/factor/request offsets.

    Each original state remains one ranking group. No graph messages cross
    requests or states, and each request keeps its own objective scale.
    """
    batches=tuple(batches)
    if not batches: raise ValueError("At least one packed state required")
    tensor_names=("x","weights","warm","context","summary","immediate","best_gain","scale")
    parts={name:[] for name in tensor_names}
    indices={name:[] for name in ("node_request","edge_index","incidence","incidence_by_node","factor_request")}
    request_offset=node_offset=factor_offset=0; ptr=[0]; ids=[]; factors=[]; keys=[]; groups=[]
    for batch in batches:
        for name in tensor_names: parts[name].append(batch[name])
        indices["node_request"].append(batch["node_request"]+request_offset)
        indices["factor_request"].append(batch["factor_request"]+request_offset)
        indices["edge_index"].append(batch["edge_index"]+node_offset)
        for name in ("incidence","incidence_by_node"):
            delta=batch[name].new_tensor([[node_offset],[factor_offset]])
            indices[name].append(batch[name]+delta)
        ptr.extend(int(v)+node_offset for v in batch["ptr"].detach().cpu().tolist()[1:])
        ids.extend(batch["node_ids"]); factors.extend(batch["factors"]); keys.extend(batch["request_keys"])
        groups.append(batch["request_count"])
        request_offset+=batch["request_count"]; node_offset+=batch["node_count"]; factor_offset+=batch["factor_count"]
    result={name:torch.cat(values,dim=0) for name,values in parts.items()}
    result.update({name:torch.cat(values,dim=1 if name in ("edge_index","incidence","incidence_by_node") else 0)
                   for name,values in indices.items()})
    result.update(ptr=batches[0]["ptr"].new_tensor(ptr),request_count=request_offset,
        node_count=node_offset,factor_count=factor_offset,node_ids=tuple(ids),factors=tuple(factors),
        request_keys=tuple(keys),group_sizes=tuple(groups),feature_names=batches[0]["feature_names"])
    return result


def merge_v4_targets(targets):
    targets=tuple(targets)
    if not targets: raise ValueError("At least one execution-target batch required")
    return {name:torch.cat([target[name] for target in targets],dim=0) for name in targets[0]}


def _sum(values,index,count):
    return values.new_zeros((count,)+values.shape[1:]).index_add(0,index,values)


def _segment_max_sorted(values,index,count):
    """Exact associative segmented prefix maximum; no scatter_reduce dependency.

    Index must be sorted (guaranteed by pack_v4). Vectorized doubling costs
    O(M log M) time and O(M) memory. It remains differentiable through maxima.
    This is an equivalent implementation, not a hardware speed claim.
    """
    if not values.numel(): return values.new_zeros(count)
    current=values; shift=1
    while shift < values.shape[0]:
        same=index[shift:]==index[:-shift]
        previous=torch.cat((current.new_full((shift,),-float("inf")),current[:-shift]))
        same=torch.cat((torch.zeros(shift,dtype=torch.bool,device=index.device),same))
        current=torch.maximum(current,torch.where(same,previous,current.new_full(current.shape,-float("inf"))))
        shift *= 2
    last=torch.cat((index[:-1]!=index[1:],torch.ones(1,dtype=torch.bool,device=index.device)))
    return _sum(current[last],index[last],count)


def capacity_project_sparse(probabilities,batch):
    if not batch["factor_count"]: return probabilities
    node,factor=batch["incidence"]
    loads=_sum(probabilities[node],factor,batch["factor_count"])
    by_node,by_factor=batch["incidence_by_node"]
    maximum=_segment_max_sorted(loads[by_factor],by_node,batch["node_count"]).clamp_min(1.)
    return probabilities/maximum


def factor_loads(probabilities,batch):
    node,factor=batch["incidence"]
    return _sum(probabilities[node],factor,batch["factor_count"])


class SparseRecoveryValueNet(nn.Module):
    """Predeclared h32/L2 local encoder, capacity or matched scalar readout.

    Both readouts share every node/conflict/clique/context input and encoder.
    Both heads are instantiated in the same order for matched initialization;
    the auxiliary mask head is available for both. Warm-mask ablation removes
    ALL warm membership/value/count input channels, with no extra parameters.
    """
    def __init__(self,hidden=32,layers=2,readout="capacity",use_warm_start=True):
        super().__init__()
        if readout not in ("capacity","free") or hidden<1 or layers<1:
            raise ValueError("Positive width/depth and capacity/free readout required")
        self.hidden,self.layers,self.readout,self.use_warm_start=hidden,layers,readout,use_warm_start
        c=len(CONTEXT_FEATURES); s=len(SUMMARY_FEATURES)
        self.embed=nn.Linear(len(NODE_FEATURES)+c,hidden)
        self.self_layers=nn.ModuleList(nn.Linear(hidden,hidden) for _ in range(layers))
        self.edge_layers=nn.ModuleList(nn.Linear(hidden,hidden,bias=False) for _ in range(layers))
        self.factor_layers=nn.ModuleList(nn.Linear(hidden,hidden,bias=False) for _ in range(layers))
        self.occupancy_head=nn.Sequential(nn.Linear(2*hidden+c+s,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.free_head=nn.Sequential(nn.Linear(2*hidden+c+s,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def forward(self,batch,return_details=False):
        x,context=batch["x"],batch["context"]
        if not self.use_warm_start:
            x=x.clone(); x[:,3]=0.
            context=context.clone(); context[:,9:11]=0.
        assignment=batch["node_request"]; count=batch["request_count"]
        h=torch.relu(self.embed(torch.cat((x,context[assignment]),dim=1)))
        source,destination=batch["edge_index"]; node,factor=batch["incidence"]
        degree=_sum(torch.ones_like(source,dtype=h.dtype),destination,batch["node_count"]).clamp_min(1.)
        factor_size=_sum(torch.ones_like(factor,dtype=h.dtype),factor,batch["factor_count"]).clamp_min(1.)
        memberships=_sum(torch.ones_like(node,dtype=h.dtype),node,batch["node_count"]).clamp_min(1.)
        for self_layer,edge_layer,factor_layer in zip(self.self_layers,self.edge_layers,self.factor_layers):
            message=_sum(edge_layer(h[source]),destination,batch["node_count"])/degree[:,None]
            pooled=_sum(h[node],factor,batch["factor_count"])/factor_size[:,None]
            relay=_sum(factor_layer(pooled[factor]),node,batch["node_count"])/memberships[:,None]
            h=torch.relu(self_layer(h)+message+relay)
        sizes=_sum(torch.ones_like(assignment,dtype=h.dtype),assignment,count).clamp_min(1.)
        pooled=_sum(h,assignment,count)/sizes[:,None]
        totals=_sum(batch["weights"],assignment,count)
        free_raw=None
        if self.readout=="free":
            weighted=_sum(h*batch["weights"][:,None],assignment,count)/totals.clamp_min(1e-12)[:,None]
            global_features=torch.cat((pooled,weighted,context,batch["summary"]),dim=1)
            free_raw=batch["immediate"]+self.free_head(global_features).squeeze(-1)
            if not return_details:
                # The free scalar deployment path pays for its encoder/head,
                # not an unused auxiliary capacity projection.
                return free_raw.clamp_min(0.)
        node_features=torch.cat((h,pooled[assignment],context[assignment],batch["summary"][assignment]),dim=1)
        logits=self.occupancy_head(node_features).squeeze(-1)
        proposed=torch.sigmoid(logits); occupancy=capacity_project_sparse(proposed,batch)
        recovery=_sum(occupancy*batch["weights"],assignment,count)
        raw=batch["immediate"]+recovery
        if self.readout=="free":
            raw=free_raw
        details=dict(raw_gain=raw,scores=raw.clamp_min(0.),recovery=recovery,
            proposed=proposed,occupancy=occupancy,logits=logits)
        return details if return_details else details["scores"]

    def parameter_counts(self):
        inactive=self.free_head if self.readout=="capacity" else self.occupancy_head
        stored=sum(p.numel() for p in self.parameters())
        return dict(stored=stored,active_scoring=stored-sum(p.numel() for p in inactive.parameters()))


class CheapSummaryValueNet(nn.Module):
    """Observable analytic summaries+q/workpoint/warm-value learned control.

    Uses no node message passing and no mask auxiliary. Summary P1 is not an
    integer bound; partition_upper_bound is an actual bound. It does not call
    greedy/search or acquire a hidden all-action feasible floor.
    """
    def __init__(self,hidden=32,use_warm_start=True):
        super().__init__(); self.hidden,self.use_warm_start=hidden,use_warm_start
        self.head=nn.Sequential(nn.Linear(len(SUMMARY_FEATURES)+len(CONTEXT_FEATURES),hidden),
            nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,batch,return_details=False):
        context=batch["context"]
        if not self.use_warm_start:
            context=context.clone(); context[:,9:11]=0.
        raw=batch["immediate"]+self.head(torch.cat((batch["summary"],context),dim=1)).squeeze(-1)
        details=dict(raw_gain=raw,scores=raw.clamp_min(0.),logits=None)
        return details if return_details else details["scores"]
    def parameter_counts(self):
        count=sum(p.numel() for p in self.parameters())
        return dict(stored=count,active_scoring=count)


def execution_targets(graph,views,recoveries,scale=None,device="cpu",on_time=None):
    """OFFLINE targets from actual valid memberships, never encoder inputs.

    None means no valid execution output: accepted outcome is incumbent/zero,
    while raw/mask regression is disabled for that request. Failure rows remain
    in same-state accepted-outcome ranking, not removed from evaluation.
    Valid rejected/late outputs still supervise signed q+recovery and masks.
    If ``on_time`` is supplied, late accepted-outcome/ranking gains are zero;
    returned_gains remains the diagnostic finite returned value. This does not
    turn a value/cost point heuristic into a completion-probability model.
    """
    views,recoveries=tuple(views),tuple(recoveries)
    if len(views)!=len(recoveries): raise ValueError("One outcome per request required")
    on_time=tuple(True for _ in views) if on_time is None else tuple(on_time)
    if len(on_time)!=len(views) or any(not isinstance(v,(bool,np.bool_)) for v in on_time):
        raise ValueError("One explicit boolean admission flag per request required")
    scale=normalization_scale(graph) if scale is None else float(scale)
    if not isfinite(scale) or scale<=0: raise ValueError("Positive finite scale required")
    raw=[]; accepted=[]; returned=[]; valid=[]; masks=[]
    for view,recovered,timely in zip(views,recoveries,on_time):
        scope=view.request.scope
        if recovered is None:
            raw.append(0.); accepted.append(0.); returned.append(0.); valid.append(False)
            masks.extend([0.]*len(scope.replacements)); continue
        actual=_check_recovery(graph,scope,recovered)
        value=(scope.immediate_gain+objective(graph,actual))/scale
        raw.append(value); returned.append(max(0.,value))
        accepted.append(max(0.,value) if timely else 0.); valid.append(True)
        masks.extend(float(v in actual) for v in scope.replacements)
    return dict(raw_gain=torch.tensor(raw,dtype=torch.float32,device=device),
        gains=torch.tensor(accepted,dtype=torch.float32,device=device),
        returned_gains=torch.tensor(returned,dtype=torch.float32,device=device),
        on_time=torch.tensor(on_time,dtype=torch.bool,device=device),
        valid=torch.tensor(valid,dtype=torch.bool,device=device),
        recovery_mask=torch.tensor(masks,dtype=torch.float32,device=device))


def recovery_value_loss(details,targets,batch,group_sizes,auxiliary_weight=.05,ranking_weight=1.,temperature=.2):
    """Signed regression before clipping, same-state realized-marginal ranking.

    Auxiliary BCE uses actual returned masks, including valid rejected updates.
    Ranking targets are [actual accepted gain-current best]_+; ties produce no
    ranking requirement. Score differences use pre-clamp raw predictions so an
    initially negative prediction still has a useful gradient. An auxiliary-free
    control removes BCE only; no raw-regression-through-zero-clamp is used.
    """
    if sum(group_sizes)!=batch["request_count"] or any(int(g)!=g or g<1 for g in group_sizes):
        raise ValueError("Positive same-state groups must cover every request")
    if temperature<=0 or auxiliary_weight<0 or ranking_weight<0:
        raise ValueError("Positive temperature and nonnegative loss weights required")
    raw=details["raw_gain"]; zero=raw.sum()*0.; valid=targets["valid"]
    regression=F.smooth_l1_loss(raw[valid],targets["raw_gain"][valid]) if bool(valid.any()) else zero
    ranks=[]; offset=0
    for size in group_sizes:
        gains=(targets["gains"][offset:offset+size]-batch["best_gain"][offset:offset+size]).clamp_min(0.)
        prediction=raw[offset:offset+size]
        desired=gains[:,None]>gains[None,:]+1e-6
        if bool(desired.any()):
            ranks.append(F.softplus(-(prediction[:,None]-prediction[None,:])[desired]/temperature).mean())
        offset += size
    ranking=torch.stack(ranks).mean() if ranks else zero
    auxiliary=zero
    if auxiliary_weight:
        if details.get("logits") is None: raise ValueError("This model has no mask-auxiliary interface")
        node_valid=valid[batch["node_request"]]
        if bool(node_valid.any()):
            node_loss=F.binary_cross_entropy_with_logits(details["logits"],targets["recovery_mask"],reduction="none")
            sums=_sum(node_loss*node_valid.to(node_loss.dtype),batch["node_request"],batch["request_count"])
            counts=_sum(node_valid.to(node_loss.dtype),batch["node_request"],batch["request_count"])
            available=counts>0
            auxiliary=(sums[available]/counts[available]).mean()
    total=regression+ranking_weight*ranking+auxiliary_weight*auxiliary
    return dict(total=total,regression=regression,ranking=ranking,auxiliary=auxiliary)


def priority_callback(model,scale=None,device="cpu"):
    """Timed deployment adapter: packing+transfer+forward+CPU gains all included."""
    static_cache=StaticPackingCache()
    def evaluate(graph,views,state):
        if not state.spent_requests:
            # Root also constructs a new policy per decision. This reset is an
            # additional boundary, not a free cross-decision feature cache.
            static_cache.reset()
        packer=pack_summary_v4 if isinstance(model,CheapSummaryValueNet) else pack_v4
        batch=packer(graph,views,state,scale=scale,device=device,static_cache=static_cache)
        with torch.no_grad(): native=model(batch)*batch["scale"]
        return PriorityEvaluation(tuple(float(v) for v in native.detach().cpu().tolist()),
            semantics="point finite recovery gain, native reward units")
    return evaluate
