"""Paid CPU preparation optimization; original model and dynamic pack unchanged.

Same-cover structural records can be reused within ONE decision for identical
ordered R/factors, independent of action/workpoint. Base/q/warm remain dynamic
and separately validated. Different nested R pools receive their own exact
cover; a full-pool factor cover is never silently substituted into a prefix.
"""
from types import FunctionType

import numpy as np
import torch
from math import fsum,log1p

from . import v4_model as model_api
from . import v4_budgeted_recovery as controller_api
from .v4_factors_fast import observable_factors_fast


def _summary_fast(weights,neighbors,factors):
    """Same statistics; vectorized exact-integer incidence counts and maxima."""
    n=len(weights);edges=sum(map(len,neighbors))//2
    sizes=np.asarray(list(map(len,factors)),np.int64)
    flat=np.fromiter((v for factor in factors for v in factor),np.int64,count=int(sizes.sum()))
    memberships=np.bincount(flat,minlength=n).astype(np.float64)
    maximum=np.ones(n,np.float64)
    np.maximum.at(maximum,flat,np.repeat(sizes,sizes))
    p1=float(fsum(float(w/d) for w,d in zip(weights,maximum)))
    remaining=set(range(n));upper=0.
    for factor in sorted(factors,key=lambda q:(-len(q),q)):
        if not remaining:break
        block=remaining.intersection(factor)
        if block:
            upper+=max(float(weights[v]) for v in block);remaining.difference_update(block)
    upper+=fsum(float(weights[v]) for v in sorted(remaining))
    stats=(float(weights.sum()),float(weights.max()) if n else 0.,
        float(weights.mean()) if n else 0.,float(weights.std()) if n else 0.,log1p(edges),
        2.*edges/max(1,n*(n-1)),log1p(len(factors)),float(max(sizes,default=0)),
        float(np.mean(sizes)) if len(sizes) else 0.,float(memberships.mean()) if n else 0.,
        sum(size>=3 for size in sizes)/max(1,len(sizes)),p1,upper)
    return stats,memberships,maximum


class FastStaticPackingCache:
    def __init__(self,summary_only=False):
        self.summary_only=bool(summary_only);self.graph=None;self.records={}
    def reset(self):self.graph=None;self.records.clear()
    def get(self,graph,scope,scale):
        if self.graph is not graph:self.reset();self.graph=graph
        # These are precisely the observable dependencies of static R tensors.
        # Action index/releases/q/base are NOT encoded static values; current
        # contexts and warm membership are reconstructed by the original pack.
        key=(tuple(scope.replacements),tuple(scope.resource_cliques),scale)
        if key not in self.records:
            neighbors,factors=observable_factors_fast(graph,scope)
            weights=np.asarray(graph.weights[list(scope.replacements)],np.float64)/scale
            stats,memberships,maximum=_summary_fast(weights,neighbors,factors)
            if self.summary_only:
                self.records[key]=(factors,weights,stats,None,None,None,None)
            else:
                n=len(weights);degrees=np.asarray(list(map(len,neighbors)),np.int64)
                x=np.zeros((n,len(model_api.NODE_FEATURES)),np.float64)
                x[:,0]=weights;x[:,1]=np.log1p(degrees);x[:,2]=degrees/max(1,n-1)
                x[:,4]=np.log1p(memberships);x[:,5]=np.log1p(maximum)
                rows=np.repeat(np.arange(n,dtype=np.int64),degrees)
                cols=np.fromiter((v for row in neighbors for v in sorted(row)),dtype=np.int64,count=int(degrees.sum()))
                edges=np.stack((rows,cols))
                sizes=np.asarray(list(map(len,factors)),np.int64)
                factor_rows=np.fromiter((v for factor in factors for v in factor),dtype=np.int64,count=int(sizes.sum()))
                factor_cols=np.repeat(np.arange(len(factors),dtype=np.int64),sizes)
                incidence=np.stack((factor_rows,factor_cols))
                by_node=incidence[:,np.lexsort((factor_cols,factor_rows))]
                self.records[key]=(factors,weights,stats,x,edges,incidence,by_node)
        return self.records[key]


def priority_callback_fast(model,scale=None,device="cpu"):
    """Full original pack/transfer/forward/CPU output, with paid fast static prep."""
    summary_only=isinstance(model,model_api.CheapSummaryValueNet)
    cache=FastStaticPackingCache(summary_only=summary_only)
    def evaluate(graph,views,state):
        if not state.spent_requests:cache.reset()
        packer=model_api.pack_summary_v4 if summary_only else model_api.pack_v4
        batch=packer(graph,views,state,scale=scale,device=device,static_cache=cache)
        with torch.no_grad():native=model(batch)*batch["scale"]
        return controller_api.PriorityEvaluation(tuple(float(v) for v in native.detach().cpu().tolist()),
            semantics="point finite recovery gain, native reward units; identical paid fast cover")
    return evaluate


def classical_priority_fast(mode="upper",inferred_factors=True):
    """Original strong values/masks/order with isolated identical-cover provider."""
    graph_identity=None;factors={}
    def factor_provider(graph,scope):
        nonlocal graph_identity
        if graph_identity is not graph:factors.clear();graph_identity=graph
        key=(tuple(scope.replacements),tuple(scope.resource_cliques))
        if key not in factors:factors[key]=observable_factors_fast(graph,scope)
        return factors[key]
    namespace=dict(controller_api.classical_priority.__globals__)
    namespace["observable_factors"]=factor_provider
    factory=FunctionType(controller_api.classical_priority.__code__,namespace,
        "classical_priority_identical_cover",controller_api.classical_priority.__defaults__,
        controller_api.classical_priority.__closure__)
    original=factory(mode,inferred_factors)
    def evaluate(graph,views,state):
        if not state.spent_requests:factors.clear()
        return original(graph,views,state)
    return evaluate
