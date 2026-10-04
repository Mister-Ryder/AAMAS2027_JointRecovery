"""Budget-conditioned joint recovery with a q-independent structural encoder.

NEW pre-fit prototype. Encoding R/factors/actual warm once per unique unit is
causal factorization for the shared R-only repair interface, not a claim that
memoization is novel. Native q is a known readout offset, never learned context.
All preparation/transfers/inference/cache lookups are paid inside the callback.
"""
from __future__ import annotations

from math import isfinite,log1p

import numpy as np
import torch
from torch import nn

from . import v4_model as original
from .v4_model_fast import FastStaticPackingCache
from .v4_budgeted_recovery import RequestView,PriorityEvaluation,_check_recovery,objective


HEAD_FEATURES=("seconds","nodes","iterations","log_native_amount",
    "log_remaining_seconds","normalized_actual_warm_value","actual_warm_fraction")
UNIT_TENSOR_KEYS=("x","weights","warm","node_request","edge_index","incidence",
    "incidence_by_node","factor_request","ptr","summary")


class DecisionEmbeddingCache:
    """Only detached observable encodings; one graph/decision/model version."""
    def __init__(self):self.graph=None;self.model=None;self.model_version=None;self.records={}
    def reset(self):self.graph=None;self.model=None;self.model_version=None;self.records.clear()
    def bind(self,graph,model):
        version=tuple((parameter._version,str(parameter.device),str(parameter.dtype)) for parameter in model.parameters())
        if self.graph is not graph or self.model is not model or self.model_version!=version:
            self.reset();self.graph=graph;self.model=model;self.model_version=version


def _unit_pack(graph,views,state,scale,device,cache):
    if not views:return None
    raw=original.pack_v4(graph,views,state,scale=scale,device=device,static_cache=cache)
    result={name:raw[name] for name in UNIT_TENSOR_KEYS}
    result.update(request_count=raw['request_count'],node_count=raw['node_count'],
        factor_count=raw['factor_count'],node_ids=raw['node_ids'],factors=raw['factors'])
    return result


def pack_factorized_v4(graph,views,state,scale=None,device='cpu',static_cache=None,
        embedding_cache=None,use_warm_start=True):
    """Actual request order preserved, with explicit request-to-unit mapping.

    Unit key is ordered R, genuine/inferred factors, observed warm and scale.
    C/E/action IDs/q/workpoint/remaining are absent from structural encoding.
    q appears ONLY as the known output offset. Expected calibrated cost remains
    the controller's allocation quantity, not a causal recovery head input.
    """
    views=tuple(views)
    if not views:raise ValueError('At least one actual request required')
    scale=original.normalization_scale(graph) if scale is None else float(scale)
    if not isfinite(scale) or scale<=0 or not isfinite(state.remaining_seconds) or state.remaining_seconds<0:
        raise ValueError('Positive finite scale and remaining time required')
    cache=FastStaticPackingCache() if static_cache is None else static_cache
    if embedding_cache is not None and embedding_cache.graph is not graph:
        raise ValueError('Embedding cache must be bound to this graph/model before packing')
    mapping={};unit_keys=[];unit_views=[];unit_records=[];unit_sizes=[]
    request_units=[];head=[];summaries=[];immediate=[];metadata=[];factor_metadata=[]
    decoder_nodes=[];decoder_request=[];weights=[];incidence=[];by_node=[];ptr=[0]
    unit_ptr=[0];factor_offset=0;validated={}
    for index,view in enumerate(views):
        scope,workpoint=view.request.scope,view.request.workpoint
        warm_key=(scope.base,scope.replacements,frozenset(view.warm_start))
        if warm_key not in validated:validated[warm_key]=_check_recovery(graph,scope,view.warm_start)
        actual=validated[warm_key];known=actual if use_warm_start else frozenset()
        record=cache.get(graph,scope,scale);factors,w,stats,_,_,local_incidence,local_by_node=record
        key=(tuple(scope.replacements),factors,known,scale)
        if key not in mapping:
            mapping[key]=len(unit_keys);unit_keys.append(key)
            unit_views.append(RequestView(view.request,known));unit_records.append(record)
            unit_sizes.append(len(w));unit_ptr.append(unit_ptr[-1]+len(w))
        unit=mapping[key];request_units.append(unit)
        head.append((float(workpoint.kind=='seconds'),float(workpoint.kind=='nodes'),
            float(workpoint.kind=='iterations'),log1p(workpoint.amount),log1p(state.remaining_seconds),
            float(objective(graph,known))/scale,len(known)/max(1,len(w))))
        summaries.append(stats);immediate.append(float(scope.immediate_gain)/scale)
        metadata.append(tuple(scope.replacements));factor_metadata.append(factors)
        n=len(w);offset=ptr[-1]
        decoder_nodes.append(np.arange(unit_ptr[unit],unit_ptr[unit]+n,dtype=np.int64))
        decoder_request.append(np.full(n,index,np.int64));weights.append(w)
        delta=np.asarray([[offset],[factor_offset]],np.int64)
        incidence.append(local_incidence+delta);by_node.append(local_by_node+delta)
        ptr.append(offset+n);factor_offset+=len(factors)
    # The cached unit state is conditional on actual observed warm; changed
    # warm keys are missing and re-encoded. No output/schedule cache is read.
    missing=[i for i,key in enumerate(unit_keys) if embedding_cache is None or key not in embedding_cache.records]
    units=_unit_pack(graph,tuple(unit_views[i] for i in missing),state,scale,device,cache)
    arrays=dict(request_unit=np.asarray(request_units,np.int64),unit_node=np.concatenate(decoder_nodes),
        node_request=np.concatenate(decoder_request),weights=np.concatenate(weights).astype(np.float32),
        context=np.asarray(head,np.float32),summary=np.asarray(summaries,np.float32),
        immediate=np.asarray(immediate,np.float32),incidence=np.concatenate(incidence,axis=1),
        incidence_by_node=np.concatenate(by_node,axis=1),ptr=np.asarray(ptr,np.int64),
        best_gain=np.full(len(views),float(state.best_gain)/scale,np.float32),
        scale=np.full(len(views),scale,np.float32))
    if any(not np.isfinite(value).all() for value in arrays.values()):raise ValueError('Finite observable features required')
    result={name:torch.as_tensor(value,device=device) for name,value in arrays.items()}
    result.update(units=units,unit_keys=tuple(unit_keys),unit_sizes=tuple(unit_sizes),
        new_unit_indices=tuple(missing),unit_count=len(unit_keys),request_count=len(views),
        node_count=ptr[-1],factor_count=factor_offset,node_ids=tuple(metadata),factors=tuple(factor_metadata),
        request_keys=tuple(view.request.key for view in views),head_feature_names=HEAD_FEATURES)
    return result


class FactorizedRecoveryValueNet(nn.Module):
    """Predeclared h32/L2 structural encoder and matched finite-recovery heads."""
    def __init__(self,hidden=32,layers=2,readout='capacity',use_warm_start=True):
        super().__init__()
        if hidden<1 or layers<1 or readout not in ('capacity','free'):
            raise ValueError('Positive dimensions and capacity/free readout required')
        self.hidden,self.layers,self.readout,self.use_warm_start=hidden,layers,readout,use_warm_start
        self.embed=nn.Linear(len(original.NODE_FEATURES),hidden)
        self.self_layers=nn.ModuleList(nn.Linear(hidden,hidden) for _ in range(layers))
        self.edge_layers=nn.ModuleList(nn.Linear(hidden,hidden,bias=False) for _ in range(layers))
        self.factor_layers=nn.ModuleList(nn.Linear(hidden,hidden,bias=False) for _ in range(layers))
        width=2*hidden+len(HEAD_FEATURES)+len(original.SUMMARY_FEATURES)
        self.occupancy_head=nn.Sequential(nn.Linear(width,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.free_head=nn.Sequential(nn.Linear(width,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def encode_units(self,units):
        x=units['x']
        if not self.use_warm_start:x=x.clone();x[:,3]=0.
        h=torch.relu(self.embed(x));source,destination=units['edge_index'];node,factor=units['incidence']
        degree=original._sum(torch.ones_like(source,dtype=h.dtype),destination,units['node_count']).clamp_min(1.)
        sizes=original._sum(torch.ones_like(factor,dtype=h.dtype),factor,units['factor_count']).clamp_min(1.)
        memberships=original._sum(torch.ones_like(node,dtype=h.dtype),node,units['node_count']).clamp_min(1.)
        for self_layer,edge_layer,factor_layer in zip(self.self_layers,self.edge_layers,self.factor_layers):
            message=original._sum(edge_layer(h[source]),destination,units['node_count'])/degree[:,None]
            pooled=original._sum(h[node],factor,units['factor_count'])/sizes[:,None]
            relay=original._sum(factor_layer(pooled[factor]),node,units['node_count'])/memberships[:,None]
            h=torch.relu(self_layer(h)+message+relay)
        assignment=units['node_request'];count=units['request_count']
        counts=original._sum(torch.ones_like(assignment,dtype=h.dtype),assignment,count).clamp_min(1.)
        pooled=original._sum(h,assignment,count)/counts[:,None]
        total=original._sum(units['weights'],assignment,count)
        weighted=original._sum(h*units['weights'][:,None],assignment,count)/total.clamp_min(1e-12)[:,None]
        return h,pooled,weighted

    def _representations(self,batch,embedding_cache):
        new=batch['new_unit_indices']
        if embedding_cache is not None and torch.is_grad_enabled():
            raise ValueError('Observable embedding cache is inference-only')
        if embedding_cache is not None:
            version=tuple((parameter._version,str(parameter.device),str(parameter.dtype)) for parameter in self.parameters())
            if embedding_cache.model is not self or embedding_cache.model_version!=version:
                raise ValueError('Embedding cache belongs to a different or changed model')
        representations={}
        if new:
            h,pooled,weighted=self.encode_units(batch['units']);offset=0
            for local,index in enumerate(new):
                size=batch['unit_sizes'][index]
                record=(h[offset:offset+size],pooled[local:local+1],weighted[local:local+1]);offset+=size
                representations[index]=record
                if embedding_cache is not None:embedding_cache.records[batch['unit_keys'][index]]=tuple(v.detach() for v in record)
        for index,key in enumerate(batch['unit_keys']):
            if index not in representations:
                if embedding_cache is None or key not in embedding_cache.records:
                    raise ValueError('Missing paid structural encoding')
                representations[index]=embedding_cache.records[key]
        rows=[representations[i] for i in range(batch['unit_count'])]
        return tuple(torch.cat([row[field] for row in rows],dim=0) for field in range(3))

    def forward(self,batch,return_details=False,embedding_cache=None):
        h,pooled,weighted=self._representations(batch,embedding_cache)
        context=batch['context']
        if not self.use_warm_start:context=context.clone();context[:,-2:]=0.
        assignment=batch['node_request'];unit=batch['request_unit'];expanded=h[batch['unit_node']]
        free_recovery=None
        if self.readout=='free':
            features=torch.cat((pooled[unit],weighted[unit],context,batch['summary']),dim=1)
            free_recovery=self.free_head(features).squeeze(-1)
            free_recovery=free_recovery*(batch['ptr'][1:]>batch['ptr'][:-1]).to(free_recovery.dtype)
            if not return_details:return (batch['immediate']+free_recovery).clamp_min(0.)
        features=torch.cat((expanded,pooled[unit][assignment],context[assignment],batch['summary'][assignment]),dim=1)
        logits=self.occupancy_head(features).squeeze(-1);proposed=torch.sigmoid(logits)
        occupancy=original.capacity_project_sparse(proposed,batch)
        recovery=original._sum(occupancy*batch['weights'],assignment,batch['request_count'])
        readout=recovery if free_recovery is None else free_recovery
        raw=batch['immediate']+readout
        details=dict(scores=raw.clamp_min(0.),raw_gain=raw,recovery=readout,
            capacity_recovery=recovery,logits=logits,proposed=proposed,occupancy=occupancy,
            encoded_units=len(batch['new_unit_indices']))
        return details if return_details else details['scores']

    def parameter_counts(self):
        inactive=self.free_head if self.readout=='capacity' else self.occupancy_head
        total=sum(p.numel() for p in self.parameters())
        return dict(stored=total,active_scoring=total-sum(p.numel() for p in inactive.parameters()))


class FactorizedSummaryValueNet(nn.Module):
    """Matched q-separated cheap learned-summary control, without graph encoder."""
    def __init__(self,hidden=32,use_warm_start=True):
        super().__init__();self.hidden,self.use_warm_start=hidden,use_warm_start
        width=len(HEAD_FEATURES)+len(original.SUMMARY_FEATURES)
        self.head=nn.Sequential(nn.Linear(width,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,batch,return_details=False):
        context=batch['context']
        if not self.use_warm_start:context=context.clone();context[:,-2:]=0.
        recovery=self.head(torch.cat((context,batch['summary']),dim=1)).squeeze(-1)
        recovery=recovery*(batch['ptr'][1:]>batch['ptr'][:-1]).to(recovery.dtype)
        raw=batch['immediate']+recovery
        details=dict(scores=raw.clamp_min(0.),raw_gain=raw,recovery=recovery,logits=None)
        return details if return_details else details['scores']

    def parameter_counts(self):
        count=sum(parameter.numel() for parameter in self.parameters())
        return dict(stored=count,active_scoring=count)


def pack_factorized_summary(graph,views,state,scale=None,device='cpu',static_cache=None,use_warm_start=True):
    """Scalar control prepares only its used summaries/context, never graph tensors."""
    views=tuple(views)
    if not views:raise ValueError('At least one actual request required')
    scale=original.normalization_scale(graph) if scale is None else float(scale)
    if not isfinite(scale) or scale<=0 or not isfinite(state.remaining_seconds) or state.remaining_seconds<0:
        raise ValueError('Positive finite scale and remaining time required')
    cache=FastStaticPackingCache(summary_only=True) if static_cache is None else static_cache
    context=[];summary=[];immediate=[];ptr=[0];validated={}
    for view in views:
        scope,workpoint=view.request.scope,view.request.workpoint
        key=(scope.base,scope.replacements,frozenset(view.warm_start))
        if key not in validated:validated[key]=_check_recovery(graph,scope,view.warm_start)
        known=validated[key] if use_warm_start else frozenset()
        _,weights,stats,_,_,_,_=cache.get(graph,scope,scale)
        context.append((float(workpoint.kind=='seconds'),float(workpoint.kind=='nodes'),
            float(workpoint.kind=='iterations'),log1p(workpoint.amount),log1p(state.remaining_seconds),
            float(objective(graph,known))/scale,len(known)/max(1,len(weights))))
        summary.append(stats);immediate.append(float(scope.immediate_gain)/scale);ptr.append(ptr[-1]+len(weights))
    arrays=dict(context=np.asarray(context,np.float32),summary=np.asarray(summary,np.float32),
        immediate=np.asarray(immediate,np.float32),ptr=np.asarray(ptr,np.int64),
        best_gain=np.full(len(views),float(state.best_gain)/scale,np.float32),scale=np.full(len(views),scale,np.float32))
    if any(not np.isfinite(value).all() for value in arrays.values()):raise ValueError('Finite observable features required')
    result={key:torch.as_tensor(value,device=device) for key,value in arrays.items()}
    result.update(request_count=len(views),request_keys=tuple(view.request.key for view in views),head_feature_names=HEAD_FEATURES)
    return result


def priority_callback_factorized(model,scale=None,device='cpu'):
    """Paid unique-unit preparation/transfer/encoding/decoder/CPU output."""
    summary=isinstance(model,FactorizedSummaryValueNet)
    static=FastStaticPackingCache(summary_only=summary);embeddings=DecisionEmbeddingCache()
    def evaluate(graph,views,state):
        if not state.spent_requests:static.reset();embeddings.reset()
        embeddings.bind(graph,model)
        batch=(pack_factorized_summary(graph,views,state,scale=scale,device=device,static_cache=static,
            use_warm_start=model.use_warm_start) if summary else
            pack_factorized_v4(graph,views,state,scale=scale,device=device,static_cache=static,
            embedding_cache=embeddings,use_warm_start=model.use_warm_start))
        with torch.no_grad():
            prediction=model(batch) if summary else model(batch,embedding_cache=embeddings)
            native=prediction*batch['scale']
        return PriorityEvaluation(tuple(float(value) for value in native.detach().cpu().tolist()),
            semantics='q-offset point finite joint recovery, native reward units')
    return evaluate


def merge_factorized_batches(batches):
    """Offline whole-state merge; preserves request scale and ranking boundaries."""
    batches=tuple(batches)
    if not batches or any(batch['new_unit_indices']!=tuple(range(batch['unit_count'])) for batch in batches):
        raise ValueError('Only uncached full-unit training batches may be merged')
    unit_tensors={name:[] for name in UNIT_TENSOR_KEYS};outer={name:[] for name in (
        'request_unit','unit_node','node_request','weights','context','summary','immediate',
        'incidence','incidence_by_node','ptr','best_gain','scale')}
    request_offset=decoder_offset=factor_offset=unit_offset=unit_node_offset=unit_factor_offset=0
    keys=[];sizes=[];node_ids=[];factors=[];request_keys=[];unit_metadata=[];unit_factors=[]
    for batch in batches:
        units=batch['units']
        for name in UNIT_TENSOR_KEYS:
            value=units[name]
            if name=='node_request':value=value+unit_offset
            elif name=='factor_request':value=value+unit_offset
            elif name=='edge_index':value=value+unit_node_offset
            elif name in ('incidence','incidence_by_node'):
                value=value+value.new_tensor([[unit_node_offset],[unit_factor_offset]])
            elif name=='ptr':value=(value if not unit_tensors[name] else value[1:])+unit_node_offset
            unit_tensors[name].append(value)
        for name in outer:
            value=batch[name]
            if name=='request_unit':value=value+unit_offset
            elif name=='unit_node':value=value+unit_node_offset
            elif name=='node_request':value=value+request_offset
            elif name in ('incidence','incidence_by_node'):
                value=value+value.new_tensor([[decoder_offset],[factor_offset]])
            elif name=='ptr':value=(value if not outer[name] else value[1:])+decoder_offset
            outer[name].append(value)
        keys.extend(batch['unit_keys']);sizes.extend(batch['unit_sizes']);node_ids.extend(batch['node_ids'])
        factors.extend(batch['factors']);request_keys.extend(batch['request_keys'])
        unit_metadata.extend(units['node_ids']);unit_factors.extend(units['factors'])
        request_offset+=batch['request_count'];decoder_offset+=batch['node_count'];factor_offset+=batch['factor_count']
        unit_offset+=batch['unit_count'];unit_node_offset+=units['node_count'];unit_factor_offset+=units['factor_count']
    merged_units={name:torch.cat(value,dim=1 if name in ('edge_index','incidence','incidence_by_node') else 0)
                  for name,value in unit_tensors.items()}
    merged_units.update(request_count=unit_offset,node_count=unit_node_offset,factor_count=unit_factor_offset,
        node_ids=tuple(unit_metadata),factors=tuple(unit_factors))
    result={name:torch.cat(value,dim=1 if name in ('incidence','incidence_by_node') else 0) for name,value in outer.items()}
    result.update(units=merged_units,unit_keys=tuple(keys),unit_sizes=tuple(sizes),
        new_unit_indices=tuple(range(unit_offset)),unit_count=unit_offset,request_count=request_offset,
        node_count=decoder_offset,factor_count=factor_offset,node_ids=tuple(node_ids),factors=tuple(factors),
        request_keys=tuple(request_keys),head_feature_names=HEAD_FEATURES)
    return result


execution_targets=original.execution_targets
recovery_value_loss=original.recovery_value_loss
merge_targets=original.merge_v4_targets
