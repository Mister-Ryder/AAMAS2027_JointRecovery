"""Shared CPU-only observable clique cover for V4 models and strong priorities.

No torch, execution kernel, labels or outcome cache is imported. Recipe follows
the deterministic greedy cover used by the V3 capacity family, now shared by
all corresponding V4 priorities so high-order information is matched.
"""
from numbers import Integral

import numpy as np


def conflict_clique_cover(neighbors,weights):
    n=len(neighbors)
    neighbors=tuple(frozenset(row) for row in neighbors)
    if any(any(not isinstance(v,Integral) or isinstance(v,bool) or v<0 or v>=n or v==i
               for v in row) for i,row in enumerate(neighbors)):
        raise ValueError("Exact local conflict IDs required")
    if any(i not in neighbors[j] for i,row in enumerate(neighbors) for j in row):
        raise ValueError("Symmetric local conflicts required")
    weights=np.asarray(weights)
    if weights.shape!=(n,) or not np.all(np.isfinite(weights)) or np.any(weights<0):
        raise ValueError("Finite nonnegative cover-order weights required")
    edges={(i,j) for i,row in enumerate(neighbors) for j in row if i<j}
    ordering=sorted(range(n),key=lambda i:(-weights[i].item(),-len(neighbors[i]),i))
    factors=set(); covered=set()
    for seed in ordering:
        if not neighbors[seed]: continue
        clique=[seed]; allowed=set(neighbors[seed])
        for vertex in ordering:
            if vertex in allowed:
                clique.append(vertex); allowed.intersection_update(neighbors[vertex])
        factor=tuple(sorted(clique))
        if len(factor)>=2:
            factors.add(factor)
            covered.update((u,v) for i,u in enumerate(factor) for v in factor[i+1:])
    factors.update(edges-covered)
    return tuple(sorted(factors))


def observable_factors(graph,scope):
    ids=tuple(scope.replacements)
    if len(ids)!=len(set(ids)) or any(not isinstance(v,Integral) or isinstance(v,bool)
            or v<0 or v>=graph.n for v in ids):
        raise ValueError("Exact distinct recovery vertex IDs required")
    local={v:i for i,v in enumerate(ids)}
    neighbors=tuple(frozenset(local[u] for u in graph.adjacency[v] if u in local) for v in ids)
    factors=set(conflict_clique_cover(neighbors,graph.weights[list(ids)]))
    for raw in scope.resource_cliques:
        if len(raw)!=len(set(raw)) or any(not isinstance(v,Integral) or isinstance(v,bool) or v not in local for v in raw):
            raise ValueError("Resource factor must be a distinct integer subset of this R")
        factor=tuple(sorted(local[v] for v in raw))
        if len(factor)<2 or any(v not in neighbors[u] for i,u in enumerate(factor) for v in factor[i+1:]):
            raise ValueError("Resource factor must be a genuine conflict clique")
        factors.add(factor)
    return neighbors,tuple(sorted(factors))
