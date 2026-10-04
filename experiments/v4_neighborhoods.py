"""Outcome-free wider coordinated neighborhoods for V4 development.

Destroy/repair is conventional LNS, not a novel wrapper. The retained research
question is learning mutually compatible finite recovery within these fixed
actions. Optional releases change an ACTION version/base, never silently a
nested scope of the original two-insert action.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from collections import deque

import numpy as np

from joint_recovery.core import is_feasible
from joint_recovery.v4_budgeted_recovery import RecoveryScope,objective


def checked_ids(values):
    values=tuple(values)
    if any(not isinstance(v,Integral) or isinstance(v,bool) for v in values):
        raise ValueError("Coordination IDs must be exact integers")
    ids=tuple(sorted(map(int,values)))
    if len(ids)!=len(set(ids)):raise ValueError("Coordination IDs must be distinct")
    return ids


@dataclass(frozen=True)
class CoordinationAction:
    inserts: tuple
    releases: tuple

    def __post_init__(self):
        object.__setattr__(self,"inserts",checked_ids(self.inserts))
        object.__setattr__(self,"releases",checked_ids(self.releases))
        if len(self.inserts)>8 or not (self.inserts or self.releases):
            raise ValueError("A bounded nonempty coordinated action is required")

    @property
    def identity(self):return self.inserts,self.releases


class NeighborhoodScopeCache:
    """Fixed C/D/base; restorable incumbent first, then weight/ID prefixes."""
    def __init__(self,graph,selected,resource_cliques=()):
        self.graph=graph;self.selected=frozenset(selected)
        self.resource_cliques=tuple(checked_ids(c) for c in resource_cliques)
        self._actions={};self._scopes={};self._clique_index=None

    def scope(self,index,action,cap):
        if isinstance(cap,bool) or not isinstance(cap,Integral) or cap<0:
            raise ValueError("An exact nonnegative prefix cap is required")
        if not isinstance(action,CoordinationAction):raise TypeError("Explicit coordination action required")
        graph=self.graph;S=self.selected
        if index not in self._actions:
            C=frozenset(action.inserts);extra=frozenset(action.releases)
            if any(v<0 or v>=graph.n or v in S for v in C) or not is_feasible(graph,C):
                raise ValueError("Commitments must be valid compatible unselected nodes")
            if not extra<=S:raise ValueError("Optional releases must belong to the original incumbent")
            mandatory=frozenset(v for c in C for v in graph.adjacency[c]&S)
            D=mandatory|extra;B=(S-D)|C
            if not is_feasible(graph,B):raise ValueError("Fixed committed base is infeasible")
            if len(set(int(graph.agents[v]) for v in D|C))<2:
                raise ValueError("This pilot requires genuinely cross-partition adjustments")
            candidates=set(D)
            for v in D:candidates.update(graph.adjacency[v])
            eligible={v for v in candidates if v not in B and not(graph.adjacency[v]&B)}
            known=eligible&S
            order=tuple(sorted(known))+tuple(sorted(eligible-known,key=lambda v:(-graph.weights[v].item(),v)))
            self._actions[index]=(action.identity,B,D,order,objective(graph,C)-objective(graph,D))
        identity,B,D,eligible,q=self._actions[index]
        if identity!=action.identity:raise ValueError("An action version cannot mutate its commitment/releases")
        key=(index,int(cap))
        if key not in self._scopes:
            R=eligible[:int(cap)];pool=frozenset(R);cliques=[]
            # Index is lazily constructed once; its full preparation is inside
            # whichever decision/pilot first requests resource factors.
            if self.resource_cliques and self._clique_index is None:
                index_by_vertex={}
                for cid,factor in enumerate(self.resource_cliques):
                    if any(v<0 or v>=graph.n for v in factor):raise ValueError("Bad original factor ID")
                    for v in factor:index_by_vertex.setdefault(v,[]).append(cid)
                self._clique_index=index_by_vertex
            touched=set(cid for v in R for cid in (self._clique_index or {}).get(v,()))
            for cid in sorted(touched):
                factor=tuple(v for v in self.resource_cliques[cid] if v in pool)
                if len(factor)>1:
                    if any(v not in graph.adjacency[u] for j,u in enumerate(factor) for v in factor[j+1:]):
                        raise ValueError("Supplied factor is not a genuine conflict clique")
                    cliques.append(factor)
            self._scopes[key]=RecoveryScope(index,action.inserts,int(cap),B,D,R,len(eligible),q,tuple(cliques))
        return self._scopes[key]

    def initial_known_warm(self,scope):
        # This is an already-known portion of the original feasible S. It is
        # never an unexecuted label or a newly solved greedy repair.
        warm=self.selected&frozenset(scope.replacements)
        if not is_feasible(self.graph,scope.base|warm):raise ValueError("Known original recovery is incompatible")
        return warm


def release_actions(graph,selected,seed=0,max_actions=16,release_sizes=(8,16,32,64)):
    """Declared graph-boundary BFS releases, independent of repair outcomes.

    A shared unselected neighbor connects two currently selected assignments
    in the boundary graph. Anchors and BFS tie order come from the fixed input
    seed, not a model or solver. These are pure destroy/repair controls first;
    later forced commitments must be separately declared action versions.
    """
    if max_actions<1 or not release_sizes:raise ValueError("Finite nonempty action grid required")
    S=frozenset(selected)
    if not is_feasible(graph,S):raise ValueError("Original incumbent is infeasible")
    rng=np.random.default_rng(seed);anchors=np.asarray(sorted(S),dtype=np.int64)
    rng.shuffle(anchors);actions=[];seen=set()
    for target in release_sizes:
        if not isinstance(target,Integral) or isinstance(target,bool) or target<1:
            raise ValueError("Positive integer release sizes required")
        quota=max(1,max_actions//len(release_sizes));used=0
        for anchor in anchors:
            D=set();queue=deque([int(anchor)]);queued={int(anchor)}
            while queue and len(D)<target:
                v=queue.popleft();D.add(v)
                adjacent=set()
                for u in graph.adjacency[v]-S:adjacent.update(graph.adjacency[u]&S)
                neighbors=sorted(adjacent-D-queued)
                if neighbors:
                    rng.shuffle(neighbors);queue.extend(neighbors);queued.update(neighbors)
            if len(set(int(graph.agents[v]) for v in D))<2:continue
            action=CoordinationAction((),tuple(D))
            if action.identity in seen:continue
            seen.add(action.identity);actions.append(action);used+=1
            if used>=quota:break
        if len(actions)>=max_actions:break
    return actions[:max_actions]


def coordination_cells(graph,selected,seed=0):
    """Eleven predeclared C(0/2/4/8) x E(0/16/64) cells, no empty no-op.

    Conditional insertions use a shared immediate-gain pool, with fixed random
    exploration; releases use the incumbent boundary graph. No oracle/model
    score is used. Aliased actual descriptors remain in coverage as aliases.
    """
    S=frozenset(selected)
    if not is_feasible(graph,S):raise ValueError("Original state must be feasible")
    rng=np.random.default_rng(seed)
    available=[v for v in range(graph.n) if v not in S]
    ranked=sorted(available,key=lambda v:(-(objective(graph,(v,))-
                  objective(graph,graph.adjacency[v]&S)),v))
    leading=set(ranked[:32])
    remaining=[v for v in available if v not in leading]
    rng.shuffle(remaining);pool=ranked[:32]+remaining[:32]
    commits=[]
    for v in pool:
        if not graph.adjacency[v]&frozenset(commits):commits.append(v)
        if len(commits)>=8:break
    anchors=sorted(S);rng.shuffle(anchors);result=[];seen={}
    for target_c in (0,2,4,8):
        for target_e in (0,16,64):
            if target_c==target_e==0:continue
            cell=dict(target_commitments=target_c,target_extra_releases=target_e)
            if len(commits)<target_c:
                result.append((None,dict(cell,status="insufficient_compatible_commitments")));continue
            C=tuple(commits[:target_c]);mandatory=frozenset(v for c in C for v in graph.adjacency[c]&S)
            E=set();queue=deque(sorted(mandatory) or anchors[:1]);queued=set(queue)
            while len(E)<target_e:
                if not queue:
                    remaining_anchors=[v for v in anchors if v not in queued]
                    if not remaining_anchors:break
                    queue.append(remaining_anchors[0]);queued.add(remaining_anchors[0])
                v=queue.popleft()
                if v not in mandatory:E.add(v)
                adjacent=set()
                for u in graph.adjacency[v]-S:adjacent.update(graph.adjacency[u]&S)
                next_ids=sorted(adjacent-queued);rng.shuffle(next_ids)
                queue.extend(next_ids);queued.update(next_ids)
            if not C and not E:
                result.append((None,dict(cell,status="no_incumbent_releases_available")));continue
            action=CoordinationAction(C,tuple(E))
            if len(set(int(graph.agents[v]) for v in mandatory|E|frozenset(C)))<2:
                result.append((None,dict(cell,status="single_partition_only")));continue
            cell.update(actual_commitments=len(C),actual_extra_releases=len(E),
                        mandatory_displacements=len(mandatory))
            if action.identity in seen:
                result.append((None,dict(cell,status="actual_action_alias",alias_cell=seen[action.identity])));continue
            seen[action.identity]=len(result)
            result.append((action,dict(cell,status="prepared")))
    if len(result)!=11:raise AssertionError("Generalized action coverage grid changed")
    return result
