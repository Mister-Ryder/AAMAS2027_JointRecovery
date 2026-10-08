#pragma once
#include "graph.hpp"
#include <deque>
namespace barr {
// Dinic with an iterative blocking-path traversal: no call-stack depth depends
// on graph size. A timed-out flow is NEVER reported as an exact cut.
class Dinic {
    struct Edge { int to, rev; Weight cap; };
    std::vector<std::vector<Edge>> a;
    std::vector<int> level, ptr;
    std::uint64_t ops=0;
    const Deadline &deadline;
    void tick() { if((++ops & 1023)==0) deadline.check(); }
    bool bfs(int s,int t) {
        deadline.check(); std::fill(level.begin(),level.end(),-1);
        std::vector<int>q(a.size()); int head=0,tail=0; q[tail++]=s; level[s]=0;
        while(head<tail) { int v=q[head++]; for(const auto&e:a[v]) {
            tick(); if(e.cap>0 && level[e.to]<0) { level[e.to]=level[v]+1; q[tail++]=e.to; }
        }}
        return level[t]>=0;
    }
    Weight path(int s,int t) {
        std::vector<int> vs(1,s), es;
        while(!vs.empty()) {
            tick(); int v=vs.back();
            if(v==t) {
                Weight f=MAX_TOTAL;
                for(std::size_t i=0;i<es.size();++i) f=std::min(f,a[vs[i]][es[i]].cap);
                for(std::size_t i=0;i<es.size();++i) { auto&e=a[vs[i]][es[i]]; e.cap-=f; a[e.to][e.rev].cap+=f; }
                return f;
            }
            while(ptr[v]<int(a[v].size())) {
                const auto&e=a[v][ptr[v]];
                if(e.cap>0 && level[e.to]==level[v]+1) break;
                ++ptr[v]; tick();
            }
            if(ptr[v]==int(a[v].size())) {
                level[v]=-1; vs.pop_back();
                if(!es.empty()) { es.pop_back(); if(!vs.empty()) ++ptr[vs.back()]; }
            } else { es.push_back(ptr[v]); vs.push_back(a[v][ptr[v]].to); }
        }
        return 0;
    }
public:
    Dinic(int n,const Deadline&d):a(n),level(n),ptr(n),deadline(d){}
    void add(int u,int v,Weight c) {
        if(c<0) throw std::logic_error("negative capacity");
        Edge x{v,int(a[v].size()),c}, y{u,int(a[u].size()),0};
        a[u].push_back(x);a[v].push_back(y);
    }
    Weight maxflow(int s,int t) {
        Weight ans=0;
        while(bfs(s,t)) { std::fill(ptr.begin(),ptr.end(),0); Weight f; while((f=path(s,t))>0) ans+=f; }
        deadline.check(); return ans;
    }
    Mask reachable(int s) {
        Mask seen(a.size(),0); std::vector<int>q(1,s); seen[s]=1;
        for(std::size_t i=0;i<q.size();++i) for(const auto&e:a[q[i]]) {
            tick(); if(e.cap>0 && !seen[e.to]) { seen[e.to]=1;q.push_back(e.to); }
        }
        deadline.check(); return seen;
    }
};
struct CutSolution { Weight value=0; std::vector<int> selected; };
// Induced active graph must have the supplied bipartition (0/1).
inline CutSolution bipartite_mwis(const Graph&g,const std::vector<int>&vs,
                                const std::vector<int>&color,const Deadline&d) {
    d.check(); if(color.size()!=g.w.size()) throw std::invalid_argument("color length");
    CutSolution result;
    if(vs.empty()) return result;
    if(vs.size()<=2) {
        for(int u:vs) if(u<0 || u>=g.n || (color[u]!=0 && color[u]!=1)) throw std::invalid_argument("invalid tiny cut domain");
        if(vs.size()==1) return {g.w[vs[0]],{vs[0]}};
        if(vs[0]==vs[1]) throw std::invalid_argument("duplicate tiny cut vertex");
        if(g.adjacent(vs[0],vs[1])) {
            if(color[vs[0]]==color[vs[1]]) throw std::logic_error("invalid tiny bipartition");
            int u=g.w[vs[0]]>=g.w[vs[1]]?vs[0]:vs[1]; return {g.w[u],{u}};
        }
        return {g.w[vs[0]]+g.w[vs[1]],{vs[0],vs[1]}};
    }
    std::vector<int>map(g.n,-1);
    Weight total=0;
    for(int i=0;i<int(vs.size());++i) {
        int u=vs[i];
        if(u<0 || u>=g.n || map[u]>=0 || (color[u]!=0 && color[u]!=1)) throw std::invalid_argument("invalid bipartite domain");
        map[u]=i; total+=g.w[u];
    }
    Dinic flow(int(vs.size())+2,d); int source=int(vs.size()),sink=source+1;
    for(int i=0;i<int(vs.size());++i) {
        if((i & 255)==0) d.check();
        int u=vs[i];
        if(color[u]==0) flow.add(source,i,g.w[u]); else flow.add(i,sink,g.w[u]);
        for(int v:g.adj[u]) if(map[v]>=0) {
            if(color[u]==color[v]) throw std::logic_error("backbone is not bipartite");
            if(color[u]==0) flow.add(i,map[v],total+1);
        }
    }
    Weight cut=flow.maxflow(source,sink);
    Mask seen=flow.reachable(source);
    for(int i=0;i<int(vs.size());++i) if((color[vs[i]]==0 && seen[i]) || (color[vs[i]]==1 && !seen[i])) {
        result.selected.push_back(vs[i]); result.value+=g.w[vs[i]];
    }
    if(result.value!=total-cut) throw std::logic_error("cut reconstruction mismatch");
    return result;
}
} // namespace barr
