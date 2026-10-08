#pragma once
#include "graph.hpp"
#include <deque>
namespace barr {
// Independent implementation of standard weighted exchange/descent/ILS ideas.
// No upstream CHILS code, AAW routine, or CHILS state is used here.
class LocalSearch {
    const Graph *g;
    std::deque<int> queue;
    Mask queued;
    std::vector<std::pair<int,bool>> undo;
    bool recording=false;
    int protected_vertex=-1;
    void enqueue(int u) { if(!queued[u]) { queued[u]=1; queue.push_back(u); } }
    void toggle(int u,bool yes) {
        if(bool(s[u])==yes) return;
        if(recording) undo.emplace_back(u,bool(s[u]));
        s[u]=yes; cost+=yes?g->w[u]:-g->w[u]; enqueue(u);
        for(int v:g->adj[u]) { blocks[v]+=yes?1:-1; blocked_weight[v]+=yes?g->w[u]:-g->w[u]; enqueue(v); }
    }
    bool can_eject(int u) const {
        if(protected_vertex<0) return true;
        return !g->adjacent(u,protected_vertex);
    }
    void multi_replace(int u) {
        if(u==protected_vertex || !s[u] || g->adj[u].size()>256) return;
        std::vector<int>candidates;
        for(int v:g->adj[u]) if(!s[v] && blocks[v]==1) candidates.push_back(v);
        if(candidates.size()<2) return;
        std::sort(candidates.begin(),candidates.end(),[&](int a,int b){ return g->w[a]!=g->w[b]?g->w[a]>g->w[b]:a<b; });
        if(candidates.size()>32) candidates.resize(32);
        std::vector<int>best;
        Weight bestw=g->w[u];
        // Several greedy orderings, including forcing each of the first 4 choices.
        // A legal move may insert more than two mutually compatible vertices.
        for(int lead=0;lead<std::min(4,int(candidates.size()));++lead) {
            std::vector<int>sel(1,candidates[lead]); Weight val=g->w[candidates[lead]];
            for(int v:candidates) if(v!=candidates[lead]) {
                bool ok=true; for(int t:sel) if(g->adjacent(v,t)) {ok=false;break;}
                if(ok) {sel.push_back(v);val+=g->w[v];}
            }
            if(val>bestw) {bestw=val;best=std::move(sel);}
        }
        if(!best.empty()) { toggle(u,false); for(int v:best) toggle(v,true); }
    }
public:
    Mask s;
    std::vector<int>blocks;
    std::vector<Weight>blocked_weight;
    Weight cost=0;
    std::uint64_t iterations=0, successful=0;
    int failures=0;
    std::mt19937_64 rng;
    LocalSearch(const Graph&graph,const Mask&initial,std::uint64_t seed)
        :g(&graph),queued(graph.n,0),s(graph.n,0),blocks(graph.n,0),blocked_weight(graph.n,0),rng(seed) { reset(initial); }
    void reset(const Mask&initial) {
        if(!g->feasible(initial)) throw std::invalid_argument("infeasible local-search initial");
        recording=false;undo.clear();protected_vertex=-1;cost=0;
        std::fill(s.begin(),s.end(),0);std::fill(blocks.begin(),blocks.end(),0);std::fill(blocked_weight.begin(),blocked_weight.end(),0);
        queue.clear();std::fill(queued.begin(),queued.end(),0);
        for(int u=0;u<g->n;++u) if(initial[u]) toggle(u,true);
        for(int u=0;u<g->n;++u) enqueue(u);
    }
    void insert(int u) {
        if(s[u]) return;
        for(int v:g->adj[u]) if(s[v]) toggle(v,false);
        toggle(u,true);
    }
    void descent(const Deadline&d) {
        std::uint64_t pops=0;
        while(!queue.empty()) {
            if((pops++ & 127)==0) d.check();
            int u=queue.front();queue.pop_front();queued[u]=0;
            if(!s[u] && g->w[u]>blocked_weight[u] && can_eject(u)) insert(u);
            else if(s[u]) multi_replace(u);
        }
    }
    void greedy_start(double alpha,double noise,const Deadline&d) {
        Mask empty(g->n,0); reset(empty);
        std::vector<std::pair<double,int>>order;order.reserve(g->n);
        std::uniform_real_distribution<double>r(0,1);
        for(int v=0;v<g->n;++v) {
            if((v & 1023)==0) d.check();
            double score=double(g->w[v])/std::pow(1.+g->adj[v].size(),alpha);
            score*=1.+noise*r(rng);order.emplace_back(-score,v);
        }
        std::sort(order.begin(),order.end());
        for(auto e:order) { if((std::uint64_t(e.second)&255)==0)d.check(); if(blocks[e.second]==0)toggle(e.second,true); }
        descent(d);
    }
    bool step(const Deadline&d) {
        d.check(); if(g->n==0) return false;
        // Do not leave an interrupted transaction in the population.
        Weight before=cost;
        undo.clear();recording=true;
        int target=int(rng()%std::uint64_t(g->n));
        try {
            int kicks=1+std::min(7,failures/64);
            if(s[target]) {
                toggle(target,false);
                std::vector<int>near=g->adj[target]; std::shuffle(near.begin(),near.end(),rng);
                for(int v:near) if(!s[v] && kicks-->0) insert(v);
            } else { insert(target);protected_vertex=target; }
            // Sparse occasional multi-site perturbations, not CHILS's queue-threshold rule.
            if((iterations & 15)==15) for(int k=1;k<kicks;++k) {
                int v=int(rng()%std::uint64_t(g->n));
                if(!s[v] && can_eject(v)) insert(v);
            }
            descent(d);protected_vertex=-1;descent(d);
        } catch(const Timeout&) {
            protected_vertex=-1;recording=false;
            for(auto it=undo.rbegin();it!=undo.rend();++it) toggle(it->first,it->second);
            undo.clear();throw;
        }
        recording=false;protected_vertex=-1;++iterations;
        if(cost<before) {
            for(auto it=undo.rbegin();it!=undo.rend();++it) toggle(it->first,it->second);
        }
        undo.clear();
        bool improved=cost>before;
        if(improved){++successful;failures=0;}else ++failures;
        return improved;
    }
    void run(int steps,const Deadline&d) {
        try { descent(d);for(int i=0;i<steps;++i) step(d); } catch(const Timeout&) {}
    }
    bool validate_state()const {
        if(!g->feasible(s) || g->value(s)!=cost) return false;
        for(int u=0;u<g->n;++u) {
            int b=0;Weight w=0;for(int v:g->adj[u]) if(s[v]) {++b;w+=g->w[v];}
            if(b!=blocks[u]||w!=blocked_weight[u])return false;
        }return true;
    }
};
} // namespace barr
