#pragma once
#include "flow.hpp"
#include <map>
#include <set>
#include <unordered_map>
namespace barr {

struct Backbone {
    Mask support, merged;
    std::vector<int> color, component;
    std::vector<std::vector<int>> parts;
    bool exact=false;
    std::uint64_t cuts=0;
};
inline Backbone fuse(const Graph&g,const Mask&p,const Mask&q,const Deadline&d) {
    if(!g.feasible(p)||!g.feasible(q)) throw std::invalid_argument("infeasible parent");
    Backbone b; b.support.resize(g.n);b.merged.assign(g.n,0);b.color.assign(g.n,-1);b.component.assign(g.n,-1);
    for(int u=0;u<g.n;++u) { b.support[u]=p[u]||q[u]; if(b.support[u]) b.color[u]=p[u]?0:1; }
    // The intersection is isolated in G[p union q]. Hence assigning it to 0 is safe.
    for(int start=0;start<g.n;++start) if(b.support[start] && b.component[start]<0) {
        int id=int(b.parts.size()); b.parts.emplace_back();auto&vs=b.parts.back();
        vs.push_back(start);b.component[start]=id;
        for(std::size_t i=0;i<vs.size();++i) for(int v:g.adj[vs[i]]) if(b.support[v]) {
            if(b.color[v]==b.color[vs[i]]) throw std::logic_error("invalid parent-union bipartition");
            if(b.component[v]<0) {b.component[v]=id;vs.push_back(v);}
        }
    }
    // Per-component fallback dominates EACH parent even when flow times out.
    for(const auto&vs:b.parts) {
        Weight wp=0,wq=0;long double rp=0,rq=0;
        for(int v:vs) {if(p[v]){wp+=g.w[v];rp+=g.original[v];}if(q[v]){wq+=g.w[v];rq+=g.original[v];}}
        const Mask&take=(wp>wq||(wp==wq && rp>=rq))?p:q;
        for(int v:vs)b.merged[v]=take[v];
    }
    try {
        for(const auto&vs:b.parts) {
            d.check(); auto cut=bipartite_mwis(g,vs,b.color,d);++b.cuts;
            for(int v:vs)b.merged[v]=0;
            for(int v:cut.selected)b.merged[v]=1;
        }
        b.exact=true;
    } catch(const Timeout&) {b.exact=false;}
    if(!g.feasible(b.merged)||g.value(b.merged)<std::max(g.value(p),g.value(q))) throw std::logic_error("fusion dominance failure");
    return b;
}

struct Kernel {
    Graph graph;
    std::vector<int> global_ids, color, outsiders;
    Mask base, parent0, parent1;
    std::vector<double> frequency;
    bool base_backbone_exact=false;
};
inline Kernel make_kernel(const Graph&g,const Backbone&b,const Mask&p,const Mask&q,
                          const std::vector<double>&frequency,const std::vector<int>&outside,
                          int max_vertices,const Deadline&d) {
    if(!b.exact) throw std::invalid_argument("kernel construction requires completed exact fusion");
    if(frequency.size()!=g.w.size())throw std::invalid_argument("frequency length");
    std::set<int> ids, comp;
    for(int v:outside) {
        if(v<0||v>=g.n||b.support[v]||!ids.insert(v).second)throw std::invalid_argument("invalid outsider");
        for(int u:g.adj[v]) if(b.support[u])comp.insert(b.component[u]);
    }
    for(int c:comp) {
        d.check(); for(int v:b.parts[c])ids.insert(v);
        if(int(ids.size())>max_vertices)throw std::length_error("whole-component kernel exceeds cap; never truncate its boundary");
    }
    if(int(ids.size())>max_vertices)throw std::length_error("kernel exceeds cap");
    Kernel k; k.global_ids.assign(ids.begin(),ids.end()); std::vector<int>map(g.n,-1);
    std::vector<Weight>w;std::vector<double>raw;std::vector<std::int64_t>own;
    for(int i=0;i<int(k.global_ids.size());++i){
        int v=k.global_ids[i];map[v]=i;w.push_back(g.w[v]);raw.push_back(g.original[v]);own.push_back(g.owners[v]);
        k.color.push_back(b.color[v]);k.base.push_back(b.merged[v]);k.parent0.push_back(p[v]);k.parent1.push_back(q[v]);k.frequency.push_back(frequency[v]);
        if(!b.support[v])k.outsiders.push_back(i);
    }
    std::vector<std::pair<int,int>>edges;
    for(int i=0;i<int(k.global_ids.size());++i){
        if((i & 255)==0)d.check();
        for(int v:g.adj[k.global_ids[i]])if(map[v]>i)edges.emplace_back(i,map[v]);
    }
    k.graph=Graph(std::move(w),std::move(edges),std::move(raw),std::move(own));k.graph.initial=k.base;k.base_backbone_exact=true;
    // Explicit full boundary check: fixed merged vertices outside k cannot conflict.
    for(int v:k.global_ids)for(int u:g.adj[v])if(map[u]<0 && b.merged[u])throw std::logic_error("unsafe fixed boundary");
    if(!k.graph.feasible(k.base))throw std::logic_error("infeasible kernel base");
    return k;
}
struct Recovery {
    Mask selected;
    Weight lower=0,upper=0;
    bool exact=false;
    std::uint64_t nodes=0,cuts=0,cache_hits=0;
    int components=0,max_component_k=0;
};
struct SearchBudget {
    std::uint64_t max_nodes=128,used=0;
    Deadline deadline;
};
inline Weight sum_mask(const std::vector<int>&vs,std::uint32_t mask,const Graph&g) {
    Weight x=0;for(int i=0;i<int(vs.size());++i)if(mask & (1u<<i))x+=g.w[vs[i]];return x;
}

// Standard branching-on-a-known-bipartite-modulator machinery, specialized here
// to a parental backbone plus omitted vertices. "k" is NOT a minimum OCT claim.
inline Recovery solve_component(const Kernel&k,const std::vector<int>&domain,SearchBudget&budget) {
    const Graph&g=k.graph;
    Recovery r;r.selected.assign(g.n,0);r.components=1;
    std::vector<int>outside,backbone;
    for(int v:domain) {if(k.color[v]<0)outside.push_back(v);else backbone.push_back(v);if(k.base[v]){r.selected[v]=1;r.lower+=g.w[v];}}
    if(outside.size()>24)throw std::invalid_argument("at most 24 outsiders per component supported");
    const int count=int(outside.size());r.max_component_k=count;
    if(count==0 && k.base_backbone_exact) {r.upper=r.lower;r.exact=true;return r;}
    CutSolution root;
    try {
        if(k.base_backbone_exact){root.value=r.lower;for(int v:backbone)if(k.base[v])root.selected.push_back(v);}
        else {root=bipartite_mwis(g,backbone,k.color,budget.deadline);++r.cuts;
            if(root.value>r.lower){std::fill(r.selected.begin(),r.selected.end(),0);for(int v:root.selected)r.selected[v]=1;r.lower=root.value;}}
    } catch(const Timeout&) {
        r.upper=0;for(int v:domain)r.upper+=g.w[v];return r;
    }
    if(count==0){r.upper=r.lower;r.exact=true;return r;}
    std::vector<std::uint32_t>neighbors(count,0);
    for(int i=0;i<count;++i)for(int j=0;j<count;++j)if(g.adjacent(outside[i],outside[j]))neighbors[i]|=1u<<j;
    auto clique_bound=[&](std::uint32_t rem) {
        // Partition the undecided outsiders into cliques; sum of maximum weights
        // is an admissible bound. Never sum individual replacement *gains*.
        std::vector<std::uint32_t>cliques;std::vector<Weight>maxima;
        for(int i=0;i<count;++i)if(rem&(1u<<i)) {
            std::size_t j=0;while(j<cliques.size() && (cliques[j]&neighbors[i])!=cliques[j])++j;
            if(j==cliques.size()){cliques.push_back(1u<<i);maxima.push_back(g.w[outside[i]]);}
            else {cliques[j]|=1u<<i;maxima[j]=std::max(maxima[j],g.w[outside[i]]);}
        }
        return std::accumulate(maxima.begin(),maxima.end(),Weight(0));
    };
    std::unordered_map<std::uint32_t,CutSolution>cache;cache.emplace(0,root);
    auto evaluate=[&](std::uint32_t chosen)->CutSolution {
        auto it=cache.find(chosen);if(it!=cache.end()){++r.cache_hits;return it->second;}
        Mask blocked(g.n,0);
        for(int i=0;i<count;++i)if(chosen&(1u<<i))for(int v:g.adj[outside[i]])blocked[v]=1;
        std::vector<int>active;for(int v:backbone)if(!blocked[v])active.push_back(v);
        auto c=bipartite_mwis(g,active,k.color,budget.deadline);++r.cuts;
        // Bounded cache memory. Different exclusion branches reuse chosen sets.
        if(cache.size()<256)cache.emplace(chosen,c);
        return c;
    };
    struct Node{std::uint32_t chosen,rem;Weight upper;std::uint64_t serial;};
    struct Less {bool operator()(const Node&a,const Node&b)const{
        if(a.upper!=b.upper)return a.upper<b.upper;
        int ca=__builtin_popcount(a.chosen),cb=__builtin_popcount(b.chosen);
        if(ca!=cb)return ca<cb;
        return a.serial>b.serial;
    }};
    std::priority_queue<Node,std::vector<Node>,Less>front;
    std::uint32_t all=(1u<<count)-1;std::uint64_t serial=0;
    front.push({0,all,root.value+clique_bound(all),serial++});
    while(!front.empty()) {
        if(front.top().upper<=r.lower){while(!front.empty())front.pop();break;}
        if(budget.used>=budget.max_nodes||budget.deadline.expired())break;
        Node n=front.top();front.pop();CutSolution cut;
        try{cut=evaluate(n.chosen);}catch(const Timeout&){front.push(n);break;}
        ++budget.used;++r.nodes;
        Weight chosen_w=sum_mask(outside,n.chosen,g);
        Weight value=chosen_w+cut.value;
        if(value>r.lower){
            r.lower=value;std::fill(r.selected.begin(),r.selected.end(),0);
            for(int v:cut.selected)r.selected[v]=1;
            for(int i=0;i<count;++i)if(n.chosen&(1u<<i))r.selected[outside[i]]=1;
        }
        Weight ub=std::min(n.upper,value+clique_bound(n.rem));
        if(ub<=r.lower||n.rem==0)continue;
        int pick=-1;double score=-1;
        for(int i=0;i<count;++i)if(n.rem&(1u<<i)){
            double sc=double(g.w[outside[i]])/(1.+__builtin_popcount(neighbors[i]&n.rem));
            if(sc>score){score=sc;pick=i;}
        }
        std::uint32_t rest=n.rem&~(1u<<pick);
        Weight exclude_ub=std::min(ub,value+clique_bound(rest));
        if(exclude_ub>r.lower)front.push({n.chosen,rest,exclude_ub,serial++});
        std::uint32_t after_include=rest&~neighbors[pick];
        Weight include_ub=std::min(ub,value+g.w[outside[pick]]+clique_bound(after_include));
        if(include_ub>r.lower)front.push({n.chosen|(1u<<pick),after_include,include_ub,serial++});
    }
    r.upper=r.lower;if(!front.empty())r.upper=std::max(r.upper,front.top().upper);
    r.exact=r.upper==r.lower;
    if(!g.feasible(r.selected)||g.value(r.selected)!=r.lower)throw std::logic_error("invalid recovery lower bound");
    return r;
}
inline Recovery recover(const Kernel&k,std::uint64_t max_nodes,const Deadline&d,bool decompose=true) {
    const auto&g=k.graph;Recovery all;all.selected.assign(g.n,0);all.exact=true;
    std::vector<std::vector<int>>parts;
    if(!decompose){parts.emplace_back(g.n);std::iota(parts.back().begin(),parts.back().end(),0);}
    else {
        Mask seen(g.n,0);
        for(int u=0;u<g.n;++u)if(!seen[u]){
            parts.emplace_back(1,u);seen[u]=1;auto&vs=parts.back();
            for(std::size_t i=0;i<vs.size();++i)for(int v:g.adj[vs[i]])if(!seen[v]){seen[v]=1;vs.push_back(v);}
        }
    }
    // Search high-potential components first, but keep bounds for EVERY component.
    auto opportunity=[&](const std::vector<int>&vs){Weight x=0;for(int v:vs)if(k.color[v]<0)x+=g.w[v];return x;};
    std::stable_sort(parts.begin(),parts.end(),[&](const auto&a,const auto&b){return opportunity(a)>opportunity(b);});
    SearchBudget budget{max_nodes,0,d};
    for(const auto&part:parts){
        auto r=solve_component(k,part,budget);
        for(int v:part)all.selected[v]=r.selected[v];
        all.lower+=r.lower;all.upper+=r.upper;all.exact=all.exact&&r.exact;
        all.nodes+=r.nodes;all.cuts+=r.cuts;all.cache_hits+=r.cache_hits;++all.components;
        all.max_component_k=std::max(all.max_component_k,r.max_component_k);
    }
    if(!g.feasible(all.selected)||all.lower<g.value(k.base))throw std::logic_error("nonmonotone kernel result");
    return all;
}
inline Recovery greedy_recover(const Kernel&k,const Deadline&d) {
    const auto&g=k.graph;Recovery r;r.selected=k.base;r.lower=g.value(k.base);r.upper=r.lower;
    for(int v:k.outsiders)r.upper+=g.w[v];
    for(int mode=0;mode<3 && !d.expired();++mode){
        std::vector<int>order=k.outsiders;
        std::sort(order.begin(),order.end(),[&](int a,int b){
            double sa=double(g.w[a])/std::pow(1.+g.adj[a].size(),mode*.5);
            double sb=double(g.w[b])/std::pow(1.+g.adj[b].size(),mode*.5);
            return sa!=sb?sa>sb:a<b;
        });
        Mask s=k.base,added(g.n,0);
        for(int v:order){bool ok=true;for(int u:g.adj[v])if(added[u]){ok=false;break;}
            if(ok){for(int u:g.adj[v])s[u]=0;s[v]=1;added[v]=1;}}
        std::vector<int>all(g.n);std::iota(all.begin(),all.end(),0);
        std::sort(all.begin(),all.end(),[&](int a,int b){return g.w[a]!=g.w[b]?g.w[a]>g.w[b]:a<b;});
        for(int v:all)if(!s[v]){bool ok=true;for(int u:g.adj[v])if(s[u]){ok=false;break;}if(ok)s[v]=1;}
        Weight val=g.value(s);if(val>r.lower){r.lower=val;r.selected=std::move(s);}
    }
    r.exact=r.lower==r.upper;return r;
}
inline Mask lift(const Graph&g,const Backbone&b,const Kernel&k,const Recovery&r) {
    if(r.selected.size()!=k.global_ids.size())throw std::invalid_argument("recovery length");
    Mask s=b.merged;for(int i=0;i<int(k.global_ids.size());++i)s[k.global_ids[i]]=r.selected[i];
    if(!g.feasible(s)||g.value(s)<g.value(b.merged))throw std::logic_error("unsafe lift");
    return s;
}
} // namespace barr
