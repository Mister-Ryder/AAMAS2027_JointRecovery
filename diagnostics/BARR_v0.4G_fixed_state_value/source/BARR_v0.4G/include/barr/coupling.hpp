#pragma once
// BARR v0.3: exact response compilation and width-bounded max-sum elimination.
// Variable elimination, min-fill and bipartite MWIS are classical. The object
// compiled here is the opportunity-cost interaction induced by a frozen BARR
// backbone. See docs/RESEARCH_ZH.md for the proposed research claim and limits.
#include "recombination.hpp"
#include <functional>
#include <memory>
#include <tuple>

namespace barr {
constexpr Weight FACTOR_NEG = -(Weight(1) << 62);
struct FactorLimit : std::runtime_error { using std::runtime_error::runtime_error; };
struct FactorOptions {
    int max_width=10, max_boundary=10, max_variables=128;
    std::uint64_t max_entries=262144;
};
struct FactorStats {
    int variables=0,components=0,max_boundary=0,induced_width=-1;
    int interaction_components=0,max_component_variables=0;
    std::uint64_t planned_entries=0,compiled_entries=0,dp_entries=0,cuts=0;
    double structure_seconds=0,compile_seconds=0,elimination_seconds=0,reconstruction_seconds=0;
    std::string status="not_started";
};
struct BoundaryComponent {std::vector<int> vertices, scope;Weight base=0;};
struct CouplingStructure {
    std::vector<BoundaryComponent> parts;
    std::vector<int> outsider_index;
    std::vector<std::pair<int,int>> conflicts;
    std::vector<std::set<int>> interaction;
};
struct EliminationPlan { std::vector<int> order;int width=0;std::uint64_t entries=0; };
struct Factor {std::vector<int> scope;std::vector<Weight> value;};
struct CompiledResponse {CouplingStructure structure;EliminationPlan plan;std::vector<Factor> factors;};
inline std::uint64_t table_size(int n) {
    if(n<0 || n>24)throw FactorLimit("table_arity_limit");
    return std::uint64_t(1)<<n;
}
inline void add_entries(std::uint64_t &used,std::uint64_t x,std::uint64_t limit) {
    if(used>limit || x>limit-used)throw FactorLimit("table_entry_limit");
    used+=x;
}
inline void validate_factor_options(const FactorOptions&o){
    if(o.max_width<0||o.max_width>20||o.max_boundary<0||o.max_boundary>20||o.max_variables<1||o.max_variables>256||o.max_entries>10000000)
        throw std::invalid_argument("invalid factor options");
}
inline CouplingStructure coupling_structure(const Kernel&k,const Deadline&d) {
    if(!k.base_backbone_exact || !k.graph.feasible(k.base))throw std::invalid_argument("exact feasible backbone required");
    const Graph&g=k.graph;const int count=int(k.outsiders.size());
    if(k.color.size()!=g.w.size())throw std::invalid_argument("kernel color length");
    CouplingStructure s;s.outsider_index.assign(g.n,-1);s.interaction.resize(count);
    for(int i=0;i<count;++i){int v=k.outsiders[i];
        if(v<0||v>=g.n||s.outsider_index[v]>=0||k.color[v]>=0||k.base[v])throw std::invalid_argument("invalid outsider map");
        s.outsider_index[v]=i;}
    for(int v=0;v<g.n;++v)if((k.color[v]<0)!=(s.outsider_index[v]>=0)||k.color[v]>1)throw std::invalid_argument("inconsistent kernel colors");
    Mask seen(g.n,0);
    for(int start=0;start<g.n;++start)if(k.color[start]>=0 && !seen[start]){
        d.check();BoundaryComponent c;c.vertices.push_back(start);seen[start]=1;std::set<int>scope;
        for(std::size_t i=0;i<c.vertices.size();++i){int u=c.vertices[i];if(k.base[u])c.base+=g.w[u];
            for(int v:g.adj[u]){
                if(s.outsider_index[v]>=0)scope.insert(s.outsider_index[v]);
                else {if(k.color[u]==k.color[v])throw std::invalid_argument("backbone not bipartite");
                    if(!seen[v]){seen[v]=1;c.vertices.push_back(v);}}
            }
            if((i&255)==0)d.check();
        }
        c.scope.assign(scope.begin(),scope.end());s.parts.push_back(std::move(c));
    }
    auto connect=[&](int u,int v){s.interaction[u].insert(v);s.interaction[v].insert(u);};
    for(const auto&c:s.parts){d.check();for(int u:c.scope)for(int v:c.scope)if(u<v)connect(u,v);}
    for(auto[u,v]:g.edges)if(s.outsider_index[u]>=0 && s.outsider_index[v]>=0){
        int a=s.outsider_index[u],b=s.outsider_index[v];if(a>b)std::swap(a,b);s.conflicts.emplace_back(a,b);connect(a,b);
    }
    return s;
}
inline EliminationPlan minfill_plan(const CouplingStructure&s,const FactorOptions&o,const Deadline&d){
    const int n=int(s.interaction.size());if(n>o.max_variables)throw FactorLimit("variable_count_limit");
    EliminationPlan plan;auto adj=s.interaction;Mask alive(n,1);
    for(int step=0;step<n;++step){d.check();int best=-1;std::tuple<std::uint64_t,int,int>best_key{UINT64_MAX,0,0};
        for(int v=0;v<n;++v)if(alive[v]){
            std::uint64_t fill=0;const auto&ns=adj[v];
            for(auto a=ns.begin();a!=ns.end();++a)for(auto b=std::next(a);b!=ns.end();++b)if(!adj[*a].count(*b))++fill;
            auto key=std::make_tuple(fill,int(ns.size()),v);if(key<best_key){best_key=key;best=v;}
            if((v&15)==0)d.check();
        }
        int width=int(adj[best].size());plan.width=std::max(plan.width,width);
        if(plan.width>o.max_width)throw FactorLimit("induced_width_limit");
        add_entries(plan.entries,table_size(width),o.max_entries);
        auto ns=adj[best];for(int u:ns)for(int v:ns)if(u<v){adj[u].insert(v);adj[v].insert(u);}
        for(int v:ns)adj[v].erase(best);
        adj[best].clear();alive[best]=0;plan.order.push_back(best);
    }
    return plan;
}
inline std::vector<int> active_backbone(const Kernel&k,const BoundaryComponent&c,const std::vector<std::uint8_t>&choice){
    Mask blocked(k.graph.n,0);
    for(int i:c.scope)if(choice[i])for(int v:k.graph.adj[k.outsiders[i]])blocked[v]=1;
    std::vector<int>vs;for(int v:c.vertices)if(!blocked[v])vs.push_back(v);return vs;
}
inline CompiledResponse compile_response(const Kernel&k,const FactorOptions&o,const Deadline&d,FactorStats&st){
    validate_factor_options(o);st.variables=int(k.outsiders.size());
    if(st.variables>o.max_variables)throw FactorLimit("variable_count_limit");
    auto begin=Clock::now();CompiledResponse c;c.structure=coupling_structure(k,d);st.components=int(c.structure.parts.size());
    Mask seen(st.variables,0);
    for(int i=0;i<st.variables;++i)if(!seen[i]){std::vector<int>vs{i};seen[i]=1;
        for(std::size_t j=0;j<vs.size();++j)for(int v:c.structure.interaction[vs[j]])if(!seen[v]){seen[v]=1;vs.push_back(v);}
        ++st.interaction_components;st.max_component_variables=std::max(st.max_component_variables,int(vs.size()));}
    for(const auto&p:c.structure.parts)st.max_boundary=std::max(st.max_boundary,int(p.scope.size()));
    if(st.max_boundary>o.max_boundary)throw FactorLimit("boundary_arity_limit");
    std::uint64_t entries=2*std::uint64_t(st.variables)+4*std::uint64_t(c.structure.conflicts.size());
    if(entries>o.max_entries)throw FactorLimit("table_entry_limit");
    for(const auto&p:c.structure.parts){st.max_boundary=std::max(st.max_boundary,int(p.scope.size()));
        if(int(p.scope.size())>o.max_boundary)throw FactorLimit("boundary_arity_limit");
        if(!p.scope.empty())add_entries(entries,table_size(int(p.scope.size())),o.max_entries);
    }
    c.plan=minfill_plan(c.structure,o,d);st.induced_width=c.plan.width;
    add_entries(entries,c.plan.entries,o.max_entries);st.planned_entries=entries;st.structure_seconds+=elapsed(begin);
    const auto&g=k.graph;begin=Clock::now();
    for(int i=0;i<st.variables;++i){c.factors.push_back({{i},{0,g.w[k.outsiders[i]]}});st.compiled_entries+=2;}
    for(auto[u,v]:c.structure.conflicts){c.factors.push_back({{u,v},{0,0,0,FACTOR_NEG}});st.compiled_entries+=4;}
    std::vector<std::uint8_t>choice(st.variables,0);
    for(const auto&p:c.structure.parts){d.check();if(p.scope.empty())continue;
        Factor f;f.scope=p.scope;f.value.resize(table_size(int(f.scope.size())),0);
        // Empty boundary assignment has zero loss: the input base is exact.
        for(std::uint64_t bits=1;bits<f.value.size();++bits){d.check();
            for(int j=0;j<int(p.scope.size());++j)choice[p.scope[j]]=std::uint8_t((bits>>j)&1);
            auto active=active_backbone(k,p,choice);auto cut=bipartite_mwis(g,active,k.color,d);++st.cuts;
            if(cut.value>p.base)throw std::logic_error("false exact-backbone claim");
            f.value[bits]=cut.value-p.base;
        }
        for(int i:p.scope)choice[i]=0;
        st.compiled_entries+=f.value.size();c.factors.push_back(std::move(f));
    }
    st.compile_seconds+=elapsed(begin);st.status="compiled";return c;
}
inline Weight evaluate_response(const CompiledResponse&c,const std::vector<std::uint8_t>&x){
    if(x.size()!=c.structure.interaction.size())throw std::invalid_argument("assignment length");
    Weight v=0;for(const auto&f:c.factors){std::size_t bits=0;for(int j=0;j<int(f.scope.size());++j){
        if(x[f.scope[j]]>1)throw std::invalid_argument("nonbinary assignment");
        bits|=std::size_t(x[f.scope[j]])<<j;}
        Weight a=f.value[bits];if(a==FACTOR_NEG)return FACTOR_NEG;v+=a;}return v;
}
struct EliminationSolution {Weight gain=0;std::vector<std::uint8_t>chosen;};
inline EliminationSolution eliminate_response(const CompiledResponse&c,const Deadline&d,FactorStats&st){
    const int n=int(c.structure.interaction.size());auto begin=Clock::now();
    // Factors are moved between buckets. No exponential table is created until
    // its complete scope has passed the preflight width and entry checks.
    std::vector<std::unique_ptr<Factor>>fs;
    for(const auto&f:c.factors)fs.emplace_back(std::make_unique<Factor>(f));
    struct Backpointer{int variable;std::vector<int>scope;std::vector<std::uint8_t>choice;};
    std::vector<Backpointer>back;
    for(int v:c.plan.order){d.check();std::vector<std::size_t>bucket;std::set<int>scope_set;
        for(std::size_t i=0;i<fs.size();++i)if(fs[i] && std::binary_search(fs[i]->scope.begin(),fs[i]->scope.end(),v)){
            bucket.push_back(i);scope_set.insert(fs[i]->scope.begin(),fs[i]->scope.end());}
        scope_set.erase(v);Factor next;next.scope.assign(scope_set.begin(),scope_set.end());
        if(int(next.scope.size())>c.plan.width)throw std::logic_error("elimination scope differs from width plan");
        std::uint64_t cells=table_size(int(next.scope.size()));next.value.assign(cells,FACTOR_NEG);
        Backpointer bp{v,next.scope,std::vector<std::uint8_t>(cells,0)};
        std::vector<std::vector<int>>positions;
        for(auto i:bucket){positions.emplace_back();for(int u:fs[i]->scope){
            if(u==v)positions.back().push_back(-1);
            else positions.back().push_back(int(std::lower_bound(next.scope.begin(),next.scope.end(),u)-next.scope.begin()));}}
        for(std::uint64_t bits=0;bits<cells;++bits){if((bits&255)==0)d.check();
            for(int val=0;val<2;++val){Weight score=0;bool feasible=true;
                for(std::size_t j=0;j<bucket.size();++j){const auto&f=*fs[bucket[j]];std::size_t idx=0;
                    for(int t=0;t<int(f.scope.size());++t){int pos=positions[j][t];idx|=std::size_t(pos<0?val:int((bits>>pos)&1))<<t;}
                    if(f.value[idx]==FACTOR_NEG){feasible=false;break;}score+=f.value[idx];
                }
                if(feasible&&score>next.value[bits]){next.value[bits]=score;bp.choice[bits]=std::uint8_t(val);}
            }
        }
        for(auto i:bucket)fs[i].reset();
        fs.emplace_back(std::make_unique<Factor>(std::move(next)));back.push_back(std::move(bp));st.dp_entries+=cells;
    }
    EliminationSolution out;out.chosen.assign(n,0);
    for(const auto&f:fs)if(f){if(!f->scope.empty()||f->value.size()!=1||f->value[0]==FACTOR_NEG)throw std::logic_error("infeasible elimination root");out.gain+=f->value[0];}
    for(auto it=back.rbegin();it!=back.rend();++it){d.check();std::size_t bits=0;for(int j=0;j<int(it->scope.size());++j)bits|=std::size_t(out.chosen[it->scope[j]])<<j;
        out.chosen[it->variable]=it->choice[bits];}
    if(out.gain<0||evaluate_response(c,out.chosen)!=out.gain)throw std::logic_error("elimination traceback mismatch");
    st.elimination_seconds+=elapsed(begin);st.status="eliminated";return out;
}
inline Mask reconstruct_response(const Kernel&k,const CompiledResponse&c,const EliminationSolution&sol,const Deadline&d,FactorStats&st){
    auto begin=Clock::now();Mask x(k.graph.n,0);
    for(int i=0;i<int(k.outsiders.size());++i)if(sol.chosen[i])x[k.outsiders[i]]=1;
    for(const auto&p:c.structure.parts){d.check();bool touched=false;for(int i:p.scope)if(sol.chosen[i])touched=true;
        if(!touched){for(int v:p.vertices)x[v]=k.base[v];}
        else{auto cut=bipartite_mwis(k.graph,active_backbone(k,p,sol.chosen),k.color,d);++st.cuts;for(int v:cut.selected)x[v]=1;}}
    if(!k.graph.feasible(x)||k.graph.value(x)!=k.graph.value(k.base)+sol.gain)throw std::logic_error("factor lift value mismatch");
    st.reconstruction_seconds+=elapsed(begin);return x;
}
struct FactoredRecovery{Recovery result;FactorStats stats;};
inline FactoredRecovery recover_factored(const Kernel&k,const FactorOptions&o,const Deadline&d){
    FactoredRecovery out;auto&r=out.result;r.selected=k.base;r.lower=k.graph.value(k.base);r.upper=r.lower;
    for(int v:k.outsiders)r.upper+=k.graph.w[v];
    try{
        auto c=compile_response(k,o,d,out.stats);auto sol=eliminate_response(c,d,out.stats);
        r.upper=r.lower+sol.gain; // exact value known even if later traceback/cut reconstruction is interrupted
        auto x=reconstruct_response(k,c,sol,d,out.stats);r.selected=std::move(x);r.lower=r.upper;r.exact=true;out.stats.status="exact";
    }catch(const FactorLimit&e){out.stats.status=e.what();}
    catch(const Timeout&){out.stats.status="timeout";}
    r.cuts=out.stats.cuts;r.components=out.stats.interaction_components;r.max_component_k=out.stats.max_component_variables;
    r.exact=r.lower==r.upper;
    return out;
}
inline bool branch_supported(const Kernel&k,bool decompose){
    if(!decompose)return k.outsiders.size()<=24;
    Mask seen(k.graph.n,0);for(int start=0;start<k.graph.n;++start)if(!seen[start]){
        std::vector<int>vs{start};seen[start]=1;int count=0;
        for(std::size_t i=0;i<vs.size();++i){int u=vs[i];if(k.color[u]<0)++count;
            for(int v:k.graph.adj[u])if(!seen[v]){seen[v]=1;vs.push_back(v);}}
        if(count>24)return false;
    }return true;
}
struct DispatchRecovery{Recovery result;FactorStats factor;std::string backend;};
inline DispatchRecovery recover_dispatch(const Kernel&k,const std::string&backend,const FactorOptions&o,
                                          std::uint64_t nodes,const Deadline&d,bool decompose=true){
    DispatchRecovery out;out.backend=backend;
    if(backend=="branch"){
        if(branch_supported(k,decompose))out.result=recover(k,nodes,d,decompose);
        else {out.result.selected=k.base;out.result.lower=k.graph.value(k.base);out.result.upper=out.result.lower;
            for(int v:k.outsiders)out.result.upper+=k.graph.w[v];
            out.backend="branch_unsupported";}
        return out;
    }
    if(backend!="factor"&&backend!="hybrid")throw std::invalid_argument("unknown recovery backend");
    auto f=recover_factored(k,o,d);out.result=std::move(f.result);out.factor=std::move(f.stats);out.backend="factor";
    if(backend=="hybrid"&&!out.result.exact&&!d.expired()){
        Recovery fallback;
        if(branch_supported(k,decompose)){fallback=recover(k,nodes,d,decompose);out.backend="factor_then_branch";}
        else{fallback=greedy_recover(k,d);out.backend="factor_then_greedy";}
        Weight upper=std::min(out.result.upper,fallback.upper);auto cuts=out.result.cuts;
        if(fallback.lower>out.result.lower)out.result=std::move(fallback);
        else{out.result.nodes=fallback.nodes;out.result.cache_hits=fallback.cache_hits;}
        out.result.cuts=cuts+fallback.cuts;out.result.upper=upper;out.result.exact=upper==out.result.lower;
    }
    return out;
}
} // namespace barr
