#include "barr/solver.hpp"
#include <iostream>
#include <sstream>
using namespace barr;
static std::uint64_t checks=0;
void require(bool x,const char*m){++checks;if(!x)throw std::runtime_error(m);}
Weight brute(const Graph&g,const Mask*allowed=nullptr){
    if(g.n>22)throw std::invalid_argument("brute limit");
    Weight best=0;std::vector<std::uint32_t>adj(g.n,0);std::uint32_t allow=(1u<<g.n)-1;
    for(auto[u,v]:g.edges){adj[u]|=1u<<v;adj[v]|=1u<<u;}
    if(allowed){allow=0;for(int v=0;v<g.n;++v)if((*allowed)[v])allow|=1u<<v;}
    std::function<void(std::uint32_t,Weight)>dfs=[&](std::uint32_t rem,Weight w){
        if(rem==0){best=std::max(best,w);return;}int v=__builtin_ctz(rem);rem&=~(1u<<v);
        dfs(rem,w);dfs(rem&~adj[v],w+g.w[v]);};dfs(allow,0);return best;
}
Mask random_is(const Graph&g,std::mt19937_64&r){Mask s(g.n,0);std::vector<int>order(g.n);std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),r);
    for(int v:order)if(r()%4){bool ok=true;for(int u:g.adj[v])if(s[u])ok=false;if(ok)s[v]=1;}
    return s;}
Graph random_graph(int n,std::mt19937_64&r){std::vector<Weight>w(n);for(auto&x:w)x=Weight(r()%31);std::vector<std::pair<int,int>>es;int threshold=5+r()%65;
    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)if(int(r()%100)<threshold)es.emplace_back(i,j);
    return Graph(w,es);}
int main(){try{
    std::mt19937_64 rng(731);int cases=600,bounded=0;
    for(int it=0;it<cases;++it){Graph g=random_graph(1+rng()%17,rng);Mask p=random_is(g,rng),q=random_is(g,rng);
        auto b=fuse(g,p,q,Deadline::after(30));require(b.exact,"fusion exact");require(g.value(b.merged)==brute(g,&b.support),"fusion optimum");
        auto expired=fuse(g,p,q,Deadline::after(0));require(g.feasible(expired.merged),"fusion timeout feasibility");
        require(g.value(expired.merged)>=std::max(g.value(p),g.value(q)),"fusion timeout parent bound");
        std::vector<int>a;for(int v=0;v<g.n;++v)if(!b.support[v]&&a.size()<8)a.push_back(v);
        std::vector<double>freq(g.n);for(int v=0;v<g.n;++v)freq[v]=(p[v]+q[v])*.5;
        auto k=make_kernel(g,b,p,q,freq,a,100,Deadline::after(30));Weight opt=brute(k.graph);
        auto rec=recover(k,100000,Deadline::after(30));require(rec.exact&&rec.lower==opt&&rec.upper==opt,"recovery optimum");
        require(g.feasible(lift(g,b,k,rec)),"full boundary lift");auto no_decomp=recover(k,100000,Deadline::after(30),false);
        require(no_decomp.exact&&no_decomp.lower==opt,"decomposition invariant");
        for(int cap:{0,1,2,8,32}){auto r=recover(k,cap,Deadline::after(30));require(r.lower<=opt&&r.upper>=opt,"bounded frontier certificate");
            require(k.graph.feasible(r.selected)&&r.lower==k.graph.value(r.selected),"bounded feasible lower");
            require(r.nodes<=std::uint64_t(cap),"shared node budget");require(!r.exact||r.lower==opt,"exact flag");++bounded;}
        auto rt=recover(k,100000,Deadline::after(0));require(rt.lower<=opt&&rt.upper>=opt,"timeout valid bounds");
        auto gr=greedy_recover(k,Deadline::after(30));require(gr.lower<=opt&&gr.upper>=opt&&k.graph.feasible(gr.selected),"greedy bounds");
        LocalSearch ls(g,p,rng());Weight before=ls.cost;ls.run(80,Deadline::after(30));require(ls.validate_state(),"local state");require(ls.cost>=before,"local monotonic ticks");
        ls.run(100000,Deadline::after(.000001));require(ls.validate_state(),"interrupted state rollback");
    }
    {Graph g({10,4,4,4},{{0,1},{0,2},{0,3}});Mask p{1,0,0,0};auto b=fuse(g,p,p,Deadline::after(3));
        auto k=make_kernel(g,b,p,p,{1,0,0,0},{1,2,3},4,Deadline::after(3));auto r=recover(k,1000,Deadline::after(3));
        require(g.value(b.merged)==10&&r.exact&&r.lower==12,"consensus expansion motif");
        bool rejected=false;try{make_kernel(g,b,p,p,{1,0,0,0},{1,2,3},3,Deadline::after(3));}catch(const std::length_error&){rejected=true;}
        require(rejected,"never truncate unsafe boundary");}
    {Graph g({10,9,9},{{0,1},{0,2},{1,2}});Mask p{1,0,0};auto b=fuse(g,p,p,Deadline::after(3));auto k=make_kernel(g,b,p,p,{1,0,0},{1,2},3,Deadline::after(3));
        auto r=recover(k,1000,Deadline::after(3));require(r.exact&&r.lower==10,"candidate conflicts enforced");}
    {Weight w=Weight(1)<<45;Graph g({w,w+1,w+2},{{0,1},{1,2}});Mask p{1,0,1},q{0,1,0};auto b=fuse(g,p,q,Deadline::after(3));
        require(g.value(b.merged)==2*w+2,"64 bit mincut");bool rejected=false;try{Graph bad({MAX_TOTAL,1},{});}catch(const std::invalid_argument&){rejected=true;}require(rejected,"weight overflow rejected");}
    {Graph g({},{});Config c;c.mode="barr";c.seconds=0;auto r=solve(g,c);require(r.selected.empty(),"empty graph");}
    {Graph g=random_graph(30,rng);g.initial=random_is(g,rng);Config c;c.mode="barr";c.seconds=5;c.max_rounds=4;c.kernel_seconds=.1;c.local_seconds=.2;
        auto s=solve(g,c);require(g.feasible(s.selected)&&g.raw_value(s.selected)>=g.raw_value(g.initial),"standalone complete");
        require(s.stats.recoveries<=s.stats.rounds*std::uint64_t(c.execute_top),"collection disabled by default");
        c.threads=2;auto t=solve(g,c);require(g.feasible(t.selected),"multithread standalone");}
    // Integration checks exercise the complete selective loop, not only the
    // sketch bound helper. These short budgets verify correctness and accounting;
    // they are not benchmark performance observations.
    for(const char*gate:{"value","random","witness"}){
        Graph g=random_graph(12,rng);g.initial=random_is(g,rng);Weight optimum=brute(g);
        Config c;c.mode="selective";c.gate=gate;c.population=2;c.local_iterations=1;
        c.seconds=.06;c.max_rounds=100000;c.local_seconds=.002;
        c.gate_warmup=0.;c.event_stale=1;c.gate_cooldown=.001;c.gate_fraction=.5;
        c.explore_every=1;c.gate_trials=3;c.max_kernel_vertices=1;
        auto s=solve(g,c);
        require(g.feasible(s.selected),"selective loop returns a feasible complete-graph schedule");
        require(g.raw_value(s.selected)>=g.raw_value(g.initial),"selective archive preserves original objective");
        require(g.value(s.selected)<=optimum,"selective result respects independently enumerated global optimum");
        require(s.stats.rounds<=std::uint64_t(c.max_rounds),"selective loop respects its round limit");
        require(s.stats.gate_epochs>0,"selective integration exercises event gating");
        require(s.stats.safe_rejects<=s.stats.sketches,"only completed sketches count as safe certificates");
        require(s.stats.recoveries<=s.stats.kernels_materialized,"structured recovery requires a materialized kernel");
        require(s.stats.accepted_recoveries<=s.stats.gate_epochs,"one selected move per selective event");
        require(s.stats.refinement_extra_ticks<=s.stats.positive_gain_ticks,"refinement gain is not larger than committed total gain");
        require(std::isfinite(s.seconds)&&std::isfinite(s.cpu_seconds),"selective completion reports finite measured times");
    }
    {Graph g({1,2},{{0,1}},{100.,1.});g.initial={1,0};
        Config c;c.mode="selective";c.population=1;c.local_iterations=1;
        c.seconds=.05;c.max_rounds=2;c.gate_warmup=0.;c.event_stale=1;
        c.gate_cooldown=.001;c.gate_fraction=.5;
        auto s=solve(g,c);
        // Integer descent prefers vertex 1, but its original objective is lower.
        // The raw-objective archive must retain the initial feasible schedule.
        require(s.selected==g.initial,"selective quantized search cannot degrade original-weight archive");
        require(g.raw_value(s.selected)==100.,"selective raw and integer objectives remain distinct");}
    std::cout<<"{\"status\":\"PASS\",\"random_graphs\":"<<cases<<",\"bounded_searches\":"<<bounded<<",\"assertions\":"<<checks<<"}\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
