#include "barr/solver.hpp"
#include <iostream>
using namespace barr;
static std::uint64_t checks=0, exhaustive_states=0, random_states=0;
static void require(bool yes,const char*message){++checks;if(!yes)throw std::runtime_error(message);}
static Mask move(const Graph&g,const Mask&incumbent,const std::vector<int>&outside){
    Mask s=incumbent;for(int v:outside)for(int u:g.adj[v])if(incumbent[u])s[u]=0;
    for(int v:outside)s[v]=1;return s;
}
static Weight brute_pair(const Graph&g,const Mask&incumbent){
    Weight best=0,base=g.value(incumbent);
    for(int a=0;a<g.n;++a)if(!incumbent[a]){
        auto one=move(g,incumbent,{a});require(g.feasible(one),"every singleton removal/insertion is feasible");
        best=std::max(best,g.value(one)-base);
        for(int b=a+1;b<g.n;++b)if(!incumbent[b]&&!g.adjacent(a,b)){
            auto two=move(g,incumbent,{a,b});require(g.feasible(two),"compatible pair removal/insertion is feasible");
            best=std::max(best,g.value(two)-base);
        }
    }
    return best;
}
static void check_state(const Graph&g,const Mask&incumbent){
    LocalSearch local(g,incumbent,71);auto opportunity=pair_scout(g,local,Deadline::after(30));
    Weight best=brute_pair(g,incumbent);auto s=move(g,incumbent,opportunity.outside);
    require(opportunity.outside.size()<=2,"pair scout move uses at most two outsiders");
    require(g.feasible(s),"pair scout witness is globally feasible");
    require(g.value(s)-g.value(incumbent)==opportunity.witness_gain,"pair scout witness has exact integer gain");
    require(opportunity.witness_gain<=best,"pair scout witness is a valid pair lower bound");
    require(opportunity.stats.complete&&!opportunity.stats.partial,"full small-graph scout completes");
    require(opportunity.stats.inspected_vertices==std::uint64_t(g.n),"full scalar scan counts every vertex");
    if(opportunity.stats.one_optimal){
        require(opportunity.complete_two_exchange,"one-optimal full scout covers every profitable pair");
        require(opportunity.witness_gain==best,"shared-blocker enumeration matches exhaustive pair optimum");
        require(opportunity.certificate_complete==(best==0),"absence certificate is equivalent to zero exhaustive pair gain");
    } else {
        require(!opportunity.complete_two_exchange&&!opportunity.certificate_complete,"positive singleton prevents a false complete-pair claim");
    }
    for(std::uint64_t epoch:{1ULL,7ULL,99917ULL}){
        auto rotated=pair_scout(g,local,Deadline::after(30),PairScoutConfig{},epoch);
        require(rotated.stats.complete&&!rotated.stats.partial,"rotated full scout still completes every seed");
        require(rotated.witness_gain==opportunity.witness_gain&&rotated.outside==opportunity.outside,"full scout rotation preserves best canonical witness");
        require(rotated.complete_two_exchange==opportunity.complete_two_exchange&&rotated.certificate_complete==opportunity.certificate_complete,"full scout rotation preserves completeness certificates");
        if(rotated.complete_two_exchange)require(rotated.witness_gain==best,"rotated complete two-exchange scout matches exhaustive optimum");
    }
}
int main(){try{
    for(std::uint32_t bits=0;bits<64;++bits){
        std::vector<std::pair<int,int>>edges;int edge=0;
        for(int a=0;a<4;++a)for(int b=a+1;b<4;++b,++edge)if(bits&(1u<<edge))edges.emplace_back(a,b);
        Graph g({0,3,5,7},edges);
        for(std::uint32_t mask=0;mask<16;++mask){Mask s(4,0);for(int v=0;v<4;++v)s[v]=(mask>>v)&1;
            if(g.feasible(s)){check_state(g,s);++exhaustive_states;}}
    }
    std::mt19937_64 rng(404071);
    for(int iteration=0;iteration<500;++iteration){
        int n=1+rng()%14;std::vector<Weight>weights(n);for(auto&x:weights)x=rng()%103;
        std::vector<std::pair<int,int>>edges;int threshold=5+rng()%70;
        for(int a=0;a<n;++a)for(int b=a+1;b<n;++b)if(int(rng()%100)<threshold)edges.emplace_back(a,b);
        Graph g(weights,edges);Mask s(n,0);std::vector<int>order(n);std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
        for(int v:order){bool legal=true;for(int u:g.adj[v])if(s[u])legal=false;if(legal)s[v]=1;}
        check_state(g,s);++random_states;
    }
    {Graph g({10,9,2},{{0,1},{0,2}});Mask s{1,0,0};LocalSearch local(g,s,17);
        auto r=pair_scout(g,local,Deadline::after(3));
        require(r.witness_gain==1&&r.outside==std::vector<int>({1,2}),"two negative singleton gains combine through shared blocker");
        require(r.complete_two_exchange&&!r.certificate_complete,"positive complete pair search is not an absence certificate");}
    {Graph g({5,7,9,5},{{0,2},{1,2},{1,3}});Mask s{1,1,0,0};LocalSearch local(g,s,17);
        auto r=pair_scout(g,local,Deadline::after(3));
        require(r.witness_gain==2&&r.outside==std::vector<int>({2,3}),"pair with distinct private and shared blockers uses exact intersection");check_state(g,s);}
    {Graph g({5,7,7,6},{{0,2},{0,3},{1,2},{1,3}});Mask s{1,1,0,0};LocalSearch local(g,s,17);
        auto r=pair_scout(g,local,Deadline::after(3));
        require(r.witness_gain==1&&r.stats.unique_pairs==1&&r.stats.exact_pairs==1,"timestamp marks deduplicate a pair sharing multiple incumbent blockers");check_state(g,s);}
    {Graph g({10,9,9},{{0,1},{0,2},{1,2}});Mask s{1,0,0};LocalSearch local(g,s,17);
        auto r=pair_scout(g,local,Deadline::after(3));
        require(r.witness_gain==0&&r.certificate_complete&&r.stats.conflict_rejects==1,"conflicting outsiders are rejected and absence certified");}
    {Graph g({10,4,4},{{0,1},{0,2}});Mask s{1,0,0};LocalSearch local(g,s,17);
        auto r=pair_scout(g,local,Deadline::after(3));
        require(r.witness_gain==0&&r.stats.upper_rejects==1&&r.stats.exact_pairs==0&&r.certificate_complete,"constant-time upper bound safely rejects worthless pair");}
    {Graph g({3,4},{});Mask s{0,0};LocalSearch local(g,s,17);
        auto r=pair_scout(g,local,Deadline::after(3));
        require(r.witness_gain==4&&r.stats.positive_singletons==2&&!r.complete_two_exchange&&!r.certificate_complete,"positive singles do not falsely certify omitted nonshared pairs");}
    {Weight large=Weight(1)<<58;Graph g({large,large-1,2},{{0,1},{0,2}});Mask s{1,0,0};LocalSearch local(g,s,17);
        auto r=pair_scout(g,local,Deadline::after(3));require(r.witness_gain==1,"64-bit pair arithmetic preserves a one-tick gain");check_state(g,s);}
    {Graph g({10,9,2},{{0,1},{0,2}});Mask s{1,0,0};LocalSearch local(g,s,17);
        auto r=pair_scout(g,local,Deadline::after(0));
        require(r.stats.partial&&!r.stats.complete&&!r.complete_two_exchange&&!r.certificate_complete,"expired scout never emits a completeness certificate");
        PairScoutConfig cfg;cfg.max_seeds=1;auto capped=pair_scout(g,local,Deadline::after(3),cfg,0);
        require(capped.witness_gain==1&&capped.stats.capped&&!capped.complete_two_exchange&&!capped.certificate_complete,"capped scout keeps a valid witness without a completeness claim");
        cfg.max_seeds=-1;bool rejected=false;try{pair_scout(g,local,Deadline::after(3),cfg);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"negative scout cap rejected");}
    {Graph g({},{});LocalSearch local(g,{},17);auto r=pair_scout(g,local,Deadline::after(3));
        require(r.certificate_complete&&r.complete_two_exchange&&r.stats.complete,"empty graph has complete zero pair certificate");}
    {std::vector<Weight>weights(4097,9);weights[0]=10;weights[1]=11;std::vector<std::pair<int,int>>edges;
        for(int v=1;v<4097;++v)edges.emplace_back(0,v);
        Graph g(weights,edges);Mask s(4097,0);s[0]=1;LocalSearch local(g,s,17);bool retained=false;
        for(double budget:{.001,.004,.016}){
            auto r=pair_scout(g,local,Deadline::after(budget));
            if(r.stats.partial&&r.witness_gain>0){
                retained=true;auto lifted=move(g,s,r.outside);
                require(g.feasible(lifted)&&g.value(lifted)-g.value(s)==r.witness_gain,"deadline partial scout preserves verified positive witness");
                require(r.witness_gain<=10&&!r.stats.complete&&!r.complete_two_exchange&&!r.certificate_complete,"partial positive scout makes no maximum-pair or absence claim");break;
            }
        }
        require(retained,"later pair-scan timeout retains a positive move found before interruption");}
    { // Two outsiders need to release two incumbent blockers jointly. A third
      // compatible outsider makes a positive-seed refinement additionally useful.
        Graph g({10,10,12,12,7},{{0,2},{1,2},{0,3},{1,3},{0,4}});g.initial={1,1,0,0,0};
        for(const std::string policy:{"direct","refine","fusion-refine"}){
            Config c;c.mode="pair";c.pair_policy=policy;c.population=1;c.seconds=.18;
            c.local_seconds=0;c.local_iterations=1;c.gate_warmup=0;c.gate_cooldown=.001;c.gate_fraction=.25;c.event_stale=1;c.pair_slice=.1;
            auto r=solve(g,c);
            require(g.feasible(r.selected)&&g.value(r.selected)==31,"native pair integration preserves feasible best full-graph result");
            require(g.raw_value(r.selected)>=g.raw_value(g.initial),"native pair integration preserves original objective archive");
            require(r.stats.rounds>=2&&r.stats.gate_epochs>0&&r.stats.accepted_recoveries>0,"native pair integration executes several local rounds and a verified positive gate");
            require(std::isfinite(r.seconds)&&r.seconds>=0&&std::isfinite(r.cpu_seconds)&&r.cpu_seconds>0,"native integration records valid wall and process CPU time");
            if(policy=="direct")require(r.stats.kernels_materialized==0&&r.stats.recoveries==0,"direct pair policy never constructs a Kernel");
            else require(r.stats.refinement_extra_ticks>=7&&r.stats.kernels_materialized>0,"positive pair seed enables extra structured recovery gain");
        }
    }
    { // Four incumbent vertices versus three compatible outsiders: every pair
      // is nonpositive, so zero gates keep both parents until a rare fusion pulse.
        std::vector<std::pair<int,int>>edges;for(int u=0;u<4;++u)for(int v=4;v<7;++v)edges.emplace_back(u,v);
        Graph g({3,3,3,3,5,5,5},edges);g.initial={1,1,1,1,0,0,0};
        Config c;c.mode="pair";c.pair_policy="fusion-refine";c.population=4;c.seconds=.35;c.local_seconds=0;c.local_iterations=1;
        c.gate_warmup=0;c.gate_cooldown=.01;c.gate_fraction=.25;c.event_stale=1;c.pair_slice=.1;c.pair_fusion_every=1;
        auto r=solve(g,c);
        require(g.feasible(r.selected)&&g.value(r.selected)==15,"rare fusion pulse maintains best parental result");
        require(r.stats.fusions>0&&r.stats.pair_certificates>0,"fusion-refine integration executes fusion and completed zero pair certificates");
        require(r.stats.kernels_materialized==0&&r.stats.recoveries==0,"zero pair gates never open a Kernel even in fusion-refine policy");
    }
    {Graph g({2,3},{{0,1}},{100.,1.});g.initial={1,0};Config c;c.mode="pair";c.pair_policy="direct";c.population=1;
        c.seconds=.08;c.local_seconds=0;c.local_iterations=1;c.gate_warmup=0;c.gate_cooldown=.001;c.gate_fraction=.25;c.event_stale=1;
        auto r=solve(g,c);
        require(r.selected==g.initial&&g.value(r.selected)==2&&g.raw_value(r.selected)==100.,"raw-versus-tick disagreement preserves the original-value archive");
    }
    {Graph g({10,9,2},{{0,1},{0,2}});g.initial={1,0,0};Config c;c.mode="pair";c.seconds=0;
        auto r=solve(g,c);require(r.selected==g.initial&&g.feasible(r.selected),"zero native search budget preserves feasible initial archive");
        c.gate_fraction=std::numeric_limits<double>::quiet_NaN();bool rejected=false;
        try{solve(g,c);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"nonfinite pair gate configuration rejected before native search");
    }
    std::cout<<"{\"status\":\"PASS\",\"exhaustive_states\":"<<exhaustive_states<<",\"random_states\":"<<random_states<<",\"assertions\":"<<checks<<"}\n";
    return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
