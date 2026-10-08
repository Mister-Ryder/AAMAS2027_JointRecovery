#include "barr/selective.hpp"
#include <iostream>
using namespace barr;
static std::uint64_t checks=0, exhaustive_domains=0, random_domains=0;
static void require(bool yes,const char*message){++checks;if(!yes)throw std::runtime_error(message);}

static Mask witness(const Graph&g,const Mask&incumbent,const std::vector<int>&outside){
    Mask s=incumbent;
    for(int v:outside)for(int u:g.adj[v])if(incumbent[u])s[u]=0;
    for(int v:outside)s[v]=1;
    return s;
}
static Weight exact_gain(const Graph&g,const Mask&incumbent,const std::vector<int>&outside){
    Weight best=0,base=g.value(incumbent);
    for(std::uint32_t bits=0;bits<(1u<<outside.size());++bits){
        std::vector<int>selected;
        for(std::size_t i=0;i<outside.size();++i)if(bits&(1u<<i))selected.push_back(outside[i]);
        Mask s=witness(g,incumbent,selected);
        if(g.feasible(s))best=std::max(best,g.value(s)-base);
    }
    return best;
}
static void check_domain(const Graph&g,const Mask&incumbent,const std::vector<int>&outside,bool check_lift=false){
    auto d=Deadline::after(30);auto sketch=selective_evaluate(g,incumbent,outside,d);
    Weight gain=exact_gain(g,incumbent,outside);
    require(sketch.upper_gain>=gain,"allocation-clique upper bound dominates every feasible coalition");
    require(sketch.witness_gain<=gain,"repair witness never exceeds exact domain gain");
    require(sketch.upper_gain!=0||gain==0,"certified zero never prunes a profitable domain");
    Mask s=witness(g,incumbent,sketch.witness_outside);
    require(g.feasible(s),"repair witness is a feasible full-graph move");
    require(g.value(s)-g.value(incumbent)==sketch.witness_gain,"repair witness gain is exact integer gain");
    if(check_lift){
        auto b=singleton_backbone(g,incumbent,d);std::vector<double>freq(g.n,0);
        for(int v=0;v<g.n;++v)freq[v]=incumbent[v];
        auto kernel=make_kernel(g,b,incumbent,incumbent,freq,outside,g.n,d);
        auto recovery=recover(kernel,100000,d);
        require(recovery.exact,"tiny singleton kernel completes exact recovery");
        require(recovery.lower-kernel.graph.value(kernel.base)==gain,"singleton kernel equals direct outsider-domain optimum");
        Mask lifted=lift(g,b,kernel,recovery);
        require(g.feasible(lifted),"selected singleton kernel preserves complete boundary");
        require(g.value(lifted)-g.value(incumbent)==gain,"lift preserves domain gain");
    }
}

int main(){try{
    // All 64 simple graphs on four vertices, all feasible incumbents and all
    // outsider subsets; each domain itself is solved by exhaustive subset search.
    for(std::uint32_t graph_bits=0;graph_bits<64;++graph_bits){
        std::vector<std::pair<int,int>>edges;int bit=0;
        for(int u=0;u<4;++u)for(int v=u+1;v<4;++v,++bit)if(graph_bits&(1u<<bit))edges.emplace_back(u,v);
        Graph g({0,3,5,7},edges);
        for(std::uint32_t incumbent_bits=0;incumbent_bits<16;++incumbent_bits){
            Mask incumbent(4,0);std::vector<int>available;
            for(int v=0;v<4;++v){incumbent[v]=(incumbent_bits>>v)&1;if(!incumbent[v])available.push_back(v);}
            if(!g.feasible(incumbent))continue;
            for(std::uint32_t domain_bits=0;domain_bits<(1u<<available.size());++domain_bits){
                std::vector<int>outside;
                for(std::size_t i=0;i<available.size();++i)if(domain_bits&(1u<<i))outside.push_back(available[i]);
                check_domain(g,incumbent,outside,(exhaustive_domains%97)==0);++exhaustive_domains;
            }
        }
    }
    std::mt19937_64 rng(40517);
    for(int iteration=0;iteration<500;++iteration){
        int n=2+rng()%11;std::vector<Weight>weights(n);for(auto&x:weights)x=rng()%101;
        std::vector<std::pair<int,int>>edges;int threshold=5+rng()%70;
        for(int u=0;u<n;++u)for(int v=u+1;v<n;++v)if(int(rng()%100)<threshold)edges.emplace_back(u,v);
        Graph g(weights,edges);Mask incumbent(n,0);std::vector<int>order(n);std::iota(order.begin(),order.end(),0);
        std::shuffle(order.begin(),order.end(),rng);
        for(int v:order)if(rng()%3){bool legal=true;for(int u:g.adj[v])if(incumbent[u])legal=false;if(legal)incumbent[v]=1;}
        std::vector<int>outside;for(int v:order)if(!incumbent[v]&&outside.size()<8)outside.push_back(v);
        check_domain(g,incumbent,outside,iteration%5==0);++random_domains;
        LocalSearch local(g,incumbent,iteration+17);SelectiveConfig cfg;cfg.pool_cap=8;cfg.random_fill=4;
        SelectiveStats a_stats,b_stats;std::mt19937_64 a_rng(731),b_rng(731);
        auto a=selective_sketches(g,local,cfg,a_rng,11,Deadline::after(30),&a_stats);
        auto b=selective_sketches(g,local,cfg,b_rng,11,Deadline::after(30),&b_stats);
        require(a.size()==b.size(),"sketch generation deterministic cardinality");
        require(a_stats.raw_candidates==g.n-members(local.s).size(),"scalar scan visits every non-incumbent candidate");
        require(a_stats.sketches==a.size(),"sketch generation counted independently of filtering");
        require(!a_stats.partial&&a_stats.timeout_trials==0,"completed sketch generation has no timeout flag");
        require(a_stats.pool_candidates<=std::uint64_t(cfg.pool_cap)*cfg.seed_trials,"random fill respects total pool cap");
        for(std::size_t i=0;i<a.size();++i){
            require(a[i].outside==b[i].outside&&a[i].upper_gain==b[i].upper_gain&&a[i].witness_gain==b[i].witness_gain,"sketch generation is seed-reproducible");
            require(a[i].outside.size()<=std::size_t(cfg.challengers),"sketch outsider cap respected");
            check_domain(g,incumbent,a[i].outside);
        }
    }
    {Graph g({10,9,2},{{0,1},{0,2}});Mask incumbent{1,0,0};
        auto s=selective_evaluate(g,incumbent,{1,2},Deadline::after(3));
        require(g.w[1]-g.w[0]<0&&g.w[2]-g.w[0]<0,"motif has negative singleton gains");
        require(s.upper_gain>=1&&s.witness_gain==1,"shared blocker retains negative-singleton positive coalition");
        check_domain(g,incumbent,{1,2},true);}
    {Graph g({10,6,6},{{0,1},{0,2},{1,2}});Mask incumbent{1,0,0};
        auto s=selective_evaluate(g,incumbent,{1,2},Deadline::after(3));
        require(s.upper_gain==0,"clique-aware blocker charge certifies conflicting six-plus-six domain zero");
        require(s.witness_gain==0,"conflicting outsiders cannot create a fake repair witness");
        check_domain(g,incumbent,{1,2},true);}
    {Graph g({10,6,6},{{0,1},{0,2}});Mask incumbent{1,0,0};
        auto s=selective_evaluate(g,incumbent,{1,2},Deadline::after(3));
        require(s.upper_gain==2&&s.witness_gain==2,"compatible six-plus-six domain remains profitable");
        check_domain(g,incumbent,{1,2},true);}
    {Graph g({5,3,3},{{0,1},{0,2}});Mask incumbent{1,0,0};
        auto s=selective_evaluate(g,incumbent,{1,2},Deadline::after(3));
        require(s.upper_gain==2&&s.witness_gain==1,"floor allocation preserves odd-weight profitable coalition");}
    {Graph g({10,4,4},{{0,1},{0,2}});Mask incumbent{1,0,0};SelectiveStats stats;
        auto s=selective_evaluate(g,incumbent,{1,2},Deadline::after(3),&stats);
        require(s.upper_gain==0&&stats.safe_zero==1,"zero allocation bound provides explicit certificate");}
    {Weight large=Weight(1)<<55;Graph g({large,large-1,2},{{0,1},{0,2}});Mask incumbent{1,0,0};
        auto s=selective_evaluate(g,incumbent,{1,2},Deadline::after(3));
        require(s.witness_gain==1&&s.upper_gain>=1,"64-bit exact reward and cost avoid overflow");check_domain(g,incumbent,{1,2},true);}
    {Graph g({},{});LocalSearch local(g,{},17);SelectiveConfig cfg;std::mt19937_64 r(17);
        require(selective_sketches(g,local,cfg,r,0,Deadline::after(3)).empty(),"empty graph yields no sketches");}
    {Graph g({3,2},{{0,1}});Mask incumbent{1,0};bool timed_out=false;
        try{selective_evaluate(g,incumbent,{1},Deadline::after(0));}catch(const Timeout&){timed_out=true;}
        require(timed_out,"sketch evaluation obeys expired deadline");
        LocalSearch local(g,incumbent,17);std::mt19937_64 r(17);SelectiveConfig cfg;cfg.pool_cap=1;cfg.random_fill=32;
        SelectiveStats stats;auto sketches=selective_sketches(g,local,cfg,r,0,Deadline::after(3),&stats);
        require(sketches.size()==1&&sketches[0].outside.size()==1,"pool cap one remains one with random fill enabled");
        require(stats.pool_candidates==std::uint64_t(cfg.seed_trials),"random fill never grows capped singleton pool");
        SelectiveStats expired_stats;
        auto expired=selective_sketches(g,local,cfg,r,0,Deadline::after(0),&expired_stats);
        require(expired.empty()&&expired_stats.partial&&expired_stats.timeout_trials==1,"interrupted scalar scan returns no fabricated sketches and marks partial");
        require(expired_stats.safe_zero==0&&expired_stats.sketches==0,"interrupted scalar scan is never counted as a zero certificate");
        cfg.challengers=25;bool rejected=false;try{selective_sketches(g,local,cfg,r,0,Deadline::after(3));}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"invalid sketch configuration rejected");}
    { // A deliberately long candidate sweep makes an interrupted later trial
      // observable. Several budgets avoid relying on one machine speed.
        const int n=768;std::vector<Weight>weights(n,10);std::vector<std::pair<int,int>>edges;
        for(int u=0;u<n;++u)for(int distance=1;distance<=10;++distance){int v=(u+distance)%n;edges.emplace_back(std::min(u,v),std::max(u,v));}
        Graph g(weights,edges);Mask incumbent(n,0);for(int v=0;v<n;v+=64)incumbent[v]=1;
        LocalSearch local(g,incumbent,17);SelectiveConfig cfg;cfg.seed_trials=256;cfg.pool_cap=256;cfg.random_fill=256;cfg.challengers=24;
        bool retained=false;
        for(double budget:{.002,.008,.032,.128}){
            SelectiveStats stats;std::mt19937_64 r(7417);
            auto sketches=selective_sketches(g,local,cfg,r,0,Deadline::after(budget),&stats);
            require(stats.sketches==sketches.size(),"interrupted generator returns exactly its completed sketches");
            if(stats.partial&&!sketches.empty()){
                retained=true;require(stats.timeout_trials==1,"later-trial timeout is counted once");
                std::uint64_t zero=0;
                for(const auto&s:sketches){
                    auto complete=selective_evaluate(g,incumbent,s.outside,Deadline::after(3));
                    require(s.upper_gain==complete.upper_gain&&s.witness_gain==complete.witness_gain,"retained prefix contains complete valid bounds and witnesses");
                    if(s.upper_gain==0)++zero;
                }
                require(zero==stats.safe_zero,"unfinished later sketch contributes no zero certificate");break;
            }
        }
        require(retained,"later-trial timeout preserves a nonempty completed sketch prefix");}
    std::cout<<"{\"status\":\"PASS\",\"exhaustive_domains\":"<<exhaustive_domains<<",\"random_domains\":"<<random_domains<<",\"assertions\":"<<checks<<"}\n";
    return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
