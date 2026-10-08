#include "barr/pair_scout.hpp"
#include <filesystem>
#include <iostream>
#include <ctime>
using namespace barr;

// Read-only final-mask mechanism audit, separate from every performance search.
// A full result covers insertion of <=2 originally unselected vertices, deleting
// their selected blocker union and keeping all other incumbent vertices fixed.
struct PairAudit {
    PairOpportunity shared;
    std::vector<int> witness;
    Weight lower=0,upper=0;
    std::uint64_t extra_candidates=0,extra_shared_rejects=0,extra_conflict_rejects=0,extra_positive=0;
    bool complete=false,extra_partial=false,brute_checked=false;
    double seconds=0,cpu_seconds=0;
};

static Mask apply(const Graph&g,const Mask&s,const std::vector<int>&outside){
    Mask t=s;for(int v:outside)for(int u:g.adj[v])if(s[u])t[u]=0;
    for(int v:outside)t[v]=1;return t;
}

// Independent direct-mask oracle: no deficits or shared-blocker formula.
static Weight brute(const Graph&g,const Mask&s){
    Weight base=g.value(s),best=0;
    for(int a=0;a<g.n;++a)if(!s[a]){
        auto one=apply(g,s,{a});if(!g.feasible(one))throw std::logic_error("infeasible singleton witness");
        best=std::max(best,g.value(one)-base);
        for(int b=a+1;b<g.n;++b)if(!s[b]&&!g.adjacent(a,b)){
            auto two=apply(g,s,{a,b});if(!g.feasible(two))throw std::logic_error("infeasible pair witness");
            best=std::max(best,g.value(two)-base);
        }
    }
    return best;
}

static PairAudit audit(const Graph&g,const Mask&s,double budget,bool check_small=true){
    if(!g.feasible(s))throw std::invalid_argument("infeasible frozen incumbent");
    auto start=Clock::now();auto cpu_start=std::clock();auto deadline=Deadline::after(budget);
    LocalSearch local(g,s,404071); // Only rebuilds incremental arrays; no search step.
    PairAudit r;r.shared=pair_scout(g,local,deadline);r.witness=r.shared.outside;r.lower=r.shared.witness_gain;
    auto remember=[&](int a,int b,Weight gain){
        std::vector<int>vertices{std::min(a,b),std::max(a,b)};
        if(gain>r.lower||(gain==r.lower&&gain>0&&r.witness.size()==2&&vertices<r.witness)){
            r.lower=gain;r.witness=std::move(vertices);
        }
    };
    // Full shared scan is complete for positives at a one-insertion optimum.
    if(r.shared.stats.complete&&r.shared.stats.one_optimal)r.complete=true;
    else if(r.shared.stats.complete){
        // Without one-optimality, a positive pair with no shared blocker must
        // include a negative-deficit vertex and have d_a+d_b<0. Enumerate every
        // such pair, deduplicate two negative seeds, and exclude shared pairs
        // already counted by the completed production scout (even weight zero).
        std::uint64_t work=0;auto check=[&](){if((work++&1023)==0)deadline.check();};
        try{
            deadline.check();std::vector<int>outside,negative;
            std::vector<Weight>deficit(g.n,0);std::vector<std::vector<int>>blockers(g.n);
            for(int v=0;v<g.n;++v){check();if(!s[v]){
                outside.push_back(v);deficit[v]=local.blocked_weight[v]-g.w[v];
                if(deficit[v]<0)negative.push_back(v);
            }}
            for(int u=0;u<g.n;++u){check();if(s[u])for(int v:g.adj[u]){check();blockers[v].push_back(u);}}
            for(int a:negative)for(int b:outside){
                check();if(a==b||(deficit[b]<0&&b<a)||deficit[a]+deficit[b]>=0)continue;
                ++r.extra_candidates;
                if(g.adjacent(a,b)){++r.extra_conflict_rejects;continue;}
                bool shared=false;std::size_t i=0,j=0;
                while(i<blockers[a].size()&&j<blockers[b].size()){
                    check();int x=blockers[a][i],y=blockers[b][j];
                    if(x==y){shared=true;break;}else if(x<y)++i;else ++j;
                }
                if(shared){++r.extra_shared_rejects;continue;}
                ++r.extra_positive;remember(a,b,-deficit[a]-deficit[b]);
            }
            r.complete=true;
        }catch(const Timeout&){r.extra_partial=true;r.complete=false;}
    }
    // Global safe bound when the audit is partial: blocker cost is nonnegative.
    Weight first=0,second=0;for(int v=0;v<g.n;++v)if(!s[v]){
        if(g.w[v]>first){second=first;first=g.w[v];}else if(g.w[v]>second)second=g.w[v];
    }
    r.upper=r.complete?r.lower:first+second;
    auto witness=apply(g,s,r.witness);
    if(!g.feasible(witness)||g.value(witness)-g.value(s)!=r.lower||r.upper<r.lower)
        throw std::logic_error("audit witness or upper bound invalid");
    if(check_small&&g.n<=64&&r.complete){
        if(brute(g,s)!=r.lower)throw std::logic_error("independent all-pair brute disagrees with audit");
        r.brute_checked=true;
    }
    r.seconds=elapsed(start);r.cpu_seconds=double(std::clock()-cpu_start)/CLOCKS_PER_SEC;return r;
}

static void self_test(){
    std::uint64_t cases=0;for(unsigned edges=0;edges<64;++edges){
        std::vector<std::pair<int,int>>es;int bit=0;
        for(int a=0;a<4;++a)for(int b=a+1;b<4;++b,++bit)if(edges&(1u<<bit))es.emplace_back(a,b);
        Graph g({0,3,5,7},es);
        for(unsigned mask=0;mask<16;++mask){Mask s(4);for(int v=0;v<4;++v)s[v]=(mask>>v)&1;
            if(g.feasible(s)){auto r=audit(g,s,10);if(!r.complete||!r.brute_checked)throw std::logic_error("small audit incomplete");++cases;}}
    }
    std::mt19937_64 rng(7130407);
    for(int iteration=0;iteration<250;++iteration){
        int n=1+rng()%12;std::vector<Weight>w(n);for(auto&x:w)x=rng()%101;
        std::vector<std::pair<int,int>>es;for(int a=0;a<n;++a)for(int b=a+1;b<n;++b)if(rng()%100<35)es.emplace_back(a,b);
        Graph g(w,es);Mask s(n,0);for(int v=0;v<n;++v){bool legal=true;for(int u:g.adj[v])if(s[u])legal=false;if(legal&&rng()%2)s[v]=1;}
        auto r=audit(g,s,10);if(!r.complete||!r.brute_checked)throw std::logic_error("random audit incomplete");++cases;
    }
    // Zero-weight shared blockers must not be double-counted in extra scan.
    {Graph g({0,3,4},{{0,1},{0,2}});auto r=audit(g,Mask{1,0,0},10);
        if(r.lower!=7||r.shared.stats.positive_pairs!=1||r.extra_positive!=0)throw std::logic_error("zero-weight shared pair duplicated");++cases;}
    {Graph g({3,4},{});auto r=audit(g,Mask{0,0},10);
        if(r.lower!=7||r.extra_positive!=1||!r.complete)throw std::logic_error("disjoint positive singleton pair missing");++cases;}
    {Graph g({10,9,2},{{0,1},{0,2}});auto r=audit(g,Mask{1,0,0},0);
        if(r.complete||r.upper<r.lower)throw std::logic_error("timeout yielded false complete certificate");++cases;}
    std::cout<<"{\"status\":\"PASS\",\"cases\":"<<cases<<",\"independent_direct_mask_brute\":true}\n";
}

static void write_json(std::ostream&o,const Graph&g,const Mask&s,const PairAudit&r,double budget,double load_seconds){
    const auto&t=r.shared.stats;auto moved=apply(g,s,r.witness);
    o<<std::setprecision(17)<<"{\"schema\":\"barr_final_pair_audit_v1\",\"scope\":\"Insert at most two originally unselected vertices, delete their selected blocker union, keep all other incumbent vertices fixed\""
     <<",\"graph_n\":"<<g.n<<",\"graph_m\":"<<g.edges.size()<<",\"incumbent_signature\":"<<signature(s)<<",\"incumbent_ticks\":"<<g.value(s)
     <<",\"incumbent_selected_count\":"<<members(s).size()<<",\"budget_seconds\":"<<budget<<",\"load_seconds\":"<<load_seconds
     <<",\"audit_seconds\":"<<r.seconds<<",\"audit_cpu_seconds\":"<<r.cpu_seconds<<",\"complete\":"<<(r.complete?"true":"false")
     <<",\"one_insertion_optimal\":"<<(t.inspected_vertices==std::uint64_t(g.n)?(t.one_optimal?"true":"false"):"null")
     <<",\"lower_gain_ticks\":"<<r.lower<<",\"upper_gain_ticks\":"<<r.upper<<",\"exact\":"<<(r.complete?"true":"false")
     <<",\"positive_opportunity_exists\":"<<(r.lower>0?"true":(r.complete?"false":"null"))
     <<",\"minimum_positive_coalition_size\":"<<(t.positive_singletons?"1":(r.lower>0?"2":"null"))
     <<",\"positive_singletons_lower_count\":"<<t.positive_singletons<<",\"positive_singletons_count_exact\":"<<(t.inspected_vertices==std::uint64_t(g.n)?"true":"false")
     <<",\"positive_pairs_lower_count\":"<<(t.positive_pairs+r.extra_positive)<<",\"positive_pairs_count_exact\":"<<(r.complete?"true":"false")
     <<",\"witness_outside\":[";
    for(std::size_t j=0;j<r.witness.size();++j){if(j)o<<',';o<<r.witness[j];}
    o<<"],\"witness_removed_selected\":[";bool comma=false;
    for(int v=0;v<g.n;++v)if(s[v]&&!moved[v]){if(comma)o<<',';comma=true;o<<v;}
    o<<"],\"witness_singleton_gain_ticks\":[";
    for(std::size_t j=0;j<r.witness.size();++j){if(j)o<<',';int v=r.witness[j];Weight blockers=0;for(int u:g.adj[v])if(s[u])blockers+=g.w[u];o<<g.w[v]-blockers;}
    o<<"],\"independent_brute_checked\":"<<(r.brute_checked?"true":"false")
     <<",\"shared_scan\":{\"complete\":"<<(t.complete?"true":"false")<<",\"partial\":"<<(t.partial?"true":"false")
     <<",\"inspected_vertices\":"<<t.inspected_vertices<<",\"outside_vertices\":"<<t.outside_vertices<<",\"seeds\":"<<t.seeds
     <<",\"pair_visits\":"<<t.pair_visits<<",\"unique_pairs\":"<<t.unique_pairs<<",\"upper_rejects\":"<<t.upper_rejects
     <<",\"conflict_rejects\":"<<t.conflict_rejects<<",\"exact_pairs\":"<<t.exact_pairs<<",\"positive_pairs\":"<<t.positive_pairs
     <<",\"intersection_visits\":"<<t.intersection_visits<<",\"seconds\":"<<r.shared.seconds<<"}"
     <<",\"nonshared_completion\":{\"partial\":"<<(r.extra_partial?"true":"false")<<",\"candidates\":"<<r.extra_candidates
     <<",\"conflict_rejects\":"<<r.extra_conflict_rejects<<",\"shared_rejects\":"<<r.extra_shared_rejects<<",\"positive_pairs\":"<<r.extra_positive<<"}}\n";
}

int main(int argc,char**argv){try{
    if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
    if(argc!=5)throw std::invalid_argument("usage: barr_pair_oracle graph.barr incumbent.txt output.json seconds; or --self-test");
    std::size_t parsed=0;double seconds=std::stod(argv[4],&parsed);
    if(parsed!=std::string(argv[4]).size()||!std::isfinite(seconds)||seconds<0||seconds>31536000.)throw std::invalid_argument("invalid audit budget");
    std::filesystem::path input=std::filesystem::weakly_canonical(argv[1]),state=std::filesystem::weakly_canonical(argv[2]),out=std::filesystem::weakly_canonical(argv[3]);
    if(input==state||input==out||state==out||std::filesystem::exists(out))throw std::invalid_argument("audit inputs/output must be distinct, output must not exist");
    auto loaded=Clock::now();Graph g=read_graph(argv[1]);std::ifstream f(argv[2]);std::string magic;int count;
    if(!(f>>magic>>count)||magic!="BARRPAIR1"||count<0||count>g.n)throw std::invalid_argument("invalid BARRPAIR1 incumbent");
    Mask s(g.n,0);for(int j=0;j<count;++j){int v;if(!(f>>v)||v<0||v>=g.n||s[v])throw std::invalid_argument("invalid/duplicate frozen selected vertex");s[v]=1;}
    if(f>>magic)throw std::invalid_argument("unexpected incumbent suffix");double load_seconds=elapsed(loaded);
    auto r=audit(g,s,seconds);std::ofstream o(argv[3]);if(!o)throw std::runtime_error("cannot open audit output");
    write_json(o,g,s,r,seconds,load_seconds);o.close();if(!o)throw std::runtime_error("audit output write failed");return 0;
}catch(const std::exception&e){std::cerr<<"pair oracle error: "<<e.what()<<'\n';return 2;}}
