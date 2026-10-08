#pragma once
#include "policy.hpp"
#include "local_search.hpp"
#include <unordered_set>

namespace barr {

// These inexpensive sketches precede graph induction and structured recovery.
// Sampling, pool truncation and ranking are heuristic. Only upper_gain==0 is a
// certificate: no subset of THIS outsider sketch can improve its incumbent.
struct SelectiveConfig {
    int seed_trials=8;
    int pool_cap=256;
    int challengers=12;
    int random_fill=32;
};
struct SelectiveStats {
    std::uint64_t raw_candidates=0, sketches=0, safe_zero=0;
    std::uint64_t neighbor_visits=0, duplicates=0, pool_candidates=0;
    std::uint64_t timeout_trials=0;
    bool partial=false;
};
struct Sketch {
    std::vector<int> outside, blockers;
    std::vector<Weight> allocated_cost, reduced_reward;
    std::vector<int> witness_outside;
    Weight upper_gain=0, witness_gain=0, union_cost=0, rewards=0;
    double shared_blocker_credit=0, predicted_work=1, score=0;
};

inline Backbone singleton_backbone(const Graph&g,const Mask&incumbent,
                                  const Deadline&d=Deadline::after(31536000.)) {
    if(incumbent.size()!=g.w.size())throw std::invalid_argument("singleton incumbent length");
    Backbone b;b.support=incumbent;b.merged=incumbent;
    b.color.assign(g.n,-1);b.component.assign(g.n,-1);
    for(int v=0;v<g.n;++v){
        if((v&511)==0)d.check();
        if(incumbent[v]>1)throw std::invalid_argument("singleton incumbent mask");
        if(incumbent[v]){b.color[v]=0;b.component[v]=int(b.parts.size());b.parts.push_back({v});}
    }
    for(std::size_t i=0;i<g.edges.size();++i){
        if((i&1023)==0)d.check();
        auto [u,v]=g.edges[i];
        if(incumbent[u]&&incumbent[v])throw std::invalid_argument("infeasible singleton incumbent");
    }
    b.exact=true;return b;
}

inline Sketch selective_evaluate(const Graph&g,const Mask&incumbent,
                                std::vector<int>outside,const Deadline&d,
                                SelectiveStats*stats=nullptr) {
    if(incumbent.size()!=g.w.size())throw std::invalid_argument("sketch incumbent length");
    std::sort(outside.begin(),outside.end());
    if(std::adjacent_find(outside.begin(),outside.end())!=outside.end())
        throw std::invalid_argument("duplicate sketch outsider");
    // Tiny outsider sets are required by the intended gate. The cap also keeps
    // clique construction and repair witnesses bounded independently of n.
    if(outside.size()>24)throw std::invalid_argument("at most 24 sketch outsiders");
    Sketch s;s.outside=std::move(outside);
    std::vector<std::vector<int>>blocked(s.outside.size());
    std::vector<Weight>unary_gain(s.outside.size());
    std::unordered_map<int,int>degree;
    std::uint64_t visits=0;
    long double independent_cost=0;
    for(std::size_t i=0;i<s.outside.size();++i){
        d.check();int v=s.outside[i];
        if(v<0||v>=g.n||incumbent[v])throw std::invalid_argument("invalid sketch outsider");
        s.rewards+=g.w[v];s.predicted_work+=1.+g.adj[v].size();
        Weight unary_cost=0;
        for(int u:g.adj[v]){
            if((visits++&511)==0)d.check();
            if(incumbent[u]){blocked[i].push_back(u);++degree[u];independent_cost+=g.w[u];unary_cost+=g.w[u];}
        }
        unary_gain[i]=g.w[v]-unary_cost;
    }
    s.blockers.reserve(degree.size());
    for(const auto&item:degree)s.blockers.push_back(item.first);
    std::sort(s.blockers.begin(),s.blockers.end());
    std::uint64_t work=0;
    for(int u:s.blockers){if((work++&511)==0)d.check();s.union_cost+=g.w[u];s.predicted_work+=1.+g.adj[u].size();}
    s.shared_blocker_credit=double(independent_cost-static_cast<long double>(s.union_cost));
    s.allocated_cost.assign(s.outside.size(),0);s.reduced_reward.resize(s.outside.size());
    for(std::size_t i=0;i<s.outside.size();++i){
        for(int u:blocked[i]){if((work++&511)==0)d.check();s.allocated_cost[i]+=g.w[u]/degree.at(u);}
        s.reduced_reward[i]=g.w[s.outside[i]]-s.allocated_cost[i];
    }
    // First construct a partition of ALL outsiders into true conflict cliques.
    // Equal-share reduced rewards only determine its heuristic construction
    // order; no outsider is removed from the sketch or the partition.
    std::vector<int>partition_order(s.outside.size());std::iota(partition_order.begin(),partition_order.end(),0);
    std::sort(partition_order.begin(),partition_order.end(),[&](int a,int b){
        return s.reduced_reward[a]!=s.reduced_reward[b]?
            s.reduced_reward[a]>s.reduced_reward[b]:s.outside[a]<s.outside[b];
    });
    std::vector<std::vector<int>>cliques;std::vector<int>clique_id(s.outside.size(),-1);
    for(int i:partition_order){
        d.check();std::size_t c=0;
        for(;c<cliques.size();++c){
            bool all=true;for(int j:cliques[c])if(!g.adjacent(s.outside[i],s.outside[j])){all=false;break;}
            if(all)break;
        }
        if(c==cliques.size())cliques.push_back({i});else cliques[c].push_back(i);
        clique_id[i]=int(c);
    }
    // Let h_b count distinct conflict cliques touched by incumbent blocker b.
    // Any independent outsider subset S selects <=h_b neighbors of b, so
    // charging each neighbor floor(w_b/h_b) allocates <=w_b total. Hence
    // W(S)-W(N_I(S)) <= sum_{v in S} reduced_reward[v], and one maximum
    // positive reduced reward per clique is an admissible whole-domain bound.
    // Clique-aware charges are >= degree-based charges, tightening the bound.
    std::unordered_map<int,std::uint32_t>covered_cliques;
    for(std::size_t i=0;i<s.outside.size();++i)for(int u:blocked[i]){
        if((work++&511)==0)d.check();covered_cliques[u]|=1u<<clique_id[i];
    }
    std::fill(s.allocated_cost.begin(),s.allocated_cost.end(),0);
    for(std::size_t i=0;i<s.outside.size();++i){
        for(int u:blocked[i]){
            if((work++&511)==0)d.check();
            s.allocated_cost[i]+=g.w[u]/__builtin_popcount(covered_cliques.at(u));
        }
        s.reduced_reward[i]=g.w[s.outside[i]]-s.allocated_cost[i];
    }
    for(const auto&clique:cliques){Weight maximum=0;for(int i:clique)maximum=std::max(maximum,s.reduced_reward[i]);s.upper_gain+=maximum;}

    // A positive repair witness is a feasible lower bound, without constructing
    // a Kernel. Continue through negative marginals so shared removal costs can
    // reveal profitable coalitions whose singleton gains are all negative.
    std::vector<int>order(s.outside.size());std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](int a,int b){
        Weight x=unary_gain[a],y=unary_gain[b];return x!=y?x>y:s.outside[a]<s.outside[b];
    });
    for(int lead=0;s.upper_gain>0&&lead<std::min(4,int(order.size()));++lead){
        d.check();std::vector<int>selected;std::unordered_set<int>paid;
        std::vector<std::uint8_t>used(order.size(),0);Weight reward=0,cost=0;
        int next=order[lead];
        while(next>=0){
            used[next]=1;selected.push_back(next);reward+=g.w[s.outside[next]];
            for(int u:blocked[next]){if((work++&511)==0)d.check();if(paid.insert(u).second)cost+=g.w[u];}
            Weight gain=reward-cost;
            if(gain>s.witness_gain){
                s.witness_gain=gain;s.witness_outside.clear();
                for(int i:selected)s.witness_outside.push_back(s.outside[i]);
                std::sort(s.witness_outside.begin(),s.witness_outside.end());
            }
            next=-1;Weight best=std::numeric_limits<Weight>::min();
            for(int i:order)if(!used[i]){
                bool ok=true;for(int j:selected)if(g.adjacent(s.outside[i],s.outside[j])){ok=false;break;}
                if(!ok)continue;
                Weight extra=0;for(int u:blocked[i]){if((work++&511)==0)d.check();if(!paid.count(u))extra+=g.w[u];}
                Weight margin=g.w[s.outside[i]]-extra;
                if(next<0||margin>best||(margin==best&&s.outside[i]<s.outside[next])){best=margin;next=i;}
            }
        }
    }
    if(s.witness_gain>s.upper_gain)throw std::logic_error("invalid selective bounds");
    s.score=(2.*double(s.witness_gain)+.05*double(s.upper_gain)+.01*s.shared_blocker_credit)/std::sqrt(s.predicted_work);
    if(stats){stats->neighbor_visits+=visits;++stats->sketches;if(s.upper_gain==0)++stats->safe_zero;}
    return s;
}

inline std::vector<Sketch> selective_sketches(const Graph&g,const LocalSearch&local,
                                           const SelectiveConfig&cfg,std::mt19937_64&rng,
                                           std::uint64_t epoch,const Deadline&d,
                                           SelectiveStats*stats=nullptr) {
    if(local.s.size()!=g.w.size()||local.blocked_weight.size()!=g.w.size())
        throw std::invalid_argument("selective local state length");
    if(cfg.seed_trials<0||cfg.seed_trials>256||cfg.pool_cap<1||cfg.pool_cap>4096||
       cfg.challengers<0||cfg.challengers>24||cfg.random_fill<0||cfg.random_fill>256)
        throw std::invalid_argument("invalid selective configuration");
    if(cfg.seed_trials==0||cfg.challengers==0)return {};
    struct Entry{double score;int id;};
    struct Better{bool operator()(const Entry&a,const Entry&b)const{
        return a.score!=b.score?a.score>b.score:a.id<b.id;
    }};
    std::priority_queue<Entry,std::vector<Entry>,Better>top;
    std::vector<int>outside;outside.reserve(g.n);
    double scale=g.n?std::max(1.,double(g.total)/g.n):1.;Better better;
    try {
        for(int v=0;v<g.n;++v){
            if((v&511)==0)d.check();
            if(!local.s[v]){
                outside.push_back(v);if(stats)++stats->raw_candidates;
                Entry e{double(g.w[v])/(scale+double(local.blocked_weight[v])),v};
                if(top.size()<128)top.push(e);
                else if(better(e,top.top())){top.pop();top.push(e);}
            }
        }
    } catch(const Timeout&) {
        // An unfinished scalar scan is not a zero-opportunity certificate.
        if(stats){stats->partial=true;++stats->timeout_trials;}
        return {};
    }
    if(outside.empty())return {};
    std::vector<Entry>seeds;while(!top.empty()){seeds.push_back(top.top());top.pop();}
    std::sort(seeds.begin(),seeds.end(),better);
    std::set<std::vector<int>>seen;std::vector<Sketch>result;
    for(int trial=0;trial<cfg.seed_trials;++trial){
        try {
        d.check();int seed=trial%3==2?outside[rng()%outside.size()]:
            seeds[(epoch*7+std::uint64_t(trial)*11)%seeds.size()].id;
        std::vector<int>pool{seed};std::unordered_set<int>in_pool{seed};
        std::uint64_t visits=0;
        for(int u:g.adj[seed]){
            if(int(pool.size())>=cfg.pool_cap)break;
            if((visits++&511)==0)d.check();
            if(local.s[u])for(int v:g.adj[u]){
                if((visits++&511)==0)d.check();
                if(!local.s[v]&&in_pool.insert(v).second)pool.push_back(v);
                if(int(pool.size())>=cfg.pool_cap)break;
            }
            if(int(pool.size())>=cfg.pool_cap)break;
        }
        int filled=0;
        for(int attempt=0;filled<cfg.random_fill&&int(pool.size())<cfg.pool_cap&&attempt<cfg.random_fill*8+8;++attempt){
            if((attempt&31)==0)d.check();int v=outside[rng()%outside.size()];
            if(in_pool.insert(v).second){pool.push_back(v);++filled;}
        }
        // Cache sparse blocker lists once per pool instead of rescanning every
        // candidate adjacency list at each of the <=12 greedy selection steps.
        std::vector<std::vector<int>>blockers(pool.size());
        for(std::size_t i=0;i<pool.size();++i)for(int u:g.adj[pool[i]]){
            if((visits++&511)==0)d.check();if(local.s[u])blockers[i].push_back(u);
        }
        std::vector<int>chosen{0};std::vector<std::uint8_t>used(pool.size(),0);used[0]=1;
        std::unordered_set<int>paid(blockers[0].begin(),blockers[0].end());
        while(int(chosen.size())<cfg.challengers&&chosen.size()<pool.size()){
            d.check();int best=-1;double best_score=-std::numeric_limits<double>::infinity();
            for(int i=0;i<int(pool.size());++i)if(!used[i]){
                Weight extra=0;for(int u:blockers[i])if(!paid.count(u))extra+=g.w[u];
                int conflicts=0;for(int j:chosen)if(g.adjacent(pool[i],pool[j]))++conflicts;
                double score=double(g.w[pool[i]])-double(extra)-.5*scale*conflicts;
                if(best<0||score>best_score||(score==best_score&&pool[i]<pool[best])){best=i;best_score=score;}
            }
            if(best<0)break;chosen.push_back(best);used[best]=1;
            for(int u:blockers[best])paid.insert(u);
        }
        std::vector<int>ids;for(int i:chosen)ids.push_back(pool[i]);std::sort(ids.begin(),ids.end());
        if(stats){stats->neighbor_visits+=visits;stats->pool_candidates+=pool.size();}
        if(!seen.insert(ids).second){if(stats)++stats->duplicates;continue;}
        result.push_back(selective_evaluate(g,local.s,std::move(ids),d,stats));
        } catch(const Timeout&) {
            // Keep completed sketches, but never expose an unfinished bound or
            // turn an interrupted evaluate call into a certified-zero sketch.
            if(stats){stats->partial=true;++stats->timeout_trials;}
            break;
        }
    }
    std::stable_sort(result.begin(),result.end(),[](const Sketch&a,const Sketch&b){
        if(a.score!=b.score)return a.score>b.score;
        if(a.witness_gain!=b.witness_gain)return a.witness_gain>b.witness_gain;
        if(a.upper_gain!=b.upper_gain)return a.upper_gain>b.upper_gain;
        return a.outside<b.outside;
    });
    return result;
}

} // namespace barr
