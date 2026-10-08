#pragma once
#include "local_search.hpp"

namespace barr {

struct PairScoutConfig {
    // Zero scans every outsider. Positive values rotate a heuristic seed cap;
    // a capped scan never certifies absence of all two-outsider improvements.
    int max_seeds=0;
};
struct PairScoutStats {
    std::uint64_t inspected_vertices=0, outside_vertices=0, seeds=0;
    std::uint64_t blocker_visits=0, pair_visits=0, unique_pairs=0;
    std::uint64_t upper_rejects=0, conflict_rejects=0, exact_pairs=0;
    std::uint64_t intersection_visits=0, positive_singletons=0, positive_pairs=0;
    bool complete=false, one_optimal=false, partial=false, capped=false;
};
struct PairOpportunity {
    std::vector<int> outside;
    Weight witness_gain=0;
    // A full shared-blocker scan covers all potentially positive pairs only
    // when every singleton replacement has nonpositive gain.
    bool complete_two_exchange=false;
    // True only for a completed absence certificate (witness_gain==0).
    bool certificate_complete=false;
    double seconds=0;
    PairScoutStats stats;
};

// For feasible incumbent I and mutually compatible outsiders a,b, write
// d_v=W(N_I(v))-w_v. Their exact insertion/removal gain is
// W(N_I(a) intersect N_I(b))-d_a-d_b. If all d_v>=0, a positive pair MUST
// share an incumbent blocker. Enumerating only blocker two-hop pairs is then
// complete for positive two-outsider exchanges, without inducing any Kernel.
// Workspace is O(n+sum_v |N_I(v)|); there is no n*n table or per-pair allocation.
inline PairOpportunity pair_scout(const Graph&g,const LocalSearch&local,
                                  const Deadline&d,const PairScoutConfig&cfg={},
                                  std::uint64_t epoch=0) {
    if(cfg.max_seeds<0)throw std::invalid_argument("negative pair scout seed cap");
    if(local.s.size()!=g.w.size()||local.blocks.size()!=g.w.size()||
       local.blocked_weight.size()!=g.w.size())throw std::invalid_argument("pair scout state length");
    const auto started=Clock::now();PairOpportunity result;result.outside.reserve(2);
    auto remember=[&](int a,int b,Weight gain){
        if(gain<=0)return;int count=b<0?1:2;
        bool better=gain>result.witness_gain;
        if(gain==result.witness_gain&&gain>0){
            if(std::size_t(count)<result.outside.size())better=true;
            else if(std::size_t(count)==result.outside.size()){
                better=a<result.outside[0]||(a==result.outside[0]&&count==2&&b<result.outside[1]);
            }
        }
        if(better){result.witness_gain=gain;result.outside.resize(count);result.outside[0]=a;if(count==2)result.outside[1]=b;}
    };
    std::uint64_t work=0;
    auto check=[&](){if((work++&1023)==0)d.check();};
    try {
        d.check();std::vector<int>outside;outside.reserve(g.n);
        std::vector<Weight>deficit(g.n,0);
        std::vector<std::size_t>offset(std::size_t(g.n)+1,0);
        for(int v=0;v<g.n;++v){
            check();++result.stats.inspected_vertices;
            if(local.s[v]>1||local.blocks[v]<0||std::size_t(local.blocks[v])>g.adj[v].size()||local.blocked_weight[v]<0)
                throw std::invalid_argument("invalid pair scout local state");
            if(local.s[v]){
                if(local.blocks[v]!=0)throw std::invalid_argument("infeasible pair scout incumbent");
                offset[v+1]=offset[v];
            } else {
                outside.push_back(v);++result.stats.outside_vertices;
                deficit[v]=local.blocked_weight[v]-g.w[v];
                offset[v+1]=offset[v]+std::size_t(local.blocks[v]);
                if(deficit[v]<0){++result.stats.positive_singletons;remember(v,-1,-deficit[v]);}
            }
        }
        result.stats.one_optimal=result.stats.positive_singletons==0;
        std::vector<int>blockers(offset.back());
        std::vector<std::size_t>cursor=offset;
        std::vector<Weight>observed_weight(g.n,0);
        // Visit selected blocker IDs in ascending order. Each outsider's flat
        // blocker list is therefore sorted, ready for allocation-free merge.
        for(int u=0;u<g.n;++u){
            check();if(!local.s[u])continue;
            for(int v:g.adj[u]){
                check();++result.stats.blocker_visits;
                if(local.s[v])throw std::invalid_argument("infeasible pair scout incumbent");
                if(cursor[v]>=offset[v+1])throw std::logic_error("pair scout blocker count mismatch");
                blockers[cursor[v]++]=u;observed_weight[v]+=g.w[u];
            }
        }
        for(int v:outside){
            check();
            if(cursor[v]!=offset[v+1]||observed_weight[v]!=local.blocked_weight[v])
                throw std::logic_error("pair scout blocked-weight state mismatch");
        }
        const std::size_t seed_limit=cfg.max_seeds==0?outside.size():
            std::min(outside.size(),std::size_t(cfg.max_seeds));
        result.stats.capped=seed_limit<outside.size();
        // Rotate even full scans: repeated deadline-limited sweeps must not
        // repeatedly censor the same high-ID seeds. A completed full rotation
        // still visits every seed once, so a<b keeps pair coverage unchanged.
        std::size_t start=outside.empty()?0:
            (result.stats.capped?epoch*std::uint64_t(seed_limit):epoch*9973ULL)%outside.size();
        std::vector<std::uint32_t>seen(g.n,0);std::uint32_t stamp=0;
        for(std::size_t seed=0;seed<seed_limit;++seed){
            d.check();int a=outside[(start+seed)%outside.size()];++result.stats.seeds;
            if(++stamp==0){std::fill(seen.begin(),seen.end(),0);++stamp;}
            for(std::size_t z=offset[a];z<offset[a+1];++z){
                check();int u=blockers[z];
                for(int b:g.adj[u]){
                    check();++result.stats.pair_visits;
                    if(b<=a||local.s[b]||seen[b]==stamp)continue;
                    seen[b]=stamp;++result.stats.unique_pairs;
                    // The sum of two deficits is within +/-2*MAX_TOTAL, safely
                    // inside signed int64. This is a proof bound, not a score.
                    Weight deficits=deficit[a]+deficit[b];
                    Weight upper=std::min(local.blocked_weight[a],local.blocked_weight[b])-deficits;
                    if(upper<=0){++result.stats.upper_rejects;continue;}
                    if(g.adjacent(a,b)){++result.stats.conflict_rejects;continue;}
                    ++result.stats.exact_pairs;Weight shared=0;
                    std::size_t i=offset[a],j=offset[b];
                    while(i<offset[a+1]&&j<offset[b+1]){
                        check();++result.stats.intersection_visits;
                        int x=blockers[i],y=blockers[j];
                        if(x==y){shared+=g.w[x];++i;++j;}
                        else if(x<y)++i;else ++j;
                    }
                    Weight gain=shared-deficits;
                    if(gain>0){++result.stats.positive_pairs;remember(a,b,gain);}
                }
            }
        }
        result.stats.complete=!result.stats.capped;
    } catch(const Timeout&) {
        // Already verified positive witnesses remain usable; unfinished scans
        // never become pair-optimality or absence certificates.
        result.stats.partial=true;result.stats.complete=false;
    }
    result.complete_two_exchange=result.stats.complete&&result.stats.one_optimal;
    result.certificate_complete=result.complete_two_exchange&&result.witness_gain==0;
    result.seconds=elapsed(started);return result;
}

} // namespace barr
