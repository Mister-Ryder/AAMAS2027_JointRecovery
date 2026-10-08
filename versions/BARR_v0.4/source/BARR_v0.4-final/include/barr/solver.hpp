#pragma once
#include "local_search.hpp"
#include "policy.hpp"
#include "selective.hpp"
#include "pair_scout.hpp"
#include <filesystem>
#include "kernel_io.hpp"
#include <atomic>
#include <cstdio>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
namespace barr {
struct Config {
    std::uint64_t seed=17,kernel_nodes=128;
    int population=4,threads=1,local_iterations=64,challengers=12,proposals=6,execute_top=2;
    int max_kernel_vertices=4096,max_gnn_vertices=4096,max_rounds=-1;
    double seconds=10.,local_seconds=.025,fusion_seconds=.10,kernel_seconds=.03;
    bool fixed_k=false,random_domains=false,decompose=true,greedy_repair=false;
    std::string mode="selective",rank="heuristic",model,events,trace,checkpoint;
    std::string recovery_backend="hybrid",kernel_snapshots;
    std::string gate="value",sketch_snapshots;
    double gate_fraction=.05,gate_cooldown=.25,gate_warmup=30.;
    int event_stale=8,explore_every=8,gate_trials=8;
    std::string pair_policy="fusion-refine";
    std::string pair_component="full"; // One ablation switch; default B operation semantics.
    double pair_slice=.5;
    int pair_max_seeds=0,pair_fusion_every=4;
    FactorOptions factor_options;
};
struct Stats {
#ifdef BARR_TEST_FIXED_CLOCK
    FixedRoundCounter rounds;
    std::uint64_t local_iterations=0,fusions=0,fusion_exact=0,recoveries=0,recovery_exact=0;
#else
    std::uint64_t rounds=0,local_iterations=0,fusions=0,fusion_exact=0,recoveries=0,recovery_exact=0;
#endif
    std::uint64_t accepted_recoveries=0,outsiders_inserted=0,consensus_released=0;
    std::uint64_t branch_nodes=0,cut_calls=0,cache_hits=0,model_batches=0,model_fallback_batches=0,restarts=0;
    std::uint64_t gate_epochs=0,raw_candidates=0,sketches=0,safe_rejects=0,neighbor_visits=0,heuristic_skips=0;
    std::uint64_t exploration_openings=0,witness_openings=0,kernels_materialized=0,gate_budget_skips=0,gate_timeouts=0,gate_oversized=0,witness_moves=0;
    std::uint64_t pair_visits=0,pair_unique=0,pair_upper_rejects=0,pair_conflict_rejects=0,pair_exact_checks=0,pair_positive_candidates=0,pair_complete=0,pair_certificates=0,pair_cached_zero=0;
    Weight refinement_extra_ticks=0,positive_gain_ticks=0;
    double gate_seconds=0,build_seconds=0,escalation_seconds=0;
    double local_seconds=0,fusion_seconds=0,proposal_seconds=0,inference_seconds=0,recovery_seconds=0;
    std::uint64_t factor_attempts=0,factor_exact=0,factor_fallbacks=0,factor_compiled_entries=0,factor_dp_entries=0;
};
inline void mask_json(std::ostream&o,const Mask&s){
    o<<'[';bool first=true;for(int i=0;i<int(s.size());++i)if(s[i]){if(!first)o<<',';o<<i;first=false;}o<<']';
}
inline void write_checkpoint(const std::string&path,const Graph&g,const Mask&s){
    if(path.empty())return;
    std::ofstream o(path+".tmp");if(!o)throw std::runtime_error("cannot write checkpoint");
    o<<std::setprecision(17)<<"{\"schema\":\"barr_checkpoint_v1\",\"tick_value\":"<<g.value(s)<<",\"original_value\":"<<g.raw_value(s)<<",\"selected\":";
    mask_json(o,s);o<<"}\n";o.close();if(!o)throw std::runtime_error("checkpoint write failed");
    if(std::rename((path+".tmp").c_str(),path.c_str())!=0)throw std::runtime_error("checkpoint rename failed");
}
inline void features_json(std::ostream&o,const Features&f){
    o<<"\"x\":[";
    for(std::size_t i=0;i<f.x.size();++i){if(i)o<<',';o<<'[';for(int j=0;j<FEATURE_DIM;++j){if(j)o<<',';o<<f.x[i][j];}o<<']';}o<<"],\"edges\":[";
    for(std::size_t i=0;i<f.edges.size();++i){if(i)o<<',';auto[u,v,t]=f.edges[i];o<<'['<<u<<','<<v<<','<<t<<']';}o<<"],\"context\":[";
    for(int j=0;j<CONTEXT_DIM;++j){if(j)o<<',';o<<f.context[j];}o<<"],\"weight_scale\":"<<f.weight_scale;
}
struct SolverResult {Mask selected;Stats stats;double seconds=0,cpu_seconds=0;};
#include "selective_solver_impl.hpp"
#include "pair_solver_impl.hpp"
inline SolverResult solve(const Graph&g,const Config&c){
    if(c.population<1||c.population>256||c.threads<1||c.threads>256||c.challengers<0||c.challengers>128||c.execute_top<1||c.proposals<1||c.local_iterations<1)
        throw std::invalid_argument("invalid solver configuration");
    if(c.mode!="local"&&c.mode!="fusion"&&c.mode!="barr"&&c.mode!="selective"&&c.mode!="pair")throw std::invalid_argument("unknown mode");
    if(c.rank!="heuristic"&&c.rank!="random"&&c.rank!="independent"&&c.rank!="gnn")throw std::invalid_argument("unknown rank");
    validate_factor_options(c.factor_options);
    if(c.recovery_backend!="hybrid"&&c.recovery_backend!="factor"&&c.recovery_backend!="branch")throw std::invalid_argument("unknown recovery backend");
    if(c.mode=="pair"){if(c.threads!=1)throw std::invalid_argument("pair currently requires one thread");return solve_pair(g,c);}
    if(c.mode=="selective"){if(c.threads!=1)throw std::invalid_argument("selective currently requires one thread");return solve_selective(g,c);}
    auto started=Clock::now();std::clock_t cpu_start=std::clock();Deadline deadline=Deadline::after(c.seconds);
    SolverResult result;result.selected=g.initial;long double bestraw=g.raw_value(result.selected);Stats&st=result.stats;
    NeuralPolicy model;if(c.rank=="gnn"){if(c.model.empty())throw std::invalid_argument("--rank gnn requires --model");model.load(c.model);}
    std::ofstream events,trace;
    if(!c.events.empty()){events.open(c.events);if(!events)throw std::runtime_error("cannot open events");events<<std::setprecision(17);}
    if(!c.trace.empty()){trace.open(c.trace);if(!trace)throw std::runtime_error("cannot open trace");trace<<std::setprecision(17);}
    write_checkpoint(c.checkpoint,g,result.selected);
    auto remember=[&](const Mask&s){
        auto value=g.raw_value(s);
        if(value>bestraw){if(!g.feasible(s))throw std::logic_error("infeasible archive candidate");result.selected=s;bestraw=value;write_checkpoint(c.checkpoint,g,s);}
    };
    if(g.n==0){result.seconds=elapsed(started);return result;}
    std::vector<std::unique_ptr<LocalSearch>>pop;
    for(int i=0;i<c.population;++i)pop.push_back(std::make_unique<LocalSearch>(g,g.initial,c.seed+0x9e3779b97f4a7c15ULL*std::uint64_t(i+1)));
    // Warm-start data is preserved in the archive; diverse constructors may be worse.
    for(int i=0;i<c.population && !deadline.expired();++i){
        try{
            auto d=deadline.slice(std::max(.01,std::min(.2,c.seconds*.04)));
            if(i==0){if(g.value(g.initial)==0)pop[i]->greedy_start(.5,0.,d);else pop[i]->descent(d);}
            else pop[i]->greedy_start((i%3)*.5,.25,d);
        }catch(const Timeout&){}
        remember(pop[i]->s);
    }
    std::mt19937_64 rng(c.seed^0xa0761d6478bd642fULL);int stale=0;
    while(!deadline.expired() && (c.max_rounds<0 || st.rounds<std::uint64_t(c.max_rounds))){
        std::uint64_t round=st.rounds++;auto begin=Clock::now();
        std::atomic<int>next{0};std::exception_ptr error;std::mutex error_mutex;
        auto worker=[&](){
            try{int i;while((i=next.fetch_add(1))<c.population){auto d=deadline.slice(c.local_seconds);pop[i]->run(c.local_iterations,d);}}
            catch(...){std::lock_guard<std::mutex>lock(error_mutex);if(!error)error=std::current_exception();}
        };
        int workers=std::min(c.threads,c.population);
        if(workers==1)worker();else{std::vector<std::thread>ts;for(int t=0;t<workers;++t)ts.emplace_back(worker);for(auto&t:ts)t.join();}
        if(error)std::rethrow_exception(error);
        st.local_seconds+=elapsed(begin);for(auto&s:pop)remember(s->s);
        if(deadline.expired())break;
        if(c.mode=="local")continue;
        int elite=0;for(int i=1;i<c.population;++i)if(pop[i]->cost>pop[elite]->cost)elite=i;
        int pa=(round%2==0)?elite:int(round%std::uint64_t(c.population)),pb=pa;Weight farthest=-1;
        for(int j=0;j<c.population;++j)if(j!=pa){Weight distance=0;for(int v=0;v<g.n;++v)if(pop[pa]->s[v]!=pop[j]->s[v])distance+=g.w[v];
            if(distance>farthest){farthest=distance;pb=j;}}
        const Mask parent0=pop[pa]->s,parent1=pop[pb]->s;
        begin=Clock::now();Backbone backbone=fuse(g,parent0,parent1,deadline.slice(c.fusion_seconds));
        st.fusion_seconds+=elapsed(begin);++st.fusions;st.fusion_exact+=backbone.exact;st.cut_calls+=backbone.cuts;
        remember(backbone.merged);Mask next_solution=backbone.merged;Weight next_value=g.value(next_solution);
        Weight fusion_gain=next_value-std::max(g.value(parent0),g.value(parent1));
        bool augmented=false;
        if(c.mode=="barr"&&backbone.exact&&!deadline.expired()){
            try{
                std::vector<double>freq(g.n,0);for(auto&s:pop)for(int v=0;v<g.n;++v)freq[v]+=double(s->s[v])/c.population;
                int nk=c.fixed_k?c.challengers:std::min(c.challengers,4*(1+std::min(5,stale/3)));
                begin=Clock::now();ProposalBuildStats ps;auto proposals=propose(g,backbone,parent0,parent1,freq,c.proposals,nk,c.max_kernel_vertices,rng,round,c.random_domains,deadline,&ps);
                st.proposal_seconds+=elapsed(begin);
                if(events.is_open())events<<"{\"event\":\"proposal_summary\",\"round\":"<<round<<",\"trials\":"<<ps.trials
                    <<",\"duplicates\":"<<ps.duplicates<<",\"oversized\":"<<ps.oversized<<",\"generated\":"<<proposals.size()<<"}\n";
                if(c.rank=="random")for(auto&pr:proposals)pr.score=std::generate_canonical<double,53>(rng);
                if(c.rank=="independent")for(auto&pr:proposals)pr.score=pr.independent;
                if(c.rank=="gnn" && !proposals.empty()){
                    begin=Clock::now();bool use_model=true;for(auto&pr:proposals)if(pr.kernel.graph.n>c.max_gnn_vertices)use_model=false;
                    if(use_model){
                        try{for(auto&pr:proposals)pr.score=model.score(features(pr.kernel,c.kernel_nodes,c.kernel_seconds),deadline);++st.model_batches;}
                        catch(const Timeout&){use_model=false;}
                    }
                    if(!use_model){for(auto&pr:proposals)pr.score=pr.heuristic;++st.model_fallback_batches;}
                    st.inference_seconds+=elapsed(begin);
                }
                std::stable_sort(proposals.begin(),proposals.end(),[](const auto&a,const auto&b){return a.score>b.score;});
                struct Observed{Features x;Recovery r;double seconds;std::size_t index;};std::vector<Observed>obs;
                int run_count=trace.is_open()?int(proposals.size()):std::min(c.execute_top,int(proposals.size()));bool complete=true;
                std::uint64_t outsiders_changed=0,consensus_changed=0;
                for(int j=0;j<run_count;++j){
                    if(deadline.remaining()<c.kernel_seconds){complete=false;if(deadline.expired())break;}
                    const auto&pr=proposals[j];
                    if(!c.kernel_snapshots.empty())write_kernel(pr.kernel,c.kernel_snapshots+"/kernel_"+std::to_string(round)+"_"+std::to_string(j)+".barrk");
                    auto rd=deadline.slice(c.kernel_seconds);begin=Clock::now();
                    DispatchRecovery dispatched;
                    if(c.greedy_repair){dispatched.result=greedy_recover(pr.kernel,rd);dispatched.backend="greedy";}
                    else dispatched=recover_dispatch(pr.kernel,c.recovery_backend,c.factor_options,c.kernel_nodes,rd,c.decompose);
                    auto rec=std::move(dispatched.result);
                    if(dispatched.factor.status!="not_started"){
                        ++st.factor_attempts;st.factor_exact+=dispatched.factor.status=="exact";
                        st.factor_fallbacks+=dispatched.backend=="factor_then_branch"||dispatched.backend=="factor_then_greedy";
                        st.factor_compiled_entries+=dispatched.factor.compiled_entries;st.factor_dp_entries+=dispatched.factor.dp_entries;
                    }
                    double used=elapsed(begin);st.recovery_seconds+=used;++st.recoveries;st.recovery_exact+=rec.exact;
                    st.branch_nodes+=rec.nodes;st.cut_calls+=rec.cuts;st.cache_hits+=rec.cache_hits;
                    Weight base=pr.kernel.graph.value(pr.kernel.base),gain=rec.lower-base;
                    bool chosen_for_commit=j<c.execute_top && g.value(backbone.merged)+gain>next_value;
                    if(chosen_for_commit){
                        next_solution=lift(g,backbone,pr.kernel,rec);next_value=g.value(next_solution);augmented=true;
                        outsiders_changed=consensus_changed=0;
                        for(int v=0;v<g.n;++v){
                            if(next_solution[v]&&freq[v]<1e-12)++outsiders_changed;
                            if(backbone.merged[v]&&!next_solution[v]&&freq[v]>1.-1e-12)++consensus_changed;
                        }
                    }
                    if(events.is_open()){
                        events<<"{\"event\":\"recovery\",\"round\":"<<round<<",\"rank\":"<<j<<",\"kernel_n\":"<<pr.kernel.graph.n<<",\"outsiders\":"<<pr.kernel.outsiders.size()
                              <<",\"base_ticks\":"<<base<<",\"lower_ticks\":"<<rec.lower<<",\"upper_ticks\":"<<rec.upper<<",\"exact\":"<<(rec.exact?"true":"false")
                              <<",\"gain_ticks\":"<<gain<<",\"nodes\":"<<rec.nodes<<",\"cuts\":"<<rec.cuts<<",\"components\":"<<rec.components
                              <<",\"max_component_k\":"<<rec.max_component_k<<",\"solver_seconds\":"<<used<<",\"in_policy_top\":"<<(j<c.execute_top?"true":"false")<<",\"backend\":\""<<dispatched.backend<<"\",\"factor\":";
                        factor_stats_json(events,dispatched.factor);events<<"}\n";
                    }
                    if(trace.is_open())obs.push_back({features(pr.kernel,c.kernel_nodes,c.kernel_seconds),std::move(rec),used,std::size_t(j)});
                }
                if(trace.is_open() && !proposals.empty()){
                    complete=complete&&obs.size()==proposals.size();
                    trace<<"{\"schema\":\"barr_counterfactual_v1\",\"round\":"<<round<<",\"complete\":"<<(complete?"true":"false")<<",\"policy_top\":"<<c.execute_top
                         <<",\"parent_signatures\":["<<signature(parent0)<<','<<signature(parent1)<<"],\"actions\":[";
                    for(std::size_t t=0;t<obs.size();++t){if(t)trace<<',';const auto&o=obs[t];const auto&k=proposals[o.index].kernel;
                        trace<<'{';features_json(trace,o.x);trace<<",\"rank\":"<<o.index<<",\"base_ticks\":"<<k.graph.value(k.base)
                            <<",\"gain_ticks\":"<<o.r.lower-k.graph.value(k.base)<<",\"upper_gain_ticks\":"<<o.r.upper-k.graph.value(k.base)
                            <<",\"exact\":"<<(o.r.exact?"true":"false")<<",\"nodes\":"<<o.r.nodes<<",\"solver_seconds\":"<<o.seconds<<'}';
                    }trace<<"]}\n";
                }
                if(augmented){++st.accepted_recoveries;st.outsiders_inserted+=outsiders_changed;st.consensus_released+=consensus_changed;}
            }catch(const Timeout&){}
        }
        remember(next_solution);
        int replace=pop[pa]->cost<=pop[pb]->cost?pa:pb;
        if(next_value>=pop[replace]->cost)pop[replace]->reset(next_solution);
        if(fusion_gain>0||augmented)stale=0;else ++stale;
        // Diversity restarts are ordinary heuristic machinery, not a novelty claim.
        if(stale>0 && stale%12==0 && c.population>1 && !deadline.expired()){
            int worst=-1;for(int i=0;i<c.population;++i)if(i!=elite && (worst<0||pop[i]->cost<pop[worst]->cost))worst=i;
            if(worst>=0){try{pop[worst]->greedy_start(.2+.8*std::generate_canonical<double,53>(rng),1.,deadline.slice(c.local_seconds));}catch(const Timeout&){}
                remember(pop[worst]->s);++st.restarts;}
        }
    }
    for(auto&s:pop){remember(s->s);st.local_iterations+=s->iterations;}
    if(!g.feasible(result.selected)||g.raw_value(result.selected)<g.raw_value(g.initial))throw std::logic_error("invalid final incumbent");
    write_checkpoint(c.checkpoint,g,result.selected);
    result.seconds=elapsed(started);result.cpu_seconds=double(std::clock()-cpu_start)/CLOCKS_PER_SEC;return result;
}
inline void result_json(std::ostream&o,const Graph&g,const Config&c,const SolverResult&r){
    const auto&s=r.stats;o<<std::setprecision(17);
    o<<"{\"schema\":\"barr_result_v1\",\"algorithm\":\"BARR-0.4E\",\"mode\":\""<<c.mode<<"\",\"rank\":\""<<c.rank
     <<"\",\"recovery_backend\":\""<<c.recovery_backend<<"\",\"tick_value\":"<<g.value(r.selected)<<",\"original_value\":"<<g.raw_value(r.selected)<<",\"initial_original_value\":"<<g.raw_value(g.initial)
     <<",\"vertices\":"<<g.n<<",\"edges\":"<<g.edges.size()<<",\"selected\":";mask_json(o,r.selected);
    o<<",\"native_seconds\":"<<r.seconds<<",\"process_cpu_seconds\":"<<r.cpu_seconds<<",\"population\":"<<c.population<<",\"threads\":"<<c.threads
     <<",\"seed\":"<<c.seed<<",\"pair_policy\":\""<<c.pair_policy<<"\",\"pair_component\":\""<<c.pair_component<<"\",\"collection_mode\":"<<(!c.trace.empty()?"true":"false")<<",\"stats\":{";
#define FIELD(name) o<<"\"" #name "\":"<<s.name<<','
    FIELD(rounds);FIELD(local_iterations);FIELD(fusions);FIELD(fusion_exact);FIELD(recoveries);FIELD(recovery_exact);FIELD(accepted_recoveries);
    FIELD(outsiders_inserted);FIELD(consensus_released);FIELD(branch_nodes);FIELD(cut_calls);FIELD(cache_hits);FIELD(model_batches);FIELD(model_fallback_batches);FIELD(restarts);
    FIELD(factor_attempts);FIELD(factor_exact);FIELD(factor_fallbacks);FIELD(factor_compiled_entries);FIELD(factor_dp_entries);
    FIELD(gate_epochs);FIELD(raw_candidates);FIELD(sketches);FIELD(safe_rejects);FIELD(neighbor_visits);FIELD(heuristic_skips);
    FIELD(exploration_openings);FIELD(witness_openings);FIELD(kernels_materialized);FIELD(gate_budget_skips);FIELD(gate_timeouts);FIELD(gate_oversized);FIELD(witness_moves);
    FIELD(pair_visits);FIELD(pair_unique);FIELD(pair_upper_rejects);FIELD(pair_conflict_rejects);FIELD(pair_exact_checks);FIELD(pair_positive_candidates);FIELD(pair_complete);FIELD(pair_certificates);FIELD(pair_cached_zero);
    FIELD(refinement_extra_ticks);FIELD(positive_gain_ticks);FIELD(gate_seconds);FIELD(build_seconds);FIELD(escalation_seconds);
    FIELD(local_seconds);FIELD(fusion_seconds);FIELD(proposal_seconds);FIELD(inference_seconds);
#undef FIELD
    o<<"\"recovery_seconds\":"<<s.recovery_seconds<<"},\"global_optimality_proven\":false,\"pair_scout_enabled\":"<<(c.mode=="pair"&&c.pair_component=="full"?"true":"false")
     <<",\"pair_snapshot_collection_enabled\":"<<(c.mode=="pair"&&c.pair_component=="full"&&!c.sketch_snapshots.empty()?"true":"false")<<",\"certificate_domain\":\""
     <<(c.mode=="pair"?(c.pair_component=="pulse-only"?"pair opportunities not evaluated; no pair absence certificate":"fixed-incumbent <=2 outsider exchanges only when one-optimal and full scout completed; individual augmented kernels otherwise")
        :"individual integer-weight augmented kernels only")<<"\"}\n";
}
} // namespace barr
