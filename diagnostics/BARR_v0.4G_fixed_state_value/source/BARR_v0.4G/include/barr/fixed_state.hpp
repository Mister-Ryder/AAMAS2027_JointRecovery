#pragma once
#include "solver.hpp"
#include "state_wire.hpp"
#include <set>
namespace barr { namespace fixed {
// Only algorithm parameters travel; replay cannot reuse capture output paths.
#define FS_CONFIG_INT(X) X(population) X(threads) X(local_iterations) X(challengers) X(proposals) X(execute_top) X(max_kernel_vertices) X(max_gnn_vertices) X(max_rounds) X(event_stale) X(explore_every) X(gate_trials) X(pair_max_seeds) X(pair_fusion_every)
#define FS_CONFIG_DOUBLE(X) X(seconds) X(local_seconds) X(fusion_seconds) X(kernel_seconds) X(gate_fraction) X(gate_cooldown) X(gate_warmup) X(pair_slice)
#define FS_CONFIG_STRING(X) X(mode) X(rank) X(recovery_backend) X(gate) X(pair_policy) X(pair_component)
inline Config frozen_config(){Config c;c.mode="pair";c.seed=101;c.seconds=360;c.threads=1;c.population=4;c.rank="heuristic";c.recovery_backend="hybrid";c.kernel_nodes=128;c.kernel_seconds=.03;c.challengers=12;c.proposals=6;c.execute_top=2;c.factor_options.max_width=10;c.factor_options.max_boundary=10;c.factor_options.max_entries=262144;c.local_iterations=64;c.local_seconds=.025;c.gate_fraction=.05;c.gate_warmup=30;c.gate_cooldown=.5;c.event_stale=8;c.pair_slice=.5;c.pair_max_seeds=0;c.pair_fusion_every=4;c.pair_policy="fusion-refine";c.pair_component="full";return c;}
inline void config_write(Writer&w,const Config&c){w.u(c.seed);w.u(c.kernel_nodes);
#define X(a) w.i(c.a);
FS_CONFIG_INT(X)
#undef X
#define X(a) w.d(c.a);
FS_CONFIG_DOUBLE(X)
#undef X
#define X(a) w.text(c.a);
FS_CONFIG_STRING(X)
#undef X
    w.u(c.fixed_k);w.u(c.random_domains);w.u(c.decompose);w.u(c.greedy_repair);w.i(c.factor_options.max_width);w.i(c.factor_options.max_boundary);w.i(c.factor_options.max_variables);w.u(c.factor_options.max_entries);
}
inline Config config_read(Reader&r){Config c;c.seed=r.u();c.kernel_nodes=r.u();
#define X(a) c.a=int(r.i());
FS_CONFIG_INT(X)
#undef X
#define X(a) c.a=r.d();
FS_CONFIG_DOUBLE(X)
#undef X
#define X(a) c.a=r.text();
FS_CONFIG_STRING(X)
#undef X
    c.fixed_k=bool(r.u());c.random_domains=bool(r.u());c.decompose=bool(r.u());c.greedy_repair=bool(r.u());c.factor_options.max_width=int(r.i());c.factor_options.max_boundary=int(r.i());c.factor_options.max_variables=int(r.i());c.factor_options.max_entries=r.u();return c;
}
inline void config_json(std::ostream&o,const Config&c){o<<"{\"seed\":"<<c.seed<<",\"kernel_nodes\":"<<c.kernel_nodes;
#define X(a) o<<",\"" #a "\":"<<c.a;
FS_CONFIG_INT(X)
FS_CONFIG_DOUBLE(X)
#undef X
#define X(a) o<<",\"" #a "\":";quoted(o,c.a);
FS_CONFIG_STRING(X)
#undef X
o<<",\"decompose\":"<<(c.decompose?"true":"false")<<",\"factor_width\":"<<c.factor_options.max_width<<",\"factor_boundary\":"<<c.factor_options.max_boundary<<",\"factor_entries\":"<<c.factor_options.max_entries<<'}';}
#define FS_STATS_U(X) X(local_iterations) X(fusions) X(fusion_exact) X(recoveries) X(recovery_exact) X(accepted_recoveries) X(outsiders_inserted) X(consensus_released) X(branch_nodes) X(cut_calls) X(cache_hits) X(model_batches) X(model_fallback_batches) X(restarts) X(gate_epochs) X(raw_candidates) X(sketches) X(safe_rejects) X(neighbor_visits) X(heuristic_skips) X(exploration_openings) X(witness_openings) X(kernels_materialized) X(gate_budget_skips) X(gate_timeouts) X(gate_oversized) X(witness_moves) X(pair_visits) X(pair_unique) X(pair_upper_rejects) X(pair_conflict_rejects) X(pair_exact_checks) X(pair_positive_candidates) X(pair_complete) X(pair_certificates) X(pair_cached_zero) X(factor_attempts) X(factor_exact) X(factor_fallbacks) X(factor_compiled_entries) X(factor_dp_entries)
#define FS_STATS_D(X) X(gate_seconds) X(build_seconds) X(escalation_seconds) X(local_seconds) X(fusion_seconds) X(proposal_seconds) X(inference_seconds) X(recovery_seconds)
inline void stats_write(Writer&w,const Stats&s){w.u(std::uint64_t(s.rounds));
#define X(a) w.u(s.a);
FS_STATS_U(X)
#undef X
    w.i(s.refinement_extra_ticks);w.i(s.positive_gain_ticks);
#define X(a) w.d(s.a);
FS_STATS_D(X)
#undef X
}
inline Stats stats_read(Reader&r){Stats s;
#ifdef BARR_TEST_FIXED_CLOCK
    s.rounds.value=r.u();
#else
    s.rounds=r.u();
#endif
#define X(a) s.a=r.u();
FS_STATS_U(X)
#undef X
    s.refinement_extra_ticks=r.i();s.positive_gain_ticks=r.i();
#define X(a) s.a=r.d();
FS_STATS_D(X)
#undef X
    return s;
}
struct State {
    std::string graph_identity,graph_input_sha256;Config config;Stats stats;
    std::vector<LocalSearchState>population;bool used_fusion=false;LocalSearchState fused_state;
    int target=0,stale=0;std::vector<Mask>certified_masks;Mask fused_certificate;
    Mask archive;Weight best_tick_max=0;long double archive_raw=0;std::string audit_rng;
    double capture_elapsed=0,capture_remaining=0,last_event=-1,pending_prefix_seconds=0,scout_seconds=0,threshold_seconds=0;
    bool prefix_normalized=false;std::uint64_t future_stream_seed=0;
    std::vector<int>outside;std::vector<Weight>unary_gains;Weight pair_gain=0;
};
inline std::string state_wire(const State&s,bool with_rng=true){Writer w;w.text("BARR_FIXED_STATE_1");w.text(s.graph_identity);w.text(s.graph_input_sha256);config_write(w,s.config);stats_write(w,s.stats);
    w.u(s.population.size());for(auto&x:s.population)local_write(w,x,with_rng);w.u(s.used_fusion);if(s.used_fusion)local_write(w,s.fused_state,with_rng);
    w.i(s.target);w.i(s.stale);w.u(s.certified_masks.size());for(auto&m:s.certified_masks)w.vec(m);w.vec(s.fused_certificate);w.vec(s.archive);w.i(s.best_tick_max);w.raw(s.archive_raw);
    if(with_rng){w.text(s.audit_rng);w.u(s.future_stream_seed);}w.d(s.capture_elapsed);w.d(s.capture_remaining);w.d(s.last_event);w.d(s.pending_prefix_seconds);w.d(s.scout_seconds);w.d(s.threshold_seconds);w.u(s.prefix_normalized);w.vec(s.outside);w.vec(s.unary_gains);w.i(s.pair_gain);return w.b;
}
inline State state_read(const std::string&b){Reader r{b};if(r.text()!="BARR_FIXED_STATE_1")throw std::runtime_error("snapshot version");State s;s.graph_identity=r.text();s.graph_input_sha256=r.text();s.config=config_read(r);s.stats=stats_read(r);
    auto n=r.length(256);for(std::size_t i=0;i<n;++i)s.population.push_back(local_read(r));s.used_fusion=bool(r.u());if(s.used_fusion)s.fused_state=local_read(r);
    s.target=int(r.i());s.stale=int(r.i());n=r.length(256);for(std::size_t i=0;i<n;++i)s.certified_masks.push_back(r.vec<std::uint8_t>());s.fused_certificate=r.vec<std::uint8_t>();s.archive=r.vec<std::uint8_t>();s.best_tick_max=r.i();s.archive_raw=r.raw();s.audit_rng=r.text();s.future_stream_seed=r.u();
    s.capture_elapsed=r.d();s.capture_remaining=r.d();s.last_event=r.d();s.pending_prefix_seconds=r.d();s.scout_seconds=r.d();s.threshold_seconds=r.d();s.prefix_normalized=bool(r.u());s.outside=r.vec<int>();s.unary_gains=r.vec<Weight>();s.pair_gain=r.i();r.end();return s;
}
inline const LocalSearchState&target_state(const State&s){return s.used_fusion?s.fused_state:s.population.at(std::size_t(s.target));}
inline Mask apply_pair(const Graph&g,const Mask&initial,const std::vector<int>&outside){Mask m=initial;for(int v:outside){if(v<0||v>=g.n||initial[v])throw std::invalid_argument("pair outside ID");for(int u:g.adj[v])m[u]=0;m[v]=1;}if(!g.feasible(m))throw std::invalid_argument("infeasible pair action");return m;}
inline void normalize_prefix(State&s){if(s.prefix_normalized)throw std::logic_error("pending prefix applied twice");s.stats.escalation_seconds+=s.pending_prefix_seconds;s.prefix_normalized=true;}
inline void validate(const Graph&g,const State&s,bool strict=true){
    if(s.graph_identity!=graph_wire(g))throw std::runtime_error("snapshot graph differs from input, including raw weights/owners/initial/edges");
    if(s.population.empty()||s.population.size()!=std::size_t(s.config.population)||s.config.threads!=1||s.target<0||s.target>=int(s.population.size())||s.stale<0)throw std::runtime_error("snapshot population/controller");
    for(auto&x:s.population){LocalSearch ls(g,g.initial,0);ls.import_state(x);}if(s.used_fusion){LocalSearch ls(g,g.initial,0);ls.import_state(s.fused_state);}rng_read(s.audit_rng);
    if(!g.feasible(s.archive)||g.raw_value(s.archive)!=s.archive_raw||s.best_tick_max<g.value(s.archive)||!s.prefix_normalized||s.pending_prefix_seconds<0||s.stats.escalation_seconds<s.pending_prefix_seconds)throw std::runtime_error("snapshot archive/prefix");
    if(s.certified_masks.size()!=s.population.size())throw std::runtime_error("snapshot cache count");for(auto&m:s.certified_masks)if(!m.empty()&&!g.feasible(m))throw std::runtime_error("snapshot cache mask");if(!s.fused_certificate.empty()&&!g.feasible(s.fused_certificate))throw std::runtime_error("snapshot fused cache");
    auto&t=target_state(s);if(s.outside.size()!=2||s.outside[0]==s.outside[1]||s.unary_gains.size()!=2)throw std::runtime_error("snapshot pair count");
    for(int j=0;j<2;++j){int v=s.outside[j];if(v<0||v>=g.n||t.s[v]||g.w[v]-t.blocked_weight[v]!=s.unary_gains[j]||(strict&&s.unary_gains[j]>=0))throw std::runtime_error("snapshot singleton proof");}
    Mask improved=apply_pair(g,t.s,s.outside);if(s.pair_gain<=0||g.value(improved)-t.cost!=s.pair_gain)throw std::runtime_error("snapshot pair proof");
}
inline std::uint64_t splitmix(std::uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
inline void rekey(State&s,std::uint64_t future_seed){if(!future_seed)return;auto root=splitmix(s.config.seed^splitmix(future_seed)^0x7ba49f0273c51de1ULL);for(std::size_t i=0;i<s.population.size();++i)s.population[i].rng=rng_text(std::mt19937_64(splitmix(root^std::uint64_t(i+1))));if(s.used_fusion)s.fused_state.rng=rng_text(std::mt19937_64(splitmix(root^0xf053dULL)));s.audit_rng=rng_text(std::mt19937_64(splitmix(root^0xa0d17ULL)));s.future_stream_seed=splitmix(root^0xf07510ULL);}
inline void solution_json(std::ostream&o,const Graph&g,const Mask&m){o<<"{\"selected\":";mask_json(o,m);o<<",\"ticks\":"<<g.value(m)<<",\"raw\":"<<std::setprecision(std::numeric_limits<long double>::max_digits10)<<g.raw_value(m)<<'}';}
inline void local_json(std::ostream&o,const LocalSearchState&x){Writer p;p.vec(x.queued);p.vec(x.blocks);p.vec(x.queue);p.vec(x.blocked_weight);p.u(x.undo.size());for(auto e:x.undo){p.i(e.first);p.u(e.second);}p.u(x.recording);p.i(x.protected_vertex);p.i(x.cost);p.u(x.iterations);p.u(x.successful);p.i(x.failures);
    o<<"{\"selected\":";mask_json(o,x.s);o<<",\"ticks\":"<<x.cost<<",\"iterations\":"<<x.iterations<<",\"successful\":"<<x.successful<<",\"failures\":"<<x.failures<<",\"private_inventory\":{\"s\":"<<x.s.size()<<",\"blocks\":"<<x.blocks.size()<<",\"blocked_weight\":"<<x.blocked_weight.size()<<",\"queue\":"<<x.queue.size()<<",\"queued\":"<<x.queued.size()<<",\"undo\":"<<x.undo.size()<<",\"recording\":"<<(x.recording?"true":"false")<<",\"protected_vertex\":"<<x.protected_vertex<<"},\"private_sha256\":";quoted(o,sha256(p.b));o<<",\"state_sha256\":";quoted(o,sha256(local_wire(x)));o<<",\"rng_state\":";quoted(o,x.rng);o<<",\"rng_sha256\":";quoted(o,sha256(x.rng));o<<'}';
}
inline void population_json(std::ostream&o,const std::vector<LocalSearchState>&p,bool internals=false){o<<'[';for(std::size_t i=0;i<p.size();++i){if(i)o<<',';if(internals)local_json(o,p[i]);else{o<<"{\"selected\":";mask_json(o,p[i].s);o<<",\"ticks\":"<<p[i].cost<<'}';}}o<<']';}
struct Diversity {std::size_t distinct_masks=0;double normalized_pairwise_hamming=0,normalized_weighted_pairwise_hamming=0;};
inline Diversity diversity(const Graph&g,const std::vector<LocalSearchState>&p){Diversity d;std::set<Mask>unique;for(auto&x:p)unique.insert(x.s);d.distinct_masks=unique.size();long double h=0,w=0;std::uint64_t pairs=0;for(std::size_t i=0;i<p.size();++i)for(std::size_t j=0;j<i;++j){++pairs;for(int v=0;v<g.n;++v)if(p[i].s[v]!=p[j].s[v]){h+=1;w+=g.w[v];}}if(pairs&&g.n)d.normalized_pairwise_hamming=double(h/(pairs*static_cast<long double>(g.n)));if(pairs&&g.total)d.normalized_weighted_pairwise_hamming=double(w/(pairs*static_cast<long double>(g.total)));return d;}
inline void diversity_json(std::ostream&o,const Diversity&d){o<<"{\"distinct_masks\":"<<d.distinct_masks<<",\"normalized_pairwise_hamming\":"<<d.normalized_pairwise_hamming<<",\"normalized_weighted_pairwise_hamming\":"<<d.normalized_weighted_pairwise_hamming<<'}';}
inline void metadata_json(std::ostream&o,const Graph&g,const State&s,const std::string&binary_hash){o<<std::setprecision(17)<<"{\"schema\":\"barr_fixed_state_meta_v1\",\"snapshot_sha256\":";quoted(o,binary_hash);o<<",\"graph_sha256\":";quoted(o,s.graph_input_sha256);o<<",\"n\":"<<g.n<<",\"m\":"<<g.edges.size()<<",\"total_ticks\":"<<g.total<<",\"config\":";config_json(o,s.config);o<<",\"population\":";population_json(o,s.population,true);o<<",\"used_fusion\":"<<(s.used_fusion?"true":"false")<<",\"feedback_target_index\":"<<s.target<<",\"target\":";local_json(o,target_state(s));o<<",\"fused_state\":";if(s.used_fusion)local_json(o,s.fused_state);else o<<"null";
    o<<",\"archive\":";solution_json(o,g,s.archive);o<<",\"best_tick_max\":"<<s.best_tick_max<<",\"controller\":{\"round\":"<<std::uint64_t(s.stats.rounds)<<",\"epoch\":"<<s.stats.gate_epochs<<",\"stale\":"<<s.stale<<",\"last_event_seconds\":"<<s.last_event<<",\"capture_elapsed_seconds\":"<<s.capture_elapsed<<",\"capture_remaining_seconds\":"<<s.capture_remaining<<",\"pending_prefix_seconds\":"<<s.pending_prefix_seconds<<",\"prefix_normalized\":true,\"escalation_seconds\":"<<s.stats.escalation_seconds<<",\"threshold_seconds\":"<<s.threshold_seconds<<",\"scout_seconds\":"<<s.scout_seconds<<",\"allstate_sha256\":";quoted(o,sha256(state_wire(s)));o<<",\"non_rng_sha256\":";quoted(o,sha256(state_wire(s,false)));o<<",\"audit_rng_state\":";quoted(o,s.audit_rng);o<<",\"audit_rng_sha256\":";quoted(o,sha256(s.audit_rng));o<<",\"certified_masks\":[";for(std::size_t i=0;i<s.certified_masks.size();++i){if(i)o<<',';mask_json(o,s.certified_masks[i]);}o<<"],\"fused_certificate\":";mask_json(o,s.fused_certificate);o<<"},\"action\":{\"outside\":["<<s.outside[0]<<','<<s.outside[1]<<"],\"unary_gains_ticks\":["<<s.unary_gains[0]<<','<<s.unary_gains[1]<<"],\"pair_gain_ticks\":"<<s.pair_gain<<",\"blockers\":[";
    for(int j=0;j<2;++j){if(j)o<<',';bool first=true;o<<'[';for(int u:g.adj[s.outside[j]])if(target_state(s).s[u]){if(!first)o<<',';first=false;o<<u;}o<<']';}o<<"],\"target\":";solution_json(o,g,target_state(s).s);o<<"},\"after_action_includes_pending_feedback\":true,\"global_optimality_claim\":false}\n";
}
struct CaptureRecord {std::string id,snapshot_path,metadata_path,snapshot_sha256;double threshold=0,elapsed=0,prefix=0,scout=0,io=0;std::uint64_t epoch=0;};
struct CaptureControl {
    std::string state_dir,graph_identity,graph_input_sha256;std::vector<double>thresholds{60,120,180};std::vector<CaptureRecord>records;
    bool due(double now,std::uint64_t epoch)const{for(auto&r:records)if(r.epoch==epoch)return false;return records.size()<thresholds.size()&&now>=thresholds[records.size()];}
    void save(const Graph&g,State s){if(!due(s.capture_elapsed,s.stats.gate_epochs))return;s.threshold_seconds=thresholds[records.size()];normalize_prefix(s);validate(g,s);auto io_started=Clock::now();auto bytes=state_wire(s);auto decoded=state_read(bytes);validate(g,decoded);if(state_wire(decoded)!=bytes)throw std::logic_error("snapshot roundtrip is not byte exact");
        std::filesystem::create_directories(state_dir);CaptureRecord r;r.id="slot_"+std::to_string(records.size())+"_epoch_"+std::to_string(s.stats.gate_epochs);r.snapshot_path=state_dir+"/"+r.id+".bin";r.metadata_path=state_dir+"/"+r.id+".meta.json";r.snapshot_sha256=sha256(bytes);r.threshold=s.threshold_seconds;r.elapsed=s.capture_elapsed;r.epoch=s.stats.gate_epochs;r.prefix=s.pending_prefix_seconds;r.scout=s.scout_seconds;
        if(std::filesystem::exists(r.snapshot_path)||std::filesystem::exists(r.metadata_path))throw std::runtime_error("refusing to overwrite capture evidence");write_bytes(r.snapshot_path,bytes);std::ofstream meta(r.metadata_path);if(!meta)throw std::runtime_error("cannot write state metadata");metadata_json(meta,g,s,r.snapshot_sha256);meta.close();if(!meta)throw std::runtime_error("metadata write failed");if(read_bytes(r.snapshot_path)!=bytes)throw std::runtime_error("snapshot disk verification failed");r.io=elapsed(io_started);records.push_back(r);
    }
};
inline std::vector<LocalSearchState> export_population(const std::vector<std::unique_ptr<LocalSearch>>&p){std::vector<LocalSearchState>r;for(auto&x:p)r.push_back(x->export_state());return r;}
struct ReplayResult {
    State ready,terminal;std::vector<LocalSearchState>after_population;Mask after_archive,after_target;
    Diversity before_diversity,after_diversity,terminal_diversity;std::string ready_hash,ready_non_rng_hash;
    double native_seconds=0,cpu_seconds=0,action_seconds=0,pair_apply_seconds=0,feedback_seconds=0,continuation_seconds=0;
    std::uint64_t local_iterations=0,fusions=0,rounds=0;double future_escalation_seconds=0;bool feedback_applied=false;
};
// All arms call this same continuation. The known pair is the only branch.
inline ReplayResult replay(const Graph&g,State source,const std::string&action,double seconds,std::uint64_t future_seed,int future_round_limit=-1){
    if(action!="recover"&&action!="continue")throw std::invalid_argument("action recover|continue required");if(!std::isfinite(seconds)||seconds<0)throw std::invalid_argument("invalid future budget");validate(g,source);const auto original_non_rng=sha256(state_wire(source,false));rekey(source,future_seed);if(sha256(state_wire(source,false))!=original_non_rng)throw std::logic_error("rekey changed non RNG state");
    ReplayResult r;r.ready=source;r.ready_hash=sha256(state_wire(source));r.ready_non_rng_hash=sha256(state_wire(source,false));
    std::vector<std::unique_ptr<LocalSearch>>pop;for(auto&x:source.population){auto p=std::make_unique<LocalSearch>(g,g.initial,0);p->import_state(x);if(local_wire(p->export_state())!=local_wire(x))throw std::logic_error("local restore mismatch");pop.push_back(std::move(p));}
    std::unique_ptr<LocalSearch>pending;LocalSearch*state=pop[source.target].get();if(source.used_fusion){pending=std::make_unique<LocalSearch>(g,g.initial,0);pending->import_state(source.fused_state);state=pending.get();}
    Stats st=source.stats;Mask archive=source.archive;Weight best=source.best_tick_max;long double raw=source.archive_raw;int stale=source.stale;double last_event=source.last_event;const Config&c=source.config;
    auto remember=[&](const Mask&s){best=std::max(best,g.value(s));auto x=g.raw_value(s);if(x>raw){if(!g.feasible(s))throw std::logic_error("invalid replay archive");archive=s;raw=x;}};
    // Exact transport validation and stream preparation end before this point.
    auto started=Clock::now();auto cpu_start=std::clock();auto deadline=Deadline::after(seconds);auto initial_started=Clock::now();
    r.before_diversity=diversity(g,source.population);
    if(action=="recover"){
        auto a=Clock::now();Mask improved=apply_pair(g,state->s,source.outside);if(g.value(improved)-state->cost!=source.pair_gain)throw std::logic_error("replay pair gain mismatch");remember(improved);state->reset(improved);stale=0;++st.witness_openings;++st.witness_moves;++st.accepted_recoveries;st.positive_gain_ticks+=source.pair_gain;r.pair_apply_seconds=elapsed(a);
    }
    auto fb=Clock::now();if(source.used_fusion&&state->cost>=pop[source.target]->cost){pop[source.target]->reset(state->s);r.feedback_applied=true;}r.feedback_seconds=elapsed(fb);r.after_target=state->s;r.after_population=export_population(pop);r.after_archive=archive;r.after_diversity=diversity(g,r.after_population);
    r.action_seconds=elapsed(initial_started);st.escalation_seconds+=r.action_seconds;auto continue_started=Clock::now();
    while(!deadline.expired()&&(future_round_limit<0||r.rounds<std::uint64_t(future_round_limit))){
        ++r.rounds;++st.rounds;Weight before=best;auto b=Clock::now();for(auto&p:pop){auto previous=p->iterations;p->run(c.local_iterations,deadline.slice(c.local_seconds));r.local_iterations+=p->iterations-previous;remember(p->s);}st.local_seconds+=elapsed(b);stale=best>before?0:stale+1;
        double now=source.capture_elapsed+elapsed(started);if(deadline.expired())break;
        if(now<c.gate_warmup||stale<c.event_stale||now-last_event<c.gate_cooldown)continue;double credit=c.gate_fraction*now-st.escalation_seconds;if(credit<.01){++st.gate_budget_skips;continue;}last_event=now;++st.gate_epochs;auto epoch=st.gate_epochs;
        int elite=0;for(int i=1;i<int(pop.size());++i)if(pop[i]->cost>pop[elite]->cost)elite=i;int target=epoch%2?elite:int((epoch/2)%pop.size());b=Clock::now();auto ed=deadline.slice(std::min(credit,c.pair_slice));
        try{if(c.pair_policy=="fusion-refine"&&epoch%std::uint64_t(c.pair_fusion_every)==0&&pop.size()>1&&ed.remaining()>.04){int far=elite;Weight distance=-1;for(int i=0;i<int(pop.size());++i)if(i!=elite){Weight z=0;for(int v=0;v<g.n;++v)if(pop[elite]->s[v]!=pop[i]->s[v])z+=g.w[v];if(z>distance){far=i;distance=z;}}
                if(distance>0){auto f=Clock::now();auto back=fuse(g,pop[elite]->s,pop[far]->s,ed.slice(c.fusion_seconds));st.fusion_seconds+=elapsed(f);++st.fusions;++r.fusions;st.fusion_exact+=back.exact;st.cut_calls+=back.cuts;remember(back.merged);LocalSearch merged(g,back.merged,source.future_stream_seed^epoch);try{merged.descent(ed.slice(c.local_seconds));}catch(const Timeout&){}remember(merged.s);target=elite;for(int i=0;i<int(pop.size());++i)if(pop[i]->cost<pop[target]->cost)target=i;if(merged.cost>=pop[target]->cost)pop[target]->reset(merged.s);}
            }}catch(const Timeout&){++st.gate_timeouts;}st.escalation_seconds+=elapsed(b);
    }
    for(auto&p:pop)remember(p->s);r.continuation_seconds=elapsed(continue_started);r.terminal=source;r.terminal.population=export_population(pop);r.terminal.archive=archive;r.terminal.archive_raw=raw;r.terminal.best_tick_max=best;r.terminal.stale=stale;r.terminal.last_event=last_event;r.terminal.stats=st;r.terminal_diversity=diversity(g,r.terminal.population);
    if(!g.feasible(archive)||raw<source.archive_raw)throw std::logic_error("replay archive regression");r.future_escalation_seconds=st.escalation_seconds-source.stats.escalation_seconds;r.native_seconds=elapsed(started);r.cpu_seconds=double(std::clock()-cpu_start)/CLOCKS_PER_SEC;return r;
}
inline void replay_json(std::ostream&o,const Graph&g,const ReplayResult&r,const std::string&action,double seconds,std::uint64_t future_seed,const std::string&snapshot_hash){o<<std::setprecision(17)<<"{\"schema\":\"barr_fixed_replay_v1\",\"action\":";quoted(o,action);o<<",\"future_seed\":"<<future_seed<<",\"budget_seconds\":"<<seconds<<",\"snapshot_sha256\":";quoted(o,snapshot_hash);o<<",\"graph_sha256\":";quoted(o,r.ready.graph_input_sha256);o<<",\"config\":";config_json(o,r.ready.config);o<<",\"ready_state_sha256\":";quoted(o,r.ready_hash);o<<",\"ready_non_rng_sha256\":";quoted(o,r.ready_non_rng_hash);o<<",\"roundtrip_equal\":true,\"io_paths_restored\":false,\"rng_plan\":\"seed0 exact captured streams and future fused seed=captured_seed XOR epoch; nonzero splitmix_v1 rekeys every existing member/fused/audit plus future fused seed; both arms identical before action\",\"after_action_includes_pending_feedback\":true,\"feedback_target_index\":"<<r.ready.target<<",\"feedback_applied\":"<<(r.feedback_applied?"true":"false")<<",\"archive_before\":";solution_json(o,g,r.ready.archive);o<<",\"archive_after_action\":";solution_json(o,g,r.after_archive);o<<",\"archive_terminal\":";solution_json(o,g,r.terminal.archive);o<<",\"best_tick_max_before\":"<<r.ready.best_tick_max<<",\"best_tick_max_terminal\":"<<r.terminal.best_tick_max;
    o<<",\"population_before\":";population_json(o,r.ready.population);o<<",\"population_after_action\":";population_json(o,r.after_population);o<<",\"population_terminal\":";population_json(o,r.terminal.population);o<<",\"target_after_action\":";solution_json(o,g,r.after_target);o<<",\"diversity_before\":";diversity_json(o,r.before_diversity);o<<",\"diversity_after_action\":";diversity_json(o,r.after_diversity);o<<",\"diversity_terminal\":";diversity_json(o,r.terminal_diversity);
    o<<",\"action_seconds\":"<<r.action_seconds<<",\"pair_apply_seconds\":"<<r.pair_apply_seconds<<",\"feedback_seconds\":"<<r.feedback_seconds<<",\"continuation_seconds\":"<<r.continuation_seconds<<",\"native_seconds\":"<<r.native_seconds<<",\"cpu_seconds\":"<<r.cpu_seconds<<",\"capture_elapsed_seconds\":"<<r.ready.capture_elapsed<<",\"capture_remaining_seconds\":"<<r.ready.capture_remaining<<",\"capture_sunk_scout_seconds\":"<<r.ready.scout_seconds<<",\"capture_sunk_prefix_seconds\":"<<r.ready.pending_prefix_seconds<<",\"future_stats\":{\"rounds\":"<<r.rounds<<",\"local_iterations\":"<<r.local_iterations<<",\"fusions\":"<<r.fusions<<",\"joint_scout_calls\":0,\"kernel_calls\":0,\"joint_recovery_calls\":0,\"escalation_seconds\":"<<r.future_escalation_seconds<<"},\"global_optimality_claim\":false,\"diagnostic_only\":true}\n";
}
} }
