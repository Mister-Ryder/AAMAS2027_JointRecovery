#include "barr/solver.hpp"
#include <iostream>
#include <regex>
#include <sstream>
namespace barr {
#include "b_frozen_pair_reference_impl.hpp"
}
using namespace barr;
#ifndef BARR_TEST_FIXED_CLOCK
#error This unit target must privately enable the fixed-round test clock.
#endif
static std::uint64_t checks=0,cases=0;
static void require(bool ok,const char*message){++checks;if(!ok)throw std::runtime_error(message);}

static std::vector<std::string> trace(const std::filesystem::path&path){
    std::ifstream input(path);std::vector<std::string>lines;std::string line;
    const std::regex timing(",\\\"(elapsed|seconds|fusion_seconds|gate_seconds)\\\":[-+0-9.eE]+");
    while(std::getline(input,line))lines.push_back(std::regex_replace(line,timing,""));
    return lines;
}

static void same_discrete(const SolverResult&a,const SolverResult&b){
    require(a.selected==b.selected,"fixed-round E full and frozen B archive memberships equal");
#define SAME(field) require(a.stats.field==b.stats.field,"fixed-round discrete field differs: " #field)
    SAME(rounds);SAME(local_iterations);SAME(fusions);SAME(fusion_exact);SAME(recoveries);SAME(recovery_exact);
    SAME(accepted_recoveries);SAME(outsiders_inserted);SAME(consensus_released);SAME(branch_nodes);SAME(cut_calls);SAME(cache_hits);
    SAME(model_batches);SAME(model_fallback_batches);SAME(restarts);SAME(gate_epochs);SAME(raw_candidates);SAME(sketches);
    SAME(safe_rejects);SAME(neighbor_visits);SAME(heuristic_skips);SAME(exploration_openings);SAME(witness_openings);
    SAME(kernels_materialized);SAME(gate_budget_skips);SAME(gate_timeouts);SAME(gate_oversized);SAME(witness_moves);
    SAME(pair_visits);SAME(pair_unique);SAME(pair_upper_rejects);SAME(pair_conflict_rejects);SAME(pair_exact_checks);
    SAME(pair_positive_candidates);SAME(pair_complete);SAME(pair_certificates);SAME(pair_cached_zero);
    SAME(refinement_extra_ticks);SAME(positive_gain_ticks);SAME(factor_attempts);SAME(factor_exact);SAME(factor_fallbacks);
    SAME(factor_compiled_entries);SAME(factor_dp_entries);
#undef SAME
}

static Config config(){
    Config c;c.mode="pair";c.pair_policy="fusion-refine";c.seconds=1000.;c.max_rounds=64;
    c.population=1;c.local_seconds=0.;c.local_iterations=1;c.gate_warmup=.2;c.gate_cooldown=.001;
    c.gate_fraction=.25;c.event_stale=1;c.pair_slice=.5;c.pair_fusion_every=1;return c;
}

static SolverResult equivalent(const Graph&g,Config c,const std::filesystem::path&dir,const std::string&name){
    c.sketch_snapshots.clear();c.events=(dir/(name+"_B.jsonl")).string();Clock::reset();
    auto reference=solve_pair_frozen_b(g,c);
    c.events=(dir/(name+"_E.jsonl")).string();Clock::reset();auto full=solve(g,c);
    require(c.pair_component=="full","default component is full");
    require(std::uint64_t(full.stats.rounds)==64&&std::uint64_t(reference.stats.rounds)==64,"both implementations finish fixed 64 rounds");
    require(full.stats.gate_epochs>0,"equivalence exercises post-warmup gates, not only initialization");
    same_discrete(reference,full);
    require(trace(dir/(name+"_B.jsonl"))==trace(dir/(name+"_E.jsonl")),"non-time event traces equal frozen B");
    require(g.feasible(full.selected)&&g.raw_value(full.selected)>=g.raw_value(g.initial),"full archive feasible and raw nondecreasing");
    ++cases;return full;
}

static void no_joint(const SolverResult&r){
    require(r.stats.gate_epochs>0,"pulse-only uses the common post-warmup event policy");
    require(r.stats.raw_candidates==0&&r.stats.sketches==0&&r.stats.neighbor_visits==0,"pulse-only performs no outsider scan or sketch");
    require(r.stats.pair_visits==0&&r.stats.pair_unique==0&&r.stats.pair_exact_checks==0&&r.stats.pair_positive_candidates==0,"pulse-only performs no pair scout");
    require(r.stats.pair_complete==0&&r.stats.pair_certificates==0&&r.stats.pair_cached_zero==0&&r.stats.safe_rejects==0,"pulse-only makes and reuses no zero opportunity certificate");
    require(r.stats.kernels_materialized==0&&r.stats.recoveries==0&&r.stats.accepted_recoveries==0&&r.stats.branch_nodes==0,"pulse-only opens no Kernel or joint solver/commit");
    require(r.stats.witness_openings==0&&r.stats.witness_moves==0&&r.stats.positive_gain_ticks==0&&r.stats.refinement_extra_ticks==0,"pulse-only records no joint witness gain");
    require(r.stats.gate_seconds==0&&r.stats.build_seconds==0&&r.stats.recovery_seconds==0,"pulse-only has no scout/build/recovery duration");
}

int main(){try{
    const auto unique=std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto directory=std::filesystem::current_path()/("pulse_ablation_test_artifacts_"+std::to_string(unique));
    std::filesystem::create_directories(directory);
    // Reading the private clock is side-effect free; only round++ moves it.
    Clock::reset();auto first=Clock::now();require(Clock::now()==first&&elapsed(first)==0.,"test clock reads do not advance time");
    {Graph g({10,4,4},{{0,1},{0,2}});g.initial={1,0,0};auto c=config();c.local_seconds=.025;c.local_iterations=4;
        auto r=equivalent(g,c,directory,"zero");
        require(r.stats.pair_certificates>0&&r.stats.accepted_recoveries==0,"equivalence includes actual full zero certificates");
        require(r.stats.local_iterations>0,"equivalence includes actual local iterations");}
    // Original full graph: a has one selected blocker, b has the common and a
    // private selected blocker. Both singleton gains are negative, pair is +1.
    Graph joint({10,3,9,5},{{0,2},{0,3},{1,3}});joint.initial={1,1,0,0};
    auto c=config();auto full=equivalent(joint,c,directory,"positive");
    require(full.stats.accepted_recoveries>0&&full.stats.positive_gain_ticks>=1&&joint.value(full.selected)==14,"full path commits the two-negative-singleton positive pair");
    {LocalSearch state(joint,joint.initial,17);auto pair=pair_scout(joint,state,Deadline::after(10));
        require(pair.outside==std::vector<int>({2,3})&&pair.witness_gain==1,"independent original graph pair witness found");
        require(joint.w[2]-state.blocked_weight[2]==-1&&joint.w[3]-state.blocked_weight[3]==-8,"both original-graph singleton gains strictly negative");
        Mask lifted=joint.initial;for(int v:pair.outside){for(int u:joint.adj[v])lifted[u]=0;lifted[v]=1;}
        require(joint.feasible(lifted)&&joint.value(lifted)-joint.value(joint.initial)==1,"positive pair validated on complete original graph");}
    {auto pulse=c;pulse.pair_component="pulse-only";pulse.sketch_snapshots=(directory/"pulse_no_snapshots").string();pulse.events=(directory/"pulse_no_joint.jsonl").string();
        Clock::reset();auto r=solve(joint,pulse);no_joint(r);
        require(r.selected==joint.initial,"single-population pulse-only does not silently commit the positive pair");
        require(!std::filesystem::exists(pulse.sketch_snapshots),"pulse-only emits no pair snapshot even when snapshot path supplied");
        auto events=trace(pulse.events);require(!events.empty(),"pulse-only emits event-policy log");
        for(auto&line:events)require(line.find("pulse_ablation_gate")!=std::string::npos&&line.find("pair_gate")==std::string::npos&&line.find("certificate")==std::string::npos,"pulse-only logs no pair gate or certificate");}
    {std::vector<std::pair<int,int>>edges;for(int u=0;u<4;++u)for(int v=4;v<7;++v)edges.emplace_back(u,v);
        Graph g({3,3,3,3,5,5,5},edges);g.initial={1,1,1,1,0,0,0};auto f=config();f.population=4;
        auto r=equivalent(g,f,directory,"fusion");require(r.stats.fusions>0&&r.stats.pair_certificates>0,"equivalence includes actual fusion pulse and full scout zero events");
        f.pair_component="pulse-only";f.events=(directory/"pulse_fusion.jsonl").string();Clock::reset();auto pulse=solve(g,f);no_joint(pulse);
        require(pulse.stats.fusions>0&&g.feasible(pulse.selected)&&g.value(pulse.selected)==15,"pulse-only preserves feasible fusion/descent archive");
        // With local steps disabled, diversity remains unless the first fused
        // mask is fed into the sole weak member. Correct feedback makes every
        // parent identical, so later eligible pulse epochs do not fuse again.
        require(pulse.stats.gate_epochs>2&&pulse.stats.fusions==1&&r.stats.fusions==1,"pulse-only and full feed back the fused mask instead of repeatedly fusing an unchanged weak member");}
    { // Snapshot-only RNG and I/O cannot change search choices in fixed rounds.
        auto sampled=c;sampled.events=(directory/"sampled.jsonl").string();sampled.sketch_snapshots=(directory/"full_snapshots").string();
        Clock::reset();auto r=solve(joint,sampled);same_discrete(full,r);
        require(trace(directory/"positive_E.jsonl")==trace(sampled.events),"diagnostic RNG/snapshots leave non-time search event trace unchanged");
        std::size_t positive_snapshots=0;std::vector<std::string>names;
        for(const auto&entry:std::filesystem::directory_iterator(sampled.sketch_snapshots))if(entry.is_regular_file()){
            std::ifstream file(entry.path());std::string text((std::istreambuf_iterator<char>(file)),{});names.push_back(entry.path().filename().string());
            require(text.find("barr_scout_snapshot_v1")!=std::string::npos&&text.find("\"scout\":\"full\"")!=std::string::npos,"E snapshots use C-compatible full schema");
            require(text.find("\"incumbent\":")!=std::string::npos&&text.find("\"outside\":")!=std::string::npos&&text.find("\"positive_singletons\":")!=std::string::npos,"snapshots freeze original incumbent/outside and singleton flags");
            if(text.find("\"gain_ticks\":1")!=std::string::npos)++positive_snapshots;
        }
        std::sort(names.begin(),names.end());require(std::adjacent_find(names.begin(),names.end())==names.end()&&positive_snapshots==1,"positive uniform hit is not duplicated into enrichment");}
    {Graph g({2,3},{{0,1}},{100.,1.});g.initial={1,0};auto raw=config();
        auto full_raw=equivalent(g,raw,directory,"raw");require(full_raw.selected==g.initial,"fixed-round full preserves raw archive against tick disagreement");
        raw.pair_component="pulse-only";Clock::reset();auto r=solve(g,raw);require(r.selected==g.initial&&g.raw_value(r.selected)==100.,"pulse-only preserves original raw-value archive");}
    {auto invalid=config();invalid.pair_component="invalid";bool caught=false;
        try{solve(joint,invalid);}catch(const std::invalid_argument&){caught=true;}require(caught,"invalid component rejected");
        invalid.pair_component="pulse-only";invalid.pair_policy="direct";caught=false;
        try{solve(joint,invalid);}catch(const std::invalid_argument&){caught=true;}require(caught,"pulse-only requires common fusion-refine policy");}
    std::ostringstream json;result_json(json,joint,c,full);
    require(json.str().find("\"pair_component\":\"full\"")!=std::string::npos,"native result records ablation component");
    std::cout<<"{\"status\":\"PASS\",\"fixed_round_equivalence_cases\":"<<cases<<",\"rounds_per_case\":64,\"post_warmup_positive_zero_fusion_exercised\":true,\"assertions\":"<<checks
             <<",\"clock_reads_idempotent\":true,\"production_time_semantics_tested\":false,\"artifacts\":\""<<directory.filename().string()<<"\"}\n";
    return 0;
}catch(const std::exception&e){std::cerr<<"pulse ablation test FAIL: "<<e.what()<<'\n';return 1;}}
