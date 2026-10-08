#include "barr/fixed_capture_impl.hpp"
#include "barr/state_cli.hpp"
#include <iostream>
using namespace barr;
using namespace barr::fixed;
static void require(bool b,const char*s){if(!b)throw std::runtime_error(s);}
template<class F>static void rejects(F f,const char*s){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,s);}
static Graph fixture_graph(){Graph g({10,10,9,12,100},{{0,2},{0,3},{1,3}});g.initial={1,1,0,0,1};return g;}
static State fixture(const Graph&g,bool fused){
    State s;s.graph_identity=graph_wire(g);s.graph_input_sha256=sha256("synthetic input fixture");s.config=frozen_config();s.config.local_iterations=3;s.config.event_stale=1;s.config.pair_fusion_every=1;s.config.gate_warmup=0;s.config.gate_cooldown=.1;s.config.gate_fraction=.25;
    for(int i=0;i<4;++i){LocalSearch l(g,g.initial,101+i);auto x=l.export_state();
        // Valid nonempty, nontrivial and ordered queue; this must not be replaced
        // by reset(mask), which would silently enqueue every graph vertex.
        x.queue={4,1,0};x.queued={1,1,0,0,1};x.iterations=12+i;x.successful=2;x.failures=i+1;s.population.push_back(x);}
    s.target=0;s.used_fusion=fused;if(fused){s.fused_state=s.population[1];Mask weak=g.initial;weak[1]=0;LocalSearch l(g,weak,998);s.population[0]=l.export_state();}
    s.certified_masks.resize(4);s.certified_masks[2]=g.initial;s.fused_certificate=g.initial;s.archive=g.initial;s.archive_raw=g.raw_value(s.archive);s.best_tick_max=g.value(s.archive);s.audit_rng=rng_text(std::mt19937_64(882));s.capture_elapsed=180;s.capture_remaining=180;s.last_event=179.9;s.pending_prefix_seconds=.02;s.scout_seconds=.01;s.threshold_seconds=60;s.stale=8;s.future_stream_seed=s.config.seed;s.stats.gate_epochs=7;s.stats.escalation_seconds=2;s.outside={2,3};s.unary_gains={-1,-8};s.pair_gain=1;normalize_prefix(s);return s;
}
static void transport_properties(){
    require(sha256("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA empty");
    require(sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA abc");
    Graph g=fixture_graph();State s=fixture(g,true);validate(g,s);auto bytes=state_wire(s);auto restored=state_read(bytes);require(bytes==state_wire(restored),"all state byte roundtrip");validate(g,restored);
    require(restored.population[0].queue.size()>0,"queue missing");require(restored.population[1].queued[2]==0&&restored.population[1].queued[4],"nontrivial queued fixture");
    for(std::size_t i=0;i<s.population.size();++i){LocalSearch a(g,g.initial,0);a.import_state(restored.population[i]);require(local_wire(a.export_state())==local_wire(s.population[i]),"private state/RNG restore");
        auto before=a.export_state();LocalSearch b(g,g.initial,999);b.import_state(before);a.run(30,Deadline::after(10));b.run(30,Deadline::after(10));require(local_wire(a.export_state())==local_wire(b.export_state()),"exact queue/RNG continuation");}
    auto no_rng=state_wire(s,false);auto seed0=s;rekey(seed0,0);require(state_wire(seed0)==bytes,"seed0 is not exact");
    for(auto seed:{1901ULL,1907ULL,1913ULL,1931ULL}){auto a=s,b=s;rekey(a,seed);rekey(b,seed);require(state_wire(a)==state_wire(b),"paired RNG rekey differs");require(state_wire(a)!=bytes,"future RNG did not change");require(state_wire(a,false)==no_rng,"rekey altered structural state");require(a.future_stream_seed!=s.future_stream_seed,"future fused constructor seed not rekeyed");}
    auto twice=s;rejects([&]{normalize_prefix(twice);},"prefix credited twice");
    require(std::abs(s.stats.escalation_seconds-2.02)<1e-12,"prefix not normalized once");
    auto bad=s.population[1];bad.recording=true;LocalSearch ls(g,g.initial,0);rejects([&]{ls.import_state(bad);},"unsafe recording accepted");bad=s.population[1];bad.undo={{1,true}};rejects([&]{ls.import_state(bad);},"unsafe undo accepted");bad=s.population[1];bad.protected_vertex=0;rejects([&]{ls.import_state(bad);},"unsafe protected accepted");bad=s.population[1];bad.queue.push_back(bad.queue[0]);rejects([&]{ls.import_state(bad);},"duplicate queue accepted");
    Graph mismatch=g;mismatch.original[0]+=.0001;rejects([&]{validate(mismatch,s);},"raw graph mismatch accepted");mismatch=g;mismatch.owners[0]=8;rejects([&]{validate(mismatch,s);},"owner graph mismatch accepted");
    auto corrupt=bytes+"x";rejects([&]{state_read(corrupt);},"wire suffix accepted");
    Writer w;long double raw=static_cast<long double>(.1)+static_cast<long double>(1e19);w.raw(raw);Reader r{w.b};require(r.raw()==raw,"long double roundtrip");r.end();
    // Populate every Stats field with nonzero distinguishable values.
    Stats st;std::uint64_t k=5;
#define X(a) st.a=++k;
    FS_STATS_U(X)
#undef X
    double d=.125;
#define X(a) st.a=(d+=.125);
    FS_STATS_D(X)
#undef X
    st.refinement_extra_ticks=77;st.positive_gain_ticks=99;Writer sw;stats_write(sw,st);Reader sr{sw.b};auto st2=stats_read(sr);Writer tw;stats_write(tw,st2);require(sw.b==tw.b,"all stats fields roundtrip");
}
static void intervention_properties(){
    Graph g=fixture_graph();
    for(bool fused:{false,true})for(auto seed:{0ULL,1901ULL,1907ULL,1913ULL,1931ULL}){
        auto s=fixture(g,fused);FixedTestClock::reset();auto control=replay(g,s,"continue",360,seed,0);
        FixedTestClock::reset();auto treatment=replay(g,s,"recover",360,seed,0);
        require(control.ready_hash==treatment.ready_hash,"arms start from different complete states");
        require(control.ready_non_rng_hash==sha256(state_wire(s,false)),"ready structural digest differs");
        require(control.after_archive==s.archive,"continue changed archive before future");
        Mask improved=apply_pair(g,target_state(s).s,s.outside);require(g.value(improved)==target_state(s).cost+1,"known pair gain");
        require(treatment.after_archive==improved,"known pair not archived");
        require(treatment.after_population[0].s==improved,"known pair feedback missing");
        require(control.after_population[0].s==(fused?s.fused_state.s:s.population[0].s),"continue fused feedback missing");
        require(control.feedback_applied==fused&&treatment.feedback_applied==fused,"pending feedback stage wrong");
        require(control.terminal.stats.escalation_seconds>=s.stats.escalation_seconds,"lost prefix credit");
        // Same-clock/same-RNG/no-op continuation is identical including private
        // queue order, all counters, controller credit, cache and RNG.
        FixedTestClock::reset();auto a=replay(g,s,"continue",360,seed,80);
        FixedTestClock::reset();auto b=replay(g,state_read(state_wire(s)),"continue",360,seed,80);
        require(state_wire(a.terminal)==state_wire(b.terminal),"no-op restored continuation differs");
        require(a.terminal.archive==b.terminal.archive&&a.local_iterations==b.local_iterations&&a.fusions==b.fusions,"no-op output differs");
        require(a.terminal.stats.recoveries==s.stats.recoveries&&a.terminal.stats.kernels_materialized==s.stats.kernels_materialized,"future joint work occurred");
    }
    // Archive raw ordering is separate from maximum integer bookkeeping.
    auto s=fixture(g,false);s.best_tick_max=g.value(s.archive)+30;validate(g,s);FixedTestClock::reset();auto a=replay(g,s,"recover",360,0,0);require(a.terminal.best_tick_max==s.best_tick_max,"integer maximum overwritten by raw archive");
    // A genuinely superior historical archive need not remain in the current
    // population. A positive pair must survive, but can add zero archive value.
    auto history=fixture(g,false);Mask weak=g.initial;weak[4]=0;
    for(std::size_t i=0;i<history.population.size();++i){LocalSearch p(g,weak,99+i);history.population[i]=p.export_state();}
    validate(g,history);FixedTestClock::reset();auto h=replay(g,history,"recover",360,0,0);
    require(h.after_archive==history.archive&&g.value(h.after_target)>g.value(weak),"positive action below historical archive mishandled");
    // General raw ordering remains correct even if tick and original objectives
    // disagree. No tick pruning may discard the raw incumbent.
    auto raw_graph=g;raw_graph.original[2]=.1;raw_graph.original[3]=.1;auto mismatch=fixture(raw_graph,false);
    validate(raw_graph,mismatch);FixedTestClock::reset();auto m=replay(raw_graph,mismatch,"recover",360,0,0);
    require(m.after_archive==mismatch.archive&&m.terminal.best_tick_max>mismatch.best_tick_max,"raw/tick archive semantics lost");
    auto cfg=frozen_config();require(cfg.gate_cooldown==.5&&cfg.gate_warmup==30&&cfg.population==4&&cfg.pair_component=="full","frozen E settings mismatch");
}
static void threshold_properties(){
    CaptureControl c;require(!c.due(59.9,1)&&c.due(180,1),"threshold eligibility");CaptureRecord a;a.epoch=1;c.records.push_back(a);require(!c.due(180,1)&&c.due(180,2),"one snapshot per epoch");a.epoch=2;c.records.push_back(a);require(!c.due(180,2)&&c.due(180,3),"eligible slots not sequential");a.epoch=3;c.records.push_back(a);require(!c.due(999,4),"more than3 snapshots");
    // With no eligible capture slot, instrumented full search follows the
    // original E policy exactly under the private deterministic test clock.
    auto g=fixture_graph();auto config=frozen_config();config.max_rounds=20;config.seconds=30;config.local_iterations=3;config.gate_warmup=0;config.event_stale=1;config.gate_fraction=.25;config.pair_fusion_every=1;config.gate_cooldown=.1;
    FixedTestClock::reset();auto old=solve_pair(g,config);CaptureControl disabled;disabled.thresholds={1000};disabled.graph_identity=graph_wire(g);disabled.graph_input_sha256=sha256("fixture");
    FixedTestClock::reset();auto current=solve_pair_capture(g,config,disabled);Writer before_writer,after_writer;stats_write(before_writer,old.stats);stats_write(after_writer,current.stats);require(old.selected==current.selected&&before_writer.b==after_writer.b,"capture observer changed uncaptured E policy");
}
static void random_transport(){
    std::mt19937_64 rng(123);for(int trial=0;trial<150;++trial){int n=1+int(rng()%18);std::vector<Weight>w(n);for(auto&x:w)x=1+rng()%1000;std::vector<std::pair<int,int>>e;for(int i=0;i<n;++i)for(int j=0;j<i;++j)if(rng()%4==0)e.emplace_back(i,j);Graph g(w,e);LocalSearch a(g,g.initial,rng());a.greedy_start(.5,.25,Deadline::after(10));a.run(1+int(rng()%20),Deadline::after(10));auto x=a.export_state();Writer wire;local_write(wire,x);Reader r{wire.b};auto y=local_read(r);r.end();LocalSearch b(g,g.initial,0);b.import_state(y);require(local_wire(x)==local_wire(b.export_state()),"random full private roundtrip");a.run(20,Deadline::after(10));b.run(20,Deadline::after(10));require(local_wire(a.export_state())==local_wire(b.export_state()),"random restored trajectory differs");}
}
static void emit_fixture(const std::string&dir){
    if(std::filesystem::exists(dir))throw std::runtime_error("fixture directory must be new");std::filesystem::create_directories(dir);
    auto g=fixture_graph();std::ostringstream input;input<<"BARR1 "<<g.n<<' '<<g.edges.size()<<'\n';
    for(int i=0;i<g.n;++i)input<<g.w[i]<<' '<<std::setprecision(17)<<g.original[i]<<" -1\n";
    for(auto e:g.edges)input<<e.first<<' '<<e.second<<'\n';auto selected=members(g.initial);input<<selected.size();for(int v:selected)input<<' '<<v;input<<'\n';
    write_bytes(dir+"/fixture.barr",input.str());auto s=fixture(g,true);s.config=frozen_config();s.graph_input_sha256=sha256(input.str());
    auto bytes=state_wire(s);write_bytes(dir+"/fixture.bin",bytes);std::ofstream meta(dir+"/fixture.meta.json");metadata_json(meta,g,s,sha256(bytes));meta.close();
    for(auto action:{"continue","recover"}){FixedTestClock::reset();auto r=replay(g,s,action,360,0);std::ofstream o(dir+"/"+action+".json");replay_json(o,g,r,action,360,0,sha256(bytes));}
    CaptureControl c;c.graph_input_sha256=s.graph_input_sha256;CaptureRecord record;record.id="fixture";record.snapshot_path=dir+"/fixture.bin";record.metadata_path=dir+"/fixture.meta.json";record.snapshot_sha256=sha256(bytes);record.threshold=60;record.elapsed=180;record.epoch=s.stats.gate_epochs;record.prefix=s.pending_prefix_seconds;record.scout=s.scout_seconds;c.records.push_back(record);
    SolverResult result;result.selected=s.archive;result.stats=s.stats;result.seconds=360;std::ofstream capture(dir+"/fixture.capture.json");capture_json(capture,g,s.config,result,c);capture.close();
    std::ofstream manifest(dir+"/fixture_manifest.json");manifest<<"{\"schema\":\"barr_fixed_state_fixture_v1\",\"synthetic_test_clock\":true,\"performance_evidence\":false,\"capture\":\"fixture.capture.json\",\"graph\":\"fixture.barr\",\"snapshot\":\"fixture.bin\",\"metadata\":\"fixture.meta.json\",\"arms\":[\"continue.json\",\"recover.json\"]}\n";
}
int main(int argc,char**argv){try{FixedTestClock::reset();transport_properties();intervention_properties();threshold_properties();random_transport();if(argc==3&&std::string(argv[1])=="--emit-fixture")emit_fixture(argv[2]);std::cout<<"{\"schema\":\"barr_fixed_state_selftest_v1\",\"passed\":true,\"checks\":[\"sha256_standard_vectors\",\"all_private_state_roundtrip\",\"rng_seed0_exact\",\"paired_rekey_nonrng_invariant\",\"prefix_exactly_once\",\"pending_fused_feedback_both_arms\",\"known_pair_only\",\"no_op_fixed_clock_equivalence\",\"uncaptured_E_policy_equivalence\",\"150_random_transport_trajectories\"],\"performance_results\":false}\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
