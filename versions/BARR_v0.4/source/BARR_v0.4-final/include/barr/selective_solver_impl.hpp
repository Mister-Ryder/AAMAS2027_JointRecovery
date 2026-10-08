#pragma once
// Included inside namespace barr after Config, Stats and SolverResult.
inline SolverResult solve_selective(const Graph&g,const Config&c) {
    if(c.gate!="value"&&c.gate!="random"&&c.gate!="witness")throw std::invalid_argument("unknown selective gate");
    if(c.gate_fraction<=0||c.gate_fraction>.5||c.gate_cooldown<=0||c.gate_trials<1||c.gate_trials>128||c.event_stale<1||c.explore_every<1)
        throw std::invalid_argument("invalid selective gate configuration");
    auto started=Clock::now();auto cpu_start=std::clock();Deadline deadline=Deadline::after(c.seconds);
    SolverResult result;result.selected=g.initial;Stats&st=result.stats;Weight best=g.value(g.initial);long double bestraw=g.raw_value(g.initial);
    std::ofstream events;if(!c.events.empty()){events.open(c.events);if(!events)throw std::runtime_error("cannot open events");events<<std::setprecision(17);}
    auto remember=[&](const Mask&s){Weight value=g.value(s);best=std::max(best,value);auto raw=g.raw_value(s);if(raw>bestraw){if(!g.feasible(s))throw std::logic_error("infeasible selective incumbent");result.selected=s;bestraw=raw;write_checkpoint(c.checkpoint,g,s);}};
    write_checkpoint(c.checkpoint,g,result.selected);
    if(g.n==0){result.seconds=elapsed(started);return result;}
    std::vector<std::unique_ptr<LocalSearch>>pop;
    for(int i=0;i<c.population;++i)pop.push_back(std::make_unique<LocalSearch>(g,g.initial,c.seed+0x9e3779b97f4a7c15ULL*std::uint64_t(i+1)));
    for(int i=0;i<c.population&&!deadline.expired();++i){try{auto d=deadline.slice(std::max(.01,std::min(.2,c.seconds*.04)));if(i==0){if(g.value(g.initial)==0)pop[i]->greedy_start(.5,0.,d);else pop[i]->descent(d);}else pop[i]->greedy_start((i%3)*.5,.25,d);}catch(const Timeout&){}remember(pop[i]->s);}
    std::mt19937_64 rng(c.seed^0x589965cc75374cc3ULL),audit_rng(c.seed^0xd1b54a32d192ed03ULL);int stale=0;double last_event=-1.;
    SelectiveConfig sc;sc.seed_trials=c.gate_trials;sc.challengers=std::min(24,c.challengers);
    while(!deadline.expired()&&(c.max_rounds<0||st.rounds<std::uint64_t(c.max_rounds))) {
        ++st.rounds;Weight before=best;auto begin=Clock::now();
        // Population scheduling and initialization match the local baseline.
        for(auto&s:pop){s->run(c.local_iterations,deadline.slice(c.local_seconds));remember(s->s);}
        st.local_seconds+=elapsed(begin);stale=best>before?0:stale+1;
        double now=elapsed(started);
        if(deadline.expired())break;
        if(now<c.gate_warmup||stale<c.event_stale||now-last_event<c.gate_cooldown)continue;
        double credit=c.gate_fraction*now-st.escalation_seconds;
        if(credit<.005){++st.gate_budget_skips;continue;}
        last_event=now;++st.gate_epochs;std::uint64_t epoch=st.gate_epochs;
        int elite=0;for(int i=1;i<c.population;++i)if(pop[i]->cost>pop[elite]->cost)elite=i;
        begin=Clock::now();auto ed=deadline.slice(std::min(credit,.1));
        SelectiveStats ss;auto gate_started=Clock::now();bool gate_finished=false;
        try {
            auto gb=Clock::now();auto sketches=selective_sketches(g,*pop[elite],sc,rng,epoch,ed,&ss);
            st.gate_seconds+=elapsed(gb);gate_finished=true;
            std::vector<int>eligible;bool has_witness=false;
            for(int j=0;j<int(sketches.size());++j)if(sketches[j].upper_gain>0){eligible.push_back(j);has_witness|=sketches[j].witness_gain>0;}
            int pick=-1;bool explore=!has_witness&&epoch%std::uint64_t(c.explore_every)==0;
            if(!eligible.empty()&&(has_witness||(c.gate!="witness"&&explore))){
                if(c.gate=="random")pick=eligible[rng()%eligible.size()];
                else for(int j:eligible)if((has_witness?sketches[j].witness_gain>0:true)&&(pick<0||sketches[j].score>sketches[pick].score))pick=j;
            }
            st.heuristic_skips+=eligible.size()-(pick>=0?1:0);
            bool uniform_sample=audit_rng()%20==0;
            if(!c.sketch_snapshots.empty()&&(uniform_sample||has_witness)){
                std::filesystem::create_directories(c.sketch_snapshots);
                std::ofstream snap(c.sketch_snapshots+"/epoch_"+std::to_string(epoch)+".json");if(!snap)throw std::runtime_error("cannot write sketch snapshot");
                snap<<std::setprecision(17)<<"{\"epoch\":"<<epoch<<",\"sample_kind\":\""<<(uniform_sample?"uniform_epoch":"positive_witness_enriched")<<"\",\"pick\":"<<pick<<",\"incumbent\":";mask_json(snap,pop[elite]->s);snap<<",\"sketches\":[";
                for(int j=0;j<int(sketches.size());++j){if(j)snap<<',';auto&z=sketches[j];snap<<"{\"outside\":[";for(std::size_t t=0;t<z.outside.size();++t){if(t)snap<<',';snap<<z.outside[t];}snap<<"],\"upper_gain_ticks\":"<<z.upper_gain<<",\"witness_gain_ticks\":"<<z.witness_gain<<",\"score\":"<<z.score<<",\"policy_selected\":"<<(j==pick?"true":"false")<<'}';}snap<<"]}\n";
            }
            if(events.is_open())events<<"{\"event\":\"gate\",\"epoch\":"<<epoch<<",\"elapsed\":"<<now<<",\"stale_rounds\":"<<stale<<",\"raw_candidates\":"<<ss.raw_candidates<<",\"sketches\":"<<sketches.size()<<",\"safe_zero\":"<<ss.safe_zero<<",\"eligible\":"<<eligible.size()<<",\"pick\":"<<pick<<",\"explore\":"<<(explore?"true":"false")<<"}\n";
            if(pick>=0) {
                auto&z=sketches[pick];if(explore)++st.exploration_openings;else ++st.witness_openings;
                const Mask incumbent=pop[elite]->s;Mask improved=incumbent;
                if(z.witness_gain>0){for(int v:z.witness_outside){for(int u:g.adj[v])improved[u]=0;improved[v]=1;}
                    if(!g.feasible(improved)||g.value(improved)-g.value(incumbent)!=z.witness_gain)throw std::logic_error("invalid cheap witness");++st.witness_moves;}
                Weight gain=z.witness_gain;bool exact=z.upper_gain==gain;Weight upper=z.upper_gain;int kn=0;double used=0;std::string backend="bound_equals_witness";
                if(!exact&&!ed.expired()) {
                    try {
                    gb=Clock::now();auto b=singleton_backbone(g,incumbent,ed);std::vector<double>freq(g.n,0);for(int v=0;v<g.n;++v)freq[v]=incumbent[v];
                    auto k=make_kernel(g,b,incumbent,incumbent,freq,z.outside,c.max_kernel_vertices,ed);kn=k.graph.n;++st.kernels_materialized;st.build_seconds+=elapsed(gb);
                    gb=Clock::now();auto dispatched=recover_dispatch(k,c.recovery_backend,c.factor_options,c.kernel_nodes,ed.slice(c.kernel_seconds),c.decompose);used=elapsed(gb);st.recovery_seconds+=used;auto&r=dispatched.result;
                    ++st.recoveries;st.branch_nodes+=r.nodes;st.cut_calls+=r.cuts;st.cache_hits+=r.cache_hits;backend=dispatched.backend;
                    if(dispatched.factor.status!="not_started"){++st.factor_attempts;st.factor_exact+=dispatched.factor.status=="exact";st.factor_fallbacks+=backend=="factor_then_branch"||backend=="factor_then_greedy";st.factor_compiled_entries+=dispatched.factor.compiled_entries;st.factor_dp_entries+=dispatched.factor.dp_entries;}
                    Weight base=k.graph.value(k.base);upper=std::min(z.upper_gain,r.upper-base);
                    if(upper<gain)throw std::logic_error("recovery upper bound below feasible witness");
                    if(r.lower-base>gain){improved=lift(g,b,k,r);st.refinement_extra_ticks+=r.lower-base-gain;gain=r.lower-base;}
                    exact=upper==gain;st.recovery_exact+=exact;
                    if(events.is_open()){events<<"{\"event\":\"factor\",\"epoch\":"<<epoch<<",\"factor\":";factor_stats_json(events,dispatched.factor);events<<"}\n";}
                    }catch(const Timeout&){++st.gate_timeouts;backend="refinement_timeout";}catch(const std::length_error&){++st.gate_oversized;backend="refinement_oversized";}
                }
                if(gain>0){for(int v=0;v<g.n;++v){st.outsiders_inserted+=improved[v]&&!incumbent[v];st.consensus_released+=incumbent[v]&&!improved[v];}remember(improved);pop[elite]->reset(improved);++st.accepted_recoveries;st.positive_gain_ticks+=gain;stale=0;}
                if(events.is_open())events<<"{\"event\":\"selective_recovery\",\"epoch\":"<<epoch<<",\"kernel_n\":"<<kn<<",\"outsiders\":"<<z.outside.size()<<",\"cheap_upper_ticks\":"<<z.upper_gain<<",\"witness_gain_ticks\":"<<z.witness_gain<<",\"gain_ticks\":"<<gain<<",\"upper_gain_ticks\":"<<upper<<",\"exact\":"<<(exact?"true":"false")<<",\"solver_seconds\":"<<used<<",\"backend\":\""<<backend<<"\"}\n";
            }
        }catch(const Timeout&){++st.gate_timeouts;}catch(const std::length_error&){++st.gate_oversized;}
        if(!gate_finished)st.gate_seconds+=elapsed(gate_started);
        st.raw_candidates+=ss.raw_candidates;st.sketches+=ss.sketches;st.safe_rejects+=ss.safe_zero;st.neighbor_visits+=ss.neighbor_visits;
        st.gate_timeouts+=ss.partial;st.escalation_seconds+=elapsed(begin);
    }
    for(auto&s:pop){remember(s->s);st.local_iterations+=s->iterations;}
    if(!g.feasible(result.selected)||g.raw_value(result.selected)<g.raw_value(g.initial))throw std::logic_error("invalid selective final incumbent");
    write_checkpoint(c.checkpoint,g,result.selected);result.seconds=elapsed(started);result.cpu_seconds=double(std::clock()-cpu_start)/CLOCKS_PER_SEC;return result;
}
