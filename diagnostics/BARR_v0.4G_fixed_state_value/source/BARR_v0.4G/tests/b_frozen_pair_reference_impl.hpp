#pragma once
// Included inside namespace barr after the solver configuration declarations.
inline SolverResult solve_pair_frozen_b(const Graph&g,const Config&c) {
    if(c.pair_policy!="direct"&&c.pair_policy!="refine"&&c.pair_policy!="fusion-refine")throw std::invalid_argument("unknown pair policy");
    if(!std::isfinite(c.gate_fraction)||!std::isfinite(c.gate_cooldown)||!std::isfinite(c.gate_warmup)||!std::isfinite(c.pair_slice)||
       c.gate_fraction<=0||c.gate_fraction>.25||c.gate_cooldown<=0||c.gate_warmup<0||c.event_stale<1||c.pair_slice<=0||c.pair_fusion_every<1||c.pair_max_seeds<0)
        throw std::invalid_argument("invalid pair gate configuration");
    auto started=Clock::now();auto cpu_start=std::clock();Deadline deadline=Deadline::after(c.seconds);
    SolverResult result;result.selected=g.initial;Stats&st=result.stats;Weight best=g.value(g.initial);long double bestraw=g.raw_value(g.initial);
    std::ofstream events;if(!c.events.empty()){events.open(c.events);if(!events)throw std::runtime_error("cannot open events");events<<std::setprecision(17);}
    auto remember=[&](const Mask&s){best=std::max(best,g.value(s));auto raw=g.raw_value(s);if(raw>bestraw){if(!g.feasible(s))throw std::logic_error("infeasible pair incumbent");result.selected=s;bestraw=raw;write_checkpoint(c.checkpoint,g,s);}};
    write_checkpoint(c.checkpoint,g,result.selected);
    if(g.n==0){result.seconds=elapsed(started);return result;}
    std::vector<std::unique_ptr<LocalSearch>>pop;
    for(int i=0;i<c.population;++i)pop.push_back(std::make_unique<LocalSearch>(g,g.initial,c.seed+0x9e3779b97f4a7c15ULL*std::uint64_t(i+1)));
    for(int i=0;i<c.population&&!deadline.expired();++i){try{auto d=deadline.slice(std::max(.01,std::min(.2,c.seconds*.04)));if(i==0){if(g.value(g.initial)==0)pop[i]->greedy_start(.5,0.,d);else pop[i]->descent(d);}else pop[i]->greedy_start((i%3)*.5,.25,d);}catch(const Timeout&){}remember(pop[i]->s);}
    // Exact masks, rather than hash equality, guard reuse of zero certificates.
    std::vector<Mask>certified_masks(c.population);Mask fused_certificate;
    int stale=0;double last_event=-1.;
    while(!deadline.expired()&&(c.max_rounds<0||st.rounds<std::uint64_t(c.max_rounds))) {
        ++st.rounds;Weight before=best;auto begin=Clock::now();
        for(auto&s:pop){s->run(c.local_iterations,deadline.slice(c.local_seconds));remember(s->s);}
        st.local_seconds+=elapsed(begin);stale=best>before?0:stale+1;
        double now=elapsed(started);if(deadline.expired())break;
        if(now<c.gate_warmup||stale<c.event_stale||now-last_event<c.gate_cooldown)continue;
        double credit=c.gate_fraction*now-st.escalation_seconds;
        if(credit<.01){++st.gate_budget_skips;continue;}
        last_event=now;++st.gate_epochs;auto epoch=st.gate_epochs;
        int elite=0;for(int i=1;i<c.population;++i)if(pop[i]->cost>pop[elite]->cost)elite=i;
        int target=epoch%2?elite:int((epoch/2)%std::uint64_t(c.population));
        begin=Clock::now();auto ed=deadline.slice(std::min(credit,c.pair_slice));
        std::unique_ptr<LocalSearch>fused;LocalSearch*state=pop[target].get();bool used_fusion=false;
        try {
            if(c.pair_policy=="fusion-refine"&&epoch%std::uint64_t(c.pair_fusion_every)==0&&c.population>1&&ed.remaining()>.04){
                int far=elite;Weight distance=-1;
                for(int i=0;i<c.population;++i)if(i!=elite){Weight z=0;for(int v=0;v<g.n;++v)if(pop[elite]->s[v]!=pop[i]->s[v])z+=g.w[v];if(z>distance){far=i;distance=z;}}
                if(distance>0){auto fb=Clock::now();auto b=fuse(g,pop[elite]->s,pop[far]->s,ed.slice(c.fusion_seconds));st.fusion_seconds+=elapsed(fb);++st.fusions;st.fusion_exact+=b.exact;st.cut_calls+=b.cuts;
                    remember(b.merged);fused=std::make_unique<LocalSearch>(g,b.merged,c.seed^epoch);try{fused->descent(ed.slice(c.local_seconds));}catch(const Timeout&){}remember(fused->s);state=fused.get();used_fusion=true;
                    target=elite;for(int i=0;i<c.population;++i)if(pop[i]->cost<pop[target]->cost)target=i;
                }
            }
            bool repeated=used_fusion?(!fused_certificate.empty()&&fused_certificate==state->s):(!certified_masks[target].empty()&&certified_masks[target]==state->s);
            if(repeated){++st.pair_cached_zero;if(used_fusion&&state->cost>=pop[target]->cost)pop[target]->reset(state->s);st.escalation_seconds+=elapsed(begin);continue;}
            auto gb=Clock::now();PairScoutConfig pc;pc.max_seeds=c.pair_max_seeds;auto opportunity=pair_scout(g,*state,ed,pc,epoch);st.gate_seconds+=elapsed(gb);
            auto&ps=opportunity.stats;st.raw_candidates+=ps.outside_vertices;st.neighbor_visits+=ps.blocker_visits+ps.pair_visits+ps.intersection_visits;
            st.pair_visits+=ps.pair_visits;st.pair_unique+=ps.unique_pairs;st.pair_upper_rejects+=ps.upper_rejects;st.pair_conflict_rejects+=ps.conflict_rejects;st.pair_exact_checks+=ps.exact_pairs;
            st.pair_positive_candidates+=ps.positive_pairs;st.pair_complete+=opportunity.complete_two_exchange;st.pair_certificates+=opportunity.certificate_complete;st.gate_timeouts+=ps.partial;
            if(opportunity.certificate_complete){if(used_fusion)fused_certificate=state->s;else certified_masks[target]=state->s;}
            if(events.is_open())events<<"{\"event\":\"pair_gate\",\"epoch\":"<<epoch<<",\"elapsed\":"<<now<<",\"target\":"<<target<<",\"fused\":"<<(used_fusion?"true":"false")<<",\"pair_visits\":"<<ps.pair_visits<<",\"unique_pairs\":"<<ps.unique_pairs<<",\"upper_rejects\":"<<ps.upper_rejects<<",\"conflict_rejects\":"<<ps.conflict_rejects<<",\"exact_pairs\":"<<ps.exact_pairs<<",\"positive_pairs\":"<<ps.positive_pairs<<",\"positive_singletons\":"<<ps.positive_singletons<<",\"complete_two_exchange\":"<<(opportunity.complete_two_exchange?"true":"false")<<",\"certificate_zero\":"<<(opportunity.certificate_complete?"true":"false")<<",\"partial\":"<<(ps.partial?"true":"false")<<",\"witness_gain_ticks\":"<<opportunity.witness_gain<<",\"seconds\":"<<opportunity.seconds<<"}\n";
            const Mask incumbent=state->s;Mask improved=incumbent;Weight gain=opportunity.witness_gain;Weight extra=0;int kn=0;bool refined=false;
            if(gain>0){
                ++st.witness_openings;++st.witness_moves;
                std::vector<Weight>unary_gains;for(int v:opportunity.outside)unary_gains.push_back(g.w[v]-state->blocked_weight[v]);
                if(!c.sketch_snapshots.empty()){
                    std::filesystem::create_directories(c.sketch_snapshots);
                    std::ofstream snap(c.sketch_snapshots+"/pair_"+std::to_string(epoch)+".json");if(!snap)throw std::runtime_error("cannot write pair snapshot");
                    snap<<"{\"epoch\":"<<epoch<<",\"fused\":"<<(used_fusion?"true":"false")<<",\"gain_ticks\":"<<gain<<",\"incumbent\":";mask_json(snap,incumbent);snap<<",\"outside\":[";for(std::size_t i=0;i<opportunity.outside.size();++i){if(i)snap<<',';snap<<opportunity.outside[i];}snap<<"]}\n";
                }
                for(int v:opportunity.outside){for(int u:g.adj[v])improved[u]=0;improved[v]=1;}
                if(!g.feasible(improved)||g.value(improved)-state->cost!=gain)throw std::logic_error("invalid pair witness");
                // Archive the verified pair before optional expensive work. An
                // interrupted expansion cannot hide the already found move.
                remember(improved);
                if(c.pair_policy!="direct"&&!ed.expired()){
                    try {
                        // Only a proven positive seed opens a larger domain. Negative
                        // singleton candidates remain eligible during expansion.
                        std::vector<int>ids=opportunity.outside;std::vector<std::uint8_t>seen(g.n,0);std::vector<int>pool;std::vector<Weight>paid(g.n,0);
                        std::uint64_t expansion_work=0;auto expansion_check=[&](){if((expansion_work++&1023)==0)ed.check();};
                        for(int v:ids)seen[v]=1;
                        for(int v:ids){for(int u:g.adj[v]){expansion_check();if(incumbent[u]){paid[u]=1;for(int x:g.adj[u]){expansion_check();if(!incumbent[x]&&!seen[x]){seen[x]=1;pool.push_back(x);}}}}}
                        std::vector<std::pair<Weight,int>>ranked;ranked.reserve(pool.size());
                        for(int v:pool){ed.check();Weight unpaid=0;for(int u:g.adj[v]){expansion_check();if(incumbent[u]&&!paid[u])unpaid+=g.w[u];}ranked.emplace_back(g.w[v]-unpaid,v);}
                        std::sort(ranked.begin(),ranked.end(),[](auto a,auto b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
                        for(auto z:ranked){if(int(ids.size())>=std::min(24,std::max(2,c.challengers)))break;ids.push_back(z.second);}
                        auto sketch=selective_evaluate(g,incumbent,std::move(ids),ed);++st.sketches;
                        if(sketch.witness_gain>gain){improved=incumbent;for(int v:sketch.witness_outside){for(int u:g.adj[v])improved[u]=0;improved[v]=1;}extra+=sketch.witness_gain-gain;gain=sketch.witness_gain;}
                        if(sketch.upper_gain>gain&&!ed.expired()){
                            gb=Clock::now();auto b=singleton_backbone(g,incumbent,ed);std::vector<double>freq(g.n,0);for(int v=0;v<g.n;++v)freq[v]=incumbent[v];auto k=make_kernel(g,b,incumbent,incumbent,freq,sketch.outside,c.max_kernel_vertices,ed);kn=k.graph.n;++st.kernels_materialized;st.build_seconds+=elapsed(gb);
                            gb=Clock::now();auto rec=recover_dispatch(k,c.recovery_backend,c.factor_options,c.kernel_nodes,ed.slice(c.kernel_seconds),c.decompose);st.recovery_seconds+=elapsed(gb);++st.recoveries;st.recovery_exact+=rec.result.exact;st.branch_nodes+=rec.result.nodes;st.cut_calls+=rec.result.cuts;st.cache_hits+=rec.result.cache_hits;
                            if(rec.factor.status!="not_started"){++st.factor_attempts;st.factor_exact+=rec.factor.status=="exact";st.factor_fallbacks+=rec.backend=="factor_then_branch"||rec.backend=="factor_then_greedy";st.factor_compiled_entries+=rec.factor.compiled_entries;st.factor_dp_entries+=rec.factor.dp_entries;}
                            Weight rg=rec.result.lower-k.graph.value(k.base);if(rg>gain){improved=lift(g,b,k,rec.result);extra+=rg-gain;gain=rg;}refined=true;
                        }
                    }catch(const Timeout&){++st.gate_timeouts;}catch(const std::length_error&){++st.gate_oversized;}
                }
                if(!g.feasible(improved)||g.value(improved)-g.value(incumbent)!=gain)throw std::logic_error("invalid pair refinement");
                for(int v=0;v<g.n;++v){st.outsiders_inserted+=improved[v]&&!incumbent[v];st.consensus_released+=incumbent[v]&&!improved[v];}
                remember(improved);state->reset(improved);++st.accepted_recoveries;st.positive_gain_ticks+=gain;st.refinement_extra_ticks+=extra;stale=0;
                if(events.is_open()){events<<"{\"event\":\"pair_commit\",\"epoch\":"<<epoch<<",\"witness_gain_ticks\":"<<opportunity.witness_gain<<",\"gain_ticks\":"<<gain<<",\"extra_ticks\":"<<extra<<",\"kernel_n\":"<<kn<<",\"refined\":"<<(refined?"true":"false")<<",\"outside\":[";for(std::size_t i=0;i<opportunity.outside.size();++i){if(i)events<<',';events<<opportunity.outside[i];}events<<"],\"unary_gains_ticks\":[";for(std::size_t i=0;i<unary_gains.size();++i){if(i)events<<',';events<<unary_gains[i];}events<<"]}\n";}
            }
            if(used_fusion&&state->cost>=pop[target]->cost)pop[target]->reset(state->s);
        }catch(const Timeout&){++st.gate_timeouts;}
        st.escalation_seconds+=elapsed(begin);
    }
    for(auto&s:pop){remember(s->s);st.local_iterations+=s->iterations;}
    if(!g.feasible(result.selected)||g.raw_value(result.selected)<g.raw_value(g.initial))throw std::logic_error("invalid pair final incumbent");
    write_checkpoint(c.checkpoint,g,result.selected);result.seconds=elapsed(started);result.cpu_seconds=double(std::clock()-cpu_start)/CLOCKS_PER_SEC;return result;
}
