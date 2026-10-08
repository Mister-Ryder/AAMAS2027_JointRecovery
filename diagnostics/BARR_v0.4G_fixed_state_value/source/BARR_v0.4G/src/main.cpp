#include "barr/solver.hpp"
#include <iostream>
#include <sstream>
#include <filesystem>
#include <set>
using namespace barr;
static int integer(const std::string&s){std::size_t p;long long v=std::stoll(s,&p);if(p!=s.size()||v<0||v>100000000)throw std::invalid_argument("invalid integer");return int(v);}
static double real(const std::string&s){std::size_t p;double v=std::stod(s,&p);if(p!=s.size()||!std::isfinite(v)||v<0||v>31536000.)throw std::invalid_argument("invalid real");return v;}
int main(int argc,char**argv){
    try{
        Config c;std::string input,output,policy_test;
        for(int i=1;i<argc;++i){
            std::string key=argv[i];
            if(key=="--help"){
                std::cout<<"BARR 0.4E: pulse ablation, no CHILS dependency.\n"
                    "--input BARR1 --output JSON --seconds 10 --seed 17 --population 4 --threads 1\n"
                    "--mode pair|selective|barr|fusion|local --rank heuristic|independent|random|gnn --model MODEL\n"
                    "--pair-component full|pulse-only (default full; pulse-only requires fusion-refine)\n"
                    "--gate value|random|witness --gate-fraction .05 --gate-warmup 30 --gate-cooldown .25\n"
                    "--event-stale 8 --explore-every 8 --gate-trials 8 --sketch-snapshots DIRECTORY\n"
                    "--challengers 12 --proposals 6 --execute-top 2 --kernel-nodes 128 --kernel-seconds .03\n"
                    "--local-iterations 64 --local-seconds .025 --fusion-seconds .10 --rounds N\n"
                    "--fixed-k --random-domains --no-decompose --greedy-repair\n"
                    "--recovery-backend hybrid|factor|branch --factor-width 10 --factor-boundary 10 --factor-entries 262144\n"
                    "--kernel-snapshots DIRECTORY (diagnostic I/O charged to time; TRAIN/DEV only in wrapper)\n"
                    "--max-kernel-vertices 4096 --max-gnn-vertices 4096 --events JSONL --trace JSONL --checkpoint JSON\n";return 0;
            }
            if(key=="--fixed-k"){c.fixed_k=true;continue;}if(key=="--random-domains"){c.random_domains=true;continue;}
            if(key=="--no-decompose"){c.decompose=false;continue;}if(key=="--greedy-repair"){c.greedy_repair=true;continue;}
            if(i+1>=argc)throw std::invalid_argument("missing value for "+key);
            std::string v=argv[++i];
            if(key=="--input")input=v;else if(key=="--output")output=v;else if(key=="--events")c.events=v;
            else if(key=="--trace")c.trace=v;else if(key=="--checkpoint")c.checkpoint=v;else if(key=="--model")c.model=v;
            else if(key=="--pair-policy")c.pair_policy=v;else if(key=="--pair-slice")c.pair_slice=real(v);else if(key=="--pair-max-seeds")c.pair_max_seeds=integer(v);else if(key=="--pair-fusion-every")c.pair_fusion_every=integer(v);
            else if(key=="--pair-component")c.pair_component=v;
            else if(key=="--gate")c.gate=v;else if(key=="--sketch-snapshots")c.sketch_snapshots=v;
            else if(key=="--gate-fraction")c.gate_fraction=real(v);else if(key=="--gate-cooldown")c.gate_cooldown=real(v);else if(key=="--gate-warmup")c.gate_warmup=real(v);
            else if(key=="--event-stale")c.event_stale=integer(v);else if(key=="--explore-every")c.explore_every=integer(v);else if(key=="--gate-trials")c.gate_trials=integer(v);
            else if(key=="--recovery-backend")c.recovery_backend=v;else if(key=="--kernel-snapshots")c.kernel_snapshots=v;
            else if(key=="--factor-width")c.factor_options.max_width=integer(v);else if(key=="--factor-boundary")c.factor_options.max_boundary=integer(v);
            else if(key=="--factor-entries")c.factor_options.max_entries=integer(v);
            else if(key=="--mode")c.mode=v;else if(key=="--rank")c.rank=v;else if(key=="--policy-test")policy_test=v;
            else if(key=="--seconds")c.seconds=real(v);else if(key=="--seed")c.seed=std::uint64_t(integer(v));
            else if(key=="--population")c.population=integer(v);else if(key=="--threads")c.threads=integer(v);
            else if(key=="--kernel-nodes")c.kernel_nodes=integer(v);else if(key=="--kernel-seconds")c.kernel_seconds=real(v);
            else if(key=="--local-seconds")c.local_seconds=real(v);else if(key=="--fusion-seconds")c.fusion_seconds=real(v);
            else if(key=="--local-iterations")c.local_iterations=integer(v);else if(key=="--rounds")c.max_rounds=integer(v);
            else if(key=="--challengers")c.challengers=integer(v);else if(key=="--proposals")c.proposals=integer(v);
            else if(key=="--execute-top")c.execute_top=integer(v);else if(key=="--max-kernel-vertices")c.max_kernel_vertices=integer(v);
            else if(key=="--max-gnn-vertices")c.max_gnn_vertices=integer(v);else throw std::invalid_argument("unknown option "+key);
        }
        if(!policy_test.empty()){
            NeuralPolicy net;net.load(c.model);std::ifstream f(policy_test);Features x;std::string magic;int n,m;
            if(!(f>>magic>>n>>m)||magic!="BARRFEAT1"||n<0||m<0)throw std::runtime_error("invalid feature file");
            x.x.resize(n);for(auto&row:x.x)for(double&v:row)if(!(f>>v)||!std::isfinite(v))throw std::runtime_error("invalid features");
            for(int i=0;i<m;++i){int u,v,t;if(!(f>>u>>v>>t)||u<0||v<0||u>=n||v>=n||(t!=0&&t!=1))throw std::runtime_error("invalid feature edge");x.edges.emplace_back(u,v,t);}
            for(double&v:x.context)if(!(f>>v))throw std::runtime_error("invalid context");
            auto y=net.forward(x,Deadline::after(60));
            std::cout<<std::setprecision(17)<<y[0]<<' '<<y[1]<<'\n';return 0;
        }
        if(input.empty())throw std::invalid_argument("--input is required");
        std::set<std::filesystem::path> paths;
        for(const auto& path: {input,output,c.events,c.trace,c.checkpoint,c.model}) {
            if(!path.empty() && !paths.insert(std::filesystem::weakly_canonical(path)).second)
                throw std::invalid_argument("input/output/model/log paths must be distinct");
        }
        Graph g=read_graph(input);auto r=solve(g,c);
        if(output.empty())result_json(std::cout,g,c,r);
        else{std::ofstream o(output+".tmp");if(!o)throw std::runtime_error("cannot write result");result_json(o,g,c,r);o.close();
            if(!o||std::rename((output+".tmp").c_str(),output.c_str())!=0)throw std::runtime_error("result write failed");}
        return 0;
    }catch(const std::exception&e){std::cerr<<"BARR error: "<<e.what()<<'\n';return 2;}
}
