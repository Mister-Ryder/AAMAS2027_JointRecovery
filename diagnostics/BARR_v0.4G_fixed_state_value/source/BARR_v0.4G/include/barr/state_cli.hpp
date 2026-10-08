#pragma once
#include "fixed_state.hpp"
#include <iostream>
#include <cstdlib>
namespace barr { namespace fixed {
struct Arguments {Config config=frozen_config();std::string input,output,state_dir,snapshot,action;std::uint64_t future_seed=0;double seconds=360;};
inline std::uint64_t cli_integer(const std::string&s){std::size_t p;auto x=std::stoull(s,&p);if(s.empty()||s[0]=='-'||p!=s.size()||x>100000000)throw std::invalid_argument("invalid integer option");return x;}
inline double cli_real(const std::string&s){std::size_t p;double x=std::stod(s,&p);if(p!=s.size()||!std::isfinite(x)||x<0||x>31536000)throw std::invalid_argument("invalid real option");return x;}
inline Arguments arguments(int argc,char**argv){Arguments a;
    for(int i=1;i<argc;++i){std::string k=argv[i];if(k=="--help"){std::cout<<"Fixed-state diagnostic only. Capture: --input --output --state-dir --seconds 360 --seed 101. Replay: --input --output --snapshot --action recover|continue --seconds 360 --future-seed 0|1901|1907|1913|1931. Frozen E options are accepted explicitly.\n";std::exit(0);}if(i+1>=argc)throw std::invalid_argument("option needs value");std::string v=argv[++i];
        if(k=="--input")a.input=v;else if(k=="--output")a.output=v;else if(k=="--state-dir")a.state_dir=v;else if(k=="--snapshot")a.snapshot=v;else if(k=="--action")a.action=v;else if(k=="--seconds")a.seconds=cli_real(v);else if(k=="--future-seed")a.future_seed=cli_integer(v);
        else if(k=="--seed")a.config.seed=cli_integer(v);else if(k=="--kernel-nodes")a.config.kernel_nodes=cli_integer(v);
#define X(name) else if(k=="--" #name)a.config.name=int(cli_integer(v));
        // CLI hyphens are handled below; single-word options are direct.
        X(population) X(threads) X(challengers) X(proposals)
#undef X
        else if(k=="--local-iterations")a.config.local_iterations=int(cli_integer(v));else if(k=="--execute-top")a.config.execute_top=int(cli_integer(v));else if(k=="--pair-max-seeds")a.config.pair_max_seeds=int(cli_integer(v));else if(k=="--pair-fusion-every")a.config.pair_fusion_every=int(cli_integer(v));else if(k=="--event-stale")a.config.event_stale=int(cli_integer(v));
        else if(k=="--kernel-seconds")a.config.kernel_seconds=cli_real(v);else if(k=="--local-seconds")a.config.local_seconds=cli_real(v);else if(k=="--fusion-seconds")a.config.fusion_seconds=cli_real(v);else if(k=="--gate-fraction")a.config.gate_fraction=cli_real(v);else if(k=="--gate-warmup")a.config.gate_warmup=cli_real(v);else if(k=="--gate-cooldown")a.config.gate_cooldown=cli_real(v);else if(k=="--pair-slice")a.config.pair_slice=cli_real(v);
        else if(k=="--factor-width")a.config.factor_options.max_width=int(cli_integer(v));else if(k=="--factor-boundary")a.config.factor_options.max_boundary=int(cli_integer(v));else if(k=="--factor-entries")a.config.factor_options.max_entries=cli_integer(v);
        else if(k=="--mode")a.config.mode=v;else if(k=="--rank")a.config.rank=v;else if(k=="--recovery-backend")a.config.recovery_backend=v;else if(k=="--pair-policy")a.config.pair_policy=v;else if(k=="--pair-component")a.config.pair_component=v;
        else if(k=="--events")a.config.events=v;else if(k=="--checkpoint")a.config.checkpoint=v;else throw std::invalid_argument("unknown diagnostic option: "+k);
    }
    if(a.input.empty()||a.output.empty())throw std::invalid_argument("--input and --output required");a.config.seconds=a.seconds;
    if(a.config.population!=4||a.config.threads!=1||a.config.mode!="pair"||a.config.pair_policy!="fusion-refine"||a.config.pair_component!="full")throw std::invalid_argument("diagnostic capture requires frozen single-thread population4 E-full");
    std::set<std::filesystem::path> paths;for(auto&p:{a.input,a.output,a.snapshot,a.config.events,a.config.checkpoint})if(!p.empty()&&!paths.insert(std::filesystem::weakly_canonical(p)).second)throw std::invalid_argument("diagnostic file paths overlap");
    if(std::filesystem::exists(a.output))throw std::runtime_error("refusing to overwrite diagnostic output");return a;
}
inline void capture_json(std::ostream&o,const Graph&g,const Config&c,const SolverResult&r,const CaptureControl&control){o<<std::setprecision(17)<<"{\"schema\":\"barr_fixed_capture_v1\",\"seed\":"<<c.seed<<",\"n\":"<<g.n<<",\"m\":"<<g.edges.size()<<",\"graph_sha256\":";quoted(o,control.graph_input_sha256);o<<",\"config\":";config_json(o,c);o<<",\"native_seconds\":"<<r.seconds<<",\"cpu_seconds\":"<<r.cpu_seconds<<",\"thresholds\":[60,120,180],\"snapshots\":[";
    for(std::size_t i=0;i<control.records.size();++i){if(i)o<<',';auto&x=control.records[i];o<<"{\"id\":";quoted(o,x.id);o<<",\"threshold_seconds\":"<<x.threshold<<",\"capture_elapsed_seconds\":"<<x.elapsed<<",\"epoch\":"<<x.epoch<<",\"snapshot_path\":";quoted(o,x.snapshot_path);o<<",\"metadata_path\":";quoted(o,x.metadata_path);o<<",\"snapshot_sha256\":";quoted(o,x.snapshot_sha256);o<<",\"roundtrip_equal\":true,\"pending_prefix_seconds\":"<<x.prefix<<",\"scout_seconds\":"<<x.scout<<",\"capture_io_seconds\":"<<x.io<<'}';}
    o<<"],\"missing_thresholds\":[";for(std::size_t i=control.records.size();i<control.thresholds.size();++i){if(i!=control.records.size())o<<',';o<<control.thresholds[i];}o<<"],\"result\":";result_json(o,g,c,r);o<<",\"diagnostic_only\":true}\n";
}
inline void atomic_output(const std::string&path,const std::string&bytes){if(std::filesystem::exists(path)||std::filesystem::exists(path+".tmp"))throw std::runtime_error("diagnostic output exists");write_bytes(path+".tmp",bytes);if(std::rename((path+".tmp").c_str(),path.c_str())!=0)throw std::runtime_error("diagnostic output rename failed");}
} }
