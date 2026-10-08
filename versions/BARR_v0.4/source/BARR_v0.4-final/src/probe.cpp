#include "barr/mechanism_audit.hpp"
#include <iostream>
#include <sstream>
using namespace barr;
int main(int argc,char**argv){
    try{
        std::string input,output,backend="factor";double seconds=30;int enumerate=12;std::uint64_t nodes=1000000;FactorOptions options;
        auto number=[](const std::string&s){std::size_t pos;int x=std::stoi(s,&pos);if(pos!=s.size()||x<0)throw std::invalid_argument("invalid integer option");return x;};
        for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--help"){
            std::cout<<"barr_probe --kernel frozen.barrk --output report.json --backend factor|branch|hybrid --seconds 30\n"
              <<"--enumerate-limit 12 --nodes 1000000 --width 10 --boundary 10 --entries 262144\n"
              <<"Offline diagnostic: exact response tables + unary/pair/coalition measurements. Not a deployment budget.\n";return 0;}
            if(i+1>=argc)throw std::invalid_argument("missing argument");
            std::string v=argv[++i];
            if(key=="--kernel")input=v;else if(key=="--output")output=v;else if(key=="--backend")backend=v;
            else if(key=="--seconds"){std::size_t pos;seconds=std::stod(v,&pos);if(pos!=v.size())throw std::invalid_argument("invalid seconds");}
            else if(key=="--enumerate-limit")enumerate=number(v);else if(key=="--nodes")nodes=number(v);
            else if(key=="--width")options.max_width=number(v);else if(key=="--boundary")options.max_boundary=number(v);
            else if(key=="--entries")options.max_entries=number(v);else throw std::invalid_argument("unknown option "+key);
        }
        if(input.empty()||output.empty()||enumerate>20)throw std::invalid_argument("kernel, output and enumerate-limit <=20 required");
        if(std::filesystem::exists(output))throw std::invalid_argument("refusing to overwrite output");
        validate_factor_options(options);auto started=Clock::now();auto d=Deadline::after(seconds);auto k=read_kernel(input,d);
        std::ostringstream mechanism;DispatchRecovery run;auto t=Clock::now();double recovery_seconds=0,audit_seconds=0;bool completed_audit=false;std::string audit_status="not_requested";
        if(backend=="factor"){
            run.backend="factor";run.result.selected=k.base;run.result.lower=k.graph.value(k.base);run.result.upper=run.result.lower;
            for(int v:k.outsiders)run.result.upper+=k.graph.w[v];
            try{auto c=compile_response(k,options,d,run.factor);auto sol=eliminate_response(c,d,run.factor);
                run.result.upper=k.graph.value(k.base)+sol.gain;run.result.selected=reconstruct_response(k,c,sol,d,run.factor);
                run.result.lower=run.result.upper;run.result.exact=true;run.factor.status="exact";recovery_seconds=elapsed(t);auto audit_start=Clock::now();
                try{auto audit=audit_response(c,sol,enumerate,d);mechanism_json(mechanism,audit);completed_audit=true;audit_status="complete";}
                catch(const Timeout&){audit_status="timeout";}
                audit_seconds=elapsed(audit_start);
            }catch(const FactorLimit&e){run.factor.status=e.what();audit_status="not_compiled";}
            catch(const Timeout&){run.factor.status="timeout";audit_status="not_completed";}
            run.result.cuts=run.factor.cuts;run.result.exact=run.result.lower==run.result.upper;
        }else run=recover_dispatch(k,backend,options,nodes,d);
        auto&r=run.result;double native=elapsed(t);if(recovery_seconds==0)recovery_seconds=native;
        std::ofstream o(output);if(!o)throw std::runtime_error("cannot write report");
        o<<std::setprecision(17)<<"{\"schema\":\"barr_mechanism_probe_v1\",\"backend\":\""<<run.backend
          <<"\",\"supported\":"<<(run.backend=="branch_unsupported"?"false":"true")<<",\"kernel_vertices\":"<<k.graph.n
          <<",\"kernel_edges\":"<<k.graph.edges.size()<<",\"outsiders\":"<<k.outsiders.size()
          <<",\"base_ticks\":"<<k.graph.value(k.base)<<",\"lower_ticks\":"<<r.lower<<",\"upper_ticks\":"<<r.upper
          <<",\"exact\":"<<(r.exact?"true":"false")<<",\"feasible\":"<<(k.graph.feasible(r.selected)?"true":"false")
          <<",\"branch_nodes\":"<<r.nodes<<",\"cut_calls\":"<<r.cuts<<",\"recovery_and_audit_seconds\":"<<native
          <<",\"recovery_seconds\":"<<recovery_seconds<<",\"audit_seconds\":"<<audit_seconds
          <<",\"total_including_snapshot_validation_seconds\":"<<elapsed(started)<<",\"audit_status\":\""<<audit_status<<"\",\"factor\":";
        factor_stats_json(o,run.factor);o<<",\"mechanism\":"<<(completed_audit?mechanism.str():"null")
          <<",\"global_optimality_proven\":false,\"timing_is_offline_diagnostic\":true}\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<"Probe error: "<<e.what()<<'\n';return 2;}
}
