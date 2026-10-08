#include "barr/selective.hpp"
#include "barr/coupling.hpp"
#include <iostream>
using namespace barr;
// Independent frozen-state diagnostics. Never invoked by performance search.
int main(int argc,char**argv){try{
    if(argc!=4)throw std::invalid_argument("usage: barr_sketch_oracle graph.barr states.txt oracle.jsonl");
    Graph g=read_graph(argv[1]);std::ifstream f(argv[2]);std::ofstream o(argv[3]);
    if(!f||!o)throw std::runtime_error("cannot open oracle files");o<<std::setprecision(17);
    std::string magic;int states;if(!(f>>magic>>states)||magic!="BARRSKETCH1"||states<0)throw std::runtime_error("invalid states file");
    for(int s=0;s<states;++s){int epoch,uniform,pick,n,count;if(!(f>>epoch>>uniform>>pick>>n>>count)||n<0||n>g.n||count<0||count>128)throw std::runtime_error("invalid state header");
        Mask incumbent(g.n,0);for(int i=0;i<n;++i){int v;if(!(f>>v)||v<0||v>=g.n||incumbent[v])throw std::runtime_error("invalid incumbent ids");incumbent[v]=1;}
        if(!g.feasible(incumbent))throw std::runtime_error("infeasible frozen state");
        auto b=singleton_backbone(g,incumbent);std::vector<double>freq(g.n,0);for(int v=0;v<g.n;++v)freq[v]=incumbent[v];
        for(int j=0;j<count;++j){int k;if(!(f>>k)||k<0||k>24)throw std::runtime_error("invalid outsider count");std::vector<int>outside(k);for(int&v:outside)if(!(f>>v))throw std::runtime_error("invalid outsider ids");
            auto started=Clock::now();auto z=selective_evaluate(g,incumbent,outside,Deadline::after(5));
            auto kernel=make_kernel(g,b,incumbent,incumbent,freq,outside,100000,Deadline::after(5));
            // 12 challengers need at most 8191 branch nodes; budgets are diagnostic.
            auto r=recover(kernel,1048576,Deadline::after(1),true);Weight base=kernel.graph.value(kernel.base);
            Weight lower=std::max(z.witness_gain,r.lower-base),upper=r.upper-base;
            if(upper<lower||z.upper_gain<lower)throw std::logic_error("oracle contradicts cheap certificate");
            o<<"{\"epoch\":"<<epoch<<",\"domain\":"<<j<<",\"uniform_epoch\":"<<(uniform?"true":"false")<<",\"policy_selected\":"<<(j==pick?"true":"false")
              <<",\"outside\":[";for(int i=0;i<k;++i){if(i)o<<',';o<<outside[i];}o<<"],\"cheap_upper_ticks\":"<<z.upper_gain<<",\"witness_gain_ticks\":"<<z.witness_gain
              <<",\"lower_gain_ticks\":"<<lower<<",\"upper_gain_ticks\":"<<upper<<",\"exact\":"<<(lower==upper?"true":"false")<<",\"kernel_n\":"<<kernel.graph.n
              <<",\"blockers\":"<<z.blockers.size()<<",\"shared_blocker_credit\":"<<z.shared_blocker_credit<<",\"branch_nodes\":"<<r.nodes<<",\"cuts\":"<<r.cuts<<",\"seconds\":"<<elapsed(started)<<"}\n";o.flush();
        }
    }
    return 0;
}catch(const std::exception&e){std::cerr<<"oracle error: "<<e.what()<<'\n';return 2;}}
