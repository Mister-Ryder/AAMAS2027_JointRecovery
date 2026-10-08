// Optional TEST HARNESS, not BARR's algorithm. Link against the unmodified pinned
// official local_search.c. It exposes only p1, not the complete CHILS program.
// Purpose: reproducible synthetic comparison when only the official p1 core is
// available. Formal experiments should use the complete published executable.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>
extern "C" {
#include "local_search.h"
volatile sig_atomic_t keep_running=1;
}
int main(int argc,char**argv){
    try{
        std::string path,warm,out;double seconds=1;int seed=17,pop=1,threads=1;
        for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("missing arg");std::string k=argv[i],v=argv[i+1];
            if(k=="-g")path=v;else if(k=="-i")warm=v;else if(k=="-o")out=v;else if(k=="-t")seconds=std::stod(v);
            else if(k=="-p")pop=std::stoi(v);else if(k=="-c")threads=std::stoi(v);else if(k=="-r")seed=std::stoi(v);
            else if(k!="-s")throw std::runtime_error("unsupported arg");}
        if(pop!=1||threads!=1)throw std::runtime_error("fixture supports official p1/c1 ONLY");
        std::ifstream input(path);int n,format;long long m;if(!(input>>n>>m>>format)||format!=10||n<1)throw std::runtime_error("bad METIS");
        std::string line;std::getline(input,line);std::vector<long long>w(n),offset(n+1);std::vector<int>edges;
        for(int u=0;u<n;++u){if(!std::getline(input,line))throw std::runtime_error("truncated METIS");std::istringstream row(line);if(!(row>>w[u])||w[u]<=0)throw std::runtime_error("bad weight");
            offset[u]=edges.size();int v;while(row>>v){if(v<1||v>n||v==u+1)throw std::runtime_error("bad edge");edges.push_back(v-1);}}
        offset[n]=edges.size();if((long long)edges.size()!=2*m)throw std::runtime_error("bad edge count");
        graph g{n,m,offset.data(),edges.data(),w.data()};omp_set_num_threads(1);
        local_search*ls=local_search_init(&g,(unsigned int)seed);std::ifstream initial(warm);int u;
        while(initial>>u){if(u<1||u>n)throw std::runtime_error("bad initial");local_search_add_vertex(&g,ls,u-1);}
        local_search_explore(&g,ls,seconds,std::numeric_limits<long long>::max(),0);
        for(int v=0;v<n;++v)if(ls->independent_set[v])for(long long i=offset[v];i<offset[v+1];++i)
            if(ls->independent_set[edges[i]])throw std::runtime_error("infeasible official p1 result");
        std::ofstream result(out);for(int v=0;v<n;++v)if(ls->independent_set[v])result<<v+1<<'\n';
        std::cout<<"official-p1-core-fixture,"<<ls->cost<<'\n';local_search_free(ls);return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}
