#pragma once
#include "coupling.hpp"
#include <filesystem>
namespace barr {
// Lossless frozen LOCAL state, not a global optimum certificate. Edges are complete
// inside the boundary-safe kernel; the original run binds it to the source graph.
inline void write_kernel(const Kernel&k,const std::string&path){
    namespace fs=std::filesystem;
    if(fs::exists(path))throw std::runtime_error("refusing to overwrite kernel snapshot");
    if(!fs::path(path).parent_path().empty())fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path+".tmp");if(!f)throw std::runtime_error("cannot write kernel snapshot");
    f<<std::setprecision(17)<<"BARRK1 "<<k.graph.n<<' '<<k.graph.edges.size()<<'\n';
    for(int i=0;i<k.graph.n;++i)f<<k.graph.w[i]<<' '<<k.graph.original[i]<<' '<<k.graph.owners[i]<<' '
        <<k.color[i]<<' '<<int(k.base[i])<<' '<<int(k.parent0[i])<<' '<<int(k.parent1[i])<<' '<<k.frequency[i]<<' '<<k.global_ids[i]<<'\n';
    for(auto[u,v]:k.graph.edges)f<<u<<' '<<v<<'\n';
    f.close();if(!f)throw std::runtime_error("kernel write failed");fs::rename(path+".tmp",path);
}
inline Kernel read_kernel(const std::string&path,const Deadline&d){
    std::ifstream f(path);std::string magic;int n;long long m;
    if(!(f>>magic>>n>>m)||magic!="BARRK1"||n<0||n>100000||m<0||m>20000000)throw std::runtime_error("invalid BARRK1 header");
    Kernel k;std::vector<Weight>w(n);std::vector<double>raw(n);std::vector<std::int64_t>own(n);std::set<int>ids;
    for(int i=0;i<n;++i){int col,base,p,q,id;double freq;
        if(!(f>>w[i]>>raw[i]>>own[i]>>col>>base>>p>>q>>freq>>id)||col< -1||col>1||base<0||base>1||p<0||p>1||q<0||q>1||!std::isfinite(freq)||freq<0||freq>1||id<0||!ids.insert(id).second)
            throw std::runtime_error("invalid BARRK1 vertex");
        if((col>=0)!=(p||q)||(col<0&&base))throw std::runtime_error("snapshot support inconsistency");
        k.color.push_back(col);k.base.push_back(base);k.parent0.push_back(p);k.parent1.push_back(q);k.frequency.push_back(freq);k.global_ids.push_back(id);
        if(col<0)k.outsiders.push_back(i);
    }
    std::vector<std::pair<int,int>>edges;edges.reserve(std::size_t(m));
    for(long long i=0;i<m;++i){int u,v;if(!(f>>u>>v))throw std::runtime_error("truncated BARRK1 edges");edges.emplace_back(u,v);}
    if(f>>magic)throw std::runtime_error("unexpected BARRK1 suffix");
    k.graph=Graph(w,edges,raw,own);k.graph.initial=k.base;
    if(!k.graph.feasible(k.parent0)||!k.graph.feasible(k.parent1)||!k.graph.feasible(k.base))throw std::runtime_error("infeasible snapshot memberships");
    k.base_backbone_exact=true;auto s=coupling_structure(k,d);
    // Do not trust an input file's exactness flag. Recompute every baseline value.
    for(const auto&part:s.parts){auto cut=bipartite_mwis(k.graph,part.vertices,k.color,d);
        if(cut.value!=part.base)throw std::runtime_error("snapshot base is not an exact backbone optimum");}
    return k;
}
inline void factor_stats_json(std::ostream&o,const FactorStats&s){
    o<<"{\"status\":\""<<s.status<<"\",\"variables\":"<<s.variables<<",\"backbone_components\":"<<s.components
      <<",\"interaction_components\":"<<s.interaction_components<<",\"max_component_variables\":"<<s.max_component_variables
      <<",\"max_boundary\":"<<s.max_boundary<<",\"induced_width\":"<<s.induced_width<<",\"planned_entries\":"<<s.planned_entries
      <<",\"compiled_entries\":"<<s.compiled_entries<<",\"dp_entries\":"<<s.dp_entries<<",\"cut_calls\":"<<s.cuts
      <<",\"structure_seconds\":"<<s.structure_seconds<<",\"compile_seconds\":"<<s.compile_seconds
      <<",\"elimination_seconds\":"<<s.elimination_seconds<<",\"reconstruction_seconds\":"<<s.reconstruction_seconds<<'}';
}
} // namespace barr
