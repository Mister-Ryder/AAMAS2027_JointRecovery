#include "barr/mechanism_audit.hpp"
#include <iostream>
using namespace barr;
static std::uint64_t checks=0;
void require(bool b,const char*s){++checks;if(!b)throw std::runtime_error(s);}
Weight brute_allowed(const Graph&g,const Mask&allow){
    Weight best=0;std::function<void(int,Mask&,Weight)>go=[&](int v,Mask&s,Weight w){
        if(v==g.n){best=std::max(best,w);return;}go(v+1,s,w);if(!allow[v])return;
        for(int u:g.adj[v]) { if(s[u])return; }
        s[v]=1;go(v+1,s,w+g.w[v]);s[v]=0;};
    Mask s(g.n,0);go(0,s,0);return best;
}
Mask ris(const Graph&g,std::mt19937_64&r){Mask s(g.n,0);std::vector<int>vs(g.n);std::iota(vs.begin(),vs.end(),0);std::shuffle(vs.begin(),vs.end(),r);
    for(int v:vs)if(r()%3){bool ok=true;for(int u:g.adj[v])if(s[u])ok=false;if(ok)s[v]=1;}
    return s;}
Kernel chain_kernel(int k){
    // Each backbone component is an edge (weights 2,1). Each challenger (3)
    // touches BOTH endpoints in two successive components, producing triangles.
    int n=2*(k+1)+k;std::vector<Weight>w(n,3);std::vector<std::pair<int,int>>es;Mask p(n,0),q(n,0);std::vector<int>a;
    for(int j=0;j<=k;++j){w[2*j]=2;w[2*j+1]=1;es.emplace_back(2*j,2*j+1);p[2*j]=1;q[2*j+1]=1;}
    for(int i=0;i<k;++i){int v=2*(k+1)+i;a.push_back(v);for(int j:{i,i+1}){es.emplace_back(v,2*j);es.emplace_back(v,2*j+1);}}
    Graph g(w,es);auto b=fuse(g,p,q,Deadline::after(10));return make_kernel(g,b,p,q,std::vector<double>(n,.5),a,n,Deadline::after(10));
}
int main(){try{
    std::mt19937_64 rng(20261007);int cases=400,assignments=0,capped=0;FactorOptions o;
    o.max_width=12;o.max_boundary=12;o.max_entries=2000000;
    for(int t=0;t<cases;++t){int n=1+rng()%14;std::vector<Weight>w(n);for(auto&x:w)x=rng()%23;std::vector<std::pair<int,int>>es;
        int threshold=10+rng()%60;for(int u=0;u<n;++u)for(int v=u+1;v<n;++v)if(int(rng()%100)<threshold)es.emplace_back(u,v);
        Graph g(w,es);auto p=ris(g,rng),q=ris(g,rng);auto b=fuse(g,p,q,Deadline::after(30));std::vector<int>a;
        for(int v=0;v<n;++v)if(!b.support[v]&&a.size()<8)a.push_back(v);
        auto k=make_kernel(g,b,p,q,std::vector<double>(n,.5),a,n,Deadline::after(30));Weight opt=brute_allowed(k.graph,Mask(k.graph.n,1));
        FactorStats st;auto c=compile_response(k,o,Deadline::after(30),st);auto sol=eliminate_response(c,Deadline::after(30),st);
        auto x=reconstruct_response(k,c,sol,Deadline::after(30),st);
        require(k.graph.value(x)==opt,"factor optimum vs brute");require(k.graph.feasible(x),"factor feasibility");
        auto rf=recover_factored(k,o,Deadline::after(30));require(rf.result.exact&&rf.result.lower==opt,"factored API exactness");
        require(g.feasible(lift(g,b,k,rf.result)),"boundary-safe global lift");
        for(std::uint64_t bits=0;bits<(1ull<<a.size());++bits){Mask chosen(k.graph.n,0),allowed(k.graph.n,0);std::vector<std::uint8_t>z(a.size());Weight val=0;
            for(int i=0;i<int(a.size());++i)if((bits>>i)&1){z[i]=1;chosen[k.outsiders[i]]=1;val+=k.graph.w[k.outsiders[i]];}
            Weight got=evaluate_response(c,z);if(!k.graph.feasible(chosen)){require(got==FACTOR_NEG,"factor conflicts");continue;}
            for(int v=0;v<k.graph.n;++v)if(k.color[v]>=0){bool ok=true;for(int u:k.graph.adj[v])if(chosen[u])ok=false;if(ok)allowed[v]=1;}
            Weight real=val+brute_allowed(k.graph,allowed)-k.graph.value(k.base);require(real==got,"exact forced response identity");++assignments;
        }
        auto audit=audit_response(c,sol,12,Deadline::after(30));require(audit.exact_gain==opt-k.graph.value(k.base),"audit exact gain");
        for(int cap:{0,1,4,16}){auto co=o;co.max_entries=cap;auto z=recover_factored(k,co,Deadline::after(30));
            require(z.result.lower<=opt&&z.result.upper>=opt,"entry cap bounds");require(k.graph.feasible(z.result.selected),"entry cap membership");
            auto h=recover_dispatch(k,"hybrid",co,8,Deadline::after(30));require(h.result.lower<=opt&&h.result.upper>=opt,"hybrid bounds");++capped;}
        auto z=recover_factored(k,o,Deadline::after(0));require(z.result.lower<=opt&&z.result.upper>=opt,"timeout bounds");
        // Permute local vertex identifiers without changing mathematical content.
        std::vector<int>perm(k.graph.n);std::iota(perm.begin(),perm.end(),0);std::shuffle(perm.begin(),perm.end(),rng);Kernel pk=k;
        std::vector<Weight>pw(k.graph.n);std::vector<std::pair<int,int>>pe;pk.outsiders.clear();
        for(int v=0;v<k.graph.n;++v){pw[perm[v]]=k.graph.w[v];pk.color[perm[v]]=k.color[v];pk.base[perm[v]]=k.base[v];if(k.color[v]<0)pk.outsiders.push_back(perm[v]);}
        for(auto[u,v]:k.graph.edges)pe.emplace_back(perm[u],perm[v]);
        pk.graph=Graph(pw,pe);pk.graph.initial=pk.base;
        auto pr=recover_factored(pk,o,Deadline::after(30));require(pr.result.exact&&pr.result.lower==opt,"vertex permutation invariant optimum");
    }
    for(int k:{3,8,16,24,32,64,96,128}){auto x=chain_kernel(k);auto r=recover_factored(x,o,Deadline::after(30));
        require(r.result.exact,"long chain exact");require(r.stats.induced_width==1,"long chain width one");
        require(r.result.lower-x.graph.value(x.base)==k-2,"long chain gain");
        require(r.stats.max_boundary==2,"long chain pair boundary");
    }
    {auto k=chain_kernel(8);auto limited=o;limited.max_width=0;auto z=recover_factored(k,limited,Deadline::after(10));
        require(!z.result.exact&&z.stats.status=="induced_width_limit","width refusal is explicit");
        auto r=recover_dispatch(k,"hybrid",limited,20000,Deadline::after(10));require(r.result.exact,"hybrid branch fallback");}
    {Graph g({5,2,2,2},{{0,1},{0,2},{0,3}});Mask p{1,0,0,0};auto b=fuse(g,p,p,Deadline::after(3));
        auto k=make_kernel(g,b,p,p,{1,0,0,0},{1,2,3},4,Deadline::after(3));FactorStats st;auto c=compile_response(k,o,Deadline::after(3),st);
        auto sol=eliminate_response(c,Deadline::after(3),st);auto a=audit_response(c,sol,12,Deadline::after(3));
        require(a.best_single==0&&a.best_pair==0&&a.exact_gain==1&&a.min_positive_size==3,"pure third-order improvement");
        Weight third=evaluate_response(c,{1,1,1})-evaluate_response(c,{1,1,0})-evaluate_response(c,{1,0,1})-evaluate_response(c,{0,1,1})
            +evaluate_response(c,{1,0,0})+evaluate_response(c,{0,1,0})+evaluate_response(c,{0,0,1});require(third==-5,"nonzero higher-order coefficient");
    }
    {auto k=chain_kernel(32);auto r=recover_dispatch(k,"branch",o,128,Deadline::after(2));require(r.backend=="branch_unsupported","no hidden truncation to 24");}
    {auto k=chain_kernel(3);auto file=(std::filesystem::temp_directory_path()/"barr_v03_roundtrip_test.barrk").string();std::filesystem::remove(file);
        write_kernel(k,file);auto same=read_kernel(file,Deadline::after(3));std::filesystem::remove(file);require(same.graph.w==k.graph.w&&same.base==k.base,"snapshot roundtrip");}
    std::cout<<"{\"status\":\"PASS\",\"random_graphs\":"<<cases<<",\"forced_assignments_checked\":"<<assignments
      <<",\"capped_runs\":"<<capped<<",\"assertions\":"<<checks<<"}\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
