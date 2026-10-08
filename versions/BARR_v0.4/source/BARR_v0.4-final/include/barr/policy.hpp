#pragma once
#include "recombination.hpp"
#include <array>
#include <tuple>
namespace barr {
constexpr int FEATURE_DIM=10, CONTEXT_DIM=6;
struct Features {
    std::vector<std::array<double,FEATURE_DIM>> x;
    std::vector<std::tuple<int,int,int>> edges; // undirected; 0 backbone, 1 outsider-related
    std::array<double,CONTEXT_DIM> context{};
    double weight_scale=1;
};
inline Features features(const Kernel&k,std::uint64_t nodes,double seconds) {
    const auto&g=k.graph;Features f;f.x.resize(g.n);
    f.weight_scale=g.n?std::max(1.,double(g.total)/g.n):1.;
    double dn=std::max(1.,std::log1p(double(g.n)));
    for(int v=0;v<g.n;++v){
        Weight bw=0,nw=0;double same=0,known=0;
        for(int u:g.adj[v]){
            nw+=g.w[u];if(k.base[u])bw+=g.w[u];
            if(g.owners[v]>=0 && g.owners[u]>=0){++known;if(g.owners[v]==g.owners[u])++same;}
        }
        double den=f.weight_scale*(1.+g.adj[v].size());
        f.x[v]={std::min(16.,g.w[v]/f.weight_scale),double(k.parent0[v]),double(k.parent1[v]),
                double(k.color[v]<0),double(k.base[v]),k.frequency[v],std::log1p(double(g.adj[v].size()))/dn,
                std::min(16.,bw/den),known?same/known:0.,std::min(16.,nw/den)};
    }
    for(auto e:g.edges)f.edges.emplace_back(e.first,e.second,(k.color[e.first]<0||k.color[e.second]<0)?1:0);
    f.context={std::log1p(double(g.n))/10.,std::log1p(double(g.edges.size()))/15.,
               g.n?double(k.outsiders.size())/g.n:0.,g.total?double(g.value(k.base))/double(g.total):0.,
               std::log1p(double(nodes))/10.,std::log1p(seconds*1000.)/10.};
    return f;
}
class NeuralPolicy {
    struct Linear {
        int rows=0,cols=0;
        std::vector<double>w,b;
        Linear()=default;
        Linear(int r,int c):rows(r),cols(c),w(std::size_t(r)*c),b(r){}
        std::vector<double> apply(const std::vector<double>&x)const {
            if(int(x.size())!=cols)throw std::logic_error("network dimension mismatch");
            auto y=b;for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)y[i]+=w[std::size_t(i)*cols+j]*x[j];return y;
        }
        void read(std::istream&f,bool bias=true){
            for(double&v:w)if(!(f>>v)||!std::isfinite(v)||std::abs(v)>1e6)throw std::runtime_error("invalid model weights");
            if(bias)for(double&v:b)if(!(f>>v)||!std::isfinite(v)||std::abs(v)>1e6)throw std::runtime_error("invalid model bias");
        }
    };
    struct Layer{Linear self,backbone,cross,high;};
    int hidden=0;
    Linear input,head,out;
    std::vector<Layer>layers;
public:
    bool loaded=false;
    void load(const std::string&path){
        std::ifstream f(path);std::string magic;int fin,layers_n,context;
        if(!(f>>magic>>fin>>hidden>>layers_n>>context)||magic!="BARRNN1"||fin!=FEATURE_DIM||context!=CONTEXT_DIM||hidden<1||hidden>64||layers_n<0||layers_n>4)
            throw std::runtime_error("unsupported BARRNN1 model");
        input=Linear(hidden,FEATURE_DIM);input.read(f);layers.clear();
        for(int i=0;i<layers_n;++i){Layer l{Linear(hidden,hidden),Linear(hidden,hidden),Linear(hidden,hidden),Linear(hidden,hidden)};
            l.self.read(f);l.backbone.read(f,false);l.cross.read(f,false);l.high.read(f,false);layers.push_back(std::move(l));}
        head=Linear(hidden,2*hidden+CONTEXT_DIM);out=Linear(2,hidden);head.read(f);out.read(f);
        std::string trailing;if(f>>trailing)throw std::runtime_error("model contains extra data");loaded=true;
    }
    std::array<double,2> forward(const Features&f,const Deadline&d)const {
        if(!loaded)throw std::logic_error("model not loaded");
        const int n=int(f.x.size());if(n==0)return {0.,-12.};
        std::vector<std::vector<double>>h(n);
        for(int v=0;v<n;++v){if((v&63)==0)d.check();h[v]=input.apply(std::vector<double>(f.x[v].begin(),f.x[v].end()));for(auto&x:h[v])x=std::max(0.,x);}
        for(const auto&l:layers){
            std::vector<std::vector<double>>m0(n,std::vector<double>(hidden,0)),m1=m0;
            std::vector<int>d0(n,0),d1(n,0);
            for(std::size_t z=0;z<f.edges.size();++z){
                if((z&1023)==0)d.check();
                auto [u,v,t]=f.edges[z];auto&m=t?m1:m0;auto&cnt=t?d1:d0;
                for(int j=0;j<hidden;++j){m[u][j]+=h[v][j];m[v][j]+=h[u][j];}++cnt[u];++cnt[v];
            }
            std::vector<std::vector<double>>next(n);
            for(int v=0;v<n;++v){
                if((v&31)==0)d.check();
                std::vector<double>hp(hidden);
                for(int j=0;j<hidden;++j){
                    double mean=(m0[v][j]+m1[v][j])/std::max(1,d0[v]+d1[v]);hp[j]=h[v][j]-mean;
                    m0[v][j]/=std::max(1,d0[v]);m1[v][j]/=std::max(1,d1[v]);
                }
                auto a=l.self.apply(h[v]),b=l.backbone.apply(m0[v]),c=l.cross.apply(m1[v]),z=l.high.apply(hp);
                for(int j=0;j<hidden;++j)a[j]=std::max(0.,a[j]+b[j]+c[j]+z[j]);
                next[v]=std::move(a);
            }h=std::move(next);
        }
        std::vector<double>pool(2*hidden+CONTEXT_DIM,0);int outside=0;
        for(int v=0;v<n;++v){for(int j=0;j<hidden;++j)pool[j]+=h[v][j]/n;
            if(f.x[v][3]>.5){++outside;for(int j=0;j<hidden;++j)pool[hidden+j]+=h[v][j];}}
        for(int j=0;j<hidden;++j)pool[hidden+j]/=std::max(1,outside);
        for(int j=0;j<CONTEXT_DIM;++j)pool[2*hidden+j]=f.context[j];
        auto a=head.apply(pool);for(double&v:a)v=std::max(0.,v);auto b=out.apply(a);
        if(!std::isfinite(b[0])||!std::isfinite(b[1]))throw std::runtime_error("nonfinite neural output");
        return {b[0],b[1]};
    }
    double score(const Features&f,const Deadline&d)const {
        auto y=forward(f,d);
        double gain=std::expm1(std::clamp(y[0],0.,20.))*f.weight_scale;
        double cost=std::exp(std::clamp(y[1],-16.,10.));return gain/cost;
    }
};

struct Proposal {
    Kernel kernel;
    double heuristic=0,independent=0,score=0;
};
struct ProposalBuildStats {std::uint64_t trials=0,duplicates=0,oversized=0;};
inline std::vector<Proposal> propose(const Graph&g,const Backbone&b,const Mask&p,const Mask&q,
        const std::vector<double>&freq,int count,int kmax,int vertex_cap,std::mt19937_64&rng,
        std::uint64_t epoch,bool random_domains,const Deadline&d,ProposalBuildStats*stats=nullptr) {
    std::vector<int>outside;std::vector<double>seed_score(g.n,0);std::vector<Weight>blocked(g.n,0);
    double scale=g.n?std::max(1.,double(g.total)/g.n):1.;
    for(int v=0;v<g.n;++v)if(!b.support[v]){
        if((v&511)==0)d.check();
        outside.push_back(v);
        for(int u:g.adj[v])if(b.merged[u])blocked[v]+=g.w[u];
        seed_score[v]=g.w[v]/(scale+blocked[v])*(1.+.25*(1.-freq[v]));
    }
    if(outside.empty()||count==0||kmax==0)return {};
    std::sort(outside.begin(),outside.end(),[&](int a,int c){return seed_score[a]!=seed_score[c]?seed_score[a]>seed_score[c]:a<c;});
    std::set<std::vector<int>>seen;
    std::vector<Proposal>proposals;
    for(int trial=0;trial<count*3 && int(proposals.size())<count;++trial){
        d.check();if(stats)++stats->trials;std::size_t top=std::min<std::size_t>(128,outside.size());
        int seed=(trial%3==2)?outside[rng()%outside.size()]:outside[(epoch*7+std::uint64_t(trial)*11)%top];
        std::set<int>pool_set{seed};
        if(!random_domains){
            for(int u:g.adj[seed])if(b.merged[u])for(int v:g.adj[u]){
                if(!b.support[v])pool_set.insert(v);
                if(pool_set.size()>=384)break;
            }
        }
        for(int attempts=0;pool_set.size()<std::size_t(std::min(384,std::max(kmax*4,kmax))) && attempts<256;++attempts)
            pool_set.insert(outside[rng()%outside.size()]);
        std::vector<int>pool(pool_set.begin(),pool_set.end()),chosen{seed};Mask paid(g.n,0);
        for(int u:g.adj[seed])if(b.merged[u])paid[u]=1;
        while(int(chosen.size())<kmax && chosen.size()<pool.size()){
            int best=-1;double best_score=-std::numeric_limits<double>::infinity();
            for(int v:pool)if(std::find(chosen.begin(),chosen.end(),v)==chosen.end()){
                Weight extra=0;for(int u:g.adj[v])if(b.merged[u]&&!paid[u])extra+=g.w[u];
                int conflicts=0;for(int x:chosen)if(g.adjacent(v,x))++conflicts;
                double score=double(g.w[v])-extra-.5*scale*conflicts;
                if(random_domains)score=std::generate_canonical<double,53>(rng);
                if(score>best_score){best_score=score;best=v;}
            }
            if(best<0)break;
            chosen.push_back(best);for(int u:g.adj[best])if(b.merged[u])paid[u]=1;
        }
        std::sort(chosen.begin(),chosen.end());if(!seen.insert(chosen).second){if(stats)++stats->duplicates;continue;}
        try{
            Proposal pr;pr.kernel=make_kernel(g,b,p,q,freq,chosen,vertex_cap,d);
            Weight rewards=0,union_cost=0;for(int v:chosen){rewards+=g.w[v];pr.independent+=double(g.w[v])-blocked[v];}
            for(int v=0;v<g.n;++v)if(paid[v])union_cost+=g.w[v];
            // A proposal score, NOT an upper bound and never used for pruning.
            double work=1.+pr.kernel.graph.edges.size()+pr.kernel.graph.n;
            pr.heuristic=(double(rewards)-union_cost+.05*rewards)/std::sqrt(work);
            pr.independent/=std::sqrt(work);pr.score=pr.heuristic;proposals.push_back(std::move(pr));
        }catch(const std::length_error&){if(stats)++stats->oversized; /* Never truncate a component. */ }
    }
    return proposals;
}
} // namespace barr
