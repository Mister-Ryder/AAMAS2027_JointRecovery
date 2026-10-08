#pragma once
// BARR: independently implemented solver. See NOTICE.md for algorithmic antecedents.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace barr {
using Weight = std::int64_t;
using Mask = std::vector<std::uint8_t>;
#ifdef BARR_TEST_FIXED_CLOCK
// Enabled only on the private pulse-ablation test target. Performance targets
// retain the original steady_clock definition verbatim in the default branch.
struct FixedTestClock {
    using duration=std::chrono::nanoseconds;
    using rep=duration::rep;using period=duration::period;
    using time_point=std::chrono::time_point<FixedTestClock,duration>;
    static constexpr bool is_steady=true;
    static inline duration ticks{0};
    static time_point now() noexcept {return time_point(ticks);}
    static void advance_round() noexcept {ticks+=std::chrono::milliseconds(100);}
    static void reset() noexcept {ticks=duration::zero();}
};
// Only ++Stats.rounds advances the private test clock. Repeated elapsed()/now()
// observations, logging, snapshots and extra branches cannot change scheduling.
struct FixedRoundCounter {
    std::uint64_t value=0;
    operator std::uint64_t() const noexcept {return value;}
    FixedRoundCounter& operator++() noexcept {++value;FixedTestClock::advance_round();return *this;}
    std::uint64_t operator++(int) noexcept {auto old=value;++*this;return old;}
};
using Clock = FixedTestClock;
#else
using Clock = std::chrono::steady_clock;
#endif
constexpr Weight MAX_TOTAL = (Weight(1) << 60) - 1;
struct Timeout : std::exception { const char* what() const noexcept override { return "cooperative timeout"; } };
struct Deadline {
    Clock::time_point end;
    static Deadline after(double s) {
        if (!std::isfinite(s) || s < 0 || s > 31536000.) throw std::invalid_argument("invalid time budget");
        return {Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s))};
    }
    bool expired() const { return Clock::now() >= end; }
    void check() const { if (expired()) throw Timeout(); }
    Deadline slice(double s) const { return {std::min(end, Deadline::after(s).end)}; }
    double remaining() const { return std::max(0., std::chrono::duration<double>(end-Clock::now()).count()); }
};
inline double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }

struct Graph {
    int n = 0;
    std::vector<Weight> w;
    std::vector<double> original;
    std::vector<std::int64_t> owners;
    std::vector<std::vector<int>> adj;
    std::vector<std::pair<int,int>> edges;
    Mask initial;
    Weight total = 0;
    Graph() = default;
    Graph(std::vector<Weight> weights, std::vector<std::pair<int,int>> es,
          std::vector<double> raw = {}, std::vector<std::int64_t> own = {})
        : n(int(weights.size())), w(std::move(weights)), original(std::move(raw)), owners(std::move(own)),
          adj(n), edges(std::move(es)), initial(n,0) {
        if (original.empty()) original.assign(w.begin(),w.end());
        if (owners.empty()) owners.assign(n,-1);
        if (original.size()!=w.size() || owners.size()!=w.size()) throw std::invalid_argument("vertex array length");
        for (int i=0;i<n;++i) {
            if(w[i]<0 || w[i]>MAX_TOTAL-total || !std::isfinite(original[i]) || original[i]<0)
                throw std::invalid_argument("negative, nonfinite or overflowing weights");
            total+=w[i];
        }
        for(auto &e:edges) {
            if(e.first<0 || e.second<0 || e.first>=n || e.second>=n || e.first==e.second)
                throw std::invalid_argument("invalid edge");
            if(e.first>e.second) std::swap(e.first,e.second);
        }
        std::sort(edges.begin(),edges.end());
        if(std::adjacent_find(edges.begin(),edges.end())!=edges.end()) throw std::invalid_argument("duplicate edge");
        for(auto e:edges) { adj[e.first].push_back(e.second); adj[e.second].push_back(e.first); }
        for(auto &a:adj) std::sort(a.begin(),a.end());
    }
    bool adjacent(int u,int v) const { return std::binary_search(adj[u].begin(),adj[u].end(),v); }
    bool feasible(const Mask &s) const {
        if(s.size()!=w.size()) return false;
        for(auto x:s) if(x>1) return false;
        for(auto e:edges) if(s[e.first] && s[e.second]) return false;
        return true;
    }
    Weight value(const Mask&s) const {
        if(s.size()!=w.size()) throw std::invalid_argument("mask length");
        Weight v=0; for(int u=0;u<n;++u) if(s[u]) v+=w[u]; return v;
    }
    long double raw_value(const Mask&s) const {
        if(s.size()!=w.size()) throw std::invalid_argument("mask length");
        long double v=0; for(int u=0;u<n;++u) if(s[u]) v+=static_cast<long double>(original[u]); return v;
    }
};
inline Graph read_graph(const std::string&path) {
    std::ifstream f(path); if(!f) throw std::runtime_error("cannot open input: "+path);
    std::string magic; long long n,m;
    if(!(f>>magic>>n>>m) || magic!="BARR1" || n<0 || n>10000000 || m<0 || m>1000000000LL)
        throw std::runtime_error("invalid BARR1 header");
    std::vector<Weight>w(n); std::vector<double>raw(n); std::vector<std::int64_t>own(n);
    for(long long i=0;i<n;++i) if(!(f>>w[i]>>raw[i]>>own[i])) throw std::runtime_error("truncated vertex data");
    std::vector<std::pair<int,int>>es; es.reserve(static_cast<std::size_t>(m));
    for(long long i=0;i<m;++i) { int u,v; if(!(f>>u>>v)) throw std::runtime_error("truncated edge data"); es.emplace_back(u,v); }
    Graph g(std::move(w),std::move(es),std::move(raw),std::move(own));
    int k; if(!(f>>k) || k<0 || k>n) throw std::runtime_error("invalid initial size");
    for(int i=0;i<k;++i) { int u; if(!(f>>u) || u<0 || u>=n || g.initial[u]) throw std::runtime_error("invalid initial vertex"); g.initial[u]=1; }
    std::string extra; if(f>>extra) throw std::runtime_error("unexpected input suffix");
    if(!g.feasible(g.initial)) throw std::runtime_error("infeasible initial solution");
    return g;
}
inline std::vector<int> members(const Mask&s) { std::vector<int>a; for(int i=0;i<int(s.size());++i) if(s[i]) a.push_back(i); return a; }
inline std::uint64_t signature(const Mask&s) {
    std::uint64_t h=1469598103934665603ULL; for(auto v:s) { h^=v; h*=1099511628211ULL; } return h;
}
} // namespace barr
