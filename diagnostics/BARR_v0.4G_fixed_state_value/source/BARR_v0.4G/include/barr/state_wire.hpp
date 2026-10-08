#pragma once
#include <array>
#include <cstring>
#include <sstream>
#include <locale>
namespace barr { namespace fixed {
inline std::uint32_t rotr(std::uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));}
inline std::string sha256(const std::string&input){
    static constexpr std::uint32_t k[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t,8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    std::string s=input;std::uint64_t bits=std::uint64_t(s.size())*8;s.push_back(char(0x80));while(s.size()%64!=56)s.push_back(0);for(int i=7;i>=0;--i)s.push_back(char(bits>>(i*8)));
    for(std::size_t off=0;off<s.size();off+=64){std::uint32_t w[64];for(int i=0;i<16;++i){w[i]=0;for(int j=0;j<4;++j)w[i]=(w[i]<<8)|std::uint8_t(s[off+4*i+j]);}for(int i=16;i<64;++i){auto a=w[i-15],b=w[i-2];w[i]=w[i-16]+(rotr(a,7)^rotr(a,18)^(a>>3))+w[i-7]+(rotr(b,17)^rotr(b,19)^(b>>10));}
        auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];for(int i=0;i<64;++i){auto t=v+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];auto u=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));v=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;}h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;}
    std::ostringstream o;o<<std::hex<<std::setfill('0');for(auto x:h)o<<std::setw(8)<<x;return o.str();
}
struct Writer {
    std::string b;
    void u(std::uint64_t x){for(int i=0;i<8;++i)b.push_back(char(x>>(8*i)));}
    void i(std::int64_t x){u(static_cast<std::uint64_t>(x));}
    void d(double x){static_assert(sizeof(double)==8,"IEEE binary64 required");std::uint64_t v;std::memcpy(&v,&x,8);u(v);}
    void text(const std::string&s){u(s.size());b+=s;}
    void raw(long double x){std::ostringstream o;o.imbue(std::locale::classic());o<<std::setprecision(std::numeric_limits<long double>::max_digits10)<<x;text(o.str());}
    template<class T>void vec(const std::vector<T>&a){u(a.size());for(auto x:a)i(static_cast<std::int64_t>(x));}
};
struct Reader {
    const std::string&b;std::size_t p=0;
    std::uint64_t u(){if(b.size()-p<8)throw std::runtime_error("truncated state wire");std::uint64_t x=0;for(int i=0;i<8;++i)x|=std::uint64_t(std::uint8_t(b[p++]))<<(8*i);return x;}
    std::int64_t i(){auto x=u();std::int64_t r;std::memcpy(&r,&x,8);return r;}
    double d(){auto v=u();double x;std::memcpy(&x,&v,8);if(!std::isfinite(x))throw std::runtime_error("nonfinite state wire");return x;}
    std::size_t length(std::size_t max=100000000){auto n=u();if(n>max||n>b.size()-p)throw std::runtime_error("state wire length");return std::size_t(n);}
    std::string text(){auto n=length(536870912);auto s=b.substr(p,n);p+=n;return s;}
    long double raw(){auto s=text();std::istringstream in(s);in.imbue(std::locale::classic());long double x;if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("invalid state raw value");in>>std::ws;if(!in.eof())throw std::runtime_error("raw suffix");return x;}
    template<class T>std::vector<T> vec(){auto n=length();std::vector<T>a;a.reserve(n);for(std::size_t j=0;j<n;++j){auto x=i();if(x<std::numeric_limits<T>::min()||x>std::numeric_limits<T>::max())throw std::runtime_error("state integer range");a.push_back(T(x));}return a;}
    void end(){if(p!=b.size())throw std::runtime_error("state wire suffix");}
};
inline std::string graph_wire(const Graph&g){Writer w;w.i(g.n);w.vec(g.w);w.u(g.original.size());for(auto x:g.original)w.d(x);w.vec(g.owners);w.vec(g.initial);w.u(g.edges.size());for(auto e:g.edges){w.i(e.first);w.i(e.second);}return w.b;}
inline void local_write(Writer&w,const LocalSearchState&x,bool with_rng=true){w.vec(x.s);w.vec(x.queued);w.vec(x.blocks);w.vec(x.queue);w.vec(x.blocked_weight);w.u(x.undo.size());for(auto e:x.undo){w.i(e.first);w.u(e.second);}w.i(x.cost);w.u(x.iterations);w.u(x.successful);w.i(x.failures);w.u(x.recording);w.i(x.protected_vertex);if(with_rng)w.text(x.rng);}
inline LocalSearchState local_read(Reader&r){LocalSearchState x;x.s=r.vec<std::uint8_t>();x.queued=r.vec<std::uint8_t>();x.blocks=r.vec<int>();x.queue=r.vec<int>();x.blocked_weight=r.vec<Weight>();auto n=r.length();for(std::size_t j=0;j<n;++j){int v=int(r.i());bool b=bool(r.u());x.undo.emplace_back(v,b);}x.cost=r.i();x.iterations=r.u();x.successful=r.u();x.failures=int(r.i());x.recording=bool(r.u());x.protected_vertex=int(r.i());x.rng=r.text();return x;}
inline std::string local_wire(const LocalSearchState&x,bool rng=true){Writer w;local_write(w,x,rng);return w.b;}
inline std::string rng_text(const std::mt19937_64&r){std::ostringstream o;o<<r;return o.str();}
inline std::mt19937_64 rng_read(const std::string&s){std::istringstream i(s);std::mt19937_64 r;if(!(i>>r))throw std::runtime_error("invalid audit RNG");i>>std::ws;if(!i.eof())throw std::runtime_error("audit RNG suffix");return r;}
inline void quoted(std::ostream&o,const std::string&s){o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<char(c);else if(c=='\n')o<<"\\n";else if(c=='\r')o<<"\\r";else if(c=='\t')o<<"\\t";else if(c<32)o<<"?";else o<<char(c);}o<<'"';}
inline std::string read_bytes(const std::string&p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read snapshot");return std::string(std::istreambuf_iterator<char>(f),{});}
inline void write_bytes(const std::string&p,const std::string&b){std::ofstream o(p,std::ios::binary);if(!o)throw std::runtime_error("cannot create snapshot");o.write(b.data(),std::streamsize(b.size()));o.close();if(!o)throw std::runtime_error("snapshot write failed");}
} }
