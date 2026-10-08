#include "barr/state_cli.hpp"
int main(int argc,char**argv){try{
    auto a=barr::fixed::arguments(argc,argv);if(a.snapshot.empty())throw std::invalid_argument("replay --snapshot required");
    auto graph=barr::read_graph(a.input);auto bytes=barr::fixed::read_bytes(a.snapshot);auto state=barr::fixed::state_read(bytes);if(barr::fixed::state_wire(state)!=bytes)throw std::runtime_error("snapshot wire roundtrip mismatch");if(state.graph_input_sha256!=barr::fixed::sha256(barr::fixed::read_bytes(a.input)))throw std::runtime_error("native input SHA differs from capture");
    auto result=barr::fixed::replay(graph,std::move(state),a.action,a.seconds,a.future_seed);std::ostringstream o;barr::fixed::replay_json(o,graph,result,a.action,a.seconds,a.future_seed,barr::fixed::sha256(bytes));barr::fixed::atomic_output(a.output,o.str());return 0;
}catch(const std::exception&e){std::cerr<<"Fixed replay error: "<<e.what()<<'\n';return 2;}}
