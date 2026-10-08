#include "barr/fixed_capture_impl.hpp"
#include "barr/state_cli.hpp"
int main(int argc,char**argv){try{
    auto a=barr::fixed::arguments(argc,argv);if(a.state_dir.empty()||!a.snapshot.empty())throw std::invalid_argument("capture --state-dir required");if(std::filesystem::exists(a.state_dir))throw std::runtime_error("capture state directory must be new");
    auto graph=barr::read_graph(a.input);barr::fixed::CaptureControl control;control.state_dir=a.state_dir;control.graph_identity=barr::fixed::graph_wire(graph);control.graph_input_sha256=barr::fixed::sha256(barr::fixed::read_bytes(a.input));
    auto result=barr::solve_pair_capture(graph,a.config,control);std::ostringstream o;barr::fixed::capture_json(o,graph,a.config,result,control);barr::fixed::atomic_output(a.output,o.str());return 0;
}catch(const std::exception&e){std::cerr<<"Fixed capture error: "<<e.what()<<'\n';return 2;}}
