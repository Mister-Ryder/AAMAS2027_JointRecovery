#pragma once
#include "kernel_io.hpp"
namespace barr {
struct MechanismAudit {
    Weight exact_gain=0,best_single=0,best_pair=0,selected_unary_sum=0,excess_over_unary=0;
    Weight additive_surrogate_optimum=0,additive_selection_true_gain=0;
    int selected_count=0,min_positive_size=-1,min_positive_lower=0,min_positive_upper=-1;
    std::uint64_t feasible_subsets=0;
    bool enumerated=false,additive_optimum_known=false,no_positive_coalition=false;
    std::vector<Weight>unary;
};
inline MechanismAudit audit_response(const CompiledResponse&c,const EliminationSolution&sol,int enumerate_limit,const Deadline&d){
    int n=int(sol.chosen.size());MechanismAudit a;a.exact_gain=sol.gain;a.unary.resize(n);std::vector<std::uint8_t>x(n,0);
    for(int i=0;i<n;++i){d.check();x[i]=1;a.unary[i]=evaluate_response(c,x);x[i]=0;a.best_single=std::max(a.best_single,a.unary[i]);
        if(sol.chosen[i]){++a.selected_count;a.selected_unary_sum+=a.unary[i];}}
    a.excess_over_unary=sol.gain-a.selected_unary_sum;a.best_pair=a.best_single;
    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){if((j&31)==0)d.check();x[i]=x[j]=1;
        a.best_pair=std::max(a.best_pair,evaluate_response(c,x));x[i]=x[j]=0;}
    if(a.exact_gain==0){a.no_positive_coalition=true;}
    else {a.min_positive_lower=a.best_single>0?1:(a.best_pair>0?2:3);a.min_positive_upper=a.selected_count;}
    if(n<=enumerate_limit){a.enumerated=true;a.additive_optimum_known=true;Weight best=-1;
        for(std::uint64_t bits=0;bits<table_size(n);++bits){if((bits&255)==0)d.check();int count=0;Weight unary=0;
            for(int i=0;i<n;++i){x[i]=(bits>>i)&1;if(x[i]){++count;unary+=a.unary[i];}}
            Weight gain=evaluate_response(c,x);if(gain==FACTOR_NEG)continue;++a.feasible_subsets;best=std::max(best,gain);
            if(gain>0&&(a.min_positive_size<0||count<a.min_positive_size))a.min_positive_size=count;
            if(unary>a.additive_surrogate_optimum){a.additive_surrogate_optimum=unary;a.additive_selection_true_gain=std::max(Weight(0),gain);}
        }
        if(best!=sol.gain)throw std::logic_error("audit enumeration disagrees with elimination");
        if(a.min_positive_size>=0)a.min_positive_lower=a.min_positive_upper=a.min_positive_size;
    }else if(a.best_single==0){a.additive_optimum_known=true; /* zero is attained by selecting nothing */}
    return a;
}
inline void mechanism_json(std::ostream&o,const MechanismAudit&a){
    o<<"{\"exact_gain_ticks\":"<<a.exact_gain<<",\"best_singleton_gain_ticks\":"<<a.best_single
      <<",\"best_size_at_most_two_gain_ticks\":"<<a.best_pair<<",\"selected_count\":"<<a.selected_count
      <<",\"selected_unary_sum_ticks\":"<<a.selected_unary_sum<<",\"excess_over_unary_ticks\":"<<a.excess_over_unary
      <<",\"all_subsets_enumerated\":"<<(a.enumerated?"true":"false")<<",\"feasible_subsets\":"<<a.feasible_subsets
      <<",\"min_positive_coalition_size\":";
    if(a.min_positive_size<0)o<<"null";else o<<a.min_positive_size;
    o<<",\"min_positive_size_lower\":"<<a.min_positive_lower<<",\"min_positive_size_upper\":";
    if(a.min_positive_upper<0)o<<"null";else o<<a.min_positive_upper;
    o<<",\"no_positive_coalition_certified\":"<<(a.no_positive_coalition?"true":"false")
      <<",\"additive_optimum_known\":"<<(a.additive_optimum_known?"true":"false")<<",\"additive_selection_actual_safe_gain_ticks\":";
    if(a.additive_optimum_known)o<<a.additive_selection_true_gain;else o<<"null";
    o<<",\"unary_forced_gain_ticks\":[";
    for(std::size_t i=0;i<a.unary.size();++i){if(i)o<<',';o<<a.unary[i];}o<<"]}";
}
} // namespace barr
