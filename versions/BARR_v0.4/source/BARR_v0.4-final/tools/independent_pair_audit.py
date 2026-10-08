"""Independent original-NPZ final-mask <=2 outsider oracle (Python 3.8+).

This file imports no BARR modules/native code. It reconstructs blocker costs
from original edge arrays and globally sort/reduces all selected-blocker pair
occurrences. Outputs are post-hoc mechanism evidence, never performance cells.
"""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class AuditLimit(Exception):
    pass


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def in_sorted(haystack, needles):
    position = np.searchsorted(haystack, needles)
    found = position < len(haystack)
    found[found] = haystack[position[found]] == needles[found]
    return found


def audit_arrays(weights, edge_u, edge_v, selected_ids, memory_mib=1024., seconds=120.):
    started = time.monotonic()
    cpu_started = time.process_time()
    def check():
        if time.monotonic()-started >= seconds:
            raise AuditLimit('diagnostic_time_budget')
    if not math.isfinite(memory_mib) or memory_mib < 0 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid diagnostic budget')
    if weights.dtype.kind not in 'iu' or edge_u.dtype.kind not in 'iu' or edge_v.dtype.kind not in 'iu':
        raise ValueError('Original weights/edge IDs must be integer arrays')
    if weights.ndim != 1 or edge_u.ndim != 1 or edge_u.shape != edge_v.shape:
        raise ValueError('Original array shape mismatch')
    n, m = len(weights), len(edge_u)
    if n*n > np.iinfo(np.int64).max:
        raise ValueError('Edge-code integer domain overflow')
    total = sum(map(int, weights))
    if np.any(weights < 0) or total > (1 << 60)-1:
        raise ValueError('Negative weights or unsafe signed-64 total')
    w = weights.astype(np.int64, copy=False)
    u = edge_u.astype(np.int64, copy=False)
    v = edge_v.astype(np.int64, copy=False)
    if np.any(u < 0) or np.any(v < 0) or np.any(u >= n) or np.any(v >= n) or np.any(u == v):
        raise ValueError('Invalid original edge')
    original_codes = np.sort(np.minimum(u,v)*n+np.maximum(u,v))
    if len(original_codes)>1 and np.any(original_codes[1:] == original_codes[:-1]):
        raise ValueError('Duplicate original edge')
    if len(set(selected_ids)) != len(selected_ids) or any(isinstance(x,bool) or not isinstance(x,int) or x<0 or x>=n for x in selected_ids):
        raise ValueError('Frozen selected IDs must be unique valid integers')
    s = np.zeros(n,dtype=bool)
    s[selected_ids] = True
    if np.any(s[u] & s[v]):
        raise ValueError('Original full-graph frozen mask is infeasible')
    cross = s[u] ^ s[v]
    cu,cv = u[cross],v[cross]
    blocker = np.where(s[cu],cu,cv)
    outsider = np.where(s[cu],cv,cu)
    blocked_weight = np.zeros(n,dtype=np.int64)
    blocked_count = np.zeros(n,dtype=np.int64)
    np.add.at(blocked_weight,outsider,w[blocker])
    np.add.at(blocked_count,outsider,1)
    outside = np.flatnonzero(~s)
    deficit = blocked_weight-w
    unary = -deficit[outside]
    positive_singletons = int(np.sum(unary > 0))
    unary_gain = max(0,int(np.max(unary))) if len(outside) else 0
    unary_witness = [int(outside[np.flatnonzero(unary == unary_gain)[0]])] if unary_gain>0 else []
    base = int(np.sum(w[s]))
    signature = 1469598103934665603
    for bit in s:
        signature = ((signature ^ int(bit))*1099511628211) & ((1 << 64)-1)
    counts = np.bincount(blocker,minlength=n)
    occurrences = sum(int(k)*(int(k)-1)//2 for k in counts[s])
    # Conservative estimate includes simultaneously held sort/order/reduction
    # temporaries, edge arrays and Python/NumPy baseline. It is a refusal guard,
    # not a promise about the allocator or an absence certificate.
    estimated_peak = occurrences*96 + m*96 + n*128 + (64 << 20)
    upper_trivial = sum(sorted(map(int,w[outside]),reverse=True)[:2])
    result = {'schema':'barr_independent_final_pair_audit_v1',
              'scope':'All insertions of at most two originally unselected vertices, deleting their selected blocker union and keeping the rest of frozen incumbent fixed.',
              'method':'Original NPZ weight_ticks and edge_u/edge_v only; independent np.add.at blocker costs, selected-blocker two-hop pair codes, global sort/reduce shared weights, exact gain and original-edge conflict search.',
              'graph_n':n,'graph_m':m,'weight_ticks_total':total,'incumbent_ticks':base,
              'incumbent_selected_count':len(selected_ids),'incumbent_signature':signature,
              'outside_count':len(outside),'one_insertion_optimal':positive_singletons==0,
              'positive_singletons_count':positive_singletons,'positive_singletons_count_exact':True,
              'pair_occurrences_before_deduplication':occurrences,'estimated_peak_bytes':estimated_peak,
              'memory_cap_bytes':int(memory_mib*(1 << 20)),'budget_seconds':seconds,
              'complete':False,'exact':False,'unknown_reason':None,
              'lower_gain_ticks':unary_gain,'upper_gain_ticks':upper_trivial,
              'witness_outside':unary_witness,'positive_pairs_lower_count':0,'positive_pairs_count_exact':False,
              'unique_pairs_sharing_selected_blocker':None,'feasible_shared_pairs':None,
              'positive_shared_pairs':0,'positive_nonshared_pairs':0,
              'best_positive_pair_gain_ticks':0,'best_positive_pair_outside':[]}
    best_pair_gain = 0
    best_pair = []
    pair_count = 0
    try:
        check()
        if estimated_peak > result['memory_cap_bytes']:
            raise AuditLimit('pair_occurrence_memory_guard')
        order = np.argsort(blocker,kind='stable')
        sorted_outside = outsider[order]
        boundaries = np.cumsum(counts,dtype=np.int64)
        codes = np.empty(occurrences,dtype=np.int64)
        credits = np.empty(occurrences,dtype=np.int64)
        position = 0
        for selected in np.flatnonzero(s):
            check()
            begin = 0 if selected==0 else int(boundaries[selected-1])
            end = int(boundaries[selected])
            neighbors = np.sort(sorted_outside[begin:end])
            for j,a in enumerate(neighbors[:-1]):
                length = len(neighbors)-j-1
                codes[position:position+length] = int(a)*n+neighbors[j+1:]
                credits[position:position+length] = w[selected]
                position += length
        assert position == occurrences
        check()
        order = np.argsort(codes,kind='quicksort')
        sorted_codes = codes[order]
        sorted_credits = credits[order]
        del codes,credits,order,sorted_outside
        if occurrences:
            starts = np.r_[0,np.flatnonzero(sorted_codes[1:]!=sorted_codes[:-1])+1]
            unique = sorted_codes[starts]
            shared = np.add.reduceat(sorted_credits,starts)
        else:
            unique = np.empty(0,dtype=np.int64)
            shared = np.empty(0,dtype=np.int64)
        del sorted_codes,sorted_credits
        check()
        pair_a,pair_b = unique//n,unique%n
        compatible = ~in_sorted(original_codes,unique)
        gains = shared-deficit[pair_a]-deficit[pair_b]
        positive = compatible & (gains > 0)
        pair_count = int(np.sum(positive))
        result['unique_pairs_sharing_selected_blocker'] = len(unique)
        result['feasible_shared_pairs'] = int(np.sum(compatible))
        result['positive_shared_pairs'] = pair_count
        if pair_count:
            best_pair_gain = int(np.max(gains[positive]))
            at = int(np.flatnonzero(positive & (gains == best_pair_gain))[0])
            best_pair = [int(pair_a[at]),int(pair_b[at])]
        # If every unary<=0, disjoint selected blocker sets cannot give a
        # positive pair. Otherwise supplement every possibly positive nonshared
        # pair, which must contain a positive singleton and deficit sum<0.
        if positive_singletons:
            negative = outside[deficit[outside] < 0]
            for a in negative:
                check()
                eligible = outside[(outside!=a)&(deficit[outside]+deficit[a]<0)]
                eligible = eligible[(deficit[eligible]>=0)|(eligible>a)]
                extra_codes = np.minimum(a,eligible)*n+np.maximum(a,eligible)
                keep = ~in_sorted(unique,extra_codes) & ~in_sorted(original_codes,extra_codes)
                eligible = eligible[keep]
                extra_gain = -deficit[a]-deficit[eligible]
                extra_count = len(eligible)
                pair_count += extra_count
                result['positive_nonshared_pairs'] += extra_count
                if extra_count:
                    gain = int(np.max(extra_gain))
                    b = int(eligible[np.flatnonzero(extra_gain == gain)[0]])
                    witness = sorted([int(a),b])
                    if gain>best_pair_gain or gain==best_pair_gain and witness<best_pair:
                        best_pair_gain,best_pair = gain,witness
        result['complete'] = result['exact'] = True
    except AuditLimit as exc:
        result['unknown_reason'] = str(exc)
    result['positive_pairs_lower_count'] = pair_count
    result['positive_pairs_count_exact'] = result['complete']
    result['best_positive_pair_gain_ticks'] = best_pair_gain
    result['best_positive_pair_outside'] = best_pair
    if best_pair_gain > result['lower_gain_ticks']:
        result['lower_gain_ticks'] = best_pair_gain
        result['witness_outside'] = best_pair
    if result['complete']:
        result['upper_gain_ticks'] = result['lower_gain_ticks']
    result['positive_opportunity_exists'] = True if result['lower_gain_ticks']>0 else False if result['complete'] else None
    result['minimum_positive_coalition_size'] = 1 if positive_singletons else 2 if pair_count else None
    def certificate(ids):
        sets = [set(map(int,blocker[outsider==vertex])) for vertex in ids]
        union = set().union(*sets) if sets else set()
        intersection = set.intersection(*sets) if len(sets)==2 else set()
        gain = sum(int(w[x]) for x in ids)-sum(int(w[x]) for x in union)
        if len(ids)==2 and in_sorted(original_codes,np.array([ids[0]*n+ids[1]],dtype=np.int64))[0]:
            raise AssertionError('Independent witness outsiders conflict')
        return {'outside':ids,'singleton_gain_ticks':[int(w[x]-blocked_weight[x]) for x in ids],
                'selected_blocker_counts':[len(x) for x in sets],
                'selected_blocker_union_count':len(union),'selected_blocker_union_ids':sorted(union),
                'shared_selected_blocker_count':len(intersection),'shared_selected_blocker_ids':sorted(intersection),
                'shared_selected_blocker_weight_ticks':sum(int(w[x]) for x in intersection),
                'selected_blocker_union_cost_ticks':sum(int(w[x]) for x in union),'exact_gain_ticks':gain}
    result['witness_certificate'] = certificate(result['witness_outside'])
    result['best_positive_pair_certificate'] = certificate(best_pair)
    assert result['witness_certificate']['exact_gain_ticks'] == result['lower_gain_ticks']
    assert result['best_positive_pair_certificate']['exact_gain_ticks'] == best_pair_gain
    result['audit_seconds'] = time.monotonic()-started
    result['audit_cpu_seconds'] = time.process_time()-cpu_started
    return result


def direct_brute(w,u,v,selected):
    n = len(w)
    adjacency = [set() for _ in range(n)]
    for a,b in zip(u,v):
        adjacency[int(a)].add(int(b));adjacency[int(b)].add(int(a))
    base = set(selected)
    outsiders = sorted(set(range(n))-base)
    best = 0
    positive_pairs = positive_singles = 0
    for index,a in enumerate(outsiders):
        new = (base-adjacency[a])|{a}
        gain = sum(int(w[x]) for x in new)-sum(int(w[x]) for x in base)
        positive_singles += gain>0
        best = max(best,gain)
        for b in outsiders[index+1:]:
            if b in adjacency[a]:
                continue
            new = (base-adjacency[a]-adjacency[b])|{a,b}
            gain = sum(int(w[x]) for x in new)-sum(int(w[x]) for x in base)
            positive_pairs += gain>0
            best = max(best,gain)
    return best,positive_singles,positive_pairs


def self_test():
    cases = 0
    for edge_bits in range(64):
        pairs = [(a,b) for a in range(4) for b in range(a+1,4)]
        edges = [pair for bit,pair in enumerate(pairs) if edge_bits & (1 << bit)]
        u = np.array([a for a,b in edges],dtype=np.int64)
        v = np.array([b for a,b in edges],dtype=np.int64)
        w = np.array([0,3,5,7],dtype=np.int64)
        for mask in range(16):
            ids = [a for a in range(4) if mask & (1 << a)]
            if any(a in ids and b in ids for a,b in edges):
                continue
            result = audit_arrays(w,u,v,ids,seconds=10)
            truth = direct_brute(w,u,v,ids)
            assert result['complete'] and (result['lower_gain_ticks'],result['positive_singletons_count'],result['positive_pairs_lower_count']) == truth
            cases += 1
    rng = np.random.RandomState(404071)
    for iteration in range(250):
        n = int(rng.randint(1,13))
        w = rng.randint(0,103,size=n).astype(np.int64)
        edges = [(a,b) for a in range(n) for b in range(a+1,n) if rng.randint(100)<35]
        u = np.array([a for a,b in edges],dtype=np.int64)
        v = np.array([b for a,b in edges],dtype=np.int64)
        ids = []
        for a in rng.permutation(n):
            if rng.randint(2) and not any((min(int(a),b),max(int(a),b)) in edges for b in ids):
                ids.append(int(a))
        result = audit_arrays(w,u,v,ids,seconds=10)
        assert result['complete'] and (result['lower_gain_ticks'],result['positive_singletons_count'],result['positive_pairs_lower_count']) == direct_brute(w,u,v,ids)
        cases += 1
    w = np.array([10,3,9,5],dtype=np.int64)
    u = np.array([0,0,1],dtype=np.int64);v = np.array([2,3,3],dtype=np.int64)
    result = audit_arrays(w,u,v,[0,1],seconds=10)
    assert result['lower_gain_ticks']==1 and result['best_positive_pair_certificate']['selected_blocker_counts']==[1,2]
    for memory,seconds in [(0,10),(1024,0)]:
        result = audit_arrays(w,u,v,[0,1],memory_mib=memory,seconds=seconds)
        assert not result['complete'] and result['positive_opportunity_exists'] is None
    print(json.dumps({'status':'PASS','independent_direct_mask_brute_cases':cases+1,'memory_and_timeout_unknown_checks':2}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--npz',type=Path)
    parser.add_argument('--mask-json',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--cpp-oracle-json',type=Path)
    parser.add_argument('--memory-mib',type=float,default=1024.)
    parser.add_argument('--seconds',type=float,default=120.)
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if any(x is None for x in (args.npz,args.mask_json,args.output_dir)):
        parser.error('--npz --mask-json --output-dir required')
    # The audit refuses any output directory that already exists, avoiding all
    # writes to performance cells or original graph/frozen mask files.
    args.output_dir.mkdir(parents=True,exist_ok=False)
    loaded = time.monotonic()
    data = json.loads(args.mask_json.read_text(encoding='utf-8'))
    ids = data['selected']
    if not isinstance(ids,list):
        raise ValueError('Native selected must be a vertex ID list')
    with np.load(args.npz,allow_pickle=False) as source:
        weights = source['weight_ticks']
        edge_u = source['edge_u']
        edge_v = source['edge_v']
    load_seconds = time.monotonic()-loaded
    result = audit_arrays(weights,edge_u,edge_v,ids,args.memory_mib,args.seconds)
    if 'tick_value' in data and int(data['tick_value']) != result['incumbent_ticks']:
        raise ValueError('Original NPZ weight_ticks objective differs from frozen native mask objective')
    result['load_seconds'] = load_seconds
    if args.cpp_oracle_json is not None:
        cpp = json.loads(args.cpp_oracle_json.read_text(encoding='utf-8'))
        if (cpp['incumbent_ticks'],cpp['incumbent_selected_count'],cpp['incumbent_signature']) != (result['incumbent_ticks'],result['incumbent_selected_count'],result['incumbent_signature']):
            raise ValueError('C++ and independent oracle frozen-state binding mismatch')
        if result['complete']:
            truth = result['lower_gain_ticks']
            assert cpp['lower_gain_ticks'] <= truth <= cpp['upper_gain_ticks']
            if cpp['complete']:
                assert cpp['lower_gain_ticks'] == truth
                assert cpp['positive_pairs_lower_count'] == result['positive_pairs_lower_count']
        elif cpp['complete']:
            assert result['lower_gain_ticks'] <= cpp['lower_gain_ticks'] <= result['upper_gain_ticks']
        result['cpp_comparison'] = {'path':str(args.cpp_oracle_json.resolve()),'sha256':sha(args.cpp_oracle_json),
                                    'cpp_complete':cpp['complete'],'both_complete':bool(result['complete'] and cpp['complete']),
                                    'bounds_agree':True,'complete_positive_count_agrees':True if result['complete'] and cpp['complete'] else None,
                                    'binding':'Signature/size/objective checked; input mask file hash independently recorded in provenance.'}
    result_path = args.output_dir/'independent_pair_audit.json'
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    provenance = {'purpose':'Independent post-hoc original-graph complete <=2 outsider mechanism truth, separate from performance search/means; no native algorithm import, solve or parameter tuning.',
                  'created_utc':datetime.now(timezone.utc).isoformat(),'npz_path':str(args.npz.resolve()),'npz_sha256':sha(args.npz),
                  'mask_json_path':str(args.mask_json.resolve()),'mask_json_sha256':sha(args.mask_json),
                  'script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__)),
                  'result_sha256':sha(result_path),'numpy_version':np.__version__,'inputs_read_only':True,
                  'memory_refusal_is_unknown':True,'elapsed_seconds_include_validation_and_enumeration':True}
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('complete','unknown_reason','lower_gain_ticks','upper_gain_ticks','positive_singletons_count','positive_pairs_lower_count','audit_seconds','audit_cpu_seconds')}))


if __name__ == '__main__':
    main()
