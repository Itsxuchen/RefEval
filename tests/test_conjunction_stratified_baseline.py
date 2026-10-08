from fractions import Fraction
from itertools import combinations, product
from math import comb

import numpy as np
import pytest

from src.conjunction_stratified_baseline import (
    integer_waterfill, pilot_allocation, select_design, infer_conditional, evaluate, validate_output, summary,
)


def test_waterfill_limits_and_tie_order():
    assert integer_waterfill(5,[1,1],[10,10],[1,1]).tolist() == [3,2]
    assert integer_waterfill(8,[100,1],[2,10],[1,1]).tolist() == [2,6]
    assert integer_waterfill(4,[0,0],[2,5],[0,0]).tolist() == [2,2]
    with pytest.raises(ValueError):
        integer_waterfill(1,[1,1],[4,4],[1,1])


def test_charged_pilot_and_explicit_small_cap_fallback():
    assert pilot_allocation([4,4],5) is None
    assert pilot_allocation([4,4],6).tolist() == [2,2]
    assert pilot_allocation([1,8],4).tolist() == [1,2]
    p=np.array([0]*4+[1]*4)
    calls=[]
    for cap in range(9):
        for policy in ('random_criterion','proportional_stratified','charged_pilot_neyman'):
            def observer(indices):
                calls.append(indices.copy())
                return np.array([int(i%3==0) for i in indices])
            d=select_design(p,cap,policy,7,observer)
            selected=np.r_[d['pilot'],d['sample']]
            assert len(selected)==len(set(selected.tolist()))==cap
            assert set(d['pilot']).isdisjoint(d['sample'])
            if policy=='charged_pilot_neyman' and cap<6:
                assert d['effective_policy']=='random_criterion'
                assert d['fallback_reason']
            for strata,n in zip(d['strata'],d['sample_counts']):
                assert n<=len(strata)
                if len(strata) and not d['fallback_reason'] and cap:
                    assert n>=1


def test_allocation_can_only_observe_pilot_labels():
    prediction=np.array([0]*20+[1]*20)
    y=np.array([0,1]*20)
    seen=[]
    d=select_design(prediction,17,'charged_pilot_neyman',9,
                    lambda idx: (seen.append(idx.copy()) or y[idx]))
    assert len(seen)==1 and np.array_equal(seen[0],d['pilot'])
    changed=y.copy();unknown=np.ones(40,bool);unknown[d['pilot']]=False;changed[unknown]=1-changed[unknown]
    d2=select_design(prediction,17,'charged_pilot_neyman',9,lambda idx:changed[idx])
    assert np.array_equal(d['sample'],d2['sample'])
    assert np.array_equal(d['sample_counts'],d2['sample_counts'])


def test_zero_remainder_and_small_layer_handling():
    exact=infer_conditional(5,2,[(0,0,0),(2,2,1)],0)
    assert exact['accuracy_estimate']==exact['accuracy_ci_lower']==exact['accuracy_ci_upper']==.6
    assert exact['normal_ci_width']==0
    unavailable=infer_conditional(5,0,[(5,1,1)],1)
    assert unavailable['normal_ci_width'] is None


def test_two_stage_conditional_design_unbiased_by_enumeration():
    # Pilot two from each four-item stratum; one additional adaptive allocation
    # is made from smoothed pilot variances, with capacity/minimum constraints.
    y=[np.array([0,0,0,1]),np.array([0,1,1,1])]
    target=Fraction(sum(int(v.sum()) for v in y),8)
    unconditional=[]
    for a,b in product(combinations(range(4),2),repeat=2):
        pilots=[a,b]
        correct=[int(y[h][list(pilots[h])].sum()) for h in range(2)]
        rates=np.array([(v+.5)/3 for v in correct])
        counts=integer_waterfill(3,2*np.sqrt(rates*(1-rates)),[2,2],[1,1])
        rest=[[j for j in range(4) if j not in pilots[h]] for h in range(2)]
        estimates=[]
        for sa,sb in product(combinations(rest[0],counts[0]),combinations(rest[1],counts[1])):
            pop=[(2,int(counts[0]),int(y[0][list(sa)].sum())),(2,int(counts[1]),int(y[1][list(sb)].sum()))]
            result=infer_conditional(8,sum(correct),pop,sum(N>n for N,n,x in pop))
            exact=(Fraction(sum(correct))+sum(Fraction(N*x,n) for N,n,x in pop))/8
            assert abs(result['accuracy_estimate']-float(exact))<1e-14
            estimates.append(exact)
        assert sum(estimates)/len(estimates)==target
        unconditional.extend([sum(estimates)/len(estimates)])
    assert sum(unconditional)/len(unconditional)==target


def test_bonferroni_conditional_coverage_small_worlds():
    # Exhaust population totals and sample counts in two residual strata.
    # Direct rational sample probabilities; integer bounds are checked at the
    # total-count scale, avoiding a float tail calculation as reference.
    from src.finite_population_intervals import hypergeom_ci
    for N1,N2 in product(range(1,5),repeat=2):
        for n1,n2 in product(range(1,N1+1),range(1,N2+1)):
            for K1,K2 in product(range(N1+1),range(N2+1)):
                covered=Fraction(0)
                for x1,x2 in product(range(max(0,n1-N1+K1),min(n1,K1)+1),
                                     range(max(0,n2-N2+K2),min(n2,K2)+1)):
                    probability=Fraction(comb(K1,x1)*comb(N1-K1,n1-x1),comb(N1,n1))*Fraction(comb(K2,x2)*comb(N2-K2,n2-x2),comb(N2,n2))
                    H=int(N1>n1)+int(N2>n2)
                    lo1,hi1=(x1,x1) if N1==n1 else hypergeom_ci(N1,n1,x1,Fraction(1,20)/H)
                    lo2,hi2=(x2,x2) if N2==n2 else hypergeom_ci(N2,n2,x2,Fraction(1,20)/H)
                    interval=infer_conditional(N1+N2,0,[(N1,n1,x1),(N2,n2,x2)],H)
                    assert abs(interval['accuracy_ci_lower']-(lo1+lo2)/(N1+N2))<1e-14
                    if lo1+lo2<=K1+K2<=hi1+hi2:covered+=probability
                assert covered>=Fraction(19,20)


def test_reproducible_metrics_and_total_cost():
    data={'gold':np.array([0,1,0,1,1,1,0,1]),'prediction':np.array([0,0,0,0,1,1,1,1]),
          'task_index':np.repeat(np.arange(4),2),'task_ids':['a','b','c','d']}
    for policy in ('random_criterion','proportional_stratified','charged_pilot_neyman'):
        row=evaluate(data,'toy',3,.75,policy)
        assert row==evaluate(data,'toy',3,.75,policy)
        assert row['pilot_queries']+row['main_queries']==row['actual_queries']==6
        assert 0<=row['accuracy_ci_lower']<=row['accuracy_ci_upper']<=1


def test_protected_ancestors_symlinks_and_external_output(tmp_path):
    root=tmp_path/'package';root.mkdir()
    (root/'artifacts/expected').mkdir(parents=True)
    for target in [root,root/'artifacts',root/'src/out',root/'artifacts/expected/rewrite',root/'data/new',root/'other',
                   root/'artifacts/reports/conjunction_policy_study',root/'artifacts/reports/conjunction_policy_study/new',
                   root/'artifacts/reports/conjunction_mechanism',root/'artifacts/reports/conjunction_mechanism/new']:
        with pytest.raises(ValueError):validate_output(target,root)
    link=tmp_path/'linked';link.symlink_to(root/'artifacts/expected',target_is_directory=True)
    with pytest.raises(ValueError):validate_output(link/'rewrite',root)
    external=tmp_path/'external'
    assert validate_output(external,root)==external
    external.mkdir();(external/'danger').symlink_to(root/'artifacts/expected',target_is_directory=True)
    with pytest.raises(ValueError):validate_output(external,root)
    assert validate_output(root/'artifacts/reports/conjunction_robustness/estimation',root)==root/'artifacts/reports/conjunction_robustness/estimation'


def test_nullable_normal_coverage_uses_arithmetic_not_boolean_reduction():
    import pandas as pd
    data={'gold':np.array([0,1,0,1,1,1,0,1]),'prediction':np.array([0,0,0,0,1,1,1,1]),
          'task_index':np.repeat(np.arange(4),2),'task_ids':['a','b','c','d']}
    rows=[evaluate(data,'toy',i,.75,'random_criterion') for i in (1,2,3)]
    frame=pd.DataFrame(rows)
    frame['normal_ci_covers']=pd.Series([np.bool_(True),np.bool_(False),None],dtype=object)
    result=summary(frame).iloc[0]
    assert result.approximate_normal_observed_coverage==.5
    assert result.approximate_normal_available==2
