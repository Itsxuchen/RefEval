"""Independent finite enumeration checks of the oracle-structure expectation."""
import itertools
import math
import numpy as np
import pytest
from src.conjunction_update_mechanism import event_probability, expected_curve, ordered_groups, permutation_strata, permute_gold


def test_uniform_group_probability_against_literal_subsets():
    for G in range(8):
        for h in range(G+1):
            subsets=list(itertools.combinations(range(G),h))
            for r in range(G+1):
                for f in range(G-r+1):
                    required=set(range(r));forbidden=set(range(r,r+f))
                    exact=sum(required<=set(s) and not forbidden.intersection(s) for s in subsets)/len(subsets)
                    assert event_probability(G,h,r,f)==pytest.approx(exact,abs=1e-14)
    assert event_probability(3,2,2,2)==0
    with pytest.raises(ValueError):event_probability(3,4,0,0)


def data(g,p,flags=(1,0,0,1)):
    return dict(gold=np.array(g),prediction=np.array(p),task_index=np.array([0,0,1,1]),
                task_ids=['a','b'],secondary_predictions=np.array([[1-x if flag else x for x,flag in zip(p,flags)]]),
                category=['binary']*4)


def direct_effect(d,queried):
    A=D=0
    for slots in ([0,1],[2,3]):
        gold=[d['gold'][i] for i in slots];pred=[d['prediction'][i] for i in slots]
        truth=all(gold);initial=all(pred)
        updated=all(d['gold'][i] if i in queried else d['prediction'][i] for i in slots)
        certified=any(i in queried and not d['gold'][i] for i in slots) or all(i in queried for i in slots)
        A+=int(not truth and not initial and updated)
        D+=int(truth and not initial and updated and not certified)
    return A,D


@pytest.mark.parametrize('policy',['random_criterion','disagreement_then_random'])
@pytest.mark.parametrize('flags',[(1,0,0,1),(1,1,0,0),(1,1,1,1),(0,0,0,0)])
def test_full_expected_task_events_by_all_grouped_permutations(policy,flags):
    for g in itertools.product((0,1),repeat=4):
        for p in itertools.product((0,1),repeat=4):
            d=data(g,p,flags);group,sizes=ordered_groups(d,policy)
            choices=[list(itertools.permutations(np.flatnonzero(group==j))) for j in range(len(sizes))]
            orders=[sum((list(x) for x in parts),[]) for parts in itertools.product(*choices)]
            rows=expected_curve(d,policy,[0,.25,.5,.75,1])
            for B,row in enumerate(rows):
                events=[direct_effect(d,set(order[:B])) for order in orders]
                a=sum(x[0] for x in events)/len(events);dd=sum(x[1] for x in events)/len(events)
                assert row['expected_introduced_fp']==pytest.approx(a,abs=1e-13)
                assert row['expected_delayed_ff']==pytest.approx(dd,abs=1e-13)
                assert row['expected_delta']==pytest.approx(dd-a,abs=1e-13)


def test_permutation_preserves_declared_strata_and_observable_data():
    d=data([0,1,1,0],[1,1,1,1]);before=d['gold'].copy();strata=permutation_strata(d)
    changed=permute_gold(d,np.random.default_rng(20261007),strata)
    for ids in strata:assert changed['gold'][ids].sum()==d['gold'][ids].sum()
    np.testing.assert_array_equal(d['gold'],before)
    for key in ['prediction','task_index','secondary_predictions']:
        np.testing.assert_array_equal(changed[key],d[key])
