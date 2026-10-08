"""Small-world independent masks and reference/anchor selection contracts."""
import itertools
import numpy as np
import pytest

from src.conjunction_reference_sensitivity import (
    CELLS, build_frames, frame_counts, identity_rank, original_peer_diagnostics, output_path, reference_changes, score_trace,
)
from src.conjunction_policy_study import replay
from src.conjunction_label_value import hybrid_queries


def toy(gold=(1,0,1), prediction=(0,1,1)):
    return dict(gold=np.array(gold), prediction=np.array(prediction), task_index=np.array([0,0,1]),
        task_ids=['t0','t1'],base_task_ids=['b0','b1'],output_ids=['o0','o1'],rater_ids=['r0','r1'],
        criterion_ids=['i0','i1','i2'],category=['binary']*3,secondary_predictions=np.array([[1,0,1]]),
        secondaries=['second'],family='test',target='and',source_hashes={})


def literal(data, queries, evaluation, share):
    cap=round(share*len(evaluation));observed=set(queries[:cap]);total={k:0 for k in (
        'certified_tasks','certified_pass','certified_fail','eager_errors','eager_FP','eager_FF',
        'gated_errors','A_introduced_FP','D_delayed_FF')}
    for t in range(len(data['task_ids'])):
        slots=np.flatnonzero(data['task_index']==t).tolist()
        truth=all(evaluation[i] for i in slots);initial=all(data['prediction'][i] for i in slots)
        certfail=any(not evaluation[i] for i in slots if i in observed)
        certpass=all(i in observed for i in slots) and truth
        cert=certfail or certpass
        end=all(evaluation[i] if i in observed else data['prediction'][i] for i in slots)
        gated=truth if cert else initial
        total['certified_tasks']+=cert;total['certified_pass']+=certpass;total['certified_fail']+=certfail
        total['eager_errors']+=end!=truth;total['eager_FP']+=end and not truth;total['eager_FF']+=not end and truth
        total['gated_errors']+=gated!=truth
        total['A_introduced_FP']+=end and not truth and not initial
        total['D_delayed_FF']+=end and truth and not initial and not cert
    total['actual_queries']=len(observed)
    return total


def test_exhaustive_evaluation_worlds_predictions_and_prefix_queries():
    worlds=list(itertools.product((0,1),repeat=3));shares=(0,1/3,2/3,1)
    count=0
    for g,p in itertools.product(worlds,worlds):
        data=toy(gold=(0,0,0),prediction=p)  # acquisition answers deliberately differ
        for order in itertools.permutations(range(3)):
            for limit in range(4):
                q=order[:limit]
                for row in score_trace(data,q,g,shares):
                    expected=literal(data,q,g,row['budget_share'])
                    assert all(row[k]==v for k,v in expected.items())
                    assert row['gated_minus_eager']==row['D_delayed_FF']-row['A_introduced_FP']
                    count+=1
    assert count==6144


def test_nonadaptive_paths_match_and_adaptive_reference_changes_are_visible():
    a=toy(gold=(0,1,1),prediction=(0,1,1));b={**a,'gold':np.array([1,1,1])}
    for policy in ('random_criterion','disagreement_then_random'):
        for seed in (1,2):
            assert np.array_equal(replay(a,policy,'release',seed)['indices'],replay(b,policy,'release',seed)['indices'])
    qa=replay(a,'sc_judge','release',1)['indices'];qb=replay(b,'sc_judge','release',1)['indices']
    assert len(qa)==2 and len(qb)==3
    cross=score_trace(a,qa,b['gold'],[1])[0]
    own=score_trace(b,qb,b['gold'],[1])[0]
    assert cross['actual_queries']==2 and cross['certified_tasks']==1 and cross['D_delayed_FF']==1
    assert own['actual_queries']==3 and own['certified_tasks']==2 and own['D_delayed_FF']==0
    for frame in (a,b):
        for cap in range(4):
            A,R=hybrid_queries(frame,cap,'release',1)
            assert len(A)+len(R)==cap


def selection_fixture():
    base=toy();base.update(task_ids=['judgmentbench/a0','judgmentbench/a1','judgmentbench/a2'],
       base_task_ids=['b0']*3,output_ids=['out']*3,rater_ids=['r0','r1','r2'],task_index=np.array([0,0,1,1,2,2]),
       criterion_ids=['c0','c1']*3,category=['binary']*6,gold=np.array([0,1,1,0,1,1]),
       prediction=np.array([0,0,1,1,0,1]),secondary_predictions=np.array([[1,0,0,1,1,1]]))
    data={}
    for name in CELLS:
        copied={**base}
        if 'mini' in name:
            copied['prediction']=base['secondary_predictions'][0].copy()
            copied['secondary_predictions']=base['prediction'][None,:].copy()
        data[name]=copied
    return data


def test_panel_and_independent_anchor_are_identity_only_and_shared():
    data=selection_fixture();frames,pairs,diag=build_frames(data)
    pair=pairs[0]
    ranked=sorted(range(3),key=lambda i:identity_rank('reference-panel-v1','out','r'+str(i),'a'+str(i)))
    anchor=min(range(3),key=lambda i:identity_rank('judge-anchor-v1','out','r'+str(i),'a'+str(i)))
    assert [pair['A_annotation_id'],pair['B_annotation_id']]==['a'+str(i) for i in ranked[:2]]
    assert pair['anchor_annotation_id']=='a'+str(anchor)
    source=data[CELLS[0]]
    for name in CELLS:
        for ref,i in zip(('A','B'),ranked[:2]):
            assert np.array_equal(frames[name][ref]['gold'],source['gold'][2*i:2*i+2])
        assert np.array_equal(frames[name]['A']['prediction'],frames[name]['B']['prediction'])
    # Changing labels and judge correctness cannot change selected identities.
    changed={name:{**d,'gold':1-d['gold'],'prediction':1-d['prediction'],'secondary_predictions':1-d['secondary_predictions']} for name,d in data.items()}
    assert build_frames(changed)[1]==pairs


def test_crossed_reference_changes_keep_fixed_query_and_rerun_distinct():
    cells={('A','A'):{'n_units':2,'x':1},('A','B'):{'n_units':2,'x':4},
           ('B','A'):{'n_units':2,'x':2},('B','B'):{'n_units':2,'x':3}}
    r=reference_changes(cells,{},('x',))
    assert {x['contrast']:x['x_change'] for x in r}=={'fixed_queries_A':3,'fixed_queries_B':1,'reacquired_own_reference':2}


def test_bad_trace_and_cross_rubric_fail_closed():
    for q in ([0,0],[-1],[3]):
        with pytest.raises(ValueError):score_trace(toy(),q,[1,1,1])
    data=selection_fixture();data[CELLS[0]]={**data[CELLS[0]],'criterion_ids':['c0','c1','wrong','c1','c0','c1']}
    with pytest.raises(ValueError):build_frames(data)


def test_original_peer_diagnostics_keep_current_judge_and_any_all_distinct():
    d=selection_fixture()[CELLS[0]]
    r=original_peer_diagnostics(d)
    assert r['n_outputs']==1 and r['n_annotations']==3
    assert r['reference_pass']==1 and r['reference_pass_all_peer_pass']==0
    assert r['initial_FF']==r['initial_FF_any_peer_fail']==r['initial_FF_all_peer_fail']==1
    assert r['judge_reference_disagreement']==2
    assert r['disagreement_any_peer_agrees_current_judge']==2
    assert r['disagreement_all_peer_agree_current_judge']==1
    assert r['human_unordered_distinct_rater_pairs']==3
    assert r['human_pair_task_equal']==1
    assert r['human_pair_criterion_equal']==2 and r['human_pair_criterion_comparisons']==6
    # Human peer conclusions hold per original annotation, independent of which
    # judge draw another annotation contains: only current-row eligibility moves.
    assert r['saved_judge_task_different_pairs']==2


def test_output_guard_blocks_protected_trees_ancestors_and_symlinks(tmp_path):
    root=tmp_path/'project';root.mkdir()
    protected=('src','tests','data','context','docs','.git','artifacts/expected',
               'artifacts/figures','artifacts/validation',
               'artifacts/reports/conjunction_policy_study',
               'artifacts/reports/conjunction_label_value','artifacts/reports/conjunction_mechanism')
    for rel in protected:
        path=root/rel;path.mkdir(parents=True,exist_ok=True)
        for target in (path,path/'child'):
            with pytest.raises(ValueError):output_path(root,target)
    for target in (root,root.parent,root/'artifacts',root/'artifacts/reports'):
        with pytest.raises(ValueError):output_path(root,target)
    link=tmp_path/'history';link.symlink_to(root/'artifacts/reports/conjunction_mechanism',target_is_directory=True)
    with pytest.raises(ValueError):output_path(root,link/'child')
    own=root/'artifacts/reports/conjunction_robustness/reference'
    assert output_path(root,None)==own
    own.mkdir(parents=True);(own/'manifest.json').write_text('{}')
    assert output_path(root,own)==own
    (own/'bad').symlink_to(root/'data',target_is_directory=True)
    with pytest.raises(ValueError):output_path(root,own)
    assert output_path(root,tmp_path/'fresh')==tmp_path/'fresh'
