"""Matched repeated-output reference sensitivity with independently fixed judge draws.

A/B select whole human vectors by identity hashes. Acquisition reference selects
queries; evaluation reference re-reads those acquired positions and defines the
scoring target. Off-diagonal runs are fixed-query rescoring counterfactuals,
not operational executions under the evaluation reference or truth adjudication.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.conjunction_policy_data import load_datasets
from src.conjunction_policy_study import replay
from src.conjunction_label_value import hybrid_queries, SHARES

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('random_criterion', 'disagreement_then_random', 'sc_judge')
GRID = tuple(i / 100 for i in range(101))
CELLS = ('JB GPT-5.4', 'JB GPT-5.4-mini', 'JB GPT-5.4 | binary_only', 'JB GPT-5.4-mini | binary_only')
METRICS = ('actual_queries', 'certified_tasks', 'certified_pass', 'certified_fail',
           'certified_initial_correct', 'certified_initial_wrong', 'certified_FP', 'certified_FF',
           'eager_errors', 'eager_FP', 'eager_FF', 'gated_errors', 'gated_FP', 'gated_FF',
           'A_introduced_FP', 'D_delayed_FF', 'gated_minus_eager', 'fixed_initial_FP', 'fixed_initial_FF')


def identity_rank(salt, output, rater, annotation):
    return hashlib.sha256(f'{salt}|{output}|{rater}|{annotation}'.encode()).hexdigest()


def unit_slots(data):
    ti = np.asarray(data['task_index'])
    return {t: np.flatnonzero(ti == i) for i, t in enumerate(data['task_ids'])}


def frame_counts(data, selected=None):
    ti = np.asarray(data['task_index']); g = np.asarray(data['gold']); p = np.asarray(data['prediction'])
    selected = set(range(len(data['task_ids']))) if selected is None else set(selected)
    counts = dict(n_units=len(selected), n_criteria=0, reference_pass=0, judge_pass=0, TP=0, TN=0, FP=0, FF=0)
    for t in sorted(selected):
        ix = np.flatnonzero(ti == t); truth, judge = bool(np.all(g[ix])), bool(np.all(p[ix]))
        counts['n_criteria'] += len(ix); counts['reference_pass'] += truth; counts['judge_pass'] += judge
        counts['TP' if truth and judge else 'FF' if truth else 'FP' if judge else 'TN'] += 1
    return counts


def original_peer_diagnostics(data):
    """Same-output, distinct-human comparisons retaining each original judge row.

    No peer judge prediction replaces the current annotation's prediction.
    References here retain the source's annotation weighting, unlike the main
    one-output-one-unit A/B frame. These disagreements are not truth-error rates.
    """
    groups=defaultdict(list)
    for t,output in enumerate(data['output_ids']): groups[output].append(t)
    groups={o:ts for o,ts in groups.items() if len({data['rater_ids'][t] for t in ts})>=2}
    slots=[np.flatnonzero(np.asarray(data['task_index'])==t) for t in range(len(data['task_ids']))]
    truth={t:bool(np.all(data['gold'][slots[t]])) for ts in groups.values() for t in ts}
    judge={t:bool(np.all(data['prediction'][slots[t]])) for t in truth}
    peers={t:[u for u in groups[data['output_ids'][t]] if data['rater_ids'][u]!=data['rater_ids'][t]] for t in truth}
    passed=[t for t in truth if truth[t]]
    ff=[t for t in truth if truth[t] and not judge[t]]
    wrong=[t for t in truth if truth[t]!=judge[t]]
    result=dict(n_outputs=len(groups),n_annotations=len(truth),n_human_raters=len({data['rater_ids'][t] for t in truth}),
        n_base_tasks=len({data['base_task_ids'][t] for t in truth}),reference_pass=len(passed),
        reference_pass_all_peer_pass=sum(all(truth[u] for u in peers[t]) for t in passed),
        initial_FF=len(ff),initial_FF_any_peer_fail=sum(any(not truth[u] for u in peers[t]) for t in ff),
        initial_FF_all_peer_fail=sum(all(not truth[u] for u in peers[t]) for t in ff),
        judge_reference_disagreement=len(wrong),
        disagreement_any_peer_agrees_current_judge=sum(any(truth[u]==judge[t] for u in peers[t]) for t in wrong),
        disagreement_all_peer_agree_current_judge=sum(all(truth[u]==judge[t] for u in peers[t]) for t in wrong),
        judge_reference_task_agreement=sum(truth[t]==judge[t] for t in truth))
    pair_n=task_equal=item_n=item_equal=judge_vector_differ=judge_verdict_differ=0
    for ts in groups.values():
        for pos,t in enumerate(ts):
            for u in ts[pos+1:]:
                if data['rater_ids'][t]==data['rater_ids'][u]: continue
                it,iu=slots[t],slots[u]
                if [data['criterion_ids'][i] for i in it]!=[data['criterion_ids'][i] for i in iu]:
                    raise ValueError('peer human rubric mismatch')
                pair_n+=1;task_equal+=truth[t]==truth[u];item_n+=len(it)
                item_equal+=int((data['gold'][it]==data['gold'][iu]).sum())
                judge_vector_differ+=int(np.any(data['prediction'][it]!=data['prediction'][iu]))
                judge_verdict_differ+=judge[t]!=judge[u]
    result.update(human_unordered_distinct_rater_pairs=pair_n,human_pair_task_equal=task_equal,
        human_pair_criterion_equal=item_equal,human_pair_criterion_comparisons=item_n,
        saved_judge_binary_vector_different_pairs=judge_vector_differ,saved_judge_task_different_pairs=judge_verdict_differ,
        definition='Peers share output_id and have a different human rater_id; current judge prediction is always the original annotation-specific saved draw. Any/all summaries condition on the stated original annotation group; pairs and criterion comparisons have different denominators. Binary criterion agreement uses the declared full/binary-only target. No true-label error rate is identified.')
    return result


def build_frames(datasets):
    """Only consume the strict loader's portable arrays and identity metadata."""
    primary = datasets[CELLS[0]]
    members = defaultdict(list)
    for i, annotation in enumerate(primary['task_ids']):
        aid = annotation.removeprefix('judgmentbench/')
        members[primary['output_ids'][i]].append(dict(annotation_id=aid, source_task_id=annotation,
            rater_id=primary['rater_ids'][i], base_task_id=primary['base_task_ids'][i]))
    pairing = []
    for output, rows in members.items():
        if len({r['rater_id'] for r in rows}) < 2: continue
        if len({r['base_task_id'] for r in rows}) != 1: raise ValueError('same output spans incompatible tasks')
        ranked = sorted(rows, key=lambda r: identity_rank('reference-panel-v1', output, r['rater_id'], r['annotation_id']))
        selected, seen = [], set()
        for r in ranked:
            if r['rater_id'] not in seen: selected.append(r); seen.add(r['rater_id'])
            if len(selected) == 2: break
        anchor = min(rows, key=lambda r: identity_rank('judge-anchor-v1', output, r['rater_id'], r['annotation_id']))
        pair = dict(base_task_id=rows[0]['base_task_id'], output_id=output, available_annotations=len(rows), available_raters=len({r['rater_id'] for r in rows}))
        for label, chosen in [('A', selected[0]), ('B', selected[1]), ('anchor', anchor)]:
            pair[label+'_annotation_id'] = chosen['annotation_id']; pair[label+'_rater_id'] = chosen['rater_id']
        pairing.append(pair)
    pairing.sort(key=lambda r: (r['base_task_id'], r['output_id']))
    if not pairing: raise ValueError('no independently repeated outputs')
    selected_outputs = {r['output_id'] for r in pairing}
    frames, diagnostics = {}, {}
    shared_identity = {k: tuple(primary[k]) for k in ('task_ids', 'base_task_ids', 'output_ids', 'rater_ids')}
    for cell_name in CELLS:
        source = datasets[cell_name]
        if any(tuple(source[k]) != v for k, v in shared_identity.items()): raise ValueError('source annotation identities differ')
        by_aid = {t.removeprefix('judgmentbench/'): ix for t, ix in unit_slots(source).items()}
        gold_a, gold_b, prediction, auxiliary, indices, criteria, category = [], [], [], [], [], [], []
        for t, pair in enumerate(pairing):
            ia, ib, ij = (by_aid[pair[k+'_annotation_id']] for k in ('A', 'B', 'anchor'))
            cids = [source['criterion_ids'][i] for i in ia]
            for member in members[pair['output_id']]:
                peer = by_aid[member['annotation_id']]
                if cids != [source['criterion_ids'][i] for i in peer]:
                    raise ValueError('repeated-output rubric is incompatible across annotations')
            if cids != [source['criterion_ids'][i] for i in ib] or cids != [source['criterion_ids'][i] for i in ij]:
                raise ValueError('same-output reference or anchor rubric mismatch')
            if [source['category'][i] for i in ia] != [source['category'][i] for i in ib]: raise ValueError('reference scoring modes mismatch')
            gold_a.extend(source['gold'][ia]); gold_b.extend(source['gold'][ib]); prediction.extend(source['prediction'][ij])
            auxiliary.append(np.asarray(source['secondary_predictions'])[:, ij])
            indices.extend([t] * len(ia)); criteria.extend(f"{pair['output_id']}::{cid}" for cid in cids)
            category.extend(source['category'][i] for i in ia)
        common = dict(name=cell_name, prediction=np.array(prediction, dtype=np.uint8), task_index=np.array(indices, dtype=int),
            secondary_predictions=np.concatenate(auxiliary, axis=1), secondaries=list(source['secondaries']),
            task_ids=[r['output_id'] for r in pairing], output_ids=[r['output_id'] for r in pairing],
            base_task_ids=[r['base_task_id'] for r in pairing], criterion_ids=criteria, category=category,
            family=source['family'], target=source['target'], source_hashes=source['source_hashes'])
        frames[cell_name] = {ref: {**common, 'gold': np.array(bits, dtype=np.uint8),
            'rater_ids': [r[ref+'_rater_id'] for r in pairing]} for ref, bits in [('A', gold_a), ('B', gold_b)]}
        A, B = frames[cell_name]['A'], frames[cell_name]['B']
        task_a = [all(A['gold'][np.asarray(indices)==t]) for t in range(len(pairing))]
        task_b = [all(B['gold'][np.asarray(indices)==t]) for t in range(len(pairing))]
        repeat_ids = [i for i, out in enumerate(source['output_ids']) if out in selected_outputs]
        diagnostics[cell_name] = dict(original_annotation_frame=frame_counts(source),
            original_repeated_peer_diagnostics=original_peer_diagnostics(source),
            repeated_original_annotation_frame=frame_counts(source, repeat_ids),
            output_uniform_A=frame_counts(A), output_uniform_B=frame_counts(B),
            reference_agreement=dict(criterion_equal=int((A['gold']==B['gold']).sum()), n_criteria=len(gold_a),
                task_equal=sum(a==b for a,b in zip(task_a,task_b)), n_outputs=len(pairing),
                A_pass_B_fail=sum(a and not b for a,b in zip(task_a,task_b)),
                A_fail_B_pass=sum(not a and b for a,b in zip(task_a,task_b))))
    # Same target and anchor must exactly swap primary/auxiliary across judges.
    for suffix in ('', ' | binary_only'):
        a,b=frames['JB GPT-5.4'+suffix]['A'],frames['JB GPT-5.4-mini'+suffix]['A']
        if not np.array_equal(a['prediction'], b['secondary_predictions'][0]): raise ValueError('anchor auxiliary mismatch')
        if not np.array_equal(b['prediction'], a['secondary_predictions'][0]): raise ValueError('anchor primary mismatch')
        for ref in ('A','B'):
            if not np.array_equal(frames['JB GPT-5.4'+suffix][ref]['gold'], frames['JB GPT-5.4-mini'+suffix][ref]['gold']):
                raise ValueError('human panel differs between judges')
    return frames, pairing, diagnostics


def score_trace(frame, queries, evaluation_gold, shares=GRID):
    """Direct masks for every prefix; production metric/delta helpers unused.

    Evaluation labels provide both queried replacement bits and target truth.
    Therefore cross-reference scoring does not preserve acquisition answers.
    """
    if not np.isin(frame['prediction'], [0,1]).all() or not np.isin(evaluation_gold, [0,1]).all():
        raise ValueError('binary prediction and evaluation labels required')
    p = np.asarray(frame['prediction'], dtype=bool); g = np.asarray(evaluation_gold, dtype=bool)
    ti = np.asarray(frame['task_index']); m=len(p); n=len(frame['task_ids'])
    q=np.asarray(queries,dtype=int)
    if len(g)!=m or len(set(q.tolist()))!=len(q) or np.any(q<0) or np.any(q>=m): raise ValueError('invalid query trace or label frame')
    if not np.array_equal(np.unique(ti),np.arange(n)) or np.any(np.diff(ti)<0): raise ValueError('contiguous nonempty task slots required')
    starts=np.r_[0,np.flatnonzero(np.diff(ti))+1]
    truth=np.logical_and.reduceat(g,starts); initial=np.logical_and.reduceat(p,starts)
    fp=initial & ~truth; ff=~initial & truth
    rank=np.full(m,m+1,dtype=int); rank[q]=np.arange(len(q))
    result=[]
    for share in shares:
        if not 0 <= float(share) <= 1: raise ValueError('budget share outside [0,1]')
        cap=round(float(share)*m); used=min(cap,len(q)); seen=rank<used
        known_fail=np.logical_or.reduceat(seen & ~g,starts)
        known_pass=np.logical_and.reduceat(seen,starts) & truth
        certified=known_fail | known_pass
        eager=np.logical_and.reduceat(np.where(seen,g,p),starts)
        gated=np.where(certified,truth,initial)
        introduced=eager & ~truth & ~initial
        delayed=eager & truth & ~initial & ~certified
        row=dict(budget_share=float(share),budget_cap=cap,n_units=n,n_criteria=m,actual_queries=used,
            certified_tasks=int(certified.sum()),certified_pass=int(known_pass.sum()),certified_fail=int(known_fail.sum()),
            certified_initial_correct=int((certified & (truth==initial)).sum()),
            certified_initial_wrong=int((certified & (truth!=initial)).sum()),
            certified_FP=int((certified & fp).sum()),certified_FF=int((certified & ff).sum()),
            eager_errors=int((eager!=truth).sum()),eager_FP=int((eager & ~truth).sum()),eager_FF=int((~eager & truth).sum()),
            gated_errors=int((gated!=truth).sum()),gated_FP=int((gated & ~truth).sum()),gated_FF=int((~gated & truth).sum()),
            A_introduced_FP=int(introduced.sum()),D_delayed_FF=int(delayed.sum()),
            gated_minus_eager=int((gated!=truth).sum()-(eager!=truth).sum()),
            fixed_initial_FP=int((fp & ~eager).sum()),fixed_initial_FF=int((ff & eager).sum()))
        assert row['gated_minus_eager']==row['D_delayed_FF']-row['A_introduced_FP']
        assert not np.any(certified & (eager!=truth))
        assert not np.any(~eager & truth & initial)
        result.append(row)
    return result


def signs(values):
    return dict(negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),positive=sum(v>0 for v in values))


def summaries(rows, keys, metrics):
    groups=defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    out=[]
    for key, part in groups.items():
        r=dict(zip(keys,key));r.update(seeds=len(part),n_units=part[0]['n_units'])
        for metric in metrics:
            vals=[x[metric] for x in part];r[metric+'_mean']=float(np.mean(vals))
            r[metric+'_per100_mean']=100*float(np.mean(vals))/r['n_units']
            r[metric+'_median']=float(np.median(vals))
            r.update({metric+'_'+k:v for k,v in signs(vals).items()})
        out.append(r)
    return out


CONTRASTS=(('fixed_queries_A',('A','A'),('A','B')),
           ('fixed_queries_B',('B','A'),('B','B')),
           ('reacquired_own_reference',('A','A'),('B','B')))


def reference_changes(conditions, identity, metrics):
    rows=[]
    for label,left,right in CONTRASTS:
        a,b=conditions[left],conditions[right]
        row={**identity,'contrast':label,'left_acquisition':left[0],'left_evaluation':left[1],
             'right_acquisition':right[0],'right_evaluation':right[1], 'n_units':a['n_units']}
        for metric in metrics:
            row[metric+'_left']=a[metric];row[metric+'_right']=b[metric];row[metric+'_change']=b[metric]-a[metric]
        rows.append(row)
    return rows


def change_summaries(rows, keys, metrics):
    result=summaries(rows, keys, tuple(k+'_change' for k in metrics))
    groups=defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    for row in result:
        part=groups[tuple(row[k] for k in keys)]
        for metric in metrics:
            left=[x[metric+'_left'] for x in part];right=[x[metric+'_right'] for x in part]
            row[metric+'_left_mean']=float(np.mean(left));row[metric+'_right_mean']=float(np.mean(right))
            row[metric+'_strict_sign_reversals']=sum(a*b<0 for a,b in zip(left,right))
            row[metric+'_same_nonzero_sign']=sum(a*b>0 for a,b in zip(left,right))
            row[metric+'_both_zero']=sum(a==0 and b==0 for a,b in zip(left,right))
            row[metric+'_one_zero']=sum((a==0)!=(b==0) for a,b in zip(left,right))
    return result


def write_csv(path,rows):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'wt',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def output_path(root,output):
    root=Path(root).resolve();target=Path(output) if output is not None else root/'artifacts/reports/conjunction_robustness/reference'
    target=(target if target.is_absolute() else root/target).resolve()
    for relative in ('src','tests','data','context','docs','.git','artifacts/expected',
                     'artifacts/figures','artifacts/validation',
                     'artifacts/reports/conjunction_policy_study',
                     'artifacts/reports/conjunction_label_value',
                     'artifacts/reports/conjunction_mechanism'):
        protected=(root/relative).resolve()
        if target==protected or target in protected.parents or protected in target.parents: raise ValueError('output overlaps protected inputs')
    if target.exists():
        if not target.is_dir(): raise ValueError('output must be a directory')
        if any(p.is_symlink() for p in target.rglob('*')): raise ValueError('output contains symlinks')
    return target


def run(root=None,output=None,seeds=31):
    root=Path(root or ROOT).resolve();output=output_path(root,output)
    if not 1<=seeds<=31:raise ValueError('seeds must be1..31')
    protocol=root/'context/conjunction_robustness_protocol.md'
    if not protocol.exists():raise FileNotFoundError('frozen robustness protocol required')
    start=time.perf_counter();datasets=load_datasets(root);frames,pairing,diagnostics=build_frames(datasets)
    rq1=[];rq1_changes=[];rq2=[];rq2_changes=[];nonadaptive_checks=0
    for name,refs in frames.items():
        m=len(refs['A']['gold']);n=len(refs['A']['task_ids'])
        for seed in range(1,seeds+1):
            traces={}
            for policy in POLICIES:
                for acq in ('A','B'):traces[(policy,acq)]=replay(refs[acq],policy,'release',seed)['indices']
                if policy!='sc_judge':
                    if not np.array_equal(traces[(policy,'A')],traces[(policy,'B')]):raise AssertionError('nonadaptive query paths depend on reference')
                    nonadaptive_checks+=1
                conditions={}
                for acq in ('A','B'):
                    for ev in ('A','B'):
                        part=score_trace(refs[acq],traces[(policy,acq)],refs[ev]['gold'])
                        conditions[(acq,ev)]={r['budget_share']:r for r in part}
                        rq1.extend(dict(dataset=name,policy=policy,seed=seed,acquisition_ref=acq,evaluation_ref=ev,**r) for r in part)
                for share in GRID:
                    rq1_changes.extend(reference_changes({k:v[share] for k,v in conditions.items()},
                        dict(dataset=name,policy=policy,seed=seed,budget_share=share),
                        ('actual_queries','certified_tasks','eager_errors','gated_errors','A_introduced_FP','D_delayed_FF','gated_minus_eager')))
            for share in SHARES:
                cap=round(share*m);conditions={}
                for acq in ('A','B'):
                    random=traces[('random_criterion',acq)][:cap]
                    A,R=hybrid_queries(refs[acq],cap,'release',seed,{'indices':traces[('sc_judge',acq)]})
                    mix=np.r_[A,R]
                    if len(random)!=len(mix) or len(mix)!=cap:raise AssertionError('RQ2 unequal actual caps')
                    for ev in ('A','B'):
                        s=score_trace(refs[acq],random,refs[ev]['gold'],[share])[0]
                        h=score_trace(refs[acq],mix,refs[ev]['gold'],[share])[0]
                        row=dict(dataset=name,seed=seed,budget_share=share,budget_cap=cap,n_units=n,n_criteria=m,
                                 acquisition_ref=acq,evaluation_ref=ev,actual_queries=cap)
                        for metric in METRICS:
                            if metric=='actual_queries':continue
                            row['srs_'+metric]=s[metric];row['mix_'+metric]=h[metric];row[metric+'_delta']=h[metric]-s[metric]
                        rq2.append(row);conditions[(acq,ev)]=row
                rq2_changes.extend(reference_changes(conditions,dict(dataset=name,seed=seed,budget_share=share),
                    ('certified_tasks_delta','eager_errors_delta','gated_errors_delta','certified_initial_wrong_delta','certified_FP_delta','certified_FF_delta')))
        print(f'{name}: {seeds} paired seeds completed; {time.perf_counter()-start:.2f}s',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'pairing.csv',pairing)
    write_csv(output/'rq1_rows.csv.gz',rq1);write_csv(output/'rq1_reference_changes.csv.gz',rq1_changes)
    write_csv(output/'rq1_summary.csv',summaries(rq1,('dataset','policy','acquisition_ref','evaluation_ref','budget_share'),METRICS))
    c1=('actual_queries','certified_tasks','eager_errors','gated_errors','A_introduced_FP','D_delayed_FF','gated_minus_eager')
    write_csv(output/'rq1_reference_change_summary.csv',change_summaries(rq1_changes,('dataset','policy','contrast','budget_share'),c1))
    write_csv(output/'rq2_rows.csv.gz',rq2);write_csv(output/'rq2_reference_changes.csv.gz',rq2_changes)
    c2=tuple(k+'_delta' for k in METRICS if k!='actual_queries')
    write_csv(output/'rq2_summary.csv',summaries(rq2,('dataset','acquisition_ref','evaluation_ref','budget_share'),c2))
    cc=('certified_tasks_delta','eager_errors_delta','gated_errors_delta','certified_initial_wrong_delta','certified_FP_delta','certified_FF_delta')
    write_csv(output/'rq2_reference_change_summary.csv',change_summaries(rq2_changes,('dataset','contrast','budget_share'),cc))
    (output/'frame_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    paths=[Path(__file__),root/'src/conjunction_policy_data.py',root/'src/conjunction_policy_study.py',root/'src/conjunction_label_value.py',protocol]
    manifest=dict(schema='reference_sensitivity_v1',created_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,
        seeds=list(range(1,seeds+1)),rq1_policies=list(POLICIES),rq1_shares=list(GRID),rq2_shares=list(SHARES),
        n_outputs=len(pairing),n_base_tasks=len({r['base_task_id'] for r in pairing}),
        rows=dict(rq1=len(rq1),rq1_changes=len(rq1_changes),rq2=len(rq2),rq2_changes=len(rq2_changes)),
        nonadaptive_identical_path_checks=nonadaptive_checks,
        source_hashes={os.path.relpath(p,root):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        data_hashes={name:datasets[name]['source_hashes'] for name in CELLS},
        output_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest.json'},
        scope=['206 repeated outputs are each one unit; A/B are hash-chosen distinct human raters, not a consensus or truth adjudication.',
               'Whole vectors and independent hash-chosen saved primary/auxiliary judge draws stay fixed across panels, policies and targets.',
               'Off-diagonal acquisition/evaluation cells retain acquired positions but re-read their labels under the evaluation panel for replacement, certification and target truth.',
               'Diagonal A/A versus B/B reruns acquisition using each panel; off-diagonal rescoring is not a deployable execution under the evaluation panel.',
               '31 seeds describe query ordering only; no reference-population confidence or new-environment prediction is claimed.',
               'Original1539 and repeated431 annotation-frame diagnostics preserve original judge/reference pairing; comparing these with206 also changes weighting and anchor draws.',
               'Counts and per100-unit rates are reported; fixed-cap counts and actual queries remain distinct when short circuit stops early.',
               'No intervals, majority vote, oracle-friendly panel selection, new judge calls, or new human labels.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path)
    parser.add_argument('--seeds',type=int,default=31)
    args=parser.parse_args();print(json.dumps(run(args.root,args.output,args.seeds),indent=2))


if __name__=='__main__':main()
