"""Export only grading bits and rebuild checker/prompt comparisons from those bits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/reports/judge_repeat_20261125'
OLD = ROOT / 'artifacts/reports/judge_panel_20261125'
ARMS = ('original','repeat','alternate')
CHECKERS = ('native','literal','without_ending')
PAIRS = (('gpt_mini','gpt_strong'),('claude_haiku','claude_opus'))


def json_write(path, data):
    text = json.dumps(data, indent=2, allow_nan=False) + '\n'
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def export():
    from src import judge_repeat_panel as acquisition
    from src import judge_panel_api as panel
    spec, units, rows, cfg = acquisition.freeze()
    if json.loads((OUT/'freeze.json').read_text()) != json.loads(json.dumps(spec)):
        raise ValueError('Acquisition identity changed')
    db = ROOT/'data/interim/judge_repeat_20261125/ledger.sqlite'
    # Read-only transport. Reconstruct accepted vectors with the frozen parser.
    conn = sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    class Receipts:
        def call(self, job, key, model, body):
            entry = conn.execute('SELECT body_hash,result FROM calls WHERE job=?',(job,)).fetchone()
            if not entry or entry[1] is None or entry[0] != panel.digest(body):
                raise ValueError('Missing or inconsistent completed receipt: '+job)
            return json.loads(entry[1])
    lookup = {u['unit_id']:u for u in units}
    results = [acquisition.grade(job,lookup,rows,cfg,Receipts()) for job in spec['jobs']]
    usage = {}
    for key in panel.JUDGES:
        receipts = [(c,json.loads(r)) for c,r in conn.execute(
            'SELECT charged,result FROM calls WHERE model_key=?',(key,))]
        usage[key] = dict(calls=len(receipts), received=sum(r['status']=='received' for _,r in receipts),
                          ledger_usd=sum(c for c,_ in receipts)/1e9)
    conn.close()
    json_write(OUT/'usage.json',usage)
    by_id = {(row['unit_id'],row['judge'],row['variant']):row for row in results}
    coverage = {r['unit_id']:r for s in (OLD/'stage2_analysis/coverage.jsonl').read_text().splitlines()
                if (r:=json.loads(s))}
    suffix = defaultdict(dict)
    for s in json.loads((OLD/'suffix_sensitivity.json').read_text())['ending_audit']:
        suffix[s['unit_id']][s['position']-1] = s
    safe = []
    for uid,orig in sorted(coverage.items()):
        refs = orig['reference']
        ends = suffix[uid]
        for pos,slot in ends.items():
            assert refs[pos] == slot['native_reference']
        preds = {'original':orig['predictions']}
        statuses = {'original':orig['judge_status']}
        for arm in ARMS[1:]:
            preds[arm] = {j:by_id[uid,j,arm]['bits'] for pair in PAIRS for j in pair}
            statuses[arm] = {j:by_id[uid,j,arm]['status'] for pair in PAIRS for j in pair}
        safe.append(dict(group=hashlib.sha256(orig['cluster_id'].encode()).hexdigest()[:20],
                         block=[orig['generator'],orig['stratum']],native=refs,
                         literal=[ends[i]['literal_reference'] if i in ends else v for i,v in enumerate(refs)],
                         keep=[i for i in range(len(refs)) if i not in ends],
                         predictions=preds,status=statuses))
    assert len(safe)==len({r['group'] for r in safe})==62
    json_write(OUT/'paired_bits.json',safe)


def inspect(row, checker, arm, judge):
    truth = row['literal'] if checker=='literal' else row['native']
    bits = row['predictions'][arm][judge]
    idx = row['keep'] if checker=='without_ending' else range(len(truth))
    if not len(idx):
        raise ValueError('Empty conjunction')
    actual = all(truth[i] for i in idx)
    if bits is None:
        return actual,None
    if len(truth)!=len(bits):
        raise ValueError('Unaligned criteria')
    pred = all(bits[i] for i in idx)
    return actual,[int(pred and not actual),int(actual and not pred),int(pred),
                   sum(bits[i]==truth[i] for i in idx),len(idx)]


def bootstrap_indices(rows):
    blocks = defaultdict(list)
    for i,r in enumerate(rows):
        blocks[tuple(r['block'])].append(i)
    rng = np.random.default_rng(20261125)
    return np.column_stack([rng.choice(ix,(9999,len(ix))) for _,ix in sorted(blocks.items())])


def contrast(rows, checker, arm, pair, resamples):
    actual = [inspect(r,checker,arm,pair[0])[0] for r in rows]
    x,y = [np.array([inspect(r,checker,arm,j)[1] for r in rows]) for j in pair]
    delta = y-x
    boot = delta[resamples].mean(axis=1)*100
    totals = lambda a:dict(zip(('FP','FF','predicted_pass','criteria_correct','M'),map(int,a.sum(axis=0))))
    return dict(N=len(rows),ref_pass=sum(actual),small=totals(x),strong=totals(y),
                delta_per100=(delta[:,:2].mean(axis=0)*100).tolist(),
                paired95_per100=np.quantile(boot[:,:2],[.025,.975],axis=0).T.tolist(),
                verdict_discordances=int((x[:,2]!=y[:,2]).sum()))


def missing_bounds(rows,checker,arm,pair):
    sums = np.zeros((2,2),dtype=int)
    for r in rows:
        ref,a = inspect(r,checker,arm,pair[0])
        _,b = inspect(r,checker,arm,pair[1])
        for i,eligible in enumerate((not ref,ref)):
            small = [a[i]] if a is not None else [0,int(eligible)]
            strong = [b[i]] if b is not None else [0,int(eligible)]
            sums[i] += [min(strong)-max(small),max(strong)-min(small)]
    return sums.tolist()


def analyze(frame):
    historical = [r for r in frame if all(bits is not None for bits in r['predictions']['original'].values())]
    output = {'interval':'95% paired percentile across source groups within generator and condition-count blocks, 9999 draws',
              'N_attempted':len(frame),'N_original_common':len(historical),'families':{}}
    for pair in PAIRS:
        common = [r for r in historical if all(r['predictions'][a][j] is not None for a in ARMS for j in pair)]
        idx_old,idx = bootstrap_indices(historical),bootstrap_indices(common)
        obj = dict(N_common=len(common),coverage={},historical={},matched={},variation={},partial_identification={})
        for arm in ARMS:
            obj['coverage'][arm]={j:dict(Counter(r['status'][arm][j] for r in frame)) for j in pair}
        for checker in CHECKERS:
            obj['historical'][checker]=contrast(historical,checker,'original',pair,idx_old)
            obj['matched'][checker]={a:contrast(common,checker,a,pair,idx) for a in ARMS}
            obj['partial_identification'][checker]={a:missing_bounds(historical,checker,a,pair) for a in ARMS}
            obj['variation'][checker]={}
            for j in pair:
                base=np.array([inspect(r,checker,'original',j)[1] for r in common])
                obj['variation'][checker][j]={}
                for arm in ARMS[1:]:
                    again=np.array([inspect(r,checker,arm,j)[1] for r in common])
                    obj['variation'][checker][j][arm]=dict(changed_verdicts=int((base[:,2]!=again[:,2]).sum()),
                                                              delta_FP_FF=(again[:,:2]-base[:,:2]).sum(axis=0).tolist())
        output['families'][pair[0]]=obj
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-ledger',action='store_true')
    p.add_argument('--input',type=Path,default=OUT/'paired_bits.json')
    p.add_argument('--output',type=Path,default=OUT/'analysis.json')
    a=p.parse_args()
    if a.from_ledger:
        export()
    json_write(a.output,analyze(json.loads(a.input.read_text())))
