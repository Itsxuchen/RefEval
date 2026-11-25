"""Regenerate paper tables and budget curves from the accompanying numerical release."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from .judge_repeat_results import analyze
except ImportError:
    from judge_repeat_results import analyze


def records(path):
    with (gzip.open(path, 'rt') if path.suffix=='.gz' else path.open()) as f:
        yield from csv.DictReader(f)


def write_rows(out,name,rows):
    slash = chr(92)
    macro = slash+'newcommand{'+slash+name+'rows}{%'+chr(10)
    cells = [' & '.join(map(str, row))+' '+slash*2 for row in rows]
    (out/(name+'.tex')).write_text(macro+chr(10).join(cells)+chr(10)+'}'+chr(10))


def compute(data,out):
    out.mkdir(exist_ok=True,parents=True)
    analysis=analyze(json.loads((data/'paired_bits.json').read_text()))
    model=analysis['families']['gpt_mini']
    table=[]
    for fmt,key in [('Official','native'),('Literal suffix','literal'),('Omit suffix','without_ending')]:
        r=model['historical'][key]
        s,b=r['small'],r['strong']
        lo,hi=r['paired95_per100'][1]
        table.append([fmt,r['ref_pass'],f"{s['criteria_correct']} / {b['criteria_correct']}",f"{s['predicted_pass']} / {b['predicted_pass']}",
                      b['FP']-s['FP'],f"+{b['FF']-s['FF']}",f"[{lo:.1f}, {hi:.1f}]"])
    write_rows(out,'checker',table)
    table=[]
    for arm,label in [('original','Original'),('repeat','Exact repeat'),('alternate','Alternative')]:
        row=[label]
        for j in ('gpt_mini','claude_haiku'):
            r=analysis['families'][j]['matched']['native'][arm]
            s,b=r['small'],r['strong']
            row.extend([b['FP']-s['FP'],f"{b['FF']-s['FF']:+}",r['verdict_discordances']])
        table.append(row)
    write_rows(out,'repeat',table)
    names={'JB GPT-5.4':'JudgmentBench','AC DeepSeek v4-pro':'Agentic coding','DR Gemini 3.1 Pro':'Deep research'}
    cost=defaultdict(lambda:defaultdict(dict))
    for r in records(data/'cost_orders.csv.gz'):
        if r['denominator']=='full_census' and int(r['percent'])==20 and r['dataset'] in names:
            cost[r['dataset'],int(r['open_to_bit'])][r['policy']][int(r['seed'])]=100*int(r['eager_errors'])/int(r['unit_count'])
    tab=[]
    for name,label in names.items():
        frame=cost[name,15]
        a,b=[np.array([d[k] for k in sorted(d)]) for d in (frame['SC'],frame['DF'])]
        delta=a-b
        lo,hi=np.quantile(delta,[.025,.975])
        tab.append([label,f'{a.mean():.2f}',f'{b.mean():.2f}',f'{delta.mean():+.2f}',f'[{lo:+.2f}, {hi:+.2f}]'])
    write_rows(out,'cost',tab)
    all_curves=defaultdict(list)
    for r in records(data/'rater_orders.csv.gz'):
        if r['dataset']=='JB GPT-5.4' and r['policy']=='disagreement_then_random':
            all_curves[r['audit']+r['score'],float(r['budget_share'])].append(int(r['gated_minus_eager']))
    fig, ax=plt.subplots(figsize=(6.1,2.85),layout='constrained')
    for combo,color,style in [('AA','#154761','-'),('AB','#154761','--'),('BB','#b74c25','-'),('BA','#b74c25','--')]:
        x=sorted(t for k,t in all_curves if k==combo)
        y=[100*statistics.mean(all_curves[combo,t])/206 for t in x]
        assert len(x)>=101 and y[0]==y[-1]==0 and all(len(all_curves[combo,t])==31 for t in x)
        ax.plot(np.array(x)*100,y,style,color=color,lw=1.6,label=f'{combo[0]} audits, {combo[1]} scores')
    ax.axhline(0,color='.5',lw=.65);ax.set(xlabel='Share of rubric items checked (%)',ylabel='Extra disagreements from gating\nper 100 outputs',xlim=(0,100))
    ax.spines[['right','top']].set_visible(False);ax.legend(ncol=2,fontsize=8,frameon=False)
    fig.savefig(out/'rater_curve.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fc=list(records(data/'fold_forecasts.csv'))
    groups=defaultdict(list)
    for r in fc:
        if r['cost']=='1:1':
            groups[r['frame'],int(r['fold'])].append(r)
    table=[]
    short={'RuVerBench DR':'Deep research','RuVerBench AC':'Agentic coding','JudgmentBench full':'JudgmentBench','ProfBench Critical':'ProfBench Critical'}
    for name in dict.fromkeys(k[0] for k in groups):
        counts=np.zeros((3,2),dtype=int)
        units=0
        for f in range(5):
            choices=groups[name,f]
            # Export preserves the predeclared model ordering, including exact ties.
            units+=int(choices[0]['test_units'])
            for m,key in enumerate(('criterion_dev_error','empirical_dev_loss','predicted_dev_loss')):
                v=np.array([float(row[key]) for row in choices])
                row=choices[np.flatnonzero(np.isclose(v,v.min(),atol=1e-12,rtol=0))[0]]
                counts[m] += [int(row['observed_test_FP']),int(row['observed_test_FF'])]
        table.append([short.get(name,name),units,*[f'{100*(fp+ff)/units:.2f}' for fp,ff in counts],
                      *[f'{100*(fp-ff)/units:+.2f}' for fp,ff in counts]])
    write_rows(out,'selectors',table)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=Path('inputs'))
    p.add_argument('--output',type=Path,default=Path('generated'))
    a=p.parse_args()
    compute(a.inputs,a.output)
