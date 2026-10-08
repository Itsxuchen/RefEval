"""Render publication figures from the separately specified robustness tables."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/refeval-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CELLS=('JB GPT-5.4','JB GPT-5.4 | binary_only','DR Gemini 3.1 Pro')
TITLES=('JudgmentBench · full strict','JudgmentBench · binary-only','RuVerBench · DR Gemini')
BLUE,ORANGE,GRAY,PURPLE='#0072B2','#D55E00','#738390','#7A5195'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def render(source,output):
 source,output=Path(source).resolve(),Path(output).resolve()
 if source==output or source in output.parents or output in source.parents:raise ValueError('separate figure and source directories required')
 for folder in ('src','tests','data','context','docs','.git','artifacts/expected','artifacts/validation','artifacts/reports'):
  protected=(ROOT/folder).resolve()
  if output==protected or protected in output.parents or output in protected.parents:raise ValueError('figure output overlaps protected files')
 if output.exists() and any(p.is_symlink() for p in output.rglob('*')):raise ValueError('output contains symlinks')
 names=['cluster/summary.json','reference/rq1_summary.csv','reference/rq2_summary.csv','reference/frame_diagnostics.json','estimation/stratified_summary.csv']
 before={name:sha(source/name) for name in names}
 cluster=json.loads((source/names[0]).read_text());r1=pd.read_csv(source/names[1]);r2=pd.read_csv(source/names[2]);e=pd.read_csv(source/names[4])
 output.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
 products=[];plot_rows=[]
 def save(fig,name):
  for ext in ('png','pdf'):
   p=output/f'{name}.{ext}';fig.savefig(p,dpi=180,bbox_inches='tight');products.append(p)
  plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(13.2,7),sharex=True,sharey='row',layout='constrained')
 x=100*np.array(cluster['budget_shares'])
 for col,(cell,title) in enumerate(zip(CELLS,TITLES)):
  q=next(v for v in cluster['rq1'] if v['dataset']==cell and v['policy']=='disagreement_then_random')
  for row,unit in enumerate(('count','per100')):
   a=axes[row,col];s=q[unit]
   a.axhline(0,color=GRAY,lw=.8);a.fill_between(x,s['pointwise_q025'],s['pointwise_q975'],color=BLUE,alpha=.17,label='95% pointwise composition range')
   a.plot(x,s['original'],color=BLUE,lw=2,label='Original exact order expectation')
   a.grid(axis='y',alpha=.16);a.set_xlim(0,100)
   if row==0:a.set_title(f'{title}\nOriginal N = {q["n_units"]:,} evaluation units')
   if row==1:a.set_xlabel('Available criterion budget (%)')
   for j,b in enumerate(x):plot_rows.append(dict(dataset=cell,unit=unit,budget_percent=b,original=s['original'][j],pointwise_lower=s['pointwise_q025'][j],pointwise_upper=s['pointwise_q975'][j],simultaneous_lower=s['simultaneous_grid_lower'][j],simultaneous_upper=s['simultaneous_grid_upper'][j]))
 axes[0,0].set_ylabel('Gated − eager\nresidual disagreements (count)');axes[1,0].set_ylabel('Gated − eager\nper 100 evaluation units')
 h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=9)
 fig.suptitle('Update effects across task compositions · disagreement then random',fontsize=16)
 save(fig,'conjunction_robustness_budget')
 fig,axes=plt.subplots(2,2,figsize=(10.4,7.1),sharex='row',layout='constrained')
 for col,cell in enumerate(CELLS[:2]):
  a=axes[0,col];q=r1[(r1.dataset==cell)&(r1.policy=='disagreement_then_random')]
  for ref,color in [('A',BLUE),('B',ORANGE)]:
   z=q[(q.acquisition_ref==ref)&(q.evaluation_ref==ref)].sort_values('budget_share');a.plot(z.budget_share*100,z.gated_minus_eager_per100_mean,color=color,lw=2,label=f'Reference {ref}')
  a.axhline(0,color=GRAY,lw=.8);a.set_title(TITLES[col]+' · fixed 206 outputs');a.set_ylabel('Gated − eager per 100 outputs');a.grid(axis='y',alpha=.16);a.legend(frameon=False)
  a=axes[1,col];q=r2[r2.dataset==cell]
  for acq,ev,color,ls in [('A','A',BLUE,'-'),('B','B',ORANGE,'-'),('A','B',ORANGE,'--'),('B','A',BLUE,'--')]:
   z=q[(q.acquisition_ref==acq)&(q.evaluation_ref==ev)].sort_values('budget_share');a.plot(z.budget_share*100,z.eager_errors_delta_per100_mean,color=color,ls=ls,marker='o',ms=3,label=f'Acquire {acq}, re-read {ev}')
  a.axhline(0,color=GRAY,lw=.8);a.set_ylabel('Mix − SRS residual disagreements\nper 100 outputs');a.set_xlabel('Available criterion budget (%)');a.grid(axis='y',alpha=.16)
 h,l=axes[1,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=4,fontsize=9)
 fig.suptitle('Changing the reference while holding judge predictions fixed',fontsize=15)
 save(fig,'conjunction_robustness_reference')
 # Show interval construction and point-estimator error separately for all nine cells.
 fig,axes=plt.subplots(1,2,figsize=(13.4,9),sharey=True,layout='constrained')
 pols=['random_criterion','proportional_stratified','charged_pilot_neyman']
 labels=['SRS','Proportional strata','Charged-pilot Neyman']
 cells=['DR Gemini 3.1 Pro','DR Qwen-plus','DR GPT-5.4 low','AC Qwen-plus','AC DeepSeek v4-pro','JB GPT-5.4','JB GPT-5.4-mini','JB GPT-5.4 | binary_only','JB GPT-5.4-mini | binary_only']
 display=['DR · Gemini 3.1 Pro','DR · Qwen-plus','DR · GPT-5.4 low','AC · Qwen-plus','AC · DeepSeek v4-pro','JB full · GPT-5.4','JB full · GPT-5.4-mini','JB binary · GPT-5.4','JB binary · GPT-5.4-mini']
 y=np.arange(len(cells));bar_height=.23
 q=e[e.budget_share==.2].set_index(['dataset','policy'])
 for a,metric,title in zip(axes,['exact_ci_width_mean','rmse'],['Mean conservative 95% interval width','Empirical RMSE across 31 query orders']):
  for k,(pol,label,color) in enumerate(zip(pols,labels,[GRAY,BLUE,ORANGE])):
   values=np.array([q.loc[(cell,pol),metric]*100 for cell in cells])
   bars=a.barh(y+(k-1)*bar_height,values,height=bar_height,color=color,label=label)
   a.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
  vmax=max(q.loc[(cell,pol),metric]*100 for cell in cells for pol in pols)
  a.set_xlim(0,vmax*1.2);a.set_xticks(np.linspace(0,15 if metric=='exact_ci_width_mean' else 2.5,6))
  a.set_title(title);a.set_xlabel('Percentage points')
  a.grid(axis='x',alpha=.16);a.set_axisbelow(True)
  for cut in (2.5,4.5,6.5):a.axhline(cut,color=GRAY,lw=.6,alpha=.4)
 axes[0].set_yticks(y,display);axes[0].invert_yaxis()
 h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,fontsize=10)
 fig.suptitle('Accuracy estimation at the same 20% label budget',fontsize=16)
 save(fig,'conjunction_robustness_estimation')
 p=output/'conjunction_robustness_budget_data.csv';pd.DataFrame(plot_rows).to_csv(p,index=False);products.append(p)
 assert before=={name:sha(source/name) for name in names}
 receipt={'source_sha256':before,'renderer_sha256':sha(Path(__file__)),'output_sha256':{p.name:sha(p) for p in products},'budget_figure':'Disagreement-then-random, all 101 caps. 399 empirical base-task resamples; RQ1 order expectation calculated exactly. Curated-task composition sensitivity, not automatic population confidence. Common vertical scales within each row. Only pointwise composition ranges are displayed; the historical constant-width simultaneous band remains in the source tables and plotting CSV.','reference_figure':'Same 206 output units and fixed primary/auxiliary predictions. Mean of 31 paired orders. Dashed RQ2 curves use fixed acquired positions and re-read queried labels under the scoring reference.','estimation_figure':'All nine cells at a matched 20% query cap. Left: mean conservative conditional 95% hypergeometric interval width, with Bonferroni combination for strata. Right: sqrt(mean squared estimation error) across 31 query orders, not a population performance guarantee. Pilot charged. The two metrics assess interval construction and point estimation separately.'}
 (output/'conjunction_robustness_render_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))
 return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'artifacts/expected/conjunction_robustness');p.add_argument('--output',type=Path,default=ROOT/'artifacts/figures');a=p.parse_args();render(a.source,a.output)
