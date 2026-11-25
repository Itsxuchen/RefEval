"""Score the development choices on paired held articles, preserving article clusters."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from render_decision_study import write_rows

root=Path(__file__).resolve().parents[1]
data=root/'inputs'
out=root/'generated'
seal=json.loads((data/'external_seal.json').read_text())
with (data/'external_groups.csv').open() as stream:
    rows=list(csv.DictReader(stream))
table=[]
audit=[]
for name,dev in seal['panels'].items():
    selected=[r for r in rows if r['panel']==name]
    gids=sorted({r['group'] for r in selected})
    models=['G-Eval GPT-4','UniEval']
    assert len(gids)==70
    by_model=defaultdict(dict)
    for row in selected:
        by_model[row['model']][row['group']]=[int(row[f]) for f in ('test_units','task_FP','task_FF','reference_passes')]
    values=np.array([[by_model[m][g] for g in gids] for m in models])
    assert np.array_equal(values[0,:,0],values[1,:,0]) and values[0,:,0].sum()==1120
    for cost in ('1:1','3:1','1:3'):
        choices=dev['costs'][cost]['choices']
        cf,ce=map(int,cost.split(':'))
        losses=cf*values[:,:,1]+ce*values[:,:,2]
        group_size=values[0,:,0]
        keys=('criterion','empirical_task','independent_task')
        idx=[models.index(choices[k]) for k in keys]
        avg=100*losses.sum(axis=1)/group_size.sum()
        rng=np.random.default_rng(20261125)
        draw=rng.integers(0,len(gids),size=(1999,len(gids)))
        dif=losses[idx[1]]-losses[idx[0]]
        b=100*dif[draw].sum(axis=1)/group_size[draw].sum(axis=1)
        const_fail_loss=ce*values[0,:,3]
        versus_const=losses[idx[1]]-const_fail_loss
        c=100*versus_const[draw].sum(axis=1)/group_size[draw].sum(axis=1)
        audit.append(dict(panel=name,cost=cost,choices=choices,task_minus_criterion=float(avg[idx[1]]-avg[idx[0]]),
                          paired95=np.quantile(b,[.025,.975]).tolist(),
                          observed_all_fail_loss=float(const_fail_loss.sum()),
                          observed_all_pass_loss=float(cf*(group_size-values[0,:,3]).sum()),
                          task_selection_minus_all_fail_per100=float(100*versus_const.sum()/group_size.sum()),
                          task_selection_minus_all_fail95=np.quantile(c,[.025,.975]).tolist()))
        lab='Fixed' if name=='primary' else 'Calibrated'
        table.append([lab,cost,*[('U' if i else 'G')+f' ({avg[i]:.2f})' for i in idx]])
write_rows(out,'external',table)
(out/'external_check.json').write_text(json.dumps(audit,indent=2)+'\n')
