"""Paired rater intervals and design-based estimator precision from numerical counts."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from src.cost_replay_20261125 import design_risk

root=Path(__file__).resolve().parents[1]
data=root/'inputs'
out=root/'generated'
out.mkdir(exist_ok=True)
with (data/'rater_groups.csv').open() as stream:
    groups=defaultdict(list)
    for row in csv.DictReader(stream):
        groups[row['policy'],row['audit'],row['score']].append(row)
draws=np.random.default_rng(20261125).multinomial(30,np.full(30,1/30),1999)
def get(key,metric):
    rows=sorted(groups[key],key=lambda r:int(r['cluster_index']))
    sizes=np.array([int(r['n_outputs']) for r in rows])
    sums=np.array([float(r[metric+'_seed_sum'])/int(r['seeds']) for r in rows])
    assert len(sizes)==30 and sum(sizes)==206
    return sums,sizes
def interval(sums,sizes):
    x=100*(draws@sums)/(draws@sizes)
    return [100*sum(sums)/sum(sizes),*np.quantile(x,[.025,.975]).tolist()]
results={}
for a,b in [('A','B'),('B','A')]:
    own,n=get(('disagreement_then_random',a,a),'gated_minus_eager')
    other,k=get(('disagreement_then_random',a,b),'gated_minus_eager')
    assert np.array_equal(n,k)
    for names,x in [(a+a,own),(a+b,other),(a+b+' minus '+a+a,other-own)]:
        results[names]=interval(x,n)
with (data/'pass_design.csv').open() as stream:
    design=list(csv.DictReader(stream))
for r in design:
    def parse(key):
        return list(map(int,r[key].split('/')))
    z=design_risk(parse('stratum_populations'),parse('stratum_pass_totals'),parse('stratum_samples'))
    for key,value in z.items():
        np.testing.assert_allclose(r[key] and float(r[key]),value,rtol=1e-11,atol=1e-12)
(out/'rater_intervals_and_designs.json').write_text(json.dumps({'rater':results,'n_designs':len(design),
    'lowest_exact_design_coverage':min(float(r['exact_95ci_coverage']) for r in design)},indent=2)+'\n')
