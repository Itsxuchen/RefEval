"""Retrospective oracle-structure decomposition of saved verdict-update effects.

This model uses full fixed reference labels before query ordering. It is not a
policy, fitted predictor, new-environment forecast, or semantic reference audit.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from src.conjunction_policy_data import load_datasets

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("random_criterion", "disagreement_then_random")
NULL_CAPS = (.05, .2, .5, .84, .95)


@lru_cache(maxsize=200000)
def event_probability(G: int, h: int, r: int, f: int) -> float:
    """Uniform h-of-G: include r specified items and exclude f disjoint items."""
    if any(not isinstance(x, (int, np.integer)) for x in (G,h,r,f)) or min(G,h,r,f)<0 or h>G:
        raise ValueError("invalid group counts")
    if r+f>G or r>h or f>G-h:
        return 0.0
    value=1.0
    for j in range(r): value *= (h-j)/(G-j)
    for j in range(f): value *= (G-h-j)/(G-r-j)
    return value


def ordered_groups(data: dict, policy: str) -> tuple[np.ndarray, np.ndarray]:
    """Release task order; within each group all permutations are equiprobable."""
    p=np.asarray(data["prediction"]); ti=np.asarray(data["task_index"])
    m=len(p)
    if policy=="random_criterion": return np.zeros(m,dtype=int),np.array([m],dtype=int)
    if policy!="disagreement_then_random": raise ValueError("unsupported policy")
    secondary=np.asarray(data["secondary_predictions"])
    flags=np.any(secondary!=p[None,:],axis=0) if len(secondary) else np.zeros(m,bool)
    counts=np.bincount(ti[flags],minlength=len(data["task_ids"]))
    task_group=np.cumsum(counts>0)-1
    sizes=counts[counts>0].tolist()
    group=np.full(m,len(sizes),dtype=int)
    group[flags]=task_group[ti[flags]]
    if np.any(~flags):sizes.append(int((~flags).sum()))
    assert np.all(group>=0)
    return group,np.asarray(sizes,dtype=int)


def structure(data: dict) -> dict:
    g,p,ti=(np.asarray(data[k]) for k in ("gold","prediction","task_index"))
    n=len(data["task_ids"]);k=np.bincount(ti,minlength=n)
    gf=np.bincount(ti,weights=(g==0),minlength=n).astype(int)
    pf=np.bincount(ti,weights=(p==0),minlength=n).astype(int)
    cf=np.bincount(ti,weights=(g==0)&(p==0),minlength=n).astype(int)
    susceptible=(gf>0)&(pf>0)&(cf==0);ff=(gf==0)&(pf>0)
    return dict(k=k,gf=gf,pf=pf,correct_fail=cf,susceptible=susceptible,initial_ff=ff,
                snapshot=dict(n_units=n,n_criteria=len(g),reference_pass=int((gf==0).sum()),
                  predicted_pass=int((pf==0).sum()),initial_fp=int(((gf>0)&(pf==0)).sum()),
                  initial_ff=int(ff.sum()),susceptible=int(susceptible.sum()),
                  correct_fail_present=int((cf>0).sum()),criterion_errors=int((g!=p).sum())))


def _event_descriptor(ti, group, n, ngroups, required, forbidden):
    rmax=np.full(n,-1,dtype=int);fmin=np.full(n,ngroups,dtype=int)
    np.maximum.at(rmax,ti[required],group[required]);np.minimum.at(fmin,ti[forbidden],group[forbidden])
    rc=np.bincount(ti[required & (group==rmax[ti])],minlength=n)
    fc=np.bincount(ti[forbidden & (group==fmin[ti])],minlength=n)
    return rmax,fmin,rc,fc


def _event_sum(descriptor, sizes, cap: int, eligible) -> float:
    ends=np.cumsum(sizes);active=int(np.searchsorted(ends,cap,side="right"))
    G=int(sizes[active]) if active<len(sizes) else 0
    h=cap-(int(ends[active-1]) if active else 0)
    rmax,fmin,rc,fc=descriptor
    valid=eligible & (rmax<=active) & (fmin>=active)
    r=np.where(rmax[valid]==active,rc[valid],0)
    f=np.where(fmin[valid]==active,fc[valid],0)
    if not len(r):return 0.0
    pairs,counts=np.unique(np.column_stack((r,f)),axis=0,return_counts=True)
    return sum(int(c)*event_probability(G,h,int(rr),int(ff)) for (rr,ff),c in zip(pairs,counts))


def expected_curve(data: dict, policy: str, shares=tuple(np.arange(101)/100)) -> list[dict]:
    """E[introduced FP], E[delayed FF], and E[gated-eager] over query ordering."""
    g,p,ti=(np.asarray(data[k]) for k in ("gold","prediction","task_index"))
    st=structure(data);n=len(data["task_ids"]);m=len(g);groups,sizes=ordered_groups(data,policy)
    empty=np.zeros(m,bool)
    intro=_event_descriptor(ti,groups,n,len(sizes),p==0,g==0)
    repaired=_event_descriptor(ti,groups,n,len(sizes),p==0,empty)
    complete=_event_descriptor(ti,groups,n,len(sizes),np.ones(m,bool),empty)
    result=[]
    for share in shares:
        if not 0<=float(share)<=1:raise ValueError("share outside [0,1]")
        cap=int(round(float(share)*m))
        A=_event_sum(intro,sizes,cap,st["susceptible"])
        repaired_ff=_event_sum(repaired,sizes,cap,st["initial_ff"])
        complete_ff=_event_sum(complete,sizes,cap,st["initial_ff"])
        D=repaired_ff-complete_ff
        if D < -1e-10:raise AssertionError("negative delayed repairs")
        D=max(D,0.)
        result.append(dict(policy=policy,task_order="release",budget_share=float(share),
             budget_cap=cap,expected_introduced_fp=A,expected_delayed_ff=D,expected_delta=D-A))
    return result


def permutation_strata(data: dict) -> list[np.ndarray]:
    """Preserve confusion margins within task length, primary prediction, category."""
    ti=np.asarray(data["task_index"]);k=np.bincount(ti)
    strata=defaultdict(list)
    for j,(task,p,c) in enumerate(zip(ti,data["prediction"],data["category"])):
        strata[(int(k[task]),int(p),str(c))].append(j)
    return [np.asarray(v,dtype=int) for _,v in sorted(strata.items())]


def permute_gold(data: dict, rng: np.random.Generator, strata=None) -> dict:
    strata=permutation_strata(data) if strata is None else strata
    original=np.asarray(data["gold"]);gold=original.copy()
    for indices in strata: gold[indices]=rng.permutation(original[indices])
    return {**data,"gold":gold}


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _sha(path: Path):return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root: Path=ROOT, output: Path|None=None, permutations: int=64, seed: int=20261007,
            preflight: bool=False) -> dict:
    root=root.resolve();output=output or root/"artifacts/reports/conjunction_mechanism"
    if permutations<1:raise ValueError("positive permutation count required")
    from src.reproduce import validate_output
    output=validate_output(Path(output),root)
    source=root/"artifacts/expected/conjunction_policy_study/budget_curves.csv.gz"
    datasets=load_datasets(root)
    protocol=root/"context/conjunction_mechanism_protocol.md"
    preserved={str(source.relative_to(root)):_sha(source),str(protocol.relative_to(root)):_sha(protocol)}
    for relative in ("src/conjunction_policy_data.py", "src/conjunction_policy_study.py"):
        preserved[relative]=_sha(root/relative)
    for data in datasets.values():preserved.update(data["source_hashes"])
    start=time.perf_counter();observed=defaultdict(list);canonical={}
    with gzip.open(source,"rt") as f:
        for r in csv.DictReader(f):
            if r["task_order"]!="release" or r["policy"] not in POLICIES:continue
            key=(r["dataset"],r["policy"],float(r["budget_share"]))
            value={"seed":int(r["seed"]),"introduced_fp":int(r["introduced_fp"]),
                   "delayed_ff":int(r["delayed_ff_corrections"]),
                   "delta":int(r["gated_residual_errors"])-int(r["residual_errors"])}
            if value["seed"]==0:canonical[key]=value
            else:observed[key].append(value)
    analytic=[];snapshots={};null_rows=[]
    if preflight:datasets={"JB GPT-5.4":datasets["JB GPT-5.4"]}
    for name,data in datasets.items():
        snapshots[name]=structure(data)["snapshot"]
        for policy in POLICIES:
            for r in expected_curve(data,policy):
                key=(name,policy,r["budget_share"]);obs=observed[key]
                assert sorted(x["seed"] for x in obs)==list(range(1,32))
                r={"dataset":name,**r,"randomized_orders":31}
                for field in ("introduced_fp","delayed_ff","delta"):
                    v=np.array([x[field] for x in obs],dtype=float)
                    r["observed_mean_"+field]=float(v.mean())
                    r["order_mcse_"+field]=float(v.std(ddof=1)/math.sqrt(len(v)))
                    r["mean_minus_expected_"+field]=float(v.mean()-r["expected_"+field])
                    r["canonical_seed0_"+field]=canonical[key][field]
                analytic.append(r)
        strata=permutation_strata(data)
        movable=[ids for ids in strata if 0<int(data["gold"][ids].sum())<len(ids)]
        snapshots[name].update(permutation_strata=len(strata),movable_strata=len(movable),
          degenerate_strata=len(strata)-len(movable),movable_criterion_slots=sum(map(len,movable)))
        word=int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],"little")
        rng=np.random.default_rng(np.random.SeedSequence([seed,word]))
        for rep in range(permutations):
            changed=permute_gold(data,rng,strata);stats=structure(changed)["snapshot"]
            for indices in strata:
                assert int(changed["gold"][indices].sum())==int(data["gold"][indices].sum())
            for policy in POLICIES:
                for r in expected_curve(changed,policy,NULL_CAPS):
                    null_rows.append(dict(dataset=name,permutation=rep,**r,
                      reference_pass=stats["reference_pass"],initial_ff=stats["initial_ff"],
                      susceptible=stats["susceptible"]))
        print(f"{name}: analytical grid and {permutations} permutations; {time.perf_counter()-start:.2f}s",flush=True)
    summary=[];nullgroups=defaultdict(list)
    for r in null_rows:nullgroups[(r["dataset"],r["policy"],r["budget_share"])].append(r)
    actual={(r["dataset"],r["policy"],r["budget_share"]):r for r in analytic}
    for key,rs in sorted(nullgroups.items()):
        a=actual[key];out=dict(dataset=key[0],policy=key[1],budget_share=key[2],permutations=len(rs))
        for field in ("expected_introduced_fp","expected_delayed_ff","expected_delta","reference_pass","initial_ff","susceptible"):
            v=np.array([r[field] for r in rs],dtype=float)
            out["original_"+field]=a[field] if field.startswith("expected_") else snapshots[key[0]][field]
            for label,value in (("mean",v.mean()),("median",np.median(v)),("q05",np.quantile(v,.05)),("q95",np.quantile(v,.95)),("min",v.min()),("max",v.max())):out["permutation_"+field+"_"+label]=float(value)
        v=np.array([r["expected_delta"] for r in rs],dtype=float)
        original=a["expected_delta"]
        out["opposite_nonzero_sign_count"]=int((((v < -1e-12) & (original > 1e-12)) | ((v > 1e-12) & (original < -1e-12))).sum())
        out["null_positive_count"]=int((v>1e-12).sum())
        out["null_zero_count"]=int((np.abs(v)<=1e-12).sum())
        out["null_negative_count"]=int((v < -1e-12).sum())
        summary.append(out)
    unchanged=all(_sha(root/p)==h for p,h in preserved.items())
    if not unchanged:raise AssertionError("original input or saved scientific result changed")
    manifest=dict(kind="retrospective oracle-structure explanation and counterfactual task-link sensitivity",
      policies=list(POLICIES),task_order="release",seeds_compared=list(range(1,32)),canonical_seed0_separate=True,
      analytic_rows=len(analytic),null_rows=len(null_rows),null_summary_rows=len(summary),permutations=permutations,
      permutation_seed=seed,null_caps=list(NULL_CAPS),sign_tolerance=1e-12,permutation_strata=["task_length","primary_prediction","criterion_category"],
      conditioning="All fixed gold labels, predictions, task positions and policy groups known before order averaging. No fitted parameters.",
      interpretation="MCSE describes 31-order sampling discrepancy, not new-task uncertainty. No pass threshold. Permutation quantiles are not confidence intervals or a semantic-noise model; permuting gold changes reference task-pass rates and is not isolated correlation causality.",
      source_hashes=preserved,preserved_sources_unchanged=unchanged,seconds=time.perf_counter()-start)
    if preflight:return manifest
    output.mkdir(parents=True,exist_ok=True)
    _write_csv(output/"update_expected_grid.csv",analytic)
    _write_csv(output/"update_permutation_rows.csv",null_rows)
    _write_csv(output/"update_permutation_summary.csv",summary)
    (output/"update_structures.json").write_text(json.dumps(snapshots,indent=2)+"\n")
    manifest["implementation_sha256"]=_sha(Path(__file__))
    manifest["output_sha256"]={p.name:_sha(p) for p in sorted(output.glob("update*")) if p.name!="update_manifest.json" and p.is_file()}
    (output/"update_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations",type=int,default=64)
    parser.add_argument("--seed",type=int,default=20261007)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    print(json.dumps(analyze(permutations=args.permutations,seed=args.seed,preflight=args.preflight),indent=2))

if __name__=="__main__":main()
