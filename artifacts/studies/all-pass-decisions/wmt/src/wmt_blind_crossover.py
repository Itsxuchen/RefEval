"""Run the predeclared WMT23 feasibility gate, seal, and held-out comparisons."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import wmt_blind_inputs as source
from src.conjunction_policy_study import make_order

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "artifacts/reports/wmt_blind_crossover_20261126"
OUT = INITIAL / "corrected"
CORRECTION = ROOT / "context/wmt_blind_source_issue_correction_20261126.md"
PLAN = ROOT / "context/wmt_blind_crossover_20261126.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store(path, data):
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Refusing to replace different evidence: {path}")
    else:
        with path.open("x") as f:
            f.write(encoded)


def read(path):
    return json.loads(path.read_text())


def split(inventory):
    def key(doc):
        return hashlib.sha256(("refeval-wmt23-first-pass-v1|" + doc).encode()).digest()
    docs = sorted(inventory, key=key)
    pivot = math.ceil(len(docs) / 3)
    return {"development": sorted(docs[:pivot]), "held": sorted(docs[pivot:])}


def paths(primary, auxiliary):
    n, k = primary.shape
    layout = dict(task_ids=np.arange(n), task_index=np.repeat(np.arange(n), k),
                  prediction=primary.ravel().astype(int),
                  secondary_predictions=auxiliary.reshape(1, -1).astype(int))
    masks = np.zeros((31, n*k), dtype=bool)
    for i in range(31):
        order = make_order(layout, "disagreement_then_random", "release", i+1)
        masks[i, order[:n*k//2]] = True
    return masks.reshape(31, n, k)


def policies(j, answers, masks):
    checked = masks[:, :, None, :]
    eager = np.where(checked, answers[None, :, :, :], j[None, :, None, :]).all(axis=3)
    any_failed = (checked & ~answers).any(axis=3)
    complete = checked.all(axis=3)
    original = j.all(axis=1)[None, :, None]
    gated = ~any_failed & (complete | original)
    assert not (gated & ~eager).any()
    return eager, gated


def frame(cache, docs):
    rows = sorted(source.load_units(cache, set(docs)), key=lambda r: (r.doc, r.sid))
    assert len({(r.doc, r.sid) for r in rows}) == len(rows)
    j = np.asarray([r.primary for r in rows], bool)
    aux = np.asarray([r.auxiliary for r in rows], bool)
    human = np.asarray([r.experts for r in rows], bool)
    assert human.shape == (len(rows), 3, 4) and j.shape == aux.shape == (len(rows), 4)
    masks = paths(j, aux)
    return rows, j, human, masks


def event_support(rows, j, experts, masks):
    h = experts.all(axis=2)
    initial = j.all(axis=1)[:, None]
    eager, gated = policies(j, experts, masks)
    events = {"up": (~initial & h).mean(axis=1), "down": (initial & ~h).mean(axis=1),
              "early": (eager & ~gated).mean(axis=(0, 2))}
    result = {}
    for name, freq in events.items():
        result[name] = {"outputs": int((freq > 0).sum()),
                        "docs": len({row.doc for row, value in zip(rows, freq) if value > 0}),
                        "output_equivalents": float(freq.sum())}
    return result


def unit_counts(j, experts, masks):
    h = experts.all(axis=2)
    peer = (h.sum(axis=1, keepdims=True) - h.astype(int)) / (h.shape[1]-1)
    original = j.all(axis=1)
    eager, gate = policies(j, experts, masks)
    early = eager & ~gate
    e = early.mean(axis=0)
    up = ~original[:, None] & h
    down = original[:, None] & ~h
    vals = {"N": np.ones(len(j)), "judge_pass": original.astype(float), "auditor_pass": h.mean(axis=1)}
    for name, decision in (("retain", original[:, None]), ("full", h),
                           ("eager", eager.mean(axis=0)), ("gated", gate.mean(axis=0))):
        vals[name+"_fp"] = (decision * (1-peer)).mean(axis=1)
        vals[name+"_ff"] = ((1.-decision)*peer).mean(axis=1)
    for key, event in (("up",up),("down",down),("early",e),("upheld",e*h),("withdrawn",e*~h)):
        vals[key] = event.mean(axis=1)
        vals[key+"_agree"] = (event * (1-peer if key == "down" else peer)).mean(axis=1)
    vals["timing_fp"] = (e * ~h * (1-peer)).mean(axis=1)
    vals["timing_ff"] = -(e * ~h * peer).mean(axis=1)
    np.testing.assert_allclose(vals["full_fp"]+vals["full_ff"]-vals["retain_fp"]-vals["retain_ff"],
                               vals["up"]+vals["down"]-2*(vals["up_agree"]+vals["down_agree"]),atol=1e-12)
    np.testing.assert_allclose(vals["gated_fp"]+vals["gated_ff"]-vals["eager_fp"]-vals["eager_ff"],
                               2*vals["early_agree"]-vals["early"],atol=1e-12)
    return vals


def measures(c):
    def pct(num, den):
        with np.errstate(invalid="ignore",divide="ignore"):
            return np.where(den>0,100*num/den,np.nan)
    n = c["N"]
    result = {k+"_per100":pct(c[k],n) for k in ("judge_pass","auditor_pass","up","down","early","withdrawn")}
    for k in ("up","down","early","upheld","withdrawn"):
        result[k+"_endorsement_pct"] = pct(c[k+"_agree"],c[k])
    result["complete_endorsement_pct"] = pct(c["up_agree"]+c["down_agree"],c["up"]+c["down"])
    result["early_minus_up_endorsement_pp"] = result["early_endorsement_pct"]-result["up_endorsement_pct"]
    result["withdrawn_fraction_early_pct"] = pct(c["withdrawn"],c["early"])
    for name in ("retain","full","eager","gated"):
        for kind in ("fp","ff"):
            result[name+"_"+kind+"_per100"] = pct(c[name+"_"+kind],n)
    for fp,ff in ((1,1),(3,1),(1,3)):
        suffix=f"_{fp}to{ff}_per100"
        for name in ("retain","full","eager","gated"):
            result[name+suffix]=pct(fp*c[name+"_fp"]+ff*c[name+"_ff"],n)
        result["full_minus_retain"+suffix]=result["full"+suffix]-result["retain"+suffix]
        result["gate_minus_eager"+suffix]=result["gated"+suffix]-result["eager"+suffix]
        result["same_early_minus_eventual"+suffix]=pct(fp*c["timing_fp"]+ff*c["timing_ff"],n)
    return result


def analyze(rows, values):
    docs = sorted({r.doc for r in rows})
    indices = np.array([docs.index(r.doc) for r in rows])
    keys = list(values)
    totals = np.stack([np.bincount(indices,weights=values[k],minlength=len(docs)) for k in keys],axis=1)
    draws = np.random.default_rng(20261125).multinomial(len(docs),np.ones(len(docs))/len(docs),size=1999)
    point = measures(dict(zip(keys,totals.sum(axis=0))))
    reps = measures(dict(zip(keys,(draws @ totals).T)))
    output = {}
    for key in point:
        valid=reps[key][np.isfinite(reps[key])]
        output[key] = {"point": float(point[key]) if np.isfinite(point[key]) else None,
                       "ci95":np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else None,
                       "defined": len(valid)}
    return output, {d:dict(zip(keys,totals[i])) for i,d in enumerate(docs)}


def code_hashes():
    files = [PLAN, CORRECTION, Path(__file__).resolve(), Path(source.__file__).resolve(),
             ROOT/"src/conjunction_policy_study.py"]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["develop","seal","held"])
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=OUT)
    a=parser.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    inv=source.inventory(a.cache)
    parts=split(inv)
    original_seal = INITIAL / "seal.json"
    if (digest(original_seal) != "66f3397c4434f2253f77f58bedb07d8dfc0754d8903d82714e71f69773f005e5"
        or read(original_seal)["split"] != parts):
        raise ValueError("Correction requires the original document split.")
    devfile,sealfile=a.out/"development.json",a.out/"seal.json"
    if a.phase=="develop":
        rows,j,h,m=frame(a.cache,parts["development"])
        assert len(rows)==sum(inv[d] for d in parts["development"])
        held_n=sum(inv[d] for d in parts["held"])
        supports=event_support(rows,j,h,m)
        for key in ("up","down"):
            supports[key]["projected_held_distinct"]=supports[key]["outputs"]*held_n/len(rows)
        go=all(v["outputs"]>=20 and v["docs"]>=5 for v in supports.values())
        go=go and all(supports[k]["output_equivalents"]>=10 and supports[k]["projected_held_distinct"]>=30 for k in ("up","down"))
        store(devfile,dict(split=parts,inventory=inv,dev_n=len(rows),held_n=held_n,
                           support=supports,passed=bool(go),protocol_sha256=digest(PLAN)))
    elif a.phase=="seal":
        dev=read(devfile)
        if not dev["passed"] or dev["split"]!=parts or dev["protocol_sha256"]!=digest(PLAN):
            raise ValueError("Cannot seal: feasibility or frozen protocol mismatch.")
        if sealfile.exists():
            raise FileExistsError(sealfile)
        store(sealfile,dict(time_utc=datetime.now(timezone.utc).isoformat(),
                            code=code_hashes(),development_sha256=digest(devfile),
                            source_sha256={k:digest(a.cache/k) for k in source.SOURCES},
                            original_seal_sha256=digest(original_seal),
                            status="mapping correction after original holdout read",
                            primary_cost=[1,1],sensitivity_costs=[[3,1],[1,3]],
                            data_inventory=inv,split=parts))
    else:
        seal=read(sealfile)
        if (seal["code"]!=code_hashes() or seal["development_sha256"]!=digest(devfile)
            or seal["data_inventory"]!=inv or seal["split"]!=parts or not read(devfile)["passed"]):
            raise ValueError("Seal mismatch before held read.")
        rows,j,h,m=frame(a.cache,parts["held"])
        assert len(rows)==sum(inv[d] for d in parts["held"])
        assert set(r.doc for r in rows)==set(parts["held"])
        # An empty and an exhaustive review reproduce both endpoint actions.
        for covered,target in ((False,j.all(axis=1)[:,None]),(True,h.all(axis=2))):
            e,g=policies(j,h,np.full((1,len(j),4),covered))
            assert (e==g).all() and (e[0]==target).all()
        numerical,groups=analyze(rows,unit_counts(j,h,m))
        store(a.out/"held.json",dict(seal_sha256=digest(sealfile),outputs=len(rows),documents=len(groups),
                                     checks_per_path=int(m[0].sum()),support=event_support(rows,j,h,m),
                                     estimates=numerical))
        store(a.out/"document_totals.json",groups)
    print(a.phase,devfile if a.phase=="develop" else sealfile if a.phase=="seal" else a.out/"held.json")

if __name__=="__main__":
    main()
