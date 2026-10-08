"""Charged, observable-only stratified controls for criterion microagreement.

The primary interval is conditional finite-population Bonferroni, not StratPPI.
All labels spent on the pilot count against the same fixed criterion cap. Saved
historical mix rows are read, never regenerated using newly modified sources.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.conjunction_policy_data import load_datasets
from src.conjunction_policy_study import rng_for
from src.conjunction_label_value import query_metrics
from src.finite_population_intervals import hypergeom_ci

SHARES = (.006, .05, .20, .50, .60, 1.0)
POLICIES = ("random_criterion", "proportional_stratified", "charged_pilot_neyman")
ROOT = Path(__file__).resolve().parents[1]


def validate_output(output, root):
    root=Path(root).resolve()
    output=Path(output)
    target=(output if output.is_absolute() else root/output).resolve()
    protected=[root/p for p in ("artifacts/expected","artifacts/figures","artifacts/validation",
                                "artifacts/reports/conjunction_label_value","artifacts/reports/conjunction_policy_study",
                                "artifacts/reports/conjunction_mechanism","expected","src","tests",
                                "data","context","docs","archive")]
    for path in protected:
        path=path.resolve()
        if target==path or path in target.parents or target in path.parents:
            raise ValueError("output overlaps source files or frozen scientific results")
    if root in target.parents and root/"artifacts" not in target.parents:
        raise ValueError("outputs inside the project must be under artifacts")
    if target.exists():
        if not target.is_dir():raise ValueError("output must be a directory")
        if any(p.is_symlink() for p in target.rglob("*")):
            raise ValueError("existing output contains symlinks; use a clean directory")
    return target


def integer_waterfill(total, weights, capacities, minimum=None):
    """Capacity-constrained proportional extras; largest remainders, index ties."""
    caps = np.asarray(capacities, dtype=int)
    weights = np.asarray(weights, dtype=float)
    low = np.zeros(len(caps), int) if minimum is None else np.asarray(minimum, dtype=int)
    if not (len(weights) == len(caps) == len(low) and np.all(low >= 0) and np.all(low <= caps)
            and low.sum() <= total <= caps.sum() and np.isfinite(weights).all() and np.all(weights >= 0)):
        raise ValueError("invalid allocation constraints")
    values = low.astype(float)
    remaining = int(total - low.sum())
    active = caps > low
    while remaining and active.any():
        indices = np.flatnonzero(active)
        w = weights[indices]
        if not w.sum():
            w = np.ones(len(indices))
        shares = remaining * w / w.sum()
        room = caps[indices] - values[indices]
        saturated = shares >= room
        if saturated.any():
            selected = indices[saturated]
            added = int(np.sum(caps[selected] - values[selected]))
            values[selected] = caps[selected]
            remaining -= added
            active[selected] = False
        else:
            values[indices] += shares
            remaining = 0
    result = np.floor(values).astype(int)
    extras = int(total - result.sum())
    candidates = np.flatnonzero(result < caps)
    ranked = sorted(candidates, key=lambda h: (-(values[h] - result[h]), h))
    for h in ranked[:extras]:
        result[h] += 1
    if result.sum() != total or np.any(result < low) or np.any(result > caps):
        raise AssertionError("integer allocation invariant failed")
    return result


def pilot_allocation(sizes, cap):
    """Return charged pilot sizes, or None when the specified design cannot fit."""
    sizes = np.asarray(sizes, dtype=int)
    base = np.minimum(2, sizes)
    remaining_layers = int((sizes > base).sum())
    if cap < int(base.sum()) + remaining_layers:
        return None
    target = min(max(int(base.sum()), int(np.floor(.2 * cap))), cap - remaining_layers)
    # Entire layers of size <=2 are already observed; retain one item elsewhere.
    capacities = np.where(sizes <= 2, sizes, sizes - 1)
    target = min(target, int(capacities.sum()))
    return integer_waterfill(target, sizes, capacities, base)


def _prefix(slots, priority, n):
    return slots[np.argsort(priority[slots], kind="stable")][:int(n)]


def select_design(prediction, cap, policy, seed, observe_pilot):
    """Selection API: has predictions and a charged pilot callback, never gold.

    The callback returns agreement bits for exactly the requested pilot indices.
    Formal remaining-stage labels are accessed later by the estimator/evaluator.
    """
    prediction = np.asarray(prediction)
    M = len(prediction)
    if not (M > 0 and policy in POLICIES and 0 <= cap <= M and np.isin(prediction, [0, 1]).all()):
        raise ValueError("invalid design")
    strata = [np.flatnonzero(prediction == p) for p in (0, 1) if np.any(prediction == p)]
    empty = np.array([], dtype=int)
    reason = ""
    effective = policy
    pilot = empty
    pilot_values = np.array([], dtype=int)
    if cap == M:
        return dict(pilot=empty, pilot_values=pilot_values, sample=np.arange(M),
                    strata=[np.arange(M)], sample_counts=np.array([M]), pilot_counts=np.array([0]), H=0,
                    effective_policy="census", fallback_reason="")
    sizes = np.array([len(s) for s in strata])
    if policy == "proportional_stratified" and cap < len(strata):
        effective, reason = "random_criterion", "cap below one sample per nonempty prediction stratum"
    if policy == "charged_pilot_neyman":
        pilot_counts = pilot_allocation(sizes, cap)
        if pilot_counts is None:
            effective, reason = "random_criterion", "cap cannot fund pilot plus one sample per remaining stratum"
    if effective == "random_criterion":
        sample = _prefix(np.arange(M), rng_for(seed, "criterion").random(M), cap)
        return dict(pilot=empty, pilot_values=pilot_values, sample=sample,
                    strata=[np.arange(M)], sample_counts=np.array([cap]), pilot_counts=np.array([0]), H=1,
                    effective_policy=effective, fallback_reason=reason)
    if policy == "proportional_stratified":
        pilot_counts=np.zeros(len(strata),dtype=int)
        counts = integer_waterfill(cap, sizes, sizes, np.ones(len(strata), int))
        priority = rng_for(seed, "stratified-proportional").random(M)
        sample = np.concatenate([_prefix(s, priority, n) for s, n in zip(strata, counts)])
    else:
        priority = rng_for(seed, "charged-neyman-pilot").random(M)
        pieces = [_prefix(s, priority, n) for s, n in zip(strata, pilot_counts)]
        pilot = np.concatenate(pieces)
        pilot_values = np.asarray(observe_pilot(pilot), dtype=int)
        if pilot_values.shape != pilot.shape or not np.isin(pilot_values, [0, 1]).all():
            raise ValueError("pilot callback must return aligned binary agreement bits")
        split_values = np.split(pilot_values, np.cumsum(pilot_counts)[:-1])
        rates = np.array([(v.sum() + .5) / (len(v) + 1) for v in split_values])
        used = np.zeros(M, bool); used[pilot] = True
        strata = [s[~used[s]] for s in strata]
        remaining_sizes = sizes - pilot_counts
        weights = remaining_sizes * np.sqrt(rates * (1 - rates))
        counts = integer_waterfill(cap-len(pilot), weights, remaining_sizes, (remaining_sizes > 0).astype(int))
        priority = rng_for(seed, "charged-neyman-remainder").random(M)
        sample = np.concatenate([_prefix(s, priority, n) for s, n in zip(strata, counts)])
    all_selected = np.r_[pilot, sample]
    if len(all_selected) != cap or len(np.unique(all_selected)) != cap:
        raise AssertionError("charged budget or no-repeat invariant failed")
    return dict(pilot=pilot, pilot_values=pilot_values, sample=sample, strata=strata,
                sample_counts=counts, pilot_counts=pilot_counts,
                H=sum(len(s)>n for s,n in zip(strata,counts)),
                effective_policy=effective, fallback_reason=reason)


def infer_conditional(M, pilot_correct, populations, H, alpha=.05):
    """populations contains (remaining N_h, sampled n_h, correct x_h)."""
    if H != sum(N>n for N,n,x in populations):
        raise ValueError("H must count the noncensused remaining strata")
    total = float(pilot_correct)
    lower = upper = int(pilot_correct)
    variance = 0.0
    variance_available = True
    estimate_available = True
    for N, n, x in populations:
        if not (0 <= x <= n <= N):
            raise ValueError("invalid conditional counts")
        if N == 0:
            continue
        lo, hi = (x,x) if n==N else hypergeom_ci(int(N), int(n), int(x), Fraction(str(alpha)) / H)
        lower += lo; upper += hi
        if not n:
            estimate_available = variance_available = False
            continue
        total += N * x / n
        if n == N:
            continue
        if n < 2:
            variance_available = False
        else:
            sample_variance = x * (n-x) / (n*(n-1))
            variance += N*N * (1-n/N) * sample_variance/n
    estimate = total/M if estimate_available else None
    normal_lo = normal_hi = None
    if variance_available and estimate_available:
        radius = float(norm.ppf(1-alpha/2)) * np.sqrt(variance) / M
        normal_lo, normal_hi = max(0.0, estimate-radius), min(1.0, estimate+radius)
    return dict(accuracy_estimate=estimate, accuracy_ci_lower=lower/M, accuracy_ci_upper=upper/M,
                accuracy_ci_width=(upper-lower)/M, normal_ci_lower=normal_lo, normal_ci_upper=normal_hi,
                normal_ci_width=None if normal_lo is None else normal_hi-normal_lo)


def evaluate(data, dataset, seed, share, policy):
    g, p = np.asarray(data["gold"]), np.asarray(data["prediction"])
    M = len(g); cap = int(round(share*M))
    design = select_design(p, cap, policy, seed, lambda idx: (g[idx] == p[idx]).astype(int))
    sample_mask = np.zeros(M, bool); sample_mask[design["sample"]] = True
    populations = []
    for strata in design["strata"]:
        idx = strata[sample_mask[strata]]
        populations.append((len(strata), len(idx), int((g[idx] == p[idx]).sum())))
    inference = infer_conditional(M, int(design["pilot_values"].sum()), populations, design["H"])
    indices = np.r_[design["pilot"], design["sample"]]
    metrics = query_metrics(data, indices)
    truth = float((g == p).mean())
    estimate = inference["accuracy_estimate"]
    return dict(dataset=dataset, seed=seed, task_order="release", budget_share=share,
                budget_cap=cap, policy=policy, effective_policy=design["effective_policy"],
                fallback_reason=design["fallback_reason"], n_criteria=M, n_tasks=len(data["task_ids"]),
                pilot_queries=len(design["pilot"]), main_queries=len(design["sample"]),
                pilot_correct=int(design["pilot_values"].sum()), H=design["H"],
                pilot_counts=json.dumps(design["pilot_counts"].tolist(),separators=(",",":")),
                main_counts=json.dumps(design["sample_counts"].tolist(),separators=(",",":")),
                conditional_counts=json.dumps(populations, separators=(",", ":")),
                query_indices_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                accuracy_truth=truth, accuracy_error=None if estimate is None else estimate-truth,
                accuracy_ci_covers=inference["accuracy_ci_lower"] <= truth <= inference["accuracy_ci_upper"],
                normal_ci_covers=None if inference["normal_ci_lower"] is None else bool(inference["normal_ci_lower"] <= truth <= inference["normal_ci_upper"]),
                inference_kind=("census" if not design["H"] else "conservative_conditional_95_Bonferroni" if design["H"] > 1 else "exact_conditional_95_SRS"),
                **inference, **metrics)


def summary(rows):
    out = []
    for keys, d in rows.groupby(["dataset", "policy", "budget_share"], sort=False):
        err = d.accuracy_error.dropna()
        out.append(dict(dataset=keys[0], policy=keys[1], budget_share=keys[2], seeds=len(d),
                        actual_queries=int(d.actual_queries.iloc[0]), n_criteria=int(d.n_criteria.iloc[0]),
                        bias=float(err.mean()), rmse=float(np.sqrt((err**2).mean())),
                        exact_ci_width_mean=float(d.accuracy_ci_width.mean()),
                        observed_coverage=float(d.accuracy_ci_covers.mean()),
                        approximate_normal_width_mean=float(d.normal_ci_width.mean()),
                        approximate_normal_observed_coverage=float(d.normal_ci_covers.dropna().astype(float).mean()),
                        approximate_normal_available=int(d.normal_ci_covers.notna().sum()),
                        criterion_errors_found_mean=float(d.criterion_errors_found.mean()),
                        certified_tasks_mean=float(d.certified_tasks.mean()), residual_errors_mean=float(d.residual_errors.mean()),
                        pilot_queries_mean=float(d.pilot_queries.mean()), fallback_count=int(d.fallback_reason.ne("").sum())))
    return pd.DataFrame(out)


def regress_saved_intervals(saved_budget, saved_tasks):
    """Reconstruct sufficient counts from saved estimates; no old query replay."""
    changes = []
    budget = pd.read_csv(saved_budget)
    for index, r in budget.iterrows():
        M, a, n = int(r.n_criteria), int(r.stageA_queries), int(r.stageR_queries)
        U = M-a
        correct = int(r.actual_queries-r.criterion_errors_found)
        if n == U or not n:
            # Census or logical bounds do not require an unidentified split.
            lo = correct/M
            hi = lo if n == U else (correct+U)/M
        else:
            x_real = (float(r.accuracy_estimate)*M-correct)/(U/n-1)
            x = int(round(x_real)); ca = correct-x
            if abs(x_real-x) > 1e-6 or not (0 <= x <= n and 0 <= ca <= a):
                raise AssertionError(f"Cannot reconstruct saved conditional counts at row {index}")
            lower, upper = hypergeom_ci(U,n,x)
            lo, hi = (ca+lower)/M,(ca+upper)/M
        oldlo, oldhi = float(r.accuracy_ci_lower),float(r.accuracy_ci_upper)
        if abs(lo-oldlo)>1e-12 or abs(hi-oldhi)>1e-12:
            changes.append(dict(source="budget_estimation_rows",row=int(index),dataset=r.dataset,seed=int(r.seed),
                                policy=r.policy,task_order=r.task_order,budget_share=float(r.budget_share),
                                old_lower=oldlo,old_upper=oldhi,new_lower=lo,new_upper=hi))
    tasks = pd.read_csv(saved_tasks)
    for index,r in tasks.iterrows():
        M,n,x=int(r.population_n),int(r.sample_n),int(r.sample_reference_pass)
        a,b=hypergeom_ci(M,n,x);lo,hi=a/M,b/M
        if abs(lo-float(r.task_pass_ci_lower))>1e-12 or abs(hi-float(r.task_pass_ci_upper))>1e-12:
            changes.append(dict(source="whole_task_sampling",row=int(index),dataset=r.dataset,seed=int(r.seed),
                                old_lower=float(r.task_pass_ci_lower),old_upper=float(r.task_pass_ci_upper),new_lower=lo,new_upper=hi))
    return {"budget_rows_checked":len(budget),"task_rows_checked":len(tasks),"changed_rows":len(changes),
            "changes":changes,"sufficient_count_reconstruction":"algebraic recovery from saved estimate and queried correct count; census/logical cases direct"}


def run(root, output, seeds=range(1,32), saved_reference=None, datasets=None):
    """Portable explicit-output driver; pass historical CSV paths for public replay."""
    root=Path(root).resolve()
    output=validate_output(output,root)
    seeds=list(seeds)
    if not seeds or len(seeds)!=len(set(seeds)) or any(s<1 for s in seeds):
        raise ValueError("use distinct positive seeds, excluding canonical seed0")
    saved_reference=saved_reference or {"budget":root/"artifacts/reports/conjunction_label_value/budget_estimation_rows.csv.gz",
                                        "tasks":root/"artifacts/reports/conjunction_label_value/whole_task_sampling.csv.gz"}
    saved_reference={k:Path(v).resolve() for k,v in saved_reference.items()}
    before={str(p.relative_to(root)) if p.is_relative_to(root) else p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in saved_reference.values()}
    started=time.perf_counter()
    datasets=load_datasets(root) if datasets is None else datasets
    records=[]
    for name,data in datasets.items():
        for seed in seeds:
            for share in SHARES:
                for policy in POLICIES:
                    records.append(evaluate(data,name,seed,share,policy))
        print(f"{name}: done, {time.perf_counter()-started:.2f}s",flush=True)
    rows=pd.DataFrame(records)
    if not (rows.actual_queries == rows.budget_cap).all() or not (rows.pilot_queries+rows.main_queries == rows.budget_cap).all():
        raise AssertionError("unequal charged label cost")
    old=pd.read_csv(saved_reference["budget"])
    old=old[old.dataset.isin(datasets) & old.seed.isin(seeds) & (old.task_order=="release")]
    old_srs=old[old.policy=="random_criterion"]
    joined=rows[rows.policy=="random_criterion"].merge(old_srs,on=["dataset","seed","budget_share"],suffixes=("_new","_old"),validate="one_to_one")
    if len(joined) != len(datasets)*len(seeds)*len(SHARES):
        raise AssertionError("incomplete saved SRS comparison")
    verified=["actual_queries","certified_tasks","residual_errors","criterion_errors_found","accuracy_estimate"]
    for field in verified:
        if not np.allclose(joined[field+"_new"],joined[field+"_old"],atol=1e-12,rtol=0):
            raise AssertionError("changed SRS control: "+field)
    regression=regress_saved_intervals(saved_reference["budget"],saved_reference["tasks"])
    output.mkdir(parents=True,exist_ok=True)
    rows.to_csv(output/"stratified_rows.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    summary(rows).to_csv(output/"stratified_summary.csv",index=False)
    old[old.policy=="sc50_then_srs"].to_csv(output/"saved_mix_comparator.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    (output/"interval_regression.json").write_text(json.dumps(regression,indent=2)+"\n")
    paths=[output/name for name in ["stratified_rows.csv.gz","stratified_summary.csv","saved_mix_comparator.csv.gz","interval_regression.json"]]
    after={str(p.relative_to(root)) if p.is_relative_to(root) else p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in saved_reference.values()}
    if before!=after:
        raise AssertionError("historical source changed")
    manifest={"rows":len(rows),"summary_rows":len(summary(rows)),"seeds":seeds,"shares":list(SHARES),"policies":list(POLICIES),
              "seconds":time.perf_counter()-started,"saved_srs_rows_verified":len(joined),"saved_srs_fields_verified":verified,
              "saved_source_sha256":before,"saved_sources_unchanged":True,
              "data_hashes":{name:data.get("source_hashes",{}) for name,data in datasets.items()},
              "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"src/finite_population_intervals.py",
                                  ROOT/"src/conjunction_label_value.py",ROOT/"src/conjunction_policy_study.py",ROOT/"src/conjunction_policy_data.py"]},
              "output_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "interval_regression_changed_rows":regression["changed_rows"],
              "scope":["Fixed-reference criterion microagreement; retrospective nine-cell development grid.",
                       "Primary interval: pointwise conservative conditional 95% Bonferroni hypergeometric; no direct StratPPI implementation.",
                       "Normal intervals share the same finite-population variance estimator and clipping for all methods; approximate, unavailable when a noncensus layer has n<2.",
                       "31-seed observed coverage, bias and RMSE are descriptive randomization summaries, not proof of coverage or population transfer.",
                       "Charged pilot q=(correct+0.5)/(n+1); deterministic constrained waterfill. Remaining labels never used for allocation.",
                       "Historical mix read only; no policy tuning, reference replacement, API calls or expert time claims."]}
    protocol=root/"context/conjunction_robustness_protocol.md"
    if protocol.exists():manifest["protocol_sha256"]=hashlib.sha256(protocol.read_bytes()).hexdigest()
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=31)
    args=parser.parse_args()
    print(json.dumps(run(args.root,args.output,range(1,args.seeds+1)),indent=2))
