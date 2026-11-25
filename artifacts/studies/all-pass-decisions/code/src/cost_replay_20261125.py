"""Paired finite-frame audit replay and guaranteed-budget unit sampling."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from src.conjunction_policy_data import ROOT, load_datasets, read_csv
from src.conjunction_policy_study import FIELDS, initial_state, metadata, replay, rng_for
from src.finite_population_intervals import _exact_tail_accept

POLICIES = {"SC": "sc_judge", "DF": "disagreement_then_random", "Random": "random_criterion"}
RATIOS = (0, 1, 5, 10, 15, 20)
PERCENT = (5, 20, 50, 100)
SEEDS = tuple(range(1, 32))
OUT = ROOT / "artifacts/reports/cost_replay_20261125"


def csv_out(path, rows):
    # Store a stable gzip header so independent reruns produce byte-identical files.
    with path.open("wb") as raw:
        stream = gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) if path.suffix == ".gz" else raw
        with io.TextIOWrapper(stream, newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


def timing_anchor():
    annotations = read_csv(ROOT / "data/raw/judgmentbench/human_annotations_rubric.csv")
    scores = read_csv(ROOT / "data/raw/judgmentbench/human_rubric_item_scores.csv")
    size = Counter(row["annotation_id"] for row in scores)
    assert {r["annotation_id"] for r in annotations} == size.keys()
    k = np.array([size[r["annotation_id"]] for r in annotations])
    t = np.array([float(r["time_spent_seconds"]) for r in annotations])
    valid = np.isfinite(t) & (t > 0)
    settings = []
    for label, keep in (("all_positive", valid), ("positive_at_most_1800_seconds", valid & (t <= 1800))):
        kt, tt = k[keep], t[keep]
        slope, intercept = np.polyfit(kt, tt, 1)
        settings.append(dict(records=label, units=len(kt), bits=int(sum(kt)), total_seconds=float(tt.sum()),
                             median_seconds=float(np.median(tt)), mean_seconds=float(tt.mean()),
                             smallest_seconds=float(tt.min()), largest_seconds=float(tt.max()),
                             distinct_raters=len({r["annotator_id"] for r, use in zip(annotations, keep) if use}),
                             slope_seconds_per_item=float(slope), intercept_seconds=float(intercept),
                             descriptive_intercept_over_slope=float(intercept/slope),
                             r_squared=float(np.corrcoef(kt, tt)[0, 1] ** 2)))
    scales = []
    for z in settings:
        for ratio in RATIOS:
            bit = z["total_seconds"] / (z["bits"] + ratio*z["units"])
            scales.append(dict(records=z["records"], open_to_bit=ratio, seconds_per_bit=bit,
                               seconds_per_open=ratio*bit))
    return dict(source="data/raw/judgmentbench/human_annotations_rubric.csv", total_rows=len(t),
                dropped_nonpositive_or_missing=int((~valid).sum()), settings=settings, calibrated_scenarios=scales)


def budget(m, n, ratio, pct, denominator="full_census"):
    census = m + ratio*n if denominator == "full_census" else m
    return int(census*pct//100)


def cost_path(trace, ratio):
    touched = np.r_[0, np.cumsum(trace["deltas"][:, FIELDS.index("tasks_touched")])]
    labels = np.arange(len(touched))
    return labels + ratio*touched


def prefixes(data, trace, policy, seed):
    n, m = len(data["task_ids"]), len(data["gold"])
    initial = trace["baseline_by_task"].sum(axis=0)
    values = np.vstack([initial, initial + np.cumsum(trace["deltas"], axis=0)])
    by_field = {name: values[:, i] for i, name in enumerate(FIELDS)}
    rows = []
    for ratio in RATIOS:
        spend = cost_path(trace, ratio)
        for denom in ("full_census", "fixed_bit_equivalents"):
            for pct in PERCENT:
                cap = budget(m, n, ratio, pct, denom)
                used = int(np.searchsorted(spend, cap, side="right")-1)
                state = {key: int(val[used]) for key, val in by_field.items()}
                efp, eff = state["residual_fp"], state["residual_ff"]
                gfp = int(initial[FIELDS.index("residual_fp")])-state["fixed_initial_fp"]
                gff = int(initial[FIELDS.index("residual_ff")])-state["certified_initial_ff"]
                if policy == "SC" and abs((gfp+gff)-(efp+eff)) > 1:
                    raise AssertionError("serial update gap exceeded one unit")
                if pct == 100 and denom == "full_census" and (state["certified_tasks"] != n or efp+eff+gfp+gff):
                    raise AssertionError("full-cost endpoint did not certify all units")
                rows.append(dict(dataset=data["name"], seed=seed, policy=policy, open_to_bit=ratio,
                                 denominator=denom, percent=pct, unit_count=n, label_count=m, cap=cap,
                                 spent=int(spend[used]), slack=cap-int(spend[used]), opened=state["tasks_touched"],
                                 labels=used, certified=state["certified_tasks"], bit_errors_found=state["criterion_errors_found"],
                                 eager_fp=efp, eager_ff=eff, eager_errors=efp+eff,
                                 gated_fp=gfp, gated_ff=gff, gated_errors=gfp+gff,
                                 gated_minus_eager=gfp+gff-efp-eff))
    return rows


def sample_sizes(strata_sizes, n):
    """Proportional two-stratum allocation, with at least one per populated stratum."""
    if len(strata_sizes) == 1:
        return (n,)
    n0, n1 = strata_sizes
    if n < 2:
        raise ValueError("Both strata must be sampled")
    first = min(max((2*n*n0+n0+n1)//(2*(n0+n1)), 1, n-n1), n0, n-1)
    return first, n-first


def guaranteed_allocation(lengths, ratio, cap):
    """Largest fixed total n under this allocation rule with all possible draws affordable."""
    n_pop = tuple(len(a) for a in lengths)
    worst = [np.r_[0, np.cumsum(np.sort(a)[::-1]+ratio)] for a in lengths]
    low, high = len(n_pop)-1, sum(n_pop)+1
    while high-low > 1:
        mid = (low+high)//2
        sizes = sample_sizes(n_pop, mid)
        if sum(cost[n] for cost, n in zip(worst, sizes)) <= cap:
            low = mid
        else:
            high = mid
    if low < len(n_pop):
        return None
    sizes = sample_sizes(n_pop, low)
    return sizes, int(sum(cost[n] for cost, n in zip(worst, sizes)))


@lru_cache(maxsize=None)
def interval_table(N, n, h):
    """Equal-tailed hypergeometric inversion; Bonferroni denominator h."""
    obs = np.arange(n+1)
    if n == N:
        return obs, obs
    threshold = Fraction(1,40*h)
    alpha_tail = float(threshold)

    def boundary(lower):
        left = np.full(n+1, -1, dtype=int)
        right = np.full(n+1, N+1, dtype=int)
        while (right-left).max() > 1:
            k = (left+right)//2
            k = np.clip(k, 0, N)
            # First total meeting the lower-tail test, or first violating the upper bound.
            probs = hypergeom.sf(obs-1,N,k,n) if lower else hypergeom.cdf(obs,N,k,n)
            accepted = probs >= alpha_tail
            for i in np.flatnonzero(np.isclose(probs,alpha_tail,rtol=1e-10,atol=1e-14)):
                accepted[i] = _exact_tail_accept(N,int(k[i]),n,int(obs[i]),lower,threshold)
            accept = accepted if lower else ~accepted
            left, right = np.where(accept, left, k), np.where(accept, k, right)
        return right if lower else right-1

    return boundary(True), boundary(False)


def design_risk(populations, pass_totals, sizes):
    """Exact design RMSE, coverage and width under independent SRSWOR within each stratum."""
    N = sum(populations)
    total_pass = sum(pass_totals)
    variance = 0.0
    lower, upper, probability = 0, 0, 1.
    for Nh, T, n in zip(populations, pass_totals, sizes, strict=True):
        if not 0 <= T <= Nh or not 1 <= n <= Nh:
            raise ValueError("incompatible stratum counts")
        if n < Nh:
            finite_variance = T*(Nh-T)/(Nh*(Nh-1))
            variance += (Nh/N)**2 * (1-n/Nh) * finite_variance/n
        support = np.arange(max(0, n-(Nh-T)), min(n, T)+1)
        pmf = hypergeom.pmf(support, Nh, T, n)
        lo, hi = interval_table(Nh, n, len(populations))
        lower = np.add.outer(lower, lo[support])
        upper = np.add.outer(upper, hi[support])
        probability = np.multiply.outer(probability, pmf)
    coverage = probability[(lower <= total_pass) & (total_pass <= upper)].sum()
    np.testing.assert_allclose(probability.sum(), 1., atol=1e-9)
    return dict(exact_rmse_pp=float(100*np.sqrt(variance)), exact_95ci_coverage=float(coverage),
                exact_mean_95ci_width_pp=float(100*((upper-lower)*probability).sum()/N))


def sample_designs(data, sc_costs, seed_list):
    st = initial_state(data)
    k, ref, judge = st["k"], (st["gf"] == 0).astype(int), (st["pf"] == 0)
    n, m = len(k), int(k.sum())
    designs = {"unit_SRS": [np.arange(n)], "predicted_verdict_stratified": [np.flatnonzero(judge == v) for v in (False,True)]}
    orders = {seed: np.argsort(rng_for(seed, "task").random(n), kind="stable") for seed in seed_list}
    rows, draws = [], []
    for name, strata in designs.items():
        strata = [ids for ids in strata if len(ids)]
        stratum_orders = {seed: [order[np.isin(order, ids)] for ids in strata] for seed, order in orders.items()}
        for ratio in RATIOS:
            for pct in PERCENT:
                cap = budget(m, n, ratio, pct)
                result = guaranteed_allocation([k[ids] for ids in strata], ratio, cap)
                if result is None:
                    raise ValueError("cap cannot guarantee one draw per populated stratum")
                sizes, worst = result
                populations = [len(ids) for ids in strata]
                pass_totals = [int(ref[ids].sum()) for ids in strata]
                base = dict(dataset=data["name"], design=name, open_to_bit=ratio, percent=pct,
                            cap=cap, sample_n=sum(sizes), stratum_populations="/".join(map(str,populations)),
                            stratum_pass_totals="/".join(map(str,pass_totals)),
                            stratum_samples="/".join(map(str,sizes)), worst_full_rubric_spend=worst)
                rows.append(dict(**base, population_pass_rate=float(ref.mean()),
                                 **design_risk(populations,pass_totals,sizes)))
                for seed in seed_list:
                    subsets = [ordered[:size] for ordered,size in zip(stratum_orders[seed], sizes)]
                    est = sum(len(ids)/n*ref[sel].mean() for ids,sel in zip(strata,subsets))
                    counts = [int(ref[sel].sum()) for sel in subsets]
                    bounds = [interval_table(len(ids),size,len(strata)) for ids,size in zip(strata,sizes)]
                    lo = sum(int(a[v]) for (a,b),v in zip(bounds,counts))
                    hi = sum(int(b[v]) for (a,b),v in zip(bounds,counts))
                    sampled = np.concatenate(subsets)
                    bits = int(sc_costs[seed][sampled].sum())
                    full = int(k[sampled].sum()) + ratio*len(sampled)
                    spent = bits + ratio*len(sampled)
                    if not (spent <= full <= worst <= cap):
                        raise AssertionError("cost exceeded guaranteed cap")
                    draws.append(dict(**base,seed=seed,estimate=float(est),ci_low=lo/n,ci_high=hi/n,
                                      covered=int(lo<=ref.sum()<=hi),sc_labels=bits,sc_spent=spent,full_spent=full))
    return rows, draws


def key_stats(values):
    a = np.asarray(values)
    return {"mean": float(a.mean()), "q025": float(np.quantile(a,.025)), "q975": float(np.quantile(a,.975)),
            "negative": int((a<0).sum()), "tie": int((a==0).sum()), "positive": int((a>0).sum())}


def contrasts(rows):
    keys = ("dataset","open_to_bit","denominator","percent")
    groups = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        groups[tuple(r[k] for k in keys)][r["seed"]][r["policy"]]=r
    results = []
    for key, pairs in groups.items():
        for a, b in (("SC","DF"),("SC","Random"),("DF","Random")):
            for metric in ("eager_errors","gated_errors","certified","bit_errors_found","labels","opened","spent"):
                dif = [pol[a][metric]-pol[b][metric] for pol in pairs.values()]
                results.append(dict(**dict(zip(keys,key)),effect="acquisition",left=a,right=b,metric=metric,
                                    n_orders=len(dif),unit_count=pairs[next(iter(pairs))][a]["unit_count"],**key_stats(dif)))
        for policy in POLICIES:
            dif = [pol[policy]["gated_minus_eager"] for pol in pairs.values()]
            results.append(dict(**dict(zip(keys,key)),effect="update",left="gated",right="eager",metric=policy,
                                n_orders=len(dif),unit_count=pairs[next(iter(pairs))][policy]["unit_count"],**key_stats(dif)))
    return results


def rank_grid(rows):
    """Rank the three policies by arithmetic mean eager error count across paired orders."""
    keys = ("dataset", "open_to_bit", "denominator", "percent")
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[tuple(row[key] for key in keys)][row["policy"]].append(row["eager_errors"])
    out = []
    for key, policies in groups.items():
        counts = [len(policies[p]) for p in POLICIES]
        if len(set(counts)) != 1:
            raise ValueError("policy sample sizes differ")
        totals = {p:sum(policies[p]) for p in POLICIES}
        scores = sorted(set(totals.values()))
        order = " < ".join(" = ".join(p for p in POLICIES if totals[p] == score) for score in scores)
        out.append(dict(**dict(zip(keys,key)), n_orders=counts[0], ascending_eager_error=order,
                        **{p:totals[p]/counts[0] for p in POLICIES}))
    return out


def summaries(rows):
    return {"paired_contrasts.csv": contrasts(rows), "ranking_grid.csv": rank_grid(rows)}


def from_orders(path, output):
    """Rebuild the paired summaries with only the text-free order-level outcomes."""
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as src:
        rows = [{k: v if k in ("dataset","policy","denominator") else int(v) for k,v in r.items()}
                for r in csv.DictReader(src)]
    for filename, table in summaries(rows).items():
        csv_out(output/filename, table)


def from_designs(path, output):
    """Recalculate all finite-population risks using only stratum sufficient counts."""
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(path)
    for row in rows:
        populations, pass_totals, sizes = (list(map(int,row[key].split("/"))) for key in
                                           ("stratum_populations","stratum_pass_totals","stratum_samples"))
        row["population_pass_rate"] = sum(pass_totals)/sum(populations)
        row.update(design_risk(populations,pass_totals,sizes))
    csv_out(output/"pass_estimation.csv", rows)


def run(output, datasets, seed_list, recorded_timing=None):
    output.mkdir(parents=True, exist_ok=True)
    all_rows, estimation, estimation_draws = [], [], []
    for name, data in datasets.items():
        sc_costs = {}
        for seed in seed_list:
            for short, policy in POLICIES.items():
                trace = replay(data, policy, "random", seed)
                all_rows.extend(prefixes(data, trace, short, seed))
                if short == "SC":
                    sc_costs[seed] = np.bincount(data["task_index"][trace["indices"]], minlength=len(data["task_ids"]))
        design, samples = sample_designs(data, sc_costs, seed_list)
        estimation.extend(design)
        estimation_draws.extend(samples)
        print(f"{name}: {len(seed_list)} paired orders complete", flush=True)
    tables = {"orders.csv.gz": all_rows, **summaries(all_rows),
              "pass_estimation.csv": estimation, "pass_estimation_draws.csv.gz": estimation_draws}
    for path, rows in tables.items():
        csv_out(output/path, rows)
    if recorded_timing is None:
        source = json.loads((ROOT/"data/interim/differentiation_screen/source_manifest.json").read_text())
        revision = source["datasets"]["judgmentbench"]["revision"]
        timing = timing_anchor()
    else:
        prior = json.loads(recorded_timing.read_text())
        revision, timing = prior["source_revision_judgmentbench"], prior["timing"]
    manifest = dict(seeds=list(seed_list),ratios=RATIOS,percents=PERCENT,policies=POLICIES,
                    source_revision_judgmentbench=revision, timing=timing,
                    timing_input="raw_completed_annotations" if recorded_timing is None else "saved_aggregate_anchor",
                    datasets={k:metadata(d) for k,d in datasets.items()},
                    files={path:dict(rows=len(rows),sha256=hashlib.sha256((output/path).read_bytes()).hexdigest())
                           for path,rows in tables.items()},
                    code_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in (
                        "src/cost_replay_20261125.py","src/conjunction_policy_study.py","src/conjunction_policy_data.py",
                        "src/finite_population_intervals.py")})
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--from-orders", type=Path, help="rebuild policy contrast and rank tables from orders.csv.gz")
    mode.add_argument("--from-designs", type=Path, help="rebuild exact survey risks from saved stratum counts")
    mode.add_argument("--timing-manifest", type=Path, help="reuse aggregate time anchors with the public numeric loader")
    args = parser.parse_args()
    if args.from_orders:
        from_orders(args.from_orders, args.output)
    elif args.from_designs:
        from_designs(args.from_designs, args.output)
    else:
        run(args.output, load_datasets(ROOT), SEEDS, args.timing_manifest)
