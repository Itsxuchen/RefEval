"""Audit manuscript anchors from public inputs and saved results; no experiments.

Run with --package-root pointing to a RefEval release. The receipt records
relative source paths, selectors, units, and executable consistency checks.
It is deliberately separate from experiment execution/reproduction manifests.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def module_at(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def unit_vectors(data):
    ti = np.asarray(data["task_index"])
    slots = [np.flatnonzero(ti == i) for i in range(len(data["task_ids"]))]
    g = np.array([np.all(data["gold"][idx]) for idx in slots])
    p = np.array([np.all(data["prediction"][idx]) for idx in slots])
    return slots, g, p


def counts(g, p, criteria):
    return dict(n_units=len(g), n_criteria=criteria, reference_pass=int(g.sum()),
                judge_pass=int(p.sum()), TP=int((g & p).sum()),
                TN=int((~g & ~p).sum()), FP=int((~g & p).sum()), FF=int((g & ~p).sum()))


@lru_cache(maxsize=200000)
def combination_event(G, h, r, f):
    """Independent combinatorial expression (not the producer recurrence)."""
    if r > h or f > G - h or r + f > G:
        return 0.0
    return (math.comb(h, r) * math.comb(G - h, f)
            / (math.comb(G, r + f) * math.comb(r + f, r)))


def analytical_curve(data, policy, shares):
    """Reconstruct group order and sum literal per-unit event probabilities."""
    m = len(data["gold"])
    slots, truth, prediction = unit_vectors(data)
    flags = np.any(data["secondary_predictions"] != data["prediction"][None, :], axis=0)
    if policy == "random_criterion":
        blocks = [np.arange(m)]
    elif policy == "disagreement_then_random":
        blocks = [idx[flags[idx]] for idx in slots if flags[idx].any()]
        if (~flags).any():
            blocks.append(np.flatnonzero(~flags))
    else:
        raise ValueError(policy)
    gid = np.empty(m, dtype=int)
    for i, idx in enumerate(blocks):
        gid[idx] = i
    sizes = np.array(list(map(len, blocks)))
    ends = np.cumsum(sizes)
    events = []
    for idx, z, hatz in zip(slots, truth, prediction):
        pred_fail = idx[data["prediction"][idx] == 0]
        ref_fail = idx[data["gold"][idx] == 0]
        if z and not hatz:
            events.append(("repair", gid[pred_fail], np.array([], dtype=int)))
            events.append(("complete", gid[idx], np.array([], dtype=int)))
        elif not z and not hatz and not np.intersect1d(pred_fail, ref_fail).size:
            events.append(("introduced", gid[pred_fail], gid[ref_fail]))
    out = []
    for share in shares:
        cap = int(round(float(share) * m))
        active = int(np.searchsorted(ends, cap, side="right"))
        G = int(sizes[active]) if active < len(sizes) else 0
        h = cap - (int(ends[active-1]) if active else 0)
        sums = dict(repair=0., complete=0., introduced=0.)
        for kind, required, forbidden in events:
            if (required > active).any() or (forbidden < active).any():
                continue
            r = int((required == active).sum())
            f = int((forbidden == active).sum())
            sums[kind] += combination_event(G, h, r, f)
        A = sums["introduced"]
        D = sums["repair"] - sums["complete"]
        out.append((A, D, D-A))
    return np.asarray(out)


def audit(root: Path):
    root = root.resolve()
    source_hashes, checks, claims = {}, [], {}

    def source(relative):
        path = root / relative
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def table(relative):
        return pd.read_csv(source("artifacts/expected/" + relative))

    def document(relative):
        return json.loads(source("artifacts/expected/" + relative).read_text())

    def equal(name, actual, expected, tolerance=1e-8):
        a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if a.shape != e.shape:
            raise AssertionError(f"{name}: shape {a.shape} != {e.shape}")
        diff = float(np.max(np.abs(a-e))) if a.size else 0.
        if not np.isfinite(a).all() or not np.isfinite(e).all() or diff > tolerance:
            raise AssertionError(f"{name}: maximum absolute difference {diff}")
        checks.append(dict(check=name, scalar_values=int(a.size), max_absolute_difference=diff, passed=True))

    def require(name, okay, n=1):
        if not bool(okay):
            raise AssertionError(name)
        checks.append(dict(check=name, instances=int(n), passed=True))

    loader = module_at(root / "src/conjunction_policy_data.py", "manuscript_data")
    datasets = loader.load_datasets(root)
    for data in datasets.values():
        for relative, digest in data["source_hashes"].items():
            if relative in source_hashes:
                continue
            require("input hash: " + relative, hashlib.sha256(source(relative).read_bytes()).hexdigest() == digest)
    saved_structures = document("conjunction_mechanism/update_structures.json")
    data_counts = []
    for name, data in datasets.items():
        slots, g, p = unit_vectors(data)
        row = counts(g, p, len(data["gold"]))
        susceptible = sum(not z and not hatz and not np.any((data["gold"][idx] == 0)
                          & (data["prediction"][idx] == 0)) for idx, z, hatz in zip(slots, g, p))
        row.update(dataset=name, n_base_tasks=len(set(data["base_task_ids"])),
                   n_outputs=len(set(data["output_ids"])), n_raters=len(set(data["rater_ids"])),
                   criterion_errors=int((data["gold"] != data["prediction"]).sum()), susceptible=int(susceptible))
        rename = dict(judge_pass="predicted_pass", FP="initial_fp", FF="initial_ff")
        fields = ["n_units", "n_criteria", "reference_pass", "judge_pass", "FP", "FF", "criterion_errors", "susceptible"]
        equal("data counts: " + name, [row[k] for k in fields],
              [saved_structures[name][rename.get(k, k)] for k in fields])
        data_counts.append(row)
    claims["data_counts"] = dict(source="data (strict validated loader); conjunction_mechanism/update_structures.json",
                                unit="criterion instances and evaluation units, as named", values=data_counts)
    jb = datasets["JB GPT-5.4"]
    jb_slots, jb_g, jb_p = unit_vectors(jb)
    reject_types = [set(np.asarray(jb["category"])[idx][jb["prediction"][idx] == 0])
                    for idx, g, p in zip(jb_slots, jb_g, jb_p) if g and not p]
    penalty_counts = dict(penalty_only=sum(x == {"occurrence_count"} for x in reject_types),
                          binary_and_penalty=sum(x == {"binary","occurrence_count"} for x in reject_types),
                          binary_only=sum(x == {"binary"} for x in reject_types))
    equal("manuscript full-target false-fail rejection types", list(penalty_counts.values()), [156,64,0])
    claims["data_counts"]["full_jb_false_fail_rejection_types"] = penalty_counts

    # Reconstruct the complete disagreement-only endpoint from criterion bits,
    # independently of replay/update helpers and of the displayed README.
    jb_flags = np.any(jb["secondary_predictions"] != jb["prediction"][None, :], axis=0)
    jb_updated_bits = np.where(jb_flags, jb["gold"], jb["prediction"])
    jb_updated = np.array([np.all(jb_updated_bits[idx]) for idx in jb_slots])
    disagreement_endpoint = dict(
        actual_queries=int(jb_flags.sum()),
        criterion_errors_found=int((jb_flags & (jb["gold"] != jb["prediction"])).sum()),
        initial_errors=int((jb_p != jb_g).sum()),
        residual_errors=int((jb_updated != jb_g).sum()),
        fixed_initial_fp=int((jb_p & ~jb_g & ~jb_updated).sum()),
        fixed_initial_ff=int((~jb_p & jb_g & jb_updated).sum()),
        introduced_fp=int((~jb_p & ~jb_g & jb_updated).sum()),
        introduced_ff=int((jb_p & jb_g & ~jb_updated).sum()))
    disagreement_endpoint["repaired_verdicts"] = (
        disagreement_endpoint["fixed_initial_fp"] + disagreement_endpoint["fixed_initial_ff"])
    equal("complete disagreement set independently reconstructed",
          list(disagreement_endpoint.values()), [4316,1217,244,245,2,2,5,0,4])
    claims["complete_disagreement_endpoint"] = dict(
        source="data/reference/judgmentbench (fixed full-target reference and both saved judge vectors)",
        selector="JB GPT-5.4 full; all slots where GPT-5.4 and GPT-5.4-mini disagree",
        unit="criterion instances and output-by-rater verdicts", values=disagreement_endpoint,
        warning="Illustrative complete-set endpoint, not an average over new tasks or an adjudication of reference truth.")

    expected = table("conjunction_mechanism/update_expected_grid.csv")
    for (name, policy), group in expected.groupby(["dataset", "policy"], sort=False):
        group = group.sort_values("budget_share")
        computed = analytical_curve(datasets[name], policy, group.budget_share)
        equal("independent analytical curve: " + name + "/" + policy, computed,
              group[["expected_introduced_fp", "expected_delayed_ff", "expected_delta"]], tolerance=2e-8)
        equal("analytical endpoint zero: " + name + "/" + policy, computed[[0,-1]], np.zeros((2,3)))
    curves = table("conjunction_policy_study/budget_curves.csv.gz")
    equal("all saved trajectories: residual count identity", curves.residual_errors, curves.residual_fp+curves.residual_ff)
    equal("all saved trajectories: gated count identity", curves.gated_residual_errors,
          curves.gated_residual_fp+curves.gated_residual_ff)
    equal("all saved trajectories: D minus A", curves.gated_residual_errors-curves.residual_errors,
          curves.delayed_ff_corrections-curves.introduced_fp)
    complete_disagreement_rows = curves[(curves.dataset == "JB GPT-5.4")
        & (curves.policy == "disagreement_only") & (curves.budget_share == 1.)]
    endpoint_fields = ["actual_queries", "criterion_errors_found", "residual_errors",
                       "fixed_initial_fp", "fixed_initial_ff", "introduced_fp"]
    require("complete disagreement endpoint present across all orders and seeds",
            len(complete_disagreement_rows) == 3*32)
    equal("complete disagreement endpoint agrees with every saved full-cap replay",
          complete_disagreement_rows[endpoint_fields],
          np.tile([disagreement_endpoint[k] for k in endpoint_fields],
                  (len(complete_disagreement_rows), 1)))
    observed = curves[(curves.task_order == "release") & (curves.seed > 0)
                      & curves.policy.isin(expected.policy.unique())].copy()
    observed["delta"] = observed.gated_residual_errors-observed.residual_errors
    keys = ["dataset", "policy", "budget_share"]
    means = observed.groupby(keys)[["introduced_fp", "delayed_ff_corrections", "delta"]].mean()
    merged = expected.set_index(keys).join(means)
    equal("all 1818 Monte Carlo means from 31 raw order rows", merged[["introduced_fp", "delayed_ff_corrections", "delta"]],
          merged[["observed_mean_introduced_fp", "observed_mean_delayed_ff", "observed_mean_delta"]])
    require("exactly 31 noncanonical order rows per RQ1 cell", (observed.groupby(keys).size() == 31).all(), len(means))
    anchors = expected[(expected.policy == "disagreement_then_random") & expected.budget_share.isin([.2,.4,.6,.8,.84])]
    fields = ["dataset", "budget_share", "budget_cap", "expected_introduced_fp", "expected_delayed_ff", "expected_delta", "observed_mean_delta", "canonical_seed0_delta"]
    extrema = []
    for name, frame in expected[expected.policy == "disagreement_then_random"].groupby("dataset", sort=False):
        for stat, key in (("analytic", "expected_delta"), ("31_order_mean", "observed_mean_delta")):
            for direction, idx in (("min", frame[key].idxmin()), ("max", frame[key].idxmax())):
                row = frame.loc[idx]
                extrema.append(dict(dataset=name, statistic=stat, extremum=direction,
                                    budget_share=row.budget_share, delta=row[key]))
    claims["rq1"] = dict(source="conjunction_mechanism/update_expected_grid.csv; conjunction_policy_study/budget_curves.csv.gz",
                         selector="release order, disagreement_then_random; observed means use seeds 1 through 31",
                         unit="gated minus eager evaluation-unit errors", values=anchors[fields].to_dict("records"), extrema=extrema)
    cost_shares = [.2, .84]
    cost_values = analytical_curve(jb, "disagreement_then_random", cost_shares)
    cost_ratios = cost_values[:, 1] / cost_values[:, 0]
    equal("expected-loss break-even ratios use unrounded expectations", cost_ratios,
          [0.7247771334551854, 9.104330494153786], tolerance=1e-9)
    claims["rq1"]["expected_loss_break_even"] = dict(
        unit="false-PASS cost divided by false-FAIL cost",
        condition="For c_FF > 0 and expected A > 0, gating has lower expected endpoint loss iff c_FP/c_FF > E[D]/E[A].",
        values=[dict(budget_share=share, expected_introduced_fp=float(row[0]),
                     expected_delayed_ff=float(row[1]), cost_ratio_crossover=float(ratio))
                for share, row, ratio in zip(cost_shares, cost_values, cost_ratios)],
        warning="Conditional on the fixed full JudgmentBench frame and disagreement-then-random priorities; ratios are computed before rounding.")
    focal_policies = curves[(curves.task_order == "release") & (curves.seed > 0) & (curves.budget_share == .2)]
    discovery_claims = []
    for name in ["DR Gemini 3.1 Pro", "JB GPT-5.4-mini", "JB GPT-5.4"]:
        sc = focal_policies[(focal_policies.dataset == name) & (focal_policies.policy == "sc_judge")].set_index("seed")
        dis = focal_policies[(focal_policies.dataset == name) & (focal_policies.policy == "disagreement_then_random")].set_index("seed")
        reversals = int(((dis.criterion_errors_found > sc.criterion_errors_found) & (dis.certified_tasks < sc.certified_tasks)).sum())
        equal("manuscript discovery/certification reversal orders: " + name, [reversals], [0 if name == "JB GPT-5.4" else 31])
        values = ["actual_queries", "criterion_errors_found", "certified_tasks", "residual_errors"]
        discovery_claims.append(dict(dataset=name, orders=31, opposing_discovery_and_certification_orders=reversals,
                                    short_circuit_marginal_medians=sc[values].median().to_dict(),
                                    disagreement_continuation_marginal_medians=dis[values].median().to_dict()))
    canonical = curves[(curves.dataset == "JB GPT-5.4") & (curves.task_order == "release") & (curves.seed == 0)
                       & (curves.budget_share == .2) & (curves.policy == "disagreement_then_random")].iloc[0]
    equal("manuscript deterministic loss example", canonical[["residual_fp","residual_ff","gated_residual_fp","gated_residual_ff"]], [28,211,22,220])
    claims["rq1"]["canonical_full_jb_20pct_loss"] = dict(eager_FP=28,eager_FF=211,gated_FP=22,gated_FF=220,
        gating_minus_eager_loss_FP_coefficient=-6, gating_minus_eager_loss_FF_coefficient=9, cost_ratio_crossover=1.5)
    claims["discovery_vs_certification"] = dict(source="conjunction_policy_study/budget_curves.csv.gz",
        selector="release order; 20% cap; seeds 1 through 31; short circuit judge-first versus disagreement continuation",
        unit="counts and marginal medians", values=discovery_claims)
    pure_certification = []
    for name, group in focal_policies[focal_policies.policy == "sc_judge"].groupby("dataset", sort=False):
        require("pure short-circuit 20pct has all noncanonical seeds: " + name,
                sorted(group.seed.tolist()) == list(range(1, 32)))
        require("pure short-circuit 20pct spends the common cap: " + name,
                (group.actual_queries == group.budget_cap).all(), len(group))
        pure_certification.append(dict(dataset=name, seeds=len(group),
            actual_queries=int(group.actual_queries.iloc[0]),
            certified_tasks_median=float(group.certified_tasks.median()),
            residual_errors_median=float(group.residual_errors.median()),
            criterion_errors_found_median=float(group.criterion_errors_found.median()),
            criterion_accuracy_point_estimate=None, criterion_accuracy_statistical_interval=None))
    for name, target in [("DR Gemini 3.1 Pro", [323,275,1]), ("JB GPT-5.4", [4697,1512,15])]:
        row = next(r for r in pure_certification if r["dataset"] == name)
        equal("manuscript pure-certification endpoint: " + name,
              [row["actual_queries"], row["certified_tasks_median"], row["residual_errors_median"]], target)
    claims["pure_certification_endpoints"] = dict(
        source="conjunction_policy_study/budget_curves.csv.gz; conjunction_label_value/budget_estimation_rows.csv.gz",
        selector="sc_judge, release order, 20% cap, seeds 1 through 31",
        unit="actual queries and separate marginal median task counts", values=pure_certification,
        warning="These selective queries have no subsequent probability sample. The stored logical agreement bounds are not a 95% sampling interval or a design-unbiased point estimate.")

    permutation = table("conjunction_mechanism/update_permutation_rows.csv")
    permutation_summary = table("conjunction_mechanism/update_permutation_summary.csv").set_index(["dataset","policy","budget_share"])
    permutation_claims = []
    for key, group in permutation.groupby(["dataset","policy","budget_share"]):
        saved = permutation_summary.loc[key]
        original = expected[(expected.dataset == key[0]) & (expected.policy == key[1]) & (expected.budget_share == key[2])].iloc[0].expected_delta
        opposite = int((group.expected_delta * original < -1e-12).sum())
        equal("matched margins summary: " + str(key), [len(group),original,group.expected_delta.min(),group.expected_delta.max(),opposite],
              [saved.permutations,saved.original_expected_delta,saved.permutation_expected_delta_min,saved.permutation_expected_delta_max,saved.opposite_nonzero_sign_count])
        permutation_claims.append(dict(dataset=key[0],policy=key[1],budget_share=key[2],configurations=len(group),
            original_expected_delta=original, configured_delta_min=float(group.expected_delta.min()),
            configured_delta_max=float(group.expected_delta.max()),opposite_sign_configurations=opposite,
            reference_pass_min=int(group.reference_pass.min()), reference_pass_max=int(group.reference_pass.max())))
    opposite_cells = sum(x["opposite_sign_configurations"] > 0 for x in permutation_claims)
    equal("manuscript counterexample cell count", [len(permutation_claims),opposite_cells], [90,44])
    claims["matched_marginal_configurations"] = dict(source="conjunction_mechanism/{update_permutation_rows.csv,update_permutation_summary.csv,update_expected_grid.csv}",
        unit="errors over fixed simulated label configurations; not a reversal probability", cells=len(permutation_claims),
        cells_admitting_opposite_sign=opposite_cells, focal=next(x for x in permutation_claims if x["dataset"] == "JB GPT-5.4" and x["policy"] == "random_criterion" and x["budget_share"] == .2))
    jb_permutations = permutation[permutation.dataset == "JB GPT-5.4"]
    require("full JB reference-pass totals agree across policy and budget for each permutation",
            (jb_permutations.groupby("permutation").reference_pass.nunique() == 1).all(), 64)
    equal("full JB actual 64-permutation reference-pass range",
          [jb_permutations.permutation.nunique(), int(jb_g.sum()), jb_permutations.reference_pass.min(),
           jb_permutations.reference_pass.max()], [64,227,108,132])
    claims["matched_marginal_configurations"]["full_jb_reference_pass_range"] = dict(
        original=227, configured_min=108, configured_max=132, distinct_configurations=64,
        warning="Range of the saved 64 reference rearrangements, not extrema over every feasible configuration.")

    allocation = table("conjunction_mechanism/allocation_rows.csv.gz")
    equal("allocation certificate decomposition", allocation.certified_tasks_delta,
          allocation.certified_initial_correct_delta+allocation.certified_initial_wrong_delta)
    equal("allocation gated error decomposition", allocation.gated_residual_errors_delta, -allocation.certified_initial_wrong_delta)
    equal("allocation eager error decomposition", allocation.residual_errors_delta,
          -allocation.certified_initial_wrong_delta-allocation.noncertified_ff_repair_delta+allocation.introduced_fp_delta)
    equal("allocation no introduced FF", allocation.introduced_ff_delta, np.zeros(len(allocation)))
    budget = table("conjunction_label_value/budget_estimation_rows.csv.gz")
    pure_budget_rows = budget[(budget.task_order == "release") & (budget.budget_share == .2)
                             & (budget.policy == "sc_judge")]
    require("pure short-circuit rows do not supply probability-sampling estimates or intervals",
            len(pure_budget_rows) == 9*31
            and pure_budget_rows.accuracy_estimate.isna().all()
            and (pure_budget_rows.stageR_queries == 0).all()
            and (pure_budget_rows.inference_kind == "logical_bounds_or_census").all(), len(pure_budget_rows))
    pairkeys = ["dataset", "task_order", "seed", "budget_share"]
    # SRS is order-invariant; its release-order run is the shared control.
    srs = budget[budget.policy == "random_criterion"].drop(columns="task_order")
    joined = budget[budget.policy == "sc50_then_srs"].merge(
        srs, on=["dataset", "seed", "budget_share"], suffixes=("_mix", "_srs"), validate="many_to_one").set_index(pairkeys)
    a = allocation.set_index(pairkeys).join(joined)
    for field in ["certified_tasks", "residual_errors"]:
        equal("paired allocation vs saved replay: " + field, a[field+"_delta"], a[field+"_mix"]-a[field+"_srs"])
    equal("paired allocation identical realized label counts", a.actual_queries_mix, a.actual_queries_srs)
    focal = allocation[(allocation.task_order == "release") & (allocation.budget_share == .2)]
    fields = ["actual_queries", "certified_tasks_delta", "certified_initial_correct_delta", "certified_initial_wrong_delta",
              "noncertified_ff_repair_delta", "introduced_fp_delta", "residual_errors_delta"]
    focal_means = focal.groupby("dataset", sort=False)[fields].mean().reset_index()
    med = budget[(budget.task_order == "release") & (budget.budget_share == .2)
                 & budget.policy.isin(["random_criterion", "sc50_then_srs"])].copy()
    med["accuracy_ci_width_pp"] = med.accuracy_ci_width * 100
    med = med.groupby(["dataset", "policy"], sort=False)[["actual_queries", "certified_tasks", "residual_errors", "accuracy_ci_width_pp"]].median().reset_index()
    claims["rq2"] = dict(source="conjunction_mechanism/allocation_rows.csv.gz; conjunction_label_value/budget_estimation_rows.csv.gz",
                         selector="release order, budget_share=0.2, seeds 1 through 31", unit="paired mix minus SRS counts; width in percentage points",
                         paired_means=focal_means.to_dict("records"), marginal_medians=med.to_dict("records"),
                         warning="Separate marginal medians need not describe any one run; their ratios are not paired relative effects.")
    expected_error_means = {
        "DR Gemini 3.1 Pro": 25/31, "DR Qwen-plus": -17/31,
        "AC Qwen-plus": -70/31, "AC DeepSeek v4-pro": -3/31, "DR GPT-5.4 low": -3/31,
        "JB GPT-5.4": -2792/31, "JB GPT-5.4-mini": -2836/31,
        "JB GPT-5.4 | binary_only": -1819/31, "JB GPT-5.4-mini | binary_only": -2028/31}
    by_name = focal_means.set_index("dataset")
    require("20pct mixed-allocation grid contains exactly nine cells", set(by_name.index) == set(expected_error_means))
    equal("all nine manuscript paired error means",
          [by_name.loc[name, "residual_errors_delta"] for name in expected_error_means],
          list(expected_error_means.values()))
    require("20pct mixed allocation increases certification in every cell and order",
            len(focal) == 9*31 and (focal.certified_tasks_delta > 0).all(), len(focal))
    require("only DR Gemini has a positive 20pct paired mean error contrast",
            list(by_name.index[by_name.residual_errors_delta > 0]) == ["DR Gemini 3.1 Pro"])
    family_ranges = {}
    for label, mask in [("RuVerBench", ~focal_means.dataset.str.startswith("JB ")),
                        ("JudgmentBench", focal_means.dataset.str.startswith("JB "))]:
        group = focal_means[mask]
        family_ranges[label] = dict(analysis_cells=len(group),
            paired_error_mean_min=float(group.residual_errors_delta.min()),
            paired_error_mean_max=float(group.residual_errors_delta.max()))
    equal("RuVerBench five-cell paired error mean range",
          [family_ranges["RuVerBench"]["paired_error_mean_min"], family_ranges["RuVerBench"]["paired_error_mean_max"]],
          [-70/31, 25/31])
    equal("JudgmentBench four-cell paired error mean range",
          [family_ranges["JudgmentBench"]["paired_error_mean_min"], family_ranges["JudgmentBench"]["paired_error_mean_max"]],
          [-2836/31, -1819/31])
    jb_focal = focal[focal.dataset == "JB GPT-5.4"]
    order_mcse = {field: float(jb_focal[field].std(ddof=1)/math.sqrt(len(jb_focal)))
                  for field in ["certified_tasks_delta", "residual_errors_delta"]}
    claims["rq2"]["nine_cell_summary"] = dict(family_ranges=family_ranges,
        certification_gain_cells=9, certification_gain_orders=9*31,
        positive_mean_error_cells=["DR Gemini 3.1 Pro"],
        full_jb_paired_mean_order_mcse=order_mcse,
        warning="Cell means describe the fixed frames; cells share tasks and are not independent replications. Order MCSE is not task-population uncertainty.")

    diag = document("conjunction_robustness/reference/frame_diagnostics.json")
    pairing = table("conjunction_robustness/reference/pairing.csv")
    matched_counts = []
    for name in diag:
        data = datasets[name]
        slots, g, p = unit_vectors(data)
        by_output = defaultdict(list)
        for i, oid in enumerate(data["output_ids"]):
            by_output[oid].append(i)
        peers = {i: [j for j in ids if data["rater_ids"][j] != data["rater_ids"][i]]
                 for ids in by_output.values() for i in ids}
        selected = [i for i in peers if peers[i]]
        peer_values = dict(n_outputs=len({data["output_ids"][i] for i in selected}), n_annotations=len(selected),
            n_human_raters=len({data["rater_ids"][i] for i in selected}), n_base_tasks=len({data["base_task_ids"][i] for i in selected}),
            reference_pass=sum(g[i] for i in selected), reference_pass_all_peer_pass=sum(g[i] and all(g[j] for j in peers[i]) for i in selected),
            initial_FF=sum(g[i] and not p[i] for i in selected),
            initial_FF_any_peer_fail=sum(g[i] and not p[i] and any(not g[j] for j in peers[i]) for i in selected),
            initial_FF_all_peer_fail=sum(g[i] and not p[i] and all(not g[j] for j in peers[i]) for i in selected),
            judge_reference_disagreement=sum(g[i] != p[i] for i in selected),
            disagreement_any_peer_agrees_current_judge=sum(g[i] != p[i] and any(g[j] == p[i] for j in peers[i]) for i in selected),
            disagreement_all_peer_agree_current_judge=sum(g[i] != p[i] and all(g[j] == p[i] for j in peers[i]) for i in selected),
            judge_reference_task_agreement=sum(g[i] == p[i] for i in selected))
        pairs = [(i,j) for ids in by_output.values() for i,j in itertools.combinations(ids,2)
                 if data["rater_ids"][i] != data["rater_ids"][j]]
        peer_values.update(human_unordered_distinct_rater_pairs=len(pairs), human_pair_task_equal=sum(g[i] == g[j] for i,j in pairs),
            human_pair_criterion_equal=sum(int((data["gold"][slots[i]] == data["gold"][slots[j]]).sum()) for i,j in pairs),
            human_pair_criterion_comparisons=sum(len(slots[i]) for i,j in pairs),
            saved_judge_binary_vector_different_pairs=sum(not np.array_equal(data["prediction"][slots[i]], data["prediction"][slots[j]]) for i,j in pairs),
            saved_judge_task_different_pairs=sum(p[i] != p[j] for i,j in pairs))
        equal("repeated-reference peer diagnostics: " + name, list(peer_values.values()),
              [diag[name]["original_repeated_peer_diagnostics"][k] for k in peer_values])
        lookup = {tid.removeprefix("judgmentbench/"): i for i, tid in enumerate(data["task_ids"])}
        a_ids = [lookup[str(x)] for x in pairing.A_annotation_id]
        b_ids = [lookup[str(x)] for x in pairing.B_annotation_id]
        j_ids = [lookup[str(x)] for x in pairing.anchor_annotation_id]
        reference_bits = {}
        for label, ids in [("A",a_ids),("B",b_ids)]:
            z, hatz = g[ids], p[j_ids]
            current = counts(z, hatz, sum(len(slots[i]) for i in ids))
            equal("matched frozen judge/reference counts: " + name + "/" + label,
                  list(current.values()), [diag[name]["output_uniform_"+label][k] for k in current])
            reference_bits[label] = np.concatenate([data["gold"][slots[i]] for i in ids])
            matched_counts.append(dict(dataset=name, reference=label, **current))
        agreement = dict(criterion_equal=int((reference_bits["A"] == reference_bits["B"]).sum()), n_criteria=len(reference_bits["A"]),
            task_equal=int((g[a_ids] == g[b_ids]).sum()), n_outputs=len(a_ids),
            A_pass_B_fail=int((g[a_ids] & ~g[b_ids]).sum()), A_fail_B_pass=int((~g[a_ids] & g[b_ids]).sum()))
        equal("matched A/B agreement: " + name, list(agreement.values()), [diag[name]["reference_agreement"][k] for k in agreement])
    reference1 = table("conjunction_robustness/reference/rq1_rows.csv.gz")
    equal("matched reference RQ1 count differences", reference1.gated_minus_eager, reference1.gated_errors-reference1.eager_errors)
    diagonal = reference1[reference1.acquisition_ref == reference1.evaluation_ref]
    equal("matched diagonal RQ1 D minus A", diagonal.gated_minus_eager, diagonal.D_delayed_FF-diagonal.A_introduced_FP)
    reference2 = table("conjunction_robustness/reference/rq2_rows.csv.gz")
    for name in ["certified_tasks", "eager_errors"]:
        equal("matched reference RQ2 paired difference: " + name, reference2[name+"_delta"], reference2["mix_"+name]-reference2["srs_"+name])
    r1 = reference1[(reference1.dataset == "JB GPT-5.4") & (reference1.policy == "disagreement_then_random")
                    & reference1.budget_share.isin([.2,.84])].copy()
    r1["delta_per100"] = 100*r1.gated_minus_eager/r1.n_units
    r1 = r1.groupby(["acquisition_ref","evaluation_ref","budget_share"])[["actual_queries","gated_minus_eager","delta_per100"]].mean().reset_index()
    r2 = reference2[(reference2.dataset == "JB GPT-5.4") & (reference2.budget_share == .2)]
    r2_claims = []
    for key, group in r2.groupby(["acquisition_ref","evaluation_ref"]):
        r2_claims.append(dict(acquisition_ref=key[0], evaluation_ref=key[1], budget_cap=int(group.budget_cap.iloc[0]),
            certified_tasks_delta_mean=group.certified_tasks_delta.mean(), eager_errors_delta_mean=group.eager_errors_delta.mean(),
            error_reduction_orders=int((group.eager_errors_delta < 0).sum()), error_increase_orders=int((group.eager_errors_delta > 0).sum()), orders=len(group)))
    claims["reference_sensitivity"] = dict(source="conjunction_robustness/reference/{pairing.csv,frame_diagnostics.json,rq1_rows.csv.gz,rq2_rows.csv.gz}",
        unit="output-level frozen judge comparison, counts or errors per 100 outputs as named", counts=matched_counts,
        original_repeated_peer_diagnostics=diag["JB GPT-5.4"]["original_repeated_peer_diagnostics"],
        subset_ratios=dict(other_rater_fails_initial_FF=32/54, all_other_raters_agree_reference_PASS=22/56),
        rq1_31_order_means=r1.to_dict("records"), rq2_31_order_means=r2_claims,
        warning="A and B are symmetric hash-selected references. Off-diagonal cells freeze acquisition reference and reevaluate; diagonal cells rerun acquisition.")
    binary_matched = diagonal[(diagonal.dataset == "JB GPT-5.4 | binary_only") & (diagonal.policy == "random_criterion") & (diagonal.budget_share == .2)]
    binary_means = binary_matched.groupby("evaluation_ref").gated_minus_eager.mean()
    equal("manuscript matched binary reference sign reversal", binary_means.loc[["A","B"]], [1,-1])
    claims["reference_sensitivity"]["binary_srs_20pct_diagonal_order_mean"] = binary_means.to_dict()

    cluster1 = table("conjunction_robustness/cluster/rq1_curves.csv.gz")
    cluster2 = table("conjunction_robustness/cluster/rq2_seed_pairs.csv.gz")
    frames = table("conjunction_robustness/cluster/rq2_frame_means.csv.gz")
    summary = document("conjunction_robustness/cluster/summary.json")
    framefields = [c[:-5] for c in frames if c.endswith("_mean")]
    recalculated = cluster2.groupby(["dataset","replicate"])[framefields].mean()
    compare = frames.set_index(["dataset","replicate"]).join(recalculated)
    for field in framefields:
        equal("cluster seed aggregation: " + field, compare[field+"_mean"], compare[field])
    for saved in summary["rq1"]:
        select = cluster1[(cluster1.dataset == saved["dataset"]) & (cluster1.policy == saved["policy"])]
        for label, field in [("count","expected_delta"),("per100","expected_delta_per100")]:
            pivot = select.pivot(index="replicate", columns="budget_share", values=field).sort_index(axis=1)
            equal("cluster original " + saved["dataset"] + "/" + saved["policy"] + "/" + label, pivot.loc[-1], saved[label]["original"])
            for prob, key in [(.025,"pointwise_q025"),(.5,"pointwise_median"),(.975,"pointwise_q975")]:
                equal("cluster quantile " + saved["dataset"] + "/" + saved["policy"] + "/" + label + "/" + key,
                      np.quantile(pivot.loc[pivot.index >= 0], prob, axis=0), saved[label][key])
    cluster_claims1, cluster_claims2 = [], []
    for (name, share), group in cluster1[(cluster1.policy == "disagreement_then_random") & cluster1.budget_share.isin([.2,.84])].groupby(["dataset","budget_share"]):
        sampled = group[group.replicate >= 0]
        cluster_claims1.append(dict(dataset=name, budget_share=share, original_expected_delta_per100=float(group.loc[group.replicate == -1,"expected_delta_per100"].iloc[0]),
            composition_quantiles_per100=np.quantile(sampled.expected_delta_per100,[.025,.5,.975]).tolist(), replicates=len(sampled)))
    for name, group in frames.groupby("dataset", sort=False):
        sampled = group[group.replicate >= 0]
        row = dict(dataset=name, replicates=len(sampled), cert_gain_replicates=int((sampled.certified_tasks_delta_mean > 0).sum()),
            error_reduction_replicates=int((sampled.residual_errors_delta_mean < 0).sum()),
            error_increase_replicates=int((sampled.residual_errors_delta_mean > 0).sum()),
            error_tie_replicates=int((sampled.residual_errors_delta_mean == 0).sum()),
            original_error_delta_per100=float(group.loc[group.replicate == -1,"residual_errors_delta_per100_mean"].iloc[0]),
            error_composition_quantiles_per100=np.quantile(sampled.residual_errors_delta_per100_mean,[.025,.5,.975]).tolist())
        saved = next(x for x in summary["rq2"] if x["dataset"] == name)
        equal("cluster RQ2 sign counts: " + name, [row["cert_gain_replicates"], row["error_reduction_replicates"]], [saved["cert_gain_replicates"],saved["error_reduction_replicates"]])
        cluster_claims2.append(row)
    claims["cluster_sensitivity"] = dict(source="conjunction_robustness/cluster/{rq1_curves.csv.gz,rq2_seed_pairs.csv.gz,rq2_frame_means.csv.gz,summary.json}",
        unit="errors per 100 evaluation units; empirical 2.5%, 50%, 97.5% task-composition quantiles", rq1=cluster_claims1, rq2=cluster_claims2,
        warning="399 base-task composition resamples; labels and rater population stay fixed. These are sensitivity ranges, not an unqualified new-population confidence claim.")

    estimation = table("conjunction_robustness/estimation/stratified_rows.csv.gz")
    intervals = module_at(root / "src/finite_population_intervals.py", "manuscript_intervals")
    equal("appendix inclusive floating-point boundary example", intervals.hypergeom_ci(16,2,0,.05), [0,13])
    require("appendix endpoint exact integer tail", math.comb(3,2)*40 == math.comb(16,2))
    saved_estimation = table("conjunction_robustness/estimation/stratified_summary.csv").set_index(["dataset","policy","budget_share"])
    estimation_claims = []
    for key, group in estimation.groupby(["dataset","policy","budget_share"]):
        values = dict(bias=group.accuracy_error.mean(), rmse=np.sqrt(np.mean(group.accuracy_error**2)),
            exact_ci_width_mean=group.accuracy_ci_width.mean(), observed_coverage=group.accuracy_ci_covers.mean(),
            approximate_normal_width_mean=group.normal_ci_width.mean(),
            criterion_errors_found_mean=group.criterion_errors_found.mean(), certified_tasks_mean=group.certified_tasks.mean(),
            residual_errors_mean=group.residual_errors.mean(), pilot_queries_mean=group.pilot_queries.mean())
        if math.isnan(values["approximate_normal_width_mean"]):
            require("normal approximation explicitly unavailable: " + str(key),
                    math.isnan(saved_estimation.loc[key,"approximate_normal_width_mean"]))
            values.pop("approximate_normal_width_mean")
        equal("estimation summary: " + str(key), list(values.values()), [saved_estimation.loc[key,k] for k in values])
        if key[2] == .2:
            estimation_claims.append(dict(dataset=key[0], policy=key[1], budget_share=key[2],
                queries=int(group.actual_queries.iloc[0]), rmse_pp=100*values["rmse"],
                guaranteed_interval_mean_width_pp=100*values["exact_ci_width_mean"],
                approximate_normal_mean_width_pp=100*values["approximate_normal_width_mean"], pilot_queries_mean=values["pilot_queries_mean"]))
    claims["estimation"] = dict(source="conjunction_robustness/estimation/{stratified_rows.csv.gz,stratified_summary.csv}",
        selector="budget_share=0.2, 31 order seeds", unit="criterion microagreement percentage points", values=estimation_claims,
        warning="Guaranteed simultaneous-stratum construction and approximate normal intervals have different coverage guarantees. RMSE is across fixed-frame random query orders.")
    claims["estimation"]["inclusive_boundary_example"] = dict(population=16,sample=2,observed_successes=0,alpha=.05,count_interval=[0,13],upper_endpoint_lower_tail="3/120 = 1/40")

    tasks = table("conjunction_label_value/whole_task_sampling.csv.gz")
    errors, subset_ok, verdict_ok = [], [], []
    task_vectors = {name: unit_vectors(data) for name, data in datasets.items()}
    for row in tasks.itertuples():
        data = datasets[row.dataset]
        slots, g, _ = task_vectors[row.dataset]
        selected = json.loads(row.selected_task_indices)
        queries = json.loads(row.short_circuit_query_indices)
        queried = np.zeros(len(data["gold"]), dtype=bool)
        queried[queries] = True
        subset_ok.append(len(set(selected)) == row.sample_n and len(set(queries)) == len(queries)
                         and set(data["task_index"][queries]).issubset(selected))
        verdict_ok.append(all(queried[slots[i]].all() if g[i]
                              else np.any(queried[slots[i]] & (data["gold"][slots[i]] == 0))
                              for i in selected))
        p = float(g.mean()); n = row.sample_n; N = len(g)
        exact_rmse = math.sqrt(p*(1-p)*(N-n)/(n*(N-1)))
        errors.append([row.full_rubric_queries-sum(len(slots[i]) for i in selected),
                       row.short_circuit_queries-len(queries), row.sample_reference_pass-int(g[selected].sum()),
                       row.task_pass_estimate-float(g[selected].mean()), row.task_pass_exact_design_rmse-exact_rmse])
    equal("whole-task cost, pass estimate and exact design RMSE", errors, np.zeros((len(errors),5)))
    require("whole-task subset sizes", all(subset_ok), len(subset_ok))
    require("short-circuit observed pass agrees with full reference", all(verdict_ok), len(verdict_ok))
    tasks_focal = tasks[tasks.requested_task_share == .2]
    task_values = tasks_focal.groupby("dataset",sort=False)[["sample_n","population_n","full_rubric_queries","short_circuit_queries","task_pass_exact_design_rmse"]].median().reset_index()
    task_values["task_pass_exact_design_rmse_pp"] = 100*task_values.pop("task_pass_exact_design_rmse")
    claims["whole_task_sampling"] = dict(source="conjunction_label_value/whole_task_sampling.csv.gz",
        selector="requested_task_share=0.2; costs are separate marginal medians over seeds 1 through 31", unit="label queries and task-pass percentage points",
        values=task_values.to_dict("records"), warning="Same sampled task set and pass estimator; different realized label costs. This is not an equal-label-budget comparison.")

    disclosure = document("information_disclosure_replay.json")
    witness_module = module_at(root / "src/release_validation.py", "manuscript_witness")
    witness_module.ROOT = root
    witness_check = witness_module.verify_disclosure_witnesses(disclosure)
    checks.append(dict(check="literal disclosure endpoint witness constraints", passed=True, **witness_check))
    disclosure_claims = []
    for run in disclosure["ruverbench"]:
        for order in sorted({r["order"] for r in run["task_link_rows"]}):
            rows = sorted([r for r in run["task_link_rows"] if r["order"] == order], key=lambda r:r["requested_fraction"])
            lower = np.array([r["pass_rate_bounds_pp"][0] for r in rows])
            upper = np.array([r["pass_rate_bounds_pp"][1] for r in rows])
            require("nested disclosure: " + run["run"] + "/" + str(order), (np.diff(lower) >= -1e-8).all() and (np.diff(upper) <= 1e-8).all(), len(rows))
            equal("complete reference-link recovery: " + run["run"] + "/" + str(order), rows[-1]["pass_rate_bounds_pp"], [run["reference_pass_pp"]]*2)
        releases = [run["baseline"],run["category_disclosure"],*run["task_link_rows"]]
        require("disclosure obtains no new labels: " + run["run"], all(r["new_reference_labels"] == 0 for r in releases), len(releases))
        equal("linear control preserved: " + run["run"], [r["linear_reference_score_pp"] for r in releases], [run["reference_mean_criteria_pp"]]*len(releases))
        disclosure_claims.append(dict(dataset=run["run"], n_tasks=run["n_tasks"], n_criteria=run["n_criteria"],
            reference_pass_pp=run["reference_pass_pp"], judge_pass_pp=run["predicted_pass_pp"],
            D0_bounds_pp=run["baseline"]["pass_rate_bounds_pp"], D0_width_pp=run["baseline"]["feasible_width_pp"],
            D1_bounds_pp=run["category_disclosure"]["pass_rate_bounds_pp"], D1_width_pp=run["category_disclosure"]["feasible_width_pp"],
            D0_correction_sign=run["baseline"]["correction_sign"], full_disclosure_width_pp=0))
        disclosure_claims[-1]["D0_count_cells"] = run["baseline"]["n_disclosed_count_cells"]
        if run["run"] == "DR Gemini 3.1 Pro":
            equal("manuscript disclosed count cells", [run["baseline"]["n_disclosed_count_cells"]], [38])
    claims["disclosure"] = dict(source="information_disclosure_replay.json; data/processed/public_judge_predictions.csv",
        unit="finite feasible task-pass percentage points", values=disclosure_claims,
        warning=witness_check["scope"])

    return clean(dict(schema_version=1, kind="manuscript numerical claim audit", status="PASS",
        method="Independent aggregation and combinatorial calculation from released fixed inputs and saved rows. No scientific experiments or optimizer reruns.",
        scope_limits=["Does not adjudicate reference truth or assert generalization beyond the stated designs.",
                      "Does not by itself certify prose completeness, citation accuracy, or correspondence of every manuscript number.",
                      "Shared input loader validates target conversion; numerical aggregation and analytical event calculation are separate from producer functions.",
                      "Disclosure feasibility is independently checked by existing literal witness validator; endpoint optimality uses saved closed solver bounds."],
        sources=source_hashes, checks=checks, claims=claims))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Defaults to PACKAGE_ROOT/artifacts/paper-checks/numerical_claim_audit.json")
    args = parser.parse_args()
    args.output = args.output or args.package_root / "artifacts/paper-checks/numerical_claim_audit.json"
    destination = args.output.resolve()
    for relative in ("data", "artifacts/expected", "src"):
        protected = (args.package_root / relative).resolve()
        if destination == protected or protected in destination.parents:
            parser.error("output must not overwrite input data, expected scientific outputs, or source code")
    receipt = audit(args.package_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(status=receipt["status"], checks=len(receipt["checks"]), sources=len(receipt["sources"]), sections=list(receipt["claims"]))))


if __name__ == "__main__":
    main()
