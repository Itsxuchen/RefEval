"""Descriptive summaries of fixed-reference policy replays; no population tests.

Requires pandas from the project dependencies. Reads completed
study outputs only. Seeds are ordering sensitivity, never independent samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/reproduced/conjunction_policy_study"
KEYS = ["dataset", "task_order", "policy"]
COMPARE = ("random_criterion", "sc_natural", "sc_random", "disagreement_only", "disagreement_then_random", "sc_disagreement")


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def signs(a: np.ndarray, b: np.ndarray) -> dict:
    return {"points": len(a), "more_certified_more_errors": int(((a > 0) & (b > 0)).sum()),
            "fewer_certified_fewer_errors": int(((a < 0) & (b < 0)).sum()),
            "certification_residual_conflict": int((a * b > 0).sum()),
            "certified_tie": int((a == 0).sum()), "residual_tie": int((b == 0).sum()),
            "more_certified_fewer_errors": int(((a > 0) & (b < 0)).sum()),
            "fewer_certified_more_errors": int(((a < 0) & (b > 0)).sum())}


def contrast(frame: pd.DataFrame, selector: str, first: str, second: str, keys: list[str]) -> pd.DataFrame:
    cols = ["certified_tasks", "residual_errors", "criterion_errors_found", "actual_queries", "budget_cap", "n_tasks"]
    a = frame.loc[frame[selector] == first, keys + cols]
    b = frame.loc[frame[selector] == second, keys + cols]
    out = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    for field in ("certified_tasks", "residual_errors", "criterion_errors_found"):
        out["delta_" + field] = out[field + "_a"] - out[field + "_b"]
        if field != "criterion_errors_found":
            out["delta_" + field + "_pp"] = 100 * out["delta_" + field] / out["n_tasks_a"]
    out["equal_actual_active"] = (out.actual_queries_a == out.actual_queries_b) & (out.actual_queries_a == out.budget_cap_a)
    return out


def contrast_summary(frame: pd.DataFrame, groups: list[str]) -> list[dict]:
    result = []
    for key, part in frame.groupby(groups, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        out = dict(zip(groups, key))
        out.update(signs(part.delta_certified_tasks.to_numpy(), part.delta_residual_errors.to_numpy()))
        out["equal_actual_active_points"] = int(part.equal_actual_active.sum())
        for metric in ("certified_tasks", "residual_errors"):
            for suffix in ("", "_pp"):
                series = part["delta_" + metric + suffix]
                out["delta_" + metric + suffix + "_min"] = float(series.min())
                out["delta_" + metric + suffix + "_max"] = float(series.max())
        if "removed_base" in part:
            out["distinct_base_removals"] = int(part.removed_base.nunique())
            flag = part.delta_certified_tasks * part.delta_residual_errors > 0
            out["base_removals_with_any_conflict"] = int(part.loc[flag, "removed_base"].nunique())
        result.append(out)
    return result


def dominance_summary(frame: pd.DataFrame, groups: list[str]) -> list[dict]:
    """Report observed same-actual-cost dominance, never a deployable oracle."""
    result = []
    for key, whole in frame.groupby(groups, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        part = whole[whole.equal_actual_active]
        dc, de, db = (part["delta_" + field] for field in ("certified_tasks", "residual_errors", "criterion_errors_found"))
        row.update(comparison_points=len(whole), equal_actual_active_points=len(part),
                   a_weakly_dominates_three=int(((dc >= 0) & (de <= 0) & (db >= 0) & ((dc > 0) | (de < 0) | (db > 0))).sum()),
                   b_weakly_dominates_three=int(((dc <= 0) & (de >= 0) & (db <= 0) & ((dc < 0) | (de > 0) | (db < 0))).sum()),
                   a_strictly_dominates_three=int(((dc > 0) & (de < 0) & (db > 0)).sum()),
                   a_more_certified=int((dc > 0).sum()), a_fewer_residual=int((de < 0).sum()), a_more_bits=int((db > 0).sum()),
                   all_three_tie=int(((dc == 0) & (de == 0) & (db == 0)).sum()),
                   certification_residual_conflict=int((dc * de > 0).sum()), certification_bit_conflict=int((dc * db < 0).sum()))
        for field in ("certified_tasks", "residual_errors", "criterion_errors_found"):
            values = part["delta_" + field]
            for stat in ("min", "median", "max"):
                row["delta_" + field + "_" + stat] = None if values.empty else float(getattr(values, stat)())
        result.append(row)
    return result


def task_dominance_summary(frame: pd.DataFrame, groups: list[str]) -> list[dict]:
    """Cross-primary comparison: criterion-error counts are not comparable."""
    result = []
    for key, whole in frame.groupby(groups, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key)); part = whole[whole.equal_actual_active]
        dc, de = part.delta_certified_tasks, part.delta_residual_errors
        row.update(comparison_points=len(whole), equal_actual_active_points=len(part),
                   hybrid_mini_dominates_tasks=int(((dc >= 0) & (de <= 0) & ((dc > 0) | (de < 0))).sum()),
                   strong_primary_dominates_tasks=int(((dc <= 0) & (de >= 0) & ((dc < 0) | (de > 0))).sum()),
                   hybrid_more_certified=int((dc > 0).sum()), hybrid_fewer_residual=int((de < 0).sum()),
                   both_tasks_tie=int(((dc == 0) & (de == 0)).sum()), coverage_error_conflict=int((dc * de > 0).sum()))
        for field in ("certified_tasks", "residual_errors"):
            values = part["delta_" + field]
            for stat in ("min", "median", "max"):
                row["hybrid_minus_strong_" + field + "_" + stat] = None if values.empty else float(getattr(values, stat)())
        result.append(row)
    return result


def strongest_primary_contrast(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    a = frame[(frame.dataset == "JB GPT-5.4-mini") & (frame.policy == "sc_disagreement")].copy()
    b = frame[(frame.dataset == "JB GPT-5.4") & (frame.policy == "sc_judge")].copy()
    a["configuration"], b["configuration"] = "hybrid_mini", "strong_primary"
    result = contrast(pd.concat([a, b]), "configuration", "hybrid_mini", "strong_primary", keys)
    if not (result.n_tasks_a == result.n_tasks_b).all() or not (result.budget_cap_a == result.budget_cap_b).all():
        raise ValueError("Cross-primary comparison has different task population or budget")
    return result.drop(columns=[c for c in result if "criterion_errors" in c])


def summarize(output: Path) -> dict:
    paths = [output / "study.json", output / "budget_curves.csv.gz", output / "leave_one_base_task_out.csv.gz"]
    study = json.loads(paths[0].read_text())
    curves = pd.read_csv(paths[1], dtype={"dataset": "category", "task_order": "category", "policy": "category"})
    if len(curves) != study["counts"]["budget_rows"] or curves.duplicated(KEYS + ["seed", "budget_share"]).any():
        raise ValueError("Incomplete or duplicated budget curve rows")
    ends = pd.DataFrame(study["endpoints"])
    random_ends = ends[ends.seed > 0]
    endpoints = []
    for key, part in random_ends.groupby(KEYS, observed=True):
        cost = part.first_all_certified.dropna()
        row = dict(zip(KEYS, key))
        row.update(randomized_seed_count=len(part), full_certification_seed_count=len(cost),
                   new_fp_ever_seeds=int((part.new_fp_events_total > 0).sum()),
                   new_fp_events_min=int(part.new_fp_events_total.min()), new_fp_events_max=int(part.new_fp_events_total.max()),
                   new_fp_events_median=float(part.new_fp_events_total.median()),
                   max_simultaneous_introduced_fp=int(part.maximum_simultaneous_introduced_fp.max()))
        for name, percentile in (("min", 0), ("p05", .05), ("median", .5), ("p95", .95), ("max", 1)):
            row["full_certification_queries_" + name] = None if cost.empty else float(cost.quantile(percentile))
        endpoints.append(row)
    pd.DataFrame(endpoints).to_csv(output / "order_sensitivity_endpoints.csv", index=False)

    anchor = curves[(curves.seed == 0) & (curves.budget_share == .2)]
    anchor.to_csv(output / "canonical_20pct.csv", index=False)
    sc = curves[curves.policy == "sc_judge"]
    order = contrast(sc, "task_order", "release", "predicted_fail_count", ["dataset", "seed", "budget_share"])
    order_anchor = order[(order.seed == 0) & (order.budget_share == .2)]
    order_anchor.to_csv(output / "task_order_canonical_20pct.csv", index=False)
    order_random20 = contrast_summary(order[(order.seed > 0) & (order.budget_share == .2)], ["dataset"])
    order_random_grid = contrast_summary(order[(order.seed > 0) & (order.budget_share > 0) & (order.budget_share < 1)], ["dataset"])
    order_seed0_grid = contrast_summary(order[(order.seed == 0) & (order.budget_share > 0) & (order.budget_share < 1)], ["dataset"])

    # Delete-one analyses preserve each trace's priority order and recalculate
    # the budget denominator. They are influence diagnostics, not confidence CIs.
    loto_columns = ["dataset", "task_order", "policy", "removed_base", "budget_share", "certified_tasks", "residual_errors",
                    "criterion_errors_found", "actual_queries", "budget_cap", "n_tasks"]
    loto = pd.read_csv(paths[2], usecols=loto_columns, dtype={"dataset": "category", "task_order": "category", "policy": "category", "removed_base": "category"})
    if len(loto) != study["counts"]["deletion_rows"] or loto.duplicated(KEYS + ["removed_base", "budget_share"]).any():
        raise ValueError("Incomplete or duplicated leave-one-base rows")
    loto_order = contrast(loto[loto.policy == "sc_judge"], "task_order", "release", "predicted_fail_count",
                          ["dataset", "removed_base", "budget_share"])
    l20 = loto_order[loto_order.budget_share == .2].copy()
    canonical = order_anchor.set_index("dataset")
    l20["baseline_delta_cert"] = l20.dataset.map(canonical.delta_certified_tasks).astype(float)
    l20["baseline_delta_error"] = l20.dataset.map(canonical.delta_residual_errors).astype(float)
    l20["preserves_both_signs"] = (np.sign(l20.delta_certified_tasks) == np.sign(l20.baseline_delta_cert)) & (np.sign(l20.delta_residual_errors) == np.sign(l20.baseline_delta_error))
    l20["reverses_both_nonzero_signs"] = (l20.baseline_delta_cert != 0) & (l20.baseline_delta_error != 0) & (np.sign(l20.delta_certified_tasks) == -np.sign(l20.baseline_delta_cert)) & (np.sign(l20.delta_residual_errors) == -np.sign(l20.baseline_delta_error))
    loto20 = contrast_summary(l20, ["dataset"])
    for row in loto20:
        part = l20[l20.dataset == row["dataset"]]
        row["preserves_both_canonical_signs"] = int(part.preserves_both_signs.sum())
        row["reverses_both_nonzero_canonical_signs"] = int(part.reverses_both_nonzero_signs.sum())
        row["at_least_one_tie"] = int(((part.delta_certified_tasks == 0) | (part.delta_residual_errors == 0)).sum())
    pd.DataFrame(loto20).to_csv(output / "leave_one_base_20pct_summary.csv", index=False)
    loto_order_grid = contrast_summary(loto_order[(loto_order.budget_share > 0) & (loto_order.budget_share < 1)], ["dataset"])
    loto_pairs = []
    for other in COMPARE:
        pair = contrast(loto, "policy", "sc_judge", other, ["dataset", "task_order", "removed_base", "budget_share"])
        active = pair[(pair.budget_share > 0) & (pair.budget_share < 1) & pair.equal_actual_active]
        for row in contrast_summary(active, ["dataset", "task_order"]):
            row.update(a="sc_judge", b=other)
            loto_pairs.append(row)

    selected_curves = curves[(curves.seed > 0) & (curves.budget_share == .2)]
    hybrid = contrast(selected_curves, "policy", "sc_disagreement", "sc_judge", ["dataset", "task_order", "seed", "budget_share"])
    hybrid_random20 = dominance_summary(hybrid, ["dataset", "task_order"])
    hybrid_loto = contrast(loto[loto.budget_share == .2], "policy", "sc_disagreement", "sc_judge", ["dataset", "task_order", "removed_base", "budget_share"])
    hybrid_loto20 = dominance_summary(hybrid_loto, ["dataset", "task_order"])
    pd.DataFrame(hybrid_random20).to_csv(output / "sc_disagreement_vs_judge_random20.csv", index=False)
    pd.DataFrame(hybrid_loto20).to_csv(output / "sc_disagreement_vs_judge_loto20.csv", index=False)
    cross = strongest_primary_contrast(curves, ["task_order", "seed", "budget_share"])
    cross20 = cross[(cross.seed > 0) & (cross.budget_share == .2)]
    cross_random20 = task_dominance_summary(cross20, ["task_order"])
    cross_anchor20 = records(cross[(cross.seed == 0) & (cross.budget_share == .2)])
    cross_grid = task_dominance_summary(cross[(cross.budget_share > 0) & (cross.budget_share < 1)], ["task_order"])
    cross_loto = strongest_primary_contrast(loto[loto.budget_share == .2], ["task_order", "removed_base", "budget_share"])
    cross_loto20 = task_dominance_summary(cross_loto, ["task_order"])
    pd.DataFrame(cross_random20).to_csv(output / "hybrid_mini_vs_strong_primary_random20.csv", index=False)
    pd.DataFrame(cross_loto20).to_csv(output / "hybrid_mini_vs_strong_primary_loto20.csv", index=False)
    fallback = contrast(selected_curves, "policy", "sc_judge", "disagreement_then_random", ["dataset", "task_order", "seed", "budget_share"])
    fallback_random20 = dominance_summary(fallback, ["dataset", "task_order"])

    eager_gated = []
    for key, part in curves.groupby(KEYS, observed=True):
        diff = part.gated_residual_errors - part.residual_errors
        row = dict(zip(KEYS, key))
        row.update(points=len(part), gated_fewer_errors_points=int((diff < 0).sum()),
                   gated_more_errors_points=int((diff > 0).sum()), gated_equal_errors_points=int((diff == 0).sum()),
                   max_gated_minus_eager_errors=int(diff.max()), min_gated_minus_eager_errors=int(diff.min()),
                   max_introduced_fp_on_grid=int(part.introduced_fp.max()),
                   max_delayed_ff_corrections_on_grid=int(part.delayed_ff_corrections.max()))
        eager_gated.append(row)
    pd.DataFrame(eager_gated).to_csv(output / "eager_gated_grid_summary.csv", index=False)

    # Optimum sets are hindsight score comparisons at the same permitted cap,
    # not a controller with access to unqueried reference labels.
    interior = curves[(curves.budget_share > 0) & (curves.budget_share < 1)].copy()
    gkeys = ["dataset", "task_order", "seed", "budget_share"]
    groups = interior.groupby(gkeys, observed=True)
    interior["best_cert"] = interior.certified_tasks == groups.certified_tasks.transform("max")
    interior["best_bits"] = interior.criterion_errors_found == groups.criterion_errors_found.transform("max")
    interior["best_error"] = interior.residual_errors == groups.residual_errors.transform("min")
    interior["common_cert_bits"] = interior.best_cert & interior.best_bits
    interior["common_cert_error"] = interior.best_cert & interior.best_error
    interior["common_bits_error"] = interior.best_bits & interior.best_error
    interior["common_all"] = interior.best_cert & interior.best_bits & interior.best_error
    interior["spends_cap"] = interior.actual_queries == interior.budget_cap
    interior["not_all_certified"] = interior.certified_tasks < interior.n_tasks
    fields = ["common_cert_bits", "common_cert_error", "common_bits_error", "common_all"]
    common = interior.groupby(gkeys, observed=True)[fields].any().reset_index()
    active = interior.groupby(gkeys, observed=True)[["spends_cap", "not_all_certified"]].all().reset_index()
    common = common.merge(active, on=gkeys, validate="one_to_one")
    best = []
    for key, part in common.groupby(["dataset", "task_order"], observed=True):
        row = dict(zip(["dataset", "task_order"], key)); row["shared_cap_points"] = len(part)
        for field in fields:
            row["no_" + field + "_optimum_points"] = int((~part[field]).sum())
        row["all_seven_policies_spend_cap_points"] = int(part.spends_cap.sum())
        row["all_seven_active_and_incomplete_points"] = int((part.spends_cap & part.not_all_certified).sum())
        for field in fields:
            row["active_no_" + field + "_optimum_points"] = int((~part[field] & part.spends_cap & part.not_all_certified).sum())
        best.append(row)
    pd.DataFrame(best).to_csv(output / "objective_optimum_sets.csv", index=False)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {os.path.relpath(p, ROOT): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + [Path(__file__)]},
        "denominators": {"datasets": len(study["datasets"]), "policies": len(study["policies"]),
                         "task_orders": len(study["task_orders"]), "seeds_total": study["seeds"], "randomized_seeds": study["seeds"] - 1,
                         "budget_grid_points": len(study["grid"]), "interior_grid_points": 99,
                         "curve_rows": len(curves), "loto_rows": len(loto), "canonical_20_rows": len(anchor)},
        "interpretation": ["All are exact or descriptive finite replay summaries; random ordering seeds are not independent empirical datasets or confidence intervals.",
                           "Pairwise policy comparisons require equal actual queries equal to the cap; shared-cap optimum comparisons permit early stopping and expose unused budget.",
                           "Task-order contrast is release minus predicted_fail_count; positive certification and positive residual differences express a coverage-error tradeoff.",
                           "LOTO removes one whole base task from the canonical transcript, preserves priorities and recalculates the subset budget. It is an influence diagnostic.",
                           "Min/max and 5th/95th percentiles summarize the 31 randomized priority seeds only, not population uncertainty.",
                           "The ex-post best set is calculated using hidden-reference evaluation and is not a deployable selection rule.",
                           "All-interior-grid optimum conflicts can be mechanical after short-circuit policies stop. The active-no-common columns require all seven policies to spend the cap and still leave at least one task uncertified.",
                           "Eager backfilling corrects every queried bit immediately. Certificate-gated reporting changes an initial task verdict only when reference-certified; it avoids new false passes but can delay fixing false fails.",
                           "All full versus binary-only JB comparisons change the scoring target; no universal policy ranking is claimed."],
        "canonical_20pct": records(anchor), "randomized_endpoint_summary": endpoints,
        "pairwise_equal_actual": study["pairwise_equal_actual"],
        "task_order_canonical_20pct": records(order_anchor), "task_order_randomized_20pct": order_random20,
        "task_order_randomized_interior_grid": order_random_grid, "task_order_canonical_interior_grid": order_seed0_grid,
        "loto_task_order_20pct": loto20, "loto_task_order_interior_grid": loto_order_grid,
        "loto_equal_actual_policy_pairs_interior_grid": loto_pairs,
        "sc_disagreement_vs_judge_definition": "a=sc_disagreement; b=sc_judge; all dominance counts require equal actual queries equal to budget cap; disagreement uses extra secondary judge run(s)",
        "sc_disagreement_vs_judge_randomized_20pct": hybrid_random20,
        "sc_disagreement_vs_judge_loto_20pct": hybrid_loto20,
        "sc_judge_vs_disagreement_fallback_randomized_20pct": fallback_random20,
        "strongest_primary_baseline": {
            "definition": "Compare full-JB sc_disagreement using mini primary and strong GPT secondary (a) against sc_judge using strong GPT as primary (b). Same reference labels and tasks. Compare certification and absolute residual error only; criterion errors differ by primary and are not comparable. Predicted-fail-count task orders are computed from each configuration's own primary. Grid counts pool seed0 and seeds1-31; random20 excludes seed0. No claim the hybrid beats the strongest available primary.",
            "canonical_20pct": cross_anchor20, "randomized_20pct": cross_random20,
            "loto_canonical_20pct": cross_loto20, "interior_grid_all_seeds": cross_grid,
        },
        "eager_gated_interior_and_endpoint_grid": eager_gated,
        "ex_post_objective_optimum_sets_interior_grid": best,
    }
    (output / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = summarize(args.output.resolve())
    print(json.dumps({"denominators": result["denominators"], "task_order_randomized_20pct": result["task_order_randomized_20pct"],
                      "loto_task_order_20pct": result["loto_task_order_20pct"]}, indent=2))


if __name__ == "__main__":
    main()
