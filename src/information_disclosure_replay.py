"""Replay observed reference-link and cross-arm-link disclosure, without new labels.

RuVerBench identified sets are integer-sharp finite-population feasibility results.
JudgmentBench covariance bounds are sharp over permutations; the derived Wald
intervals are only an exploratory superpopulation diagnostic, not exact coverage.
No independence, error-rate extrapolation, new judges, or human-time model is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/processed/public_judge_predictions.csv"
FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
BASE_SEED = 20260907


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference_runs(path: Path = PREDICTIONS) -> dict[str, list[dict]]:
    """Fail closed on identities, ordering, labels, or cross-judge reference drift."""
    runs: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    seen_criterion_ids = set()
    reference = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if not all(raw.get(key, "").strip() for key in
                       ("run", "domain", "task_id", "criterion_id", "category")):
                raise ValueError("missing identity or category")
            try:
                slot = int(raw["criterion_index"])
            except (ValueError, KeyError) as exc:
                raise ValueError("invalid criterion index") from exc
            if slot < 0:
                raise ValueError("malformed criterion identity")
            identity = (raw["run"], raw["domain"], raw["task_id"], slot)
            if identity in seen:
                raise ValueError("duplicate criterion identity")
            seen.add(identity)
            named_identity = (raw["run"], raw["domain"], raw["task_id"], raw["criterion_id"])
            if named_identity in seen_criterion_ids:
                raise ValueError("duplicate named criterion identity")
            seen_criterion_ids.add(named_identity)
            if raw["gold"] not in ("True", "False") or raw["prediction"] not in ("True", "False"):
                raise ValueError("non-binary reference or prediction")
            ref_id = identity[1:]
            ref_value = (raw["gold"], raw["category"], raw["criterion_id"])
            if ref_id in reference and reference[ref_id] != ref_value:
                raise ValueError("reference differs between judge runs")
            reference[ref_id] = ref_value
            runs[raw["run"]].append({
                "task_id": f"{raw['domain']}/{raw['task_id']}",
                "criterion_id": raw["criterion_id"], "slot": slot,
                "category": raw["category"], "prediction": int(raw["prediction"] == "True"),
                "gold": int(raw["gold"] == "True"),
            })
    if not runs:
        raise ValueError("empty reference table")
    for rows in runs.values():
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["task_id"]].append(row)
        for task_rows in by_task.values():
            if sorted(row["slot"] for row in task_rows) != list(range(len(task_rows))):
                raise ValueError("noncontiguous or duplicate criterion slots")
            for row in task_rows:
                row["k"] = len(task_rows)
        rows.sort(key=lambda row: (row["task_id"], row["slot"]))
    return dict(sorted(runs.items()))


def disclose_reference(rows: list[dict], by_category: bool,
                       revealed_tasks: set[str] | None = None) -> dict:
    """Publisher constructs the permitted release; solver never receives hidden gold.

    D0: k x predicted-label reference-success counts. D1: refine each group by
    observed category. These are released gold/prediction confusion margins, with
    total group sizes already determined by the visible prediction slot map.
    """
    revealed_tasks = set() if revealed_tasks is None else set(revealed_tasks)
    known_tasks = {row["task_id"] for row in rows}
    if not revealed_tasks <= known_tasks:
        raise ValueError("unknown revealed task")
    groups = defaultdict(lambda: {"slots": [], "reference_successes": 0})
    visible, links = [], {}
    for index, row in enumerate(rows):
        if row["gold"] not in (0, 1) or row["prediction"] not in (0, 1):
            raise ValueError("non-binary labels")
        visible.append({key: row[key] for key in ("task_id", "slot", "k", "category", "prediction")})
        key = (row["k"], row["prediction"], row["category"] if by_category else "all")
        groups[key]["slots"].append(index)
        groups[key]["reference_successes"] += row["gold"]
        if row["task_id"] in revealed_tasks:
            links[index] = row["gold"]
    return {"visible_slots": visible, "groups": list(groups.values()),
            "reference_links": links, "category_margins": by_category}


def conjunction_bounds(release: dict, time_limit: float = 30.0) -> dict:
    """Integer-sharp reference Pass@1 extrema using only an observed release.

    Binary x_i denotes an unknown reference label and binary y_t the conjunction.
    y_t <= x_i for every slot, and sum_i x_i - y_t <= k_t - 1. Marginal counts
    and revealed x_i are equalities. The program separates exactly over k.
    """
    visible = release["visible_slots"]
    n_criteria = len(visible)
    if not visible:
        raise ValueError("empty disclosure")
    tasks = defaultdict(list)
    seen = set()
    for index, row in enumerate(visible):
        identity = (row["task_id"], row["slot"])
        if identity in seen:
            raise ValueError("duplicate visible slot")
        seen.add(identity)
        tasks[row["task_id"]].append(index)
    for slots in tasks.values():
        if (sorted(visible[i]["slot"] for i in slots) != list(range(len(slots)))
                or any(visible[i]["k"] != len(slots) for i in slots)):
            raise ValueError("inconsistent task size or slots")
        slots.sort(key=lambda index: visible[index]["slot"])
    covered = []
    for group in release["groups"]:
        covered.extend(group["slots"])
        count = group["reference_successes"]
        if int(count) != count or not 0 <= count <= len(group["slots"]):
            raise ValueError("infeasible marginal count")
    if sorted(covered) != list(range(n_criteria)):
        raise ValueError("marginal groups must partition slots")
    for index, value in release["reference_links"].items():
        if index not in range(n_criteria) or value not in (0, 1):
            raise ValueError("invalid restored reference link")

    total_extrema = [0, 0]
    certificates = []
    for k in sorted({row["k"] for row in visible}):
        task_keys = sorted(task for task, slots in tasks.items() if len(slots) == k)
        slot_ids = [i for task in task_keys for i in tasks[task]]
        local = {slot: i for i, slot in enumerate(slot_ids)}
        m, nt = len(slot_ids), len(task_keys)
        group_rows = [group for group in release["groups"] if group["slots"][0] in local]
        if any(any(i not in local for i in group["slots"]) for group in group_rows):
            raise ValueError("marginal group crosses criterion-count strata")
        nvars = m + nt
        matrix = lil_matrix((m + nt + len(group_rows), nvars), dtype=float)
        lo = np.full(matrix.shape[0], -np.inf)
        hi = np.zeros(matrix.shape[0])
        ri = 0
        for ti, task in enumerate(task_keys):
            for index in tasks[task]:
                matrix[ri, m + ti] = 1
                matrix[ri, local[index]] = -1
                ri += 1
            for index in tasks[task]:
                matrix[ri, local[index]] = 1
            matrix[ri, m + ti] = -1
            hi[ri] = k - 1
            ri += 1
        for group in group_rows:
            for index in group["slots"]:
                matrix[ri, local[index]] = 1
            lo[ri] = hi[ri] = group["reference_successes"]
            ri += 1
        matrix = matrix.tocsc()
        lower, upper = np.zeros(nvars), np.ones(nvars)
        for index, value in release["reference_links"].items():
            if index in local:
                lower[local[index]] = upper[local[index]] = value
        objective = np.r_[np.zeros(m), np.ones(nt)]
        for endpoint, sign in enumerate((1.0, -1.0)):
            solution = milp(
                sign * objective, integrality=np.ones(nvars), bounds=Bounds(lower, upper),
                constraints=LinearConstraint(matrix, lo, hi),
                options={"mip_rel_gap": 0.0, "time_limit": time_limit},
            )
            if not solution.success or solution.status != 0:
                raise RuntimeError(f"integer optimum not certified: k={k}, {solution.message}")
            witness = np.rint(solution.x)
            lhs = matrix @ witness
            violation = max(float(np.max(np.maximum(lo - lhs, 0))),
                            float(np.max(np.maximum(lhs - hi, 0))),
                            float(np.max(np.maximum(lower - witness, 0))),
                            float(np.max(np.maximum(witness - upper, 0))))
            value = int(objective @ witness)
            if violation > 1e-8 or abs(solution.fun - sign * value) > 1e-7:
                raise RuntimeError("integer witness fails independent constraint check")
            if abs(solution.mip_dual_bound - solution.fun) > 1e-7:
                raise RuntimeError("integer endpoint optimality gap not closed")
            total_extrema[endpoint] += value
            certificates.append({"k": k, "endpoint": "lower" if endpoint == 0 else "upper",
                                 "pass_count": value, "constraint_violation": violation,
                                 "objective_gap": float(abs(solution.mip_dual_bound - solution.fun)),
                                 "reference_label_witness_bits": "".join(str(int(v)) for v in witness[:m]),
                                 "task_pass_witness_bits": "".join(str(int(v)) for v in witness[m:])})
    n = len(tasks)
    return {"pass_count_bounds": total_extrema,
            "pass_rate_bounds": [value / n for value in total_extrema],
            "n_tasks": n, "solver_certificates": certificates}


def decision_state(bounds: list[float] | tuple[float, float], margin: float = 0.01) -> str:
    lo, hi = bounds
    if lo > margin + 1e-12:
        return "A_better"
    if hi < -margin - 1e-12:
        return "B_better"
    if lo >= -margin - 1e-12 and hi <= margin + 1e-12:
        return "practically_equivalent"
    return "unresolved"


def disclose_pairing(a: np.ndarray, b: np.ndarray,
                     revealed_pairs: tuple[int, ...] | list[int] = ()) -> dict:
    """Release known marginal score multisets and observed selected pairs only."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (a.ndim != 1 or b.shape != a.shape or len(a) < 2
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
        raise ValueError("paired arrays must be finite, one-dimensional and equally sized")
    if len(set(revealed_pairs)) != len(revealed_pairs) or any(i not in range(len(a)) for i in revealed_pairs):
        raise ValueError("duplicate or unknown revealed pair")
    revealed = set(revealed_pairs)
    remaining = [i for i in range(len(a)) if i not in revealed]
    return {"a_scores": np.sort(a).tolist(), "b_scores": np.sort(b).tolist(),
            "revealed_score_pairs": [(float(a[i]), float(b[i])) for i in sorted(revealed)],
            "remaining_a_scores": np.sort(a[remaining]).tolist(),
            "remaining_b_scores": np.sort(b[remaining]).tolist()}


def pairing_bounds(release: dict) -> dict:
    """Sharp extrema from only marginal multisets and explicitly released pairs.

    Rearrangement attains both remaining product extrema; mean gap is unchanged.
    This function never receives hidden task identities or actual hidden pairing.
    """
    a, b = np.asarray(release["a_scores"]), np.asarray(release["b_scores"])
    n = len(a)
    aa, bb = np.sort(release["remaining_a_scores"]), np.sort(release["remaining_b_scores"])
    known = release["revealed_score_pairs"]
    if len(aa) != len(bb) or len(aa) + len(known) != n:
        raise ValueError("inconsistent disclosed score counts")
    if (not np.array_equal(np.sort([*aa, *(pair[0] for pair in known)]), np.sort(a))
            or not np.array_equal(np.sort([*bb, *(pair[1] for pair in known)]), np.sort(b))):
        raise ValueError("disclosed pairs inconsistent with marginal scores")
    fixed_product = float(sum(left * right for left, right in known))
    products = [fixed_product + float(aa @ bb[::-1]), fixed_product + float(aa @ bb)]
    covariance = [(value - n * float(a.mean() * b.mean())) / (n - 1) for value in products]
    variance_sum = float(a.var(ddof=1) + b.var(ddof=1))
    se = [math.sqrt(max(0.0, (variance_sum - 2 * cov) / n)) for cov in covariance[::-1]]
    gap = float(a.mean() - b.mean())
    intervals = [[gap - 1.96 * value, gap + 1.96 * value] for value in se]
    return {"n_tasks": n, "revealed_pairs": len(known), "mean_gap": gap,
            "mean_gap_bounds": [gap, gap], "covariance_bounds": covariance,
            "paired_se_bounds": se, "wald95_at_min_se": intervals[0],
            "wald95_at_max_se": intervals[1],
            "state_at_min_se": decision_state(intervals[0]),
            "state_at_max_se": decision_state(intervals[1]),
            "robust_exploratory_state": decision_state(intervals[1])}


def run_ruverbench(runs: dict[str, list[dict]], repeats: int) -> list[dict]:
    output = []
    for run_index, (run, rows) in enumerate(runs.items()):
        started = time.perf_counter()
        task_rows = defaultdict(list)
        for row in rows:
            task_rows[row["task_id"]].append(row)
        task_ids = sorted(task_rows)
        n = len(task_ids)
        gold_pass = np.asarray([all(row["gold"] for row in task_rows[task]) for task in task_ids])
        pred_pass = np.asarray([all(row["prediction"] for row in task_rows[task]) for task in task_ids])
        gold_mean = float(np.mean([np.mean([row["gold"] for row in task_rows[task]]) for task in task_ids]))
        pred_mean = float(np.mean([np.mean([row["prediction"] for row in task_rows[task]]) for task in task_ids]))
        truth = float(gold_pass.mean())
        pred = float(pred_pass.mean())
        truth_sign = "positive" if truth > pred else "negative" if truth < pred else "zero"

        def evaluate(stage: str, category: bool, selected: set[str], order: int | None,
                     fraction: float) -> dict:
            release = disclose_reference(rows, category, selected)
            result = conjunction_bounds(release)
            lo, hi = result["pass_rate_bounds"]
            if not lo - 1e-10 <= truth <= hi + 1e-10:
                raise AssertionError("true conjunction excluded")
            # The control is reconstructed only from disclosed group totals.
            linear = sum(group["reference_successes"] /
                         release["visible_slots"][group["slots"][0]]["k"]
                         for group in release["groups"]) / n
            if abs(linear - gold_mean) > 1e-10:
                raise AssertionError("linear control changed under link hiding")
            correction = [100 * (lo - pred), 100 * (hi - pred)]
            sign = ("positive" if correction[0] > 1e-9 else
                    "negative" if correction[1] < -1e-9 else
                    "zero" if max(abs(value) for value in correction) < 1e-9 else "unresolved")
            return {"stage": stage, "order": order, "requested_fraction": fraction,
                    "restored_tasks": len(selected), "restored_task_fraction": len(selected) / n,
                    "restored_reference_links": len(release["reference_links"]),
                    "new_reference_labels": 0, "n_disclosed_count_cells": len(release["groups"]),
                    "pass_rate_bounds_pp": [100 * lo, 100 * hi],
                    "correction_bounds_pp": correction, "correction_sign": sign,
                    "feasible_width_pp": 100 * (hi - lo),
                    "linear_reference_score_pp": 100 * linear,
                    "linear_correction_pp": 100 * (linear - pred_mean),
                    "solver_certificates": result["solver_certificates"]}

        baseline = evaluate("D0_k_confusion", False, set(), None, 0)
        category = evaluate("D1_k_category_confusion", True, set(), None, 0)
        if category["feasible_width_pp"] > baseline["feasible_width_pp"] + 1e-8:
            raise AssertionError("category disclosure expands feasible set")
        full = evaluate("D2_task_links", True, set(task_ids), None, 1)
        if full["feasible_width_pp"] > 1e-8:
            raise AssertionError("complete reference linkage failed to recover score")
        ladders, orders = [], []
        for repeat in range(repeats):
            seed = BASE_SEED + 1000 * run_index + repeat
            shuffled = np.random.default_rng(seed).permutation(task_ids).tolist()
            orders.append({"order": repeat, "seed": seed, "task_order": shuffled})
            previous = category["pass_rate_bounds_pp"]
            for fraction in FRACTIONS:
                count = math.floor(fraction * n)
                if fraction == 0:
                    row = {**category, "stage": "D2_task_links", "order": repeat}
                elif fraction == 1:
                    row = {**full, "order": repeat}
                else:
                    row = evaluate("D2_task_links", True, set(shuffled[:count]), repeat, fraction)
                current = row["pass_rate_bounds_pp"]
                if current[0] < previous[0] - 1e-8 or current[1] > previous[1] + 1e-8:
                    raise AssertionError("task disclosure is not nested")
                previous = current
                ladders.append(row)
        curve = []
        for fraction in FRACTIONS:
            cells = [row for row in ladders if row["requested_fraction"] == fraction]
            widths = [row["feasible_width_pp"] for row in cells]
            curve.append({"requested_fraction": fraction, "restored_tasks": cells[0]["restored_tasks"],
                          "width_mean_pp": float(np.mean(widths)), "width_min_pp": min(widths),
                          "width_max_pp": max(widths),
                          "correct_sign_identified_orders": sum(row["correction_sign"] == truth_sign for row in cells),
                          "n_orders": repeats,
                          "restored_reference_links_min": min(row["restored_reference_links"] for row in cells),
                          "restored_reference_links_max": max(row["restored_reference_links"] for row in cells)})
        joint_counts = {f"pred{p}_ref{g}": int(sum((pred_pass == p) & (gold_pass == g)))
                        for p in (0, 1) for g in (0, 1)}
        entry = {"run": run, "n_tasks": n, "n_criteria": len(rows),
                 "reference_pass_pp": 100 * truth, "predicted_pass_pp": 100 * pred,
                 "reference_minus_judge_pp": 100 * (truth - pred),
                 "reference_correction_sign": truth_sign,
                 "reference_mean_criteria_pp": 100 * gold_mean,
                 "predicted_mean_criteria_pp": 100 * pred_mean,
                 "baseline": baseline, "category_disclosure": category,
                 "task_link_curve": curve, "task_link_rows": ladders, "orders": orders,
                 "alternative_sufficient_summary": {
                     "observed_task_verdict_confusion_counts": joint_counts,
                     "recovers_reference_pass_without_row_links": True,
                     "note": "Four observed task-level counts also recover correction; publisher must already possess task linkage. This is a sufficient summary, not a minimal release theorem."},
                 "seconds": time.perf_counter() - started}
        output.append(entry)
        print(f"{run}: D0 width={baseline['feasible_width_pp']:.2f} pp; D1={category['feasible_width_pp']:.2f} pp; {entry['seconds']:.1f}s", flush=True)
    return output


def run_judgmentbench(repeats: int) -> dict:
    from src.judgmentbench_paired_gap import FILES, ITEM_FILES, RUBRIC_FILE, load_aligned_blocks
    blocks = load_aligned_blocks()
    ids = [row["block_id"] for row in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate aligned block identity")
    task_ids = sorted({row["task_id"] for row in blocks})
    n = len(task_ids)
    orders = [{"order": repeat, "seed": BASE_SEED + 9000 + repeat,
               "task_order": np.random.default_rng(BASE_SEED + 9000 + repeat).permutation(task_ids).tolist()}
              for repeat in range(repeats)]
    output = []
    for source in FILES:
        for metric in ("pass", "mean_criteria"):
            a = np.asarray([np.mean([row[f"{source}_excellent_{metric}"] for row in blocks
                                    if row["task_id"] == task]) for task in task_ids])
            b = np.asarray([np.mean([row[f"{source}_good_{metric}"] for row in blocks
                                    if row["task_id"] == task]) for task in task_ids])
            cells = []
            for order in orders:
                indices = [task_ids.index(task) for task in order["task_order"]]
                previous = None
                for fraction in FRACTIONS:
                    count = math.floor(fraction * n)
                    cell = pairing_bounds(disclose_pairing(a, b, indices[:count]))
                    cell.update({"order": order["order"], "requested_fraction": fraction,
                                 "new_arm_scores": 0})
                    current = cell["paired_se_bounds"]
                    actual_se = float((a - b).std(ddof=1) / np.sqrt(n))
                    if not current[0] - 1e-8 <= actual_se <= current[1] + 1e-8:
                        raise AssertionError("true pairing excluded")
                    if previous and (current[0] < previous[0] - 1e-9 or current[1] > previous[1] + 1e-9):
                        raise AssertionError("pair linkage disclosure not nested")
                    previous = current
                    cells.append(cell)
            full = pairing_bounds(disclose_pairing(a, b, list(range(n))))
            if abs(full["paired_se_bounds"][1] - full["paired_se_bounds"][0]) > 1e-8:
                raise AssertionError("complete pair linkage not recovered")
            curve = []
            for fraction in FRACTIONS:
                rows = [row for row in cells if row["requested_fraction"] == fraction]
                curve.append({"requested_fraction": fraction, "restored_pairs": rows[0]["revealed_pairs"],
                              "mean_se_lower_pp": 100 * float(np.mean([row["paired_se_bounds"][0] for row in rows])),
                              "mean_se_upper_pp": 100 * float(np.mean([row["paired_se_bounds"][1] for row in rows])),
                              "minimum_se_lower_pp": 100 * min(row["paired_se_bounds"][0] for row in rows),
                              "maximum_se_upper_pp": 100 * max(row["paired_se_bounds"][1] for row in rows),
                              "robust_states": {state: sum(row["robust_exploratory_state"] == state for row in rows)
                                                for state in ("A_better", "B_better", "practically_equivalent", "unresolved")}})
            output.append({"source": source, "metric": metric,
                           "mean_gap_pp": 100 * full["mean_gap"],
                           "per_arm_task_scores": {"task_ids": task_ids, "A_excellent": a.tolist(), "B_good": b.tolist()},
                           "oracle_actual_paired_se": float((a - b).std(ddof=1) / np.sqrt(n)),
                           "zero_pairing": pairing_bounds(disclose_pairing(a, b)), "full_pairing": full,
                           "pair_link_curve": curve, "pair_link_rows": cells})
    return {"n_tasks": n, "n_blocks": len(blocks), "orders": orders, "comparisons": output,
            "provenance": {str(path.relative_to(ROOT)): _sha256(path) for path in [*FILES.values(), *ITEM_FILES.values(), RUBRIC_FILE]}}


def write_report(result: dict, path: Path) -> None:
    lines = ["# Observed-information disclosure replay", "", "## Scope and design", "",
             "This is a retrospective replay on complete existing public reference labels. It adds no labels, judge calls or experts. Raw label counts remain fixed; only released counts and label-to-task / cross-arm links change. Labels are relative to each release's reference, not proof of occupational capability or independent expert consensus.", "",
             "RuVerBench keeps the same predicted criterion/task map and fixed task-uniform weights within each of all four runs. D0 releases exact k × predicted-label reference-success counts. This is already stronger than pooled balanced accuracy and sufficient to identify task-weighted Mean Criteria. D1 refines the same counts by observed criterion category. D2 progressively restores actual reference-label links for random whole-task prefixes. This is a reference-minus-judge score-correction diagnostic on fixed outputs; correction signs are not solver-model rankings. Runs are solved separately and their summaries are never combined.", "",
             "The binary feasibility program uses unknown criterion labels x and task conjunctions y, constraints y ≤ x for every task slot and sum(x) − y ≤ k − 1, exact disclosed group totals, and equality-fixed restored labels. It separates exactly over k. Both endpoints have an attained integer witness and a closed solver objective bound; these are sharp extrema over the declared finite label-assignment space, not a continuous LP relaxation and not confidence intervals. The optimization function receives no hidden reference labels. Retained complete truth is used only to construct releases and check containment, nesting and full recovery.", "",
             f"Partial-link fractions are {list(FRACTIONS)} with floor(fraction × task count). {result['config']['repeats']} deterministic random orders per population use base seed {BASE_SEED}; every seed and entire order is saved in JSON. Fractions are nested within each order. All runs and the existing excellent–good JudgmentBench comparison for all three sources and both metrics are reported; no outcome-favorable cases are selected. Random-order ranges are disclosure-order sensitivity, not sampling confidence intervals.", "",
             "![Information disclosure replay](figures/information_disclosure_replay.png)", "",
             "Figure: left, sharp conjunction feasible widths under observed disclosures, with the linear control at zero; right, sharp paired-SE bounds for human strict JudgmentBench scores. Shading and dotted envelopes summarize ten disclosure orders, not confidence intervals. The complete six-source/metric pairing audit is reported below.", "",
             "## RuVerBench: nonlinear correction and a linear control", "",
             "Correction = reference score minus judge score. A positive value means the judge understates the fixed-reference pass rate. Feasible widths and corrections below are percentage points.", "",
             "| Run | Tasks / criteria | Actual correction | D0 width | D1 width | 50% links mean width | 75% links mean width | Full width |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["ruverbench"]:
        curve = {cell["requested_fraction"]: cell for cell in row["task_link_curve"]}
        lines.append(f"| {row['run']} | {row['n_tasks']} / {row['n_criteria']} | {row['reference_minus_judge_pp']:+.3f} | {row['baseline']['feasible_width_pp']:.3f} | {row['category_disclosure']['feasible_width_pp']:.3f} | {curve[0.5]['width_mean_pp']:.3f} | {curve[0.75]['width_mean_pp']:.3f} | {curve[1.0]['width_mean_pp']:.3f} |")
    lines += ["", "The task-uniform Mean Criteria reference score and correction are point-identified at D0 and unchanged at every disclosure stage. Thus the nonzero conjunction widths are not a generic failure to know the amount of positive criterion evidence. Category refinement can help because errors and passes are not interchangeable across all prediction/type slots; no independence is imposed.", "",
              "| Run | D0 correction range | D1 correction range | Correct sign at 25% | 50% | 75% | 100% |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["ruverbench"]:
        curve = {cell["requested_fraction"]: cell for cell in row["task_link_curve"]}
        bounds = lambda values: f"[{values[0]:+.3f}, {values[1]:+.3f}]"
        rates = " | ".join(f"{curve[f]['correct_sign_identified_orders']}/{result['config']['repeats']}" for f in (0.25, 0.5, 0.75, 1.0))
        lines.append(f"| {row['run']} | {bounds(row['baseline']['correction_bounds_pp'])} | {bounds(row['category_disclosure']['correction_bounds_pp'])} | {rates} |")
    lines += ["", "A compact alternative release is the observed 2 × 2 table of judge-task verdict against reference-task verdict: its four counts also recover the fixed-reference score correction without exposing every criterion link. Actual four-cell tables are in JSON. This is a sufficient summary for this aggregate target, not a claim that it is the unique or globally minimal release. Producing it still requires the publisher to possess the complete task linkage. A reference pass total alone suffices if correction is the sole target; the joint table additionally identifies task-verdict agreement.", "",
              "Restoration releases existing reference-to-task links, not new annotations. JSON reports the exact number of linked criteria because a given fraction of tasks contains a variable number of criteria. These retrospective link counts cannot be called expert minutes, real audit cost, or prospective policy efficiency.", "",
              "## JudgmentBench: the mean gap is already identified; pairing controls precision", "",
              "Each constructed excellent/good arm first averages its existing blocks within task, then weights all 30 tasks equally. At zero cross-arm disclosure the complete multiset of 30 scores in each arm is already known, so the finite-sample mean gap is exactly known. What is hidden is which score in B belongs with each score in A. Restored pairs fix the actual mapping; arbitrary bijections remain possible among the others. The rearrangement inequality gives attained minimum/maximum product sums, hence sharp sample covariance and paired-standard-error extrema. No arm means, task weights, labels or marginal score distributions change.", "",
              "For interpretability only, each feasible SE is mapped to gap ± 1.96 SE and the existing illustrative 1 pp practical margin: A better, B better, practically equivalent, or unresolved. These Wald intervals assume a superpopulation view of tasks and are exploratory at N = 30; they have no exact coverage claim, do not capture annotator sampling, and are not finite-population feasibility intervals. 'Robust state' means the same non-unresolved conclusion is licensed even by the largest feasible SE under this diagnostic. For the fully enumerated fixed sample itself, pairing uncertainty is not uncertainty about its known mean gap.", "",
              "| Source / metric | Fixed gap (pp) | No links: feasible SE (pp) | Full links: actual SE (pp) | Full exploratory state |", "|---|---:|---:|---:|---|"]
    for row in result["judgmentbench"]["comparisons"]:
        lo, hi = row["zero_pairing"]["paired_se_bounds"]
        full = row["full_pairing"]
        lines.append(f"| {row['source']} / {row['metric']} | {row['mean_gap_pp']:+.3f} | [{100*lo:.3f}, {100*hi:.3f}] | {100*row['oracle_actual_paired_se']:.3f} | {full['robust_exploratory_state']} |")
    unresolved = sum(row["full_pairing"]["robust_exploratory_state"] == "unresolved" for row in result["judgmentbench"]["comparisons"])
    lines += ["", f"Even complete pairing leaves {unresolved} of 6 source × metric comparisons unresolved under this exploratory rule. These are constructed quality tiers, not named solver-model arms. The three sources reuse the same 30 tasks and are not independent replications.", "",
              "## Provenance, verification and limits", "",
              f"- Prediction rows: {result['population']['criterion_records']}; distinct domain/task IDs: {result['population']['distinct_tasks']}; run-specific task evaluations: {result['population']['run_task_evaluations']}. Duplicated tasks across judge runs are not extra independent tasks.",
              "- Source SHA-256 hashes, exact random orders, every interval, restored-link counts and attained integer endpoint witnesses are in the companion JSON. Witness bit strings follow sorted task IDs and increasing criterion slots within each k stratum. Feasible intervals are endpoint hulls; they do not assert that every continuous interior score is an attainable integer-world value.",
              "- Automated validation checks exhaustive tiny label worlds, exhaustive tiny pair permutations, duplicate/malformed identities, truth containment, nested sets, the fixed linear control, and full-link recovery.",
              "- This empirical disclosure experiment restores observed links; an assumed independent coupling is not treated as observed co-occurrence data.",
              "- Identification bounds condition on fixed public references, visible prediction maps, exact count disclosures, and allowed within-group relabeling. They do not describe all realistic error-generating mechanisms or establish transportability to other benchmarks.",
              "- No raw data were edited and no paid calls or human study were executed.", "",
              "Reproduce: `.venv/bin/python -m src.information_disclosure_replay --repeats 10`.",
              "Validate: `.venv/bin/python -m pytest tests/test_information_disclosure_replay.py -q`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_result(result: dict, path: Path) -> None:
    """Static research-figure contract.

    Question: how much does actual linkage shrink fixed-label feasible ranges?
    Left: ordered disclosure stage line/dot comparison, all 4 runs, 7 stages;
    exact source rows plus mean/min/max across 10 nested orders. Linear target=0.
    Right: uncertainty envelope over 6 pairing fractions, human strict existing
    primary diagnostic (all 6 source/metric comparisons retained in report).
    Categorical spacing on left; numerical fractions on right. Matplotlib PNG,
    14 x 5.6 inches, white background; four explicit roots and marker distinction.
    QA: inspect exported image, labels, zero anchors, coverage-vs-feasibility note.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "reference-label-value-mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "reference-label-value-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#7B6D2A", "#D0782E", "#3276A8", "#B65B83")
    markers = ("s", "^", "o", "D")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#777777", "text.color": "#222222",
                         "axes.labelcolor": "#222222", "xtick.color": "#444444",
                         "ytick.color": "#444444"}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.6),
                                        gridspec_kw={"width_ratios": [1.35, 1]})
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.27, top=0.76, wspace=0.23)
        fig.suptitle("Observed-information disclosure on fixed public labels", x=0.06,
                     y=0.97, ha="left", fontsize=16, fontweight="bold")
        fig.text(0.06, 0.905,
                 f"RuVerBench: 4,916 criterion judgments / 494 distinct tasks; {result['config']['repeats']} nested reveal orders; no new labels",
                 fontsize=11, color="#555555")
        left.set_title("A. RuVerBench: conjunction feasible width", loc="left", pad=18,
                       fontsize=12, fontweight="bold")
        for row, color, marker in zip(result["ruverbench"], colors, markers):
            curve = row["task_link_curve"]
            centers = [row["baseline"]["feasible_width_pp"], *[cell["width_mean_pp"] for cell in curve]]
            lower = [centers[0], *[cell["width_min_pp"] for cell in curve]]
            upper = [centers[0], *[cell["width_max_pp"] for cell in curve]]
            x = np.arange(len(centers))
            left.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
            left.plot(x, centers, color=color, marker=marker, markersize=5,
                      linewidth=1.6, label=f"{row['run']} (N={row['n_tasks']})")
        left.axhline(0, color="#333333", linestyle="--", linewidth=1.2,
                     label="Mean Criteria width = 0 throughout")
        left.set_xticks(range(7), ["D0", "D1\n0%", "10%", "25%", "50%", "75%", "100%"])
        left.set_xlabel("Count summaries → reference links for a fraction of whole tasks", labelpad=9)
        left.set_ylabel("Sharp feasible width (percentage points)")
        left.set_ylim(-1, 41)
        left.set_xlim(-0.15, 6.15)
        left.set_yticks([0, 10, 20, 30, 40])
        left.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        left.legend(loc="upper right", fontsize=8.2, frameon=False, handlelength=2.4)
        left.text(0.01, -0.28, "D0: reference-success counts by k × prediction.\n"
                  "D1: same counts additionally stratified by criterion category.\n"
                  "Stage spacing is categorical; shaded bands are min–max across orders.",
                  transform=left.transAxes, fontsize=9, color="#555555", va="top")

        human = next(row for row in result["judgmentbench"]["comparisons"]
                     if row["source"] == "human" and row["metric"] == "pass")
        curve = human["pair_link_curve"]
        fraction = [100 * cell["requested_fraction"] for cell in curve]
        lower = [cell["mean_se_lower_pp"] for cell in curve]
        upper = [cell["mean_se_upper_pp"] for cell in curve]
        right.fill_between(fraction, lower, upper, color="#3276A8", alpha=0.15)
        right.plot(fraction, lower, color="#3276A8", marker="o", markersize=5, linewidth=1.7,
                   label="Mean feasible SE bounds")
        right.plot(fraction, upper, color="#3276A8", marker="o", markersize=5, linewidth=1.7)
        right.plot(fraction, [cell["minimum_se_lower_pp"] for cell in curve], color="#3276A8",
                   linestyle=":", linewidth=1.2, label="Min–max envelope across orders")
        right.plot(fraction, [cell["maximum_se_upper_pp"] for cell in curve], color="#3276A8",
                   linestyle=":", linewidth=1.2)
        actual = 100 * human["oracle_actual_paired_se"]
        right.axhline(actual, color="#333333", linestyle="--", linewidth=1.2,
                      label=f"Actual paired SE = {actual:.2f} pp")
        right.set_title("B. JudgmentBench: human strict-score pairing", loc="left", pad=18,
                        fontsize=12, fontweight="bold")
        right.set_xlabel("Restored same-task pairs (%)", labelpad=9)
        right.set_ylabel("Feasible paired standard error (pp)")
        right.set_xticks([0, 10, 25, 50, 75, 100])
        right.set_ylim(0, 7.5)
        right.set_xlim(-2, 102)
        right.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        right.legend(loc="upper right", fontsize=8.2, frameon=False)
        right.text(0.01, -0.33,
                   f"N = 30 tasks. Fixed mean gap = {human['mean_gap_pp']:+.2f} pp at every stage.\n"
                   "Full pairing remains unresolved under exploratory Wald + 1 pp rule.",
                   transform=right.transAxes, fontsize=9, color="#555555", va="top")
        fig.text(0.06, 0.025,
                 "Source: released RuVerBench predictions and aligned JudgmentBench rubric/item scores. Feasible ranges and order spreads are not confidence intervals.",
                 fontsize=9, color="#555555")
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)


def run_experiment(repeats: int, output: Path, figure: Path) -> dict:
    started = time.perf_counter()
    runs = load_reference_runs()
    result = {"schema_version": 1, "config": {"repeats": repeats, "fractions": FRACTIONS,
                                              "base_seed": BASE_SEED, "illustrative_margin": 0.01},
              "population": {"criterion_records": sum(len(rows) for rows in runs.values()),
                             "distinct_tasks": len({row["task_id"] for rows in runs.values() for row in rows}),
                             "run_task_evaluations": sum(len({row["task_id"] for row in rows}) for rows in runs.values())},
              "prediction_provenance": {str(PREDICTIONS.relative_to(ROOT)): _sha256(PREDICTIONS)},
              "ruverbench": run_ruverbench(runs, repeats),
              "judgmentbench": run_judgmentbench(repeats)}
    result["seconds"] = time.perf_counter() - started
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Path(figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(result, path.with_suffix(".md"))
    plot_result(result, figure)
    print(f"Wrote {path} and {path.with_suffix('.md')} in {result['seconds']:.1f}s", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/reproduced/information_disclosure_replay.json")
    parser.add_argument("--figure", type=Path, default=ROOT / "artifacts/reproduced/figures/information_disclosure_replay.png")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    run_experiment(args.repeats, args.out, args.figure)


if __name__ == "__main__":
    main()
