"""Whole-unit SRS controls for fixed-reference conjunction assessment.

Fix the number of sampled tasks/annotation units, not a criterion-query budget.
Full-rubric and short-circuit queries certify the same sampled task outcomes.
The full-rubric arm also supports a design-unbiased microaccuracy estimator;
that estimator is NOT available from short-circuit labels alone.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from src.conjunction_policy_data import load_datasets


def prepare(data: dict) -> dict:
    gold, prediction, ti = (np.asarray(data[k]) for k in ("gold", "prediction", "task_index"))
    n = len(data["task_ids"])
    if len(gold) != len(prediction) or len(gold) != len(ti) or n == 0:
        raise ValueError("Nonempty aligned tasks required")
    if not np.isin(gold, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Binary labels required")
    if ti.min() != 0 or ti.max() != n - 1 or np.any(np.diff(ti) < 0):
        raise ValueError("Contiguous ordered task indices required")
    k = np.bincount(ti, minlength=n).astype(int)
    if np.any(k == 0):
        raise ValueError("Empty task")
    failures = np.bincount(ti, weights=1 - gold, minlength=n)
    correct = np.bincount(ti, weights=gold == prediction, minlength=n).astype(int)
    return {"n": n, "m": len(gold), "k": k, "correct": correct,
            "pass": (failures == 0).astype(int), "gold": gold, "prediction": prediction, "ti": ti}


def criterion_order(data: dict, seed: int) -> np.ndarray:
    """Observable primary labels and random ties only; no reference access."""
    prediction = np.asarray(data["prediction"])
    ti = np.asarray(data["task_index"])
    ties = np.random.default_rng(np.random.SeedSequence([20260914, seed, 2])).random(len(prediction))
    return np.lexsort((ties, prediction, ti))


def sample_tasks(n: int, sample_n: int, seed: int) -> np.ndarray:
    if not 1 <= sample_n <= n:
        raise ValueError("Sample size must lie in 1..N")
    rng = np.random.default_rng(np.random.SeedSequence([20260914, seed, 1]))
    # Permutation prefixes give SRSWOR at each fixed n and nested task budgets.
    return np.sort(rng.permutation(n)[:sample_n])


def design_rmse(values: np.ndarray, sample_n: int, scale: float = 1.0) -> float:
    """Exact finite-population RMSE of scale times an unbiased SRS mean."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if not 1 <= sample_n <= n:
        raise ValueError("Sample size must lie in 1..N")
    if n == 1 or sample_n == n:
        return 0.0
    return float(abs(scale) * np.sqrt((1 - sample_n / n) * np.var(values, ddof=1) / sample_n))


def evaluate_sample(data: dict, selected: Iterable[int], seed: int = 1) -> dict:
    info = prepare(data)
    selection = np.asarray(list(selected), dtype=int)
    if len(selection) == 0 or len(np.unique(selection)) != len(selection) or np.any(selection < 0) or np.any(selection >= info["n"]):
        raise ValueError("Require unique valid sampled tasks")
    selected_mask = np.zeros(info["n"], dtype=bool)
    selected_mask[selection] = True
    observed_fail = np.zeros(info["n"], dtype=bool)
    observed = np.zeros(info["n"], dtype=int)
    query_indices = []
    for index in criterion_order(data, seed):
        task = info["ti"][index]
        if not selected_mask[task] or observed_fail[task]:
            continue
        query_indices.append(int(index))
        observed[task] += 1
        observed_fail[task] |= info["gold"][index] == 0
    certified = observed_fail | (observed == info["k"])
    if not certified[selection].all():
        raise AssertionError("Every selected task must be certified")
    verdicts = (~observed_fail[selection]).astype(int)
    if not np.array_equal(verdicts, info["pass"][selection]):
        raise AssertionError("Short circuit and full rubric disagree")
    sample_n = len(selection)
    full_queries = int(info["k"][selection].sum())
    full_correct = int(info["correct"][selection].sum())
    full_errors = full_queries - full_correct
    return {
        "sample_n": sample_n, "population_n": info["n"], "population_criteria": info["m"],
        "selected_task_indices": selection.tolist(), "selected_task_ids": [data["task_ids"][i] for i in selection],
        "short_circuit_query_indices": query_indices, "sample_reference_pass": int(verdicts.sum()),
        "full_rubric_queries": full_queries, "short_circuit_queries": len(query_indices),
        "task_pass_estimate": float(verdicts.mean()), "population_task_pass": float(info["pass"].mean()),
        "task_pass_exact_design_rmse": design_rmse(info["pass"], sample_n),
        "full_rubric_correct_criteria": full_correct,
        "full_rubric_error_criteria": full_errors,
        "microaccuracy_ht_full_rubric": float(info["n"] * full_correct / (sample_n * info["m"])),
        "microaccuracy_difference_full_rubric": float(1 - info["n"] * full_errors / (sample_n * info["m"])),
        "microaccuracy_naive_sample_ratio_full_rubric": float(full_correct / full_queries),
        "population_microaccuracy": float(info["correct"].sum() / info["m"]),
        "microaccuracy_ht_exact_design_rmse": design_rmse(info["correct"], sample_n, info["n"] / info["m"]),
        "microaccuracy_difference_exact_design_rmse": design_rmse(info["k"] - info["correct"], sample_n, info["n"] / info["m"]),
    }


def run_task_sampling(datasets: dict | None = None, seeds: Iterable[int] = range(1, 32),
                      task_shares: Iterable[float] = (.05, .20, .50)) -> dict:
    datasets = load_datasets() if datasets is None else datasets
    seeds, task_shares = list(seeds), list(task_shares)
    if any(not 0 < share <= 1 for share in task_shares):
        raise ValueError("Task shares must lie in (0,1]")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds")
    rows = []
    populations = {}
    for name, data in datasets.items():
        info = prepare(data)
        populations[name] = {"tasks": info["n"], "criteria": info["m"], "family": data.get("family"),
                             "target": data.get("target"), "source_hashes": data.get("source_hashes", {})}
        for share in task_shares:
            sample_n = min(info["n"], max(1, int(round(share * info["n"]))))
            for seed in seeds:
                selected = sample_tasks(info["n"], sample_n, seed)
                rows.append({"dataset": name, "seed": seed, "requested_task_share": share,
                             "actual_task_share": sample_n / info["n"], **evaluate_sample(data, selected, seed)})
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seeds": seeds, "requested_task_shares": task_shares, "datasets": populations, "rows": rows,
            "boundaries": ["SRS without replacement over fixed task/annotation units, with fixed sampled unit count and random label cost; not a matched criterion-cap experiment.",
                           "JB units are output-by-rater annotations, not 1539 independent base problems; design-unbiasedness targets the finite annotation population.",
                           "The task-pass sample mean is available from both query arms. Microaccuracy Horvitz-Thompson estimates require all labels of sampled rubrics; short-circuit does not supply those labels.",
                           "HT microaccuracy is N/(sample_n*M) times the sampled total of correct criterion labels; values above one are possible and are not clipped.",
                           "The difference estimator uses the known criterion count M: one minus N/(sample_n*M) times sampled errors. It is also design-unbiased and can have lower variance; no optimality is asserted and out-of-range values are not clipped.",
                           "Exact finite-population RMSE uses known population dispersion for retrospective design evaluation, not an estimated confidence interval available to a deployed sampler.",
                           "Seeds describe repeated sampling designs and random within-judge-label ties; no human time, reference validity, or claimed optimal policy."]}


def run_all(datasets: dict | None = None, seeds: Iterable[int] = range(1, 32)) -> list[dict]:
    """CSV-compatible wrapper for the parent study's existing output driver."""
    result = run_task_sampling(datasets, seeds=seeds)
    return [{key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
             for key, value in row.items()} for row in result["rows"]]
