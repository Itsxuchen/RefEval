"""Design risk at the manuscript's 20% criterion cap.

SRS, proportional strata and charged-pilot Neyman integrate their full sampling
design. The mix integrates the fresh SRS exactly, then averages conditional
variances over 31 first-stage realizations; its remaining Monte Carlo error is
reported explicitly. Historical empirical-RMSE tables are never overwritten.
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from src.conjunction_policy_data import load_datasets
from src.conjunction_label_value import hybrid_queries
from src.conjunction_stratified_baseline import integer_waterfill, pilot_allocation, validate_output

ROOT = Path(__file__).resolve().parents[1]
SHARE = .20


def binary_mean_variance(N, K, n):
    """Exact SRSWOR variance of a binary mean, including finite correction."""
    if not (0 <= K <= N and 0 <= n <= N and (n > 0 or N == 0)):
        raise ValueError("a nonempty population requires a positive sample")
    if n == N:
        return 0.0
    return (N - n) * K * (N - K) / (n * N * N * (N - 1))


def conditional_variance(M, sizes, correct, counts):
    """Known first-stage total plus independent stratum remainder estimates."""
    return math.fsum((int(N) / M) ** 2 * binary_mean_variance(int(N), int(K), int(n))
                     for N, K, n in zip(sizes, correct, counts))


def exact_design_variance(sizes, correct, cap, policy):
    """Integrate all pilot count outcomes using the actual integer allocation."""
    sizes, correct = np.asarray(sizes, int), np.asarray(correct, int)
    if (sizes.ndim != 1 or sizes.shape != correct.shape or not len(sizes)
            or np.any(sizes <= 0) or np.any(correct < 0) or np.any(correct > sizes)
            or not 1 <= cap <= sizes.sum()):
        raise ValueError("invalid finite population or budget")
    if policy not in ("random_criterion", "proportional_stratified", "charged_pilot_neyman"):
        raise ValueError("unknown policy")
    M = int(sizes.sum())
    result = dict(pilot_outcomes=0, pilot_probability_mass=1.0)
    if cap == M:
        return dict(result, variance=0.0)
    pilot = pilot_allocation(sizes, cap) if policy == "charged_pilot_neyman" else None
    fallback = ((policy == "charged_pilot_neyman" and pilot is None)
                or (policy == "proportional_stratified" and cap < len(sizes)))
    if policy == "random_criterion" or fallback:
        return dict(result, variance=binary_mean_variance(M, int(correct.sum()), cap))
    if policy == "proportional_stratified":
        counts = integer_waterfill(cap, sizes, sizes, np.ones(len(sizes), int))
        return dict(result, variance=conditional_variance(M, sizes, correct, counts))

    remaining = sizes - pilot
    supports = [np.arange(max(0, int(a - (N - K))), min(int(a), int(K)) + 1)
                for N, K, a in zip(sizes, correct, pilot)]
    probabilities = [hypergeom.pmf(t, N, K, a)
                     for t, N, K, a in zip(supports, sizes, correct, pilot)]
    terms, masses = [], []
    for index in itertools.product(*(range(len(t)) for t in supports)):
        t = np.array([supports[h][j] for h, j in enumerate(index)])
        probability = math.prod(float(probabilities[h][j]) for h, j in enumerate(index))
        rates = (t + .5) / (pilot + 1)
        counts = integer_waterfill(cap - int(pilot.sum()),
                                   remaining * np.sqrt(rates * (1 - rates)),
                                   remaining, (remaining > 0).astype(int))
        terms.append(probability * conditional_variance(M, remaining, correct - t, counts))
        masses.append(probability)
    mass = math.fsum(masses)
    if not math.isclose(mass, 1.0, rel_tol=0, abs_tol=1e-10):
        raise AssertionError("pilot probability mass is incomplete")
    return dict(variance=math.fsum(terms), pilot_outcomes=len(terms), pilot_probability_mass=mass)


def summarize_conditional_variances(values):
    """Rao-Blackwellized risk estimate and delta-method Monte Carlo SE."""
    values = np.asarray(values, float)
    if len(values) < 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("at least two finite nonnegative conditional variances required")
    variance = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(len(values))
    rmse = math.sqrt(variance)
    return dict(variance=variance, rmse=rmse, variance_mcse=se,
                rmse_mcse=se / (2 * rmse) if rmse else 0.0,
                conditional_variance_cv=sd / variance if variance else 0.0)


def compute(datasets, seeds=range(1, 32)):
    seeds = list(seeds)
    if len(seeds) < 2 or len(seeds) != len(set(seeds)) or any(s < 1 for s in seeds):
        raise ValueError("use at least two distinct positive first-stage seeds")
    rows, conditional = [], []
    for name, data in datasets.items():
        pred = np.asarray(data["prediction"])
        agreement = np.asarray(data["gold"]) == pred
        M, K = len(pred), int(agreement.sum())
        cap = int(round(SHARE * M))
        strata = [pred == h for h in (0, 1) if np.any(pred == h)]
        sizes = [int(s.sum()) for s in strata]
        correct = [int(agreement[s].sum()) for s in strata]
        base = dict(dataset=name, budget_share=SHARE, budget_cap=cap,
                    n_criteria=M, population_correct=K)
        for policy in ("random_criterion", "proportional_stratified", "charged_pilot_neyman"):
            result = exact_design_variance(sizes, correct, cap, policy)
            rows.append(dict(base, policy=policy, calculation="exact_design", **result,
                             rmse=math.sqrt(result["variance"]), variance_mcse=0.0, rmse_mcse=0.0,
                             conditional_variance_cv=0.0, first_stage_replicates=0))
        variances = []
        for seed in seeds:
            A, R = hybrid_queries(data, cap, "release", seed)
            U, n, remaining_correct = M - len(A), len(R), K - int(agreement[A].sum())
            variance = conditional_variance(M, [U], [remaining_correct], [n])
            variances.append(variance)
            conditional.append(dict(base, seed=seed, stageA_queries=len(A), stageR_queries=n,
                                    stageA_correct=int(agreement[A].sum()),
                                    remaining_correct=remaining_correct, conditional_variance=variance))
        rows.append(dict(base, policy="sc50_then_srs", calculation="first_stage_monte_carlo",
                         **summarize_conditional_variances(variances), pilot_outcomes=0,
                         pilot_probability_mass=1.0, first_stage_replicates=len(seeds)))
        print(f"Design RMSE: {name}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(conditional)


def run(root, output):
    root = Path(root).resolve()
    output = validate_output(output, root)
    rows, conditional = compute(load_datasets(root))
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "design_rmse.csv", index=False)
    conditional.to_csv(output / "mix_conditional_variances.csv", index=False)
    return {"scope": "20% criterion cap; exact three-design risks and mix first-stage Monte Carlo",
            "summary_rows": len(rows), "conditional_rows": len(conditional)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.root, args.output))
