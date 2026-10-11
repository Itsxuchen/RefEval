from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest

from src.conjunction_design_rmse import (
    binary_mean_variance, conditional_variance, exact_design_variance,
    summarize_conditional_variances,
)
from src.conjunction_stratified_baseline import integer_waterfill, pilot_allocation


def test_binary_variance_matches_exhaustive_sampling():
    for N in range(1, 8):
        for K in range(N + 1):
            y = [1] * K + [0] * (N - K)
            for n in range(1, N + 1):
                errors = [(Fraction(sum(y[i] for i in sample), n) - Fraction(K, N)) ** 2
                          for sample in combinations(range(N), n)]
                assert binary_mean_variance(N, K, n) == pytest.approx(float(sum(errors) / len(errors)), abs=1e-15)


def test_pilot_risk_matches_exhaustive_label_sampling():
    # Enumerate identities and all second-stage samples, independently of the
    # implementation's aggregation by hypergeometric pilot success counts.
    groups = [np.array([1, 1, 0, 0, 0]), np.array([1, 1, 1, 1, 0, 0])]
    sizes = np.array([len(g) for g in groups]); M = int(sizes.sum()); cap = 8
    pilot = pilot_allocation(sizes, cap); remaining = sizes - pilot
    truth = Fraction(sum(int(g.sum()) for g in groups), M)
    risks = []
    for chosen in product(*(combinations(range(len(g)), int(a)) for g, a in zip(groups, pilot))):
        t = np.array([int(g[list(s)].sum()) for g, s in zip(groups, chosen)])
        q = (t + .5) / (pilot + 1)
        counts = integer_waterfill(cap - int(pilot.sum()), remaining * np.sqrt(q * (1 - q)), remaining, [1, 1])
        rests = [[i for i in range(len(g)) if i not in chosen[h]] for h, g in enumerate(groups)]
        estimates = []
        for samples in product(*(combinations(s, int(n)) for s, n in zip(rests, counts))):
            estimate = (Fraction(int(t.sum())) + sum(Fraction(int(R) * int(g[list(s)].sum()), int(n))
                        for R, g, s, n in zip(remaining, groups, samples, counts))) / M
            estimates.append(estimate)
        assert sum(estimates) / len(estimates) == truth
        risks.append(sum((e - truth) ** 2 for e in estimates) / len(estimates))
    actual = exact_design_variance(sizes, [2, 4], cap, "charged_pilot_neyman")
    assert actual["variance"] == pytest.approx(float(sum(risks) / len(risks)), abs=1e-15)
    assert actual["pilot_probability_mass"] == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize("policy", ["random_criterion", "proportional_stratified", "charged_pilot_neyman"])
def test_census_and_low_budget_fallback(policy):
    assert exact_design_variance([4, 4], [1, 3], 8, policy)["variance"] == 0
    assert exact_design_variance([4, 4], [1, 3], 1, policy)["variance"] == binary_mean_variance(8, 4, 1)
    assert exact_design_variance([7], [0], 5, policy)["variance"] == 0


def test_proportional_variance_matches_identity_enumeration():
    groups = [[1, 0, 0], [1, 1, 1, 0, 0]]; truth = Fraction(4, 8)
    counts = integer_waterfill(5, [3, 5], [3, 5], [1, 1])
    estimates = [(Fraction(3 * sum(a), int(counts[0])) + Fraction(5 * sum(b), int(counts[1]))) / 8
                 for a, b in product(combinations(groups[0], int(counts[0])), combinations(groups[1], int(counts[1])))]
    expected = sum((v - truth) ** 2 for v in estimates) / len(estimates)
    assert exact_design_variance([3, 5], [1, 3], 5, "proportional_stratified")["variance"] == pytest.approx(float(expected), abs=1e-15)


def test_mix_averages_variances_before_square_root_and_reports_mcse():
    result = summarize_conditional_variances([.01, .04])
    assert result["rmse"] == pytest.approx(np.sqrt(.025))
    assert result["rmse"] != pytest.approx((.1 + .2) / 2)
    assert result["variance_mcse"] == pytest.approx(.015)
    assert result["rmse_mcse"] == pytest.approx(.015 / (2 * np.sqrt(.025)))
    assert summarize_conditional_variances([0, 0])["rmse_mcse"] == 0
    # A fixed first-stage total contributes no conditional sampling variance.
    assert conditional_variance(10, [6], [3], [2]) == pytest.approx((6 / 10) ** 2 * binary_mean_variance(6, 3, 2))


def test_invalid_designs():
    with pytest.raises(ValueError):
        binary_mean_variance(5, 3, 0)
    with pytest.raises(ValueError):
        exact_design_variance([5], [6], 2, "random_criterion")
    with pytest.raises(ValueError):
        summarize_conditional_variances([.1])
