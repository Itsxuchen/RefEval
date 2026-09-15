"""Finite-population tests using combinatorics and hidden-label enumeration."""
from __future__ import annotations

import itertools
import math

import numpy as np

from src import conjunction_label_value as value


def choose(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def pmf(M, n, K, x):
    return choose(K, x) * choose(M - K, n - x) / choose(M, n)


def combinatorial_interval(M, n, x, alpha=.05):
    accepted = []
    for K in range(M + 1):
        lower_tail = sum(pmf(M, n, K, t) for t in range(x + 1))
        upper_tail = sum(pmf(M, n, K, t) for t in range(x, n + 1))
        if min(lower_tail, upper_tail) >= alpha / 2:
            accepted.append(K)
    return min(accepted), max(accepted)


def world(gold, prediction, sizes=None):
    g, p = np.asarray(gold, dtype=int), np.asarray(prediction, dtype=int)
    sizes = sizes or [len(g)]
    n, m = len(sizes), len(g)
    return {"gold": g, "prediction": p, "task_index": np.repeat(np.arange(n), sizes),
            "task_ids": [f"t{i}" for i in range(n)], "base_task_ids": [f"b{i}" for i in range(n)],
            "criterion_ids": [f"c{i}" for i in range(m)], "category": ["binary"] * m,
            "secondary_predictions": (1 - p)[None, :], "secondaries": ["secondary"],
            "family": "synthetic", "target": "all_pass", "source_hashes": {}}


def test_exact_hypergeometric_inversion_and_coverage_by_combinatorics():
    for M in range(1, 11):
        for n in range(M + 1):
            intervals = []
            for x in range(n + 1):
                interval = value.hypergeom_ci(M, n, x)
                assert interval == combinatorial_interval(M, n, x)
                intervals.append(interval)
            for K in range(M + 1):
                coverage = sum(pmf(M, n, K, x) for x, (low, high) in enumerate(intervals) if low <= K <= high)
                assert coverage >= .95 - 1e-12, (M, n, K, coverage)


def test_conditional_accuracy_estimator_unbiased_for_every_small_remainder():
    for M in range(1, 8):
        for qA in range(M + 1):
            U = M - qA
            for correctA in range(qA + 1):
                for K in range(U + 1):
                    truth = (correctA + K) / M
                    for qR in range(1, U + 1):
                        expectation = coverage = 0.0
                        for correctR in range(qR + 1):
                            probability = pmf(U, qR, K, correctR)
                            if not probability:
                                continue
                            result = value.accuracy_inference(M, qA, correctA, qR, correctR)
                            expectation += probability * result["estimate"]
                            lo, hi = combinatorial_interval(U, qR, correctR)
                            assert math.isclose(result["ci_lower"], (correctA + lo) / M)
                            assert math.isclose(result["ci_upper"], (correctA + hi) / M)
                            coverage += probability * (result["ci_lower"] - 1e-12 <= truth <= result["ci_upper"] + 1e-12)
                        assert math.isclose(expectation, truth, abs_tol=1e-12)
                        assert coverage >= .95 - 1e-12


def test_no_random_remainder_means_identification_only_except_at_census():
    for M in range(1, 8):
        for qA in range(M + 1):
            for correctA in range(qA + 1):
                out = value.accuracy_inference(M, qA, correctA, 0, 0)
                assert out["ci_lower"] == correctA / M
                assert out["ci_upper"] == (correctA + M - qA) / M
                assert out["estimate"] == (correctA / M if qA == M else None)


def test_accuracy_and_pass_bounds_equal_extrema_over_hidden_completions():
    M = 3
    for prediction in itertools.product((0, 1), repeat=M):
        for mask in itertools.product((False, True), repeat=M):
            indices = np.flatnonzero(mask)
            unobserved = [i for i in range(M) if not mask[i]]
            for observed in itertools.product((0, 1), repeat=len(indices)):
                g = np.ones(M, dtype=int)
                g[indices] = observed
                data = world(g, prediction, [1, 2])
                report = value.query_metrics(data, indices)
                passes, accuracies, reports = [], [], []
                for hidden in itertools.product((0, 1), repeat=len(unobserved)):
                    completion = g.copy(); completion[unobserved] = hidden
                    complete_data = world(completion, prediction, [1, 2])
                    passes.append((int(completion[0]) + int(all(completion[1:]))) / 2)
                    accuracies.append(sum(completion[i] == prediction[i] for i in range(M)) / M)
                    reports.append(value.query_metrics(complete_data, indices))
                assert math.isclose(report["pass_lower"], min(passes))
                assert math.isclose(report["pass_upper"], max(passes))
                assert math.isclose(report["accuracy_lower"], min(accuracies))
                assert math.isclose(report["accuracy_upper"], max(accuracies))
                assert math.isclose(report["pass_upper"] - report["pass_lower"], 1 - report["certified_tasks"] / 2)
                # Bounds and certification are observable even if hidden truth
                # changes; residual errors deliberately remain oracle metrics.
                for field in ("pass_lower", "pass_upper", "accuracy_lower", "accuracy_upper", "certified_tasks"):
                    assert all(r[field] == report[field] for r in reports)


def test_hybrid_has_disjoint_acquisition_stages_and_exact_total_cap():
    data = world([1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1, 0], [2, 2, 3])
    for cap, order, seed in itertools.product(range(8), ("release", "predicted_fail_count", "random"), (0, 1, 19)):
        A, R = value.hybrid_queries(data, cap, order, seed)
        assert len(A) <= cap // 2
        assert len(A) + len(R) == cap
        assert len(set(A.tolist())) == len(A)
        assert len(set(R.tolist())) == len(R)
        assert not set(A.tolist()) & set(R.tolist())
        assert set(A.tolist()) | set(R.tolist()) <= set(range(7))


def test_hybrid_never_uses_unobserved_gold_to_select_either_stage():
    for prediction in itertools.product((0, 1), repeat=3):
        worlds = [world(g, prediction, [1, 2]) for g in itertools.product((0, 1), repeat=3)]
        for cap, order, seed in itertools.product(range(4), ("release", "predicted_fail_count", "random"), (0, 11)):
            selections = [value.hybrid_queries(data, cap, order, seed) for data in worlds]
            for i, j in itertools.combinations(range(len(worlds)), 2):
                a, b = selections[i][0].tolist(), selections[j][0].tolist()
                same_history = True
                for pos in range(max(len(a), len(b)) + 1):
                    x, y = a[pos] if pos < len(a) else None, b[pos] if pos < len(b) else None
                    assert x == y
                    if x is None:
                        break
                    if worlds[i]["gold"][x] != worlds[j]["gold"][y]:
                        same_history = False
                        break
                if same_history:
                    assert np.array_equal(selections[i][1], selections[j][1])


def test_f27_extra_margins_can_identify_score_without_querying_all_tasks():
    demo = value.f27_bridge_demo()
    before, after = demo["examples"]
    assert before["linked_only_pass_range"] == [0., 1.]
    assert before["f27_with_exact_margins"]["pass_rate_bounds"] == [0., .5]
    assert after["linked_only_pass_range"] == [.5, 1.]
    assert after["f27_with_exact_margins"]["pass_rate_bounds"] == [.5, .5]
