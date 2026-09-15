"""Exhaustive finite-population checks for the task-SRS control."""
from itertools import combinations

import numpy as np
import pytest

from src.conjunction_task_sampling import criterion_order, design_rmse, evaluate_sample, prepare, run_task_sampling, sample_tasks


def world(tasks):
    return {"task_ids": [f"task{i}" for i in range(len(tasks))],
            "task_index": np.array([i for i, rows in enumerate(tasks) for _ in rows]),
            "gold": np.array([g for rows in tasks for g, p in rows]),
            "prediction": np.array([p for rows in tasks for g, p in rows])}


def test_exhaustive_srs_means_and_rmse_with_unequal_rubrics():
    data = world([[(1, 1)], [(0, 1), (1, 0), (1, 0)], [(1, 1), (1, 1)]])
    info = prepare(data)
    for n in (1, 2, 3):
        rows = [evaluate_sample(data, selection) for selection in combinations(range(3), n)]
        passes = np.array([row["task_pass_estimate"] for row in rows])
        ht = np.array([row["microaccuracy_ht_full_rubric"] for row in rows])
        difference = np.array([row["microaccuracy_difference_full_rubric"] for row in rows])
        pass_truth = info["pass"].mean()
        acc_truth = info["correct"].sum() / info["m"]
        assert passes.mean() == pytest.approx(pass_truth)
        assert ht.mean() == pytest.approx(acc_truth)
        assert difference.mean() == pytest.approx(acc_truth)
        assert np.sqrt(np.mean((passes - pass_truth) ** 2)) == pytest.approx(design_rmse(info["pass"], n))
        assert np.sqrt(np.mean((ht - acc_truth) ** 2)) == pytest.approx(design_rmse(info["correct"], n, info["n"] / info["m"]))
        assert np.sqrt(np.mean((difference - acc_truth) ** 2)) == pytest.approx(design_rmse(info["k"] - info["correct"], n, info["n"] / info["m"]))
        assert rows[0]["microaccuracy_difference_exact_design_rmse"] == pytest.approx(np.sqrt(np.mean((difference - acc_truth) ** 2)))


def test_naive_ratio_can_be_biased_and_ht_can_exceed_one():
    data = world([[(1, 1)] * 5, [(1, 0)]])
    rows = [evaluate_sample(data, [i]) for i in range(2)]
    assert np.mean([r["microaccuracy_naive_sample_ratio_full_rubric"] for r in rows]) == .5
    assert rows[0]["population_microaccuracy"] == pytest.approx(5 / 6)
    assert np.mean([r["microaccuracy_ht_full_rubric"] for r in rows]) == pytest.approx(5 / 6)
    assert rows[0]["microaccuracy_ht_full_rubric"] == pytest.approx(5 / 3)


def test_short_circuit_has_same_sample_outcomes_with_lower_cost():
    data = world([[(0, 0), (1, 1), (1, 1)], [(1, 0), (1, 1)]])
    row = evaluate_sample(data, [0, 1], seed=5)
    assert row["full_rubric_queries"] == 5
    assert row["short_circuit_queries"] == 3
    assert row["sample_reference_pass"] == 1 and row["task_pass_estimate"] == .5
    assert row["task_pass_exact_design_rmse"] == row["microaccuracy_ht_exact_design_rmse"] == 0


def test_difference_estimator_uses_known_total_without_clipping():
    # Correct counts vary with rubric length, whereas the error total is fixed.
    data = world([[(1, 1)] * 5, [(1, 1)]])
    rows = [evaluate_sample(data, [i]) for i in range(2)]
    assert rows[0]["microaccuracy_ht_exact_design_rmse"] > 0
    assert all(row["microaccuracy_difference_full_rubric"] == 1 for row in rows)
    assert all(row["microaccuracy_difference_exact_design_rmse"] == 0 for row in rows)
    erroneous = world([[(1, 0)] * 5, [(1, 1)]])
    row = evaluate_sample(erroneous, [0])
    assert row["microaccuracy_difference_full_rubric"] == pytest.approx(-2 / 3)


def test_priority_and_task_selection_do_not_read_reference():
    data = world([[(0, 0), (1, 1), (1, 0)], [(1, 0), (1, 1)]])
    changed = {**data, "gold": 1 - data["gold"]}
    assert np.array_equal(criterion_order(data, 9), criterion_order(changed, 9))
    assert set(sample_tasks(10, 2, 9)) <= set(sample_tasks(10, 5, 9))
    assert len(set(sample_tasks(10, 5, 9))) == 5


def test_raw_rows_have_selected_ids_actual_costs_and_declared_boundaries():
    data = world([[(1, 1)], [(0, 1), (1, 0), (1, 1)]])
    result = run_task_sampling({"synthetic": data}, seeds=[1, 2], task_shares=[.5, 1])
    assert len(result["rows"]) == 4
    for row in result["rows"]:
        assert len(row["selected_task_ids"]) == row["sample_n"]
        assert len(row["short_circuit_query_indices"]) == row["short_circuit_queries"]
        assert row["short_circuit_queries"] <= row["full_rubric_queries"]
    assert any("not a matched criterion-cap" in text for text in result["boundaries"])
    assert any("require all labels" in text for text in result["boundaries"])


@pytest.mark.parametrize("selected", [[], [0, 0], [-1], [2]])
def test_invalid_task_selection_fails(selected):
    data = world([[(1, 1)], [(0, 0)]])
    with pytest.raises(ValueError, match="unique valid"):
        evaluate_sample(data, selected)
