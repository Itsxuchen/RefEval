"""Independent tiny-world checks for the observed-information replay."""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict

import numpy as np
import pytest

from src.information_disclosure_replay import (
    conjunction_bounds, decision_state, disclose_pairing, disclose_reference,
    load_reference_runs, pairing_bounds,
)


def toy_rows(gold=(1, 1, 0, 1, 0, 1, 0)):
    sizes = (1, 2, 2, 2)
    rows = []
    for task, k in enumerate(sizes):
        for slot in range(k):
            index = len(rows)
            rows.append({"task_id": f"task{task}", "slot": slot, "k": k,
                         "category": "logic" if (task + slot) % 2 else "format",
                         "prediction": int(index % 3 != 0), "gold": gold[index]})
    return rows


def enumerate_reference_extrema(release):
    """Enumerate literal binary assignments, without conjunction MILP constraints."""
    slots = release["visible_slots"]
    task_indices = defaultdict(list)
    for index, row in enumerate(slots):
        task_indices[row["task_id"]].append(index)
    feasible = []
    for labels in itertools.product((0, 1), repeat=len(slots)):
        if any(labels[index] != value for index, value in release["reference_links"].items()):
            continue
        if any(sum(labels[index] for index in group["slots"]) != group["reference_successes"]
               for group in release["groups"]):
            continue
        feasible.append(sum(all(labels[index] for index in indices)
                            for indices in task_indices.values()))
    return [min(feasible), max(feasible)]


@pytest.mark.parametrize("gold", [
    (0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 0, 1, 0, 1, 0), (0, 1, 1, 0, 0, 0, 1),
    (1, 0, 1, 1, 1, 0, 0), (0, 0, 0, 1, 1, 1, 1),
])
@pytest.mark.parametrize("category,selected", [(False, set()), (True, set()),
                                               (True, {"task2"}), (True, {"task0", "task3"})])
def test_integer_extrema_equal_independently_enumerated_worlds(gold, category, selected):
    release = disclose_reference(toy_rows(gold), category, selected)
    computed = conjunction_bounds(release)
    assert computed["pass_count_bounds"] == enumerate_reference_extrema(release)
    # Stored endpoint witnesses independently reconstruct literal conjunctions.
    for certificate in computed["solver_certificates"]:
        k = certificate["k"]
        bits = list(map(int, certificate["reference_label_witness_bits"]))
        conjunction = [int(all(bits[start:start+k])) for start in range(0, len(bits), k)]
        assert "".join(map(str, conjunction)) == certificate["task_pass_witness_bits"]
        assert sum(conjunction) == certificate["pass_count"]
        assert certificate["objective_gap"] <= 1e-7


def test_conjunction_classical_pooled_count_case_and_linear_control():
    rows = [{"task_id": f"t{i//2}", "slot": i % 2, "k": 2,
             "category": "same", "prediction": 1, "gold": int(i < 3)} for i in range(6)]
    release = disclose_reference(rows, False)
    # Three successes across three two-slot tasks allow either 0 or 1 passes.
    assert conjunction_bounds(release)["pass_count_bounds"] == [0, 1]
    assert sum(group["reference_successes"] / 2 for group in release["groups"]) / 3 == 0.5


def test_nested_truth_containment_and_full_recovery_without_hidden_gold():
    rows = toy_rows()
    task_names = sorted({row["task_id"] for row in rows})
    truth = sum(all(row["gold"] for row in rows if row["task_id"] == task) for task in task_names)
    releases = [disclose_reference(rows, False), disclose_reference(rows, True)]
    releases += [disclose_reference(rows, True, set(task_names[:count])) for count in range(1, 5)]
    previous = [-1, 100]
    for release in releases:
        assert not any("gold" in row for row in release["visible_slots"])
        current = conjunction_bounds(release)["pass_count_bounds"]
        assert previous[0] <= current[0] <= truth <= current[1] <= previous[1]
        previous = current
    assert previous == [truth, truth]


@pytest.mark.parametrize("selected", [(), (1,), (1, 3), (0, 1, 2, 3)])
def test_pairing_extrema_equal_all_permutations(selected):
    a = np.array([0.0, 0.25, 0.75, 1.0])
    b = np.array([0.5, 0.0, 1.0, 0.25])
    release = disclose_pairing(a, b, selected)
    result = pairing_bounds(release)
    remaining = [i for i in range(len(a)) if i not in selected]
    covariance, se, gaps = [], [], []
    for permutation in itertools.permutations(remaining):
        mapped = b.copy()
        mapped[remaining] = b[list(permutation)]
        covariance.append(float(np.cov(a, mapped, ddof=1)[0, 1]))
        se.append(float((a - mapped).std(ddof=1) / np.sqrt(len(a))))
        gaps.append(float(np.mean(a - mapped)))
    np.testing.assert_allclose(result["covariance_bounds"], [min(covariance), max(covariance)], atol=1e-10)
    np.testing.assert_allclose(result["paired_se_bounds"], [min(se), max(se)], atol=1e-10)
    np.testing.assert_allclose(gaps, result["mean_gap"], atol=1e-10)
    assert "actual_paired_se" not in result


def test_perfect_pairing_zero_variance_and_hidden_mapping_not_in_release():
    a = np.array([0.0, 0.25, 0.5, 1.0])
    fully = pairing_bounds(disclose_pairing(a, a, [0, 1, 2, 3]))
    np.testing.assert_allclose(fully["paired_se_bounds"], [0, 0], atol=1e-9)
    assert disclose_pairing(a, a) == disclose_pairing(a, a[::-1])
    assert fully["robust_exploratory_state"] == "practically_equivalent"


def test_pairing_nested_intervals_and_full_actual_se():
    a = np.array([0.0, 0.1, 0.3, 0.8, 1.0])
    b = np.array([0.2, 0.0, 0.4, 0.9, 0.8])
    order = [2, 4, 0, 1, 3]
    previous = [-1, 100]
    for count in range(6):
        row = pairing_bounds(disclose_pairing(a, b, order[:count]))
        current = row["paired_se_bounds"]
        assert previous[0] - 1e-8 <= current[0] <= current[1] <= previous[1] + 1e-8
        assert row["mean_gap_bounds"][0] == row["mean_gap_bounds"][1]
        previous = current
    np.testing.assert_allclose(previous, np.repeat((a-b).std(ddof=1)/np.sqrt(5), 2), atol=1e-9)


def test_malformed_pair_identity_rejected():
    with pytest.raises(ValueError, match="duplicate or unknown"):
        disclose_pairing(np.array([0, 1]), np.array([0, 1]), [0, 0])
    with pytest.raises(ValueError, match="duplicate or unknown"):
        disclose_pairing(np.array([0, 1]), np.array([0, 1]), [2])


def write_csv(path, rows):
    fields = ["run", "domain", "task_id", "criterion_id", "criterion_index", "category", "gold", "prediction"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reference_csv_row(**kwargs):
    return {"run": "judge", "domain": "domain", "task_id": "task", "criterion_id": "named_rubric",
            "criterion_index": "0", "category": "logic", "gold": "True", "prediction": "False", **kwargs}


@pytest.mark.parametrize("rows,match", [
    ([reference_csv_row(), reference_csv_row()], "duplicate"),
    ([reference_csv_row(criterion_index="-1")], "malformed"),
    ([reference_csv_row(criterion_index="2")], "noncontiguous"),
    ([reference_csv_row(criterion_index="bad")], "invalid criterion index"),
    ([reference_csv_row(prediction="unknown")], "non-binary"),
    ([reference_csv_row(task_id="")], "missing identity"),
    ([reference_csv_row(), reference_csv_row(run="other", gold="False")], "reference differs"),
])
def test_malformed_reference_identifiers_and_labels_rejected(tmp_path, rows, match):
    path = tmp_path / "test.csv"
    write_csv(path, rows)
    with pytest.raises(ValueError, match=match):
        load_reference_runs(path)


def test_release_parser_accepts_named_criteria_and_distinct_domain_task_ids(tmp_path):
    path = tmp_path / "test.csv"
    write_csv(path, [reference_csv_row(), reference_csv_row(domain="different")])
    assert len(load_reference_runs(path)["judge"]) == 2


def test_actual_public_population_contract():
    runs = load_reference_runs()
    assert len(runs) == 4
    assert sum(len(rows) for rows in runs.values()) == 4916
    assert len({row["task_id"] for rows in runs.values() for row in rows}) == 494
    for rows in runs.values():
        release = disclose_reference(rows, True)
        bounds = conjunction_bounds(release)["pass_rate_bounds"]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["task_id"]].append(row["gold"])
        truth = np.mean([all(labels) for labels in grouped.values()])
        assert bounds[0] <= truth <= bounds[1]


def test_decision_rule_keeps_unresolved_and_equivalence_distinct():
    assert decision_state([-0.05, 0.05]) == "unresolved"
    assert decision_state([-0.01, 0.01]) == "practically_equivalent"
    assert decision_state([0.02, 0.10]) == "A_better"
    assert decision_state([-0.10, -0.02]) == "B_better"
