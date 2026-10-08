"""Small-world checks of composition reconstruction and recomputed policies."""
import itertools
import numpy as np
import pytest

from src.conjunction_cluster_sensitivity import (
    allocation_panel, composition_draws, curve_summary, peak_record, query_indices,
    resample_frame, validate_output_path, validate_query_equivalence,
)
from src.conjunction_label_value import hybrid_queries
from src.conjunction_policy_study import make_order
from src.conjunction_update_mechanism import expected_curve


def toy():
    return dict(name="toy", gold=np.array([1, 1, 0, 1, 0]),
                prediction=np.array([0, 1, 1, 0, 1]),
                secondary_predictions=np.array([[1, 1, 0, 1, 1]]),
                task_index=np.array([0, 0, 1, 2, 2]), task_ids=["u0", "u1", "u2"],
                base_task_ids=["a", "b", "a"], output_ids=["out0", "out1", "out2"],
                rater_ids=["r0", "r1", "r2"], criterion_ids=["i0", "i1", "j0", "i0", "i1"],
                category=["binary"] * 5, source_hashes={})


def literal_metrics(data, queried):
    certified = errors = 0
    for t in range(len(data["task_ids"])):
        slots = [i for i, tt in enumerate(data["task_index"]) if tt == t]
        truth = all(data["gold"][i] for i in slots)
        updated = all(data["gold"][i] if i in queried else data["prediction"][i] for i in slots)
        certified += any(i in queried and not data["gold"][i] for i in slots) or all(i in queried for i in slots)
        errors += truth != updated
    return int(certified), int(errors)


def test_whole_base_cloning_preserves_order_raters_and_bits():
    data = toy()
    changed = resample_frame(data, {"a": 2, "b": 0})
    assert changed["task_ids"] == ["u0::clone_0", "u0::clone_1", "u2::clone_0", "u2::clone_1"]
    assert changed["base_task_ids"] == ["a::clone_0", "a::clone_1", "a::clone_0", "a::clone_1"]
    assert changed["rater_ids"] == ["r0", "r0", "r2", "r2"]
    np.testing.assert_array_equal(changed["gold"], [1, 1, 1, 1, 1, 0, 1, 0])
    np.testing.assert_array_equal(changed["prediction"], [0, 1, 0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(changed["task_index"], np.repeat(np.arange(4), 2))
    assert len(data["gold"]) == 5 and data["task_ids"] == ["u0", "u1", "u2"]


def test_invalid_multiplicity_rejected():
    for counts in ({"a": 0, "b": 0}, {"a": -1, "b": 3}, {"a": 1.5, "b": 1}, {"a": 1}):
        with pytest.raises(ValueError):
            resample_frame(toy(), counts)


def test_draws_are_family_shared_reproducible_and_fixed_size():
    a = composition_draws(["a", "b", "c"], 30, 20261008, "JB")
    np.testing.assert_array_equal(a, composition_draws(["a", "b", "c"], 30, 20261008, "JB"))
    np.testing.assert_array_equal(a.sum(axis=1), np.full(30, 3))
    assert not np.array_equal(a, composition_draws(["a", "b", "c"], 30, 20261008, "DR"))


def test_resampled_expected_curve_matches_literal_uniform_subsets():
    data = resample_frame(toy(), {"a": 2, "b": 0})
    m = len(data["gold"])
    for row in expected_curve(data, "random_criterion", [0, .2, .5, 1]):
        cap = row["budget_cap"]
        assert cap == round(row["budget_share"] * 8)
        differences = []
        for subset in itertools.combinations(range(m), cap):
            queried = set(subset)
            _, eager = literal_metrics(data, queried)
            gated = 0
            for t in range(4):
                slots = [i for i, tt in enumerate(data["task_index"]) if tt == t]
                truth = all(data["gold"][i] for i in slots)
                initial = all(data["prediction"][i] for i in slots)
                cert = any(i in queried and not data["gold"][i] for i in slots) or all(i in queried for i in slots)
                gated += not cert and initial != truth
            differences.append(gated - eager)
        assert row["expected_delta"] == pytest.approx(np.mean(differences), abs=1e-13)


def test_allocation_replays_new_budget_and_matches_literal_task_arithmetic():
    data = resample_frame(toy(), {"a": 2, "b": 0})
    rows, mean = allocation_panel(data, seeds=(1, 7, 31))
    assert mean["budget_cap"] == 2  # Original frame's rounded 20% cap was 1.
    for row in rows:
        seed = row["query_seed"]
        srs = set(make_order(data, "random_criterion", "release", seed)[:2])
        A, R = hybrid_queries(data, 2, "release", seed)
        mix = set(np.r_[A, R])
        sc, se = literal_metrics(data, srs)
        mc, me = literal_metrics(data, mix)
        assert row["certified_tasks_delta"] == mc - sc
        assert row["residual_errors_delta"] == me - se
    assert mean["residual_errors_delta_per100_mean"] == pytest.approx(25 * mean["residual_errors_delta_mean"])


def test_vectorized_query_path_matches_existing_replay_in_small_worlds():
    data = toy()
    for gold in itertools.product((0, 1), repeat=5):
        for pred in itertools.product((0, 1), repeat=5):
            world = {**data, "gold": np.asarray(gold), "prediction": np.asarray(pred)}
            assert validate_query_equivalence(world, seeds=(1, 7)) == 4
    changed = resample_frame(data, {"a": 2, "b": 0})
    assert validate_query_equivalence(changed, seeds=(1, 7, 31)) == 6
    with pytest.raises(ValueError):
        query_indices(data, "disagreement_only", 1)


def test_grid_band_and_peak_recomputed_instead_of_using_original_peak_budget():
    original = np.array([0., 2., 0.])
    draws = np.array([[0., 1., 3.], [0., 4., 0.]])
    result = curve_summary(original, draws)
    assert result["sup_deviation_q95"] == pytest.approx(2.95)
    assert result["pointwise_q975"][2] == pytest.approx(2.925)
    curve = [dict(expected_delta=v, budget_share=b, n_units=10) for b, v in zip([0., .5, 1.], draws[0])]
    peak = peak_record(curve)
    assert peak["max_budget_first"] == 1 and peak["max_delta_per100"] == 30
    zeros = [dict(expected_delta=0, budget_share=b, n_units=10) for b in [0., .5, 1.]]
    flat = peak_record(zeros)
    assert flat["max_tied_grid_points"] == 3 and flat["max_budget_first"] == 0 and flat["max_budget_last"] == 1


def test_output_guard_preserves_empty_protected_trees_and_symlink_targets(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    for name in ("src", "data", "tests", "context", "validation", "artifacts/expected", "artifacts/figures"):
        protected = root / name
        protected.mkdir(parents=True, exist_ok=True)
        for candidate in (protected, protected / "new-output"):
            with pytest.raises(ValueError):
                validate_output_path(root, candidate)
    for candidate in (root, root.parent, root / "artifacts"):
        with pytest.raises(ValueError):
            validate_output_path(root, candidate)
    link = tmp_path / "input-link"
    link.symlink_to(root / "data", target_is_directory=True)
    with pytest.raises(ValueError):
        validate_output_path(root, link / "new-output")
    destination = root / "artifacts/reports/conjunction_robustness/cluster"
    assert validate_output_path(root, destination) == destination
    external = tmp_path / "fresh-replay"
    assert validate_output_path(root, external) == external
    external.mkdir()
    (external / "keep.txt").write_text("preserve")
    with pytest.raises(ValueError):
        validate_output_path(root, external)
