"""Portable output guards and fail-closed scientific comparison checks."""
import copy
import csv
import json
from pathlib import Path

import pytest

from src.reproduce import configuration, validate_output
from src.release_validation import assert_equal, compare_csv, verify_disclosure_witnesses

ROOT = Path(__file__).resolve().parents[1]


def test_public_schedule_and_smoke_are_explicit():
    assert configuration("full") == dict(mode="full", policy_seeds=32, label_seeds=31,
        disclosure_orders=10, leave_one_base_task_out=True, policy_summaries_and_figures=True)
    assert configuration("smoke")["policy_seeds"] == 2
    assert not configuration("smoke")["leave_one_base_task_out"]
    with pytest.raises(ValueError):
        configuration("other")


@pytest.mark.parametrize("relative", [".", "artifacts", "artifacts/expected", "artifacts/expected/sub", "data/cache", "src/new", "artifacts/figures/new", "artifacts/validation"])
def test_output_cannot_overwrite_inputs_or_expected(tmp_path, relative):
    with pytest.raises(ValueError):
        validate_output(Path(relative), tmp_path)


def test_output_guard_resolves_aliases_and_child_links(tmp_path):
    expected = tmp_path / "artifacts/expected"
    expected.mkdir(parents=True)
    alias = tmp_path / "artifacts/alias"
    alias.symlink_to(expected, target_is_directory=True)
    with pytest.raises(ValueError):
        validate_output(alias, tmp_path)
    fresh = tmp_path / "artifacts/new"
    assert validate_output(fresh, tmp_path) == fresh
    fresh.mkdir()
    (fresh / "study.json").symlink_to(expected / "study.json")
    with pytest.raises(ValueError):
        validate_output(fresh, tmp_path)


def test_metadata_is_not_scientific_but_counts_and_identities_are():
    a = {"rows": 7, "rate": 0.3, "source_hashes": {"a": "new"}, "created_utc": "today"}
    b = {"rows": 7, "rate": 0.30000000001, "source_hashes": {"b": "old"}}
    assert assert_equal(a, b) == 2
    with pytest.raises(AssertionError):
        assert_equal({**a, "rows": 8}, b)
    with pytest.raises(AssertionError):
        assert_equal({"dataset": "A"}, {"dataset": "B"})
    with pytest.raises(AssertionError):
        assert_equal(float("nan"), float("nan"))


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "metric"])
        writer.writeheader()
        writer.writerows(rows)


def test_csv_subset_checks_every_selected_scientific_field(tmp_path):
    actual, expected = tmp_path / "a.csv", tmp_path / "e.csv"
    table(expected, [dict(id="A", metric="0.3"), dict(id="B", metric="0.6")])
    table(actual, [dict(id="B", metric="0.60000000001")])
    assert compare_csv(actual, expected, subset=True, key_fields=("id",))["rows"] == 1
    with pytest.raises(AssertionError):
        compare_csv(actual, expected)
    table(actual, [dict(id="B", metric="0.7")])
    with pytest.raises(AssertionError):
        compare_csv(actual, expected, subset=True, key_fields=("id",))
    table(actual, [dict(id="C", metric="0.6")])
    with pytest.raises(AssertionError):
        compare_csv(actual, expected, subset=True, key_fields=("id",))
    table(actual, [dict(id="B", metric="0.6")] * 2)
    with pytest.raises(AssertionError):
        compare_csv(actual, expected, subset=True, key_fields=("id",))


def test_stored_reference_witnesses_and_corruption_without_solver():
    result = json.loads((ROOT / "artifacts/expected/information_disclosure_replay.json").read_text())
    assert verify_disclosure_witnesses(result)["endpoint_witnesses"] > 1000
    changed = copy.deepcopy(result)
    certificate = changed["ruverbench"][0]["baseline"]["solver_certificates"][0]
    certificate["reference_label_witness_bits"] = "2" + certificate["reference_label_witness_bits"][1:]
    with pytest.raises(AssertionError, match="malformed"):
        verify_disclosure_witnesses(changed)


def test_projected_loaders_work_outside_project_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from src.conjunction_policy_data import load_datasets
    from src.judgmentbench_paired_gap import load_aligned_blocks
    from src.information_disclosure_replay import load_reference_runs
    assert len(load_datasets()) == 9
    assert len(load_reference_runs()) == 4
    blocks = load_aligned_blocks()
    assert len(blocks) == 513
