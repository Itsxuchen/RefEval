"""Input-contract tests: reject corrupt joins and preserve scoring constructs."""
import csv
from pathlib import Path

import numpy as np
import pytest

from src.conjunction_policy_data import aligned, load_datasets, read_judgmentbench, read_ruver_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ruver_row(**changes):
    return {"run": "DR primary", "task_id": "task1", "criterion_id": "item1", "criterion_index": "0",
            "category": "facts", "domain": "deepresearch", "gold": "True", "prediction": "False", **changes}


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows + rows, "Duplicate criterion slot"),
    (lambda rows: [dict(rows[0], prediction="maybe")], "Non-binary"),
    (lambda rows: [dict(rows[0], gold="2")], "Non-binary"),
    (lambda rows: rows + [dict(rows[0], criterion_index="1")], "Duplicate criterion id within task"),
    (lambda rows: [dict(rows[0], criterion_index="2")], "Noncontiguous"),
])
def test_ruver_invalid_rows_fail_closed(tmp_path, mutation, match):
    path = tmp_path / "predictions.csv"
    write_csv(path, mutation([ruver_row()]))
    with pytest.raises(ValueError, match=match):
        read_ruver_rows([path])


@pytest.mark.parametrize("key,value", [("category", "logic"), ("gold", 0), ("criterion_id", "other"), ("task_id", "other")])
def test_secondary_alignment_covers_identity_category_and_reference(key, value):
    row = {"task_id": "task1", "criterion_id": "item1", "slot": 0, "category": "facts", "gold": 1, "prediction": 0}
    with pytest.raises(ValueError, match="alignment mismatch"):
        aligned([row], [dict(row, **{key: value})], "secondary")


def jb_fixture(root):
    raw = root / "data/reference/judgmentbench"
    items = [
        {"rubric_item_id": "item1", "task_id": "task1", "item_order": "1", "scoring_mode": "binary", "weight": "2", "max_score": "2"},
        {"rubric_item_id": "item2", "task_id": "task1", "item_order": "2", "scoring_mode": "occurrence_count", "weight": "-1", "max_score": "0"},
    ]
    write_csv(raw / "rubric_items.csv", items)
    common = {"annotation_id": "ann1", "annotation_order": "1", "task_id": "task1", "task_slot_order": "1",
              "method_step_order": "1", "output_id": "out1", "output_quality_level": "good",
              "output_quality_level_order": "2", "rubric_total_points": "2", "rubric_max_points": "2"}
    for source in ("human", "gpt_5_4", "gpt_5_4_mini"):
        annotation = dict(common)
        annotation["annotator_id" if source == "human" else "corresponding_annotator_id"] = "rater1"
        if source != "human":
            annotation["annotation_id"] = "autograder_ann1"
        write_csv(raw / f"{source}_annotations_rubric.csv", [annotation])
        scores = [{"annotation_id": annotation["annotation_id"], "rubric_item_id": "item1", "score_order": "1", "raw_value": "1", "awarded_points": "2"},
                  {"annotation_id": annotation["annotation_id"], "rubric_item_id": "item2", "score_order": "2", "raw_value": "0", "awarded_points": "0"}]
        write_csv(raw / f"{source}_rubric_item_scores.csv", scores)
    write_csv(root / "data/reference/judgmentbench/outputs_metadata.csv",
              [{"output_id": "out1", "task_id": "task1", "quality_level": "good"}])
    return raw


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "Duplicate score"),
    ("nonbinary", "Out-of-domain"),
    ("award", "Incorrect award"),
    ("metadata", "metadata mismatch"),
    ("rater", "Rater mapping"),
    ("missing", "Incomplete rubric"),
])
def test_jb_corruption_is_rejected(tmp_path, mutation, match):
    raw = jb_fixture(tmp_path)
    score_path = raw / "gpt_5_4_rubric_item_scores.csv"
    with score_path.open() as stream:
        scores = list(csv.DictReader(stream))
    if mutation == "duplicate":
        scores.append(scores[0].copy())
    elif mutation == "nonbinary":
        scores[0]["raw_value"] = "2"
    elif mutation == "award":
        scores[0]["awarded_points"] = "1"
    elif mutation == "missing":
        scores.pop()
    elif mutation in {"metadata", "rater"}:
        path = raw / "gpt_5_4_annotations_rubric.csv"
        with path.open() as stream:
            metadata = list(csv.DictReader(stream))
        metadata[0]["output_id" if mutation == "metadata" else "corresponding_annotator_id"] = "wrong"
        write_csv(path, metadata)
    write_csv(score_path, scores)
    with pytest.raises(ValueError, match=match):
        read_judgmentbench(tmp_path)


def test_jb_full_credit_maps_binary_one_and_penalty_zero(tmp_path):
    jb_fixture(tmp_path)
    items, labels, metadata, paths = read_judgmentbench(tmp_path)
    assert labels["human"]["ann1"] == {"item1": 1, "item2": 1}
    assert items["item2"]["scoring_mode"] == "occurrence_count"
    assert metadata["ann1"]["annotator_id"] == "rater1"
    assert len(paths) == 8


@pytest.fixture(scope="module")
def datasets():
    return load_datasets()


def test_canonical_counts_and_real_cluster_metadata(datasets):
    assert len(datasets) == 9
    for name, data in datasets.items():
        n, m = (284, 1615) if name.startswith("DR ") else (210, 843) if name.startswith("AC ") else (1539, 20229 if "binary_only" in name else 23487)
        assert len(data["task_ids"]) == n and len(data["gold"]) == m
        assert data["secondary_predictions"].shape == (len(data["secondaries"]), m)
        assert np.all(np.diff(data["task_index"]) >= 0)
        assert len(data["category"]) == len(data["criterion_ids"]) == m
        assert not data["gold"].flags.writeable
        if name.startswith("JB "):
            assert len(set(data["base_task_ids"])) == 30
            assert len(set(data["output_ids"])) == 1314
            assert len(set(data["rater_ids"])) == 49
            assert len(data["source_hashes"]) == 8


def task_confusion(data):
    starts = np.flatnonzero(np.r_[True, np.diff(data["task_index"]) != 0])
    gold = np.minimum.reduceat(data["gold"], starts)
    pred = np.minimum.reduceat(data["prediction"], starts)
    return int(np.sum((gold == 0) & (pred == 1))), int(np.sum((gold == 1) & (pred == 0)))


def test_canonical_target_change_is_explicit_and_preserves_units(datasets):
    full = datasets["JB GPT-5.4"]
    reduced = datasets["JB GPT-5.4 | binary_only"]
    assert task_confusion(full) == (24, 220)
    assert task_confusion(reduced) == (181, 67)
    assert set(reduced["category"]) == {"binary"}
    assert "changed target" in reduced["target"]
    for key in ("task_ids", "base_task_ids", "output_ids", "rater_ids"):
        assert full[key] == reduced[key]
    assert set(reduced["source_hashes"]) == set(full["source_hashes"])
