"""Align projected numeric JudgmentBench labels into paired quality-tier blocks.

The fixed human scores are references; constructed tiers are not named systems.
This module only supplies the F28 replay loader, not a standalone report.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


RAW = Path(__file__).resolve().parents[1] / "data/reference/judgmentbench"
FILES = {
    "human": RAW / "human_annotations_rubric.csv",
    "gpt_5_4": RAW / "gpt_5_4_annotations_rubric.csv",
    "gpt_5_4_mini": RAW / "gpt_5_4_mini_annotations_rubric.csv",
}
ITEM_FILES = {
    "human": RAW / "human_rubric_item_scores.csv",
    "gpt_5_4": RAW / "gpt_5_4_rubric_item_scores.csv",
    "gpt_5_4_mini": RAW / "gpt_5_4_mini_rubric_item_scores.csv",
}
RUBRIC_FILE = RAW / "rubric_items.csv"
QUALITY_LEVELS = ("intermediate", "good", "excellent")
ALIGNMENT_FIELDS = (
    "annotation_order",
    "task_id",
    "task_slot_order",
    "method_step_order",
    "output_id",
    "output_quality_level",
    "output_quality_level_order",
    "rubric_max_points",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_aligned_blocks(
    files: dict[str, Path] = FILES,
    item_files: dict[str, Path] = ITEM_FILES,
    rubric_file: Path = RUBRIC_FILE,
) -> list[dict]:
    tables = {name: _read(path) for name, path in files.items()}
    lengths = {name: len(rows) for name, rows in tables.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"annotation tables have different lengths: {lengths}")

    human = tables["human"]
    for judge in ("gpt_5_4", "gpt_5_4_mini"):
        for index, (left, right) in enumerate(zip(human, tables[judge])):
            mismatch = [field for field in ALIGNMENT_FIELDS if left[field] != right[field]]
            if left["annotator_id"] != right["corresponding_annotator_id"]:
                mismatch.append("annotator_id")
            if mismatch:
                raise ValueError(
                    f"{judge} row {index} is not aligned on {sorted(set(mismatch))}"
                )

    rubric_rows = _read(rubric_file)
    rubric_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    rubric_lookup = {}
    for row in rubric_rows:
        rubric_by_task[row["task_id"]].append(row)
        rubric_lookup[row["rubric_item_id"]] = row
    for task_id in rubric_by_task:
        rubric_by_task[task_id].sort(key=lambda row: int(row["item_order"]))

    item_tables = {name: _read(path) for name, path in item_files.items()}
    items_by_annotation: dict[str, dict[str, list[dict[str, str]]]] = {}
    for source, rows in item_tables.items():
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["annotation_id"]].append(row)
        for annotation_id in grouped:
            grouped[annotation_id].sort(key=lambda row: int(row["score_order"]))
        items_by_annotation[source] = grouped

    blocks: dict[tuple[str, str, str], dict] = {}
    for index, human_row in enumerate(human):
        key = (
            human_row["annotator_id"],
            human_row["task_id"],
            human_row["task_slot_order"],
        )
        block = blocks.setdefault(
            key,
            {
                "block_id": "/".join(key),
                "annotator_id": key[0],
                "task_id": key[1],
                "task_slot_order": key[2],
                "scores": defaultdict(dict),
            },
        )
        quality = human_row["output_quality_level"]
        for source, row in (
            ("human", human_row),
            ("gpt_5_4", tables["gpt_5_4"][index]),
            ("gpt_5_4_mini", tables["gpt_5_4_mini"][index]),
        ):
            points = float(row["rubric_total_points"])
            maximum = float(row["rubric_max_points"])
            if maximum <= 0:
                raise ValueError("rubric_max_points must be positive")
            item_rows = items_by_annotation[source].get(row["annotation_id"], [])
            expected = rubric_by_task[row["task_id"]]
            if [item["rubric_item_id"] for item in item_rows] != [
                item["rubric_item_id"] for item in expected
            ]:
                raise ValueError(
                    f"{source} item scores do not align for annotation {row['annotation_id']}"
                )
            item_met = []
            for item_score in item_rows:
                rubric = rubric_lookup[item_score["rubric_item_id"]]
                raw = float(item_score["raw_value"])
                awarded = float(item_score["awarded_points"])
                if rubric["scoring_mode"] == "binary":
                    met = awarded >= float(rubric["max_score"]) - 1e-12
                elif rubric["scoring_mode"] == "occurrence_count":
                    met = raw == 0.0
                else:
                    raise ValueError(f"unknown scoring mode: {rubric['scoring_mode']}")
                item_met.append(float(met))
            pass_all_items = float(all(item_met))
            pass_from_total = float(points >= maximum - 1e-12)
            if pass_all_items != pass_from_total:
                raise ValueError(
                    f"strict item conjunction disagrees with rubric total for {row['annotation_id']}"
                )
            block["scores"][source][quality] = {
                "normalized": points / maximum,
                "mean_criteria": float(np.mean(item_met)),
                "pass_all": pass_all_items,
                "n_items": len(item_met),
                "output_id": row["output_id"],
            }

    output: list[dict] = []
    for block in blocks.values():
        for source in FILES:
            if set(block["scores"][source]) != set(QUALITY_LEVELS):
                raise ValueError(f"incomplete quality triplet: {block['block_id']} {source}")
        row = {key: value for key, value in block.items() if key != "scores"}
        for source in FILES:
            scores = block["scores"][source]
            row[f"{source}_excellent_pass"] = scores["excellent"]["pass_all"]
            row[f"{source}_good_pass"] = scores["good"]["pass_all"]
            row[f"{source}_pass_gap"] = (
                scores["excellent"]["pass_all"] - scores["good"]["pass_all"]
            )
            row[f"{source}_score_gap"] = (
                scores["excellent"]["normalized"] - scores["good"]["normalized"]
            )
            row[f"{source}_excellent_mean_criteria"] = scores["excellent"][
                "mean_criteria"
            ]
            row[f"{source}_good_mean_criteria"] = scores["good"]["mean_criteria"]
            row[f"{source}_mean_criteria_gap"] = (
                scores["excellent"]["mean_criteria"]
                - scores["good"]["mean_criteria"]
            )
            row[f"{source}_rubric_items"] = scores["excellent"]["n_items"]
        output.append(row)
    return sorted(output, key=lambda row: row["block_id"])

