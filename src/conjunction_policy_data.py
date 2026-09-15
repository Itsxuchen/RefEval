"""Strict offline inputs for the matched-budget conjunction policy study.

The reference is a fixed released bit, not newly collected human truth. JB's
unit is an output-by-rater annotation; ``base_task_ids`` preserves clustering.
Binary-only is a different researcher-defined target, not a cleaned version.
No policy, query order, or sampling claim is implemented in this loader.
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUVER_NAMES = ("DR Gemini 3.1 Pro", "DR Qwen-plus", "AC Qwen-plus", "AC DeepSeek v4-pro", "DR GPT-5.4 low")
JB_NAMES = {"gpt_5_4": "JB GPT-5.4", "gpt_5_4_mini": "JB GPT-5.4-mini"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def binary(value: str) -> int:
    require(value in {"0", "1", "False", "True"}, f"Non-binary label: {value!r}")
    return int(value in {"1", "True"})


def hashes(paths: list[Path], root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def indexed(rows: list[dict], field: str, prefix: str = "") -> dict[str, dict]:
    result = {}
    for row in rows:
        value = row[field]
        require(value.startswith(prefix), f"Missing identity prefix {prefix!r}: {value}")
        key = value[len(prefix):]
        require(bool(key) and key not in result, f"Empty or duplicate {field}: {key}")
        result[key] = row
    return result


def read_ruver_rows(paths: list[Path]) -> dict[str, list[dict]]:
    """Read full runs without silently collapsing duplicate identities or runs."""
    runs = defaultdict(list)
    seen = set()
    for path in paths:
        for raw in read_csv(path):
            name, task, cid = raw["run"], raw["task_id"], raw["criterion_id"]
            require(bool(name) and bool(task) and bool(cid), "Empty RuVer identity")
            slot = int(raw["criterion_index"])
            require(slot >= 0, f"Negative criterion slot: {task}/{slot}")
            identity = (name, task, slot)
            require(identity not in seen, f"Duplicate criterion slot: {identity}")
            seen.add(identity)
            runs[name].append({"task_id": task, "criterion_id": cid, "slot": slot,
                               "category": raw["category"], "gold": binary(raw["gold"]),
                               "prediction": binary(raw["prediction"]), "domain": raw["domain"]})
    for name, rows in runs.items():
        require(len({(r["task_id"], r["criterion_id"]) for r in rows}) == len(rows), f"Duplicate criterion id within task: {name}")
        rows.sort(key=lambda r: (r["task_id"], r["slot"]))
        per_task = defaultdict(list)
        for row in rows:
            per_task[row["task_id"]].append(row["slot"])
        require(all(slots == list(range(len(slots))) for slots in per_task.values()),
                f"Noncontiguous retained criterion slots: {name}")
    return dict(runs)


def aligned(primary: list[dict], secondary: list[dict], label: str) -> None:
    fields = ("task_id", "criterion_id", "slot", "category", "gold")
    require(len(primary) == len(secondary), f"Population mismatch: {label}")
    for first, second in zip(primary, secondary):
        require(all(first[key] == second[key] for key in fields),
                f"Identity/category/gold alignment mismatch: {label}/{first['task_id']}/{first['slot']}")


def pack(name: str, rows: list[dict], peers: dict[str, list[dict]], metadata: dict[str, dict],
         source_hashes: dict[str, str], family: str, target: str) -> dict:
    require(bool(rows), f"Empty dataset: {name}")
    for peer_name, peer_rows in peers.items():
        aligned(rows, peer_rows, peer_name)
    task_ids = list(dict.fromkeys(r["task_id"] for r in rows))
    ids = {task: i for i, task in enumerate(task_ids)}
    require(set(task_ids) == set(metadata), f"Task metadata coverage mismatch: {name}")
    arrays = {
        "gold": np.array([r["gold"] for r in rows], dtype=np.uint8),
        "prediction": np.array([r["prediction"] for r in rows], dtype=np.uint8),
        "task_index": np.array([ids[r["task_id"]] for r in rows], dtype=np.int64),
        "secondary_predictions": np.array([[r["prediction"] for r in peer] for peer in peers.values()],
                                            dtype=np.uint8).reshape(len(peers), len(rows)),
    }
    require(np.all(np.diff(arrays["task_index"]) >= 0), f"Noncontiguous task rows: {name}")
    require(all(r[k] in (0, 1) for r in rows for k in ("gold", "prediction")), "Non-binary packed label")
    for array in arrays.values():
        array.flags.writeable = False
    return {"name": name, **arrays, "criterion_ids": [r["criterion_id"] for r in rows],
            "category": [r["category"] for r in rows], "task_ids": task_ids,
            "base_task_ids": [metadata[t]["base_task_id"] for t in task_ids],
            "output_ids": [metadata[t]["output_id"] for t in task_ids],
            "rater_ids": [metadata[t]["rater_id"] for t in task_ids],
            "secondaries": list(peers), "source_hashes": source_hashes,
            "target": target, "family": family}


def read_judgmentbench(root: Path) -> tuple[dict, dict, dict, list[Path]]:
    """Validate raw domains, points, complete rubrics and repeated-output joins."""
    raw = root / "data/reference/judgmentbench"
    paths = [raw / "rubric_items.csv"]
    items = indexed(read_csv(paths[0]), "rubric_item_id")
    by_task = defaultdict(set)
    for item_id, item in items.items():
        mode = item["scoring_mode"]
        weight, maximum = Decimal(item["weight"]), Decimal(item["max_score"])
        require(mode in {"binary", "occurrence_count"}, f"Unknown scoring mode: {mode}")
        require((mode == "binary" and weight == maximum and weight > 0) or
                (mode == "occurrence_count" and weight < 0 and maximum == 0),
                f"Unexpected scoring weights: {item_id}")
        require(int(item["item_order"]) > 0, f"Invalid item order: {item_id}")
        by_task[item["task_id"]].add(item_id)
    for task, ids in by_task.items():
        require(len(ids) == len({items[i]["item_order"] for i in ids}), f"Duplicate item order: {task}")
    labels, metadata = {}, {}
    for source in ("human", *JB_NAMES):
        annotation_path, score_path = raw / f"{source}_annotations_rubric.csv", raw / f"{source}_rubric_item_scores.csv"
        paths.extend([annotation_path, score_path])
        prefix = "" if source == "human" else "autograder_"
        annotations = indexed(read_csv(annotation_path), "annotation_id", prefix)
        table, totals = defaultdict(dict), defaultdict(Decimal)
        for row in read_csv(score_path):
            aid, item_id = row["annotation_id"], row["rubric_item_id"]
            require(aid.startswith(prefix), f"Unexpected score annotation: {aid}")
            aid = aid[len(prefix):]
            require(aid in annotations and item_id in items, f"Unknown score identity: {aid}/{item_id}")
            require(item_id not in table[aid], f"Duplicate score: {aid}/{item_id}")
            item = items[item_id]
            require(item["task_id"] == annotations[aid]["task_id"], f"Cross-task item: {aid}/{item_id}")
            value = Decimal(row["raw_value"])
            require(value.is_finite() and value == value.to_integral_value(), f"Invalid raw value: {aid}/{item_id}")
            require(value in (0, 1) if item["scoring_mode"] == "binary" else value >= 0,
                    f"Out-of-domain raw value: {aid}/{item_id}")
            award = Decimal(row["awarded_points"])
            require(award == Decimal(item["weight"]) * value, f"Incorrect award: {aid}/{item_id}")
            require(int(row["score_order"]) == int(item["item_order"]), f"Order mismatch: {aid}/{item_id}")
            table[aid][item_id] = int(value == (1 if item["scoring_mode"] == "binary" else 0))
            totals[aid] += award
        require(set(table) == set(annotations), f"Missing annotations: {source}")
        for aid, annotation in annotations.items():
            require(set(table[aid]) == by_task[annotation["task_id"]], f"Incomplete rubric: {source}/{aid}")
            maximum = Decimal(annotation["rubric_max_points"])
            require(totals[aid] == Decimal(annotation["rubric_total_points"]), f"Total mismatch: {source}/{aid}")
            require(sum(Decimal(items[i]["max_score"]) for i in table[aid]) == maximum,
                    f"Maximum mismatch: {source}/{aid}")
            require(all(table[aid].values()) == (totals[aid] == maximum), f"Full credit mismatch: {source}/{aid}")
        labels[source], metadata[source] = dict(table), annotations
    for source in JB_NAMES:
        require(set(metadata[source]) == set(metadata["human"]), f"Source population mismatch: {source}")
        for aid, human in metadata["human"].items():
            judge = metadata[source][aid]
            for key in ("task_id", "output_id", "output_quality_level", "output_quality_level_order",
                        "annotation_order", "task_slot_order", "method_step_order", "rubric_max_points"):
                require(human[key] == judge[key], f"Annotation metadata mismatch: {source}/{aid}/{key}")
            require(human["annotator_id"] == judge["corresponding_annotator_id"], f"Rater mapping mismatch: {aid}")
    output_path = root / "data/reference/judgmentbench/outputs_metadata.csv"
    paths.append(output_path)
    outputs = indexed(read_csv(output_path), "output_id")
    for aid, human in metadata["human"].items():
        require(human["output_id"] in outputs, f"Missing output: {aid}")
        output = outputs[human["output_id"]]
        require(human["task_id"] == output["task_id"] and human["output_quality_level"] == output["quality_level"],
                f"Output identity mismatch: {aid}")
    return items, labels, metadata["human"], paths


def load_datasets(root: Path | None = None) -> dict[str, dict]:
    root = (root or ROOT).resolve()
    paths = [root / "data/processed/public_judge_predictions.csv", root / "data/processed/gpt54_low_dr_predictions.csv"]
    runs = read_ruver_rows(paths)
    require(set(runs) == set(RUVER_NAMES), f"Unexpected RuVer runs: {sorted(runs)}")
    datasets = {}
    for name in RUVER_NAMES:
        rows = runs[name]
        domain = "deepresearch" if name.startswith("DR ") else "agenticcoding"
        require({r["domain"] for r in rows} == {domain}, f"Unexpected source domain: {name}")
        peers = {peer: runs[peer] for peer in RUVER_NAMES if peer != name and peer[:2] == name[:2]}
        tasks = list(dict.fromkeys(r["task_id"] for r in rows))
        meta = {task: {"base_task_id": task, "output_id": task, "rater_id": "released_reference"} for task in tasks}
        datasets[name] = pack(name, rows, peers, meta, hashes(paths, root), "RuVerBench",
                              "all retained binary criteria pass against the fixed released reference")
    items, labels, human_meta, paths = read_judgmentbench(root)
    for binary_only in (False, True):
        suffix = " | binary_only" if binary_only else ""
        jb_runs = {}
        meta = {}
        for source, name in JB_NAMES.items():
            rows = []
            for aid in sorted(labels["human"]):
                task = f"judgmentbench/{aid}"
                human = human_meta[aid]
                meta[task] = {"base_task_id": human["task_id"], "output_id": human["output_id"], "rater_id": human["annotator_id"]}
                kept = sorted((i for i in labels["human"][aid] if not binary_only or items[i]["scoring_mode"] == "binary"),
                              key=lambda i: int(items[i]["item_order"]))
                require(bool(kept), f"Empty rubric after binary-only filtering: {aid}")
                for slot, item_id in enumerate(kept):
                    rows.append({"task_id": task, "criterion_id": item_id, "slot": slot,
                                 "category": items[item_id]["scoring_mode"], "gold": labels["human"][aid][item_id],
                                 "prediction": labels[source][aid][item_id]})
            jb_runs[name + suffix] = rows
        for name, rows in jb_runs.items():
            datasets[name] = pack(name, rows, {n: r for n, r in jb_runs.items() if n != name}, meta,
                                  hashes(paths, root), "JudgmentBench",
                                  "all positive binary items receive full credit; penalty items excluded (changed target)" if binary_only else
                                  "all positive binary items receive full credit AND all penalty occurrence counts are zero")
    return datasets
