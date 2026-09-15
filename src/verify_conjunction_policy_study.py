"""Independently recompute canonical real-data metrics from acquired query sets.

The strict production loader and policy selector are reused. The production
metric deltas, curve builder and metric helpers are never used. Thus this is an
independent arithmetic check, not an independent raw-data or policy audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.conjunction_policy_data import load_datasets
from src.conjunction_policy_study import replay

ROOT = Path(__file__).resolve().parents[1]


def direct_metrics(data: dict, indices: list[int], cap: int) -> dict:
    """Literal observed-bit replacement and per-task certification calculation."""
    queried = set(indices)
    if len(queried) != len(indices):
        raise AssertionError("repeated query")
    by_task, sequence = defaultdict(list), defaultdict(list)
    for i, t in enumerate(data["task_index"]):
        by_task[int(t)].append(i)
    for i in indices:
        sequence[int(data["task_index"][i])].append(i)
    g, p = data["gold"], data["prediction"]
    counts = dict.fromkeys(("criterion_errors_found", "certified_tasks", "residual_fp", "residual_ff",
                           "introduced_fp", "fixed_initial_fp", "fixed_initial_ff", "tasks_touched",
                           "new_fp_events", "certified_initial_ff", "gated_residual_fp", "gated_residual_ff"), 0)
    initial_errors = 0
    for t, slots in by_task.items():
        observed = set(slots) & queried
        truth, initial = all(g[i] for i in slots), all(p[i] for i in slots)
        current = all(g[i] if i in observed else p[i] for i in slots)
        certified = any(not g[i] for i in observed) or len(observed) == len(slots)
        gated = truth if certified else initial
        counts["criterion_errors_found"] += sum(g[i] != p[i] for i in observed)
        counts["certified_tasks"] += certified
        counts["residual_fp"] += current and not truth
        counts["residual_ff"] += truth and not current
        counts["introduced_fp"] += current and not truth and not initial
        counts["fixed_initial_fp"] += initial and not truth and not current
        counts["fixed_initial_ff"] += truth and not initial and current
        counts["tasks_touched"] += bool(observed)
        counts["certified_initial_ff"] += truth and not initial and certified
        counts["gated_residual_fp"] += gated and not truth
        counts["gated_residual_ff"] += truth and not gated
        initial_errors += initial != truth
        if certified and current != truth:
            raise AssertionError("certified but incorrect")
        working = {i: bool(p[i]) for i in slots}
        before = initial
        for i in sequence[t]:
            working[i] = bool(g[i])
            after = all(working.values())
            counts["new_fp_events"] += after and not before and not truth
            before = after
    counts = {k: int(v) for k, v in counts.items()}
    counts.update(budget_cap=cap, actual_queries=len(indices), n_tasks=len(by_task), n_criteria=len(g),
                  primary_criterion_errors=int((g != p).sum()), initial_task_errors=initial_errors,
                  residual_errors=counts["residual_fp"] + counts["residual_ff"],
                  uncertified_tasks=len(by_task) - counts["certified_tasks"], unused_budget=cap - len(indices),
                  gated_residual_errors=counts["gated_residual_fp"] + counts["gated_residual_ff"],
                  delayed_ff_corrections=counts["gated_residual_ff"] - counts["residual_ff"])
    return counts


def check_hashes(mapping: dict, root: Path) -> list[dict]:
    checks = []
    for name, expected in mapping.items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        checks.append({"path": name, "expected": expected, "actual": actual, "matches": expected == actual})
    return checks


def verify(root: Path = ROOT, directory: Path | None = None) -> dict:
    directory = directory or root / "artifacts/reproduced/conjunction_policy_study"
    study = json.loads((directory / "study.json").read_text())
    data = load_datasets(root)
    rows = []
    for saved in study["canonical_20pct"]:
        if saved["task_order"] not in {"release", "predicted_fail_count"}:
            continue
        cell = data[saved["dataset"]]
        # Deliberately discard all returned evaluator deltas.
        transcript = replay(cell, saved["policy"], saved["task_order"], 0)["indices"]
        cap = int(round(.2 * len(cell["gold"])))
        indices = transcript[:cap].tolist()
        expected = direct_metrics(cell, indices, cap)
        mismatches = {key: {"saved": saved[key], "recomputed": value}
                      for key, value in expected.items() if saved[key] != value}
        rows.append({"dataset": saved["dataset"], "policy": saved["policy"], "task_order": saved["task_order"],
                     "actual_queries": len(indices), "metrics_checked": len(expected), "mismatches": mismatches})
    source_map = dict(study["source_hashes"])
    for cell in study["datasets"].values():
        for name, value in cell["source_hashes"].items():
            if name in source_map and source_map[name] != value:
                raise AssertionError(f"conflicting source receipt: {name}")
            source_map[name] = value
    source_checks = check_hashes(source_map, root)
    okay = (len(rows) == 126 and all(not r["mismatches"] for r in rows)
            and all(r["matches"] for r in source_checks))
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": okay,
            "scope": "Canonical seed0,20percent budget,9cells,7policies,release and predicted-fail task orders; arithmetic independent of production metrics; loader and query selection reused.",
            "not_checked_here": ["other seeds and budget points", "raw joins independently of loader", "full query-selection implementation independently", "population inference", "human reference validity"],
            "canonical_rows_checked": len(rows), "numeric_fields_checked": sum(r["metrics_checked"] for r in rows),
            "canonical_rows": rows, "study_source_hash_checks": source_checks,
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "small_world_validation_location": "tests/test_conjunction_policy_study.py",
            "exhaustive_world_scope": {"worlds": 4 + 16 + 64, "policies": 7, "task_orders": 3, "seeds": 2,
                                       "traces": (4 + 16 + 64) * 7 * 3 * 2,
                                       "integer_budget_snapshots": (4 * 2 + 16 * 3 + 64 * 4) * 7 * 3 * 2}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/reproduced/conjunction_policy_study")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reproduced/conjunction_policy_study/validation.json")
    args = parser.parse_args()
    result = verify(directory=args.directory)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "canonical_rows_checked", "numeric_fields_checked")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
