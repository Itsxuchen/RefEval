"""Offline reproduction entry point; never writes frozen expected results.

Smoke executes bounded, real-data subsets. Full executes the complete published
seed grids, deletion sensitivity, summaries, figures and information replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "artifacts/expected"


def configuration(mode: str) -> dict:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    return {"mode": mode, "policy_seeds": 32 if mode == "full" else 2,
            "label_seeds": 31 if mode == "full" else 1,
            "disclosure_orders": 10 if mode == "full" else 1,
            "leave_one_base_task_out": mode == "full",
            "policy_summaries_and_figures": mode == "full"}


def validate_output(output: Path, root: Path = ROOT) -> Path:
    """Refuse source/data/reference overlap, including symlink aliases."""
    root = root.resolve()
    target = (output if output.is_absolute() else root / output).resolve()
    protected = [root / p for p in ("artifacts/expected", "artifacts/figures", "artifacts/validation", "src", "tests", "data", "context", "docs")]
    for path in protected:
        path = path.resolve()
        if target == path or path in target.parents or target in path.parents:
            raise ValueError("output overlaps protected package files or frozen expected results")
    if root in target.parents and root / "artifacts" not in target.parents:
        raise ValueError("outputs within the package must be inside artifacts")
    if target.exists():
        if not target.is_dir():
            raise ValueError("output must be a directory")
        if any(p.is_symlink() for p in target.rglob("*")):
            raise ValueError("existing output contains symlinks; choose a clean output directory")
    return target


def digest_tree(directory: Path) -> dict:
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reproduce(mode: str, output: Path) -> dict:
    config = configuration(mode)
    output = validate_output(output)
    frozen_before = digest_tree(EXPECTED)
    if not frozen_before:
        raise FileNotFoundError("frozen expected results are missing")
    from src.conjunction_policy_data import load_datasets
    from src.conjunction_policy_study import run_study
    from src.conjunction_label_value import run_experiment as run_label
    from src.conjunction_error_flow import run_experiment as run_flow
    from src.information_disclosure_replay import run_experiment as run_disclosure
    from src.verify_conjunction_policy_study import verify as verify_policy
    from src.verify_conjunction_label_value import verify as verify_label
    from src.release_validation import verify_outputs

    started = time.perf_counter()
    stages = {}
    output.mkdir(parents=True, exist_ok=True)
    data = load_datasets(ROOT)
    if len(data) != 9:
        raise AssertionError("expected nine fixed-reference cells")
    policy, label = output / "conjunction_policy_study", output / "conjunction_label_value"

    def stage(name, action):
        print(f"[{mode}] {name}", flush=True)
        tick = time.perf_counter()
        result = action()
        stages[name] = round(time.perf_counter() - tick, 3)
        return result

    stage("policy_grid", lambda: run_study(data, config["policy_seeds"], policy,
                                           loto=config["leave_one_base_task_out"]))
    stage("error_flow", lambda: run_flow(data, policy / "error_flow.json"))
    if config["policy_summaries_and_figures"]:
        from src.summarize_conjunction_policy_study import summarize
        from src.plot_conjunction_policy_study import plot
        stage("policy_summary", lambda: summarize(policy))
        stage("policy_figures", lambda: plot(policy, output / "figures/conjunction_policy_study"))
    stage("label_value_and_task_sampling", lambda: run_label(data, config["label_seeds"], label))
    stage("information_disclosure", lambda: run_disclosure(config["disclosure_orders"],
          output / "information_disclosure_replay.json", output / "figures/information_disclosure_replay.png"))
    direct_policy = stage("independent_policy_arithmetic", lambda: verify_policy(ROOT, policy))
    if not direct_policy["passed"]:
        raise AssertionError("independent policy arithmetic failed")
    write_json(policy / "validation.json", direct_policy)
    direct_label = stage("independent_label_arithmetic", lambda: verify_label(label, policy))
    write_json(label / "validation.json", direct_label)
    validation = stage("frozen_scientific_comparison", lambda: verify_outputs(output, EXPECTED, mode))
    if digest_tree(EXPECTED) != frozen_before:
        raise AssertionError("frozen expected files changed during reproduction")
    validation["frozen_expected_files_unchanged"] = len(frozen_before)
    validation["direct_arithmetic"] = {
        "policy_canonical_rows": direct_policy["canonical_rows_checked"],
        "policy_numeric_fields": direct_policy["numeric_fields_checked"],
        "label_hybrid_rows": direct_label["independently_scored_hybrid_rows"],
        "whole_task_rows": direct_label["raw_rechecked_whole_task_rows"]}
    write_json(output / "verification.json", validation)
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "configuration": config,
               "output": os.path.relpath(output, ROOT), "datasets": list(data),
               "stage_seconds": stages, "elapsed_seconds": round(time.perf_counter() - started, 3),
               "passed": True, "complete_reproduction": mode == "full",
               "scope": "Fixed released numeric references; no model APIs or new human labels."}
    write_json(output / "run_manifest.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reproduced"))
    args = parser.parse_args()
    try:
        validate_output(args.output)
    except ValueError as error:
        parser.error(str(error))
    reproduce(args.mode, args.output)


if __name__ == "__main__":
    main()
