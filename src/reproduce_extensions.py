"""Reproduce and verify the complete retrospective mechanism/robustness extension.

The original study has its own src.reproduce entry point. This command runs the
new, separately specified grids from the same local numeric inputs. It never
writes expected tables. --verify-only rechecks every scientific extension table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.reproduce import ROOT, EXPECTED, digest_tree, validate_output, write_json
from src.release_validation import assert_equal, compare_csv

CONFIG = {"mechanism_permutations": 64, "order_seeds": 31,
          "cluster_replicates": 399, "cluster_seed": 20261008,
          "reference_cells": 4, "stratified_cells": 9}
DIRECTORIES = ("conjunction_mechanism", "conjunction_robustness/reference",
               "conjunction_robustness/cluster", "conjunction_robustness/estimation")
JSON_TARGETS = ("conjunction_mechanism/update_structures.json",
                "conjunction_robustness/reference/frame_diagnostics.json",
                "conjunction_robustness/cluster/summary.json",
                "conjunction_robustness/estimation/interval_regression.json")


def verify(output: Path) -> dict:
    output = validate_output(output)
    tables = {}
    for folder in DIRECTORIES:
        expected = EXPECTED / folder
        if not expected.is_dir():
            raise FileNotFoundError(f"Missing extension targets: {folder}")
        wanted = sorted(p.name for p in expected.iterdir()
                        if p.name.endswith((".csv", ".csv.gz")))
        actual = sorted(p.name for p in (output / folder).iterdir()
                        if p.name.endswith((".csv", ".csv.gz")))
        if actual != wanted or not wanted:
            raise AssertionError(f"Scientific table inventory differs: {folder}")
        for name in wanted:
            relative = f"{folder}/{name}"
            tables[relative] = compare_csv(output / relative, EXPECTED / relative)
    payloads = {}
    for relative in JSON_TARGETS:
        payloads[relative] = assert_equal(json.loads((output / relative).read_text()),
                                         json.loads((EXPECTED / relative).read_text()), relative)
    result = {"passed": True, "configuration": CONFIG, "csv_checks": tables,
              "json_scientific_leaves": payloads,
              "scientific_rows": sum(r["rows"] for r in tables.values()),
              "comparison_tolerance": {"absolute": 1e-8, "relative": 1e-9},
              "scope": "All extension scientific CSV rows and four scientific JSON payloads; source paths, timestamps and compressed byte headers may differ."}
    write_json(output / "extension_verification.json", result)
    return result


def reproduce(output: Path) -> dict:
    output = validate_output(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory to preserve previous runs")
    from src.conjunction_update_mechanism import analyze
    from src.conjunction_allocation_mechanism import run as allocation
    from src.conjunction_reference_sensitivity import run as reference
    from src.conjunction_cluster_sensitivity import run as cluster
    from src.conjunction_stratified_baseline import run as estimation
    frozen = digest_tree(EXPECTED)
    started, stages = time.perf_counter(), {}
    output.mkdir(parents=True, exist_ok=True)

    def stage(name, action):
        print(f"[extensions] {name}", flush=True)
        tick = time.perf_counter()
        value = action()
        stages[name] = round(time.perf_counter() - tick, 3)
        return value

    mechanism = output / DIRECTORIES[0]
    stage("mechanism_update", lambda: analyze(ROOT, mechanism, permutations=64))
    stage("mechanism_allocation", lambda: allocation(mechanism, seeds=31))
    stage("matched_reference", lambda: reference(ROOT, output / DIRECTORIES[1], seeds=31))
    stage("task_composition", lambda: cluster(ROOT, output / DIRECTORIES[2],
          ROOT / "context/conjunction_robustness_protocol.md", replicates=399, seed=20261008))
    stage("stratified_estimation", lambda: estimation(ROOT, output / DIRECTORIES[3],
          seeds=range(1, 32), saved_reference={
              "budget": EXPECTED / "conjunction_label_value/budget_estimation_rows.csv.gz",
              "tasks": EXPECTED / "conjunction_label_value/whole_task_sampling.csv.gz"}))
    from src.render_conjunction_mechanism import main as mechanism_figures
    from src.render_conjunction_robustness import render as robustness_figures
    stage("mechanism_figures", lambda: mechanism_figures(mechanism, output / "figures"))
    stage("robustness_figures", lambda: robustness_figures(output / "conjunction_robustness", output / "figures"))
    result = stage("all_scientific_targets", lambda: verify(output))
    if frozen != digest_tree(EXPECTED):
        raise AssertionError("Frozen expected files changed during extension replay")
    sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ("src", "context") for p in sorted((ROOT / folder).iterdir())
               if p.is_file() and p.suffix in (".py", ".md")}
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "configuration": CONFIG, "passed": result["passed"],
               "complete_extension_reproduction": True,
               "stage_seconds": stages, "seconds": round(time.perf_counter() - started, 3),
               "sources_sha256": sources, "frozen_expected_files_unchanged": len(frozen),
               "scope": "Retrospective numeric extension; run src.reproduce separately for the original policy/disclosure study. No API calls or newly collected labels."}
    write_json(output / "extension_run_manifest.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reproduced-extensions"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = validate_output(args.output)
    if args.verify_only:
        receipt = json.loads((output / "extension_run_manifest.json").read_text())
        if receipt["configuration"] != CONFIG or not receipt["passed"]:
            raise AssertionError("A successful matching full extension run is required")
        print(json.dumps(verify(output), indent=2))
    else:
        reproduce(output)


if __name__ == "__main__":
    main()
