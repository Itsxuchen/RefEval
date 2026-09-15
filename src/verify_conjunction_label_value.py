"""Verify saved extension against prior baselines and raw fixed references."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from src.conjunction_policy_data import load_datasets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reproduced/conjunction_label_value"


def key(r):
    return r["dataset"], r["task_order"], r["policy"], int(r["seed"]), float(r["budget_share"])


def verify(directory: Path = OUT, policy_directory: Path | None = None) -> dict:
    OUT = directory
    policy_directory = policy_directory or ROOT / "artifacts/reproduced/conjunction_policy_study"
    manifest = json.loads((OUT / "manifest.json").read_text())
    all_hashes = dict(manifest["source_hashes"])
    for hashes in manifest["data_hashes"].values():
        all_hashes.update(hashes)
    for p, digest in all_hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest, p
    with gzip.open(OUT / "budget_estimation_rows.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == manifest["rows"] and len({key(r) for r in rows}) == len(rows)
    baseline = {key(r): r for r in rows if r["policy"] != "sc50_then_srs" and float(r["budget_share"]) != .006}
    checked = set()
    with gzip.open(policy_directory / "budget_curves.csv.gz", "rt") as f:
        for old in csv.DictReader(f):
            kk = key(old)
            if kk not in baseline:
                continue
            new = baseline[kk]
            for field in ("actual_queries", "certified_tasks", "residual_errors", "criterion_errors_found", "budget_cap"):
                assert int(new[field]) == int(old[field]), (kk, field)
            assert abs(float(new["pass_width"]) - int(old["uncertified_tasks"]) / int(old["n_tasks"])) < 1e-12
            checked.add(kk)
    assert checked == set(baseline), (len(checked), len(baseline))
    for r in rows:
        used, cap = int(r["actual_queries"]), int(r["budget_cap"])
        assert used <= cap and used == int(r["stageA_queries"]) + int(r["stageR_queries"])
        if r["policy"] != "sc_judge":
            assert used == cap
        assert abs(float(r["pass_width"]) - (1 - int(r["certified_tasks"]) / int(r["n_tasks"]))) < 1e-12
    datasets = load_datasets()
    # Reuse the selection transcript but score real hybrid sets in a separate
    # task-by-task loop, not through the production vectorized metric helper.
    from src.conjunction_label_value import hybrid_queries
    hybrid_checked = 0
    for r in rows:
        if r["policy"] != "sc50_then_srs" or int(r["seed"]) != 1 or float(r["budget_share"]) != .2:
            continue
        data = datasets[r["dataset"]]
        A, R = hybrid_queries(data, int(r["budget_cap"]), r["task_order"], 1)
        observed = set(np.r_[A, R].tolist())
        cert = residual = errors = passes = 0
        for t in range(len(data["task_ids"])):
            slots = np.flatnonzero(data["task_index"] == t).tolist()
            gold, pred = data["gold"], data["prediction"]
            known_fail = any(gold[i] == 0 for i in slots if i in observed)
            complete = all(i in observed for i in slots)
            cert += known_fail or complete
            passes += complete and not known_fail
            truth = all(gold[i] for i in slots)
            current = all(gold[i] if i in observed else pred[i] for i in slots)
            residual += current != truth
            errors += sum(gold[i] != pred[i] for i in slots if i in observed)
        assert cert == int(r["certified_tasks"]) and residual == int(r["residual_errors"])
        assert errors == int(r["criterion_errors_found"])
        assert abs(passes / len(data["task_ids"]) - float(r["pass_lower"])) < 1e-12
        hybrid_checked += 1
    with gzip.open(OUT / "whole_task_sampling.csv.gz", "rt") as f:
        task_rows = list(csv.DictReader(f))
    for r in task_rows:
        data = datasets[r["dataset"]]
        selected = json.loads(r["selected_task_indices"])
        selected_ids = json.loads(r["selected_task_ids"])
        queries = json.loads(r["short_circuit_query_indices"])
        assert selected_ids == [data["task_ids"][i] for i in selected]
        assert len(set(queries)) == len(queries) == int(r["short_circuit_queries"])
        assert len(set(selected)) == len(selected) == int(r["sample_n"])
        ti, g, p = data["task_index"], data["gold"], data["prediction"]
        selected_mask = np.isin(ti, selected)
        assert selected_mask[queries].all()
        assert selected_mask.sum() == int(r["full_rubric_queries"])
        observed = np.zeros(len(g), bool); observed[queries] = True
        success = 0
        for t in selected:
            mask = ti == t
            assert np.any(g[mask & observed] == 0) or observed[mask].all()
            success += int(g[mask].all())
        assert success == int(r["sample_reference_pass"])
        assert abs(success / len(selected) - float(r["task_pass_estimate"])) < 1e-12
        correct = int((g[selected_mask] == p[selected_mask]).sum())
        errors = int(selected_mask.sum()) - correct
        N, n, M = len(data["task_ids"]), len(selected), len(g)
        assert abs(N * correct / (n * M) - float(r["microaccuracy_ht_full_rubric"])) < 1e-12
        assert abs(1 - N * errors / (n * M) - float(r["microaccuracy_difference_full_rubric"])) < 1e-12
    validation = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
                      unique_source_data_hashes=len(all_hashes), allocation_rows=len(rows),
                      independently_reconciled_prior_baseline_rows=len(checked),
                      compared_fields_per_baseline=6, raw_rechecked_whole_task_rows=len(task_rows),
                      independently_scored_hybrid_rows=hybrid_checked,
                      verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      prior_budget_csv_sha256=hashlib.sha256((policy_directory / "budget_curves.csv.gz").read_bytes()).hexdigest(),
                      checked_result_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.csv*")},
                      scope="Baseline CSV was regenerated by the policy replay. Queried-set arithmetic agrees. Whole-task records rechecked directly;27 real hybrid rows independently scored while reusing selection. Exhaustive estimator/CI tests are separate; not an independent second selection implementation.")
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=OUT)
    parser.add_argument("--policy-directory", type=Path, default=ROOT / "artifacts/reproduced/conjunction_policy_study")
    args = parser.parse_args()
    result = verify(args.directory, args.policy_directory)
    (args.directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
