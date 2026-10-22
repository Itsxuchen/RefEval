"""Staged, reference-sealed analysis for aligned new-P / saved-R WildIFEval.

This module never calls a model. Real reference access is an explicit separate
stage after immutable design and prediction receipts have passed validation.
The primary bootstrap fixes the fitted pilot, as protocol section 5 specifies.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from src import wildifeval_forecast as forecast
from src import wildifeval_policy as policy


PRODUCERS = ("deepseek-v3", "llama3.1-405b", "llama3.3-70b", "mistral-large", "qwen2.5-72b")
CODE_PATHS = ("src/wildifeval_external_run.py", "src/wildifeval_forecast.py",
              "src/wildifeval_policy.py", "src/finite_population_intervals.py",
              "src/wildifeval_flashlite_design.py")
GENERATION_ARTIFACTS = {"jobs", "units", "results", "config", "receipt", "calibration", "runner",
                        "job_manifest", "source_manifest", "completion", "failures", "accounting_summary", "fee"}
MIGRATION_ARTIFACTS = {"predecessor_runner", "predecessor_config", "predecessor_freeze",
                       "predecessor_results", "predecessor_failures", "predecessor_ledger_backup",
                       "migration_receipt", "migration_script", "compatibility_diagnostic"}
METADATA_FIELDS = {"unit_id", "task_id", "cluster_id", "producer", "k", "split",
                   "criterion_ids", "criterion_hashes", "job_ids", "prompt_sha256",
                   "output_hash", "p"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_hash(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("Duplicate JSON key; refusing ambiguous input")
        out[key] = value
    return out


def read_json(path: Path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line, object_pairs_hook=_unique) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_json(path: Path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _relative(root: Path, path: Path | str) -> str:
    root = Path(root).resolve()
    path = Path(path)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError as exc:
        raise ValueError("Frozen files must remain inside project root") from exc


def _path(root: Path, rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts or _relative(root, path) != str(path):
        raise ValueError("Receipt paths must be canonical project-relative paths")
    return Path(root) / path


def default_config() -> dict:
    """The unchanged original gates, split, primary endpoint and seed choices."""
    return {"schema_version": "wildifeval-two-role-analysis-v1", "diagnostic": False,
            "producers": list(PRODUCERS), "benchmark_tasks": 7523,
            "minimum_clusters": 500, "minimum_pilot_clusters": 100,
            "minimum_confirmation_clusters": 300, "minimum_retention": .7,
            "pilot_share": .2, "split_salt": "refeval-external-v1",
            "budget_shares": [.2, .5, .8], "minimum_pool_clusters": 20,
            "bootstrap_replicates": 999, "bootstrap_seed": 20261010,
            "priority_seeds": list(policy.SEEDS), "display_shares": list(policy.GRID),
            "policies": ["srs", "mix50", "serial"], "reference_role": "saved_deepseek-v3",
            "reference_interpretation": "machine reference, not adjudicated human truth",
            "primary_bootstrap": "fixed_pilot_confirmation_clusters",
            "reference_threshold": .5, "amendment_path": None}


def validate_config(config: Mapping) -> None:
    baseline = default_config()
    if set(config) != set(baseline):
        raise ValueError("Unexpected/missing analysis configuration fields")
    if not config.get("amendment_path"):
        raise ValueError("Two-role analysis requires a dated protocol amendment")
    mutable = {"amendment_path", "diagnostic"}
    if config["diagnostic"]:
        mutable |= {"producers", "benchmark_tasks", "minimum_clusters", "minimum_pilot_clusters",
                    "minimum_confirmation_clusters", "minimum_retention", "minimum_pool_clusters",
                    "bootstrap_replicates", "priority_seeds", "display_shares"}
    for key in set(baseline) - mutable:
        if config[key] != baseline[key]:
            raise ValueError("Configuration changes a prespecified analysis field: " + key)
    if not config["producers"] or len(set(config["producers"])) != len(config["producers"]):
        raise ValueError("Unique nonempty producer list required")
    if not isinstance(config["diagnostic"], bool):
        raise ValueError("Diagnostic flag must be boolean")
    if min(config[key] for key in ("benchmark_tasks", "minimum_clusters", "minimum_pilot_clusters",
                                    "minimum_confirmation_clusters", "minimum_pool_clusters",
                                    "bootstrap_replicates")) < 1:
        raise ValueError("Positive sample and bootstrap counts required")
    if not config["priority_seeds"] or len(set(config["priority_seeds"])) != len(config["priority_seeds"]):
        raise ValueError("Distinct priority seeds required")


def _validate_units(units: Iterable[Mapping], config: Mapping, *, require_p: bool) -> list[dict]:
    """Copy an explicit allowlist; reference fields are rejected without reading values."""
    result, ids, jobs, criteria = [], set(), set(), set()
    required = METADATA_FIELDS - ({"p"} if not require_p else set())
    for unit in units:
        if set(unit) - METADATA_FIELDS or required - set(unit):
            raise ValueError("Metadata contains unexpected/reference fields or lacks required fields")
        row = {key: unit[key] for key in required}
        if row["producer"] not in config["producers"]:
            raise ValueError("Producer is outside the frozen cohort")
        k = row["k"]
        if type(k) is not int or not 1 <= k <= 8:
            raise ValueError("WildIFEval requires 1..8 complete criteria")
        for key in ("criterion_ids", "criterion_hashes", "job_ids"):
            row[key] = list(row[key])
            if len(row[key]) != k or len(set(row[key])) != k:
                raise ValueError("Criterion metadata is incomplete or duplicated")
        if row["unit_id"] in ids or jobs.intersection(row["job_ids"]) or criteria.intersection(row["criterion_ids"]):
            raise ValueError("Duplicate immutable unit/job/criterion IDs")
        ids.add(row["unit_id"]); jobs.update(row["job_ids"]); criteria.update(row["criterion_ids"])
        if require_p:
            row["p"] = list(row["p"])
            if len(row["p"]) != k or any(type(x) is not int or x not in (0, 1) for x in row["p"]):
                raise ValueError("P must be a complete binary vector")
        result.append(row)
    if not result:
        raise ValueError("Empty metadata frame")
    return sorted(result, key=lambda row: row["unit_id"])


def _assign_split(units: list[dict], config: Mapping) -> None:
    clusters = sorted({row["cluster_id"] for row in units}, key=lambda cluster: (
        hashlib.sha256((config["split_salt"] + "|" + cluster).encode()).hexdigest(), cluster))
    pilot = set(clusters[:int(config["pilot_share"] * len(clusters))])
    for row in units:
        row["split"] = "pilot" if row["cluster_id"] in pilot else "confirmation"


def _frame_report(units: list[dict], config: Mapping) -> dict:
    producers = config["producers"]
    by_task = defaultdict(list)
    for row in units:
        by_task[row["task_id"]].append(row)
    for rows in by_task.values():
        if sorted(row["producer"] for row in rows) != sorted(producers):
            raise ValueError("All producer cells must share complete tasks")
        if len({row["cluster_id"] for row in rows}) != 1 or len({row["split"] for row in rows}) != 1:
            raise ValueError("Task descendants cross clusters or partitions")
        if len({(row["prompt_sha256"], tuple(row["criterion_hashes"])) for row in rows}) != 1:
            raise ValueError("Producer descendants do not share the same complete task rubric")
    clusters = {row["cluster_id"] for row in units}
    pilot = {row["cluster_id"] for row in units if row["split"] == "pilot"}
    confirm = {row["cluster_id"] for row in units if row["split"] == "confirmation"}
    if pilot & confirm or pilot | confirm != clusters:
        raise ValueError("Cluster leakage or invalid partition")
    expected = [dict(row) for row in units]
    _assign_split(expected, config)
    if any(a["split"] != b["split"] for a, b in zip(units, expected)):
        raise ValueError("Partition differs from prespecified hash split")
    counts = {}
    for producer in producers:
        rows = [row for row in units if row["producer"] == producer]
        counts[producer] = {"units": len(rows), "retention": len(rows) / config["benchmark_tasks"],
                            "criteria": sum(row["k"] for row in rows),
                            "pilot_units": sum(row["split"] == "pilot" for row in rows),
                            "pilot_criteria": sum(row["k"] for row in rows if row["split"] == "pilot"),
                            "confirmation_units": sum(row["split"] == "confirmation" for row in rows),
                            "confirmation_criteria": sum(row["k"] for row in rows if row["split"] == "confirmation")}
    gates = {"clusters": len(clusters) >= config["minimum_clusters"],
             "pilot_clusters": len(pilot) >= config["minimum_pilot_clusters"],
             "confirmation_clusters": len(confirm) >= config["minimum_confirmation_clusters"],
             "retention": all(row["retention"] >= config["minimum_retention"] for row in counts.values())}
    return {"units": len(units), "tasks": len(by_task), "clusters": len(clusters),
            "pilot_clusters": len(pilot), "confirmation_clusters": len(confirm),
            "producer_counts": counts, "gates": gates, "feasible": all(gates.values()),
            "ids_sha256": object_hash([{key: row[key] for key in ("unit_id", "task_id", "cluster_id", "split")}
                                      for row in units]),
            "reference_values_opened": False, "diagnostic": config["diagnostic"]}


def prepare_metadata(units: Iterable[Mapping], generated_labels: Mapping[str, int | None], config: Mapping) -> tuple[list[dict], dict]:
    """Apply new-P missingness at whole-task level, then split the final frame.

    The input builder has already checked saved-R availability without exposing
    reference values. Missing/invalid generated labels never become FAIL.
    """
    validate_config(config)
    rows = _validate_units(units, config, require_p=False)
    known_jobs = {job for row in rows for job in row["job_ids"]}
    if set(generated_labels) - known_jobs:
        raise ValueError("Generated labels contain an unknown job ID")
    invalid = {job for job in known_jobs if type(generated_labels.get(job)) is not int or generated_labels[job] not in (0, 1)}
    incomplete = {row["task_id"] for row in rows if any(job in invalid for job in row["job_ids"])}
    groups = defaultdict(set)
    for row in rows:
        groups[row["task_id"]].add(row["producer"])
    incomplete |= {task for task, found in groups.items() if found != set(config["producers"])}
    excluded = [{"unit_id": row["unit_id"], "task_id": row["task_id"], "producer": row["producer"],
                 "reason": "incomplete_new_p_or_cross_producer_task"} for row in rows if row["task_id"] in incomplete]
    kept = [{**row, "p": [generated_labels[job] for job in row["job_ids"]]} for row in rows if row["task_id"] not in incomplete]
    if not kept:
        return [], {"feasible": False, "reason": "no_complete_common_tasks", "excluded": excluded,
                    "invalid_or_missing_jobs": len(invalid), "reference_values_opened": False}
    _assign_split(kept, config)
    report = _frame_report(kept, config)
    report.update(excluded=excluded, invalid_or_missing_jobs=len(invalid),
                  source_units=len(rows), generated_label_records=len(generated_labels))
    return kept, report


def read_generated_labels(jobs_path: Path, results_path: Path, *, expected_model: str,
                          expected_system_fingerprint: str | None = None) -> dict[str, int]:
    """Validate runner exports against immutable requests, without any R access.

    Failed requests are recorded separately by the runner and are absent here.
    Their jobs remain missing during whole-task eligibility construction.
    """
    from src.wildifeval_scoring import PARSER_VERSION, EVIDENCE_SERIALIZATION_VERSION, score_from_tokens
    jobs = {}
    with Path(jobs_path).open() as handle:
        for line in handle:
            row = json.loads(line, object_pairs_hook=_unique)
            if row["job_id"] in jobs:
                raise ValueError("Duplicate input job ID")
            if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
                raise ValueError("Job prompt hash mismatch")
            jobs[row["job_id"]] = {key: row[key] for key in ("unit_id", "criterion_id", "prompt_sha256")}
    labels = {}
    with Path(results_path).open() as handle:
        for line in handle:
            row = json.loads(line, object_pairs_hook=_unique)
            jid = row["job_id"]
            if jid not in jobs or jid in labels:
                raise ValueError("Unknown/duplicate successful job ID")
            if row.get("status") != "success" or row.get("model") != expected_model:
                raise ValueError("Generated result has wrong model or non-success status")
            if (row.get("parser_version") != PARSER_VERSION or row.get("finish_reason") != "stop"
                    or row.get("service_tier") not in (None, "default")):
                raise ValueError("Generated result differs from frozen parser/completion/service settings")
            if any(row.get(key) != value for key, value in jobs[jid].items()):
                raise ValueError("Generated result association hash/ID mismatch")
            if type(row.get("label")) is not int or row["label"] not in (0, 1):
                raise ValueError("Generated result is not a valid binary prediction")
            probability = row.get("prob_yes")
            if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Generated result has invalid Yes probability")
            if row["label"] != int(probability >= .5):
                raise ValueError("Generated prediction disagrees with the frozen threshold")
            recomputed = score_from_tokens(row.get("top_logprobs"))
            if recomputed["prob_yes"] != probability:
                raise ValueError("Generated probability does not reproduce saved token evidence")
            version = row.get("evidence_serialization_version")
            if version is not None:
                invalid, position = row.get("invalid_prefix_positions"), recomputed["score_position"]
                if (version != EVIDENCE_SERIALIZATION_VERSION or len(row["top_logprobs"]) != position + 1
                        or not isinstance(invalid, list) or any(type(i) is not int for i in invalid)
                        or invalid != sorted(set(invalid))
                        or any(i < 0 or i >= position or row["top_logprobs"][i] != {"top_logprobs": []} for i in invalid)):
                    raise ValueError("Generated evidence serialization version or prefix is invalid")
            if expected_system_fingerprint is not None and row.get("system_fingerprint") != expected_system_fingerprint:
                raise ValueError("Generated result fingerprint differs from the frozen model")
            labels[jid] = row["label"]
    return labels


def _generation_roles(freeze: Mapping) -> set[str]:
    from src import wildifeval_flashlite_design as flashlite
    if flashlite.is_gemini_receipt(freeze):     # protocol section 14: Gemini batch archive labelled by A1
        return set(flashlite.ROLES)
    return GENERATION_ARTIFACTS | (MIGRATION_ARTIFACTS if "previous_generation_freeze_sha256" in freeze else set())


def _jsonl_records(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line, object_pairs_hook=_unique)


def _fixed_scoring_logic(path: Path) -> str:
    """Independently compare the frozen scoring, request and pricing expressions."""
    functions = {"score_from_tokens", "request_body", "request_parameters", "usage_cost", "reservation_nano"}
    constants = {"PARSER_VERSION", "RATES", "MAX_INPUT_TOKENS", "SHORT_CONTEXT_TOKENS"}
    selected = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    selected[target.id] = ast.dump(node.value, include_attributes=False)
    if set(selected) != functions | constants:
        raise ValueError("Incomplete predecessor scoring source")
    from src.wildifeval_scoring import digest
    return digest(selected)


def _verify_serialization_migration(paths: Mapping[str, Path], generation_config: Mapping,
                                    freeze: Mapping, *, diagnostic: bool) -> dict:
    """Preserve legacy rows and bind the optional prospective format repair.

    The backup is hash-checked; analysis never opens the execution database.
    """
    from src import wildifeval_scoring as scoring
    version = scoring.EVIDENCE_SERIALIZATION_VERSION
    if "previous_generation_freeze_sha256" not in freeze:
        counts = Counter(row.get("evidence_serialization_version") for row in _jsonl_records(paths["results"]))
        if set(counts) - {version}:
            raise ValueError("Unversioned result requires an explicit predecessor migration archive")
        return {"legacy_success_records": 0, "selected_prefix_v2_success_records": counts[version],
                "preserved_terminal_failures": 0, "migration": False}
    receipt, predecessor = read_json(paths["migration_receipt"]), read_json(paths["predecessor_freeze"])
    old_config = read_json(paths["predecessor_config"])
    hashes = {"old_config_sha256": scoring.digest(old_config), "new_config_sha256": scoring.digest(generation_config)}
    for key, role in (("old_config_file_sha256", "predecessor_config"), ("new_config_file_sha256", "config"),
                      ("old_runner_sha256", "predecessor_runner"), ("new_runner_sha256", "runner"),
                      ("old_generation_freeze_sha256", "predecessor_freeze"), ("new_generation_freeze_sha256", "receipt"),
                      ("calibration_sha256", "calibration"), ("diagnostic_sha256", "compatibility_diagnostic"),
                      ("old_results_sha256", "predecessor_results"), ("old_failures_sha256", "predecessor_failures"),
                      ("migration_script_sha256", "migration_script"), ("jobs_sha256", "jobs")):
        hashes[key] = sha256(paths[role])
    if (receipt.get("status") != "completed" or receipt.get("kind") != "selected-prefix-evidence-v2"
            or any(receipt.get(key) != value for key, value in hashes.items())
            or receipt.get("database_backup_sha256") != sha256(paths["predecessor_ledger_backup"])
            or receipt.get("ledger_id") != freeze["ledger_id"]
            or receipt.get("state_directory_sha256") != freeze.get("state_directory_sha256")
            or receipt.get("evidence_serialization_version") != version
            or receipt.get("migration_id") != scoring.digest({**hashes, "ledger_id": freeze["ledger_id"], "kind": receipt.get("kind")})):
        raise ValueError("Migration archive does not bind the predecessor and successor evidence")
    if sha256(paths["migration_script"]) != sha256(Path(__file__).with_name("wildifeval_scoring_migration.py")):
        raise ValueError("Migration verifier source differs from the bound implementation")
    if (freeze.get("previous_generation_freeze_sha256") != hashes["old_generation_freeze_sha256"]
            or freeze.get("migration_script_sha256") != hashes["migration_script_sha256"]
            or freeze.get("compatibility_diagnostic_sha256") != hashes["diagnostic_sha256"]
            or freeze.get("evidence_serialization_version") != version
            or predecessor.get("status") != "frozen"
            or predecessor.get("config_sha256") != hashes["old_config_sha256"]
            or predecessor.get("runner_sha256") != hashes["old_runner_sha256"]
            or predecessor.get("jobs_sha256") != hashes["jobs_sha256"]
            or predecessor.get("calibration_sha256") != hashes["calibration_sha256"]
            or any(predecessor.get(key) != freeze.get(key) for key in
                   ("ledger_id", "state_directory_sha256", "canary_job_ids_sha256"))):
        raise ValueError("Migration predecessor freeze chain mismatch")
    scoring.validate_config(old_config)
    if ({key: value for key, value in old_config.items() if key != "budget_usd"}
            != {key: value for key, value in generation_config.items() if key != "budget_usd"}
            or scoring.usd_to_nano(old_config["budget_usd"]) != 100_000_000_000
            or scoring.usd_to_nano(generation_config["budget_usd"]) != 99_500_000_000
            or receipt.get("old_budget_nano") != 100_000_000_000
            or receipt.get("new_budget_nano") != 99_500_000_000
            or receipt.get("diagnostic_budget_reserved_nano") != 500_000_000):
        raise ValueError("Migration changed configuration beyond its diagnostic budget reserve")
    logic = _fixed_scoring_logic(paths["runner"])
    if logic != _fixed_scoring_logic(paths["predecessor_runner"]) or receipt.get("fixed_logic_ast_sha256") != logic:
        raise ValueError("Migration changed score, request or pricing logic")
    if any(receipt.get(key) is not True for key in
           ("request_parameters_unchanged", "score_algorithm_unchanged", "existing_results_unchanged", "existing_missing_not_retried")):
        raise ValueError("Migration lacks immutable-record guarantees")
    if (any(type(receipt.get(key)) is not int or receipt[key] < 0 for key in
            ("charged_nano_unchanged", "prior_nano_unchanged"))
            or receipt["prior_nano_unchanged"] > receipt["charged_nano_unchanged"]
            or receipt["charged_nano_unchanged"] > receipt["new_budget_nano"]
            or not isinstance(receipt.get("immutable_records_sha256"), str)
            or len(receipt["immutable_records_sha256"]) != 64):
        raise ValueError("Invalid preserved accounting in migration archive")
    probe = read_json(paths["compatibility_diagnostic"])
    if (probe.get("source") != "unrelated_synthetic_logprob_compatibility"
            or not 0 <= scoring.usd_to_nano(probe.get("budget_usd", -1)) <= 500_000_000
            or type(probe.get("charged_or_reserved_nano")) is not int
            or not 0 <= probe["charged_or_reserved_nano"] <= 500_000_000
            or receipt.get("diagnostic_charged_or_reserved_nano") != probe["charged_or_reserved_nano"]
            or not isinstance(probe.get("cases"), list) or len(probe["cases"]) > 32):
        raise ValueError("Migration diagnostic was not an unrelated bounded probe")
    old_success, old_terminal, old_ids = {}, {}, set()
    for role in ("predecessor_results", "predecessor_failures"):
        for row in _jsonl_records(paths[role]):
            jid = row.get("job_id")
            if not isinstance(jid, str) or jid in old_ids:
                raise ValueError("Duplicate or invalid predecessor job")
            old_ids.add(jid)
            if role == "predecessor_results":
                if row.get("status") != "success" or "evidence_serialization_version" in row:
                    raise ValueError("Predecessor success is not an unchanged v1 record")
                old_success[jid] = canonical_bytes(row)
            elif row.get("status") in ("terminal_missing", "exhausted", "failed"):
                old_terminal[jid] = canonical_bytes(row)
            elif row.get("status") != "pending" or row.get("attempt_count") != 0:
                raise ValueError("Predecessor has unfinished attempts outside the completed canary")
    if old_ids != {row["job_id"] for row in _jsonl_records(paths["jobs"])}:
        raise ValueError("Predecessor exports do not account for the complete planned frame")
    for prefix, rows in (("old_success", old_success), ("old_terminal_failure", old_terminal)):
        ids = sorted(rows)
        if (receipt.get(prefix + "_ids") != ids or receipt.get(prefix + "_ids_sha256") != scoring.digest(ids)
                or receipt.get(prefix + "_jobs") != len(ids)):
            raise ValueError("Migration preserved-job identity mismatch")
    if not diagnostic:
        groups = defaultdict(list)
        for row in _jsonl_records(paths["jobs"]):
            groups[row["producer"]].append(row["job_id"])
        canary = {jid for ids in groups.values() for jid in sorted(ids, key=lambda jid: scoring.text_hash("fee-calibration-v1|"+jid))[:32]}
        if set(old_success) | set(old_terminal) != canary or len(old_success) != 152 or len(old_terminal) != 8:
            raise ValueError("Migration predecessor differs from the documented 152/8 canary")
    retained, later = set(), 0
    for row in _jsonl_records(paths["results"]):
        jid = row["job_id"]
        if jid in old_success:
            if canonical_bytes(row) != old_success[jid]:
                raise ValueError("A predecessor success was rewritten or relabelled")
            retained.add(jid)
        elif jid in old_terminal:
            raise ValueError("A terminal predecessor failure was retried into a success")
        elif row.get("evidence_serialization_version") != version:
            raise ValueError("A later success lacks its v2 serialization identity")
        else:
            later += 1
    retained_failures = set()
    for row in _jsonl_records(paths["failures"]):
        jid = row["job_id"]
        if jid in old_terminal:
            if canonical_bytes(row) != old_terminal[jid]:
                raise ValueError("A terminal predecessor failure changed or was retried")
            retained_failures.add(jid)
    if retained != set(old_success) or retained_failures != set(old_terminal):
        raise ValueError("Migration lost predecessor successes or terminal failures")
    return {"migration": True, "migration_id": receipt["migration_id"],
            "legacy_success_records": len(retained), "selected_prefix_v2_success_records": later,
            "preserved_terminal_failures": len(retained_failures)}


def _verify_generation(root: Path, artifacts: Mapping[str, str], config: Mapping,
                       metadata: list[dict], reference_files: Mapping[str, str]) -> dict:
    if "receipt" not in artifacts or set(artifacts) != _generation_roles(read_json(_path(root, artifacts["receipt"]))):
        raise ValueError("Complete new-P generation evidence bundle required, including any migration predecessor")
    from src import wildifeval_flashlite_design as flashlite
    if flashlite.is_gemini_receipt(read_json(_path(root, artifacts["receipt"]))):
        return flashlite.verify_generation(root, artifacts, config, metadata, reference_files)
    paths = {key: _path(root, value) for key, value in artifacts.items()}
    manifest = read_json(paths["job_manifest"])
    for role, name in (("jobs", "jobs.jsonl"), ("units", "units.jsonl")):
        if sha256(paths[role]) != manifest["files"][name]["sha256"]:
            raise ValueError("Generated job/unit input differs from builder manifest")
    if sha256(paths["source_manifest"]) != manifest["source_inputs"]["source_manifest_sha256"]:
        raise ValueError("Source manifest differs from the generation source binding")
    source = read_json(paths["source_manifest"])
    source_files = {row["path"]: row["sha256"] for row in source["files"]}
    if len(source_files) != len(source["files"]):
        raise ValueError("Duplicate paths in source manifest")
    for producer, rel in reference_files.items():
        expected = "predictions/llm_aaj_scores/llm-aaj-deepseek-v3." + producer + "-0shot-wild-if-eval.json"
        if source_files.get(expected) != sha256(_path(root, rel)):
            raise ValueError("Saved reference does not match the pinned source manifest")
    generation_config = read_json(paths["config"])
    from src import wildifeval_scoring as scoring
    if sha256(paths["runner"]) != sha256(Path(scoring.__file__)):
        raise ValueError("Generation evidence uses a different scoring implementation")
    # Validate archived inference provenance without requiring the original
    # machine's absolute runtime/SQLite path. Analysis never resumes API work.
    scoring.validate_config(generation_config)
    calibration, generation_freeze = read_json(paths["calibration"]), read_json(paths["receipt"])
    if (generation_config.get("state") != "ready" or calibration.get("status") != "passed"
            or calibration.get("source") != "unrelated_synthetic_probe"
            or calibration.get("model_requested") != generation_config["model"]
            or calibration.get("response_model") != generation_config["expected_response_model"]
            or calibration.get("request_parameters") != scoring.request_parameters(generation_config)
            or calibration.get("usage_bound_ok") is not True or calibration.get("score_parse_ok") is not True):
        raise ValueError("invalid_calibration_or_generation_configuration")
    fingerprint = generation_config.get("expected_system_fingerprint")
    if fingerprint is not None and calibration.get("system_fingerprint") != fingerprint:
        raise ValueError("Calibration fingerprint differs from generation configuration")
    if (generation_freeze.get("status") != "frozen"
            or generation_freeze.get("config_sha256") != scoring.digest(generation_config)
            or generation_freeze.get("jobs_sha256") != generation_config["jobs_sha256"]
            or generation_freeze.get("calibration_sha256") != sha256(paths["calibration"])
            or generation_freeze.get("runner_sha256") != sha256(paths["runner"])):
        raise ValueError("Generation freeze does not bind calibration, config and runner")
    if not isinstance(generation_freeze.get("ledger_id"), str) or not generation_freeze["ledger_id"]:
        raise ValueError("Generation freeze lacks its execution ledger identity")
    if generation_config["jobs_sha256"] != sha256(paths["jobs"]):
        raise ValueError("Generation configuration uses another job set")
    completion = read_json(paths["completion"])
    planned = manifest.get("counts", {}).get("jobs")
    if (completion.get("status") != "complete" or completion.get("pending_jobs") != 0
            or completion.get("inflight_jobs") != 0 or completion.get("halted_reason") is not None
            or type(planned) is not int or planned < 1
            or completion.get("planned_jobs") != planned or completion.get("terminal_jobs") != planned
            or completion.get("success_jobs", -1) + completion.get("failure_jobs", -1) != planned):
        raise ValueError("Generation is incomplete; pending or halted jobs cannot become analysis exclusions")
    if not config["diagnostic"] and planned != 134485:
        raise ValueError("Production generation did not cover all 134485 planned criterion jobs")
    if completion.get("ledger_id") != generation_freeze["ledger_id"]:
        raise ValueError("Completion receipt belongs to a different execution ledger")
    if completion.get("config_sha256") != scoring.digest(generation_config):
        raise ValueError("Completion receipt belongs to a different generation configuration")
    for key, role in (("jobs_sha256", "jobs"), ("runner_sha256", "runner"),
                      ("generation_freeze_sha256", "receipt"), ("calibration_sha256", "calibration"),
                      ("results_sha256", "results"), ("failures_sha256", "failures"),
                      ("accounting_summary_sha256", "accounting_summary")):
        if completion.get(key) != sha256(paths[role]):
            raise ValueError("Completion receipt does not bind its generation artifact: " + role)
    accounting = read_json(paths["accounting_summary"])
    counts = accounting.get("jobs", {})
    if (set(counts) - {"success", "terminal_missing", "exhausted", "failed", "pending", "inflight"}
            or any(type(value) is not int or value < 0 for value in counts.values())
            or counts.get("pending", 0) or counts.get("inflight", 0) or sum(counts.values()) != planned
            or counts.get("success", 0) != completion["success_jobs"]
            or accounting.get("halted_reason") is not None
            or accounting.get("ledger_id") != generation_freeze["ledger_id"]
            or accounting.get("config_sha256") != scoring.digest(generation_config)):
        raise ValueError("Accounting summary does not establish completed generation")
    fee = read_json(paths["fee"])
    if (fee.get("status") != "passed" or fee.get("ledger_id") != generation_freeze["ledger_id"]
            or fee.get("config_sha256") != scoring.digest(generation_config)
            or fee.get("canary_job_ids_sha256") != generation_freeze.get("canary_job_ids_sha256")
            or fee.get("complete") is not True or fee.get("estimable") is not True
            or type(fee.get("projected_total_usd")) not in (int, float)
            or not 0 <= fee["projected_total_usd"] <= generation_config["budget_usd"]):
        raise ValueError("Missing/mismatched fee-stage approval in generation archive")
    model = generation_config.get("expected_response_model")
    if not isinstance(model, str) or not model:
        raise ValueError("Generation config lacks the fixed new-P model")
    labels = read_generated_labels(paths["jobs"], paths["results"], expected_model=model,
                                   expected_system_fingerprint=generation_config.get("expected_system_fingerprint"))
    source_units = read_jsonl(paths["units"])
    expected_jobs = {jid for row in source_units for jid in row["job_ids"]}
    failures = read_jsonl(paths["failures"])
    failure_ids = set()
    for row in failures:
        if (row.get("job_id") in failure_ids or row.get("job_id") in labels
                or row.get("status") not in ("terminal_missing", "exhausted", "failed")
                or "label" in row or "prob_yes" in row):
            raise ValueError("Failure export contains a repeated, nonterminal or labelled job")
        failure_ids.add(row["job_id"])
    if (len(expected_jobs) != planned or set(labels) | failure_ids != expected_jobs
            or len(labels) != completion["success_jobs"] or len(failure_ids) != completion["failure_jobs"]):
        raise ValueError("Completed exports do not account for every planned job exactly once")
    serialization = _verify_serialization_migration(paths, generation_config, generation_freeze,
                                                   diagnostic=config["diagnostic"])
    rebuilt, report = prepare_metadata(source_units, labels, config)
    if canonical_bytes(rebuilt) != canonical_bytes(metadata):
        raise ValueError("Final P metadata does not reproduce immutable generation results")
    if not report["feasible"]:
        raise ValueError("Regenerated P frame fails feasibility")
    return {"fixed_model": model, "successful_prediction_records": len(labels),
            "execution_ledger_id": generation_freeze["ledger_id"], "serialization": serialization,
            "final_metadata_sha256": object_hash(metadata), "reference_values_opened": False}


def create_design_freeze(root: Path, *, metadata_path: Path, preflight_path: Path, config_path: Path,
                         reference_files: Mapping[str, str], required_files: Iterable[Path | str],
                         output_path: Path, generation_artifacts: Mapping[str, str] | None = None) -> dict:
    """Freeze inputs/code before the first reference outcome can be loaded."""
    root = Path(root)
    config = read_json(config_path); validate_config(config)
    units = _validate_units(read_jsonl(metadata_path), config, require_p=True)
    report = _frame_report(units, config)
    if not report["feasible"]:
        raise ValueError("Masked feasibility gate failed; reference remains sealed")
    saved = read_json(preflight_path)
    if any(saved.get(key) != value for key, value in report.items()):
        raise ValueError("Preflight receipt does not reproduce final metadata counts")
    if set(reference_files) != set(config["producers"]):
        raise ValueError("One saved reference file required for every producer")
    generation = {_key: _relative(root, value) for _key, value in (generation_artifacts or {}).items()}
    verification = None
    if not config["diagnostic"] or generation:
        verification = _verify_generation(root, generation, config, units,
                                           {key: _relative(root, value) for key, value in reference_files.items()})
    paths = set(map(str, required_files)) | set(CODE_PATHS) | set(reference_files.values())
    paths |= set(generation.values())
    paths |= {str(metadata_path), str(preflight_path), str(config_path), config["amendment_path"]}
    files = {_relative(root, path): sha256(root / path if not Path(path).is_absolute() else Path(path)) for path in paths}
    receipt = {"schema_version": "wildifeval-two-role-freeze-v1", "stage": "design_freeze",
               "created_utc": datetime.now(timezone.utc).isoformat(), "diagnostic": config["diagnostic"],
               "frozen_files": dict(sorted(files.items())), "metadata_path": _relative(root, metadata_path),
               "preflight_path": _relative(root, preflight_path), "config_path": _relative(root, config_path),
               "reference_files": {key: _relative(root, value) for key, value in reference_files.items()},
               "generation_artifacts": generation, "generation_verification": verification,
               "frame": report, "reference_values_opened": False}
    _write_json(output_path, receipt)
    return receipt


def validate_design(root: Path, receipt_path: Path) -> tuple[dict, dict, list[dict]]:
    receipt = read_json(receipt_path)
    if receipt.get("stage") != "design_freeze" or receipt.get("reference_values_opened") is not False:
        raise ValueError("Invalid design-freeze stage")
    if not receipt.get("frozen_files"):
        raise ValueError("Empty freeze manifest")
    required = set(CODE_PATHS) | {receipt[key] for key in ("metadata_path", "config_path", "preflight_path")}
    required |= set(receipt["reference_files"].values())
    if not required <= set(receipt["frozen_files"]):
        raise ValueError("Design omits required code or data hashes")
    for rel, expected in receipt["frozen_files"].items():
        if sha256(_path(root, rel)) != expected:
            raise ValueError("Frozen input/code changed: " + rel)
    for rel in CODE_PATHS:
        if sha256(Path(__file__).parent / Path(rel).name) != receipt["frozen_files"][rel]:
            raise ValueError("Executing analysis code differs from frozen source: " + rel)
    config = read_json(_path(root, receipt["config_path"])); validate_config(config)
    if config["amendment_path"] not in receipt["frozen_files"]:
        raise ValueError("Protocol amendment was not frozen")
    if not config["diagnostic"]:
        generation = receipt.get("generation_artifacts", {})
        verification = receipt.get("generation_verification")
        if ("receipt" not in generation or set(generation) != _generation_roles(read_json(_path(root, generation["receipt"])))
                or not verification or not set(generation.values()) <= set(receipt["frozen_files"])):
            raise ValueError("Production freeze lacks verified new-P generation evidence")
    if set(receipt["reference_files"]) != set(config["producers"]):
        raise ValueError("Reference role map differs from the frozen producer cohort")
    units = _validate_units(read_jsonl(_path(root, receipt["metadata_path"])), config, require_p=True)
    report = _frame_report(units, config)
    if report != receipt["frame"] or not report["feasible"]:
        raise ValueError("Frozen frame failed revalidation")
    if receipt.get("generation_verification") and receipt["generation_verification"]["final_metadata_sha256"] != object_hash(units):
        raise ValueError("Frozen generation verification does not bind the final P metadata")
    return receipt, config, units


class _SealedNumber:
    """JSON numeric token; only selected-split tokens may be thresholded."""
    __slots__ = ("_literal",)

    def __init__(self, literal):
        self._literal = literal

    def bit(self):
        value = Decimal(self._literal)
        if not value.is_finite() or not 0 <= value <= 1:
            raise ValueError("Selected reference score is invalid")
        return int(value >= Decimal("0.5"))


def _read_references(root: Path, receipt: Mapping, units: list[dict], split: str) -> list[dict]:
    """Only called after gate checks; never thresholds the other partition."""
    selected = [{**row} for row in units if row["split"] == split]
    for producer, rel in receipt["reference_files"].items():
        targets = {row["prompt_sha256"]: row for row in selected if row["producer"] == producer}
        if len(targets) != sum(row["producer"] == producer for row in selected):
            raise ValueError("Ambiguous prompt keys in frozen producer frame")
        raw = json.loads(_path(root, rel).read_text(), object_pairs_hook=_unique,
                         parse_int=_SealedNumber, parse_float=_SealedNumber,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Invalid reference JSON")))
        for prompt, record in raw.items():
            key = hashlib.sha256(prompt.encode()).hexdigest()
            unit = targets.get(key)
            if unit is None:
                continue
            if not isinstance(record, dict) or not isinstance(record.get("scores"), dict):
                raise ValueError("Selected reference record has an invalid schema")
            scores = {hashlib.sha256(criterion.encode()).hexdigest(): value for criterion, value in record["scores"].items()}
            if set(scores) != set(unit["criterion_hashes"]):
                raise ValueError("Selected reference criteria do not exactly match frozen task")
            values = [scores[key] for key in unit["criterion_hashes"]]
            if any(not isinstance(value, _SealedNumber) for value in values):
                raise ValueError("Selected reference contains missing/non-numeric values")
            unit["r"] = [value.bit() for value in values]
        if any("r" not in unit for unit in targets.values()):
            raise ValueError("Selected reference task is absent")
    return selected


def _validate_prediction(root: Path, design_path: Path, receipt_path: Path, units: list[dict], config: Mapping) -> dict:
    receipt = read_json(receipt_path)
    if receipt.get("stage") != "prediction_freeze" or receipt.get("design_receipt_sha256") != sha256(design_path):
        raise ValueError("Missing or mismatched prediction freeze")
    files = receipt.get("prediction_files", {})
    if not files or receipt.get("predictions_path") not in files:
        raise ValueError("Prediction receipt omits forecast payload")
    for rel, expected in files.items():
        if sha256(_path(root, rel)) != expected:
            raise ValueError("Frozen prediction payload changed")
    payload = read_json(_path(root, receipt["predictions_path"]))
    confirmation = [row for row in units if row["split"] == "confirmation"]
    recomputed = forecast.predict_frame(payload["fit"], confirmation, config["budget_shares"])
    if canonical_bytes(recomputed) != canonical_bytes(payload["primary_predictions"]):
        raise ValueError("Saved forecasts differ from frozen fit and confirmation metadata")
    if payload["design_receipt_sha256"] != sha256(design_path):
        raise ValueError("Forecast payload belongs to another design")
    return payload


def load_stage(root: Path, design_path: Path, *, stage: str, prediction_path: Path | None = None) -> list[dict]:
    """The only public reference loader; validation precedes any score parsing."""
    if stage not in ("pilot", "confirmation_metadata", "confirmation"):
        raise ValueError("Unknown analysis stage")
    receipt, config, units = validate_design(root, design_path)
    if stage == "confirmation_metadata":
        return [row for row in units if row["split"] == "confirmation"]
    if stage == "confirmation":
        if prediction_path is None:
            raise ValueError("Confirmation reference is sealed without frozen predictions")
        _validate_prediction(root, design_path, prediction_path, units, config)
    return _read_references(root, receipt, units, stage)


def run_pilot(root: Path, design_path: Path, output_dir: Path) -> dict:
    """Fit once, save every forecast, then issue the confirmation-access receipt."""
    receipt, config, metadata = validate_design(root, design_path)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "attempt.json", {"stage": "pilot_started", "design_receipt_sha256": sha256(design_path)})
    try:
        units = load_stage(root, design_path, stage="pilot")
        confirmation = [row for row in metadata if row["split"] == "confirmation"]
        fit = forecast.fit_forecaster(units, "r", config["minimum_pool_clusters"])
        predictions = forecast.predict_frame(fit, confirmation, config["budget_shares"])
        allocation = policy.allocation_forecasts(units, references=("r",), seeds=config["priority_seeds"],
                                                  minimum_clusters=config["minimum_pool_clusters"])
        payload = {"stage": "pilot_predictions", "design_receipt_sha256": sha256(design_path), "fit": fit,
                   "primary_predictions": predictions, "support_manifest": forecast.support_manifest(fit, confirmation),
                   "allocation_forecasts": allocation,
                   "pilot_reference_bits": sum(row["k"] for row in units),
                   "confirmation_reference_bits_revealed": 0,
                   "reference_interpretation": config["reference_interpretation"], "diagnostic": config["diagnostic"]}
        path = output_dir / "predictions.json"; _write_json(path, payload)
        result = {"stage": "prediction_freeze", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "design_receipt_sha256": sha256(design_path), "predictions_path": _relative(root, path),
                  "prediction_files": {_relative(root, path): sha256(path)}, "diagnostic": config["diagnostic"],
                  "n_primary_predictions": len(predictions), "n_primary_cells": len(predictions) // 4,
                  "confirmation_reference_bits_revealed": 0}
        _write_json(output_dir / "prediction_freeze.json", result)
        return result
    except Exception as exc:
        _write_json(output_dir / "failed.json", {"stage": "pilot_failed", "error_type": type(exc).__name__,
                                                  "message": str(exc), "confirmation_opened": False})
        raise


def _secondary_summary(rows: list[dict], predictions: list[dict]) -> list[dict]:
    result = []
    for pred in predictions:
        selected = [row for row in rows if row["producer"] == pred["producer"] and abs(row["budget_share"] - .2) < 1e-12]
        by_seed = defaultdict(dict)
        for row in selected:
            by_seed[row["seed"]][row["policy"]] = row
        differences = []
        for seed, policies in sorted(by_seed.items()):
            srs, mix = policies["srs"], policies["mix50"]
            if srs["actual_queries"] != mix["actual_queries"]:
                raise AssertionError("SRS and mixed costs differ")
            delta = {name: mix[name] - srs[name] for name in ("C", "CW", "D", "A", "eager_errors")}
            if delta["eager_errors"] != -delta["CW"] - delta["D"] + delta["A"]:
                raise AssertionError("Paired allocation decomposition failed")
            differences.append({"seed": seed, **{key + "_per100": 100 * value / mix["n_units"] for key, value in delta.items()}})
        if not differences:
            raise ValueError("Missing prespecified 20% policy endpoint")
        observed = float(np.mean([row["eager_errors_per100"] for row in differences]))
        result.append({**pred, "observed_repair_contrast_per100": observed,
                       "event_forecast_absolute_error": abs(observed - pred["repair_forecast_per100"]),
                       "certificate_forecast_absolute_error": abs(observed - pred["certificate_only_forecast_per100"]),
                       "paired_seed_rows": differences,
                       "uncertainty_scope": "query-priority randomization; not a task-composition confidence interval"})
    return result


def _write_jsonl_gz(path: Path, rows: Iterable[dict]) -> None:
    # Zero timestamp makes compressed payloads reproducible across replays.
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
            for row in rows:
                handle.write(canonical_bytes(row) + b"\n")


def run_confirmation(root: Path, design_path: Path, prediction_path: Path, output_dir: Path) -> dict:
    """Score every frozen primary prediction and execute all specified controls."""
    receipt, config, metadata = validate_design(root, design_path)
    frozen = _validate_prediction(root, design_path, prediction_path, metadata, config)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "attempt.json", {"stage": "confirmation_started", "design_receipt_sha256": sha256(design_path),
                                               "prediction_receipt_sha256": sha256(prediction_path)})
    opened = False
    try:
        opened = True
        units = load_stage(root, design_path, stage="confirmation", prediction_path=prediction_path)
        primary = forecast.compare_forecasts(frozen["fit"], units, "r", config["budget_shares"],
                                              config["bootstrap_replicates"], config["bootstrap_seed"])
        pilot_metadata = [row for row in metadata if row["split"] == "pilot"]
        rows = []
        for producer in config["producers"]:
            cell = [row for row in units if row["producer"] == producer]
            rows.extend(policy.replay_cell(cell, "r", seeds=config["priority_seeds"], shares=config["display_shares"],
                                          policies=config["policies"], interval_shares=(.2,))["rows"])
        secondary = _secondary_summary(rows, frozen["allocation_forecasts"]["predictions"])
        controls = policy.no_pilot_controls(units, pilot_metadata, reference_key="r", shares=config["budget_shares"],
                                             seeds=config["priority_seeds"], policies=config["policies"])
        ledger = []
        for producer, counts in receipt["frame"]["producer_counts"].items():
            for share in config["budget_shares"]:
                queried = round(share * counts["confirmation_criteria"])
                ledger.append({"producer": producer, "confirmation_share": share,
                               "pilot_reference_bits": counts["pilot_criteria"], "confirmation_query_cap": queried,
                               "charged_reference_bits": counts["pilot_criteria"] + queried,
                               "validation_only_confirmation_bits": counts["confirmation_criteria"] - queried,
                               "no_pilot_cap": min(counts["confirmation_criteria"], counts["pilot_criteria"] + queried)})
        _write_json(output_dir / "primary.json", primary)
        _write_json(output_dir / "secondary.json", secondary)
        _write_json(output_dir / "reference_budget.json", {"per_protocol_costs": ledger,
                   "unique_pilot_bits_opened": frozen["pilot_reference_bits"],
                   "unique_confirmation_bits_opened_for_validation": sum(row["k"] for row in units),
                   "api_costs": "Tracked separately by the new-P generation ledger; reference replay is not API expenditure",
                   "seed_and_budget_runs_reuse_fixed_labels": True})
        _write_jsonl_gz(output_dir / "policy_rows.jsonl.gz", rows)
        _write_jsonl_gz(output_dir / "no_pilot_controls.jsonl.gz", controls)
        output_files = {_relative(root, path): sha256(path) for path in sorted(output_dir.iterdir())}
        result = {"stage": "confirmation_complete", "design_receipt_sha256": sha256(design_path),
                  "prediction_receipt_sha256": sha256(prediction_path), "result_files": output_files,
                  "primary_decision": primary["summary"]["decision"], "diagnostic": config["diagnostic"],
                  "reference_interpretation": config["reference_interpretation"],
                  "auxiliary_disagreement_and_reference_replacement": "omitted in the two-role amendment",
                  "primary_bootstrap": config["primary_bootstrap"], "secondary_intervals": "not task-composition CIs"}
        _write_json(output_dir / "result_receipt.json", result)
        return result
    except Exception as exc:
        _write_json(output_dir / "failed.json", {"stage": "confirmation_failed", "error_type": type(exc).__name__,
                                                  "message": str(exc), "confirmation_opened": opened})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "pilot", "confirmation"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "validate":
        _, _, units = validate_design(args.root, args.design)
        print(json.dumps({"validated": True, "units": len(units), "reference_values_opened": False}))
    elif args.stage == "pilot":
        if args.output is None:
            parser.error("--output is required")
        print(json.dumps(run_pilot(args.root, args.design, args.output)))
    else:
        if args.output is None or args.predictions is None:
            parser.error("--output and --predictions are required")
        print(json.dumps(run_confirmation(args.root, args.design, args.predictions, args.output)))


if __name__ == "__main__":
    main()
