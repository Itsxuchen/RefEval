"""Synthetic-only stage, cohort, inference and corruption checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from src import wildifeval_external_run as run
from src import wildifeval_forecast as forecast


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value))


def synthetic_units(n=20, producers=("a", "b")):
    units, labels, raw = [], {}, {p: {} for p in producers}
    for i in range(n):
        for producer in producers:
            uid = producer + "-" + str(i)
            criteria = ["criterion A", "criterion B"]
            jobs = [uid + ":job:" + str(j) for j in range(2)]
            units.append(dict(unit_id=uid, task_id=str(i), cluster_id="cluster-" + str(i),
                producer=producer, k=2, split="unassigned", criterion_ids=[uid + ":" + str(j) for j in range(2)],
                criterion_hashes=[digest(c) for c in criteria], job_ids=jobs,
                prompt_sha256=digest("prompt " + str(i)), output_hash=digest(uid + " output")))
            labels.update(zip(jobs, (0, 1)))
            reference = ((1, 1), (1, 0), (0, 1), (0, 0))[i % 4]
            raw[producer]["prompt " + str(i)] = {"scores": dict(zip(criteria, reference))}
    return units, labels, raw


def config():
    result = run.default_config()
    result.update(diagnostic=True, producers=["a", "b"], benchmark_tasks=20,
                  minimum_clusters=5, minimum_pilot_clusters=1, minimum_confirmation_clusters=4,
                  minimum_retention=.7, minimum_pool_clusters=1, bootstrap_replicates=7,
                  priority_seeds=[20261011, 20261012], display_shares=[0, .2, .5, .8, 1],
                  amendment_path="amendment.md")
    return result


@pytest.fixture
def frozen(tmp_path):
    cfg = config()
    units, labels, references = synthetic_units()
    metadata, report = run.prepare_metadata(units, labels, cfg)
    for rel in run.CODE_PATHS:
        target = tmp_path / rel
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(run.__file__).parent / Path(rel).name, target)
    (tmp_path / "amendment.md").write_text("Synthetic test protocol only")
    (tmp_path / "units.jsonl").write_text("\n".join(json.dumps(row) for row in metadata) + "\n")
    dump(tmp_path / "config.json", cfg)
    dump(tmp_path / "preflight.json", report)
    for producer, raw in references.items():
        dump(tmp_path / (producer + ".json"), raw)
    design = tmp_path / "design.json"
    run.create_design_freeze(tmp_path, metadata_path=tmp_path / "units.jsonl",
        preflight_path=tmp_path / "preflight.json", config_path=tmp_path / "config.json",
        reference_files={p: p + ".json" for p in cfg["producers"]}, required_files=[], output_path=design)
    return tmp_path, design, metadata, report


def test_missing_label_excludes_whole_task_all_producers_and_resplits():
    units, labels, _ = synthetic_units()
    del labels[units[0]["job_ids"][0]]
    kept, report = run.prepare_metadata(units, labels, config())
    assert len(kept) == 38
    assert len(report["excluded"]) == 2
    assert {row["task_id"] for row in report["excluded"]} == {"0"}
    assert report["pilot_clusters"] == 3
    assert all(row["p"] == [0, 1] for row in kept)


def test_non_binary_and_unknown_prediction_never_become_false():
    units, labels, _ = synthetic_units()
    labels[units[0]["job_ids"][0]] = True
    kept, report = run.prepare_metadata(units, labels, config())
    assert len(kept) == 38 and report["invalid_or_missing_jobs"] == 1
    labels["unknown"] = 0
    with pytest.raises(ValueError, match="unknown job"):
        run.prepare_metadata(units, labels, config())


def test_metadata_rejects_reference_without_accessing_value():
    class Poison(dict):
        def __getitem__(self, key):
            if key == "r":
                raise AssertionError("leaked reference")
            return super().__getitem__(key)
    units, labels, _ = synthetic_units()
    units[0] = Poison(units[0], r=object())
    with pytest.raises(ValueError, match="reference fields"):
        run.prepare_metadata(units, labels, config())


def test_production_constants_cannot_silently_change():
    cfg = run.default_config()
    cfg["amendment_path"] = "amendment.md"
    run.validate_config(cfg)
    cfg["bootstrap_replicates"] = 99
    with pytest.raises(ValueError, match="prespecified"):
        run.validate_config(cfg)


def test_failed_feasibility_does_not_freeze(frozen):
    root, _, _, _ = frozen
    cfg = run.read_json(root / "config.json")
    cfg["minimum_clusters"] = 100
    dump(root / "config.json", cfg)
    with pytest.raises(ValueError, match="gate failed"):
        run.create_design_freeze(root, metadata_path=root / "units.jsonl", preflight_path=root / "preflight.json",
            config_path=root / "config.json", reference_files={p: p + ".json" for p in cfg["producers"]},
            required_files=[], output_path=root / "never.json")
    assert not (root / "never.json").exists()


def test_confirmation_is_sealed_and_metadata_no_references(frozen, monkeypatch):
    root, design, _, _ = frozen
    def forbidden(*args, **kwargs):
        raise AssertionError("reference parser was invoked")
    monkeypatch.setattr(run, "_read_references", forbidden)
    with pytest.raises(ValueError, match="sealed"):
        run.load_stage(root, design, stage="confirmation")
    metadata = run.load_stage(root, design, stage="confirmation_metadata")
    assert all("r" not in row and "r2" not in row for row in metadata)


def test_only_pilot_numeric_tokens_are_thresholded(frozen, monkeypatch):
    root, design, _, report = frozen
    original = run._SealedNumber.bit
    calls = []
    def observe(self):
        calls.append(1)
        return original(self)
    monkeypatch.setattr(run._SealedNumber, "bit", observe)
    pilot = run.load_stage(root, design, stage="pilot")
    expected = sum(row["pilot_criteria"] for row in report["producer_counts"].values())
    assert len(calls) == expected == sum(row["k"] for row in pilot)
    assert all(row["split"] == "pilot" for row in pilot)


@pytest.mark.parametrize("filename", ["units.jsonl", "config.json", "a.json", "amendment.md", "src/wildifeval_forecast.py"])
def test_any_bound_input_change_blocks_unmask_before_parser(frozen, monkeypatch, filename):
    root, design, _, _ = frozen
    with (root / filename).open("a") as handle:
        handle.write(" ")
    monkeypatch.setattr(run, "_read_references", lambda *a: (_ for _ in ()).throw(AssertionError("reference opened")))
    with pytest.raises(ValueError, match="Frozen input/code changed"):
        run.load_stage(root, design, stage="pilot")


def test_complete_pipeline_uses_frozen_predictions_and_records_costs(frozen):
    root, design, metadata, report = frozen
    freeze = run.run_pilot(root, design, root / "pilot")
    assert freeze["n_primary_cells"] == 6 and freeze["n_primary_predictions"] == 24
    assert freeze["confirmation_reference_bits_revealed"] == 0
    payload = run.read_json(root / "pilot/predictions.json")
    expected = forecast.predict_frame(payload["fit"], [row for row in metadata if row["split"] == "confirmation"])
    assert expected == payload["primary_predictions"]
    result = run.run_confirmation(root, design, root / "pilot/prediction_freeze.json", root / "confirmation")
    assert result["diagnostic"] is True
    primary = run.read_json(root / "confirmation/primary.json")
    assert primary["summary"]["bootstrap_replicates"] == 7
    assert primary["summary"]["interval_scope"].endswith("fixed pilot fit")
    budget = run.read_json(root / "confirmation/reference_budget.json")
    assert budget["unique_pilot_bits_opened"] == 16
    assert budget["unique_confirmation_bits_opened_for_validation"] == 64
    for row in budget["per_protocol_costs"]:
        assert row["charged_reference_bits"] == row["pilot_reference_bits"] + row["confirmation_query_cap"]
    for row in run.read_json(root / "confirmation/secondary.json"):
        for seed in row["paired_seed_rows"]:
            assert seed["eager_errors_per100"] == pytest.approx(-seed["CW_per100"] - seed["D_per100"] + seed["A_per100"])
    with pytest.raises(FileExistsError):
        run.run_confirmation(root, design, root / "pilot/prediction_freeze.json", root / "confirmation")


def test_corrupted_frozen_prediction_blocks_confirmation(frozen, monkeypatch):
    root, design, _, _ = frozen
    run.run_pilot(root, design, root / "pilot")
    payload_path = root / "pilot/predictions.json"
    payload = run.read_json(payload_path)
    payload["primary_predictions"][0]["forecast_per100"] += 1
    dump(payload_path, payload)
    monkeypatch.setattr(run, "_read_references", lambda *a: (_ for _ in ()).throw(AssertionError("reference opened")))
    with pytest.raises(ValueError, match="payload changed"):
        run.load_stage(root, design, stage="confirmation", prediction_path=root / "pilot/prediction_freeze.json")


def test_modified_forecast_even_with_updated_payload_hash_blocks_confirmation(frozen, monkeypatch):
    root, design, _, _ = frozen
    run.run_pilot(root, design, root / "pilot")
    path = root / "pilot/predictions.json"
    payload = run.read_json(path)
    payload["primary_predictions"][0]["forecast_per100"] += 1
    dump(path, payload)
    freeze_path = root / "pilot/prediction_freeze.json"
    receipt = run.read_json(freeze_path)
    receipt["prediction_files"]["pilot/predictions.json"] = run.sha256(path)
    dump(freeze_path, receipt)
    monkeypatch.setattr(run, "_read_references", lambda *a: (_ for _ in ()).throw(AssertionError("reference opened")))
    with pytest.raises(ValueError, match="Saved forecasts differ"):
        run.load_stage(root, design, stage="confirmation", prediction_path=freeze_path)


def test_forecast_failure_preserves_attempt_and_never_freezes(frozen, monkeypatch):
    root, design, _, _ = frozen
    def fail(*args, **kwargs):
        raise ValueError("synthetic support failure")
    monkeypatch.setattr(forecast, "predict_frame", fail)
    with pytest.raises(ValueError, match="support failure"):
        run.run_pilot(root, design, root / "pilot")
    assert (root / "pilot/attempt.json").exists()
    assert run.read_json(root / "pilot/failed.json")["confirmation_opened"] is False
    assert not (root / "pilot/prediction_freeze.json").exists()


def test_exact_source_constraint_association_is_required(frozen):
    root, design, _, _ = frozen
    raw = run.read_json(root / "a.json")
    for record in raw.values():
        record["scores"]["criterion C"] = record["scores"].pop("criterion A")
    dump(root / "a.json", raw)
    receipt = run.read_json(design)
    receipt["frozen_files"]["a.json"] = run.sha256(root / "a.json")
    dump(design, receipt)
    with pytest.raises(ValueError, match="exactly match"):
        run.load_stage(root, design, stage="pilot")


def test_generated_exports_require_model_request_and_unique_id(tmp_path):
    from src.wildifeval_scoring import PARSER_VERSION
    job = dict(job_id="job", unit_id="unit", criterion_id="criterion", prompt="text", prompt_sha256=digest("text"))
    result = {key: value for key, value in job.items() if key != "prompt"}
    result.update(model="fixed-model", status="success", label=1, prob_yes=1.0,
                  parser_version=PARSER_VERSION, finish_reason="stop", service_tier="default",
                  top_logprobs=[{"top_logprobs": [{"token": "Yes", "logprob": 0.0}]}])
    jobs, results = tmp_path / "jobs.jsonl", tmp_path / "results.jsonl"
    jobs.write_text(json.dumps(job) + "\n")
    results.write_text(json.dumps(result) + "\n")
    assert run.read_generated_labels(jobs, results, expected_model="fixed-model") == {"job": 1}
    with pytest.raises(ValueError, match="wrong model"):
        run.read_generated_labels(jobs, results, expected_model="another-model")
    results.write_text((json.dumps(result) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        run.read_generated_labels(jobs, results, expected_model="fixed-model")
    result["prompt_sha256"] = "wrong"
    results.write_text(json.dumps(result) + "\n")
    with pytest.raises(ValueError, match="association"):
        run.read_generated_labels(jobs, results, expected_model="fixed-model")
    result["prompt_sha256"] = job["prompt_sha256"]
    result["label"] = 0
    results.write_text(json.dumps(result) + "\n")
    with pytest.raises(ValueError, match="threshold"):
        run.read_generated_labels(jobs, results, expected_model="fixed-model")
    result["label"] = 1
    result["prob_yes"] = .8
    results.write_text(json.dumps(result) + "\n")
    with pytest.raises(ValueError, match="token evidence"):
        run.read_generated_labels(jobs, results, expected_model="fixed-model")


def generation_bundle(root, metadata):
    from src import wildifeval_scoring as scoring
    units, labels, _ = synthetic_units()
    artifacts = {key: "generation_" + key + ".json" for key in run.GENERATION_ARTIFACTS}
    artifacts["runner"] = "src/wildifeval_scoring.py"
    shutil.copyfile(scoring.__file__, root / artifacts["runner"])
    (root / artifacts["units"]).write_text("\n".join(json.dumps(row) for row in units) + "\n")
    jobs, results = [], []
    for unit in units:
        for jid, cid in zip(unit["job_ids"], unit["criterion_ids"]):
            prompt = "Synthetic grading prompt " + jid
            row = dict(job_id=jid, unit_id=unit["unit_id"], criterion_id=cid, prompt=prompt, prompt_sha256=digest(prompt))
            jobs.append(row)
            value = {key: value for key, value in row.items() if key != "prompt"}
            label = labels[jid]
            value.update(model="gpt-6-luna", status="success", label=label, prob_yes=float(label),
                         parser_version=scoring.PARSER_VERSION, finish_reason="stop", service_tier="default",
                         evidence_serialization_version=scoring.EVIDENCE_SERIALIZATION_VERSION,
                         invalid_prefix_positions=[],
                         top_logprobs=[{"top_logprobs": [{"token": "Yes" if label else "No", "logprob": 0.0}]}])
            results.append(value)
    for role, rows in (("jobs", jobs), ("results", results)):
        (root / artifacts[role]).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    source = {"files": [{"path": "predictions/llm_aaj_scores/llm-aaj-deepseek-v3." + p + "-0shot-wild-if-eval.json",
                           "sha256": run.sha256(root / (p + ".json"))} for p in ("a", "b")]}
    dump(root / artifacts["source_manifest"], source)
    job_manifest = {"counts": {"jobs": len(jobs)}, "files": {key + ".jsonl": {"sha256": run.sha256(root / artifacts[key])} for key in ("jobs", "units")},
                    "source_inputs": {"source_manifest_sha256": run.sha256(root / artifacts["source_manifest"])}}
    dump(root / artifacts["job_manifest"], job_manifest)
    config = scoring.config_example(run.sha256(root / artifacts["jobs"]))
    config["state"] = "ready"
    dump(root / artifacts["config"], config)
    calibration = dict(status="passed", source="unrelated_synthetic_probe", model_requested=config["model"],
                       response_model=config["expected_response_model"], request_parameters=scoring.request_parameters(config),
                       usage_bound_ok=True, score_parse_ok=True)
    dump(root / artifacts["calibration"], calibration)
    receipt = dict(status="frozen", ledger_id="synthetic-ledger", config_sha256=scoring.digest(config), jobs_sha256=config["jobs_sha256"],
                   calibration_sha256=run.sha256(root / artifacts["calibration"]), runner_sha256=run.sha256(root / artifacts["runner"]))
    dump(root / artifacts["receipt"], receipt)
    (root / artifacts["failures"]).write_text("")
    dump(root / artifacts["accounting_summary"], {"jobs": {"success": len(jobs)}, "halted_reason": None,
                                                 "ledger_id": "synthetic-ledger", "config_sha256": scoring.digest(config)})
    dump(root / artifacts["fee"], dict(status="passed", ledger_id="synthetic-ledger", config_sha256=scoring.digest(config),
                                        canary_job_ids_sha256=None, complete=True, estimable=True, projected_total_usd=.01))
    completion = dict(status="complete", planned_jobs=len(jobs), terminal_jobs=len(jobs), pending_jobs=0,
                      inflight_jobs=0, success_jobs=len(jobs), failure_jobs=0, halted_reason=None,
                      ledger_id="synthetic-ledger", config_sha256=scoring.digest(config))
    for key, role in (("jobs_sha256", "jobs"), ("runner_sha256", "runner"), ("generation_freeze_sha256", "receipt"),
                      ("calibration_sha256", "calibration"), ("results_sha256", "results"), ("failures_sha256", "failures"),
                      ("accounting_summary_sha256", "accounting_summary")):
        completion[key] = run.sha256(root / artifacts[role])
    dump(root / artifacts["completion"], completion)
    return artifacts


def test_generation_chain_rebuilds_p_and_freezes_all_evidence(frozen):
    root, _, metadata, _ = frozen
    artifacts = generation_bundle(root, metadata)
    receipt = run.create_design_freeze(root, metadata_path=root / "units.jsonl", preflight_path=root / "preflight.json",
        config_path=root / "config.json", reference_files={p: p + ".json" for p in ("a", "b")},
        required_files=[], output_path=root / "verified_design.json", generation_artifacts=artifacts)
    assert set(artifacts.values()) <= set(receipt["frozen_files"])
    assert receipt["generation_verification"]["successful_prediction_records"] == 80
    run.validate_design(root, root / "verified_design.json")


def test_hand_edited_p_cannot_become_generation_backed_metadata(frozen):
    root, _, metadata, _ = frozen
    artifacts = generation_bundle(root, metadata)
    metadata[0]["p"][0] = 1
    (root / "units.jsonl").write_text("\n".join(json.dumps(row) for row in metadata) + "\n")
    with pytest.raises(ValueError, match="does not reproduce immutable generation"):
        run.create_design_freeze(root, metadata_path=root / "units.jsonl", preflight_path=root / "preflight.json",
            config_path=root / "config.json", reference_files={p: p + ".json" for p in ("a", "b")},
            required_files=[], output_path=root / "never.json", generation_artifacts=artifacts)


def test_failed_calibration_cannot_freeze_analysis(frozen):
    root, _, metadata, _ = frozen
    artifacts = generation_bundle(root, metadata)
    cal = run.read_json(root / artifacts["calibration"])
    cal["status"] = "failed"
    dump(root / artifacts["calibration"], cal)
    with pytest.raises(ValueError, match="invalid_calibration"):
        run.create_design_freeze(root, metadata_path=root / "units.jsonl", preflight_path=root / "preflight.json",
            config_path=root / "config.json", reference_files={p: p + ".json" for p in ("a", "b")},
            required_files=[], output_path=root / "never.json", generation_artifacts=artifacts)


def test_partial_but_feasible_generation_cannot_freeze_analysis(frozen):
    root, _, metadata, _ = frozen
    artifacts = generation_bundle(root, metadata)
    # The complete metadata passes the coverage gates, but the execution ledger
    # still records one job awaiting completion. It must not be called missing.
    completion = run.read_json(root / artifacts["completion"])
    completion.update(status="incomplete", pending_jobs=1, terminal_jobs=79, success_jobs=79)
    dump(root / artifacts["completion"], completion)
    with pytest.raises(ValueError, match="Generation is incomplete"):
        run.create_design_freeze(root, metadata_path=root / "units.jsonl", preflight_path=root / "preflight.json",
            config_path=root / "config.json", reference_files={p: p + ".json" for p in ("a", "b")},
            required_files=[], output_path=root / "never.json", generation_artifacts=artifacts)


def test_completed_exports_must_account_for_every_job(frozen):
    root, _, metadata, _ = frozen
    artifacts = generation_bundle(root, metadata)
    result_path = root / artifacts["results"]
    result_path.write_text("\n".join(result_path.read_text().splitlines()[:-1]) + "\n")
    completion = run.read_json(root / artifacts["completion"])
    completion["results_sha256"] = run.sha256(result_path)
    dump(root / artifacts["completion"], completion)
    with pytest.raises(ValueError, match="every planned job exactly once"):
        run.create_design_freeze(root, metadata_path=root / "units.jsonl", preflight_path=root / "preflight.json",
            config_path=root / "config.json", reference_files={p: p + ".json" for p in ("a", "b")},
            required_files=[], output_path=root / "never.json", generation_artifacts=artifacts)


def migrated_bundle(root, metadata):
    """Synthetic mixed-version archive: two old successes, one terminal failure."""
    from src import wildifeval_scoring as scoring
    from src import wildifeval_scoring_migration as migration
    artifacts = generation_bundle(root, metadata)
    artifacts.update({key: "generation_" + key + ".json" for key in run.MIGRATION_ARTIFACTS})
    paths = {key: root / value for key, value in artifacts.items()}
    old_config = run.read_json(paths["config"])
    dump(paths["predecessor_config"], old_config)
    new_config = {**old_config, "budget_usd": 99.5}
    dump(paths["config"], new_config)
    paths["predecessor_runner"].write_text(paths["runner"].read_text() + "\n# Synthetic predecessor identity.\n")
    shutil.copyfile(migration.__file__, paths["migration_script"])
    paths["predecessor_ledger_backup"].write_bytes(b"synthetic backup hash fixture; not opened as SQLite")
    old_freeze = run.read_json(paths["receipt"])
    old_freeze.update(runner_sha256=run.sha256(paths["predecessor_runner"]), state_directory_sha256="synthetic-state-hash")
    dump(paths["predecessor_freeze"], old_freeze)
    probe = dict(source="unrelated_synthetic_logprob_compatibility", budget_usd=.5,
                 charged_or_reserved_nano=1000, cases=[{"synthetic": True}])
    dump(paths["compatibility_diagnostic"], probe)
    freeze = {**old_freeze, "config_sha256": scoring.digest(new_config),
              "runner_sha256": run.sha256(paths["runner"]),
              "previous_generation_freeze_sha256": run.sha256(paths["predecessor_freeze"]),
              "migration_script_sha256": run.sha256(paths["migration_script"]),
              "compatibility_diagnostic_sha256": run.sha256(paths["compatibility_diagnostic"]),
              "evidence_serialization_version": scoring.EVIDENCE_SERIALIZATION_VERSION}
    dump(paths["receipt"], freeze)
    results = run.read_jsonl(paths["results"])
    old_success = deepcopy(results[:2])
    for row in old_success:
        row.pop("evidence_serialization_version")
        row.pop("invalid_prefix_positions")
    old_failure = [{"job_id": row["job_id"], "status": "terminal_missing" if i == 0 else "pending",
                    "attempt_count": 1 if i == 0 else 0} for i, row in enumerate(results[2:])]
    final_results = old_success + results[3:]
    final_failures = old_failure[:1]
    for role, rows in (("predecessor_results", old_success), ("predecessor_failures", old_failure),
                       ("results", final_results), ("failures", final_failures)):
        paths[role].write_text("".join(json.dumps(row) + "\n" for row in rows))
    hashes = {"old_config_sha256": scoring.digest(old_config), "new_config_sha256": scoring.digest(new_config)}
    for key, role in (("old_config_file_sha256", "predecessor_config"), ("new_config_file_sha256", "config"),
                      ("old_runner_sha256", "predecessor_runner"), ("new_runner_sha256", "runner"),
                      ("old_generation_freeze_sha256", "predecessor_freeze"), ("new_generation_freeze_sha256", "receipt"),
                      ("calibration_sha256", "calibration"), ("diagnostic_sha256", "compatibility_diagnostic"),
                      ("old_results_sha256", "predecessor_results"), ("old_failures_sha256", "predecessor_failures"),
                      ("migration_script_sha256", "migration_script"), ("jobs_sha256", "jobs")):
        hashes[key] = run.sha256(paths[role])
    receipt = dict(status="completed", kind="selected-prefix-evidence-v2", **hashes,
                   ledger_id=freeze["ledger_id"], state_directory_sha256=freeze["state_directory_sha256"],
                   database_backup_sha256=run.sha256(paths["predecessor_ledger_backup"]),
                   evidence_serialization_version=scoring.EVIDENCE_SERIALIZATION_VERSION,
                   fixed_logic_ast_sha256=run._fixed_scoring_logic(paths["runner"]),
                   charged_nano_unchanged=5000, prior_nano_unchanged=2000,
                   immutable_records_sha256="a"*64, old_budget_nano=100_000_000_000,
                   new_budget_nano=99_500_000_000, diagnostic_budget_reserved_nano=500_000_000,
                   diagnostic_charged_or_reserved_nano=1000, request_parameters_unchanged=True,
                   score_algorithm_unchanged=True, existing_results_unchanged=True, existing_missing_not_retried=True)
    receipt["migration_id"] = scoring.digest({**hashes, "ledger_id": freeze["ledger_id"], "kind": receipt["kind"]})
    for prefix, rows in (("old_success", old_success), ("old_terminal_failure", final_failures)):
        ids = sorted(row["job_id"] for row in rows)
        receipt.update({prefix+"_ids": ids, prefix+"_ids_sha256": scoring.digest(ids), prefix+"_jobs": len(ids)})
    dump(paths["migration_receipt"], receipt)
    for role in ("accounting_summary", "fee", "completion"):
        payload = run.read_json(paths[role])
        payload["config_sha256"] = scoring.digest(new_config)
        if role == "accounting_summary":
            payload["jobs"] = {"success": len(final_results), "terminal_missing": 1}
        elif role == "completion":
            payload.update(success_jobs=len(final_results), failure_jobs=1)
            for key, bound in (("generation_freeze_sha256", "receipt"), ("results_sha256", "results"),
                               ("failures_sha256", "failures"), ("accounting_summary_sha256", "accounting_summary")):
                payload[key] = run.sha256(paths[bound])
        dump(paths[role], payload)
    return artifacts, paths, new_config, freeze


def test_migration_archive_preserves_legacy_identity_and_all_terminal_records(frozen):
    root, _, metadata, _ = frozen
    artifacts, paths, generation_config, freeze = migrated_bundle(root, metadata)
    result = run._verify_serialization_migration(paths, generation_config, freeze, diagnostic=True)
    assert result["legacy_success_records"] == 2
    assert result["selected_prefix_v2_success_records"] == 77
    assert result["preserved_terminal_failures"] == 1
    cfg = run.read_json(root / "config.json")
    labels = run.read_generated_labels(paths["jobs"], paths["results"], expected_model="gpt-6-luna")
    units, report = run.prepare_metadata(run.read_jsonl(paths["units"]), labels, cfg)
    (root / "units.jsonl").write_text("".join(json.dumps(row) + "\n" for row in units))
    dump(root / "preflight.json", report)
    receipt = run.create_design_freeze(root, metadata_path=root / "units.jsonl", preflight_path=root / "preflight.json",
        config_path=root / "config.json", reference_files={p: p+".json" for p in ("a", "b")},
        required_files=[], output_path=root / "migrated_design.json", generation_artifacts=artifacts)
    assert receipt["generation_verification"]["serialization"] == result
    assert set(artifacts.values()) <= set(receipt["frozen_files"])
    run.validate_design(root, root / "migrated_design.json")


@pytest.mark.parametrize("case,match", [
    ("rewrite_success", "rewritten or relabelled"), ("retry_terminal", "retried into a success"),
    ("retry_failure", "changed or was retried"), ("hide_version", "lacks its v2"),
    ("drop_terminal", "lost predecessor"), ("tamper_backup", "does not bind"),
    ("tamper_predecessor", "does not bind"), ("change_request", "beyond its diagnostic"),
])
def test_migration_rejects_history_rewriting_and_provenance_corruption(frozen, case, match):
    root, _, metadata, _ = frozen
    _, paths, cfg, freeze = migrated_bundle(root, metadata)
    rows = run.read_jsonl(paths["results"])
    if case == "rewrite_success":
        rows[0]["evidence_serialization_version"] = "selected-prefix-v2"
    elif case == "retry_terminal":
        rows.append({"job_id": run.read_jsonl(paths["failures"])[0]["job_id"], "status": "success"})
    elif case == "hide_version":
        rows[-1].pop("evidence_serialization_version")
    elif case == "retry_failure":
        failures = run.read_jsonl(paths["failures"])
        failures[0]["attempt_count"] = 2
        paths["failures"].write_text(json.dumps(failures[0]) + "\n")
    elif case == "drop_terminal":
        paths["failures"].write_text("")
    elif case == "tamper_backup":
        paths["predecessor_ledger_backup"].write_bytes(b"changed")
    elif case == "tamper_predecessor":
        paths["predecessor_runner"].write_text("changed")
    elif case == "change_request":
        # Even a self-consistent edited hash chain cannot authorize a request change.
        cfg["max_attempts"] = 2
        dump(paths["config"], cfg)
        from src import wildifeval_scoring as scoring
        receipt = run.read_json(paths["migration_receipt"])
        receipt["new_config_sha256"] = scoring.digest(cfg)
        receipt["new_config_file_sha256"] = run.sha256(paths["config"])
        keys = [key for key in receipt if key.endswith("sha256") and key in {
            "old_config_sha256", "new_config_sha256", "old_config_file_sha256", "new_config_file_sha256",
            "old_runner_sha256", "new_runner_sha256", "old_generation_freeze_sha256", "new_generation_freeze_sha256",
            "calibration_sha256", "diagnostic_sha256", "old_results_sha256", "old_failures_sha256",
            "migration_script_sha256", "jobs_sha256"}]
        receipt["migration_id"] = scoring.digest({**{key: receipt[key] for key in keys},
            "ledger_id": freeze["ledger_id"], "kind": receipt["kind"]})
        dump(paths["migration_receipt"], receipt)
    paths["results"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=match):
        run._verify_serialization_migration(paths, cfg, freeze, diagnostic=True)


def test_migrated_generation_requires_the_whole_predecessor_bundle(frozen):
    root, _, metadata, _ = frozen
    artifacts, _, _, _ = migrated_bundle(root, metadata)
    del artifacts["predecessor_results"]
    with pytest.raises(ValueError, match="migration predecessor"):
        run._verify_generation(root, artifacts, config(), metadata, {p: p+".json" for p in ("a", "b")})


def test_prefix_version_cannot_change_or_append_unused_positions(tmp_path):
    from src.wildifeval_scoring import PARSER_VERSION
    job = dict(job_id="job", unit_id="unit", criterion_id="criterion", prompt="text", prompt_sha256=digest("text"))
    row = {key: value for key, value in job.items() if key != "prompt"}
    row.update(status="success", model="fixed-model", label=1, prob_yes=1., parser_version=PARSER_VERSION,
               finish_reason="stop", service_tier="default", evidence_serialization_version="selected-prefix-v2",
               invalid_prefix_positions=[], top_logprobs=[{"top_logprobs": [{"token": "Yes", "logprob": 0.}]}])
    jobs, results = tmp_path / "jobs.jsonl", tmp_path / "results.jsonl"
    jobs.write_text(json.dumps(job) + "\n")
    results.write_text(json.dumps(row) + "\n")
    assert run.read_generated_labels(jobs, results, expected_model="fixed-model") == {"job": 1}
    row["top_logprobs"].append({"top_logprobs": []})
    results.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="serialization"):
        run.read_generated_labels(jobs, results, expected_model="fixed-model")
