# WildIFEval: aligned grading and staged external validation

**Status, 2026-10-22:** P is complete and the completed-data design freeze is issued. Pilot R opens after this publication; no reference value had been read when it was made. Sections 14 to 18 of the [protocol](../context/conjunction_external_validation_protocol.md) record the route, and the receipts are in [`artifacts/expected/wildifeval_flashlite/`](../artifacts/expected/wildifeval_flashlite/). They contain counts, hashes and identifiers but no prompt, output or answer text.

- **Judge (section 14).** The `gpt-6-luna` run of section 12 stopped at its cap after 7,357 predictions, which are archived and excluded. P is `gemini-3.1-flash-lite` (temperature 0, thinking LOW, 2,048 output tokens) through Gemini API Batch Mode. All 134,485 requests returned an answer, at a settled cost of 29.84 USD.
- **Parser A1 (section 15).** The pinned prompt does not ask for Yes or No, so the section 14 yes/no parser left 18,996 answers (14.1%) missing. A1 keeps every section 14 label and reads explicit first-line verdicts, which leaves 4,430 missing (3.3%). The section 14 parser fails the 70% retention gate, and that result is reported as a sensitivity analysis.
- **Repair sensitivity (section 16).** The 2,398 unblocked missing jobs were asked once more with "Answer with Yes or No." added, for 0.45 USD. This repaired P is a sensitivity analysis only.
- **Design freeze (sections 17 and 18).** The freeze re-derives every request, answer, cost and label from the raw batch files. Its frame has 6,188 complete tasks (retention 0.8225 per producer) in 5,742 clusters: 1,148 pilot and 4,594 confirmation. All gates pass.
- **Correction (section 18).** An earlier Vertex AI batch run with the same configuration had also completed, which section 14 had misreported. Its outputs serve only as a replication check. Criterion labels agree 97.0% between the two runs, and output verdicts agree 96.0%.

The pilot stage runs in the private research tree, which holds the job texts, the P metadata and the saved references:

```bash
python -m src.wildifeval_external_run validate --design artifacts/reports/wildifeval_external/flashlite_design/design_freeze.json
python -m src.wildifeval_external_run pilot --design artifacts/reports/wildifeval_external/flashlite_design/design_freeze.json --output artifacts/reports/wildifeval_external/flashlite_pilot
```

The pilot stage reads R for pilot clusters only. It fits the frozen forecasters and writes the prediction freeze before any confirmation value can be read.

**Earlier status, 2026-10-16:** the author adopted the two-role study and two unrelated API calibration cases passed. The 160-request fee canary and evidence-only compatibility repair are complete, and full grading is running. Dataset-level P is not complete; this page reports no completed-data design freeze, pilot fit or held-out result. It describes the executable workflow and the completed API calibration.

The [original masked preflight](WILDIFEVAL.md) failed because saved judges did not share complete rubrics. The [dated amendment](../context/conjunction_external_validation_protocol.md#12-author-adopted-two-role-execution-amendment--2026-10-16) instead generates a fixed new P on the exact saved outputs and conditions, with saved DeepSeek scores as R. R is a machine reference. Its historical output association still depends on upstream producer/task provenance because its release lacks evaluated-output hashes. New P records output hashes but does not repair that historical gap.

The candidate input contains 134,485 criterion requests, 37,090 outputs, 7,418 tasks and 6,847 clusters across five producing models. These are input counts, not completed predictions. Generation failures can reduce the final common frame. The statistical pilot opens R only after that frame has been checked and frozen.

## 1. Rebuild private jobs from pinned sources

Run commands from the repository root. Install the optional source reader and API client, then follow [the preflight download instructions](WILDIFEVAL.md#reproduce-the-preflight). Keep source text, jobs, credentials and runtime state outside Git. The execution paths below use the ignored `artifacts/wildifeval-execution-private/` directory; the source cache remains separate.

```bash
python -m pip install ".[external,scoring]"
export WILD_RUN=artifacts/wildifeval-execution-private/new-p-v1
mkdir -p "$WILD_RUN"
curl --fail --location \
  https://raw.githubusercontent.com/gililior/wild-if-eval-code/82ebff7068b9faeadc5468d55935d62b1601d89c/scripts/evaluate_llms/constants.py \
  --output "$WILD_RUN/constants.py"
python -m src.wildifeval_jobs \
  --sources artifacts/wildifeval-cache \
  --masked artifacts/wildifeval-preflight/masked_units.csv \
  --preflight-report artifacts/wildifeval-preflight/report.json \
  --template "$WILD_RUN/constants.py" \
  --private-dir "$WILD_RUN" \
  --manifest "$WILD_RUN/job_manifest.json"
```

The builder validates pinned source and template hashes, complete ordered criteria and saved output hashes. It reads the reference **availability masks**, without opening reference values. `jobs.jsonl` contains private grading prompts; `units.jsonl` contains their immutable associations. Existing job artifacts are never overwritten.

## 2. Calibrate the API, fix the configuration and initialize accounting

The implemented configuration uses `gpt-6-luna`, one user message, temperature 0, no reasoning, top-five token log probabilities and a 1,000-token completion cap. The parser follows the pinned upstream Yes/No probability rule; probability at least 0.5 gives P=PASS. Unparseable responses, refusal and truncation are missing outcomes, not FAIL labels. The runner does not retry these responses to obtain a preferred answer.

Before a new generation, obtain a successful **unrelated synthetic** calibration under these exact settings. Save its returned model identity, request parameters, parser checks, usage checks and cost accounting as `api_calibration.json`. The runner validates this receipt. No calibration CLI is provided; an archived successful receipt documents its own run and is not evidence that a different account or future endpoint still behaves identically.

For inspection, the example configuration is available through:

```bash
python -m src.wildifeval_scoring example-config
```

After validating the successful calibration, create the ready configuration without changing its model, prompt or parsing settings:

```python
import json, os
from pathlib import Path
from src import wildifeval_scoring as scoring

work = Path(os.environ["WILD_RUN"])
calibration = json.loads((work / "api_calibration.json").read_text())
config = scoring.config_example(scoring.file_hash(work / "jobs.jsonl"))
assert calibration["status"] == "passed"
assert calibration["model_requested"] == config["model"]
assert calibration["request_parameters"] == scoring.request_parameters(config)
assert calibration["usage_bound_ok"] and calibration["score_parse_ok"]
config.update(state="ready",
              expected_response_model=calibration["response_model"],
              expected_system_fingerprint=calibration["system_fingerprint"],
              prior_reserved_usd=calibration["prior_unresolved_reserve_usd"]
                  + calibration["known_conservative_nano"] / scoring.NANODOLLARS)
scoring.validate_config(config)
with (work / "generation_config.json").open("x") as handle:
    json.dump(config, handle, indent=2)
```

```bash
python -m src.wildifeval_scoring check \
  --jobs "$WILD_RUN/jobs.jsonl" --config "$WILD_RUN/generation_config.json"
python -m src.wildifeval_scoring make-canary \
  --jobs "$WILD_RUN/jobs.jsonl" --output "$WILD_RUN/canary.json"
python -m src.wildifeval_scoring init-ledger \
  --jobs "$WILD_RUN/jobs.jsonl" --config "$WILD_RUN/generation_config.json" \
  --state "$WILD_RUN/state" > "$WILD_RUN/ledger_init.json"
```

These three commands make no API calls. The ledger fixes the configuration and accounting identity. Its budget includes prior charges/reserves and every subsequent dispatch. The current configuration caps the run at $100 using conservative token-cost reservations; this is not a provider invoice or an account-wide spending lock. Unknown billing outcomes retain their reservations.

Create the generation freeze only after those checks. It binds the ledger, code, jobs and successful calibration; it is separate from the later analysis freeze:

```python
import json, os
from pathlib import Path
from src import wildifeval_scoring as scoring

work = Path(os.environ["WILD_RUN"])
config = json.loads((work / "generation_config.json").read_text())
ledger = json.loads((work / "ledger_init.json").read_text())
receipt = dict(status="frozen", config_sha256=scoring.digest(config),
    jobs_sha256=scoring.file_hash(work / "jobs.jsonl"),
    calibration_sha256=scoring.file_hash(work / "api_calibration.json"),
    runner_sha256=scoring.file_hash(Path(scoring.__file__)),
    ledger_id=ledger["ledger_id"], state_directory_sha256=ledger["state_directory_sha256"],
    canary_job_ids_sha256=scoring.file_hash(work / "canary.json"))
with (work / "generation_freeze.json").open("x") as handle:
    json.dump(receipt, handle, indent=2)
```

## 3. Measure cost on the fixed canary, then generate P

The canary selects 32 immutable job IDs per producer by the frozen hash rule, for 160 requests. It checks API usage and cost on actual prompts. It is **not the statistical pilot** and exposes no R. Its successful predictions are retained in the same ledger and reused in the full run.

The following two `run` commands incur API charges. Replace the key-file argument with an authorized private file; never put its contents in a command, artifact or repository.

```bash
python -m src.wildifeval_scoring run \
  --jobs "$WILD_RUN/jobs.jsonl" --config "$WILD_RUN/generation_config.json" \
  --state "$WILD_RUN/state" --key-file /absolute/private/key-file \
  --calibration-receipt "$WILD_RUN/api_calibration.json" \
  --generation-freeze "$WILD_RUN/generation_freeze.json" \
  --job-ids "$WILD_RUN/canary.json" --phase-cap-usd 1
python -m src.wildifeval_scoring fee-report \
  --jobs "$WILD_RUN/jobs.jsonl" --config "$WILD_RUN/generation_config.json" \
  --state "$WILD_RUN/state" \
  --calibration-receipt "$WILD_RUN/api_calibration.json" \
  --generation-freeze "$WILD_RUN/generation_freeze.json" \
  --output "$WILD_RUN/fee_candidate.json"
```

Review the fee report's completeness, known and unresolved charges, per-producer support and projected total. Preserve the candidate. A passed copy, `fee_passed.json`, may change `status` to `passed` only after this review; keep every calculated field unchanged. The full-run guard independently recomputes the canary accounting and requires a complete, estimable projection within the frozen budget. A projection remains an estimate; per-dispatch reservations enforce the separate ceiling. Neither review nor selection uses prediction accuracy or R.

```bash
python -m src.wildifeval_scoring run \
  --jobs "$WILD_RUN/jobs.jsonl" --config "$WILD_RUN/generation_config.json" \
  --state "$WILD_RUN/state" --key-file /absolute/private/key-file \
  --calibration-receipt "$WILD_RUN/api_calibration.json" \
  --generation-freeze "$WILD_RUN/generation_freeze.json" \
  --fee-receipt "$WILD_RUN/fee_passed.json"
```

Resume using the same state directory, configuration and receipts; successful job IDs are not generated again. Use `--resume-after-quota` only after the quota problem is resolved. Transient failures have at most three attempts; quota/authentication/configuration failures stop the run. A model, parser or source change requires a separately documented design change rather than silently modifying frozen files.

After generation stops, finalize its exports offline:

```bash
python -m src.wildifeval_scoring finalize \
  --jobs "$WILD_RUN/jobs.jsonl" --config "$WILD_RUN/generation_config.json" \
  --state "$WILD_RUN/state" \
  --calibration-receipt "$WILD_RUN/api_calibration.json" \
  --generation-freeze "$WILD_RUN/generation_freeze.json"
```

`state/completion.json` must say `complete`: all 134,485 planned requests are terminal, none are pending or in flight, and no global halt remains. `results.jsonl` and `failures.jsonl` must account for every job exactly once. The finalizer exits unsuccessfully for an incomplete run. **Passing the 70% retention threshold does not permit freezing an unfinished generation.** Keep the SQLite ledger, exports, summary, completion record and invocation history.

### The recorded canary compatibility repair

The first canary retained 152 successful predictions and eight terminal missing records. Those eight responses had a usable scoring position but failed the old serializer's validation of later, unused log-probability positions. After unrelated synthetic diagnostics, the runner was revised to store only the prefix through the selected scoring position. Earlier malformed positions are represented by empty lists that preserve their indices. The Yes/No score, threshold, request settings and existing records are unchanged.

This is a prospective storage repair triggered by observed API formatting, and is recorded as such. The 152 existing successful rows remain v1 records; they are not described as generated by v2. The eight failures remain missing and are not retried. Later successes carry `evidence_serialization_version="selected-prefix-v2"`. No R was used in diagnosing or choosing the repair.

The actual run continues in the same ledger with a $99.50 core ceiling and $0.50 reserved separately for the unrelated diagnostics. Its old code, configuration, freeze and complete exports are archived in `generation_v1/`. The migration uses a SQLite backup, preserves charges and reservations, and binds both versions in its receipt. It cannot run while requests are in flight. The offline migration command has explicit predecessor arguments:

```bash
python -m src.wildifeval_scoring_migration \
  --jobs "$WILD_RUN/jobs.jsonl" --state "$WILD_RUN/state" \
  --old-config "$WILD_RUN/generation_v1/config.json" \
  --new-config "$WILD_RUN/generation_config_v2.json" \
  --old-runner "$WILD_RUN/generation_v1/scoring.py" \
  --old-freeze "$WILD_RUN/generation_v1/generation_freeze.json" \
  --new-freeze "$WILD_RUN/generation_freeze_v2.json" \
  --calibration "$WILD_RUN/api_calibration.json" \
  --diagnostic "$WILD_RUN/compatibility_diagnostic.json" \
  --old-results "$WILD_RUN/generation_v1/results.jsonl" \
  --old-failures "$WILD_RUN/generation_v1/failures.jsonl" \
  --backup "$WILD_RUN/generation_v1/ledger.sqlite" \
  --receipt "$WILD_RUN/migration_v1_v2.json"
```

Before this command, create the v2 configuration by changing only the ceiling from 100 to 99.5. Create its freeze with the same ledger and canary identity, updated config/runner hashes, and `previous_generation_freeze_sha256`, `migration_script_sha256`, `compatibility_diagnostic_sha256` and `evidence_serialization_version`. The migration verifies the archived exports against the ledger and refuses any change to scoring, request or pricing logic. It creates the backup itself. Resume with the v2 configuration and freeze, and create a new fee report bound to them. This migration applies to the recorded v1 canary; a fresh run using the current runner starts with v2 records and has no predecessor migration.

## 4. Recheck the final common frame and freeze the analysis

The following preparation reads generated P and reference availability/provenance, but no R values. It validates the export's request associations, token evidence, probability and label. A terminal missing P condition excludes the whole task across all five producers. The original hash split is recomputed on the final eligible clusters.

```python
import json, os
from pathlib import Path
from src import wildifeval_external_run as analysis

root = Path(".").resolve()
work = Path(os.environ["WILD_RUN"])
out = work / "analysis"
out.mkdir(exist_ok=False)
generation = analysis.read_json(work / "generation_config.json")
config = analysis.default_config()
config["amendment_path"] = "context/conjunction_external_validation_protocol.md"
labels = analysis.read_generated_labels(work / "jobs.jsonl", work / "state/results.jsonl",
    expected_model=generation["expected_response_model"],
    expected_system_fingerprint=generation["expected_system_fingerprint"])
units, preflight = analysis.prepare_metadata(analysis.read_jsonl(work / "units.jsonl"), labels, config)
for name, value in (("config.json", config), ("preflight.json", preflight)):
    with (out / name).open("x") as handle:
        json.dump(value, handle, indent=2)
if not preflight["feasible"]:
    raise SystemExit("Final coverage gates failed; preserve exclusions and stop before R access")
with (out / "units.jsonl").open("x") as handle:
    for unit in units:
        handle.write(json.dumps(unit, sort_keys=True) + "\n")

sources = Path("artifacts/wildifeval-cache")
artifacts = {
    "jobs": work / "jobs.jsonl", "units": work / "units.jsonl",
    "results": work / "state/results.jsonl", "failures": work / "state/failures.jsonl",
    "config": work / "generation_config.json", "receipt": work / "generation_freeze.json",
    "calibration": work / "api_calibration.json", "runner": Path("src/wildifeval_scoring.py"),
    "job_manifest": work / "job_manifest.json", "source_manifest": sources / "manifest.json",
    "completion": work / "state/completion.json", "accounting_summary": work / "state/summary.json",
    "fee": work / "fee_passed.json"}
# For the recorded migrated run, use its active v2 config/freeze and fee receipt
# above, and include every immutable predecessor artifact below. Omit this
# branch for a fresh current-runner generation with no predecessor.
if "previous_generation_freeze_sha256" in analysis.read_json(artifacts["receipt"]):
    artifacts.update({
        "predecessor_config": work / "generation_v1/config.json",
        "predecessor_runner": work / "generation_v1/scoring.py",
        "predecessor_freeze": work / "generation_v1/generation_freeze.json",
        "predecessor_results": work / "generation_v1/results.jsonl",
        "predecessor_failures": work / "generation_v1/failures.jsonl",
        "predecessor_ledger_backup": work / "generation_v1/ledger.sqlite",
        "migration_receipt": work / "migration_v1_v2.json",
        "migration_script": Path("src/wildifeval_scoring_migration.py"),
        "compatibility_diagnostic": work / "compatibility_diagnostic.json"})
references = {producer: str(sources / "predictions/llm_aaj_scores" /
    ("llm-aaj-deepseek-v3." + producer + "-0shot-wild-if-eval.json")) for producer in config["producers"]}
analysis.create_design_freeze(root, metadata_path=out / "units.jsonl",
    preflight_path=out / "preflight.json", config_path=out / "config.json",
    reference_files=references, generation_artifacts={key: str(path) for key, path in artifacts.items()},
    required_files=["src/wildifeval_jobs.py", "src/wildifeval_scoring.py"],
    output_path=out / "design_freeze.json")
```

The unchanged gates require at least 500 eligible clusters, 100 pilot clusters, 300 confirmation clusters and 70% retention in each producer. The design freezer independently rebuilds P and its complete-case frame from the generation archive and binds code, protocol, source files, all IDs, exclusions and split. For a migrated archive, it checks the predecessor hashes and unchanged scoring logic, verifies that the 152 old successful rows remain identical and all eight terminal failures remain terminal, and counts v1 and v2 successes separately. Preserve and publish the non-text design record before proceeding. Do not publish source prompts or outputs alongside it.

## 5. Fit the pilot, publish predictions, then open confirmation

```bash
python -m src.wildifeval_external_run validate --root . \
  --design "$WILD_RUN/analysis/design_freeze.json"
python -m src.wildifeval_external_run pilot --root . \
  --design "$WILD_RUN/analysis/design_freeze.json" \
  --output "$WILD_RUN/analysis/pilot"
```

The pilot command opens only pilot R. It saves `predictions.json` and `prediction_freeze.json`: 15 producer-by-budget cells for each of four predictors, plus allocation predictions and support counts. Publish this prediction record before confirmation access. If the pilot cannot support the prescribed forecasts, preserve its failed attempt and stop; do not inspect confirmation to repair the predictor.

```bash
python -m src.wildifeval_external_run confirmation --root . \
  --design "$WILD_RUN/analysis/design_freeze.json" \
  --predictions "$WILD_RUN/analysis/pilot/prediction_freeze.json" \
  --output "$WILD_RUN/analysis/confirmation"
```

The confirmation command verifies both freezes and reproduces the saved predictions before reading confirmation R. It reports every producer and the fixed 20%, 50% and 80% budgets. The primary comparison uses 999 paired confirmation-cluster resamples with the pilot fit fixed; every draw recomputes the criterion frame, budgets, forecasts and exact outcomes. The prespecified interval determines support, contradiction or an unresolved result.

Secondary outputs compare SRS, 50/50 allocation and serial querying over 101 priority seeds and the full budget grid, with the 20% allocation endpoint retained. Seed variability is query-order variability, not a task-composition confidence interval. The reference-budget ledger counts pilot acquisitions, operational queries and validation-only reveals separately; no-pilot controls use matched total reference costs. Auxiliary-disagreement and alternative-reference analyses are omitted in this two-role study. All three manuscript RQs remain unchanged.

The result directory contains `primary.json`, `secondary.json`, `reference_budget.json`, compressed policy and no-pilot rows, and a hash-bound result receipt. Failed attempts remain alongside their stage records. Synthetic tests verify these controls and accounting identities; only a completed, frozen real-data analysis can test the external scientific explanation.

An exact replay uses the frozen prediction exports. Regenerating P through a live API is a new acquisition and can produce different predictions even with the same requested settings.
