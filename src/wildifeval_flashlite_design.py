"""Completed-data design freeze for the Flash-Lite P archive (protocol section 14, step 6).

The section 12 analysis code verifies an OpenAI generation archive. This module verifies the Gemini API batch
archive of section 14, labelled by parser A1 of section 15, and leaves everything else to
``src.wildifeval_external_run`` unchanged: whole-task missingness, the hash split, the gates, the design freeze
and the sealed pilot stage. The verification re-derives every P label from the raw batch outputs and checks
every request against its frozen prompt. It reads no reference value; the saved reference files are only hashed
against the pinned source manifest.

    python -m src.wildifeval_flashlite_design freeze

``freeze`` writes the batch-file manifest, the P metadata, the preflight frame report and the design freeze.
The pilot stage then runs through ``python -m src.wildifeval_external_run pilot``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src import wildifeval_gemini_batch as batch
from src import wildifeval_parser_v2 as a1
from src import wildifeval_vertex_judge as online

ROOT = online.ROOT
GEMINI_PROTOCOL = "wildifeval-p-gemini-batch-v3"
ROLES = frozenset({"jobs", "units", "job_manifest", "source_manifest", "receipt", "fee", "summary", "results",
                   "batch_runner", "shared_runner", "batch_files", "relabel_receipt", "relabelled", "parser"})
DESIGN_DIR = Path("artifacts/reports/wildifeval_external/flashlite_design")   # public receipts, no text
PRIVATE_DIR = Path("data/interim/wildifeval_external/flashlite_design")      # P metadata
CONFIG_PATH = Path("context/wildifeval_two_role_analysis.json")
REFERENCE_PATTERN = "data/interim/wildifeval_external/sources/predictions/llm_aaj_scores/llm-aaj-deepseek-v3.{}-0shot-wild-if-eval.json"
ARTIFACTS = {
    "jobs": "data/interim/wildifeval_external/luna_run/jobs.jsonl",
    "units": "data/interim/wildifeval_external/luna_run/units.jsonl",
    "job_manifest": "artifacts/reports/wildifeval_external/luna_run/job_manifest.json",
    "source_manifest": "data/interim/wildifeval_external/sources/manifest.json",
    "receipt": "artifacts/reports/wildifeval_external/flashlite_gemini_batch/freeze.json",
    "fee": "artifacts/reports/wildifeval_external/flashlite_gemini_batch/fee_receipt.json",
    "summary": "artifacts/reports/wildifeval_external/flashlite_gemini_batch/summary.json",
    "results": "data/interim/wildifeval_external/flashlite_gemini_batch/results.jsonl",
    "batch_runner": "src/wildifeval_gemini_batch.py",
    "shared_runner": "src/wildifeval_vertex_judge.py",
    "batch_files": str(DESIGN_DIR / "batch_files.json"),
    "relabel_receipt": "artifacts/reports/wildifeval_external/flashlite_gemini_batch/parser_v2_receipt.json",
    "relabelled": "data/interim/wildifeval_external/flashlite_gemini_batch/results_parser_v2.jsonl",
    "parser": "src/wildifeval_parser_v2.py",
}
SENSITIVITY_FILES = ("artifacts/reports/wildifeval_external/flashlite_gemini_repair/freeze.json",
                     "artifacts/reports/wildifeval_external/flashlite_gemini_repair/summary.json",
                     "artifacts/reports/wildifeval_external/flashlite_replication/replication_receipt.json",
                     "src/wildifeval_run_replication.py")   # sections 16 and 18; never inputs to P


def is_gemini_receipt(freeze: Mapping[str, Any]) -> bool:
    return freeze.get("protocol_version") == GEMINI_PROTOCOL


def section_sha256(protocol: Path, heading: str) -> str:
    text = protocol.read_text(encoding="utf-8")
    if heading not in text:
        raise ValueError("Protocol lacks section: " + heading)
    return hashlib.sha256((heading + text.split(heading, 1)[1].split("\n## ", 1)[0]).encode("utf-8")).hexdigest()


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verify_generation(root: Path, artifacts: Mapping[str, str], config: Mapping, metadata: list[dict],
                      reference_files: Mapping[str, str]) -> dict:
    """Audit the archive from raw batch files to the final P metadata; any mismatch stops the freeze."""
    from src import wildifeval_external_run as run
    if set(artifacts) != ROLES:
        raise ValueError("Complete Flash-Lite generation evidence bundle required")
    p = {key: run._path(root, value) for key, value in artifacts.items()}
    for role, module in (("batch_runner", batch), ("shared_runner", online), ("parser", a1)):
        if run.sha256(p[role]) != run.sha256(Path(module.__file__)):
            raise ValueError("Executing code differs from the archived " + role)

    # Builder inputs and saved references are those of the pinned source manifest.
    manifest = run.read_json(p["job_manifest"])
    for role, name in (("jobs", "jobs.jsonl"), ("units", "units.jsonl")):
        if run.sha256(p[role]) != manifest["files"][name]["sha256"]:
            raise ValueError("Generated job/unit input differs from builder manifest")
    if run.sha256(p["source_manifest"]) != manifest["source_inputs"]["source_manifest_sha256"]:
        raise ValueError("Source manifest differs from the generation source binding")
    source = {row["path"]: row["sha256"] for row in run.read_json(p["source_manifest"])["files"]}
    for producer, rel in reference_files.items():
        expected = "predictions/llm_aaj_scores/llm-aaj-deepseek-v3." + producer + "-0shot-wild-if-eval.json"
        if source.get(expected) != run.sha256(run._path(root, rel)):
            raise ValueError("Saved reference does not match the pinned source manifest")
    planned = manifest["counts"]["jobs"]
    if not config["diagnostic"] and planned != 134485:
        raise ValueError("Production generation did not cover all 134485 planned criterion jobs")

    # The run freeze binds the job file, the code, the configuration and protocol section 14.
    freeze = run.read_json(p["receipt"])
    protocol = run._path(root, config["amendment_path"])
    if (not is_gemini_receipt(freeze) or freeze.get("status") != "frozen" or freeze.get("jobs") != planned
            or freeze.get("jobs_sha256") != run.sha256(p["jobs"]) or freeze.get("reference_values_opened") is not False
            or freeze.get("code_sha256") != {"batch": run.sha256(p["batch_runner"]), "shared": run.sha256(p["shared_runner"])}
            or freeze.get("parser_version") != online.PARSER_VERSION or freeze.get("config") != batch.CONFIG
            or freeze.get("protocol_section_sha256") != section_sha256(protocol, online.SECTION_HEADING)):
        raise ValueError("Run freeze does not bind the job file, code, configuration and section 14")
    model = freeze["config"]["model"]
    fee, summary = run.read_json(p["fee"]), run.read_json(p["summary"])
    if fee.get("passed") is not True or fee.get("cap_usd") != freeze["cost"]["cap_usd"]:
        raise ValueError("Fee gate did not pass under the frozen cap")
    if (summary.get("protocol_version") != GEMINI_PROTOCOL or summary.get("results_sha256") != run.sha256(p["results"])
            or summary.get("model_versions") != [model] or sum(summary["states"].values()) != planned
            or set(summary["states"]) - {"done", "error"} or summary["settled_usd"] > freeze["cost"]["cap_usd"]):
        raise ValueError("Run summary does not establish a completed, capped, single-model export")

    # Every request carries its frozen prompt and configuration; every answer matches one request.
    prompts = {row["job_id"]: row["prompt_sha256"] for row in _jsonl(p["jobs"])}
    if len(prompts) != planned:
        raise ValueError("Job file count differs from the builder manifest")
    files = run.read_json(p["batch_files"])
    keys = {b["key"] for b in summary["batches"]}
    if set(files.get("inputs", {})) != keys or set(files.get("outputs", {})) != keys:
        raise ValueError("Batch-file manifest does not cover every batch job")
    for kind in ("inputs", "outputs"):
        for entry in files[kind].values():
            if run.sha256(run._path(root, entry["path"])) != entry["sha256"]:
                raise ValueError("Archived batch file changed: " + entry["path"])
    generation = batch.request_line("job", "prompt")["request"]["generation_config"]
    sent: dict[str, int] = {}
    for key in sorted(keys):
        for item in _jsonl(run._path(root, files["inputs"][key]["path"])):
            job_id, request = item["key"], item["request"]
            contents = request.get("contents")
            if (set(request) != {"contents", "generation_config"} or request["generation_config"] != generation
                    or not isinstance(contents, list) or len(contents) != 1 or contents[0].get("role") != "user"
                    or len(contents[0].get("parts", [])) != 1):
                raise ValueError("A request differs from the frozen request format")
            text = contents[0]["parts"][0]["text"]
            if job_id not in prompts or hashlib.sha256(text.encode()).hexdigest() != prompts[job_id]:
                raise ValueError("A request does not carry its frozen prompt")
            sent[job_id] = sent.get(job_id, 0) + 1
    if set(sent) != set(prompts):
        raise ValueError("Not every frozen job was sent")
    answers: dict[str, dict] = {}
    for key in sorted(keys):
        for obj in _jsonl(run._path(root, files["outputs"][key]["path"])):
            job_id, rec, _ = batch.parse_result_line(obj)
            if job_id not in sent:
                raise ValueError("An answer matches no request")
            if rec is not None:
                if job_id in answers:
                    raise ValueError("A job has two answers")
                if not str(rec["model_version"]).startswith(model):
                    raise ValueError("An answer comes from another model")
                answers[job_id] = rec
    cost = sum(batch.batch_cost(rec["usage"]) for rec in answers.values())
    if abs(cost + summary.get("probe_usd", 0.0) - summary["settled_usd"]) > 1e-6:
        raise ValueError("Recomputed cost differs from the settled cost")

    # The export and the A1 relabelling reproduce from the raw answers.
    v1: dict[str, int | None] = {}
    for row in _jsonl(p["results"]):
        rec = answers.get(row["job_id"])
        if row["job_id"] in v1 or row["job_id"] not in prompts:
            raise ValueError("Export rows are not one per job")
        if rec is None:
            if row.get("label") is not None:
                raise ValueError("An export row without an answer has a label")
        elif (row.get("raw_text") != rec["text"] or row.get("finish_reason") != rec["finish_reason"]
              or row.get("label") != online.parse_verdict(rec["text"])[0]):
            raise ValueError("Export row differs from its raw answer")
        v1[row["job_id"]] = row.get("label")
    if len(v1) != planned:
        raise ValueError("Export does not cover every job")
    receipt = run.read_json(p["relabel_receipt"])
    if (receipt.get("source_results_sha256") != run.sha256(p["results"])
            or receipt.get("output_sha256") != run.sha256(p["relabelled"])
            or receipt.get("code_sha256") != run.sha256(p["parser"]) or receipt.get("parser_version") != a1.PARSER_VERSION
            or receipt.get("protocol_section_sha256") != section_sha256(protocol, a1.SECTION_HEADING)):
        raise ValueError("A1 receipt does not bind the export, output, code and section 15")
    labels: dict[str, int] = {}
    rules: dict[str, int] = {}
    seen: set[str] = set()
    for row in _jsonl(p["relabelled"]):
        job_id = row["job_id"]
        text = answers[job_id]["text"] if job_id in answers else None
        label, rule = a1.parse(text)
        if job_id in seen or row.get("label") != label or row.get("parser_rule") != rule or row.get("label_v1") != v1[job_id]:
            raise ValueError("A1 label does not reproduce from the raw answer")
        if v1[job_id] is not None and label != v1[job_id]:
            raise ValueError("A1 changed a section 14 label")
        seen.add(job_id)
        rules[rule] = rules.get(rule, 0) + 1
        if label is not None:
            labels[job_id] = label
    if seen != set(prompts):
        raise ValueError("A1 output does not cover every job")

    rebuilt, report = run.prepare_metadata(run.read_jsonl(p["units"]), labels, config)
    if run.canonical_bytes(rebuilt) != run.canonical_bytes(metadata):
        raise ValueError("Final P metadata does not reproduce the archived A1 labels")
    if not report["feasible"]:
        raise ValueError("Regenerated P frame fails feasibility")
    return {"fixed_model": model, "parser_version": a1.PARSER_VERSION, "successful_prediction_records": len(labels),
            "audit": {"jobs": planned, "requests_sent": sum(sent.values()), "answers": len(answers),
                      "section14_labels": sum(v is not None for v in v1.values()), "a1_rules": dict(sorted(rules.items())),
                      "recomputed_cost_usd": round(cost, 9), "batch_jobs": len(keys)},
            "final_metadata_sha256": run.object_hash(metadata), "reference_values_opened": False}


def write_batch_files(root: Path, summary_path: Path, run_dir: Path, out: Path) -> dict:
    from src import wildifeval_external_run as run
    keys = sorted(b["key"] for b in json.loads(summary_path.read_text())["batches"])
    record = {kind: {key: {"path": run._relative(root, run_dir / folder / f"{key}.jsonl"),
                           "sha256": run.sha256(run_dir / folder / f"{key}.jsonl")} for key in keys}
              for kind, folder in (("inputs", "batch_inputs"), ("outputs", "batch_outputs"))}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def build(root: Path = ROOT) -> dict:
    """Write the batch-file manifest, P metadata and preflight, then issue the design freeze."""
    from src import wildifeval_external_run as run
    root = Path(root)
    files = write_batch_files(root, root / ARTIFACTS["summary"], root / Path(ARTIFACTS["results"]).parent,
                              root / ARTIFACTS["batch_files"])
    config = run.read_json(root / CONFIG_PATH)
    labels = {row["job_id"]: row["label"] for row in _jsonl(root / ARTIFACTS["relabelled"]) if row["label"] is not None}
    metadata, report = run.prepare_metadata(run.read_jsonl(root / ARTIFACTS["units"]), labels, config)
    if not report["feasible"]:
        raise ValueError("A1 frame fails feasibility; reference remains sealed")
    (root / PRIVATE_DIR).mkdir(parents=True, exist_ok=True)
    metadata_path, preflight_path = root / PRIVATE_DIR / "units_p.jsonl", root / DESIGN_DIR / "preflight.json"
    with metadata_path.open("x") as handle:
        handle.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata))
    with preflight_path.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    required = [entry["path"] for kind in files.values() for entry in kind.values()] + list(SENSITIVITY_FILES)
    return run.create_design_freeze(
        root, metadata_path=metadata_path.relative_to(root), preflight_path=preflight_path.relative_to(root),
        config_path=CONFIG_PATH, reference_files={p: REFERENCE_PATTERN.format(p) for p in config["producers"]},
        required_files=required, output_path=root / DESIGN_DIR / "design_freeze.json", generation_artifacts=ARTIFACTS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["freeze"])
    parser.parse_args(argv)
    receipt = build()
    frame = receipt["frame"]
    print(json.dumps({"design_freeze": str(DESIGN_DIR / "design_freeze.json"),
                      **{k: frame[k] for k in ("units", "tasks", "clusters", "pilot_clusters", "confirmation_clusters",
                                               "gates", "feasible")},
                      "audit": receipt["generation_verification"]["audit"], "reference_values_opened": False}, indent=2))


if __name__ == "__main__":
    main()
