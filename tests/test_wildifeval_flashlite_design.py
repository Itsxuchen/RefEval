"""Synthetic Gemini-archive checks for the section 14 step 6 design freeze. No network, no real data."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src import wildifeval_external_run as run
from src import wildifeval_flashlite_design as flashlite
from src import wildifeval_gemini_batch as batch
from src import wildifeval_parser_v2 as a1
from src import wildifeval_vertex_judge as online
from tests.test_wildifeval_external_run import config, dump, synthetic_units

MODEL = batch.CONFIG["model"]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def answer(jid: str, label: int) -> str:
    if label:
        return "Yes"
    return "No." if int(sha(jid), 16) % 2 else "The constraint is **not satisfied**."   # A1 reads the second


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def archive(root: Path) -> dict:
    """Build a complete synthetic archive under root and return the generation artifacts."""
    for rel in set(run.CODE_PATHS) | {"src/wildifeval_gemini_batch.py", "src/wildifeval_vertex_judge.py",
                                      "src/wildifeval_parser_v2.py"}:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(run.__file__).parent / Path(rel).name, root / rel)
    (root / "amendment.md").write_text(f"# P\n\n{online.SECTION_HEADING}\n\nsection 14\n\n{a1.SECTION_HEADING}\n\nsection 15\n")
    cfg = config()
    dump(root / "config.json", cfg)
    units, labels, references = synthetic_units()
    for producer, raw in references.items():
        dump(root / f"{producer}.json", raw)
    jobs = []
    for unit in units:
        for jid in unit["job_ids"]:
            prompt = f"Synthetic prompt {jid}\n\n###Answer:"
            jobs.append({"job_id": jid, "unit_id": unit["unit_id"], "producer": unit["producer"], "prompt": prompt,
                         "prompt_sha256": sha(prompt), "task_id": unit["task_id"], "cluster_id": unit["cluster_id"],
                         "split": "pilot"})
    write_jsonl(root / "jobs.jsonl", jobs)
    write_jsonl(root / "units.jsonl", units)
    dump(root / "source_manifest.json", {"files": [
        {"path": f"predictions/llm_aaj_scores/llm-aaj-deepseek-v3.{p}-0shot-wild-if-eval.json",
         "sha256": run.sha256(root / f"{p}.json")} for p in cfg["producers"]]})
    dump(root / "job_manifest.json", {"counts": {"jobs": len(jobs)},
                                      "files": {n: {"sha256": run.sha256(root / n)} for n in ("jobs.jsonl", "units.jsonl")},
                                      "source_inputs": {"source_manifest_sha256": run.sha256(root / "source_manifest.json")}})
    run_dir, report = root / "run", root / "report"
    write_jsonl(run_dir / "batch_inputs" / "full-r0-s00.jsonl", [batch.request_line(j["job_id"], j["prompt"]) for j in jobs])
    outputs, results, cost = [], [], 0.0
    for j in jobs:
        text = answer(j["job_id"], labels[j["job_id"]])
        usage = {"promptTokenCount": 30, "candidatesTokenCount": 2, "thoughtsTokenCount": 50}
        outputs.append({"key": j["job_id"], "response": {
            "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "modelVersion": MODEL, "usageMetadata": usage}})
        cost += batch.batch_cost({"prompt": 30, "candidates": 2, "thoughts": 50})
        results.append({**{k: j[k] for k in ("job_id", "unit_id", "producer", "task_id", "cluster_id", "split")},
                        "parser_version": online.PARSER_VERSION, "label": online.parse_verdict(text)[0],
                        "status": "success", "raw_text": text, "finish_reason": "STOP", "model_version": MODEL})
    write_jsonl(run_dir / "batch_outputs" / "full-r0-s00.jsonl", outputs)
    write_jsonl(run_dir / "results.jsonl", results)
    report.mkdir()
    dump(report / "freeze.json", {
        "protocol_version": flashlite.GEMINI_PROTOCOL, "status": "frozen", "jobs": len(jobs),
        "jobs_sha256": run.sha256(root / "jobs.jsonl"), "parser_version": online.PARSER_VERSION, "config": batch.CONFIG,
        "code_sha256": {"batch": run.sha256(root / "src/wildifeval_gemini_batch.py"),
                        "shared": run.sha256(root / "src/wildifeval_vertex_judge.py")},
        "protocol_section_sha256": flashlite.section_sha256(root / "amendment.md", online.SECTION_HEADING),
        "reference_values_opened": False, "cost": {"cap_usd": 100.0}})
    dump(report / "fee_receipt.json", {"passed": True, "cap_usd": 100.0})
    dump(report / "summary.json", {"protocol_version": flashlite.GEMINI_PROTOCOL, "states": {"done": len(jobs)},
                                   "model_versions": [MODEL], "settled_usd": cost, "probe_usd": 0.0,
                                   "batches": [{"key": "full-r0-s00"}], "results_sha256": run.sha256(run_dir / "results.jsonl")})
    a1.relabel(run_dir, report, root / "amendment.md", expected_rows=len(jobs))
    # relabel binds the live parser file; the archived copy is identical
    flashlite.write_batch_files(root, report / "summary.json", run_dir, report / "batch_files.json")
    artifacts = {"jobs": "jobs.jsonl", "units": "units.jsonl", "job_manifest": "job_manifest.json",
                 "source_manifest": "source_manifest.json", "receipt": "report/freeze.json", "fee": "report/fee_receipt.json",
                 "summary": "report/summary.json", "results": "run/results.jsonl",
                 "batch_runner": "src/wildifeval_gemini_batch.py", "shared_runner": "src/wildifeval_vertex_judge.py",
                 "batch_files": "report/batch_files.json", "relabel_receipt": "report/parser_v2_receipt.json",
                 "relabelled": "run/results_parser_v2.jsonl", "parser": "src/wildifeval_parser_v2.py"}
    a1_labels = {r["job_id"]: r["label"] for r in run.read_jsonl(run_dir / "results_parser_v2.jsonl") if r["label"] is not None}
    metadata, frame = run.prepare_metadata(units, a1_labels, cfg)
    write_jsonl(root / "units_p.jsonl", metadata)
    dump(root / "preflight.json", frame)
    return {"artifacts": artifacts, "metadata": metadata, "cfg": cfg, "labels": labels}


def freeze(root: Path, artifacts: dict, name: str = "design.json") -> dict:
    return run.create_design_freeze(root, metadata_path=root / "units_p.jsonl", preflight_path=root / "preflight.json",
                                    config_path=root / "config.json", reference_files={p: f"{p}.json" for p in ("a", "b")},
                                    required_files=[], output_path=root / name, generation_artifacts=artifacts)


def test_gemini_archive_freezes_and_reaches_the_sealed_pilot(tmp_path):
    built = archive(tmp_path)
    receipt = freeze(tmp_path, built["artifacts"])
    audit = receipt["generation_verification"]["audit"]
    assert audit["jobs"] == audit["requests_sent"] == audit["answers"] == 80
    assert audit["a1_rules"]["verdict-phrase"] > 0 and audit["a1_rules"]["v1-yes-no"] > 0
    assert receipt["generation_verification"]["successful_prediction_records"] == 80
    assert set(built["artifacts"].values()) <= set(receipt["frozen_files"])
    assert "src/wildifeval_flashlite_design.py" in receipt["frozen_files"]
    run.validate_design(tmp_path, tmp_path / "design.json")
    # A1 recovers the phrase answers, so P equals the synthetic labels
    for row in built["metadata"]:
        assert row["p"] == [built["labels"][j] for j in row["job_ids"]]
    result = run.run_pilot(tmp_path, tmp_path / "design.json", tmp_path / "pilot")
    assert result["stage"] == "prediction_freeze" and result["confirmation_reference_bits_revealed"] == 0


def rewrite(path: Path, change) -> None:
    rows = run.read_jsonl(path)
    change(rows)
    write_jsonl(path, rows)


@pytest.mark.parametrize("case, match", [
    ("answer_text", "batch file changed"),
    ("relabel_row", "does not bind"),
    ("relabel_row_rehashed", "does not reproduce"),
    ("prompt", "frozen prompt"),
    ("hand_edited_p", "does not reproduce"),
    ("other_model", "another model"),
])
def test_any_break_in_the_chain_blocks_the_freeze(tmp_path, case, match):
    built = archive(tmp_path)
    report, run_dir = tmp_path / "report", tmp_path / "run"
    if case == "answer_text":
        rewrite(run_dir / "batch_outputs" / "full-r0-s00.jsonl",
                lambda rows: rows[0]["response"]["candidates"][0]["content"]["parts"][0].update(text="No"))
    elif case == "relabel_row":
        rewrite(run_dir / "results_parser_v2.jsonl", lambda rows: rows[0].update(label=1 - (rows[0]["label"] or 0)))
    elif case == "relabel_row_rehashed":
        rewrite(run_dir / "results_parser_v2.jsonl", lambda rows: rows[0].update(label=1 - (rows[0]["label"] or 0)))
        receipt = json.loads((report / "parser_v2_receipt.json").read_text())
        dump(report / "parser_v2_receipt.json", {**receipt, "output_sha256": run.sha256(run_dir / "results_parser_v2.jsonl")})
    elif case in ("prompt", "other_model"):
        if case == "prompt":
            rewrite(run_dir / "batch_inputs" / "full-r0-s00.jsonl",
                    lambda rows: rows[0]["request"]["contents"][0]["parts"][0].update(text="edited prompt"))
        else:
            rewrite(run_dir / "batch_outputs" / "full-r0-s00.jsonl",
                    lambda rows: rows[0]["response"].update(modelVersion="gemini-other"))
        (report / "batch_files.json").unlink()
        flashlite.write_batch_files(tmp_path, report / "summary.json", run_dir, report / "batch_files.json")
    else:
        rewrite(tmp_path / "units_p.jsonl", lambda rows: rows[0]["p"].__setitem__(0, 1 - rows[0]["p"][0]))
        dump(tmp_path / "preflight.json", run._frame_report(run.read_jsonl(tmp_path / "units_p.jsonl"), built["cfg"]))
    with pytest.raises(ValueError, match=match):
        freeze(tmp_path, built["artifacts"])
    assert not (tmp_path / "design.json").exists()
