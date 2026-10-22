"""Offline tests for the section 18 replication check. No network, no reference values."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from src import wildifeval_run_replication as rep
from src import wildifeval_vertex_batch as vertex
from src import wildifeval_vertex_judge as online


class FakeStore:
    def __init__(self, blobs: dict[str, str]):
        self.blobs, self.downloads = blobs, 0

    def list(self, prefix):
        return [u for u in self.blobs if u.startswith(prefix)]

    def download(self, uri, local):
        self.downloads += 1
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(self.blobs[uri])


def line(prompt: str, text: str | None) -> str:
    out = {"request": vertex.request_line(prompt)["request"], "status": "",
           "response": {"candidates": ([{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}] if text else []),
                        "modelVersion": "gemini-3.1-flash-lite",
                        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1, "thoughtsTokenCount": 5}}}
    return json.dumps(out) + "\n"


def setup(tmp: Path, vertex_texts: list[str | None], p_texts: list[str | None]):
    jobs, p_rows, p_results, blobs = [], [], [], {}
    for i, (vt, pt) in enumerate(zip(vertex_texts, p_texts)):
        prompt = f"prompt {i}\n\n###Answer:"
        jid = f"j{i}"
        jobs.append({"job_id": jid, "producer": "a", "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        label, rule = rep.a1.parse(pt)
        p_rows.append({"job_id": jid, "unit_id": f"u{i // 2}", "task_id": f"t{i // 2}", "producer": "a", "label": label,
                       "parser_rule": rule, "label_v1": online.parse_verdict(pt)[0]})
        p_results.append({"job_id": jid, "raw_text": pt})
        key = "s0" if i < 3 else "s1"
        uri = f"gs://b/out/{key}/prediction-model-x/predictions.jsonl"
        blobs[uri] = blobs.get(uri, "") + line(prompt, vt)
    (tmp / "jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    (tmp / "p.jsonl").write_text("".join(json.dumps(r) + "\n" for r in p_rows))
    (tmp / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in p_results))
    vdir = tmp / "vertex"
    (vdir / "state").mkdir(parents=True)
    con = sqlite3.connect(vdir / "state" / "ledger.sqlite")
    con.execute("CREATE TABLE batches (key TEXT, n INTEGER, name TEXT, output_uri TEXT)")
    con.execute("CREATE TABLE jobs (job_id TEXT, batch_key TEXT)")
    con.executemany("INSERT INTO batches VALUES (?, ?, ?, ?)", [("s0", 3, "n0", "gs://b/out/s0"), ("s1", 3, "n1", "gs://b/out/s1")])
    con.executemany("INSERT INTO jobs VALUES (?, ?)", [(f"j{i}", "s0" if i < 3 else "s1") for i in range(len(jobs))])
    con.commit(); con.close()
    return vdir, FakeStore(blobs)


def test_replication_compares_labels_without_changing_p(tmp_path):
    vtexts = ["Yes", "No", "The constraint is satisfied.", None, "Yes", "The constraint is partially satisfied."]
    ptexts = ["Yes", "Yes", "No", "Yes", "Yes", "No"]
    vdir, store = setup(tmp_path, vtexts, ptexts)
    before = (tmp_path / "p.jsonl").read_bytes()
    receipt = rep.run(lambda: store, vdir, tmp_path / "p.jsonl", tmp_path / "results.jsonl", tmp_path / "report",
                      tmp_path / "jobs.jsonl")
    assert receipt["criterion_pairs"] == {"agree": 2, "p_pass_vertex_fail": 1, "p_fail_vertex_pass": 1, "vertex_missing": 2}
    assert receipt["identical_answer_text"] == 2
    assert receipt["unit_verdicts_where_both_complete"] == {"p_pass_vertex_fail": 1}
    assert receipt["vertex_tasks_complete"] == 1 and receipt["tasks_complete_in_both"] == 1
    assert (tmp_path / "p.jsonl").read_bytes() == before
    text = (tmp_path / "report" / "replication_receipt.json").read_text()
    assert "partially" not in text and receipt["reference_values_opened"] is False
    downloads = store.downloads
    rep.run(lambda: store, vdir, tmp_path / "p.jsonl", tmp_path / "results.jsonl", tmp_path / "report", tmp_path / "jobs.jsonl")
    assert store.downloads == downloads          # local copies are reused


def test_unmatched_or_duplicate_output_stops(tmp_path):
    vdir, store = setup(tmp_path, ["Yes"] * 6, ["Yes"] * 6)
    uri = "gs://b/out/s0/prediction-model-x/predictions.jsonl"
    store.blobs[uri] += store.blobs[uri].splitlines(keepends=True)[0]       # a job answered twice
    with pytest.raises(online.StopRun):
        rep.run(lambda: store, vdir, tmp_path / "p.jsonl", tmp_path / "results.jsonl", tmp_path / "report",
                tmp_path / "jobs.jsonl")
