"""Offline tests for repair sensitivity R1 (protocol section 16). No network, no SDK, no reference values.

Runs under pytest or directly: python tests/test_wildifeval_repair_batch.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import wildifeval_parser_v2 as a1  # noqa: E402
from src import wildifeval_repair_batch as rb  # noqa: E402
from src import wildifeval_vertex_judge as w  # noqa: E402
from tests.test_wildifeval_gemini_batch import FakeBackend, Patch, answer_for, make_jobs  # noqa: E402


def rule_for(i: int) -> tuple[str, int | None]:
    if i % 5 == 0:
        return "missing-partial", None
    if i % 5 == 1:
        return "missing-empty", None          # a provider block: never re-sent
    if i % 7 == 2:
        return "missing-abstain", None
    return "v1-yes-no", i % 2


def setup(tmp: Path, per_producer: int = 10) -> dict[str, Path]:
    jobs_path = make_jobs(tmp, per_producer)
    src, src_report, run, report = tmp / "src", tmp / "src_report", tmp / "run", tmp / "report"
    src.mkdir(); src_report.mkdir()
    with jobs_path.open() as handle, (src / "results_parser_v2.jsonl").open("w") as out:
        for n, line in enumerate(handle):
            job = json.loads(line)
            rule, label = rule_for(n)
            out.write(json.dumps({k: job[k] for k in ("job_id", "producer", "task_id", "cluster_id", "unit_id", "split")}
                                 | {"label": label, "parser_rule": rule, "status": "success"}) + "\n")
    (src_report / "parser_v2_receipt.json").write_text(
        json.dumps({"output_sha256": w.sha256_file(src / "results_parser_v2.jsonl")}))
    protocol = tmp / "protocol.md"
    protocol.write_text(f"# P\n\n{rb.SECTION_HEADING}\n\nSynthetic section.\n\n## 17. later\n")
    return {"jobs_path": jobs_path, "source_dir": src, "source_report_dir": src_report, "protocol_path": protocol,
            "run": run, "report": report}


def start(p: dict[str, Path], backend: FakeBackend, cap: float = 5.0):
    kwargs = {k: p[k] for k in ("jobs_path", "source_dir", "source_report_dir", "protocol_path")}
    rb.write_freeze(cap, p["report"], **kwargs)
    frozen, jobs = rb.verify_freeze(p["report"], **kwargs)
    ctx = rb.context(frozen, jobs, backend, p["run"], p["report"], jobs_path=p["jobs_path"])
    ctx.log = lambda m: None
    rows = rb.a1_rows(p["source_dir"], p["source_report_dir"])
    for _ in range(40):
        if rb.step(ctx, rows):
            return ctx, jobs, rows
    raise AssertionError("the repair run did not finish")


def test_repair_prompt_adds_one_line():
    prompt = "Below is ...\n\n###Question:\nIs the following constraint satisfied: \"c\"?\n\n###Answer:"
    out = rb.repair_prompt(prompt)
    assert out == prompt[: -len(rb.ANSWER_TAIL)] + "\nAnswer with Yes or No.\n\n###Answer:"
    try:
        rb.repair_prompt("no marker")
        raise AssertionError("a prompt without the answer marker was accepted")
    except rb.StopRun:
        pass


def test_end_to_end_repairs_only_unblocked_missing_answers():
    with tempfile.TemporaryDirectory() as d:
        p = setup(Path(d))
        fb = FakeBackend()
        ctx, jobs, rows = start(p, fb)
        expected = {r["job_id"] for r in rows if r["parser_rule"] in rb.REPAIR_RULES}
        assert {j.job_id for j in jobs} == expected and expected
        sent = [json.loads(l) for ref, text in fb.files.items() if ref.startswith("files/in-")
                for l in text.splitlines()]
        assert {s["key"] for s in sent} == expected and len(sent) == len(expected)
        for s in sent:
            text = s["request"]["contents"][0]["parts"][0]["text"]
            assert text.endswith("\nAnswer with Yes or No.\n\n###Answer:") and text.count(rb.INSTRUCTION) == 1
        out = [json.loads(l) for l in (p["run"] / "results_repaired.jsonl").open()]
        assert [r["job_id"] for r in out] == [r["job_id"] for r in rows]
        by_key = {s["key"]: s["request"]["contents"][0]["parts"][0]["text"] for s in sent}
        for src, r in zip(rows, out):
            assert r["label_a1"] == src["label"]
            if src["job_id"] in expected:
                assert r["label"] == r["repair_label"] == a1.parse(answer_for(by_key[src["job_id"]]))[0]
            else:
                assert r["label"] == src["label"] and "repair_label" not in r
        summary_text = (p["report"] / "summary.json").read_text()
        summary = json.loads(summary_text)
        assert "cannot tell" not in summary_text and "repair_text" not in summary_text
        assert sum(sum(v.values()) for v in summary["repair_outcomes"].values()) == len(expected)
        assert summary["labels"]["a1"]["missing"] >= summary["labels"]["repaired"]["missing"]
        assert summary["results_sha256"] == w.sha256_file(p["run"] / "results_repaired.jsonl")


def test_request_without_response_is_resubmitted_under_its_own_phase():
    with tempfile.TemporaryDirectory() as d:
        p = setup(Path(d))
        fb = FakeBackend(error=lambda text, n: "output 5\n" in text and n == 1)
        ctx, jobs, _ = start(p, fb)
        phases = {b["phase"] for b in ctx.ledger.batches()}
        assert phases == {rb.PHASE, rb.RETRY_PHASE}
        assert all(r["state"] == "done" for r in ctx.ledger.job_states().values())
        assert all(job["display"].endswith(("repair-r0-s00", "repair-retry-r1-s00")) for job in fb.jobs.values())


def test_freeze_detects_a_changed_instruction():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            p = setup(Path(d))
            kwargs = {k: p[k] for k in ("jobs_path", "source_dir", "source_report_dir", "protocol_path")}
            rb.write_freeze(5.0, p["report"], **kwargs)
            patch.set(rb, "INSTRUCTION", "Answer Yes.")
            try:
                rb.verify_freeze(p["report"], **kwargs)
                raise AssertionError("a changed instruction passed the freeze check")
            except rb.StopRun:
                pass
    finally:
        patch.undo()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
