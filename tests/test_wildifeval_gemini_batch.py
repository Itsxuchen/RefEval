"""Offline tests for the WildIFEval Gemini API batch runner with a fake backend.

No network, no SDK, no key, no reference values. Runs under pytest or directly:
    python tests/test_wildifeval_gemini_batch.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import wildifeval_gemini_batch as g  # noqa: E402
from src import wildifeval_vertex_judge as w  # noqa: E402

PRODUCERS = w.PRODUCERS


def answer_for(text: str) -> str:
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return "I cannot tell." if h % 11 == 0 else ("No." if h % 3 == 0 else "Yes")


def make_jobs(tmp: Path, per_producer: int = 40, shared: int = 3) -> Path:
    path = tmp / "jobs.jsonl"
    with path.open("w") as out:
        for p in PRODUCERS:
            for i in range(per_producer):
                who = "same output" if i < shared else f"{p} output"
                prompt = f"###Instruction:\n{who} {i}\n\n###Question:\nIs the following constraint satisfied: \"c{i}\"?\n\n###Answer:"
                out.write(json.dumps({"job_id": hashlib.sha256(f"{p}-{i}".encode()).hexdigest(), "producer": p,
                                      "unit_id": f"u{i}", "cluster_id": f"c{i}", "task_id": f"t{i}",
                                      "criterion_id": f"k{i}", "criterion_hash": "h", "criterion_index": 0,
                                      "output_hash": "o", "split": "pilot", "k": 1, "prompt": prompt,
                                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}) + "\n")
    return path


class Patch:
    def __init__(self):
        self.saved = []

    def set(self, obj, name, value):
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self.saved):
            setattr(obj, name, value)


def response(text: str, thoughts: int = 120, version: str = "gemini-3.1-flash-lite"):
    parts = [{"text": "Let me check the constraint.", "thought": True}, {"text": answer_for(text)}]
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
            "modelVersion": version, "responseId": "x",
            "usageMetadata": {"promptTokenCount": len(text) // 4, "candidatesTokenCount": 1,
                              "thoughtsTokenCount": thoughts}}


class FakeBackend:
    """Files and batch jobs in memory. A job runs for one poll, then its result file appears."""

    def __init__(self, error=lambda text, n: False, drop=lambda text, n: False, thoughts=120,
                 version="gemini-3.1-flash-lite", max_running=1000, rekey=lambda k: k):
        self.error, self.drop, self.thoughts, self.version = error, drop, thoughts, version
        self.max_running, self.rekey = max_running, rekey
        self.files: dict[str, str] = {}
        self.jobs: dict[str, dict] = {}
        self.seen: dict[str, int] = {}
        self.lose_create_reply = False
        self.uploads = 0

    def upload(self, local: Path, display_name: str) -> str:
        self.uploads += 1
        ref = f"files/in-{len(self.files) + 1}"
        self.files[ref] = Path(local).read_text()
        return ref

    def running(self):
        return sum(1 for j in self.jobs.values() if j["polls"] < 2)

    def create(self, file_ref, display_name):
        if self.running() >= self.max_running:
            raise g.RateLimited("enqueued tokens exceeded")
        name = f"batches/{len(self.jobs) + 1}"
        self.jobs[name] = {"src": file_ref, "display": display_name, "polls": 0, "result": None}
        if self.lose_create_reply:
            self.lose_create_reply = False
            raise ConnectionError("reply lost after the job was created")
        return name

    def find(self, display_name):
        for name, job in self.jobs.items():
            if job["display"] == display_name:
                return name
        return None

    def get(self, name):
        job = self.jobs[name]
        job["polls"] += 1
        if job["polls"] < 2:
            return "JOB_STATE_RUNNING", None, None
        if job["result"] is None:
            lines = []
            for raw in self.files[job["src"]].splitlines():
                item = json.loads(raw)
                text = item["request"]["contents"][0]["parts"][0]["text"]
                self.seen[text] = self.seen.get(text, 0) + 1
                if self.drop(text, self.seen[text]):
                    continue
                key = self.rekey(item["key"])
                if self.error(text, self.seen[text]):
                    lines.append({"key": key, "error": {"code": 13, "message": "Internal error"}})
                else:
                    lines.append({"key": key, "response": response(text, self.thoughts, self.version)})
            ref = f"files/out-{name.split('/')[-1]}"
            self.files[ref] = "".join(json.dumps(x) + "\n" for x in lines)
            job["result"] = ref
        return "JOB_STATE_SUCCEEDED", None, job["result"]

    def download(self, result_ref, local: Path):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(self.files[result_ref])

    def probe(self, text):
        return {"text": "OK" if "single word OK" in text else "Yes", "finish_reason": "STOP",
                "usage": {"prompt": 10, "candidates": 1, "thoughts": 100}, "model_version": self.version,
                "response_id": "r"}


def setup(tmp: Path, patch: Patch, cap: float = 5.0, per_producer: int = 40, max_shard: int = 60):
    jobs_path = make_jobs(tmp, per_producer)
    protocol = tmp / "protocol.md"
    protocol.write_text("# P\n\n" + w.SECTION_HEADING + "\n\nSynthetic section.\n")
    patch.set(w, "EXPECTED_JOBS_SHA256", w.sha256_file(jobs_path))
    patch.set(w, "EXPECTED_JOBS", per_producer * len(PRODUCERS))
    patch.set(w, "PROTOCOL_PATH", protocol)
    patch.set(w, "CANARY_PER_PRODUCER", 4)
    patch.set(g, "MAX_SHARD", max_shard)
    report, run = tmp / "report", tmp / "run"
    g.write_freeze(cap, jobs_path, report)
    return jobs_path, report, run


def context(jobs_path: Path, report: Path, run: Path, backend=None):
    frozen, jobs = g.verify_freeze(jobs_path, report)
    ledger = g.BatchLedger(run / "state" / "ledger.sqlite", frozen["cost"]["cap_usd"])
    ledger.register(jobs)
    return g.Context(ledger, jobs, w.PromptReader(jobs_path), backend or FakeBackend(), run, report,
                     log=lambda m: None)


def drive(ctx, jobs_path, limit=60):
    for _ in range(limit):
        if g.step(ctx, jobs_path):
            return True
    raise AssertionError("the run did not finish")


# ----------------------------------------------------------------------------- units

def test_request_line_and_rates():
    line = g.request_line("abc", "prompt text")
    assert line["key"] == "abc" and line["request"]["contents"][0]["parts"][0]["text"] == "prompt text"
    assert line["request"]["generation_config"] == {"temperature": 0.0, "max_output_tokens": 2048,
                                                    "thinking_config": {"thinking_level": "LOW"}}
    assert abs(g.batch_cost({"prompt": 1_000_000}) - 0.125) < 1e-12
    assert abs(g.batch_cost({"candidates": 400_000, "thoughts": 600_000}) - 0.75) < 1e-12


def test_parse_result_line():
    key, rec, err = g.parse_result_line({"key": "k1", "response": response("abcd" * 10)})
    assert key == "k1" and err is None and rec["usage"]["thoughts"] == 120 and "Let me" not in rec["text"]
    assert rec["model_version"] == "gemini-3.1-flash-lite"
    key, rec, err = g.parse_result_line({"key": "k2", "error": {"code": 13, "message": "Internal"}})
    assert key == "k2" and rec is None and "Internal" in err
    key, rec, err = g.parse_result_line({"key": "k3", "status": {"code": 8}})
    assert rec is None and "8" in err
    top = {"key": "k4", **response("zzzz")}
    key, rec, err = g.parse_result_line(top)
    assert key == "k4" and rec is not None and err is None
    blocked = {"key": "k5", "response": {"promptFeedback": {"blockReason": "SAFETY"},
                                         "usageMetadata": {"promptTokenCount": 5}}}
    key, rec, err = g.parse_result_line(blocked)
    assert err is None and rec["text"] is None and rec["usage"]["prompt"] == 5


def test_shards_respect_both_caps():
    with tempfile.TemporaryDirectory() as d:
        jobs = w.load_jobs(make_jobs(Path(d), 40))
        size = sum(j.length for j in jobs)
        for max_size, max_bytes in ((7, 10**9), (1000, size // 5), (50, size // 3)):
            shards = g.make_shards(jobs, max_size, max_bytes)
            assert sorted(j.job_id for s in shards for j in s) == sorted(j.job_id for j in jobs)
            assert all(len(s) <= max_size and sum(j.length for j in s) <= max_bytes for s in shards)


def test_key_loading():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "k.rtf"
        fake = "AIza" + "x" * 35
        f.write_text('{\\rtf1 export OPENAI_API_KEY="sk-proj-abc"\\\nexport GEMINI_API_KEY="' + fake + '"\\\n}')
        old = os.environ.pop("GEMINI_API_KEY", None)
        try:
            assert g.load_api_key(f) == fake
            os.environ["GEMINI_API_KEY"] = "AIza" + "y" * 35
            assert g.load_api_key(f) == "AIza" + "y" * 35
            os.environ["GEMINI_API_KEY"] = "not-a-key"
            assert g.load_api_key(f) == fake
            f.write_text("nothing here")
            try:
                g.load_api_key(f)
            except g.StopRun as exc:
                assert "AIza" not in str(exc)
            else:
                raise AssertionError("a missing key was accepted")
        finally:
            os.environ.pop("GEMINI_API_KEY", None)
            if old is not None:
                os.environ["GEMINI_API_KEY"] = old


# ----------------------------------------------------------------------------- workflow

def test_end_to_end_with_rate_limit_and_retries():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run = setup(Path(d), patch)

            def error(text, n):    # one prompt fails once, another fails every time
                return ("output 7" in text and n == 1) or ("output 11\n" in text and "llama3.3" in text)

            def drop(text, n):     # one result line is missing the first time
                return "output 9" in text and "qwen" in text and n == 1

            fb = FakeBackend(error=error, drop=drop, max_running=2)
            ctx = context(jobs_path, report, run, fb)
            assert drive(ctx, jobs_path)
            summary = json.loads((report / "summary.json").read_text())
            rows = ctx.ledger.job_states()
            failed = [r for r in rows.values() if r["state"] == "error"]
            assert len(failed) == 1 and failed[0]["submissions"] == 1 + g.MAX_RETRY_ROUNDS
            assert summary["states"] == {"done": 199, "error": 1}
            assert ctx.ledger.meta_get("max_active") == "2"          # learned from the refused creation
            displays = [j["display"] for j in fb.jobs.values()]
            assert len(displays) == len(set(displays)) == len(ctx.ledger.batches())
            assert fb.uploads == len(ctx.ledger.batches())           # a refused creation reuses its upload
            assert {b["phase"] for b in ctx.ledger.batches()} == {"canary", "full", "retry"}
            settled = sum(r["cost_usd"] for r in rows.values())
            assert abs(summary["settled_usd"] - settled - summary["probe_usd"]) < 1e-9
            results = [json.loads(l) for l in (run / "results.jsonl").open()]
            assert [r["job_id"] for r in results] == [json.loads(l)["job_id"] for l in jobs_path.open()]
            for r in results:
                if r["status"] in ("success", "unparseable"):
                    assert r["label"] == w.parse_verdict(r["raw_text"])[0] and "Let me" not in r["raw_text"]
                else:
                    assert r["label"] is None and r["status"] == "no_response"
            assert "raw_text" not in (report / "summary.json").read_text()
            n = len(fb.jobs)
            assert g.step(ctx, jobs_path) and len(fb.jobs) == n
    finally:
        patch.undo()


def test_gate_blocks_then_recorded_cap_raise():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run = setup(Path(d), patch, cap=0.2)
            fb = FakeBackend(thoughts=2000)
            ctx = context(jobs_path, report, run, fb)
            try:
                drive(ctx, jobs_path)
            except g.StopRun as exc:
                assert "projected" in str(exc)
            else:
                raise AssertionError("the gate did not stop the run")
            assert {b["phase"] for b in ctx.ledger.batches()} == {"canary"}
            assert not json.loads((report / "fee_receipt.json").read_text())["passed"]
            w.raise_cap(1.0, "canary projection above cap", report)
            ctx = context(jobs_path, report, run, fb)
            assert drive(ctx, jobs_path) and ctx.ledger.meta_get("gate") == "passed"
    finally:
        patch.undo()


def test_submission_respects_cap():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run = setup(Path(d), patch, cap=0.01)
            ctx = context(jobs_path, report, run)
            try:
                g.step(ctx, jobs_path)
            except g.StopRun as exc:
                assert "could exceed" in str(exc)
            else:
                raise AssertionError("the canary was submitted above the cap")
            assert ctx.backend.jobs == {} and ctx.backend.uploads == 0
    finally:
        patch.undo()


def test_lost_create_reply_is_adopted():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run = setup(Path(d), patch)
            fb = FakeBackend()
            fb.lose_create_reply = True
            ctx = context(jobs_path, report, run, fb)
            try:
                g.step(ctx, jobs_path)
            except ConnectionError:
                pass
            assert len(fb.jobs) == 1
            ctx = context(jobs_path, report, run, fb)
            assert drive(ctx, jobs_path)
            displays = [j["display"] for j in fb.jobs.values()]
            assert len(displays) == len(set(displays))
    finally:
        patch.undo()


def test_unmatched_key_stops_without_settling():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run = setup(Path(d), patch)
            fb = FakeBackend(rekey=lambda k: "request-" + k[:6])
            ctx = context(jobs_path, report, run, fb)
            try:
                drive(ctx, jobs_path)
            except g.StopRun as exc:
                assert "did not match" in str(exc)
            else:
                raise AssertionError("unknown result keys were accepted")
            assert ctx.ledger.settled() == float(ctx.ledger.meta_get("probe_usd"))
    finally:
        patch.undo()


def test_wrong_model_version_stops():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run = setup(Path(d), patch)
            fb = FakeBackend()
            ctx = context(jobs_path, report, run, fb)
            g.probe(ctx)
            fb.version = "gemini-2.5-flash-lite"
            try:
                drive(ctx, jobs_path)
            except g.StopRun as exc:
                assert "model versions" in str(exc)
            else:
                raise AssertionError("a different model was accepted")
            assert not ctx.ledger.batches("full")
    finally:
        patch.undo()


def test_freeze_binds_plan():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, _ = setup(Path(d), patch)
            frozen = json.loads((report / "freeze.json").read_text())
            assert frozen["config"]["mode"] == "gemini-api-batch" and frozen["config"]["price_out_per_m"] == 0.75
            assert sum(frozen["shard_plan"]["full_sizes"]) == 180
            patch.set(g, "MAX_SHARD", 1000)
            try:
                g.verify_freeze(jobs_path, report)
            except g.StopRun as exc:
                assert "shard_plan" in str(exc)
            else:
                raise AssertionError("a changed shard plan passed verification")
    finally:
        patch.undo()


def test_cmd_run_resumes_across_calls():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run = setup(Path(d), patch)
            fb = FakeBackend()
            patch.set(g, "GeminiBatchBackend", lambda key: fb)
            patch.set(g, "load_api_key", lambda path=None: "AIza" + "z" * 35)
            args = types.SimpleNamespace(jobs=jobs_path, run_dir=run, report_dir=report, wait=False)
            for _ in range(20):
                g.cmd_run(args)
                if (report / "summary.json").exists():
                    break
            assert json.loads((report / "summary.json").read_text())["states"] == {"done": 200}
            for p in list(run.rglob("*")) + list(report.rglob("*")):
                if p.is_file() and p.suffix in (".json", ".jsonl", ".txt"):
                    assert "AIza" not in p.read_text()
    finally:
        patch.undo()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
