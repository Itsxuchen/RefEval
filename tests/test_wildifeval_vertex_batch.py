"""Offline tests for the WildIFEval Vertex batch runner with a fake Cloud Storage and a fake batch API.

No network, no SDK, no reference values. Runs under pytest or directly:
    python tests/test_wildifeval_vertex_batch.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import wildifeval_vertex_batch as b  # noqa: E402
from src import wildifeval_vertex_judge as w  # noqa: E402

PRODUCERS = w.PRODUCERS


def answer_for(text: str) -> str:
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return "I cannot tell." if h % 11 == 0 else ("No." if h % 3 == 0 else "Yes")


def make_jobs(tmp: Path, per_producer: int = 40, shared: int = 3) -> Path:
    """The first `shared` criteria have the same prompt for every producer, as identical outputs do in WildIFEval."""
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


class FakeStore:
    def __init__(self, root: Path):
        self.root = root
        self.uploads = []

    def _local(self, uri: str) -> Path:
        bucket, path = b.split_uri(uri)
        return self.root / bucket / path

    def ensure_bucket(self, bucket_uri, location, create):
        (self.root / b.split_uri(bucket_uri)[0]).mkdir(parents=True, exist_ok=True)
        return location.upper()

    def upload(self, local: Path, uri: str) -> None:
        dest = self._local(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        self.uploads.append(uri)

    def list(self, prefix_uri: str) -> list[str]:
        bucket, path = b.split_uri(prefix_uri)
        base = self.root / bucket
        if not base.exists():
            return []
        return sorted(f"gs://{bucket}/{p.relative_to(base).as_posix()}" for p in base.rglob("*")
                      if p.is_file() and p.relative_to(base).as_posix().startswith(path))

    def download(self, uri: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._local(uri), local)


def response(text: str, answer: str | None = None, thoughts: int = 120, version: str = "gemini-3.1-flash-lite"):
    parts = [{"text": "Let me check the constraint.", "thought": True}, {"text": answer or answer_for(text)}]
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
            "modelVersion": version,
            "usageMetadata": {"promptTokenCount": len(text) // 4, "candidatesTokenCount": 1,
                              "thoughtsTokenCount": thoughts, "trafficType": "BATCH"}}


class FakeBatches:
    """Each job runs for one poll and then writes a predictions file under its output prefix."""

    def __init__(self, store: FakeStore, error=lambda text, n: False, drop=lambda text, n: False,
                 thoughts: int = 120, version: str = "gemini-3.1-flash-lite", mangle=lambda text: text):
        self.store, self.error, self.drop, self.thoughts, self.version, self.mangle = store, error, drop, thoughts, version, mangle
        self.jobs: dict[str, dict] = {}
        self.seen: dict[str, int] = {}
        self.fail_create_after_register = False

    def create(self, src, dest, display_name):
        name = f"projects/p/locations/global/batchPredictionJobs/{len(self.jobs) + 1}"
        self.jobs[name] = {"src": src, "dest": dest, "display": display_name, "polls": 0}
        if self.fail_create_after_register:
            self.fail_create_after_register = False
            raise ConnectionError("response lost after the job was created")
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
            return "JOB_STATE_RUNNING", None
        if job["polls"] == 2:
            lines = []
            for raw in self.store._local(job["src"]).read_text().splitlines():
                req = json.loads(raw)["request"]
                text = req["contents"][0]["parts"][0]["text"]
                self.seen[text] = self.seen.get(text, 0) + 1
                if self.drop(text, self.seen[text]):
                    continue
                echo = json.loads(json.dumps(req))
                echo["contents"][0]["parts"][0]["text"] = self.mangle(text)
                if self.error(text, self.seen[text]):
                    lines.append({"status": "Internal error occurred", "processed_time": "t", "request": echo})
                else:
                    lines.append({"status": "", "processed_time": "t", "request": echo,
                                  "response": response(text, thoughts=self.thoughts, version=self.version)})
            tmp = self.store.root / "_tmp.jsonl"
            tmp.write_text("".join(json.dumps(x) + "\n" for x in lines))
            self.store.upload(tmp, job["dest"] + "/prediction-model-2026-10-21T00:00:00Z/predictions.jsonl")
        return "JOB_STATE_SUCCEEDED", None

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
    patch.set(b, "MAX_SHARD", max_shard)
    report, run = tmp / "report", tmp / "run"
    b.write_freeze(cap, jobs_path, report)
    return jobs_path, report, run


def context(tmp: Path, jobs_path: Path, report: Path, run: Path, batches=None, logs=None):
    frozen, jobs = b.verify_freeze(jobs_path, report)
    ledger = b.BatchLedger(run / "state" / "ledger.sqlite", frozen["cost"]["cap_usd"])
    ledger.register(jobs)
    store = batches.store if batches else FakeStore(tmp / "gcs")
    batches = batches or FakeBatches(store)
    return b.Context(ledger, jobs, w.PromptReader(jobs_path), batches, store, "gs://bucket", run, report,
                     log=(logs.append if logs is not None else (lambda m: None)))


def drive(ctx, jobs_path, limit=40):
    for _ in range(limit):
        if b.step(ctx, jobs_path):
            return True
    raise AssertionError("the run did not finish")


# ----------------------------------------------------------------------------- units

def test_shards_keep_identical_prompts_apart():
    with tempfile.TemporaryDirectory() as d:
        jobs = w.load_jobs(make_jobs(Path(d), 40, shared=5))
        for size in (7, 60, 1000):
            shards = b.make_shards(jobs, size)
            assert sorted(j.job_id for s in shards for j in s) == sorted(j.job_id for j in jobs)
            for s in shards:
                assert len(s) <= size and len({j.prompt_sha256 for j in s}) == len(s)
            assert b.plan_sha256(shards) == b.plan_sha256(b.make_shards(jobs, size))
        assert len(b.make_shards(jobs, 1000)) == len(PRODUCERS)   # five copies of a prompt need five shards


def test_parse_output_line():
    text = "###Question:\nIs it?\n\n###Answer:"
    sha = hashlib.sha256(text.encode()).hexdigest()
    req = b.request_line(text)["request"]
    got_sha, rec, err = b.parse_output_line({"status": "", "request": req, "response": response(text, "No.")})
    assert got_sha == sha and err is None and rec["text"] == "No." and rec["usage"]["thoughts"] == 120
    assert rec["traffic_type"] == "BATCH" and rec["model_version"] == "gemini-3.1-flash-lite"
    assert w.parse_verdict(rec["text"])[0] == 0
    _, rec, err = b.parse_output_line({"status": "Quota exceeded", "request": req})
    assert rec is None and "Quota" in err
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 5}}
    _, rec, err = b.parse_output_line({"status": "", "request": req, "response": blocked})
    assert err is None and rec["text"] is None and rec["usage"]["prompt"] == 5
    snake = {"candidates": [{"content": {"parts": [{"text": "Yes"}]}, "finish_reason": "STOP"}],
             "usage_metadata": {"prompt_token_count": 3, "candidates_token_count": 1, "thoughts_token_count": 2},
             "model_version": "gemini-3.1-flash-lite"}
    _, rec, _ = b.parse_output_line({"request": req, "response": snake})
    assert rec["usage"] == {"prompt": 3, "candidates": 1, "thoughts": 2} and rec["finish_reason"] == "STOP"
    line = b.request_line(text)
    assert line["request"]["generationConfig"] == {"temperature": 0.0, "maxOutputTokens": 2048,
                                                   "thinkingConfig": {"thinkingLevel": "LOW"}}


def test_batch_rates():
    assert abs(b.batch_cost({"prompt": 1_000_000}) - 0.125) < 1e-12
    assert abs(b.batch_cost({"candidates": 400_000, "thoughts": 600_000}) - 0.75) < 1e-12


# ----------------------------------------------------------------------------- workflow

def test_end_to_end_with_retries():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            jobs_path, report, run = setup(tmp, patch)
            store = FakeStore(tmp / "gcs")

            def error(text, n):    # one prompt fails once, another fails every time
                return ("output 7" in text and n == 1) or "output 11\n" in text and "llama3.3" in text

            def drop(text, n):     # one prompt is missing from the first output file
                return "output 9" in text and "qwen" in text and n == 1

            fb = FakeBatches(store, error=error, drop=drop)
            logs = []
            ctx = context(tmp, jobs_path, report, run, fb, logs)
            assert drive(ctx, jobs_path)
            summary = json.loads((report / "summary.json").read_text())
            rows = ctx.ledger.job_states()
            always = [r for r in rows.values() if r["state"] == "error"]
            assert len(always) == 1 and always[0]["submissions"] == 1 + b.MAX_RETRY_ROUNDS
            assert summary["states"] == {"done": 199, "error": 1}
            once = [j for j in ctx.jobs if "output 7" in ctx.reader.prompt(j) or
                    ("output 9" in ctx.reader.prompt(j) and j.producer == "qwen2.5-72b")]
            assert all(rows[j.job_id]["state"] == "done" and rows[j.job_id]["submissions"] == 2 for j in once)
            # every created job belongs to a stored plan shard, and no shard was created twice
            displays = [j["display"] for j in fb.jobs.values()]
            assert len(displays) == len(set(displays)) == len(ctx.ledger.batches())
            assert {bt["phase"] for bt in ctx.ledger.batches()} == {"canary", "full", "retry"}
            assert ctx.ledger.meta_get("gate") == "passed"
            settled = sum(r["cost_usd"] for r in rows.values())
            assert abs(summary["settled_usd"] - settled - summary["probe_usd"]) < 1e-9
            results = [json.loads(l) for l in (run / "results.jsonl").open()]
            assert [r["job_id"] for r in results] == [json.loads(l)["job_id"] for l in jobs_path.open()]
            for r in results:
                if r["status"] in ("success", "unparseable"):
                    assert r["label"] == w.parse_verdict(r["raw_text"])[0] and r["traffic_type"] == "BATCH"
                    assert "Let me check" not in r["raw_text"]          # thought parts never reach the parser
                else:
                    assert r["label"] is None and r["status"] == "no_response"
            assert "raw_text" not in (report / "summary.json").read_text()
            assert b.step(ctx, jobs_path) and len(fb.jobs) == len(displays)   # finished runs submit nothing new
    finally:
        patch.undo()


def test_gate_blocks_then_recorded_cap_raise():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            jobs_path, report, run = setup(tmp, patch, cap=0.2)
            fb = FakeBatches(FakeStore(tmp / "gcs"), thoughts=2000)   # about 0.375 USD projected with the margin
            ctx = context(tmp, jobs_path, report, run, fb)
            try:
                drive(ctx, jobs_path)
            except b.StopRun as exc:
                assert "projected" in str(exc)
            else:
                raise AssertionError("the gate did not stop the run")
            assert {bt["phase"] for bt in ctx.ledger.batches()} == {"canary"}
            receipt = json.loads((report / "fee_receipt.json").read_text())
            assert not receipt["passed"] and receipt["traffic_types"] == {"BATCH": 20}
            w.raise_cap(1.0, "canary projection above cap", report)
            ctx = context(tmp, jobs_path, report, run, fb)
            assert drive(ctx, jobs_path)
            assert ctx.ledger.meta_get("gate") == "passed"
            assert json.loads((report / "freeze.json").read_text())["cost"]["cap_history"][0]["to_usd"] == 1.0
    finally:
        patch.undo()


def test_submission_respects_cap():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            jobs_path, report, run = setup(tmp, patch, cap=0.01)   # below the worst-case bound of the canary
            ctx = context(tmp, jobs_path, report, run)
            try:
                b.step(ctx, jobs_path)
            except b.StopRun as exc:
                assert "could exceed" in str(exc)
            else:
                raise AssertionError("the canary was submitted above the cap")
            assert ctx.batches.jobs == {} and ctx.store.uploads == []
    finally:
        patch.undo()


def test_lost_create_response_is_adopted():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            jobs_path, report, run = setup(tmp, patch)
            fb = FakeBatches(FakeStore(tmp / "gcs"))
            fb.fail_create_after_register = True
            ctx = context(tmp, jobs_path, report, run, fb)
            try:
                b.step(ctx, jobs_path)
            except ConnectionError:
                pass
            assert len(fb.jobs) == 1
            ctx = context(tmp, jobs_path, report, run, fb)   # restart after the crash
            assert drive(ctx, jobs_path)
            displays = [j["display"] for j in fb.jobs.values()]
            assert len(displays) == len(set(displays))
    finally:
        patch.undo()


def test_unmatched_output_stops_without_settling():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            jobs_path, report, run = setup(tmp, patch)
            fb = FakeBatches(FakeStore(tmp / "gcs"), mangle=lambda t: t.strip() + " ")
            ctx = context(tmp, jobs_path, report, run, fb)
            try:
                drive(ctx, jobs_path)
            except b.StopRun as exc:
                assert "did not match" in str(exc)
            else:
                raise AssertionError("unmatched output lines were accepted")
            assert all(r["state"] == "submitted" for r in ctx.ledger.job_states().values() if r["batch_key"])
            assert ctx.ledger.settled() == float(ctx.ledger.meta_get("probe_usd"))
    finally:
        patch.undo()


def test_wrong_model_version_stops():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            jobs_path, report, run = setup(tmp, patch)
            fb = FakeBatches(FakeStore(tmp / "gcs"))
            ctx = context(tmp, jobs_path, report, run, fb)
            b.probe(ctx)
            fb.version = "gemini-2.5-flash-lite"
            try:
                drive(ctx, jobs_path)
            except b.StopRun as exc:
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
            assert frozen["config"]["price_in_per_m"] == 0.125 and frozen["config"]["mode"] == "vertex-batch"
            jobs = w.load_jobs(jobs_path)
            canary = set(w.canary_ids(jobs))
            full = b.make_shards([j for j in jobs if j.job_id not in canary])
            assert frozen["shard_plan"]["full_sizes"] == [len(s) for s in full] and max(map(len, full)) <= 60
            assert sum(frozen["shard_plan"]["full_sizes"]) == 180 and len(full) >= 3
            patch.set(b, "MAX_SHARD", 1000)
            try:
                b.verify_freeze(jobs_path, report)
            except b.StopRun as exc:
                assert "shard_plan" in str(exc)
            else:
                raise AssertionError("a changed shard plan passed verification")
    finally:
        patch.undo()


def test_cmd_run_resumes_across_calls():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            jobs_path, report, run = setup(tmp, patch)
            store = FakeStore(tmp / "gcs")
            fb = FakeBatches(store)
            patch.set(b, "GcsStore", lambda project: store)
            patch.set(b, "VertexBatches", lambda project, location: fb)
            args = types.SimpleNamespace(jobs=jobs_path, run_dir=run, report_dir=report, project="p",
                                         bucket="gs://bucket", location="global", bucket_location="us-central1",
                                         create_bucket=True, wait=False)
            for _ in range(20):
                b.cmd_run(args)
                if (report / "summary.json").exists():
                    break
            summary = json.loads((report / "summary.json").read_text())
            assert summary["states"] == {"done": 200}
            assert all(not u.startswith("gs://bucket/") or "/wildifeval_flashlite_batch_v3/" in u for u in store.uploads)
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
            traceback.print_exc(limit=3)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
