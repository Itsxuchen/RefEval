"""Offline tests for the WildIFEval Vertex judge. No network, no SDK, no reference values.

Runs under pytest or directly: python tests/test_wildifeval_vertex_judge.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import wildifeval_vertex_judge as w  # noqa: E402

PRODUCERS = w.PRODUCERS


class FakeHTTPError(Exception):
    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.code = code


def answer_for(text: str) -> str:
    """Deterministic fake answer: about a third No, one in eleven unparseable."""
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return "I cannot tell from this." if h % 11 == 0 else ("No." if h % 3 == 0 else "Yes")


def ok_record(text: str, answer: str | None = None) -> dict:
    return {"text": answer if answer is not None else answer_for(text), "finish_reason": "STOP",
            "usage": {"prompt": len(text) // 4, "candidates": 1, "thoughts": 120},
            "model_version": "gemini-3.1-flash-lite", "response_id": "r"}


def fake_request(client, text):
    return ok_record(text)


def make_jobs(tmp: Path, per_producer: int = 40) -> Path:
    path = tmp / "jobs.jsonl"
    with path.open("w") as out:
        for p in PRODUCERS:
            for i in range(per_producer):
                prompt = f"###Instruction:\nsynthetic {p} {i}\n\n###Question:\nIs the following constraint satisfied: \"c{i}\"?\n\n###Answer:"
                out.write(json.dumps({"job_id": f"{p}-{i:04d}", "producer": p, "unit_id": f"u{i}", "cluster_id": f"c{i}",
                                      "task_id": f"t{i}", "criterion_id": f"k{i}", "criterion_hash": "h",
                                      "criterion_index": 0, "output_hash": "o", "split": "pilot", "k": 1,
                                      "prompt": prompt,
                                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}) + "\n")
    return path


class Patch:
    """Minimal monkeypatch so the tests run without pytest."""

    def __init__(self):
        self.saved = []

    def set(self, obj, name, value):
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self.saved):
            setattr(obj, name, value)


def setup_env(tmp: Path, patch: Patch, per_producer: int = 40, cap: float = 5.0):
    jobs_path = make_jobs(tmp, per_producer)
    protocol = tmp / "protocol.md"
    protocol.write_text("# P\n\n" + w.SECTION_HEADING + "\n\nSynthetic section.\n")
    patch.set(w, "EXPECTED_JOBS_SHA256", w.sha256_file(jobs_path))
    patch.set(w, "EXPECTED_JOBS", per_producer * len(PRODUCERS))
    patch.set(w, "PROTOCOL_PATH", protocol)
    patch.set(w, "CANARY_PER_PRODUCER", 4)
    report, run = tmp / "report", tmp / "run"
    w.write_freeze(cap, jobs_path, report)
    return jobs_path, report, run


def no_sleep(_):
    return None


# ----------------------------------------------------------------------------- parser and arithmetic

def test_parser():
    cases = {"Yes": 1, "No.": 0, "**Yes**": 1, "yes, the constraint is met": 1, "NO": 0, "Answer: No": 0,
             "The answer is yes": 1, "Yes/No": 1, "1. Yes": 1, "Yesterday it was fine": None, "Not satisfied": None,
             "I think the constraint is satisfied: Yes": None, "": None, None: None, "   ": None, "Sí": None}
    for text, label in cases.items():
        assert w.parse_verdict(text)[0] == label, (text, w.parse_verdict(text))
    assert w.parse_verdict("The answer is yes")[1] == 3


def test_cost_arithmetic():
    assert abs(w.usage_cost({"prompt": 1_000_000, "candidates": 0, "thoughts": 0}) - 0.25) < 1e-12
    assert abs(w.usage_cost({"prompt": 0, "candidates": 500_000, "thoughts": 500_000}) - 1.50) < 1e-12
    r = w.reserve_usd(2000)
    assert abs(r - ((1000 + 50) * 0.25 + 4096 * 1.5) / 1e6) < 1e-15
    typical = w.usage_cost({"prompt": 600, "candidates": 1, "thoughts": 150})
    assert r > typical


def test_canary_selection():
    with tempfile.TemporaryDirectory() as d:
        jobs = w.load_jobs(make_jobs(Path(d)))
        ids = w.canary_ids(jobs)
        assert len(ids) == w.CANARY_PER_PRODUCER * len(PRODUCERS) == len(set(ids))
        assert ids == w.canary_ids(list(reversed(jobs)))   # independent of file order
        for p in PRODUCERS:
            pool = sorted(j.job_id for j in jobs if j.producer == p)
            mine = [i for i in ids if i.startswith(p + "-")]
            expect = sorted(pool, key=lambda j: hashlib.sha256(f"fee-calibration-v1|{j}".encode()).hexdigest())
            assert mine == expect[:w.CANARY_PER_PRODUCER]


def test_prompt_reader_checks_hash():
    with tempfile.TemporaryDirectory() as d:
        path = make_jobs(Path(d), 3)
        jobs = w.load_jobs(path)
        reader = w.PromptReader(path)
        assert reader.prompt(jobs[4]).endswith("###Answer:")
        bad = w.Job(jobs[0].job_id, jobs[0].producer, jobs[0].offset, jobs[0].length, "0" * 64)
        try:
            reader.prompt(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered prompt hash was accepted")


# ----------------------------------------------------------------------------- freeze

def test_freeze_verify_and_cap():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, _ = setup_env(Path(d), patch)
            frozen, jobs = w.verify_freeze(jobs_path, report)
            assert frozen["reference_values_opened"] is False and len(jobs) == 200
            try:
                w.write_freeze(5.0, jobs_path, report)
            except w.StopRun:
                pass
            else:
                raise AssertionError("second freeze was allowed")
            try:
                w.raise_cap(4.0, "lower", report)
            except w.StopRun:
                pass
            else:
                raise AssertionError("a lower cap was allowed")
            cost = w.raise_cap(9.0, "canary projection above cap", report)
            assert cost["cap_usd"] == 9.0 and cost["cap_history"][0]["from_usd"] == 5.0
            w.verify_freeze(jobs_path, report)          # a cap change keeps the freeze valid
            patch.set(w, "CONFIG", {**w.CONFIG, "temperature": 0.5})
            try:
                w.verify_freeze(jobs_path, report)
            except w.StopRun as exc:
                assert "config" in str(exc)
            else:
                raise AssertionError("changed configuration passed verification")
    finally:
        patch.undo()


def test_missing_section_blocks_freeze():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            protocol = Path(d) / "protocol.md"
            protocol.write_text("# no amendment\n")
            patch.set(w, "PROTOCOL_PATH", protocol)
            try:
                w.protocol_section()
            except w.StopRun:
                pass
            else:
                raise AssertionError("missing section 14 was accepted")
    finally:
        patch.undo()


# ----------------------------------------------------------------------------- ledger and run loop

def env_ledger(tmp: Path, patch: Patch, cap: float = 5.0, per_producer: int = 40):
    jobs_path, report, run = setup_env(tmp, patch, per_producer, cap)
    jobs = w.load_jobs(jobs_path)
    ledger = w.Ledger(run / "state" / "ledger.sqlite", cap)
    return jobs_path, report, run, jobs, ledger, w.PromptReader(jobs_path)


def test_success_and_settlement():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run, jobs, ledger, reader = env_ledger(Path(d), patch)
            w.run_jobs(None, ledger, jobs, reader, "full", 4, fake_request, no_sleep)
            rows = ledger.rows()
            assert len(rows) == 200 and all(r["state"] == "done" for r in rows)
            t = ledger.totals()
            assert abs(t["inflight_usd"]) < 1e-12 and t["unknown_usd"] == 0
            assert abs(t["settled_usd"] - sum(r["cost_usd"] for r in rows)) < 1e-9
            calls = []
            w.run_jobs(None, ledger, jobs, reader, "full", 4, lambda c, t: calls.append(t), no_sleep)
            assert calls == []                              # resume sends nothing for finished jobs
    finally:
        patch.undo()


def test_429_is_released_and_cools_down():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            _, _, _, jobs, ledger, reader = env_ledger(Path(d), patch)
            seen, sleeps = {}, []

            def flaky(client, text):
                seen[text] = seen.get(text, 0) + 1
                if seen[text] <= 4:          # more than MAX_ATTEMPTS, fewer than MAX_ATTEMPTS_429
                    raise FakeHTTPError(429)
                return ok_record(text)

            w.run_jobs(None, ledger, jobs[:10], reader, "full", 1, flaky, sleeps.append)
            rows = ledger.rows()
            assert all(r["state"] == "done" and r["attempts"] == 5 for r in rows)
            t = ledger.totals()
            assert t["unknown_usd"] == 0 and abs(t["settled_usd"] - sum(r["cost_usd"] for r in rows)) < 1e-9
            assert sleeps and max(sleeps) <= 121.0
    finally:
        patch.undo()


def test_timeout_keeps_unknown_charge():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            _, _, _, jobs, ledger, reader = env_ledger(Path(d), patch)

            def timeout(client, text):
                raise TimeoutError("read timed out")

            w.run_jobs(None, ledger, jobs[:2], reader, "full", 1, timeout, no_sleep)
            rows = ledger.rows()
            assert all(r["state"] == "unknown" and r["attempts"] == w.MAX_ATTEMPTS for r in rows)
            expect = sum(w.reserve_usd(len(reader.prompt(j).encode())) for j in jobs[:2]) * w.MAX_ATTEMPTS
            assert abs(ledger.totals()["unknown_usd"] - expect) < 1e-9
            w.run_jobs(None, ledger, jobs[:2], reader, "full", 1, fake_request, no_sleep)   # retried on resume
            assert all(r["state"] == "done" for r in ledger.rows())
    finally:
        patch.undo()


def test_400_rejects_one_job_and_401_stops():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            _, _, _, jobs, ledger, reader = env_ledger(Path(d), patch)
            target = reader.prompt(jobs[0])

            def bad_one(client, text):
                if text == target:
                    raise FakeHTTPError(400)
                return ok_record(text)

            w.run_jobs(None, ledger, jobs[:5], reader, "full", 1, bad_one, no_sleep)
            states = ledger.states()
            assert states[jobs[0].job_id] == "rejected" and sum(s == "done" for s in states.values()) == 4
            calls = []
            w.run_jobs(None, ledger, jobs[:5], reader, "full", 1, lambda c, t: calls.append(t), no_sleep)
            assert calls == []                              # a rejected job is terminal

            def denied(client, text):
                raise FakeHTTPError(401)

            try:
                w.run_jobs(None, ledger, jobs[5:15], reader, "full", 2, denied, no_sleep)
            except w.StopRun as exc:
                assert "401" in str(exc)
            else:
                raise AssertionError("HTTP 401 did not stop the run")
            assert ledger.totals()["unknown_usd"] == 0
    finally:
        patch.undo()


def test_ctrl_c_stops_cleanly():
    import _thread
    import threading
    import time
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            _, _, _, jobs, ledger, reader = env_ledger(Path(d), patch)

            def slow(client, text):
                time.sleep(0.02)
                return ok_record(text)

            threading.Timer(0.3, _thread.interrupt_main).start()
            try:
                w.run_jobs(None, ledger, jobs, reader, "full", 4, slow, no_sleep)
            except w.StopRun as exc:
                assert "interrupted" in str(exc)
            else:
                raise AssertionError("the interrupt was not handled")
            states = ledger.states()
            assert 0 < sum(s == "done" for s in states.values()) < 200
            assert "inflight" not in states.values() and ledger.totals()["unknown_usd"] == 0
            assert abs(ledger.totals()["inflight_usd"]) < 1e-12
            w.run_jobs(None, ledger, jobs, reader, "full", 4, fake_request, no_sleep)   # resume completes
            assert all(s == "done" for s in ledger.states().values())
    finally:
        patch.undo()


def test_persistent_429_stops():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            _, _, _, jobs, ledger, reader = env_ledger(Path(d), patch)
            patch.set(w, "MAX_429_WITHOUT_SUCCESS", 12)

            def always(client, text):
                raise FakeHTTPError(429)

            try:
                w.run_jobs(None, ledger, jobs[:20], reader, "full", 1, always, no_sleep)
            except w.StopRun as exc:
                assert "429" in str(exc)
            else:
                raise AssertionError("persistent 429 did not stop the run")
            assert ledger.totals()["settled_usd"] == 0 and ledger.totals()["unknown_usd"] == 0
    finally:
        patch.undo()


def test_cap_stops_before_overspend():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            _, _, _, jobs, ledger, reader = env_ledger(Path(d), patch, cap=0.03)
            try:
                w.run_jobs(None, ledger, jobs, reader, "full", 1, fake_request, no_sleep)
            except w.StopRun as exc:
                assert "cap" in str(exc)
            else:
                raise AssertionError("the cap did not stop the run")
            assert ledger.committed() <= 0.03 + 1e-12
            assert sum(r["state"] == "done" for r in ledger.rows()) > 0
    finally:
        patch.undo()


def test_interrupted_inflight_becomes_unknown():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            _, _, run, jobs, ledger, _ = env_ledger(Path(d), patch)
            ledger.ensure(jobs[0], "full")
            ledger.begin(jobs[0].job_id, 0.01)
            ledger.con.close()                               # simulated crash with one request in flight
            reopened = w.Ledger(run / "state" / "ledger.sqlite", 5.0)
            assert reopened.state(jobs[0].job_id) == "unknown"
            assert abs(reopened.totals()["unknown_usd"] - 0.01) < 1e-12
    finally:
        patch.undo()


# ----------------------------------------------------------------------------- probe, projection, export, run

def test_probe():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            _, _, _, _, ledger, _ = env_ledger(Path(d), patch)

            def good(client, text):
                return ok_record(text, "OK" if "OK" in text else "Yes")

            w.probe(None, ledger, good)
            assert ledger.meta_get("probe_ok") == "1"
    finally:
        patch.undo()
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            _, _, _, _, ledger, _ = env_ledger(Path(d), patch)
            for bad in (lambda c, t: ok_record(t, "No"),
                        lambda c, t: {**ok_record(t, "Yes"), "model_version": "gemini-2.5-flash"}):
                try:
                    w.probe(None, ledger, bad)
                except w.StopRun:
                    pass
                else:
                    raise AssertionError("a bad probe passed")
            assert ledger.meta_get("probe_ok") is None
    finally:
        patch.undo()


def test_end_to_end_run_and_export():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run = setup_env(Path(d), patch, cap=5.0)
            patch.set(w, "make_client", lambda project, location: object())
            patch.set(w, "default_request", lambda c, t: ok_record(t, "OK") if "single word OK" in t else ok_record(t))
            args = types.SimpleNamespace(jobs=jobs_path, run_dir=run, report_dir=report, location="global",
                                         workers=4, project="test")
            w.cmd_run(args)
            receipt = json.loads((report / "fee_receipt.json").read_text())
            assert receipt["passed"] and all(v["canary_done"] == 4 for v in receipt["producers"].values())
            summary = json.loads((report / "summary.json").read_text())
            assert summary["states"] == {"done": 200}
            assert sum(summary["labels"].values()) == 200 and summary["labels"]["missing"] > 0
            assert "raw_text" not in (report / "summary.json").read_text()
            results = [json.loads(l) for l in (run / "results.jsonl").open()]
            assert [r["job_id"] for r in results] == [json.loads(l)["job_id"] for l in jobs_path.open()]
            for r in results:
                assert r["label"] == w.parse_verdict(r["raw_text"])[0]
                assert (r["status"] == "unparseable") == (r["label"] is None)
                assert "prompt" not in r
    finally:
        patch.undo()


def test_export_keeps_missing_jobs():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run, jobs, ledger, reader = env_ledger(Path(d), patch)
            target = reader.prompt(jobs[0])

            def bad_one(client, text):
                if text == target:
                    raise FakeHTTPError(400)
                return ok_record(text)

            w.run_jobs(None, ledger, jobs[:10], reader, "full", 1, bad_one, no_sleep)
            summary = w.export(ledger, jobs_path, run, report)
            assert summary["states"] == {"rejected": 1, "done": 9, "not_started": 190}
            rows = [json.loads(l) for l in (run / "results.jsonl").open()]
            assert len(rows) == 200 and sum(r["label"] is None for r in rows) == summary["labels"]["missing"]
            assert {r["status"] for r in rows if r["label"] is None} >= {"rejected", "not_started"}
    finally:
        patch.undo()


def test_cost_gate_blocks_full_run():
    patch = Patch()
    try:
        with tempfile.TemporaryDirectory() as d:
            jobs_path, report, run = setup_env(Path(d), patch, cap=0.03)   # projection about 0.047 USD
            patch.set(w, "make_client", lambda project, location: object())
            sent = []

            def request(c, t):
                sent.append(t)
                return ok_record(t, "OK") if "single word OK" in t else ok_record(t)

            patch.set(w, "default_request", request)
            args = types.SimpleNamespace(jobs=jobs_path, run_dir=run, report_dir=report, location="global",
                                         workers=2, project="test")
            try:
                w.cmd_run(args)
            except w.StopRun as exc:
                assert "projected" in str(exc)
            else:
                raise AssertionError("the cost gate did not stop the run")
            assert len(sent) == 2 + 4 * len(PRODUCERS)       # probes and canary only
            assert not json.loads((report / "fee_receipt.json").read_text())["passed"]
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
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
