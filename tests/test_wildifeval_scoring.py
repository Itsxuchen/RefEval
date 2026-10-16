"""No network or real credentials: fault-injected durable scoring checks."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

import src.wildifeval_scoring as s


def make_jobs(path, n=4, producers=2):
    jobs = []
    for i in range(n):
        prompt = "Synthetic rubric question " + str(i)
        criterion = "criterion-" + str(i)
        h = s.text_hash(prompt)
        jobs.append(dict(job_id=s.text_hash("wildifeval-new-p-v1|"+criterion+"|"+h),
                         unit_id="unit-"+str(i), task_id="task-"+str(i), cluster_id="cluster-"+str(i),
                         producer="producer-"+str(i % producers), criterion_index=0, criterion_id=criterion,
                         k=1, split="pilot", prompt=prompt, prompt_sha256=h,
                         output_hash=s.text_hash("output-"+str(i)), criterion_hash=s.text_hash("criterion-"+str(i))))
    path.write_text("".join(json.dumps(j)+"\n" for j in jobs))
    return jobs


def setup_run(tmp_path, n=4, cap=100, max_attempts=3):
    jobs_path = tmp_path / "jobs.jsonl"
    jobs = make_jobs(jobs_path, n=n)
    config = s.config_example(s.file_hash(jobs_path))
    config.update(state="ready", concurrency=4, requests_per_second=100,
                  prior_reserved_usd=.001, budget_usd=cap, max_attempts=max_attempts)
    state = tmp_path / "state"
    ledger = s.Ledger(state, config, jobs_path)
    info = ledger.summary()
    ledger.close()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(dict(status="passed", source="unrelated_synthetic_probe",
                                          model_requested=config["model"], response_model=config["expected_response_model"],
                                          system_fingerprint=None, request_parameters=s.request_parameters(config),
                                          usage_bound_ok=True, score_parse_ok=True)))
    freeze = tmp_path / "freeze.json"
    canary = tmp_path / "fixed_canary.json"
    s.make_canary_manifest(jobs_path, canary)
    freeze.write_text(json.dumps(dict(status="frozen", config_sha256=s.digest(config),
                                     jobs_sha256=config["jobs_sha256"], calibration_sha256=s.file_hash(calibration),
                                     runner_sha256=s.file_hash(s.__file__), ledger_id=info["ledger_id"],
                                     canary_job_ids_sha256=s.file_hash(canary),
                                     state_directory_sha256=s.text_hash(str(state.resolve())))))
    return jobs, jobs_path, config, state, calibration, freeze


def response(prob=.75, **extra):
    return dict(id="synthetic-response", model="gpt-6-luna", system_fingerprint="fixture-fp", service_tier="default",
                usage={"prompt_tokens": 100, "completion_tokens": 1,
                       "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 20}},
                choices=[{"finish_reason": "stop", "message": {"content": "Yes", "refusal": None},
                          "logprobs": {"content": [{"token": "Yes", "top_logprobs": [
                              {"token": "Yes", "logprob": math.log(prob)},
                              {"token": "No", "logprob": math.log(1-prob)}]}]}}], **extra)


class APIError(Exception):
    def __init__(self, status, code=None, error_type=None, retry_after=None):
        super().__init__("sk-DO_NOT_LOG_SECRET_PROVIDER_BODY")
        self.status_code, self.code = status, code
        self.body = {"error": {"type": error_type, "code": code, "message": "sk-DO_NOT_LOG_SECRET_PROVIDER_BODY"}}
        self.response = SimpleNamespace(headers={"retry-after": retry_after} if retry_after is not None else {})


def run(setup, request, **kwargs):
    _, jobs, config, state, calibration, freeze = setup
    kwargs.setdefault("job_ids_path", jobs.parent / "fixed_canary.json")
    if kwargs["job_ids_path"] is not None:
        kwargs.setdefault("phase_cap_usd", 1)
    return s.run_jobs(jobs, config, state, request, calibration_receipt=calibration, generation_freeze=freeze, **kwargs)


def approve_fixture_fee(fixture):
    path = fixture[1].parent / "fees.json"
    report = s.make_fee_report(fixture[1], fixture[2], fixture[3], fixture[4], fixture[5], path)
    report["status"] = "passed"
    path.write_text(json.dumps(report))
    return path


def test_upstream_probability_regex_positions_and_mass():
    positions = [{"top_logprobs": [{"token": "yesterday", "logprob": -.1}]}]
    positions.append({"top_logprobs": [{"token": " YES,", "logprob": math.log(.3)},
                                      {"token": "Ġno_", "logprob": math.log(.1)}]})
    result = s.score_from_tokens(positions)
    assert result["prob_yes"] == pytest.approx(.75)
    assert result["score_position"] == 1
    assert result["yes_no_mass"] == pytest.approx(.4)
    near = [{"top_logprobs": [{"token": "yes", "logprob": math.log(.00001)}]}]
    with pytest.raises(s.GuardError):
        s.score_from_tokens(near)
    with pytest.raises(s.GuardError):
        s.score_from_tokens(positions[:1]*5+positions[1:])


@pytest.mark.parametrize("text,expected", [("Yes", 1), ("(NO)", 0), ("yesterday", None), ("nobody", None)])
def test_score_word_boundaries(text, expected):
    value = [{"top_logprobs": [{"token": text, "logprob": -.2}]}]
    if expected is None:
        with pytest.raises(s.GuardError):
            s.score_from_tokens(value)
    else:
        assert s.score_from_tokens(value)["prob_yes"] == expected


def test_threshold_tie_passes_and_truncation_refusal_missing():
    config = s.config_example()
    assert s.parse_response(response(.5), config)["label"] == 1
    r = response()
    r["choices"][0]["finish_reason"] = "length"
    with pytest.raises(s.GuardError, match="nonstop_completion"):
        s.parse_response(r, config)
    r = response()
    r["choices"][0]["message"]["refusal"] = "refused"
    with pytest.raises(s.GuardError, match="refusal"):
        s.parse_response(r, config)


def test_cost_exact_rates_and_model_context_reservation():
    config = s.config_example()
    cost = s.usage_cost(response()["usage"], config)
    assert cost["modeled_nano"] == 70*100+10*10+20*125+500
    assert cost["conservative_nano"] == 100*125+500
    assert s.reservation_nano(config) == 922000*250+1000*750
    upper = s.usage_cost({"prompt_tokens": 922000, "completion_tokens": 1000}, config)
    assert upper["conservative_nano"] == s.reservation_nano(config)
    with pytest.raises(s.GuardError, match="provider_usage_exceeds_frozen_bound"):
        s.usage_cost({"prompt_tokens": 922001, "completion_tokens": 1}, config)
    with pytest.raises(s.GuardError, match="invalid_usage"):
        s.usage_cost({"prompt_tokens": 100, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 101}}, config)


def test_quota_type_overrides_rate_limit_code_and_sanitizes():
    error = s.safe_error(APIError(429, "rate_limit_exceeded", "insufficient_quota", "10"))
    assert error["fatal"] and not error["retryable"]
    assert error["code"] == "insufficient_quota"
    assert error["retry_after_seconds"] == 10
    assert "SECRET" not in json.dumps(error)
    rate = s.safe_error(APIError(429, "slow_down", retry_after="4"))
    assert rate["retryable"] and not rate["fatal"]
    assert s.safe_error(APIError(401, "invalid_api_key"))["fatal"]
    assert s.safe_error(RuntimeError("secret"))["fatal"]


def test_request_configuration_no_hidden_sdk_retries_or_tools(tmp_path):
    job = make_jobs(tmp_path / "jobs", n=1)[0]
    body = s.request_body(job, s.config_example())
    assert body["model"] == "gpt-6-luna"
    assert body["max_completion_tokens"] == 1000
    assert body["reasoning_effort"] == "none" and body["temperature"] == 0
    assert body["top_logprobs"] == 5 and body["logprobs"] is True
    assert body["service_tier"] == "default" and body["store"] is False
    assert "tools" not in body


def test_draft_config_cannot_run_or_read_key(tmp_path):
    fixture = setup_run(tmp_path)
    fixture[2]["state"] = "awaiting_api_calibration"
    with pytest.raises(s.GuardError, match="awaiting_api_calibration"):
        run(fixture, lambda body: pytest.fail("must not call provider"))


def test_freeze_prevents_new_state_and_changed_receipts(tmp_path):
    fixture = setup_run(tmp_path)
    _, jobs, config, state, calibration, freeze = fixture
    with pytest.raises(s.GuardError, match="frozen_ledger_path_missing_or_changed"):
        s.run_jobs(jobs, config, tmp_path / "different", lambda _: pytest.fail(), calibration_receipt=calibration, generation_freeze=freeze)
    record = json.loads(freeze.read_text())
    record["ledger_id"] = "different-id"
    freeze.write_text(json.dumps(record))
    with pytest.raises(s.GuardError, match="frozen_ledger_identity_changed"):
        run(fixture, lambda _: pytest.fail())


def test_success_resume_no_duplicate_and_export_shape(tmp_path):
    fixture = setup_run(tmp_path)
    calls = []
    result = run(fixture, lambda body: calls.append(body) or response(), limit=2)
    assert result["jobs"] == {"pending": 2, "success": 2}
    result = run(fixture, lambda body: calls.append(body) or response())
    assert result["jobs"] == {"success": 4} and len(calls) == 4
    result = run(fixture, lambda body: pytest.fail("successes must not repeat"))
    rows = [json.loads(x) for x in (fixture[3] / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(r["status"] == "success" and r["label"] == 1 for r in rows)
    assert all("prompt_sha256" in r and "prob_yes" in r and "model" in r for r in rows)
    assert all("prompt" not in r and "r" not in r for r in rows)
    assert result["committed_or_unknown_usd"] == pytest.approx(.001+4*.000013)


def test_quota_initial_dispatch_singleflight_and_no_retry(tmp_path):
    fixture = setup_run(tmp_path, n=20)
    calls = []
    def fail(body):
        calls.append(body)
        raise APIError(429, "credit_balance_exhausted", "insufficient_quota")
    result = run(fixture, fail)
    assert len(calls) == 1
    assert result["halted_reason"] == "insufficient_quota"
    assert result["attempts"] == {"fatal": 1}
    assert result["committed_or_unknown_usd"] == pytest.approx(.001+.23125)
    assert run(fixture, lambda _: pytest.fail())["halted_reason"] == "insufficient_quota"


def test_resume_after_quota_preserves_unknown_charge(tmp_path):
    fixture = setup_run(tmp_path, n=1)
    def fail(_):
        raise APIError(429, "insufficient_quota")
    run(fixture, fail)
    result = run(fixture, lambda _: response(), resume_after_quota=True)
    assert result["jobs"] == {"success": 1}
    assert result["committed_or_unknown_usd"] == pytest.approx(.001+.23125+.000013)


def test_unparseable_success_is_terminal_no_regeneration(tmp_path):
    fixture = setup_run(tmp_path, n=1)
    bad = response()
    bad["choices"][0]["logprobs"] = {"content": []}
    result = run(fixture, lambda _: bad)
    assert result["jobs"] == {"terminal_missing": 1}
    assert result["attempts"] == {"terminal_missing": 1}
    run(fixture, lambda _: pytest.fail("do not resample invalid answers"))
    row = json.loads((fixture[3] / "failures.jsonl").read_text())
    assert "label" not in row


@pytest.mark.parametrize("field,value,code", [("model", "another-model", "response_model_changed"), ("service_tier", "priority", "response_service_tier_changed")])
def test_price_identity_drift_keeps_full_reservation(tmp_path, field, value, code):
    fixture = setup_run(tmp_path, n=2)
    changed = response()
    changed[field] = value
    result = run(fixture, lambda _: changed)
    assert result["halted_reason"] == code
    assert result["committed_or_unknown_usd"] == pytest.approx(.001+.23125)
    assert result["attempts"] == {"fatal": 1}


def test_crash_unknown_reservations_and_attempt_bound(tmp_path):
    fixture = setup_run(tmp_path, n=1, max_attempts=2)
    _, jobs, config, state, _, _ = fixture
    ledger = s.Ledger(state, config, jobs)
    claim, reason = ledger.claim()
    assert reason == "claimed"
    ledger.close()  # Simulates abrupt process loss between dispatch and result.
    ledger = s.Ledger(state, config, jobs)
    claim2, reason = ledger.claim()
    assert reason == "claimed" and claim2["attempt_no"] == 2
    ledger.close()
    ledger = s.Ledger(state, config, jobs)
    assert ledger.claim()[1] == "done"
    summary = ledger.summary()
    ledger.close()
    assert summary["jobs"] == {"exhausted": 1}
    assert summary["attempts"] == {"unknown_after_crash": 2}
    assert summary["committed_or_unknown_usd"] == pytest.approx(.001+2*.23125)


def test_parallel_claims_cannot_oversubscribe_cap(tmp_path):
    fixture = setup_run(tmp_path, n=40, cap=1)
    _, jobs, config, state, _, _ = fixture
    ledger = s.Ledger(state, config, jobs)
    with ThreadPoolExecutor(max_workers=20) as pool:
        claims = list(pool.map(lambda _: ledger.claim(), range(40)))
    assert sum(reason == "claimed" for _, reason in claims) == 4
    assert ledger.summary()["committed_or_unknown_usd"] <= 1
    ledger.close()


def test_phase_cap_does_not_reset_global_budget(tmp_path):
    fixture = setup_run(tmp_path, n=1)
    calls = []
    result = run(fixture, lambda _: calls.append(1) or response(), phase_cap_usd=.2)
    assert not calls and result["jobs"] == {"pending": 1}
    result = run(fixture, lambda _: response(), phase_cap_usd=1)
    assert result["jobs"] == {"success": 1}
    assert result["prior_reserved_usd"] == .001


def test_frozen_canary_balanced_and_reused_for_full_run(tmp_path):
    fixture = setup_run(tmp_path, n=70)
    canary = tmp_path / "canary.json"
    selected = s.make_canary_manifest(fixture[1], canary)
    assert selected["jobs"] == 64
    record = json.loads(fixture[-1].read_text())
    record["canary_job_ids_sha256"] = s.file_hash(canary)
    fixture[-1].write_text(json.dumps(record))
    result = run(fixture, lambda _: response(), job_ids_path=canary, phase_cap_usd=1)
    assert result["jobs"] == {"pending": 6, "success": 64}
    fees = approve_fixture_fee(fixture)
    result = run(fixture, lambda _: response(), job_ids_path=None, fee_receipt=fees)
    assert result["jobs"] == {"success": 70}
    assert result["attempts"] == {"success": 70}


def test_invalid_reference_or_changed_prompt_rejected(tmp_path):
    job = make_jobs(tmp_path / "j", n=1)[0]
    job["r"] = 1
    with pytest.raises(s.GuardError, match="reference_field_in_jobs"):
        s.validate_job(job)
    del job["r"]
    job["prompt"] += "changed"
    with pytest.raises(s.GuardError, match="job_prompt_hash_mismatch"):
        s.validate_job(job)


def test_state_lock_and_config_identity(tmp_path):
    fixture = setup_run(tmp_path)
    _, jobs, config, state, _, _ = fixture
    first = s.Ledger(state, config, jobs)
    with pytest.raises(s.GuardError, match="another_runner_owns_state"):
        s.Ledger(state, config, jobs)
    first.close()
    changed = dict(config, concurrency=2)
    with pytest.raises(s.GuardError, match="state_configuration_mismatch"):
        s.Ledger(state, changed, jobs)


def test_fake_key_reader_never_returns_error_contents(tmp_path):
    path = tmp_path / "fake.env"
    fake = "sk-"+"fixture_not_real_"*3
    path.write_text("OPENAI_API_KEY="+fake)
    assert s.read_existing_key(path) == fake
    path.write_text(fake+"\n"+fake+"different")
    with pytest.raises(s.GuardError, match="key_file_requires_one_unique_key"):
        s.read_existing_key(path)


def test_full_run_cannot_skip_canary_and_fee_review(tmp_path):
    fixture = setup_run(tmp_path)
    with pytest.raises(s.GuardError, match="full_run_requires_fee_receipt"):
        run(fixture, lambda _: pytest.fail(), job_ids_path=None)
    fees = approve_fixture_fee(fixture)
    with pytest.raises(s.GuardError, match="fee_calibration_not_passed"):
        run(fixture, lambda _: pytest.fail(), job_ids_path=None, fee_receipt=fees)


def test_fee_gate_rejects_partial_canary_and_wrong_ledger(tmp_path):
    fixture = setup_run(tmp_path)
    run(fixture, lambda _: response(), limit=1)
    fees = approve_fixture_fee(fixture)
    assert json.loads(fees.read_text())["complete"] is False
    with pytest.raises(s.GuardError, match="fee_calibration_not_passed"):
        run(fixture, lambda _: pytest.fail(), job_ids_path=None, fee_receipt=fees)
    run(fixture, lambda _: response())
    fees.unlink()
    fees = approve_fixture_fee(fixture)
    record = json.loads(fees.read_text())
    record["ledger_id"] = "wrong-ledger"
    fees.write_text(json.dumps(record))
    with pytest.raises(s.GuardError, match="fee_calibration_not_passed"):
        run(fixture, lambda _: pytest.fail(), job_ids_path=None, fee_receipt=fees)


def test_canary_cannot_raise_phase_cap(tmp_path):
    fixture = setup_run(tmp_path)
    with pytest.raises(s.GuardError, match="canary_requires_cumulative_one_dollar_cap"):
        run(fixture, lambda _: pytest.fail(), phase_cap_usd=2)


def test_completion_never_treats_pending_as_missing(tmp_path):
    fixture = setup_run(tmp_path)
    result = run(fixture, lambda _: response(), limit=1)
    receipt = json.loads((fixture[3]/"completion.json").read_text())
    assert receipt["status"] == "incomplete" and receipt["pending_jobs"] == 3
    assert receipt["planned_jobs"] == 4 and receipt["failure_jobs"] == 0
    result = run(fixture, lambda _: response())
    receipt = json.loads((fixture[3]/"completion.json").read_text())
    assert receipt["status"] == "complete" and receipt["pending_jobs"] == 0
    assert receipt["results_sha256"] == s.file_hash(fixture[3]/"results.jsonl")
    assert receipt["generation_freeze_sha256"] == s.file_hash(fixture[-1])


def test_interrupt_stops_dispatch_instead_of_finishing_entire_pool(tmp_path):
    import time
    fixture = setup_run(tmp_path, n=60)
    calls = []
    def request(_):
        calls.append(1)
        time.sleep(.01)
        return response()
    def interrupt(_):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run(fixture, request, progress=interrupt)
    assert len(calls) <= 5
    ledger = s.Ledger(fixture[3], fixture[2], fixture[1])
    assert ledger.summary()["jobs"].get("pending", 0) >= 55
    ledger.close()


def test_retry_attempts_are_hard_bounded_and_unknown_charges_retained(tmp_path, monkeypatch):
    fixture = setup_run(tmp_path, n=1)
    # Remove wall-clock delay while retaining the real retry/accounting path.
    monkeypatch.setattr(s.Pacer, "cooldown", lambda *args: None)
    original = s.Ledger.finish
    def finish_without_delay(self, claim, **kwargs):
        kwargs["delay"] = 0
        return original(self, claim, **kwargs)
    monkeypatch.setattr(s.Ledger, "finish", finish_without_delay)
    calls = []
    def unavailable(_):
        calls.append(1)
        raise APIError(503, "server_is_overloaded")
    result = run(fixture, unavailable)
    assert len(calls) == 3 and result["jobs"] == {"exhausted": 1}
    assert result["committed_or_unknown_usd"] == pytest.approx(.001+3*.23125)


def test_invalid_skipped_prefix_is_serialized_without_nan_or_resample(tmp_path):
    fixture = setup_run(tmp_path, n=1)
    bad = response()
    bad["choices"][0]["logprobs"]["content"].insert(0, {"top_logprobs": [{"token": "x", "logprob": float("nan")}]})
    result = run(fixture, lambda _: bad)
    assert result["jobs"] == {"success": 1}
    row = json.loads((fixture[3]/"results.jsonl").read_text())
    assert row["invalid_prefix_positions"] == [0]
    assert row["score_position"] == 1
    assert row["top_logprobs"][0] == {"top_logprobs": []}
    assert row["evidence_serialization_version"] == "selected-prefix-v2"
    assert s.score_from_tokens(row["top_logprobs"])["prob_yes"] == row["prob_yes"]
    assert result["committed_or_unknown_usd"] == pytest.approx(.001+.000013)


def test_unexpected_worker_error_stops_other_dispatchers(tmp_path, monkeypatch):
    fixture = setup_run(tmp_path, n=64)
    calls = []
    def request(_):
        calls.append(1)
        if len(calls) == 2:
            raise APIError(503)
        return response()
    def broken_error_handler(_):
        raise ValueError("synthetic internal failure")
    monkeypatch.setattr(s, "safe_error", broken_error_handler)
    with pytest.raises(ValueError, match="synthetic internal failure"):
        run(fixture, request)
    assert len(calls) <= 5
    ledger = s.Ledger(fixture[3], fixture[2], fixture[1])
    assert ledger.halted() == "runner_unexpected_error"
    ledger.close()


def test_post_selection_positive_logprob_does_not_change_score():
    raw = response()
    positions = raw["choices"][0]["logprobs"]["content"]
    positions.append({"top_logprobs": [{"token": ".", "logprob": 3.814697265625e-06}]})
    score = s.score_from_tokens(positions)
    parsed = s.parse_response(raw, s.config_example())
    assert parsed["prob_yes"] == score["prob_yes"]
    assert parsed["label"] == 1 and parsed["score_position"] == 0
    assert len(parsed["top_logprobs"]) == 1
    assert parsed["invalid_prefix_positions"] == []
    assert s.score_from_tokens(parsed["top_logprobs"]) == score
