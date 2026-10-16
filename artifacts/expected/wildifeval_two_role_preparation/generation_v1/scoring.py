"""Resumable, budget-reserved WildIFEval P scoring; never reads reference labels.

Real execution requires a successful unrelated-input calibration receipt and a
generation freeze. SDK retries are disabled. SQLite is the durable source of
truth; JSONL exports contain successful P scores or separate unlabeled failures.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import email.utils
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

NANODOLLARS = 1_000_000_000
MAX_INPUT_TOKENS = 922_000
SHORT_CONTEXT_TOKENS = 272_000
# Nanodollars per token. Cache-write prices are a replacement rate, not additive.
RATES = {"short": dict(input=100, cached=10, write=125, output=500),
         "long": dict(input=200, cached=20, write=250, output=750)}
QUOTA_CODES = {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached", "billing_not_active"}
KNOWN_ERROR_CODES = QUOTA_CODES | {"rate_limit_exceeded", "slow_down", "server_is_overloaded", "invalid_api_key", "model_not_found", "account_deactivated", "invalid_request_error", "unsupported_parameter", "unsupported_value"}
PARSER_VERSION = "wild-top5-position-top5-token-yes-no-mass-gt-1e-4-v1"
UPSTREAM_PARSER_URL = "https://github.com/gililior/wild-if-eval-code/blob/82ebff7068b9faeadc5468d55935d62b1601d89c/scripts/arena_filtering/classify_constrained_generation_tasks.py"
META_FIELDS = ("job_id", "unit_id", "task_id", "cluster_id", "producer", "criterion_index", "criterion_id", "k", "split", "prompt_sha256", "output_hash", "criterion_hash")


class GuardError(ValueError):
    """Messages are fixed public codes; never include provider bodies or secrets."""


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def usd_to_nano(value):
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise GuardError("invalid_budget_amount")
    return int((amount * NANODOLLARS).to_integral_value(rounding=ROUND_CEILING))


def config_example(jobs_sha256=""):
    return {
        "schema_version": 1, "state": "awaiting_api_calibration", "jobs_sha256": jobs_sha256,
        "model": "gpt-6-luna", "expected_response_model": "gpt-6-luna",
        "expected_system_fingerprint": None, "max_completion_tokens": 1000,
        "max_attempts": 3, "concurrency": 32, "requests_per_second": 8,
        "timeout_seconds": 120, "budget_usd": 100, "prior_reserved_usd": .001,
        "reasoning_effort": "none", "temperature": 0, "logprobs": True,
        "top_logprobs": 5, "service_tier": "default", "store": False,
        "base_url": "https://api.openai.com/v1", "parser_version": PARSER_VERSION,
        "input_reservation_tokens": MAX_INPUT_TOKENS,
        "reservation_price_basis": "documented maximum input at standard long-context cache-write rate; output cap at long-context output rate",
        "pricing_source": "https://developers.openai.com/api/docs/pricing",
        "model_limit_source": "https://developers.openai.com/api/docs/models/gpt-6-luna",
    }


def validate_config(config):
    fixed = config_example()
    if set(config) != set(fixed):
        raise GuardError("unexpected_configuration_fields")
    for key in ("schema_version", "model", "max_completion_tokens", "reasoning_effort", "temperature", "logprobs", "top_logprobs", "service_tier", "store", "base_url", "parser_version", "input_reservation_tokens"):
        if config.get(key) != fixed[key]:
            raise GuardError("nonfrozen_request_configuration")
    if not re.fullmatch(r"[0-9a-f]{64}", str(config.get("jobs_sha256", ""))):
        raise GuardError("invalid_jobs_hash")
    if not isinstance(config.get("expected_response_model"), str) or not config["expected_response_model"].startswith("gpt-6-luna"):
        raise GuardError("invalid_expected_response_model")
    fingerprint = config.get("expected_system_fingerprint")
    if fingerprint is not None and (not isinstance(fingerprint, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", fingerprint)):
        raise GuardError("invalid_expected_fingerprint")
    for key, lower, upper in (("max_attempts", 1, 3), ("concurrency", 1, 64), ("timeout_seconds", 1, 600)):
        if type(config.get(key)) is not int or not lower <= config[key] <= upper:
            raise GuardError("invalid_execution_limit")
    rps = config.get("requests_per_second")
    if not isinstance(rps, (float, int)) or not math.isfinite(rps) or not 0 < rps <= 100:
        raise GuardError("invalid_request_rate")
    cap, prior = usd_to_nano(config["budget_usd"]), usd_to_nano(config["prior_reserved_usd"])
    if not 0 < cap <= 100 * NANODOLLARS or prior > cap:
        raise GuardError("budget_exceeds_authorization")
    return config


def request_body(job, config):
    return {"model": config["model"], "messages": [{"role": "user", "content": job["prompt"]}],
            **{key: config[key] for key in ("reasoning_effort", "temperature", "logprobs", "top_logprobs", "max_completion_tokens", "service_tier", "store")}}


def request_parameters(config):
    return {key: config[key] for key in ("reasoning_effort", "temperature", "logprobs", "top_logprobs", "max_completion_tokens", "service_tier", "store")}


def validate_execution_gates(config, calibration_path, freeze_path, state, job_ids_path=None):
    validate_config(config)
    if config.get("state") != "ready":
        raise GuardError("awaiting_api_calibration")
    if calibration_path is None or freeze_path is None:
        raise GuardError("missing_execution_receipts")
    calibration = json.loads(Path(calibration_path).read_text())
    freeze = json.loads(Path(freeze_path).read_text())
    if (calibration.get("status") != "passed" or calibration.get("source") != "unrelated_synthetic_probe"
            or calibration.get("model_requested") != config["model"]
            or calibration.get("response_model") != config["expected_response_model"]
            or calibration.get("request_parameters") != request_parameters(config)
            or calibration.get("usage_bound_ok") is not True or calibration.get("score_parse_ok") is not True):
        raise GuardError("invalid_calibration_receipt")
    if config.get("expected_system_fingerprint") is not None and calibration.get("system_fingerprint") != config["expected_system_fingerprint"]:
        raise GuardError("calibration_fingerprint_mismatch")
    if (freeze.get("status") != "frozen" or freeze.get("config_sha256") != digest(config)
            or freeze.get("jobs_sha256") != config["jobs_sha256"]
            or freeze.get("calibration_sha256") != file_hash(calibration_path)
            or freeze.get("runner_sha256") != file_hash(__file__)):
        raise GuardError("invalid_generation_freeze")
    if freeze.get("state_directory_sha256") != text_hash(str(Path(state).resolve())) or not (Path(state) / "ledger.sqlite").is_file():
        raise GuardError("frozen_ledger_path_missing_or_changed")
    db = sqlite3.connect(f"file:{(Path(state)/'ledger.sqlite').resolve()}?mode=ro", uri=True)
    try:
        identity = db.execute("SELECT value FROM metadata WHERE key='ledger_id'").fetchone()
    finally:
        db.close()
    if identity is None or freeze.get("ledger_id") != identity[0]:
        raise GuardError("frozen_ledger_identity_changed")
    if job_ids_path is not None and freeze.get("canary_job_ids_sha256") != file_hash(job_ids_path):
        raise GuardError("canary_manifest_not_frozen")


def reservation_nano(config):
    return MAX_INPUT_TOKENS * RATES["long"]["write"] + config["max_completion_tokens"] * RATES["long"]["output"]


def usage_cost(usage, config):
    if not isinstance(usage, dict):
        raise GuardError("missing_usage")
    prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        raise GuardError("invalid_usage")
    cached, writes = details.get("cached_tokens", 0) or 0, details.get("cache_write_tokens", 0) or 0
    if any(type(v) is not int or v < 0 for v in (prompt, output, cached, writes)) or cached + writes > prompt:
        raise GuardError("invalid_usage")
    if prompt > MAX_INPUT_TOKENS or output > config["max_completion_tokens"]:
        raise GuardError("provider_usage_exceeds_frozen_bound")
    tier = "long" if prompt > SHORT_CONTEXT_TOKENS else "short"
    rate = RATES[tier]
    conservative = prompt * rate["write"] + output * rate["output"]
    modeled = (prompt-cached-writes)*rate["input"] + cached*rate["cached"] + writes*rate["write"] + output*rate["output"]
    clean = dict(prompt_tokens=prompt, completion_tokens=output,
                 prompt_tokens_details=dict(cached_tokens=cached, cache_write_tokens=writes),
                 cache_write_tokens_reported="cache_write_tokens" in details)
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if type(reasoning) is int and 0 <= reasoning <= output:
        clean["reasoning_tokens"] = reasoning
    return {"usage": clean, "modeled_nano": modeled, "conservative_nano": conservative, "price_tier": tier}


def score_from_tokens(positions):
    """Upstream calc_score on valid logprobs; invalid positions are skipped.

    Search first five generated positions. At the first position with Yes/No
    mass > 0.0001, normalize their masses. Word boundaries/variants match the
    pinned upstream regular expressions; 'yesterday' is not a Yes token.
    """
    if not isinstance(positions, list):
        raise GuardError("unparseable_probability")
    patterns = [re.compile(rf"(\W|Ġ|_)*{name}(\W|Ġ|_)*", re.IGNORECASE) for name in ("yes", "no")]
    for index, position in enumerate(positions[:5]):
        try:
            top = position["top_logprobs"]
            if not isinstance(top, list) or len(top) > 5:
                continue
            masses = [0., 0.]
            for token in top:
                text, logprob = token["token"], token["logprob"]
                if not isinstance(text, str) or not isinstance(logprob, (float, int)) or not math.isfinite(logprob) or logprob > 0:
                    raise ValueError("invalid_logprob")
                for j, pattern in enumerate(patterns):
                    if pattern.fullmatch(text):
                        masses[j] += math.exp(logprob)
            total = sum(masses)
            if total > .0001 and total <= 1.000001:
                return dict(prob_yes=masses[0]/total, score_position=index, yes_no_mass=total)
        except (TypeError, ValueError, KeyError, OverflowError):
            continue
    raise GuardError("unparseable_probability")


def parse_response(response, config):
    if not isinstance(response, dict):
        raise GuardError("invalid_response")
    if response.get("model") != config["expected_response_model"]:
        raise GuardError("response_model_changed")
    expected = config.get("expected_system_fingerprint")
    if expected is not None and response.get("system_fingerprint") != expected:
        raise GuardError("response_fingerprint_changed")
    if response.get("service_tier") not in (None, "default"):
        raise GuardError("response_service_tier_changed")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise GuardError("invalid_choices")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise GuardError("nonstop_completion")
    message = choice.get("message") or {}
    if message.get("refusal"):
        raise GuardError("refusal")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise GuardError("empty_completion")
    positions = (choice.get("logprobs") or {}).get("content")
    score = score_from_tokens(positions)
    # Only the five positions used by the frozen parser are needed to reproduce P.
    clean_positions = []
    for position in positions[:5]:
        try:
            top = position["top_logprobs"]
            if not isinstance(top, list) or len(top) > 5:
                raise ValueError()
            clean = []
            for token in top:
                value = token["logprob"]
                if not isinstance(token["token"], str) or not isinstance(value, (float, int)) or not math.isfinite(value) or value > 0:
                    raise ValueError()
                clean.append({"token": token["token"], "logprob": value})
            clean_positions.append({"top_logprobs": clean})
        except (TypeError, KeyError, ValueError):
            raise GuardError("malformed_logprobs") from None
    return {**score, "label": int(score["prob_yes"] >= .5), "status": "success",
            "model": response["model"], "system_fingerprint": response.get("system_fingerprint"),
            "service_tier": response.get("service_tier"), "parser_version": PARSER_VERSION,
            "top_logprobs": clean_positions, "finish_reason": choice["finish_reason"],
            "response_text_sha256": text_hash(content), "response_id_sha256": text_hash(str(response.get("id", "")))}


def safe_error(exc, now=None):
    """Never return exception strings, provider bodies, request headers, or keys."""
    status = getattr(exc, "status_code", None)
    status = status if type(status) is int and 100 <= status <= 599 else None
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    error_type = None
    if isinstance(body, dict):
        data = body.get("error", body)
        if isinstance(data, dict):
            code = code or data.get("code")
            error_type = data.get("type")
    code = code if isinstance(code, str) and code in KNOWN_ERROR_CODES else "other"
    if isinstance(error_type, str) and error_type in QUOTA_CODES:
        code = error_type
    fatal = code in QUOTA_CODES or (status is not None and 400 <= status < 500 and status not in (408, 409, 429))
    retryable = not fatal and (status in (408, 409, 429) or (status is not None and status >= 500)
                               or type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectionError"))
    if not fatal and not retryable and status is None:
        fatal, code = True, "unrecognized_request_error"
    delay = None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw = headers.get("retry-after")
        if isinstance(raw, str):
            try:
                delay = float(raw)
            except ValueError:
                try:
                    stamp = email.utils.parsedate_to_datetime(raw).timestamp()
                    delay = max(0., stamp-(time.time() if now is None else now))
                except (ValueError, TypeError, OverflowError):
                    pass
    if delay is not None and (not math.isfinite(delay) or delay < 0):
        delay = None
    return dict(status=status, code=code, fatal=fatal, retryable=retryable, retry_after_seconds=delay)


def validate_job(job):
    if not isinstance(job, dict) or any(key not in job for key in META_FIELDS + ("prompt",)):
        raise GuardError("invalid_job_schema")
    if not isinstance(job["prompt"], str) or text_hash(job["prompt"]) != job["prompt_sha256"]:
        raise GuardError("job_prompt_hash_mismatch")
    expected = text_hash("wildifeval-new-p-v1|" + job["criterion_id"] + "|" + job["prompt_sha256"])
    if job["job_id"] != expected:
        raise GuardError("job_id_mismatch")
    if any(key in job for key in ("r", "r2", "reference", "reference_label")):
        raise GuardError("reference_field_in_jobs")
    if set(job) != set(META_FIELDS + ("prompt",)):
        raise GuardError("unexpected_job_fields")


class Ledger:
    def __init__(self, directory, config, jobs_path, *, require_existing=False):
        self.directory, self.config, self.jobs_path = Path(directory), config, Path(jobs_path)
        validate_config(config)
        if file_hash(jobs_path) != config["jobs_sha256"]:
            raise GuardError("jobs_file_hash_mismatch")
        if require_existing and not (self.directory / "ledger.sqlite").is_file():
            raise GuardError("existing_frozen_ledger_required")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock_handle = (self.directory / "runner.lock").open("a+")
        try:
            fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_handle.close()
            raise GuardError("another_runner_owns_state") from None
        self.guard = threading.RLock()
        self.db = None
        try:
            self.db = sqlite3.connect(self.directory / "ledger.sqlite", timeout=60, check_same_thread=False, isolation_level=None)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS account(id INTEGER PRIMARY KEY CHECK(id=1),budget_nano INTEGER,charge_nano INTEGER,prior_nano INTEGER,halted_reason TEXT);
          CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,offset INTEGER,length INTEGER,metadata_json TEXT,status TEXT,attempt_count INTEGER,available_at REAL);
          CREATE INDEX IF NOT EXISTS pending_jobs ON jobs(status,available_at,ordinal);
          CREATE INDEX IF NOT EXISTS pending_order ON jobs(status,ordinal);
          CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,job_id TEXT,attempt_no INTEGER,started_at REAL,finished_at REAL,status TEXT,reserved_nano INTEGER,charged_nano INTEGER,usage_json TEXT,result_json TEXT,error_code TEXT,error_status INTEGER,UNIQUE(job_id,attempt_no));
          CREATE INDEX IF NOT EXISTS attempt_status ON attempts(status);
            """)
            self._initialize()
        except BaseException:
            self.close()
            raise

    @contextmanager
    def transaction(self):
        with self.guard:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def _initialize(self):
        with self.transaction() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='config_sha256'").fetchone()
            if row:
                if row[0] != digest(self.config):
                    raise GuardError("state_configuration_mismatch")
                if db.execute("SELECT value FROM metadata WHERE key='runner_sha256'").fetchone()[0] != file_hash(__file__):
                    raise GuardError("state_runner_changed")
                # An interrupted dispatch can have incurred its full charge.
                # Keep every reservation, count the attempt, and allow at most
                # the predeclared number of further attempts.
                db.execute("UPDATE attempts SET status='unknown_after_crash' WHERE status='inflight'")
                db.execute("UPDATE jobs SET status=CASE WHEN attempt_count<? THEN 'pending' ELSE 'exhausted' END WHERE status='inflight'", (self.config["max_attempts"],))
                return
            db.execute("INSERT INTO metadata VALUES('config_sha256',?)", (digest(self.config),))
            db.execute("INSERT INTO metadata VALUES('config_json',?)", (encoded(self.config).decode(),))
            db.execute("INSERT INTO metadata VALUES('ledger_id',?)", (str(uuid.uuid4()),))
            db.execute("INSERT INTO metadata VALUES('runner_sha256',?)", (file_hash(__file__),))
            db.execute("INSERT INTO account VALUES(1,?,?,?,NULL)", (usd_to_nano(self.config["budget_usd"]), usd_to_nano(self.config["prior_reserved_usd"]), usd_to_nano(self.config["prior_reserved_usd"])))
            with self.jobs_path.open("rb") as stream:
                ordinal = 0
                while True:
                    offset, line = stream.tell(), stream.readline()
                    if not line:
                        break
                    job = json.loads(line)
                    validate_job(job)
                    meta = {key: job[key] for key in META_FIELDS}
                    db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,'pending',0,0)", (job["job_id"], ordinal, offset, len(line), encoded(meta).decode()))
                    ordinal += 1
            if ordinal == 0:
                raise GuardError("empty_job_list")

    def close(self):
        if self.db is not None:
            self.db.close()
        fcntl.flock(self.lock_handle, fcntl.LOCK_UN)
        self.lock_handle.close()

    def recover_quota_halt(self):
        with self.transaction() as db:
            reason = db.execute("SELECT halted_reason FROM account").fetchone()[0]
            if reason not in QUOTA_CODES:
                raise GuardError("halt_is_not_quota_related")
            db.execute("UPDATE account SET halted_reason=NULL")

    def halted(self):
        with self.guard:
            return self.db.execute("SELECT halted_reason FROM account").fetchone()[0]

    def claim(self, *, phase_cap_nano=None, allowed_ids=None):
        with self.transaction() as db:
            account = db.execute("SELECT * FROM account").fetchone()
            if account["halted_reason"]:
                return None, "halted"
            extra = "" if allowed_ids is None else " AND job_id IN ("+",".join("?" for _ in allowed_ids)+")"
            parameters = () if allowed_ids is None else tuple(allowed_ids)
            # Full runs consume an ordinal prefix; this avoids sorting the entire
            # remaining 134k-row queue before every request. Canary IN queries
            # instead use the small ID set through SQLite's ordinary optimizer.
            table = "jobs INDEXED BY pending_order" if allowed_ids is None else "jobs"
            row = db.execute("SELECT * FROM "+table+" WHERE status='pending' AND available_at<=?"+extra+" ORDER BY ordinal LIMIT 1", (time.time(),)+parameters).fetchone()
            if row is None:
                pending = db.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'"+extra, parameters).fetchone()[0]
                return None, "waiting" if pending else "done"
            reserve = reservation_nano(self.config)
            cap = min(account["budget_nano"], phase_cap_nano) if phase_cap_nano is not None else account["budget_nano"]
            if account["charge_nano"] + reserve > cap:
                inflight = db.execute("SELECT COUNT(*) FROM attempts WHERE status='inflight'").fetchone()[0]
                return None, "waiting" if inflight else "budget_exhausted"
            attempt = row["attempt_count"] + 1
            if attempt > self.config["max_attempts"]:
                raise GuardError("attempt_bound_violated")
            db.execute("UPDATE account SET charge_nano=charge_nano+?", (reserve,))
            cur = db.execute("INSERT INTO attempts(job_id,attempt_no,started_at,status,reserved_nano,charged_nano) VALUES(?,?,?,'inflight',?,?)", (row["job_id"], attempt, time.time(), reserve, reserve))
            db.execute("UPDATE jobs SET status='inflight',attempt_count=? WHERE job_id=?", (attempt, row["job_id"]))
            return dict(row, attempt_id=cur.lastrowid, attempt_no=attempt), "claimed"

    def load_job(self, claim):
        with self.jobs_path.open("rb") as stream:
            stream.seek(claim["offset"])
            job = json.loads(stream.read(claim["length"]))
        validate_job(job)
        if job["job_id"] != claim["job_id"] or encoded({k: job[k] for k in META_FIELDS}).decode() != claim["metadata_json"]:
            raise GuardError("job_changed_during_run")
        return job

    def finish(self, claim, *, result=None, cost=None, error=None, missing=None, fatal=None, delay=0, cancelled=False):
        with self.transaction() as db:
            attempt = db.execute("SELECT * FROM attempts WHERE id=?", (claim["attempt_id"],)).fetchone()
            if attempt["status"] != "inflight":
                raise GuardError("attempt_already_finalized")
            charge = 0 if cancelled else (cost["conservative_nano"] if cost else attempt["reserved_nano"])
            if charge > attempt["reserved_nano"]:
                fatal = "actual_charge_exceeds_reservation"
            status = "cancelled_before_dispatch" if cancelled else ("success" if result is not None else "terminal_missing" if missing else "fatal" if fatal else "api_error")
            if result is not None:
                job_status = "success"
            elif missing or cancelled:
                job_status = "terminal_missing" if missing else "pending"
            elif claim["attempt_no"] >= self.config["max_attempts"]:
                job_status = "exhausted"
            elif (error and error["retryable"]) or fatal in QUOTA_CODES:
                job_status = "pending"
            else:
                job_status = "failed"
            if cancelled:
                # No API dispatch occurred; do not consume an attempt allowance.
                db.execute("DELETE FROM attempts WHERE id=?", (claim["attempt_id"],))
                db.execute("UPDATE jobs SET status='pending',attempt_count=attempt_count-1 WHERE job_id=?", (claim["job_id"],))
            else:
                db.execute("UPDATE attempts SET finished_at=?,status=?,charged_nano=?,usage_json=?,result_json=?,error_code=?,error_status=? WHERE id=?", (time.time(), status, charge, encoded(cost).decode() if cost else None, encoded(result).decode() if result else None, fatal or missing or (error or {}).get("code"), (error or {}).get("status"), claim["attempt_id"]))
                db.execute("UPDATE jobs SET status=?,available_at=? WHERE job_id=?", (job_status, time.time()+delay, claim["job_id"]))
            db.execute("UPDATE account SET charge_nano=charge_nano+?", (charge-attempt["charged_nano"],))
            if fatal:
                db.execute("UPDATE account SET halted_reason=COALESCE(halted_reason,?)", (fatal,))

    def summary(self):
        with self.guard:
            account = dict(self.db.execute("SELECT * FROM account").fetchone())
            jobs = dict(self.db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall())
            attempts = dict(self.db.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status").fetchall())
            costs = self.db.execute("SELECT COALESCE(SUM(charged_nano),0) FROM attempts").fetchone()[0]
            if costs + account["prior_nano"] != account["charge_nano"]:
                raise GuardError("ledger_charge_identity_failed")
            return dict(jobs=jobs, attempts=attempts, budget_usd=account["budget_nano"]/NANODOLLARS,
                        committed_or_unknown_usd=account["charge_nano"]/NANODOLLARS,
                        prior_reserved_usd=account["prior_nano"]/NANODOLLARS,
                        remaining_reservation_usd=(account["budget_nano"]-account["charge_nano"])/NANODOLLARS,
                        halted_reason=account["halted_reason"], config_sha256=digest(self.config),
                        ledger_id=self.db.execute("SELECT value FROM metadata WHERE key='ledger_id'").fetchone()[0],
                        cost_scope="usage-based cache-write upper charge plus all unresolved reservations; not an invoice")

    def export(self):
        with self.guard:
            for filename, success in (("results.jsonl", True), ("failures.jsonl", False)):
                target = self.directory / filename
                temp = target.with_suffix(".tmp")
                if success:
                    rows = self.db.execute("SELECT j.metadata_json,a.result_json,a.usage_json FROM jobs j JOIN attempts a ON j.job_id=a.job_id WHERE a.status='success' ORDER BY j.ordinal")
                else:
                    rows = self.db.execute("SELECT metadata_json,status,attempt_count FROM jobs WHERE status<>'success' ORDER BY ordinal")
                with temp.open("wb") as handle:
                    for row in rows:
                        meta = json.loads(row[0])
                        value = {**meta, **json.loads(row[1]), "cost": json.loads(row[2])} if success else {**meta, "status": row[1], "attempt_count": row[2]}
                        handle.write(encoded(value)+b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
            summary = self.summary()
            (self.directory / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
            jobs = summary["jobs"]
            pending, inflight = jobs.get("pending", 0), jobs.get("inflight", 0)
            terminal = sum(n for key, n in jobs.items() if key in ("success", "terminal_missing", "exhausted", "failed"))
            complete = terminal == sum(jobs.values()) and not pending and not inflight and not summary["halted_reason"]
            def metadata(key):
                row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
                return row[0] if row else None
            completion = dict(schema_version=1, status="complete" if complete else "incomplete",
                              planned_jobs=sum(jobs.values()), terminal_jobs=terminal,
                              pending_jobs=pending, inflight_jobs=inflight, success_jobs=jobs.get("success", 0),
                              failure_jobs=terminal-jobs.get("success", 0), jobs_sha256=self.config["jobs_sha256"],
                              config_sha256=digest(self.config), runner_sha256=file_hash(__file__),
                              ledger_id=summary["ledger_id"], state_directory_sha256=text_hash(str(self.directory.resolve())),
                              generation_freeze_sha256=metadata("generation_freeze_sha256"),
                              calibration_sha256=metadata("calibration_sha256"),
                              results_sha256=file_hash(self.directory/"results.jsonl"),
                              failures_sha256=file_hash(self.directory/"failures.jsonl"),
                              accounting_summary_sha256=file_hash(self.directory/"summary.json"),
                              committed_or_unknown_usd=summary["committed_or_unknown_usd"], halted_reason=summary["halted_reason"])
            target = self.directory / "completion.json"
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(completion, indent=2)+"\n")
            os.replace(temp, target)
            summary["generation_completion"] = completion["status"]
            return summary


class Pacer:
    def __init__(self, requests_per_second):
        self.guard = threading.Lock()
        self.interval = 1 / requests_per_second
        self.next_at = 0.

    def wait(self, ledger, stop=None):
        while not ledger.halted() and not (stop and stop.is_set()):
            with self.guard:
                delay = self.next_at-time.monotonic()
                if delay <= 0:
                    self.next_at = time.monotonic()+self.interval
                    return True
            time.sleep(min(delay, .2))
        return False

    def cooldown(self, delay):
        with self.guard:
            self.next_at = max(self.next_at, time.monotonic()+delay)


def expected_canary_ids(ledger):
    groups = {}
    with ledger.guard:
        for row in ledger.db.execute("SELECT job_id,metadata_json FROM jobs"):
            groups.setdefault(json.loads(row[1])["producer"], []).append(row[0])
    return {job_id for ids in groups.values() for job_id in sorted(ids, key=lambda j: text_hash("fee-calibration-v1|"+j))[:32]}


def canary_accounting(ledger):
    """Fee-only audit, excluding P values, reference labels, and task accuracy."""
    selected = expected_canary_ids(ledger)
    by_producer, selected_rows, attempts = {}, [], []
    with ledger.guard:
        for row in ledger.db.execute("SELECT job_id,metadata_json,status,attempt_count FROM jobs ORDER BY job_id"):
            producer = json.loads(row[1])["producer"]
            cell = by_producer.setdefault(producer, dict(total_jobs=0, selected_jobs=0, successful_jobs=0, known_cost_jobs=0, known_nano=0, unknown_nano=0))
            cell["total_jobs"] += 1
            if row[0] in selected:
                cell["selected_jobs"] += 1
                cell["successful_jobs"] += row[2] == "success"
                selected_rows.append((row[0], producer, row[2], row[3]))
        known_jobs = set()
        producers = {r[0]: r[1] for r in selected_rows}
        for row in ledger.db.execute("SELECT job_id,attempt_no,status,charged_nano,usage_json FROM attempts ORDER BY job_id,attempt_no"):
            if row[0] not in selected:
                continue
            record = tuple(row)
            attempts.append(record)
            if row[4] is not None:
                known_jobs.add(row[0])
                by_producer[producers[row[0]]]["known_nano"] += row[3]
            else:
                by_producer[producers[row[0]]]["unknown_nano"] += row[3]
        prior = ledger.db.execute("SELECT prior_nano FROM account").fetchone()[0]
    for job_id in known_jobs:
        by_producer[producers[job_id]]["known_cost_jobs"] += 1
    complete = bool(selected_rows) and all(r[2] not in ("pending", "inflight") for r in selected_rows)
    projection = Decimal(prior)
    estimable = all(c["known_cost_jobs"] and c["successful_jobs"] for c in by_producer.values())
    if estimable:
        for cell in by_producer.values():
            projection += cell["known_nano"] + cell["unknown_nano"]
            projection += Decimal(cell["known_nano"])*(cell["total_jobs"]-cell["selected_jobs"])/cell["known_cost_jobs"]
    projected_usd = float(projection/Decimal(NANODOLLARS)) if estimable else None
    return dict(complete=complete, estimable=estimable, selected_jobs=len(selected),
                by_producer=by_producer, projected_total_usd=projected_usd,
                canary_accounting_sha256=digest({"jobs": selected_rows, "attempts": attempts}),
                scope="Fee projection from fixed canary, plus unresolved charges; no P values or reference outcomes; not a spending guarantee")


def validate_fee_receipt(ledger, receipt_path, freeze):
    if receipt_path is None:
        raise GuardError("full_run_requires_fee_receipt")
    receipt = json.loads(Path(receipt_path).read_text())
    current = canary_accounting(ledger)
    if (receipt.get("status") != "passed" or not current["complete"] or not current["estimable"]
            or receipt.get("config_sha256") != digest(ledger.config)
            or receipt.get("ledger_id") != ledger.summary()["ledger_id"]
            or receipt.get("canary_job_ids_sha256") != freeze.get("canary_job_ids_sha256")
            or receipt.get("canary_accounting_sha256") != current["canary_accounting_sha256"]
            or receipt.get("projected_total_usd") != current["projected_total_usd"]
            or not 0 <= current["projected_total_usd"] <= float(ledger.config["budget_usd"])):
        raise GuardError("fee_calibration_not_passed")
    return current


def run_jobs(jobs_path, config, state, request_fn, *, calibration_receipt, generation_freeze, limit=None, resume_after_quota=False, progress=None, phase_cap_usd=None, job_ids_path=None, fee_receipt=None):
    """Injected request_fn permits synthetic validation without credentials."""
    validate_execution_gates(config, calibration_receipt, generation_freeze, state, job_ids_path)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise GuardError("invalid_dispatch_limit")
    phase_cap_nano = None if phase_cap_usd is None else usd_to_nano(phase_cap_usd)
    if phase_cap_nano is not None and not 0 < phase_cap_nano <= usd_to_nano(config["budget_usd"]):
        raise GuardError("invalid_phase_cap")
    allowed_ids = None
    if job_ids_path is not None:
        if phase_cap_nano is None or phase_cap_nano > NANODOLLARS:
            raise GuardError("canary_requires_cumulative_one_dollar_cap")
        manifest = json.loads(Path(job_ids_path).read_text())
        if manifest.get("jobs_sha256") != config["jobs_sha256"] or manifest.get("selection_rule") != "32-per-producer-smallest-sha256-fee-calibration-v1-job-id":
            raise GuardError("invalid_canary_manifest")
        allowed_ids = tuple(manifest.get("job_ids", ()))
        if not allowed_ids or len(set(allowed_ids)) != len(allowed_ids):
            raise GuardError("invalid_canary_ids")
    ledger = Ledger(state, config, jobs_path, require_existing=True)
    try:
        if allowed_ids is not None:
            if set(allowed_ids) != expected_canary_ids(ledger):
                raise GuardError("canary_selection_changed")
        else:
            validate_fee_receipt(ledger, fee_receipt, json.loads(Path(generation_freeze).read_text()))
        with ledger.guard:
            for key, value in (("generation_freeze_sha256", file_hash(generation_freeze)), ("calibration_sha256", file_hash(calibration_receipt))):
                old = ledger.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
                if old is not None and old[0] != value:
                    raise GuardError("execution_receipt_changed_after_start")
                ledger.db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)", (key, value))
            ledger.db.execute("CREATE TABLE IF NOT EXISTS invocations(id INTEGER PRIMARY KEY,started_at REAL,phase_cap_nano INTEGER,allowlist_sha256 TEXT,dispatch_limit INTEGER,fee_receipt_sha256 TEXT)")
            ledger.db.execute("INSERT INTO invocations(started_at,phase_cap_nano,allowlist_sha256,dispatch_limit,fee_receipt_sha256) VALUES(?,?,?,?,?)", (time.time(), phase_cap_nano, file_hash(job_ids_path) if job_ids_path else None, limit, file_hash(fee_receipt) if fee_receipt else None))
    except BaseException:
        ledger.close()
        raise
    pacer = Pacer(config["requests_per_second"])
    stop = threading.Event()
    dispatched, dispatch_lock = [0], threading.Lock()
    fatal_parse = {"response_model_changed", "response_fingerprint_changed", "response_service_tier_changed"}

    def process_one():
        if not pacer.wait(ledger, stop):
            return "halted"
        with dispatch_lock:
            if stop.is_set():
                return "halted"
            if limit is not None and dispatched[0] >= limit:
                return "limit"
            claim, reason = ledger.claim(phase_cap_nano=phase_cap_nano, allowed_ids=allowed_ids)
            if claim is None:
                return reason
            dispatched[0] += 1
        try:
            job = ledger.load_job(claim)
        except Exception:
            ledger.finish(claim, fatal="job_changed_during_run", cancelled=False)
            return "halted"
        if ledger.halted() or stop.is_set():
            ledger.finish(claim, cancelled=True)
            return "halted"
        try:
            response = request_fn(request_body(job, config))
        except Exception as exc:
            error = safe_error(exc)
            delay = max(error["retry_after_seconds"] or 0., 2 ** (claim["attempt_no"]-1))
            if error["retryable"]:
                pacer.cooldown(delay)
            ledger.finish(claim, error=error, fatal=error["code"] if error["fatal"] else None, delay=delay)
            return "halted" if error["fatal"] else "retry" if error["retryable"] else "failed"
        cost = None
        try:
            cost = usage_cost(response.get("usage"), config)
        except (GuardError, AttributeError) as exc:
            code = str(exc) if isinstance(exc, GuardError) else "invalid_response"
            ledger.finish(claim, fatal=code)
            return "halted"
        try:
            result = {**{key: job[key] for key in META_FIELDS}, **parse_response(response, config)}
        except GuardError as exc:
            code = str(exc)
            ledger.finish(claim, cost=None if code in fatal_parse else cost,
                          fatal=code if code in fatal_parse else None, missing=None if code in fatal_parse else code)
            return "halted" if code in fatal_parse else "missing"
        except Exception:
            ledger.finish(claim, cost=cost, missing="invalid_response")
            return "missing"
        ledger.finish(claim, result=result, cost=cost)
        return "success"

    def worker():
        try:
            while True:
                reason = process_one()
                if reason in ("halted", "done", "limit", "budget_exhausted"):
                    return reason
                if reason == "waiting":
                    time.sleep(.2)
        except BaseException:
            stop.set()
            try:
                with ledger.transaction() as db:
                    db.execute("UPDATE account SET halted_reason=COALESCE(halted_reason,'runner_unexpected_error')")
            except Exception:
                pass  # Already-reserved attempts remain durable even on disk errors.
            raise

    try:
        if resume_after_quota:
            ledger.recover_quota_halt()
        # Do not launch a concurrent batch before one successful API request in
        # this invocation. Quota/auth/unsupported-model errors thus make one call.
        while True:
            first = process_one()
            if first == "success":
                break
            if first in ("halted", "done", "limit", "budget_exhausted"):
                return ledger.export()
            if first == "waiting":
                time.sleep(.2)
        pool = ThreadPoolExecutor(max_workers=config["concurrency"])
        try:
            futures = [pool.submit(worker) for _ in range(config["concurrency"])]
            while not all(f.done() for f in futures):
                if progress:
                    progress(ledger.summary())
                time.sleep(.25 if progress is None else 10)
            for future in futures:
                future.result()
        except BaseException:
            stop.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return ledger.export()
    finally:
        ledger.close()


def read_existing_key(path):
    # Credential reuse must already be authorized. Never report contents/errors.
    try:
        text = Path(path).read_text()
    except (OSError, UnicodeError):
        raise GuardError("key_file_unreadable") from None
    found = set(re.findall(r"(?<![A-Za-z0-9_-])(sk-[A-Za-z0-9_-]{16,})(?![A-Za-z0-9_-])", text))
    if len(found) != 1:
        raise GuardError("key_file_requires_one_unique_key")
    return found.pop()


def make_canary_manifest(jobs_path, output_path):
    """Predetermined 32 requests per producer; selected using IDs, not outcomes."""
    if Path(output_path).exists():
        raise GuardError("canary_output_exists")
    groups = {}
    with Path(jobs_path).open() as handle:
        for line in handle:
            job = json.loads(line)
            validate_job(job)
            groups.setdefault(job["producer"], []).append(job["job_id"])
    selected = [j for ids in groups.values() for j in sorted(ids, key=lambda j: text_hash("fee-calibration-v1|"+j))[:32]]
    result = {"schema_version": 1, "selection_rule": "32-per-producer-smallest-sha256-fee-calibration-v1-job-id",
              "jobs_sha256": file_hash(jobs_path), "count_per_producer": 32,
              "job_ids": sorted(selected, key=lambda j: text_hash("fee-calibration-v1|"+j))}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2)+"\n")
    return {"status": "canary_selected_without_outcomes", "jobs": len(selected), "sha256": file_hash(output_path)}


def make_fee_report(jobs_path, config, state, calibration, freeze_path, output):
    """Create a fee-only review candidate; the authoring agent must review it."""
    if Path(output).exists():
        raise GuardError("fee_report_already_exists")
    validate_execution_gates(config, calibration, freeze_path, state)
    ledger = Ledger(state, config, jobs_path, require_existing=True)
    try:
        report = {**canary_accounting(ledger), "status": "review_required", "config_sha256": digest(config),
                  "ledger_id": ledger.summary()["ledger_id"],
                  "canary_job_ids_sha256": json.loads(Path(freeze_path).read_text()).get("canary_job_ids_sha256")}
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2)+"\n")
        return report
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("example-config")
    canary = sub.add_parser("make-canary")
    canary.add_argument("--jobs", type=Path, required=True)
    canary.add_argument("--output", type=Path, required=True)
    for name in ("check", "init-ledger", "fee-report", "finalize", "run"):
        command = sub.add_parser(name)
        command.add_argument("--jobs", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        if name in ("init-ledger", "fee-report", "finalize", "run"):
            command.add_argument("--state", type=Path, required=True)
        if name in ("fee-report", "finalize", "run"):
            command.add_argument("--calibration-receipt", type=Path, required=True)
            command.add_argument("--generation-freeze", type=Path, required=True)
        if name == "fee-report":
            command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--key-file", type=Path, required=True)
            command.add_argument("--limit", type=int)
            command.add_argument("--resume-after-quota", action="store_true")
            command.add_argument("--phase-cap-usd", type=float)
            command.add_argument("--job-ids", type=Path)
            command.add_argument("--fee-receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "example-config":
            print(json.dumps(config_example(), indent=2))
            return
        if args.command == "make-canary":
            print(json.dumps(make_canary_manifest(args.jobs, args.output)))
            return
        config = validate_config(json.loads(args.config.read_text()))
        if file_hash(args.jobs) != config["jobs_sha256"]:
            raise GuardError("jobs_file_hash_mismatch")
        if args.command == "check":
            n = 0
            with args.jobs.open() as handle:
                for line in handle:
                    validate_job(json.loads(line))
                    n += 1
            print(json.dumps(dict(status="offline_inputs_checked", jobs=n, state=config.get("state"), config_sha256=digest(config), per_dispatch_reservation_usd=reservation_nano(config)/NANODOLLARS, api_calls=0)))
            return
        if args.command == "init-ledger":
            if config.get("state") != "ready":
                raise GuardError("awaiting_api_calibration")
            if (args.state / "ledger.sqlite").exists():
                raise GuardError("ledger_already_exists")
            ledger = Ledger(args.state, config, args.jobs)
            try:
                print(json.dumps({**ledger.summary(), "state_directory_sha256": text_hash(str(args.state.resolve())), "api_calls": 0}))
            finally:
                ledger.close()
            return
        if args.command == "fee-report":
            print(json.dumps(make_fee_report(args.jobs, config, args.state, args.calibration_receipt, args.generation_freeze, args.output)))
            return
        if args.command == "finalize":
            validate_execution_gates(config, args.calibration_receipt, args.generation_freeze, args.state)
            ledger = Ledger(args.state, config, args.jobs, require_existing=True)
            try:
                summary = ledger.export()
                print(json.dumps(summary))
                if summary["generation_completion"] != "complete":
                    raise SystemExit(3)
            finally:
                ledger.close()
            return
        validate_execution_gates(config, args.calibration_receipt, args.generation_freeze, args.state, args.job_ids)
        for logger_name in ("openai", "httpx", "httpcore"):
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        from openai import OpenAI
        client = OpenAI(api_key=read_existing_key(args.key_file), base_url="https://api.openai.com/v1", max_retries=0, timeout=config["timeout_seconds"])
        def request(body):
            return client.chat.completions.create(**body).model_dump(mode="json")
        try:
            summary = run_jobs(args.jobs, config, args.state, request,
                               calibration_receipt=args.calibration_receipt, generation_freeze=args.generation_freeze,
                               limit=args.limit, resume_after_quota=args.resume_after_quota,
                               phase_cap_usd=args.phase_cap_usd, job_ids_path=args.job_ids,
                               fee_receipt=args.fee_receipt,
                               progress=lambda row: print(json.dumps(row), flush=True))
            print(json.dumps(summary), flush=True)
        finally:
            client.close()
    except KeyboardInterrupt:
        print(json.dumps({"status": "stopped", "reason": "interrupted_reservations_preserved"}), flush=True)
        raise SystemExit(130) from None
    except GuardError as exc:
        print(json.dumps({"status": "stopped", "reason": str(exc)}), flush=True)
        raise SystemExit(2) from None
    except Exception:
        # In particular, do not let an SDK exception render a request or key.
        print(json.dumps({"status": "stopped", "reason": "unexpected_error_inspect_local_ledger"}), flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
