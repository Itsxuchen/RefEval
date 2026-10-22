"""WildIFEval primary judge P through Gemini API Batch Mode (protocol section 14).

Batch Mode bills half the standard rate. Every frozen job is one JSONL request line keyed by its job_id, with the
pinned upstream prompt as the only user message, temperature 0, thinking level LOW and a 2048-token output cap.
The parser, job index and prompt checks are shared with ``src.wildifeval_vertex_judge``.

The API key is read at run time from the environment variable GEMINI_API_KEY or from the untracked local file
``secrete_key.rtf``. It is never printed, logged or written anywhere. No reference score R is read here.

    python -m src.wildifeval_gemini_batch freeze --cap-usd 100
    python -m src.wildifeval_gemini_batch run --wait
    python -m src.wildifeval_gemini_batch status

``run`` is a resumable state machine: two synthetic online probes, the fixed 160-request canary as one batch job,
the cost gate, the remaining jobs in shards that respect the enqueued-token limit, at most two resubmissions of
requests without a response, and the export of P. Batch jobs keep running at Google after this script exits.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable

from src import wildifeval_vertex_judge as online

ROOT = online.ROOT
JOBS_PATH = online.JOBS_PATH
KEY_FILE = ROOT / "secrete_key.rtf"
RUN_DIR = ROOT / "data" / "interim" / "wildifeval_external" / "flashlite_gemini_batch"          # private
REPORT_DIR = ROOT / "artifacts" / "reports" / "wildifeval_external" / "flashlite_gemini_batch"  # public, no text
StopRun = online.StopRun
Job = online.Job

PROTOCOL_VERSION = "wildifeval-p-gemini-batch-v3"
CONFIG: dict[str, Any] = {"model": online.CONFIG["model"], "thinking_level": online.CONFIG["thinking_level"],
                          "temperature": online.CONFIG["temperature"],
                          "max_output_tokens": online.CONFIG["max_output_tokens"],
                          "mode": "gemini-api-batch", "price_in_per_m": 0.125, "price_out_per_m": 0.75}
PRICE_SOURCE = ("https://ai.google.dev/gemini-api/docs/pricing, Batch row for gemini-3.1-flash-lite, read "
                "2026-10-21: 0.125 / 0.75 USD per million tokens, thinking counted as output")
MAX_SHARD = 10_000              # requests per batch job
MAX_SHARD_BYTES = 28_000_000    # job-file bytes per batch job: about 8M input tokens, under the Tier-1 10M limit
MAX_RETRY_ROUNDS = 2
PROJECTION_MARGIN = online.PROJECTION_MARGIN
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
POLL_SECONDS = 120
DISPLAY_PREFIX = "wildifeval-flashlite-v3"


class RateLimited(RuntimeError):
    """Batch creation refused because the enqueued-token or concurrency limit is reached."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_api_key(path: Path | None = None) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not re.fullmatch(r"AIza[0-9A-Za-z_\-]{30,}", key):
        try:
            text = (path or KEY_FILE).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise StopRun(f"no GEMINI_API_KEY in the environment and no key file: {exc.__class__.__name__}") from None
        found = re.search(r'GEMINI_API_KEY\s*=\s*"?(AIza[0-9A-Za-z_\-]{30,})', text)
        if not found:
            raise StopRun("the key file has no GEMINI_API_KEY entry")
        key = found.group(1)
    return key


def batch_cost(usage: dict[str, int]) -> float:
    out = usage.get("candidates", 0) + usage.get("thoughts", 0)
    return (usage.get("prompt", 0) * CONFIG["price_in_per_m"] + out * CONFIG["price_out_per_m"]) / 1e6


def request_line(job_id: str, prompt: str) -> dict[str, Any]:
    return {"key": job_id,
            "request": {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generation_config": {"temperature": CONFIG["temperature"],
                                              "max_output_tokens": CONFIG["max_output_tokens"],
                                              "thinking_config": {"thinking_level": CONFIG["thinking_level"]}}}}


def make_shards(jobs: list[Job], max_size: int | None = None, max_bytes: int | None = None) -> list[list[Job]]:
    """First-fit in job-file order under a request cap and a byte cap."""
    max_size, max_bytes = max_size or MAX_SHARD, max_bytes or MAX_SHARD_BYTES
    shards: list[list[Job]] = []
    sizes: list[int] = []
    for job in jobs:
        for i, shard in enumerate(shards):
            if len(shard) < max_size and sizes[i] + job.length <= max_bytes:
                shard.append(job); sizes[i] += job.length
                break
        else:
            shards.append([job]); sizes.append(job.length)
    return shards


def plan_sha256(shards: list[list[Job]]) -> str:
    return hashlib.sha256("\n\n".join("\n".join(j.job_id for j in s) for s in shards).encode()).hexdigest()


def _first(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def parse_result_line(obj: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Return (key, response record or None, error). A line is a keyed response or a keyed status."""
    key = obj.get("key")
    resp = obj.get("response")
    if resp is None and "candidates" in obj:
        resp = obj
    if not resp:
        err = _first(obj, "error", "status", default="no response")
        return key, None, (json.dumps(err) if not isinstance(err, str) else err)[:300]
    cands = resp.get("candidates") or []
    cand = cands[0] if cands else {}
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text") or "" for p in parts if not p.get("thought")) or None
    usage = _first(resp, "usageMetadata", "usage_metadata", default={}) or {}
    rec = {"text": text, "finish_reason": _first(cand, "finishReason", "finish_reason"),
           "usage": {"prompt": int(_first(usage, "promptTokenCount", "prompt_token_count", default=0)),
                     "candidates": int(_first(usage, "candidatesTokenCount", "candidates_token_count", default=0)),
                     "thoughts": int(_first(usage, "thoughtsTokenCount", "thoughts_token_count", default=0))},
           "model_version": _first(resp, "modelVersion", "model_version"),
           "response_id": _first(resp, "responseId", "response_id")}
    return key, rec, None


# --------------------------------------------------------------------------- Gemini API adapter (not used in tests)

class GeminiBatchBackend:
    def __init__(self, api_key: str):
        from google import genai  # imported lazily so tests run without the SDK
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=600_000, retry_options=types.HttpRetryOptions(attempts=1)))

    def upload(self, local: Path, display_name: str) -> str:
        f = self.client.files.upload(file=str(local), config=self.types.UploadFileConfig(
            display_name=display_name, mime_type="jsonl"))
        return f.name

    def create(self, file_ref: str, display_name: str) -> str:
        try:
            job = self.client.batches.create(model=CONFIG["model"], src=file_ref,
                                             config={"display_name": display_name})
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "code", None) == 429:
                raise RateLimited(str(exc)[:300]) from exc
            raise
        return job.name

    def find(self, display_name: str) -> str | None:
        for job in self.client.batches.list():
            if getattr(job, "display_name", None) == display_name:
                return job.name
        return None

    def get(self, name: str) -> tuple[str, str | None, str | None]:
        job = self.client.batches.get(name=name)
        state = getattr(job.state, "name", None) or str(job.state)
        error = getattr(job, "error", None)
        dest = getattr(job, "dest", None)
        return str(state), (str(error)[:500] if error else None), (getattr(dest, "file_name", None) if dest else None)

    def download(self, result_ref: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.client.files.download(file=result_ref))

    def probe(self, text: str) -> dict[str, Any]:
        return online.default_request(self.client, text)


# --------------------------------------------------------------------------- ledger

class BatchLedger:
    def __init__(self, path: Path, cap_usd: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = cap_usd
        self.con = sqlite3.connect(path, isolation_level=None)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("""CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, producer TEXT, phase TEXT, state TEXT, submissions INTEGER DEFAULT 0,
            batch_key TEXT, raw_text TEXT, finish_reason TEXT, prompt_tokens INTEGER, candidates_tokens INTEGER,
            thoughts_tokens INTEGER, cost_usd REAL DEFAULT 0, model_version TEXT, error TEXT, updated_utc TEXT)""")
        self.con.execute("""CREATE TABLE IF NOT EXISTS batches (
            key TEXT PRIMARY KEY, phase TEXT, round INTEGER, n INTEGER, local_path TEXT, file_ref TEXT,
            display_name TEXT, name TEXT, state TEXT, error TEXT, result_ref TEXT, submitted_utc TEXT,
            collected INTEGER DEFAULT 0, settled_usd REAL DEFAULT 0, matched INTEGER DEFAULT 0,
            expected_usd REAL DEFAULT 0)""")
        self.con.execute("CREATE TABLE IF NOT EXISTS events (utc TEXT, kind TEXT, detail TEXT)")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")

    def event(self, kind: str, detail: str) -> None:
        self.con.execute("INSERT INTO events VALUES (?, ?, ?)", (now(), kind, detail))

    def meta_get(self, key: str, default: str | None = None) -> str | None:
        row = self.con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def meta_set(self, key: str, value: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def settled(self) -> float:
        jobs = self.con.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM jobs").fetchone()[0]
        return float(jobs) + float(self.meta_get("probe_usd", "0"))

    def register(self, jobs: list[Job]) -> None:
        self.con.execute("BEGIN")
        self.con.executemany("INSERT OR IGNORE INTO jobs (job_id, producer, state, updated_utc) "
                             "VALUES (?, ?, 'pending', ?)", [(j.job_id, j.producer, now()) for j in jobs])
        self.con.execute("COMMIT")

    def job_states(self) -> dict[str, sqlite3.Row]:
        return {r["job_id"]: r for r in self.con.execute("SELECT * FROM jobs")}

    def batches(self, phase: str | None = None) -> list[sqlite3.Row]:
        if phase is None:
            return self.con.execute("SELECT * FROM batches ORDER BY key").fetchall()
        return self.con.execute("SELECT * FROM batches WHERE phase = ? ORDER BY key", (phase,)).fetchall()


# --------------------------------------------------------------------------- workflow

class Context:
    def __init__(self, ledger: BatchLedger, jobs: list[Job], reader: online.PromptReader, backend: Any,
                 run_dir: Path, report_dir: Path, log: Callable[[str], None] = print):
        self.ledger, self.jobs, self.reader, self.backend = ledger, jobs, reader, backend
        self.run_dir, self.report_dir, self.log = run_dir, report_dir, log
        self.by_id = {j.job_id: j for j in jobs}
        self.canary = set(online.canary_ids(jobs))


def probe(ctx: Context) -> None:
    """Two unrelated synthetic online requests at the standard rate; they check access, model and parser."""
    if ctx.ledger.meta_get("probe_ok") == "1":
        return
    for name, text, expected in online.SYNTHETIC_PROBES:
        try:
            rec = ctx.backend.probe(text)
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {str(exc)[:300]}"
            ctx.ledger.event("probe_failed", f"{name}: {message}")
            raise StopRun(f"probe {name} failed: {message}") from exc
        cost = online.usage_cost(rec["usage"])
        ctx.ledger.meta_set("probe_usd", f"{float(ctx.ledger.meta_get('probe_usd', '0')) + cost:.9f}")
        label, _ = online.parse_verdict(rec["text"])
        ok = bool((rec["text"] or "").strip()) and (expected is None or label == int(expected == "yes"))
        ctx.ledger.event("probe", json.dumps({"probe": name, "model_version": rec["model_version"], "text": rec["text"],
                                              "usage": rec["usage"], "cost_usd": cost, "ok": ok}))
        ctx.log(f"probe {name}: model_version={rec['model_version']} answer={rec['text']!r} usage={rec['usage']}")
        if not ok:
            raise StopRun(f"probe {name} did not return a usable answer")
        if rec["model_version"] and not str(rec["model_version"]).startswith(CONFIG["model"]):
            raise StopRun(f"returned model {rec['model_version']} differs from {CONFIG['model']}")
    ctx.ledger.meta_set("probe_ok", "1")


def expected_usd(ctx: Context, shard: list[Job]) -> float:
    """Projected cost times the margin once the canary receipt exists; before that a worst-case bound."""
    receipt = ctx.ledger.meta_get("receipt")
    if receipt is None:   # the job line is longer than its prompt, so two bytes per token of the line bounds input
        return sum((j.length / 2 + 50) * CONFIG["price_in_per_m"] / 1e6
                   + 2 * CONFIG["max_output_tokens"] * CONFIG["price_out_per_m"] / 1e6 for j in shard)
    means = {p: v["mean_cost_usd"] for p, v in json.loads(receipt)["producers"].items()}
    return PROJECTION_MARGIN * sum(means[j.producer] for j in shard)


def committed(ctx: Context) -> float:
    pending = ctx.ledger.con.execute("SELECT COALESCE(SUM(expected_usd), 0) FROM batches WHERE collected = 0").fetchone()[0]
    return ctx.ledger.settled() + float(pending)


def active(ctx: Context) -> int:
    return sum(1 for b in ctx.ledger.batches() if b["name"] and b["state"] not in TERMINAL)


def submit(ctx: Context, phase: str, round_: int, plan: list[list[Job]]) -> None:
    """Write, upload and create every shard of a plan that has no batch job yet. Safe to call again."""
    con = ctx.ledger.con
    for i, shard in enumerate(plan):
        key = f"{phase}-r{round_}-s{i:02d}"
        row = con.execute("SELECT * FROM batches WHERE key = ?", (key,)).fetchone()
        if row is not None and row["name"]:
            continue
        limit = int(ctx.ledger.meta_get("max_active", "1000"))
        if active(ctx) >= limit:
            return
        display = f"{DISPLAY_PREFIX}-{key}"
        if row is None:
            expected = expected_usd(ctx, shard)
            if committed(ctx) + expected > ctx.ledger.cap + 1e-12:
                raise StopRun(f"submitting {key} could exceed the cost cap "
                              f"(committed {committed(ctx):.2f} + expected {expected:.2f} > {ctx.ledger.cap:.2f} USD)")
            local = ctx.run_dir / "batch_inputs" / f"{key}.jsonl"
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("w", encoding="utf-8") as out:
                for job in shard:
                    out.write(json.dumps(request_line(job.job_id, ctx.reader.prompt(job)), ensure_ascii=False) + "\n")
            con.execute("BEGIN")
            con.execute("INSERT INTO batches (key, phase, round, n, local_path, display_name, state, expected_usd) "
                        "VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)",
                        (key, phase, round_, len(shard), str(local), display, expected))
            con.executemany("UPDATE jobs SET phase = ?, state = 'submitted', batch_key = ?, "
                            "submissions = submissions + 1, updated_utc = ? WHERE job_id = ?",
                            [(phase, key, now(), j.job_id) for j in shard])
            con.execute("COMMIT")
            row = con.execute("SELECT * FROM batches WHERE key = ?", (key,)).fetchone()
        name = ctx.backend.find(display) if row["file_ref"] else None   # adopt a job created before a lost reply
        if name is None:
            file_ref = row["file_ref"]
            if not file_ref:
                file_ref = ctx.backend.upload(Path(row["local_path"]), display)
                con.execute("UPDATE batches SET file_ref = ? WHERE key = ?", (file_ref, key))
            try:
                name = ctx.backend.create(file_ref, display)
            except RateLimited as exc:
                running = active(ctx)
                if running == 0:
                    raise StopRun(f"{key} was refused with no batch job running: {exc}") from exc
                ctx.ledger.meta_set("max_active", str(running))
                ctx.ledger.event("rate_limited", json.dumps({"key": key, "active": running}))
                ctx.log(f"{key} waits: the enqueued-token limit allows {running} running batch job(s)")
                return
        con.execute("UPDATE batches SET name = ?, state = 'JOB_STATE_PENDING', submitted_utc = ? WHERE key = ?",
                    (name, now(), key))
        ctx.ledger.event("submit", json.dumps({"key": key, "name": name, "n": len(shard)}))
        ctx.log(f"submitted {key}: {len(shard)} requests as {name}")


def ensure_round(ctx: Context, phase: str, round_: int, make: Callable[[], list[list[Job]]] | None) -> bool:
    key = f"plan:{phase}-r{round_}"
    stored = ctx.ledger.meta_get(key)
    if stored is None:
        if make is None:
            return False
        shards = make()
        if not shards:
            return False
        stored = json.dumps([[j.job_id for j in s] for s in shards])
        ctx.ledger.meta_set(key, stored)
    submit(ctx, phase, round_, [[ctx.by_id[i] for i in s] for s in json.loads(stored)])
    return True


def poll(ctx: Context) -> None:
    for row in ctx.ledger.batches():
        if row["name"] and not row["collected"] and row["state"] not in TERMINAL:
            state, error, result_ref = ctx.backend.get(row["name"])
            if state != row["state"] or result_ref != row["result_ref"]:
                ctx.ledger.con.execute("UPDATE batches SET state = ?, error = ?, result_ref = ? WHERE key = ?",
                                       (state, error, result_ref, row["key"]))
                ctx.ledger.event("state", json.dumps({"key": row["key"], "state": state, "error": error}))


def collect(ctx: Context) -> None:
    """Download finished results, match each line to its job by key and settle its cost."""
    con = ctx.ledger.con
    for row in ctx.ledger.batches():
        if row["collected"] or row["state"] not in TERMINAL:
            continue
        key = row["key"]
        expected = {r["job_id"] for r in
                    con.execute("SELECT job_id FROM jobs WHERE batch_key = ? AND state = 'submitted'", (key,))}
        seen: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
        unmatched = 0
        if row["result_ref"]:
            local = ctx.run_dir / "batch_outputs" / f"{key}.jsonl"
            ctx.backend.download(row["result_ref"], local)
            with local.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    job_id, rec, err = parse_result_line(json.loads(line))
                    if job_id not in expected or job_id in seen:
                        unmatched += 1
                        continue
                    seen[job_id] = (rec, err)
        elif row["state"] == "JOB_STATE_SUCCEEDED":
            raise StopRun(f"{key} succeeded but reports no result file")
        if unmatched:
            ctx.ledger.event("unmatched", json.dumps({"key": key, "lines": unmatched}))
            raise StopRun(f"{key}: {unmatched} result lines did not match exactly one submitted job; "
                          "nothing was settled for this batch")
        con.execute("BEGIN")
        settled, versions = 0.0, set()
        for job_id in expected:
            rec, err = seen.get(job_id, (None, "no result line" if row["state"] == "JOB_STATE_SUCCEEDED"
                                         else f"batch {row['state']}: {row['error']}"))
            if rec is None:
                con.execute("UPDATE jobs SET state = 'error', error = ?, updated_utc = ? WHERE job_id = ?",
                            (err, now(), job_id))
                continue
            cost = batch_cost(rec["usage"])
            settled += cost
            versions.add(str(rec["model_version"]))
            con.execute("UPDATE jobs SET state = 'done', raw_text = ?, finish_reason = ?, prompt_tokens = ?, "
                        "candidates_tokens = ?, thoughts_tokens = ?, cost_usd = ?, model_version = ?, error = NULL, "
                        "updated_utc = ? WHERE job_id = ?",
                        (rec["text"], rec["finish_reason"], rec["usage"]["prompt"], rec["usage"]["candidates"],
                         rec["usage"]["thoughts"], cost, rec["model_version"], now(), job_id))
        con.execute("UPDATE batches SET collected = 1, settled_usd = ?, matched = ? WHERE key = ?",
                    (settled, len(seen), key))
        con.execute("COMMIT")
        ctx.ledger.event("collect", json.dumps({"key": key, "state": row["state"], "matched": len(seen),
                                                "expected": len(expected), "settled_usd": settled,
                                                "model_versions": sorted(versions)}))
        ctx.log(f"collected {key}: {sum(1 for r, _ in seen.values() if r)}/{len(expected)} answered, {settled:.4f} USD")
        bad = [v for v in versions if v != "None" and not v.startswith(CONFIG["model"])]
        if bad:
            raise StopRun(f"{key} returned model versions {bad}, not {CONFIG['model']}")
        if ctx.ledger.settled() > ctx.ledger.cap:
            ctx.ledger.event("cap_exceeded", f"{ctx.ledger.settled():.4f}")
            raise StopRun("settled cost exceeds the cap; nothing further is submitted")


def uncollected(ctx: Context, phase: str | None = None) -> list[str]:
    return [b["key"] for b in ctx.ledger.batches(phase) if not b["collected"]]


def projection(ctx: Context) -> dict[str, Any]:
    rows = ctx.ledger.job_states()
    receipt: dict[str, Any] = {"producers": {}}
    total = 0.0
    for producer in online.PRODUCERS:
        canary = [rows[j.job_id] for j in ctx.jobs if j.producer == producer and j.job_id in ctx.canary]
        done = [r for r in canary if r["state"] == "done"]
        n_jobs = sum(1 for j in ctx.jobs if j.producer == producer)
        if not done:
            raise StopRun(f"no answered canary requests for {producer}")
        mean = sum(r["cost_usd"] for r in done) / len(done)
        receipt["producers"][producer] = {
            "canary": len(canary), "answered": len(done),
            "parsed": sum(online.parse_verdict(r["raw_text"])[0] is not None for r in done),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in done) / len(done),
            "mean_thoughts_tokens": sum(r["thoughts_tokens"] for r in done) / len(done),
            "mean_cost_usd": mean, "jobs": n_jobs, "projected_usd": mean * n_jobs}
        total += mean * n_jobs
    receipt.update({"projected_total_usd": total, "margin": PROJECTION_MARGIN, "cap_usd": ctx.ledger.cap,
                    "passed": total * PROJECTION_MARGIN <= ctx.ledger.cap, "settled_usd": ctx.ledger.settled(),
                    "rates": {k: CONFIG[k] for k in ("price_in_per_m", "price_out_per_m")}, "utc": now()})
    return receipt


def retry_candidates(ctx: Context) -> list[Job]:
    ids = {r[0] for r in ctx.ledger.con.execute(
        "SELECT job_id FROM jobs WHERE state = 'error' AND submissions <= ?", (MAX_RETRY_ROUNDS,))}
    return [j for j in ctx.jobs if j.job_id in ids]


def export(ctx: Context, jobs_path: Path) -> dict[str, Any]:
    rows = ctx.ledger.job_states()
    states: dict[str, int] = {}
    labels = {"pass": 0, "fail": 0, "missing": 0}
    finish: dict[str, int] = {}
    versions: set[str] = set()
    out_path = ctx.run_dir / "results.jsonl"
    with jobs_path.open() as source, out_path.open("w", encoding="utf-8") as out:
        for line in source:
            job = json.loads(line)
            r = rows.get(job["job_id"])
            state = r["state"] if r else "not_started"
            states[state] = states.get(state, 0) + 1
            ids = {k: job[k] for k in ("job_id", "unit_id", "cluster_id", "task_id", "producer", "criterion_id",
                                        "criterion_hash", "criterion_index", "output_hash", "prompt_sha256", "split", "k")}
            base = {"parser_version": online.PARSER_VERSION, "model": CONFIG["model"], "mode": CONFIG["mode"],
                    "submissions": r["submissions"] if r else 0}
            if state != "done":
                labels["missing"] += 1
                out.write(json.dumps(ids | base | {"label": None, "status": "no_response" if state == "error" else state,
                                                   "error": r["error"] if r else None}, ensure_ascii=False) + "\n")
                continue
            label, position = online.parse_verdict(r["raw_text"])
            labels["missing" if label is None else ("pass" if label else "fail")] += 1
            finish[str(r["finish_reason"])] = finish.get(str(r["finish_reason"]), 0) + 1
            versions.add(str(r["model_version"]))
            out.write(json.dumps(ids | base | {
                "label": label, "answer_word_position": position,
                "status": "success" if label is not None else "unparseable",
                "model_version": r["model_version"], "raw_text": r["raw_text"], "finish_reason": r["finish_reason"],
                "prompt_tokens": r["prompt_tokens"], "candidates_tokens": r["candidates_tokens"],
                "thoughts_tokens": r["thoughts_tokens"], "cost_usd": r["cost_usd"]}, ensure_ascii=False) + "\n")
    summary = {"protocol_version": PROTOCOL_VERSION, "states": states, "labels": labels, "finish_reasons": finish,
               "model_versions": sorted(versions), "settled_usd": ctx.ledger.settled(),
               "probe_usd": float(ctx.ledger.meta_get("probe_usd", "0")), "cap_usd": ctx.ledger.cap,
               "batches": [{k: b[k] for k in ("key", "n", "state", "matched", "settled_usd")} for b in ctx.ledger.batches()],
               "results_sha256": online.sha256_file(out_path), "utc": now()}
    ctx.report_dir.mkdir(parents=True, exist_ok=True)
    (ctx.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ctx.ledger.meta_set("exported", now())
    return summary


def step(ctx: Context, jobs_path: Path) -> bool:
    """Advance the run as far as possible without waiting. Returns True once P is exported."""
    probe(ctx)
    poll(ctx)
    collect(ctx)
    ensure_round(ctx, "canary", 0, lambda: make_shards([j for j in ctx.jobs if j.job_id in ctx.canary]))
    if uncollected(ctx, "canary"):
        ctx.log("canary: " + ", ".join(f"{b['key']}={b['state']}" for b in ctx.ledger.batches("canary")))
        return False
    gate_cap = ctx.ledger.meta_get("gate_cap")
    if ctx.ledger.meta_get("gate") is None or (gate_cap is not None and float(gate_cap) != ctx.ledger.cap):
        receipt = projection(ctx)   # recomputed from the same canary rows after a recorded cap change
        ctx.report_dir.mkdir(parents=True, exist_ok=True)
        (ctx.report_dir / "fee_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        ctx.ledger.meta_set("receipt", json.dumps(receipt))
        ctx.ledger.meta_set("gate", "passed" if receipt["passed"] else "failed")
        ctx.ledger.meta_set("gate_cap", repr(ctx.ledger.cap))
        ctx.log(f"canary projection: {receipt['projected_total_usd']:.2f} USD for all requests "
                f"(cap {receipt['cap_usd']:.2f} USD, margin {PROJECTION_MARGIN})")
    if ctx.ledger.meta_get("gate") != "passed":
        raise StopRun("projected cost times the margin exceeds the cap; the full run was not submitted")
    ensure_round(ctx, "full", 0, lambda: make_shards([j for j in ctx.jobs if j.job_id not in ctx.canary]))
    if uncollected(ctx):
        done = sum(1 for b in ctx.ledger.batches() if b["collected"])
        ctx.log(f"{done}/{len(ctx.ledger.batches())} batch jobs collected; "
                + ", ".join(f"{b['key']}={b['state']}" for b in ctx.ledger.batches() if not b["collected"]))
        return False
    for round_ in range(1, MAX_RETRY_ROUNDS + 1):
        if ctx.ledger.meta_get(f"plan:retry-r{round_}") is None:
            candidates = retry_candidates(ctx)
            if not candidates:
                break
            ensure_round(ctx, "retry", round_, lambda: make_shards(candidates))
            return False
        ensure_round(ctx, "retry", round_, None)
        if uncollected(ctx):
            ctx.log(f"retry round {round_} still running")
            return False
    summary = export(ctx, jobs_path)
    ctx.log(json.dumps({k: summary[k] for k in ("states", "labels", "settled_usd", "model_versions")}, indent=2))
    return True


# --------------------------------------------------------------------------- freeze and commands

def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def freeze_record(jobs_path: Path, cap_usd: float) -> tuple[dict[str, Any], list[Job]]:
    base, jobs = online.freeze_record(jobs_path, cap_usd)   # job hash, count, producers and section 14
    canary = set(online.canary_ids(jobs))
    canary_plan = make_shards([j for j in jobs if j.job_id in canary])
    full_plan = make_shards([j for j in jobs if j.job_id not in canary])
    record = {**base, "protocol_version": PROTOCOL_VERSION, "config": CONFIG,
              "code_sha256": {"batch": code_sha256(), "shared": online.code_sha256()},
              "shard_plan": {"max_shard": MAX_SHARD, "max_shard_bytes": MAX_SHARD_BYTES,
                             "canary_shards": len(canary_plan), "canary_sha256": plan_sha256(canary_plan),
                             "full_shards": len(full_plan), "full_sizes": [len(s) for s in full_plan],
                             "full_sha256": plan_sha256(full_plan)},
              "retry_rounds": MAX_RETRY_ROUNDS,
              "cost": {"cap_usd": cap_usd, "price_source": PRICE_SOURCE, "projection_margin": PROJECTION_MARGIN,
                       "probe_rates": {k: online.CONFIG[k] for k in ("price_in_per_m", "price_out_per_m")},
                       "accounting": ("each answered request settles at its returned usage and the batch rates; a "
                                      "request without a response is not billed; a running batch job cannot stop at "
                                      "the cap, so the cap applies to the canary projection and to every later "
                                      "submission")}}
    return record, jobs


def write_freeze(cap_usd: float, jobs_path: Path = JOBS_PATH, report_dir: Path = REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "freeze.json"
    if path.exists():
        raise StopRun("freeze.json already exists; a new freeze needs a new protocol amendment")
    record, _ = freeze_record(jobs_path, cap_usd)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def verify_freeze(jobs_path: Path = JOBS_PATH, report_dir: Path = REPORT_DIR) -> tuple[dict[str, Any], list[Job]]:
    frozen = json.loads((report_dir / "freeze.json").read_text())
    current, jobs = freeze_record(jobs_path, frozen["cost"]["cap_usd"])
    for key in ("jobs_sha256", "config", "parser_version", "code_sha256", "protocol_section_sha256", "canary",
                "shard_plan", "retry_rounds"):
        if frozen[key] != current[key]:
            raise StopRun(f"{key} changed after the freeze")
    return frozen, jobs


def cmd_run(args: argparse.Namespace) -> None:
    frozen, jobs = verify_freeze(args.jobs, args.report_dir)
    ledger = BatchLedger(args.run_dir / "state" / "ledger.sqlite", frozen["cost"]["cap_usd"])
    ledger.register(jobs)
    backend = GeminiBatchBackend(load_api_key())
    ledger.event("run", json.dumps({"wait": args.wait}))
    ctx = Context(ledger, jobs, online.PromptReader(args.jobs), backend, args.run_dir, args.report_dir)
    while not step(ctx, args.jobs):
        if not args.wait:
            print("nothing more to do now; run the same command again later (or add --wait)")
            return
        time.sleep(POLL_SECONDS)


def cmd_status(args: argparse.Namespace) -> None:
    frozen = json.loads((args.report_dir / "freeze.json").read_text())
    ledger = BatchLedger(args.run_dir / "state" / "ledger.sqlite", frozen["cost"]["cap_usd"])
    states = {r[0]: r[1] for r in ledger.con.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state")}
    print(json.dumps({"cap_usd": ledger.cap, "settled_usd": ledger.settled(), "gate": ledger.meta_get("gate"),
                      "job_states": states,
                      "batches": [{k: b[k] for k in ("key", "n", "state", "collected", "matched", "settled_usd")}
                                  for b in ledger.batches()]}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=Path, default=JOBS_PATH)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="write freeze.json; no request is sent")
    f.add_argument("--cap-usd", type=float, required=True)
    r = sub.add_parser("run", help="probe, canary, cost gate, batch jobs, retries and export; resumable")
    r.add_argument("--wait", action="store_true", help=f"poll every {POLL_SECONDS} s until P is exported")
    sub.add_parser("status", help="print batch jobs, job states and settled cost")
    c = sub.add_parser("raise-cap", help="record a higher cost cap after a failed cost gate")
    c.add_argument("--cap-usd", type=float, required=True)
    c.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            print(write_freeze(args.cap_usd, args.jobs, args.report_dir))
        elif args.command == "run":
            cmd_run(args)
        elif args.command == "raise-cap":
            print(json.dumps(online.raise_cap(args.cap_usd, args.reason, args.report_dir), indent=2))
        else:
            cmd_status(args)
    except StopRun as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
