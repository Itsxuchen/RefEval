"""WildIFEval primary judge P through Vertex AI batch inference (protocol section 14).

Batch inference bills half the standard rate. Every frozen job is sent once as one JSONL request line with the
pinned upstream prompt as the only user message, temperature 0, thinking level LOW and a 2048-token output cap.
The parser, job index and prompt checks are shared with ``src.wildifeval_vertex_judge``.

Credentials come from Google Application Default Credentials on the machine that runs the script. The script reads
no key file. No reference score R is read here.

    python -m src.wildifeval_vertex_batch freeze --cap-usd 100
    python -m src.wildifeval_vertex_batch run --project PROJECT_ID --bucket gs://BUCKET --create-bucket
    python -m src.wildifeval_vertex_batch run --project PROJECT_ID --bucket gs://BUCKET --wait
    python -m src.wildifeval_vertex_batch status

``run`` is a resumable state machine. Each call does whatever is ready and then returns, or keeps polling with
``--wait``. It sends two synthetic online probes, submits the fixed 160-request canary as a batch job, applies the
cost gate, submits the remaining jobs in shards of at most 30,000 requests, retries requests that returned no
response at most twice, and exports P. A batch job keeps running at Google after this script exits.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable

from src import wildifeval_vertex_judge as online

ROOT = online.ROOT
JOBS_PATH = online.JOBS_PATH
RUN_DIR = ROOT / "data" / "interim" / "wildifeval_external" / "flashlite_batch"          # private: prompts, answers
REPORT_DIR = ROOT / "artifacts" / "reports" / "wildifeval_external" / "flashlite_batch"  # public receipts, no text
StopRun = online.StopRun
Job = online.Job

PROTOCOL_VERSION = "wildifeval-p-vertex-batch-v3"
CONFIG: dict[str, Any] = {"model": online.CONFIG["model"], "thinking_level": online.CONFIG["thinking_level"],
                          "temperature": online.CONFIG["temperature"],
                          "max_output_tokens": online.CONFIG["max_output_tokens"],
                          "mode": "vertex-batch", "price_in_per_m": 0.125, "price_out_per_m": 0.75}
PRICE_SOURCE = ("https://ai.google.dev/gemini-api/docs/pricing, Batch row for gemini-3.1-flash-lite, read "
                "2026-10-21 (0.125 / 0.75 USD per million tokens, thinking counted as output). Vertex batch is "
                "assumed to bill the same half rate; the Google Cloud invoice is the final record")
GCS_PREFIX = "wildifeval_flashlite_batch_v3"
MAX_SHARD = 30_000
MAX_RETRY_ROUNDS = 2
PROJECTION_MARGIN = online.PROJECTION_MARGIN
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
            "JOB_STATE_PARTIALLY_SUCCEEDED"}
POLL_SECONDS = 120


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def batch_cost(usage: dict[str, int]) -> float:
    out = usage.get("candidates", 0) + usage.get("thoughts", 0)
    return (usage.get("prompt", 0) * CONFIG["price_in_per_m"] + out * CONFIG["price_out_per_m"]) / 1e6


def request_line(prompt: str) -> dict[str, Any]:
    return {"request": {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": CONFIG["temperature"],
                                             "maxOutputTokens": CONFIG["max_output_tokens"],
                                             "thinkingConfig": {"thinkingLevel": CONFIG["thinking_level"]}}}}


def make_shards(jobs: list[Job], max_size: int | None = None) -> list[list[Job]]:
    """First-fit in job-file order. Identical prompts never share a shard, so every output line maps to one job."""
    max_size = max_size or MAX_SHARD
    shards: list[list[Job]] = []
    seen: list[set[str]] = []
    for job in jobs:
        for shard, hashes in zip(shards, seen):
            if len(shard) < max_size and job.prompt_sha256 not in hashes:
                shard.append(job); hashes.add(job.prompt_sha256)
                break
        else:
            shards.append([job]); seen.append({job.prompt_sha256})
    return shards


def plan_sha256(shards: list[list[Job]]) -> str:
    return hashlib.sha256("\n\n".join("\n".join(j.job_id for j in s) for s in shards).encode()).hexdigest()


def _first(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_output_line(obj: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Return (prompt SHA-256, response record or None, error). The echoed request identifies the job."""
    try:
        prompt = obj["request"]["contents"][0]["parts"][0]["text"]
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    except (KeyError, IndexError, TypeError):
        sha = None
    status = obj.get("status") or ""
    resp = obj.get("response")
    if not resp or status:
        return sha, None, (str(status)[:300] or "no response")
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
           "response_id": _first(resp, "responseId", "response_id"),
           "traffic_type": _first(usage, "trafficType", "traffic_type")}
    return sha, rec, None


def split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise StopRun(f"not a Cloud Storage URI: {uri}")
    bucket, _, path = uri[5:].partition("/")
    return bucket, path


# --------------------------------------------------------------------------- Google adapters (not used in tests)

class GcsStore:
    def __init__(self, project: str):
        from google.cloud import storage  # imported lazily so tests run without the SDK
        self.client = storage.Client(project=project)

    def ensure_bucket(self, bucket_uri: str, location: str, create: bool) -> str:
        name, _ = split_uri(bucket_uri)
        bucket = self.client.lookup_bucket(name)
        if bucket is None:
            if not create:
                raise StopRun(f"bucket {bucket_uri} does not exist; add --create-bucket")
            bucket = self.client.create_bucket(name, location=location)
        return str(bucket.location)

    def upload(self, local: Path, uri: str) -> None:
        name, path = split_uri(uri)
        self.client.bucket(name).blob(path).upload_from_filename(str(local), timeout=600)

    def list(self, prefix_uri: str) -> list[str]:
        name, path = split_uri(prefix_uri)
        return [f"gs://{name}/{b.name}" for b in self.client.list_blobs(name, prefix=path)]

    def download(self, uri: str, local: Path) -> None:
        name, path = split_uri(uri)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.client.bucket(name).blob(path).download_to_filename(str(local), timeout=600)


class VertexBatches:
    def __init__(self, project: str, location: str):
        self.client = online.make_client(project, location)

    def create(self, src: str, dest: str, display_name: str) -> str:
        from google.genai import types
        job = self.client.batches.create(model=CONFIG["model"], src=src,
                                         config=types.CreateBatchJobConfig(dest=dest, display_name=display_name))
        return job.name

    def find(self, display_name: str) -> str | None:
        for job in self.client.batches.list():
            if getattr(job, "display_name", None) == display_name:
                return job.name
        return None

    def get(self, name: str) -> tuple[str, str | None]:
        job = self.client.batches.get(name=name)
        state = getattr(job.state, "value", None) or str(job.state)
        error = getattr(job, "error", None)
        return str(state), (str(error)[:500] if error else None)

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
            thoughts_tokens INTEGER, cost_usd REAL DEFAULT 0, model_version TEXT, traffic_type TEXT, error TEXT,
            updated_utc TEXT)""")
        self.con.execute("""CREATE TABLE IF NOT EXISTS batches (
            key TEXT PRIMARY KEY, phase TEXT, round INTEGER, n INTEGER, input_uri TEXT, output_uri TEXT,
            display_name TEXT, name TEXT, state TEXT, error TEXT, submitted_utc TEXT, collected INTEGER DEFAULT 0,
            settled_usd REAL DEFAULT 0, matched INTEGER DEFAULT 0, unmatched INTEGER DEFAULT 0, expected_usd REAL DEFAULT 0)""")
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
    def __init__(self, ledger: BatchLedger, jobs: list[Job], reader: online.PromptReader, batches: Any, store: Any,
                 bucket_uri: str, run_dir: Path, report_dir: Path, log: Callable[[str], None] = print):
        self.ledger, self.jobs, self.reader, self.batches, self.store = ledger, jobs, reader, batches, store
        self.bucket_uri, self.run_dir, self.report_dir, self.log = bucket_uri.rstrip("/"), run_dir, report_dir, log
        self.by_id = {j.job_id: j for j in jobs}
        self.canary = set(online.canary_ids(jobs))


def probe(ctx: Context) -> None:
    """Two unrelated synthetic online requests at the standard rate; they check access, model and parser."""
    if ctx.ledger.meta_get("probe_ok") == "1":
        return
    for name, text, expected in online.SYNTHETIC_PROBES:
        try:
            rec = ctx.batches.probe(text)
        except Exception as exc:  # noqa: BLE001
            ctx.ledger.event("probe_failed", f"{name}: {type(exc).__name__}: {str(exc)[:300]}")
            raise StopRun(f"probe {name} failed: {type(exc).__name__}: {str(exc)[:300]}") from exc
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
    if receipt is None:
        # the job line is longer than its prompt, so two bytes per token of the line bounds the input
        return sum((j.length / 2 + 50) * CONFIG["price_in_per_m"] / 1e6
                   + 2 * CONFIG["max_output_tokens"] * CONFIG["price_out_per_m"] / 1e6 for j in shard)
    means = {p: v["mean_cost_usd"] for p, v in json.loads(receipt)["producers"].items()}
    return PROJECTION_MARGIN * sum(means[j.producer] for j in shard)


def committed(ctx: Context) -> float:
    """Settled cost plus the expected cost of every batch that has not been collected."""
    pending = ctx.ledger.con.execute("SELECT COALESCE(SUM(expected_usd), 0) FROM batches WHERE collected = 0").fetchone()[0]
    return ctx.ledger.settled() + float(pending)


def submit(ctx: Context, phase: str, round_: int, plan: list[list[Job]]) -> None:
    """Upload and submit every shard of a plan that has no batch job yet. Safe to call again after a crash."""
    con = ctx.ledger.con
    for i, shard in enumerate(plan):
        key = f"{phase}-r{round_}-s{i:02d}"
        row = con.execute("SELECT * FROM batches WHERE key = ?", (key,)).fetchone()
        if row is not None and row["name"]:
            continue
        display = f"{GCS_PREFIX}-{key}"
        input_uri = f"{ctx.bucket_uri}/{GCS_PREFIX}/input/{key}.jsonl"
        output_uri = f"{ctx.bucket_uri}/{GCS_PREFIX}/output/{key}"
        if row is None:
            expected = expected_usd(ctx, shard)
            if committed(ctx) + expected > ctx.ledger.cap + 1e-12:
                raise StopRun(f"submitting {key} could exceed the cost cap "
                              f"(committed {committed(ctx):.2f} + expected {expected:.2f} > {ctx.ledger.cap:.2f} USD)")
            local = ctx.run_dir / "batch_inputs" / f"{key}.jsonl"
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("w", encoding="utf-8") as out:
                for job in shard:
                    out.write(json.dumps(request_line(ctx.reader.prompt(job)), ensure_ascii=False) + "\n")
            ctx.store.upload(local, input_uri)
            con.execute("BEGIN")
            con.execute("INSERT INTO batches (key, phase, round, n, input_uri, output_uri, display_name, state, "
                        "expected_usd) VALUES (?, ?, ?, ?, ?, ?, ?, 'creating', ?)",
                        (key, phase, round_, len(shard), input_uri, output_uri, display, expected))
            con.executemany("UPDATE jobs SET phase = ?, state = 'submitted', batch_key = ?, "
                            "submissions = submissions + 1, updated_utc = ? WHERE job_id = ?",
                            [(phase, key, now(), j.job_id) for j in shard])
            con.execute("COMMIT")
            name = None
        else:   # a crash after the upload: adopt a job that was already created rather than paying twice
            name = ctx.batches.find(display)
        if name is None:
            name = ctx.batches.create(input_uri, output_uri, display)
        con.execute("UPDATE batches SET name = ?, state = 'JOB_STATE_PENDING', submitted_utc = ? WHERE key = ?",
                    (name, now(), key))
        ctx.ledger.event("submit", json.dumps({"key": key, "name": name, "n": len(shard)}))
        ctx.log(f"submitted {key}: {len(shard)} requests as {name}")


def ensure_round(ctx: Context, phase: str, round_: int, make: Callable[[], list[list[Job]]] | None) -> bool:
    """Store the shard plan once, then submit whatever part of it is missing. Returns False if nothing to do."""
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
            state, error = ctx.batches.get(row["name"])
            if state != row["state"]:
                ctx.ledger.con.execute("UPDATE batches SET state = ?, error = ? WHERE key = ?", (state, error, row["key"]))
                ctx.ledger.event("state", json.dumps({"key": row["key"], "state": state, "error": error}))


def collect(ctx: Context) -> None:
    """Download finished outputs, match each line to its job by the echoed prompt and settle its cost."""
    con = ctx.ledger.con
    for row in ctx.ledger.batches():
        if row["collected"] or row["state"] not in TERMINAL:
            continue
        key = row["key"]
        expected = {ctx.by_id[r["job_id"]].prompt_sha256: r["job_id"] for r in
                    con.execute("SELECT job_id FROM jobs WHERE batch_key = ? AND state = 'submitted'", (key,))}
        _, out_path = split_uri(row["output_uri"])
        files = sorted(u for u in ctx.store.list(row["output_uri"] + "/")
                       if Path(split_uri(u)[1]).name.startswith("predictions") and u.endswith(".jsonl"))
        if not files and row["state"] != "JOB_STATE_FAILED":
            raise StopRun(f"{key} ended {row['state']} but no predictions file was found under {row['output_uri']}; "
                          f"files present: {ctx.store.list(row['output_uri'] + '/')[:10]}")
        seen: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
        unmatched = 0
        for uri in files:
            rel = split_uri(uri)[1][len(out_path):].lstrip("/")
            local = ctx.run_dir / "batch_outputs" / key / rel
            ctx.store.download(uri, local)
            with local.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    sha, rec, err = parse_output_line(json.loads(line))
                    job_id = expected.get(sha)
                    if job_id is None or job_id in seen:
                        unmatched += 1
                        continue
                    seen[job_id] = (rec, err)
        if unmatched:
            ctx.ledger.event("unmatched", json.dumps({"key": key, "lines": unmatched}))
            raise StopRun(f"{key}: {unmatched} output lines did not match exactly one submitted job; "
                          "nothing was settled for this batch")
        con.execute("BEGIN")
        settled, versions = 0.0, set()
        for job_id in expected.values():
            rec, err = seen.get(job_id, (None, "no output line" if row["state"] != "JOB_STATE_FAILED"
                                                else f"batch failed: {row['error']}"))
            if rec is None:
                con.execute("UPDATE jobs SET state = 'error', error = ?, updated_utc = ? WHERE job_id = ?",
                            (err, now(), job_id))
                continue
            cost = batch_cost(rec["usage"])
            settled += cost
            versions.add(str(rec["model_version"]))
            con.execute("UPDATE jobs SET state = 'done', raw_text = ?, finish_reason = ?, prompt_tokens = ?, "
                        "candidates_tokens = ?, thoughts_tokens = ?, cost_usd = ?, model_version = ?, "
                        "traffic_type = ?, error = NULL, updated_utc = ? WHERE job_id = ?",
                        (rec["text"], rec["finish_reason"], rec["usage"]["prompt"], rec["usage"]["candidates"],
                         rec["usage"]["thoughts"], cost, rec["model_version"], rec["traffic_type"], now(), job_id))
        con.execute("UPDATE batches SET collected = 1, settled_usd = ?, matched = ?, unmatched = 0 WHERE key = ?",
                    (settled, len(seen), key))
        con.execute("COMMIT")
        ctx.ledger.event("collect", json.dumps({"key": key, "state": row["state"], "matched": len(seen),
                                                "expected": len(expected), "settled_usd": settled,
                                                "model_versions": sorted(versions)}))
        ctx.log(f"collected {key}: {len(seen)}/{len(expected)} answered, {settled:.4f} USD")
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
    traffic: dict[str, int] = {}
    for producer in online.PRODUCERS:
        canary = [rows[j.job_id] for j in ctx.jobs if j.producer == producer and j.job_id in ctx.canary]
        done = [r for r in canary if r["state"] == "done"]
        n_jobs = sum(1 for j in ctx.jobs if j.producer == producer)
        if not done:
            raise StopRun(f"no answered canary requests for {producer}")
        for r in done:
            traffic[str(r["traffic_type"])] = traffic.get(str(r["traffic_type"]), 0) + 1
        mean = sum(r["cost_usd"] for r in done) / len(done)
        receipt["producers"][producer] = {
            "canary": len(canary), "answered": len(done),
            "parsed": sum(online.parse_verdict(r["raw_text"])[0] is not None for r in done),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in done) / len(done),
            "mean_output_tokens": sum(r["candidates_tokens"] + r["thoughts_tokens"] for r in done) / len(done),
            "mean_cost_usd": mean, "jobs": n_jobs, "projected_usd": mean * n_jobs}
        total += mean * n_jobs
    receipt.update({"projected_total_usd": total, "margin": PROJECTION_MARGIN, "cap_usd": ctx.ledger.cap,
                    "passed": total * PROJECTION_MARGIN <= ctx.ledger.cap, "settled_usd": ctx.ledger.settled(),
                    "traffic_types": traffic, "rates": {k: CONFIG[k] for k in ("price_in_per_m", "price_out_per_m")},
                    "utc": now()})
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
    traffic: dict[str, int] = {}
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
            traffic[str(r["traffic_type"])] = traffic.get(str(r["traffic_type"]), 0) + 1
            versions.add(str(r["model_version"]))
            out.write(json.dumps(ids | base | {
                "label": label, "answer_word_position": position,
                "status": "success" if label is not None else "unparseable",
                "model_version": r["model_version"], "raw_text": r["raw_text"], "finish_reason": r["finish_reason"],
                "prompt_tokens": r["prompt_tokens"], "candidates_tokens": r["candidates_tokens"],
                "thoughts_tokens": r["thoughts_tokens"], "cost_usd": r["cost_usd"],
                "traffic_type": r["traffic_type"]}, ensure_ascii=False) + "\n")
    summary = {"protocol_version": PROTOCOL_VERSION, "states": states, "labels": labels, "finish_reasons": finish,
               "model_versions": sorted(versions), "traffic_types": traffic, "settled_usd": ctx.ledger.settled(),
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
    ensure_round(ctx, "canary", 0, lambda: make_shards([j for j in ctx.jobs if j.job_id in ctx.canary]))
    poll(ctx)
    collect(ctx)
    if uncollected(ctx, "canary"):
        ctx.log("canary batch still running")
        return False
    gate_cap = ctx.ledger.meta_get("gate_cap")
    if ctx.ledger.meta_get("gate") is None or (gate_cap is not None and float(gate_cap) != ctx.ledger.cap):
        receipt = projection(ctx)   # recomputed from the same canary rows after a recorded cap change
        (ctx.report_dir / "fee_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        ctx.ledger.meta_set("receipt", json.dumps(receipt))
        ctx.ledger.meta_set("gate", "passed" if receipt["passed"] else "failed")
        ctx.ledger.meta_set("gate_cap", repr(ctx.ledger.cap))
        ctx.log(f"canary projection: {receipt['projected_total_usd']:.2f} USD for all requests "
                f"(cap {receipt['cap_usd']:.2f} USD, margin {PROJECTION_MARGIN})")
    if ctx.ledger.meta_get("gate") != "passed":
        raise StopRun("projected cost times the margin exceeds the cap; the full run was not submitted")
    ensure_round(ctx, "full", 0, lambda: make_shards([j for j in ctx.jobs if j.job_id not in ctx.canary]))
    poll(ctx)
    collect(ctx)
    if uncollected(ctx):
        ctx.log("running: " + ", ".join(f"{b['key']}={b['state']}" for b in ctx.ledger.batches() if not b["collected"]))
        return False
    for round_ in range(1, MAX_RETRY_ROUNDS + 1):
        if ctx.ledger.meta_get(f"plan:retry-r{round_}") is None:
            candidates = retry_candidates(ctx)
            if not candidates:
                break
            ensure_round(ctx, "retry", round_, lambda: make_shards(candidates))
            return False
        ensure_round(ctx, "retry", round_, None)
        poll(ctx)
        collect(ctx)
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
              "shard_plan": {"max_shard": MAX_SHARD, "canary_shards": len(canary_plan),
                             "canary_sha256": plan_sha256(canary_plan), "full_shards": len(full_plan),
                             "full_sizes": [len(s) for s in full_plan], "full_sha256": plan_sha256(full_plan)},
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
    store, batches = GcsStore(args.project), VertexBatches(args.project, args.location)
    bucket_location = store.ensure_bucket(args.bucket, args.bucket_location, args.create_bucket)
    ledger.event("run", json.dumps({"location": args.location, "bucket": args.bucket,
                                    "bucket_location": bucket_location, "wait": args.wait}))
    ctx = Context(ledger, jobs, online.PromptReader(args.jobs), batches, store, args.bucket, args.run_dir,
                  args.report_dir)
    while not step(ctx, args.jobs):
        if not args.wait:
            print("nothing more to do now; run the same command again later (or add --wait)")
            return
        time.sleep(POLL_SECONDS)


def cmd_status(args: argparse.Namespace) -> None:
    frozen = json.loads((args.report_dir / "freeze.json").read_text())
    ledger = BatchLedger(args.run_dir / "state" / "ledger.sqlite", frozen["cost"]["cap_usd"])
    states: dict[str, int] = {}
    for r in ledger.con.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state"):
        states[r[0]] = r[1]
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
    r = sub.add_parser("run", help="probe, canary, cost gate, full batches, retries and export; resumable")
    r.add_argument("--project", required=True, help="Google Cloud project ID with Vertex AI enabled")
    r.add_argument("--bucket", required=True, help="gs://bucket for batch input and output")
    r.add_argument("--location", default="global", help="Vertex AI location of the model and batch jobs")
    r.add_argument("--bucket-location", default="us-central1")
    r.add_argument("--create-bucket", action="store_true")
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
