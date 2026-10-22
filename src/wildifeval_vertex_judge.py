"""WildIFEval primary judge P on Vertex AI after the OpenAI run stopped (protocol section 14).

The job file is the frozen two-role job file (one pinned upstream prompt per criterion), checked by its SHA-256.
Every request sends that prompt as one user message to ``gemini-3.1-flash-lite`` with temperature 0, thinking
level LOW and a 2048-token output cap. Gemini 3.x models return no log probabilities, so the answer text decides:
the first of the first five alphabetic words that equals "yes" or "no" (case-insensitive) gives PASS or FAIL,
and an answer without such a word is missing, never FAIL, and is not regenerated.

Credentials come from Google Application Default Credentials on the machine that runs the script
(``gcloud auth application-default login``). The script reads no key file and ignores API-key variables.
No reference score R is read here.

    python -m src.wildifeval_vertex_judge freeze --cap-usd 100
    python -m src.wildifeval_vertex_judge run --project YOUR_PROJECT_ID
    python -m src.wildifeval_vertex_judge status

``run`` verifies the freeze, sends two unrelated synthetic probes, runs the fixed 160-request fee canary, stops
unless the projected total times 1.25 fits under the cap, and otherwise completes every job and exports P.
It resumes after an interruption. If only the cost gate stops the run, ``raise-cap --cap-usd X`` records a higher
cap with a reason; section 14 allows this because the canary reveals cost and no reference value.

    python -m src.wildifeval_vertex_judge raise-cap --cap-usd 150 --reason "canary projection above cap"
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
JOBS_PATH = ROOT / "data" / "interim" / "wildifeval_external" / "luna_run" / "jobs.jsonl"
RUN_DIR = ROOT / "data" / "interim" / "wildifeval_external" / "flashlite_run"          # private: prompts never leave it
REPORT_DIR = ROOT / "artifacts" / "reports" / "wildifeval_external" / "flashlite_run"  # public receipts, no text
PROTOCOL_PATH = ROOT / "context" / "conjunction_external_validation_protocol.md"
SECTION_HEADING = "## 14. Primary-judge replacement before completion — 2026-10-21"

PROTOCOL_VERSION = "wildifeval-p-vertex-v3"
EXPECTED_JOBS_SHA256 = "914683b47714464ed85fea1f7505cbadc5f91e3246b4c959052e5c62aa857cd3"
EXPECTED_JOBS = 134485
PRODUCERS = ("deepseek-v3", "llama3.1-405b", "llama3.3-70b", "mistral-large", "qwen2.5-72b")
CONFIG: dict[str, Any] = {"model": "gemini-3.1-flash-lite", "thinking_level": "LOW", "temperature": 0.0,
                          "max_output_tokens": 2048, "price_in_per_m": 0.25, "price_out_per_m": 1.50}
PRICE_SOURCE = ("https://ai.google.dev/gemini-api/docs/pricing, standard tier, as frozen for the ProfBench judge "
                "on 2026-10-18; a Vertex invoice may differ, so the cap is a client-side ceiling")
PARSER_VERSION = "first-five-words-yes-no-text-v1"
CANARY_SALT = "fee-calibration-v1"   # the same selection rule as the OpenAI fee canary
CANARY_PER_PRODUCER = 32
PROJECTION_MARGIN = 1.25
MAX_ATTEMPTS = 3
MAX_ATTEMPTS_429 = 8
TIMEOUT_MS = 180_000
MAX_CONSECUTIVE_FAILURES = 20        # 429 responses do not count; they trigger the shared cooldown instead
MAX_429_WITHOUT_SUCCESS = 300        # persistent throttling stops the run instead of crawling
UNKNOWN_SHARE_STOP = 0.05
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
FATAL_CODES = {401, 403, 404}        # 400 rejects one job (terminal missing); it does not stop the run
SYNTHETIC_PROBES = (
    ("probe-ok", "Reply with the single word OK.", None),
    ("probe-yes-no", "###Instruction:\nAnswer with Yes or No.\n\n###Question:\nIs Paris the capital of France?\n\n###Answer:", "yes"),
)
WORD = re.compile(r"[A-Za-z]+")


class StopRun(RuntimeError):
    """Raised when a frozen stop rule fires; the ledger keeps every completed result."""


@dataclass(frozen=True)
class Job:
    job_id: str
    producer: str
    offset: int
    length: int
    prompt_sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_verdict(text: str | None) -> tuple[int | None, int | None]:
    """Return (label, word position). The first 'yes' or 'no' among the first five alphabetic words decides."""
    for position, word in enumerate(WORD.findall(text or "")[:5]):
        low = word.lower()
        if low in ("yes", "no"):
            return int(low == "yes"), position
    return None, None


def load_jobs(path: Path = JOBS_PATH) -> list[Job]:
    """Index the job file by byte offset; prompts are read only when a request is sent."""
    jobs, offset = [], 0
    with path.open("rb") as handle:
        for raw in handle:
            row = json.loads(raw)
            jobs.append(Job(row["job_id"], row["producer"], offset, len(raw), row["prompt_sha256"]))
            offset += len(raw)
    if len({j.job_id for j in jobs}) != len(jobs):
        raise ValueError("duplicate job identifier")
    return jobs


class PromptReader:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.handle = path.open("rb")

    def prompt(self, job: Job) -> str:
        with self.lock:
            self.handle.seek(job.offset)
            row = json.loads(self.handle.read(job.length))
        if row["job_id"] != job.job_id or hashlib.sha256(row["prompt"].encode()).hexdigest() != job.prompt_sha256:
            raise ValueError(f"prompt does not match its frozen hash: {job.job_id}")
        return row["prompt"]


def canary_ids(jobs: list[Job]) -> list[str]:
    chosen = []
    for producer in sorted({j.producer for j in jobs}):
        pool = sorted((hashlib.sha256(f"{CANARY_SALT}|{j.job_id}".encode()).hexdigest(), j.job_id)
                      for j in jobs if j.producer == producer)
        chosen += [job_id for _, job_id in pool[:CANARY_PER_PRODUCER]]
    return chosen


def reserve_usd(prompt_bytes: int) -> float:
    """Conservative reservation before a request: two bytes per input token plus 50, and twice the output cap.
    The same rule as the ProfBench runner; known usage replaces it when the response arrives."""
    return ((math.ceil(prompt_bytes / 2) + 50) * CONFIG["price_in_per_m"]
            + 2 * CONFIG["max_output_tokens"] * CONFIG["price_out_per_m"]) / 1e6


def usage_cost(usage: dict[str, int]) -> float:
    out = usage.get("candidates", 0) + usage.get("thoughts", 0)
    return (usage.get("prompt", 0) * CONFIG["price_in_per_m"] + out * CONFIG["price_out_per_m"]) / 1e6


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def protocol_section(path: Path | None = None) -> str:
    text = (path or PROTOCOL_PATH).read_text(encoding="utf-8")
    if SECTION_HEADING not in text:
        raise StopRun("protocol section 14 is missing; the run is not authorized")
    section = text.split(SECTION_HEADING, 1)[1].split("\n## ", 1)[0]
    return SECTION_HEADING + section


# --------------------------------------------------------------------------- freeze

def freeze_record(jobs_path: Path, cap_usd: float) -> tuple[dict[str, Any], list[Job]]:
    digest = sha256_file(jobs_path)
    if digest != EXPECTED_JOBS_SHA256:
        raise StopRun(f"job file hash {digest} differs from the frozen two-role jobs")
    jobs = load_jobs(jobs_path)
    if len(jobs) != EXPECTED_JOBS or set(j.producer for j in jobs) != set(PRODUCERS):
        raise StopRun("job count or producers differ from the frozen two-role jobs")
    canary = canary_ids(jobs)
    record = {"protocol_version": PROTOCOL_VERSION, "status": "frozen",
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "jobs_sha256": digest, "jobs": len(jobs), "config": CONFIG, "parser_version": PARSER_VERSION,
              "code_sha256": code_sha256(),
              "protocol_section_sha256": hashlib.sha256(protocol_section().encode()).hexdigest(),
              "canary": {"salt": CANARY_SALT, "per_producer": CANARY_PER_PRODUCER,
                         "job_ids_sha256": hashlib.sha256("\n".join(canary).encode()).hexdigest()},
              "cost": {"cap_usd": cap_usd, "price_source": PRICE_SOURCE, "projection_margin": PROJECTION_MARGIN,
                       "accounting": ("conservative reservation before each request; known usage settles it; an HTTP "
                                      "error response, including 429, is rejected and unbilled; a timeout or lost "
                                      "response keeps its reservation as an unknown charge")},
              "reference_values_opened": False,
              "excluded_predecessor": "all OpenAI gpt-6-luna results of luna_run (v1/v2); none enters analysis"}
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
    for key in ("jobs_sha256", "config", "parser_version", "code_sha256", "protocol_section_sha256", "canary"):
        if frozen[key] != current[key]:
            raise StopRun(f"{key} changed after the freeze")
    return frozen, jobs


# --------------------------------------------------------------------------- ledger

class Ledger:
    """SQLite ledger of requests, results and money; safe for a thread pool."""

    def __init__(self, path: Path, cap_usd: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cap, self.lock = cap_usd, threading.RLock()
        self.con = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("""CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, producer TEXT, phase TEXT, state TEXT, attempts INTEGER DEFAULT 0,
            reserve_usd REAL DEFAULT 0, raw_text TEXT, finish_reason TEXT, prompt_tokens INTEGER,
            candidates_tokens INTEGER, thoughts_tokens INTEGER, cost_usd REAL DEFAULT 0,
            model_version TEXT, response_id TEXT, error TEXT, updated_utc TEXT)""")
        self.con.execute("CREATE TABLE IF NOT EXISTS events (utc TEXT, kind TEXT, detail TEXT)")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.inflight = 0.0
        for job_id, reserve in self.con.execute("SELECT job_id, reserve_usd FROM jobs WHERE state = 'inflight'").fetchall():
            self._set(job_id, state="unknown", error="interrupted while in flight")
            self.event("unknown", f"{job_id} interrupted; reserve {reserve:.6f} USD kept")
            self.add_unknown(reserve)

    @staticmethod
    def now() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    def event(self, kind: str, detail: str) -> None:
        with self.lock:
            self.con.execute("INSERT INTO events VALUES (?, ?, ?)", (self.now(), kind, detail))

    def meta_get(self, key: str, default: str | None = None) -> str | None:
        with self.lock:
            row = self.con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def meta_set(self, key: str, value: str) -> None:
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def add_unknown(self, amount: float) -> None:
        with self.lock:
            self.meta_set("unknown_usd", f"{float(self.meta_get('unknown_usd', '0')) + amount:.9f}")

    def add_settled(self, amount: float) -> None:
        with self.lock:
            self.meta_set("settled_usd", f"{float(self.meta_get('settled_usd', '0')) + amount:.9f}")

    def totals(self) -> dict[str, float]:
        with self.lock:
            return {"settled_usd": float(self.meta_get("settled_usd", "0")),
                    "unknown_usd": float(self.meta_get("unknown_usd", "0")), "inflight_usd": self.inflight}

    def committed(self) -> float:
        t = self.totals()
        return t["settled_usd"] + t["unknown_usd"] + t["inflight_usd"]

    def ensure(self, job: Job, phase: str) -> None:
        with self.lock:
            self.con.execute("INSERT OR IGNORE INTO jobs (job_id, producer, phase, state, updated_utc) "
                             "VALUES (?, ?, ?, 'pending', ?)", (job.job_id, job.producer, phase, self.now()))

    def ensure_many(self, jobs: list[Job], phase: str) -> None:
        with self.lock:
            self.con.execute("BEGIN")
            self.con.executemany("INSERT OR IGNORE INTO jobs (job_id, producer, phase, state, updated_utc) "
                                 "VALUES (?, ?, ?, 'pending', ?)",
                                 [(j.job_id, j.producer, phase, self.now()) for j in jobs])
            self.con.execute("COMMIT")

    def states(self) -> dict[str, str]:
        with self.lock:
            return dict(self.con.execute("SELECT job_id, state FROM jobs").fetchall())

    def state(self, job_id: str) -> str | None:
        with self.lock:
            row = self.con.execute("SELECT state FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return row[0] if row else None

    def _set(self, job_id: str, **fields: Any) -> None:
        fields["updated_utc"] = self.now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.lock:
            self.con.execute(f"UPDATE jobs SET {cols} WHERE job_id = ?", (*fields.values(), job_id))

    def begin(self, job_id: str, reserve: float) -> None:
        with self.lock:
            if self.committed() + reserve > self.cap + 1e-12:
                raise StopRun("the next request could exceed the frozen cost cap")
            self.inflight += reserve
            self.con.execute("UPDATE jobs SET attempts = attempts + 1, state = 'inflight', reserve_usd = ?, "
                             "updated_utc = ? WHERE job_id = ?", (reserve, self.now(), job_id))

    def settle(self, job_id: str, reserve: float, result: dict[str, Any]) -> None:
        with self.lock:
            self.inflight -= reserve
            self.add_settled(result["cost_usd"])
            self._set(job_id, state="done", reserve_usd=0.0, error=None, **result)

    def release(self, job_id: str, reserve: float, error: str, state: str) -> None:
        """An HTTP error response, including 429: the request was rejected and is not billed.
        state is 'pending' (retry now), 'failed' (retries used up; retried on resume) or 'rejected' (terminal)."""
        with self.lock:
            self.inflight -= reserve
            self._set(job_id, state=state, reserve_usd=0.0, error=error)

    def keep_unknown(self, job_id: str, reserve: float, error: str, final: bool) -> None:  # retried on resume
        """A timeout or lost response: the reservation stays as an unknown charge."""
        with self.lock:
            self.inflight -= reserve
            self.add_unknown(reserve)
            self._set(job_id, state="unknown" if final else "pending", reserve_usd=0.0, error=error)
            self.event("unknown", f"{job_id}: {error}; reserve {reserve:.6f} USD kept")

    def rows(self) -> list[sqlite3.Row]:
        with self.lock:
            self.con.row_factory = sqlite3.Row
            try:
                return self.con.execute("SELECT * FROM jobs ORDER BY job_id").fetchall()
            finally:
                self.con.row_factory = None


# --------------------------------------------------------------------------- requests

def make_client(project: str, location: str) -> Any:
    from google import genai  # imported lazily so tests run without the SDK
    from google.genai import types
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        os.environ.pop(name, None)  # force Application Default Credentials
    return genai.Client(vertexai=True, project=project, location=location,
                        http_options=types.HttpOptions(timeout=TIMEOUT_MS,
                                                       retry_options=types.HttpRetryOptions(attempts=1)))


def default_request(client: Any, text: str) -> dict[str, Any]:
    from google.genai import types
    config = types.GenerateContentConfig(
        temperature=CONFIG["temperature"], max_output_tokens=CONFIG["max_output_tokens"],
        thinking_config=types.ThinkingConfig(thinking_level=CONFIG["thinking_level"]))
    response = client.models.generate_content(model=CONFIG["model"], contents=text, config=config)
    usage = getattr(response, "usage_metadata", None)
    try:
        answer = response.text
    except Exception:  # no text part, for example a blocked answer
        answer = None
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    return {"text": answer, "finish_reason": getattr(finish, "value", None) or (str(finish) if finish else None),
            "usage": {"prompt": getattr(usage, "prompt_token_count", 0) or 0,
                      "candidates": getattr(usage, "candidates_token_count", 0) or 0,
                      "thoughts": getattr(usage, "thoughts_token_count", 0) or 0},
            "model_version": getattr(response, "model_version", None),
            "response_id": getattr(response, "response_id", None)}


def classify_error(exc: BaseException) -> tuple[bool, bool, bool, str, int | None]:
    """(definite, retryable, fatal, message, code). An HTTP error response is definite and unbilled."""
    code = getattr(exc, "code", None)
    message = f"{type(exc).__name__}: {str(exc)[:300]}"
    if isinstance(code, int):
        return True, code in RETRYABLE_CODES, code in FATAL_CODES, f"HTTP {code} {message}", code
    return False, True, False, message, None


Requester = Callable[[Any, str], dict[str, Any]]


def run_jobs(client: Any, ledger: Ledger, jobs: list[Job], reader: PromptReader, phase: str, workers: int,
             request: Requester = default_request, sleep: Callable[[float], None] = time.sleep,
             clock: Callable[[], float] = time.monotonic) -> None:
    known = ledger.states()
    todo = [j for j in jobs if known.get(j.job_id) in (None, "pending", "failed", "unknown")]
    ledger.ensure_many(todo, phase)
    stop = threading.Event()
    shared = {"consecutive": 0, "cooldown_until": 0.0, "throttled": 0}
    reasons: list[str] = []

    def one(job: Job) -> None:
        attempt, limit = 0, MAX_ATTEMPTS
        while True:
            attempt += 1
            if stop.is_set():
                return
            wait = shared["cooldown_until"] - clock()
            if wait > 0:
                sleep(wait)
            prompt = reader.prompt(job)
            reserve = reserve_usd(len(prompt.encode("utf-8")))
            try:
                ledger.begin(job.job_id, reserve)
            except StopRun as exc:
                reasons.append(str(exc)); stop.set(); return
            try:
                rec = request(client, prompt)
            except Exception as exc:  # noqa: BLE001 - every failure is classified and recorded
                definite, retryable, fatal, message, code = classify_error(exc)
                if code == 429:
                    limit = MAX_ATTEMPTS_429
                final = fatal or not retryable or attempt >= limit
                if definite:
                    state = "pending" if not final else ("failed" if retryable and not fatal else "rejected")
                    ledger.release(job.job_id, reserve, message, state)
                else:
                    ledger.keep_unknown(job.job_id, reserve, message, final)
                if code == 429:   # throttle every worker instead of counting a failure
                    pause = min(120.0, 10.0 * 2.0 ** (attempt - 1)) + random.random()
                    with ledger.lock:
                        shared["cooldown_until"] = max(shared["cooldown_until"], clock() + pause)
                        shared["throttled"] += 1
                else:
                    with ledger.lock:
                        shared["consecutive"] += 1
                too_many = shared["consecutive"] >= MAX_CONSECUTIVE_FAILURES
                too_throttled = shared["throttled"] >= MAX_429_WITHOUT_SUCCESS
                too_unknown = ledger.totals()["unknown_usd"] > UNKNOWN_SHARE_STOP * ledger.cap
                if fatal or too_many or too_throttled or too_unknown:
                    reasons.append(message if fatal else
                                   "too many consecutive failures" if too_many else
                                   "persistent HTTP 429 without a success; lower --workers or raise the Vertex quota"
                                   if too_throttled else "unknown charges exceeded the stop share")
                    stop.set(); return
                if final:
                    return
                if code != 429:
                    sleep(min(60.0, 2.0 * 2.0 ** (attempt - 1) + random.random()))
                continue
            with ledger.lock:
                shared["consecutive"] = 0
                shared["throttled"] = 0
            ledger.settle(job.job_id, reserve, {
                "raw_text": rec["text"], "finish_reason": rec["finish_reason"],
                "prompt_tokens": rec["usage"]["prompt"], "candidates_tokens": rec["usage"]["candidates"],
                "thoughts_tokens": rec["usage"]["thoughts"], "cost_usd": usage_cost(rec["usage"]),
                "model_version": rec["model_version"], "response_id": rec["response_id"]})
            return

    pool = cf.ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        for i, _ in enumerate(cf.as_completed([pool.submit(one, job) for job in todo]), start=1):
            if i % 500 == 0 or i == len(todo):
                t = ledger.totals()
                print(f"[{phase}] {i}/{len(todo)} finished; settled {t['settled_usd']:.2f} USD, "
                      f"unknown {t['unknown_usd']:.2f} USD", flush=True)
    except KeyboardInterrupt:
        # Ctrl-C: send nothing new, let requests in flight settle, then stop; the run resumes later
        stop.set()
        reasons.append("interrupted by the user")
        print("interrupt received; waiting for requests in flight to finish", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if stop.is_set():
        ledger.event("stop", "; ".join(dict.fromkeys(reasons)))
        raise StopRun("; ".join(dict.fromkeys(reasons)))


def probe(client: Any, ledger: Ledger, request: Requester = default_request) -> None:
    """Two unrelated synthetic requests at the production configuration; no dataset text."""
    if ledger.meta_get("probe_ok") == "1":
        return
    records = []
    for name, text, expected in SYNTHETIC_PROBES:
        reserve = reserve_usd(len(text.encode()))
        if ledger.committed() + reserve > ledger.cap:
            raise StopRun("the probe could exceed the frozen cost cap")
        try:
            rec = request(client, text)
        except Exception as exc:  # noqa: BLE001
            definite, _, _, message, _ = classify_error(exc)
            if not definite:
                ledger.add_unknown(reserve)
            ledger.event("probe_failed", f"{name}: {message}")
            raise StopRun(f"probe {name} failed: {message}") from exc
        cost = usage_cost(rec["usage"])
        ledger.add_settled(cost)
        label, _ = parse_verdict(rec["text"])
        ok = bool((rec["text"] or "").strip()) and (expected is None or label == int(expected == "yes"))
        records.append({"probe": name, "model_version": rec["model_version"], "text": rec["text"],
                        "usage": rec["usage"], "cost_usd": cost, "ok": ok})
        print(f"probe {name}: model_version={rec['model_version']} answer={rec['text']!r} usage={rec['usage']}")
        if not ok:
            ledger.event("probe_failed", json.dumps(records[-1]))
            raise StopRun(f"probe {name} did not return a usable answer")
        if rec["model_version"] and not str(rec["model_version"]).startswith(CONFIG["model"]):
            raise StopRun(f"returned model {rec['model_version']} differs from {CONFIG['model']}")
    ledger.event("probe", json.dumps(records))
    ledger.meta_set("probe_ok", "1")


def projection(ledger: Ledger, jobs: list[Job], canary: set[str]) -> dict[str, Any]:
    rows = {r["job_id"]: r for r in ledger.rows()}
    receipt: dict[str, Any] = {"producers": {}}
    total = 0.0
    for producer in PRODUCERS:
        done = [rows[j.job_id] for j in jobs if j.producer == producer and j.job_id in canary
                and rows.get(j.job_id) is not None and rows[j.job_id]["state"] == "done"]
        n_jobs = sum(1 for j in jobs if j.producer == producer)
        if not done:
            raise StopRun(f"no completed canary requests for {producer}")
        mean = sum(r["cost_usd"] for r in done) / len(done)
        parsed = sum(parse_verdict(r["raw_text"])[0] is not None for r in done)
        receipt["producers"][producer] = {"canary_done": len(done), "parsed": parsed, "mean_cost_usd": mean,
                                          "jobs": n_jobs, "projected_usd": mean * n_jobs}
        total += mean * n_jobs
    receipt.update({"projected_total_usd": total, "margin": PROJECTION_MARGIN, "cap_usd": ledger.cap,
                    "passed": total * PROJECTION_MARGIN <= ledger.cap, **ledger.totals(), "utc": ledger.now()})
    return receipt


def export(ledger: Ledger, jobs_path: Path, run_dir: Path, report_dir: Path) -> dict[str, Any]:
    """Private results with answer text go to run_dir; the public summary carries counts only."""
    rows = {r["job_id"]: r for r in ledger.rows()}
    states: dict[str, int] = {}
    labels = {"pass": 0, "fail": 0, "missing": 0}
    finish: dict[str, int] = {}
    versions: set[str] = set()
    with jobs_path.open() as source, (run_dir / "results.jsonl").open("w", encoding="utf-8") as out:
        for line in source:
            job = json.loads(line)
            r = rows.get(job["job_id"])
            state = r["state"] if r else "not_started"
            states[state] = states.get(state, 0) + 1
            ids = {k: job[k] for k in ("job_id", "unit_id", "cluster_id", "task_id", "producer", "criterion_id",
                                        "criterion_hash", "criterion_index", "output_hash", "prompt_sha256", "split", "k")}
            if state != "done":   # every job gets a record; a job without an answer is missing
                labels["missing"] += 1
                out.write(json.dumps(ids | {"label": None, "status": state, "parser_version": PARSER_VERSION,
                                            "model": CONFIG["model"], "error": r["error"] if r else None},
                                     ensure_ascii=False) + "\n")
                continue
            label, position = parse_verdict(r["raw_text"])
            labels["missing" if label is None else ("pass" if label else "fail")] += 1
            finish[str(r["finish_reason"])] = finish.get(str(r["finish_reason"]), 0) + 1
            versions.add(str(r["model_version"]))
            out.write(json.dumps(ids | {"label": label, "answer_word_position": position, "parser_version": PARSER_VERSION,
                                        "status": "success" if label is not None else "unparseable",
                                        "model": CONFIG["model"], "model_version": r["model_version"],
                                        "raw_text": r["raw_text"], "finish_reason": r["finish_reason"],
                                        "prompt_tokens": r["prompt_tokens"], "candidates_tokens": r["candidates_tokens"],
                                        "thoughts_tokens": r["thoughts_tokens"], "cost_usd": r["cost_usd"]},
                                 ensure_ascii=False) + "\n")
    summary = {"protocol_version": PROTOCOL_VERSION, "states": states, "labels": labels, "finish_reasons": finish,
               "model_versions": sorted(versions), **ledger.totals(), "utc": ledger.now(),
               "results_sha256": sha256_file(run_dir / "results.jsonl")}
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# --------------------------------------------------------------------------- commands

def cmd_run(args: argparse.Namespace) -> None:
    frozen, jobs = verify_freeze(args.jobs, args.report_dir)
    ledger = Ledger(args.run_dir / "state" / "ledger.sqlite", frozen["cost"]["cap_usd"])
    ledger.event("run", json.dumps({"location": args.location, "workers": args.workers}))
    reader = PromptReader(args.jobs)
    client, request = make_client(args.project, args.location), default_request
    probe(client, ledger, request)
    canary = set(canary_ids(jobs))
    run_jobs(client, ledger, [j for j in jobs if j.job_id in canary], reader, "canary", args.workers, request)
    receipt = projection(ledger, jobs, canary)
    (args.report_dir / "fee_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"canary projection: {receipt['projected_total_usd']:.2f} USD for all requests "
          f"(cap {receipt['cap_usd']:.2f} USD, margin {PROJECTION_MARGIN})")
    if not receipt["passed"]:
        raise StopRun("projected cost times the margin exceeds the cap; the full run was not started")
    run_jobs(client, ledger, [j for j in jobs if j.job_id not in canary], reader, "full", args.workers, request)
    print(json.dumps(export(ledger, args.jobs, args.run_dir, args.report_dir), indent=2))


def raise_cap(cap_usd: float, reason: str, report_dir: Path = REPORT_DIR) -> dict[str, Any]:
    """Record a higher cap. Allowed only upward and only with a reason; every change stays in the freeze file."""
    path = report_dir / "freeze.json"
    frozen = json.loads(path.read_text())
    old = frozen["cost"]["cap_usd"]
    if cap_usd <= old:
        raise StopRun(f"the new cap {cap_usd} must exceed the current cap {old}")
    if not reason.strip():
        raise StopRun("a reason is required")
    frozen["cost"].setdefault("cap_history", []).append(
        {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "from_usd": old, "to_usd": cap_usd,
         "reason": reason, "reference_values_opened": False})
    frozen["cost"]["cap_usd"] = cap_usd
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen["cost"]


def cmd_status(args: argparse.Namespace) -> None:
    frozen = json.loads((args.report_dir / "freeze.json").read_text())
    ledger = Ledger(args.run_dir / "state" / "ledger.sqlite", frozen["cost"]["cap_usd"])
    states: dict[str, int] = {}
    for r in ledger.rows():
        states[r["state"]] = states.get(r["state"], 0) + 1
    print(json.dumps({"cap_usd": ledger.cap, **ledger.totals(), "states": states,
                      "not_started": frozen["jobs"] - sum(states.values())}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=Path, default=JOBS_PATH)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="write freeze.json; no request is sent")
    f.add_argument("--cap-usd", type=float, required=True)
    r = sub.add_parser("run", help="probe, canary, cost gate, full run and export; resumable")
    r.add_argument("--project", required=True, help="Google Cloud project ID with Vertex AI enabled")
    r.add_argument("--location", default="global")
    r.add_argument("--workers", type=int, default=8)
    sub.add_parser("status", help="print ledger totals and job states")
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
            print(json.dumps(raise_cap(args.cap_usd, args.reason, args.report_dir), indent=2))
        else:
            cmd_status(args)
    except StopRun as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
