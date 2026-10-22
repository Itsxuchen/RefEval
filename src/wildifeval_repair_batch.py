"""Repair sensitivity R1 for WildIFEval P (protocol section 16): ask once more with an explicit Yes/No request.

The repair set is every job that parser A1 leaves missing for a reason other than a provider block: abstaining,
partial, no-verdict and conflicting answers. Each is sent once more to ``gemini-3.1-flash-lite`` in Gemini API
Batch Mode with the section 14 configuration. The prompt is the frozen job prompt with one added line,
"Answer with Yes or No.", directly above the final "###Answer:". Jobs blocked by the provider are not re-sent.
Parser A1 reads the new answers. The repaired P fills only missing labels and is a sensitivity analysis; the
primary analysis keeps A1. A request without a response is resubmitted at most twice; no answer is regenerated.

The batch mechanics (ledger, submission under the cap, polling, key matching, settlement) are those of
``src.wildifeval_gemini_batch``; this module chooses the jobs, adds the line and exports the result.
No reference score R is read here.

    python -m src.wildifeval_repair_batch freeze --cap-usd 10
    python -m src.wildifeval_repair_batch run --wait
    python -m src.wildifeval_repair_batch status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src import wildifeval_gemini_batch as batch
from src import wildifeval_parser_v2 as a1
from src import wildifeval_vertex_judge as online

ROOT = online.ROOT
JOBS_PATH = online.JOBS_PATH
SOURCE_DIR = a1.RUN_DIR                       # A1 predictions and the section 14 export
SOURCE_REPORT_DIR = a1.REPORT_DIR
RUN_DIR = ROOT / "data" / "interim" / "wildifeval_external" / "flashlite_gemini_repair"          # private
REPORT_DIR = ROOT / "artifacts" / "reports" / "wildifeval_external" / "flashlite_gemini_repair"  # public, no text
PROTOCOL_PATH = online.PROTOCOL_PATH
SECTION_HEADING = "## 16. Repair sensitivity R1 before reference access — 2026-10-22"

PROTOCOL_VERSION = "wildifeval-p-repair-r1"
REPAIR_RULES = ("missing-abstain", "missing-conflict", "missing-no-verdict", "missing-partial")
INSTRUCTION = "Answer with Yes or No."
ANSWER_TAIL = "\n\n###Answer:"
PHASE = "repair"
RETRY_PHASE = "repair-retry"   # distinct batch display names from the section 14 run
StopRun = online.StopRun


def repair_prompt(prompt: str) -> str:
    """The frozen prompt with the instruction as its own line above the final answer marker."""
    if not prompt.endswith(ANSWER_TAIL):
        raise StopRun("a frozen prompt does not end with the answer marker")
    return prompt[: -len(ANSWER_TAIL)] + "\n" + INSTRUCTION + ANSWER_TAIL


class RepairReader:
    """Reads and verifies the frozen prompt, then adds the instruction."""

    def __init__(self, path: Path):
        self.inner = online.PromptReader(path)

    def prompt(self, job: online.Job) -> str:
        return repair_prompt(self.inner.prompt(job))


def protocol_section(path: Path | None = None) -> str:
    text = (path or PROTOCOL_PATH).read_text(encoding="utf-8")
    if SECTION_HEADING not in text:
        raise StopRun("protocol section 16 is missing; the repair run is not authorized")
    return SECTION_HEADING + text.split(SECTION_HEADING, 1)[1].split("\n## ", 1)[0]


def a1_rows(source_dir: Path = SOURCE_DIR, source_report_dir: Path = SOURCE_REPORT_DIR) -> list[dict[str, Any]]:
    path = source_dir / "results_parser_v2.jsonl"
    receipt = json.loads((source_report_dir / "parser_v2_receipt.json").read_text(encoding="utf-8"))
    if online.sha256_file(path) != receipt["output_sha256"]:
        raise StopRun("results_parser_v2.jsonl does not match its receipt")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def freeze_record(cap_usd: float, jobs_path: Path = JOBS_PATH, source_dir: Path = SOURCE_DIR,
                  source_report_dir: Path = SOURCE_REPORT_DIR,
                  protocol_path: Path | None = None) -> tuple[dict[str, Any], list[online.Job]]:
    jobs_sha = online.sha256_file(jobs_path)
    if jobs_path == JOBS_PATH and jobs_sha != online.EXPECTED_JOBS_SHA256:
        raise StopRun("the job file changed")
    chosen = {r["job_id"] for r in a1_rows(source_dir, source_report_dir) if r["parser_rule"] in REPAIR_RULES}
    jobs = [j for j in online.load_jobs(jobs_path) if j.job_id in chosen]
    if len(jobs) != len(chosen):
        raise StopRun("a repair job is missing from the job file")
    plan = batch.make_shards(jobs)
    record = {
        "protocol_version": PROTOCOL_VERSION, "status": "frozen", "created_utc": batch.now(),
        "jobs_sha256": jobs_sha,
        "source_a1_sha256": online.sha256_file(source_dir / "results_parser_v2.jsonl"),
        "repair_rules": list(REPAIR_RULES), "jobs": len(jobs),
        "job_ids_sha256": hashlib.sha256("\n".join(j.job_id for j in jobs).encode()).hexdigest(),
        "instruction": INSTRUCTION, "config": batch.CONFIG, "parser_version": a1.PARSER_VERSION,
        "code_sha256": {"repair": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "batch": batch.code_sha256(), "shared": online.code_sha256(),
                        "parser": online.sha256_file(Path(a1.__file__))},
        "protocol_section_sha256": a1.sha256_text(protocol_section(protocol_path)),
        "shard_plan": {"shards": len(plan), "sizes": [len(s) for s in plan], "sha256": batch.plan_sha256(plan)},
        "retry_rounds": batch.MAX_RETRY_ROUNDS,
        "cost": {"cap_usd": cap_usd, "price_source": batch.PRICE_SOURCE,
                 "accounting": ("each answered request settles at its returned usage and the batch rates; a "
                                "submission needs the worst-case bound of section 14 to fit under the cap")},
        "reference_values_opened": False,
    }
    return record, jobs


def write_freeze(cap_usd: float, report_dir: Path = REPORT_DIR, **kwargs: Any) -> Path:
    path = report_dir / "freeze.json"
    if path.exists():
        raise StopRun("freeze.json already exists; a new freeze needs a new protocol amendment")
    record, _ = freeze_record(cap_usd, **kwargs)
    report_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def verify_freeze(report_dir: Path = REPORT_DIR, **kwargs: Any) -> tuple[dict[str, Any], list[online.Job]]:
    frozen = json.loads((report_dir / "freeze.json").read_text())
    current, jobs = freeze_record(frozen["cost"]["cap_usd"], **kwargs)
    for key in ("jobs_sha256", "source_a1_sha256", "repair_rules", "jobs", "job_ids_sha256", "instruction", "config",
                "parser_version", "code_sha256", "protocol_section_sha256", "shard_plan", "retry_rounds"):
        if frozen[key] != current[key]:
            raise StopRun(f"{key} changed after the freeze")
    return frozen, jobs


def step(ctx: batch.Context, source_rows: list[dict[str, Any]]) -> bool:
    """Advance the repair run without waiting. Returns True once the repaired P is exported."""
    batch.poll(ctx)
    batch.collect(ctx)
    batch.ensure_round(ctx, PHASE, 0, lambda: batch.make_shards(ctx.jobs))
    if batch.uncollected(ctx):
        ctx.log(", ".join(f"{b['key']}={b['state']}" for b in ctx.ledger.batches() if not b["collected"]))
        return False
    for round_ in range(1, batch.MAX_RETRY_ROUNDS + 1):
        if ctx.ledger.meta_get(f"plan:{RETRY_PHASE}-r{round_}") is None:
            candidates = batch.retry_candidates(ctx)
            if not candidates:
                break
            batch.ensure_round(ctx, RETRY_PHASE, round_, lambda: batch.make_shards(candidates))
            return False
        batch.ensure_round(ctx, RETRY_PHASE, round_, None)
        if batch.uncollected(ctx):
            ctx.log(f"retry round {round_} still running")
            return False
    summary = export(ctx, source_rows)
    ctx.log(json.dumps({k: summary[k] for k in ("repair_outcomes", "labels", "settled_usd")}, indent=2))
    return True


def export(ctx: batch.Context, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    states = ctx.ledger.job_states()
    repaired_path = ctx.run_dir / "results_repaired.jsonl"
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    labels = {"a1": Counter(), "repaired": Counter()}
    complete = {"a1": defaultdict(lambda: True), "repaired": defaultdict(lambda: True)}
    task_cluster: dict[str, tuple[str, str]] = {}
    versions: set[str] = set()
    with repaired_path.open("w", encoding="utf-8") as out:
        for r in source_rows:
            row = {k: v for k, v in r.items() if k not in ("status",)}
            repair = states.get(r["job_id"])
            label = r["label"]
            if repair is not None:
                new, rule = (a1.parse(repair["raw_text"]) if repair["state"] == "done" else (None, "no-response"))
                outcomes[r["parser_rule"]]["missing" if new is None else ("pass" if new else "fail")] += 1
                if repair["state"] == "done":
                    versions.add(str(repair["model_version"]))
                row.update({"repair_label": new, "repair_rule": rule, "repair_text": repair["raw_text"],
                            "repair_finish_reason": repair["finish_reason"], "repair_submissions": repair["submissions"],
                            "repair_cost_usd": repair["cost_usd"]})
                label = new
            elif r["label"] is None and r["parser_rule"] in REPAIR_RULES:
                raise StopRun(f"repair job {r['job_id']} has no ledger row")
            row.update({"protocol_version": PROTOCOL_VERSION, "label_a1": r["label"], "label": label})
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            for name, value in (("a1", r["label"]), ("repaired", label)):
                labels[name]["missing" if value is None else ("pass" if value else "fail")] += 1
                complete[name][r["task_id"]] &= value is not None
            task_cluster[r["task_id"]] = (r["cluster_id"], r["split"])

    def frame(name: str) -> dict[str, Any]:
        kept = [t for t, ok in complete[name].items() if ok]
        pilot = {task_cluster[t][0] for t in kept if task_cluster[t][1] == "pilot"}
        confirmation = {task_cluster[t][0] for t in kept if task_cluster[t][1] == "confirmation"}
        return {"tasks_complete": len(kept), "task_retention": round(len(kept) / len(complete[name]), 6),
                "clusters": len(pilot | confirmation), "pilot_clusters": len(pilot),
                "confirmation_clusters": len(confirmation)}

    finish = Counter(str(s["finish_reason"]) for s in states.values() if s["state"] == "done")
    summary = {
        "protocol_version": PROTOCOL_VERSION, "jobs": len(states),
        "states": dict(Counter(s["state"] for s in states.values())), "finish_reasons": dict(finish),
        "repair_outcomes": {k: dict(sorted(v.items())) for k, v in sorted(outcomes.items())},
        "labels": {k: dict(sorted(v.items())) for k, v in labels.items()},
        "common_frame_preview": {name: frame(name) for name in labels},
        "model_versions": sorted(versions), "settled_usd": ctx.ledger.settled(), "cap_usd": ctx.ledger.cap,
        "batches": [{k: b[k] for k in ("key", "n", "state", "matched", "settled_usd")} for b in ctx.ledger.batches()],
        "results_sha256": online.sha256_file(repaired_path),
        "note": ("Sensitivity only: repaired P mixes the section 14 prompt with the one-line Yes/No variant for the "
                 "repair set. The primary analysis keeps A1."),
        "reference_values_opened": False, "utc": batch.now(),
    }
    ctx.report_dir.mkdir(parents=True, exist_ok=True)
    (ctx.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ctx.ledger.meta_set("exported", batch.now())
    return summary


def context(frozen: dict[str, Any], jobs: list[online.Job], backend: Any, run_dir: Path, report_dir: Path,
            jobs_path: Path = JOBS_PATH) -> batch.Context:
    ledger = batch.BatchLedger(run_dir / "state" / "ledger.sqlite", frozen["cost"]["cap_usd"])
    ledger.register(jobs)
    return batch.Context(ledger, jobs, RepairReader(jobs_path), backend, run_dir, report_dir,
                         log=lambda s: print(s, flush=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze", help="write freeze.json; no request is sent")
    f.add_argument("--cap-usd", type=float, required=True)
    r = sub.add_parser("run", help="submit, collect, resubmit requests without a response, export; resumable")
    r.add_argument("--wait", action="store_true", help=f"poll every {batch.POLL_SECONDS} s until the export")
    sub.add_parser("status", help="print batch jobs, job states and settled cost")
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            print(write_freeze(args.cap_usd))
        elif args.command == "run":
            frozen, jobs = verify_freeze()
            ctx = context(frozen, jobs, batch.GeminiBatchBackend(batch.load_api_key()), RUN_DIR, REPORT_DIR)
            ctx.ledger.event("run", json.dumps({"wait": args.wait}))
            rows = a1_rows()
            while not step(ctx, rows):
                if not args.wait:
                    print("nothing more to do now; run the same command again later (or add --wait)")
                    return
                time.sleep(batch.POLL_SECONDS)
        else:
            frozen = json.loads((REPORT_DIR / "freeze.json").read_text())
            ledger = batch.BatchLedger(RUN_DIR / "state" / "ledger.sqlite", frozen["cost"]["cap_usd"])
            states = {s: n for s, n in ledger.con.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state")}
            print(json.dumps({"cap_usd": ledger.cap, "settled_usd": ledger.settled(), "job_states": states,
                              "batches": [{k: b[k] for k in ("key", "n", "state", "collected", "settled_usd")}
                                          for b in ledger.batches()]}, indent=2))
    except StopRun as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
