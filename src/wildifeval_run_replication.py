"""Replication check on P (protocol section 18): compare the completed Vertex AI batch run with the Gemini API run.

Both runs sent the same 134,485 frozen prompts to ``gemini-3.1-flash-lite`` with the same configuration. This
module downloads the Vertex outputs, matches each output line to its job by the echoed prompt hash within its
batch job, applies the section 14 parser and parser A1 unchanged, and compares the labels with P. It submits no
request, never changes P or the frame, and reads no reference value.

    python -m src.wildifeval_run_replication run

``run`` downloads each output file once (later calls reuse local copies), writes the private comparison next to the Vertex
archive and a public receipt with counts and hashes only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from src import wildifeval_parser_v2 as a1
from src import wildifeval_vertex_batch as vertex
from src import wildifeval_vertex_judge as online

ROOT = online.ROOT
VERTEX_DIR = ROOT / "data" / "interim" / "wildifeval_external" / "flashlite_batch"                   # private
P_PATH = ROOT / "data" / "interim" / "wildifeval_external" / "flashlite_gemini_batch" / "results_parser_v2.jsonl"
P_RESULTS = ROOT / "data" / "interim" / "wildifeval_external" / "flashlite_gemini_batch" / "results.jsonl"
REPORT_DIR = ROOT / "artifacts" / "reports" / "wildifeval_external" / "flashlite_replication"           # public


def batch_rows(vertex_dir: Path) -> tuple[list[dict], dict[str, str]]:
    """Batch jobs and the batch of every submitted job, read from the Vertex ledger without writing to it."""
    con = sqlite3.connect(f"file:{vertex_dir / 'state' / 'ledger.sqlite'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    batches = [dict(r) for r in con.execute("SELECT key, n, name, output_uri FROM batches ORDER BY key")]
    assignment = {r["job_id"]: r["batch_key"] for r in con.execute("SELECT job_id, batch_key FROM jobs")}
    con.close()
    return batches, assignment


def download(store: Any, batches: list[dict], out_dir: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {}
    for b in batches:
        _, prefix = vertex.split_uri(b["output_uri"])
        uris = sorted(u for u in store.list(b["output_uri"] + "/")
                      if Path(vertex.split_uri(u)[1]).name.startswith("predictions") and u.endswith(".jsonl"))
        if not uris:
            raise online.StopRun(f"no predictions file under {b['output_uri']}")
        files[b["key"]] = []
        for uri in uris:
            local = out_dir / b["key"] / vertex.split_uri(uri)[1][len(prefix):].lstrip("/")
            if not local.exists():
                store.download(uri, local)
            files[b["key"]].append(local)
    return files


def vertex_answers(files: dict[str, list[Path]], assignment: dict[str, str],
                   prompts: dict[str, str]) -> dict[str, dict]:
    """Match every output line to exactly one job of its batch job by the echoed prompt hash."""
    expected: dict[str, dict[str, str]] = defaultdict(dict)
    for job_id, key in assignment.items():
        expected[key][prompts[job_id]] = job_id
    answers: dict[str, dict] = {}
    for key, paths in files.items():
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    sha, rec, err = vertex.parse_output_line(json.loads(line))
                    job_id = expected[key].get(sha)
                    if job_id is None or job_id in answers:
                        raise online.StopRun(f"{key}: an output line matches no job or a job twice")
                    answers[job_id] = rec if rec is not None else {"text": None, "finish_reason": None,
                                                                   "model_version": None, "usage": {}, "error": err}
    if set(answers) != set(assignment):
        raise online.StopRun(f"{len(set(assignment) - set(answers))} jobs have no Vertex output line")
    return answers


def compare(answers: dict[str, dict], p_rows: list[dict], p_text: dict[str, str | None]) -> tuple[dict, list[dict]]:
    pair = Counter()
    by_producer: dict[str, Counter] = defaultdict(Counter)
    by_rule: dict[str, Counter] = defaultdict(Counter)
    v1_pair = Counter()
    complete_v = defaultdict(lambda: True)
    complete_both = defaultdict(lambda: True)
    unit_verdicts: dict[str, list[tuple[int | None, int | None]]] = defaultdict(list)
    identical_text = 0
    versions = Counter()
    cost = 0.0
    rows = []
    for r in p_rows:
        rec = answers[r["job_id"]]
        versions[str(rec.get("model_version"))] += 1
        cost += vertex.batch_cost(rec.get("usage") or {}) if rec.get("usage") else 0.0
        text = rec.get("text")
        v_label, v_rule = a1.parse(text)
        v_v1, _ = online.parse_verdict(text)
        p_label = r["label"]
        identical_text += text is not None and text == p_text.get(r["job_id"])
        if p_label is None or v_label is None:
            kind = "both_missing" if p_label is None and v_label is None else (
                "p_missing" if p_label is None else "vertex_missing")
        else:
            kind = "agree" if p_label == v_label else ("p_pass_vertex_fail" if p_label else "p_fail_vertex_pass")
        pair[kind] += 1
        by_producer[r["producer"]][kind] += 1
        by_rule[r["parser_rule"]][kind] += 1
        if r["label_v1"] is not None and v_v1 is not None:
            v1_pair["agree" if r["label_v1"] == v_v1 else "disagree"] += 1
        complete_v[r["task_id"]] &= v_label is not None
        complete_both[r["task_id"]] &= v_label is not None and p_label is not None
        unit_verdicts[r["unit_id"]].append((p_label, v_label))
        rows.append({"job_id": r["job_id"], "unit_id": r["unit_id"], "task_id": r["task_id"], "producer": r["producer"],
                     "p_label": p_label, "p_rule": r["parser_rule"], "vertex_label": v_label, "vertex_rule": v_rule,
                     "vertex_label_v1": v_v1, "vertex_finish_reason": rec.get("finish_reason"),
                     "identical_text": text is not None and text == p_text.get(r["job_id"])})
    both = [u for u, pairs in unit_verdicts.items() if all(p is not None and v is not None for p, v in pairs)]
    verdict = Counter()
    for u in both:
        p_all = all(p for p, _ in unit_verdicts[u])
        v_all = all(v for _, v in unit_verdicts[u])
        verdict["agree" if p_all == v_all else ("p_pass_vertex_fail" if p_all else "p_fail_vertex_pass")] += 1
    labelled = pair["agree"] + pair["p_pass_vertex_fail"] + pair["p_fail_vertex_pass"]
    summary = {
        "jobs": len(p_rows), "identical_answer_text": identical_text,
        "criterion_pairs": dict(sorted(pair.items())),
        "criterion_agreement_where_both_labelled": round(pair["agree"] / labelled, 6) if labelled else None,
        "section14_parser_pairs": dict(sorted(v1_pair.items())),
        "by_producer": {k: dict(sorted(v.items())) for k, v in sorted(by_producer.items())},
        "by_p_rule": {k: dict(sorted(v.items())) for k, v in sorted(by_rule.items())},
        "vertex_tasks_complete": sum(complete_v.values()), "tasks_complete_in_both": sum(complete_both.values()),
        "tasks": len(complete_v),
        "unit_verdicts_where_both_complete": dict(sorted(verdict.items())),
        "unit_verdict_agreement": round(verdict["agree"] / len(both), 6) if both else None,
        "vertex_model_versions": dict(versions), "vertex_cost_at_frozen_batch_rates_usd": round(cost, 6),
    }
    return summary, rows


def run(store_factory: Callable[[], Any], vertex_dir: Path = VERTEX_DIR, p_path: Path = P_PATH,
        p_results: Path = P_RESULTS, report_dir: Path = REPORT_DIR, jobs_path: Path = online.JOBS_PATH) -> dict:
    batches, assignment = batch_rows(vertex_dir)
    prompts = {j.job_id: j.prompt_sha256 for j in online.load_jobs(jobs_path)}
    if set(assignment) != set(prompts):
        raise online.StopRun("the Vertex ledger does not cover the frozen job file")
    files = download(store_factory(), batches, vertex_dir / "replication_outputs")   # skips files already local
    answers = vertex_answers(files, assignment, prompts)
    p_rows = [json.loads(line) for line in p_path.open(encoding="utf-8")]
    p_text = {}
    with p_results.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            p_text[r["job_id"]] = r.get("raw_text")
    summary, rows = compare(answers, p_rows, p_text)
    private = vertex_dir / "replication_comparison.jsonl"
    with private.open("w", encoding="utf-8") as out:
        out.write("".join(json.dumps(r) + "\n" for r in rows))
    receipt = {"section": 18, "purpose": "replication check on P; never changes P or the frame",
               "batch_jobs": [{"key": b["key"], "n": b["n"]} for b in batches],
               "output_files": {str(p.relative_to(vertex_dir)): online.sha256_file(p)
                                for paths in files.values() for p in paths},
               "p_sha256": online.sha256_file(p_path), "comparison_sha256": online.sha256_file(private),
               "code_sha256": online.sha256_file(Path(__file__)), **summary,
               "reference_values_opened": False,
               "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "replication_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run"])
    parser.parse_args(argv)
    receipt = run(lambda: vertex.GcsStore("gen-lang-client-0244811936"))
    keys = ("identical_answer_text", "criterion_pairs", "criterion_agreement_where_both_labelled",
            "section14_parser_pairs", "vertex_tasks_complete", "tasks_complete_in_both",
            "unit_verdicts_where_both_complete", "unit_verdict_agreement", "vertex_model_versions",
            "vertex_cost_at_frozen_batch_rates_usd")
    print(json.dumps({k: receipt[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
