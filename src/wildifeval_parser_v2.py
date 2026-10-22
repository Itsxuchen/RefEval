"""Parser amendment A1 for WildIFEval P (protocol section 15): read explicit verdicts the frozen parser misses.

The pinned upstream prompt ends with "Is the following constraint satisfied: ...? ###Answer:" and does not ask for
Yes or No, so ``gemini-3.1-flash-lite`` often answers "The constraint is (not) satisfied." The section 14 parser
reads only "yes" or "no" among the first five alphabetic words and marks such answers missing. This module re-reads
the stored answer texts of the completed batch run. It sends no request and reads no reference score R.

Rules, applied in order to each stored answer:

1. A label from the frozen section 14 parser is kept unchanged.
2. An empty answer stays missing.
3. The first non-empty line is the decision window. Quoted spans and markdown emphasis are removed from it.
4. A first word from a closed list of yes/no words in other languages, directly followed by punctuation, decides.
5. A partial or abstaining verdict ("partially", "not applicable", "cannot be determined") stays missing.
   The sensitivity variant reads a partial verdict as FAIL.
6. Clauses opened by "if" or "whether" are dropped ("To determine if the constraint is satisfied, ...").
   Exactly one of a negative or a positive satisfaction phrase then decides; both or neither stay missing.

    python -m src.wildifeval_parser_v2 relabel

``relabel`` checks the frozen export against the run's ``summary.json``, writes the private relabelled predictions
next to it without answer texts, and writes a public receipt with counts and hashes only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src import wildifeval_vertex_judge as online

ROOT = online.ROOT
RUN_DIR = ROOT / "data" / "interim" / "wildifeval_external" / "flashlite_gemini_batch"          # private
REPORT_DIR = ROOT / "artifacts" / "reports" / "wildifeval_external" / "flashlite_gemini_batch"  # public, no text
PROTOCOL_PATH = online.PROTOCOL_PATH
SECTION_HEADING = "## 15. Parser amendment A1 before reference access — 2026-10-22"

PARSER_VERSION = "first-line-verdict-text-v2"
V1_PARSER_VERSION = online.PARSER_VERSION
EXPECTED_ROWS = online.EXPECTED_JOBS

# First word of the window, directly followed by punctuation. Words that also mean something else at the start of
# a sentence in another language are left out (for example "si", "tak", "так", "da", "ano").
YES_WORDS = frozenset({"sí", "sim", "sì", "ja", "oui", "да", "ya", "是", "是的", "はい", "네", "예", "evet", "بله",
                       "igen", "kyllä", "ναι"})
NO_WORDS = frozenset({"nein", "nei", "nej", "nee", "non", "não", "нет", "ні", "nie", "tidak", "不", "不是", "否",
                      "いいえ", "아니요", "아니오", "hayır", "خیر", "نه", "nem", "ei", "όχι"})
FIRST_WORD = re.compile(r"^\W*([^\W\d_]+)[,.!:;，、。،]")
QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|`[^`\n]*`')
MARKUP = re.compile(r"[*_#>]+")
ADVERB = r"(?:fully |entirely |completely |clearly |successfully )?"
NEGATIVE = re.compile(r"(?:\bnot|n't) " + ADVERB + r"(?:satisfied|met|fulfill?ed|satisfy|meet|fulfill?)\b"
                      r"|\bfail(?:s|ed)? to " + ADVERB + r"(?:satisfy|meet|fulfill?)\b|\bunsatisfied\b")
POSITIVE = re.compile(r"\b(?:is|was|are|were|been) " + ADVERB + r"(?:satisfied|met|fulfill?ed)\b"
                      r"|\b(?:satisfies|meets|fulfill?s)\b")
PARTIAL = re.compile(r"\bpartial(?:ly)?\b")
ABSTAIN = re.compile(r"\bnot (?:yet )?applicable\b|\bn/a\b|\bnot evaluable\b|\b(?:cannot|can't|can not) be (?:determined|"
                     r"evaluated|assessed|verified|judged)\b|\b(?:unable|impossible) to (?:determine|evaluate|assess|"
                     r"verify|judge)\b")
CONDITION = re.compile(r"\b(?:if|whether)\b[^,.;:!?]*")   # "to determine if the constraint is satisfied, ..."


def window(text: str) -> str:
    """First non-empty line, lower-cased, without quoted spans or markdown emphasis."""
    line = next((l for l in text.splitlines() if l.strip()), "")
    line = MARKUP.sub("", QUOTED.sub(" ", line))
    return " ".join(line.lower().split())


def parse(text: str | None, partial_as_fail: bool = False) -> tuple[int | None, str]:
    """Return (label, rule). The rule names the step that decided, or why the answer stays missing."""
    label, _ = online.parse_verdict(text)
    if label is not None:
        return label, "v1-yes-no"
    if not (text or "").strip():
        return None, "missing-empty"
    w = window(text)
    m = FIRST_WORD.match(w)
    if m and m.group(1) in YES_WORDS:
        return 1, "first-word-lexicon"
    if m and m.group(1) in NO_WORDS:
        return 0, "first-word-lexicon"
    if ABSTAIN.search(w):
        return None, "missing-abstain"
    if PARTIAL.search(w):
        return (0, "partial-as-fail") if partial_as_fail else (None, "missing-partial")
    w = CONDITION.sub(" ", w)
    negative = bool(NEGATIVE.search(w))
    positive = bool(POSITIVE.search(NEGATIVE.sub(" ", w)))
    if negative and positive:
        return None, "missing-conflict"
    if negative or positive:
        return int(positive), "verdict-phrase"
    return None, "missing-no-verdict"


def protocol_section(path: Path | None = None) -> str:
    text = (path or PROTOCOL_PATH).read_text(encoding="utf-8")
    if SECTION_HEADING not in text:
        raise online.StopRun("protocol section 15 is missing; the relabelling is not authorized")
    return SECTION_HEADING + text.split(SECTION_HEADING, 1)[1].split("\n## ", 1)[0]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relabel(run_dir: Path = RUN_DIR, report_dir: Path = REPORT_DIR, protocol_path: Path | None = None,
            expected_rows: int = EXPECTED_ROWS) -> dict[str, Any]:
    source, summary_path = run_dir / "results.jsonl", report_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_sha = online.sha256_file(source)
    if source_sha != summary["results_sha256"]:
        raise online.StopRun("results.jsonl does not match the hash in summary.json")
    section = protocol_section(protocol_path)

    rules: Counter[str] = Counter()
    by_producer: dict[str, Counter[str]] = defaultdict(Counter)
    labels = {name: Counter() for name in ("v1", "v2", "partial_as_fail")}
    complete = {name: defaultdict(lambda: True) for name in labels}
    task_cluster: dict[str, tuple[str, str]] = {}
    units: set[str] = set()
    out_path = run_dir / "results_parser_v2.jsonl"
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    rows = 0
    with source.open(encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as out:
        for line in src:
            r = json.loads(line)
            rows += 1
            text = r.pop("raw_text", None)
            v1, _ = online.parse_verdict(text)
            if v1 != r["label"] or r["parser_version"] != V1_PARSER_VERSION:
                raise online.StopRun(f"stored label of {r['job_id']} differs from the frozen parser")
            v2, rule = parse(text)
            sens, _ = parse(text, partial_as_fail=True)
            if v1 is not None and (v2 != v1 or sens != v1):
                raise online.StopRun(f"amendment changed a frozen label: {r['job_id']}")
            rules[rule] += 1
            by_producer[r["producer"]][rule] += 1
            for name, value in (("v1", v1), ("v2", v2), ("partial_as_fail", sens)):
                labels[name]["missing" if value is None else ("pass" if value else "fail")] += 1
                complete[name][r["task_id"]] &= value is not None
            task_cluster[r["task_id"]] = (r["cluster_id"], r["split"])
            units.add(r["unit_id"])
            if r["status"] in ("success", "unparseable"):
                r["status"] = "success" if v2 is not None else "unparseable"
            r.update({"parser_version": PARSER_VERSION, "label": v2, "label_v1": v1, "parser_rule": rule,
                      "label_partial_as_fail": sens})
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    if rows != expected_rows:
        tmp_path.unlink()
        raise online.StopRun(f"expected {expected_rows} rows, read {rows}")
    tmp_path.replace(out_path)

    def frame(name: str) -> dict[str, Any]:
        kept = [t for t, ok in complete[name].items() if ok]
        clusters = {s: {task_cluster[t][0] for t in kept if task_cluster[t][1] == s} for s in ("pilot", "confirmation")}
        n_clusters = len(clusters["pilot"] | clusters["confirmation"])
        share = len(kept) / len(complete[name])
        return {"tasks_complete": len(kept), "task_retention": round(share, 6), "clusters": n_clusters,
                "pilot_clusters": len(clusters["pilot"]), "confirmation_clusters": len(clusters["confirmation"]),
                "gates_preview": {"clusters_ge_500": n_clusters >= 500, "pilot_ge_100": len(clusters["pilot"]) >= 100,
                                  "confirmation_ge_300": len(clusters["confirmation"]) >= 300,
                                  "retention_ge_0.70_each_producer": share >= 0.70}}

    receipt = {
        "amendment": "A1", "parser_version": PARSER_VERSION, "replaces_for_primary": V1_PARSER_VERSION,
        "source_results_sha256": source_sha, "output_sha256": online.sha256_file(out_path),
        "code_sha256": online.sha256_file(Path(__file__)), "protocol_section_sha256": sha256_text(section),
        "rows": rows, "tasks": len(complete["v2"]), "units": len(units),
        "rules": dict(sorted(rules.items())), "rules_by_producer": {p: dict(sorted(c.items()))
                                                                    for p, c in sorted(by_producer.items())},
        "labels": {k: dict(sorted(v.items())) for k, v in labels.items()},
        "common_frame_preview": {name: frame(name) for name in labels},
        "note": ("Missingness preview from P alone: a task is complete when every criterion of all five producers "
                 "has a label, and a cluster counts when at least one of its tasks is complete. It is not the "
                 "completed-data design freeze of section 14 step 6."),
        "reference_values_opened": False,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "parser_v2_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["relabel"])
    parser.parse_args(argv)
    try:
        receipt = relabel()
    except online.StopRun as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({k: receipt[k] for k in ("rules", "labels", "common_frame_preview")}, indent=2))


if __name__ == "__main__":
    main()
