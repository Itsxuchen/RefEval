"""Offline tests for parser amendment A1 (protocol section 15). No network, no reference values.

Runs under pytest or directly: python tests/test_wildifeval_parser_v2.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import wildifeval_parser_v2 as p2  # noqa: E402
from src import wildifeval_vertex_judge as w  # noqa: E402

CASES = [
    # (answer text, primary label, rule, label with partial read as FAIL)
    ("Yes", 1, "v1-yes-no", 1),
    ("No, the response is too short.", 0, "v1-yes-no", 0),
    ("The constraint is partially satisfied, so no.", None, "missing-partial", 0),   # v1 sees no "no" in five words
    ("The constraint is **not satisfied**.\n\nThe response ...", 0, "verdict-phrase", 0),
    ("The constraint is **satisfied**.", 1, "verdict-phrase", 1),
    ("The constraint is not fully satisfied.", 0, "verdict-phrase", 0),
    ("The response does not satisfy the constraint.", 0, "verdict-phrase", 0),
    ("The response fails to meet the constraint.", 0, "verdict-phrase", 0),
    ("The response satisfies the constraint.", 1, "verdict-phrase", 1),
    ('The constraint "Make sure the plan is satisfied" is not met.', 0, "verdict-phrase", 0),
    ("To determine if the constraint is satisfied, let's count the words:", None, "missing-no-verdict", None),
    ("The constraint is **partially satisfied**.", None, "missing-partial", 0),
    ("The constraint is not applicable (and therefore satisfied by default).", None, "missing-abstain", None),
    ("The constraint is not satisfied (or rather, it is not yet applicable).", None, "missing-abstain", None),
    ("The constraint cannot be evaluated.", None, "missing-abstain", None),
    ("The constraint is satisfied, but the constraint is not satisfied.", None, "missing-conflict", None),
    ("Reasoning:\nThe constraint is satisfied.", None, "missing-no-verdict", None),   # verdict not on the first line
    ("Sí, la restricción se cumple.", 1, "first-word-lexicon", 1),
    ("**Ja**, die Vorgabe ist erfüllt.", 1, "first-word-lexicon", 1),
    ("Non, la contrainte n'est pas respectée.", 0, "first-word-lexicon", 0),
    ("Нет. Ограничение не соблюдено.", 0, "first-word-lexicon", 0),
    ("是的，該限制條件已滿足。", 1, "first-word-lexicon", 1),
    ("Tak, to ograniczenie zostało spełnione.", None, "missing-no-verdict", None),   # ambiguous word left out
    ("Ja die Vorgabe ist erfüllt", None, "missing-no-verdict", None),                 # no punctuation after it
    ("", None, "missing-empty", None),
    (None, None, "missing-empty", None),
]


def test_rules_on_fixed_cases():
    for text, label, rule, sens in CASES:
        assert p2.parse(text) == (label, rule), text
        assert p2.parse(text, partial_as_fail=True)[0] == sens, text


def test_frozen_labels_are_never_changed():
    texts = [t for t, *_ in CASES] + ["Yes, but the constraint is not satisfied.", "no", "Answer: YES",
                                      "The constraint is satisfied. No issues.", "It is not. No."]
    for text in texts:
        v1, _ = w.parse_verdict(text)
        if v1 is not None:
            assert p2.parse(text) == (v1, "v1-yes-no")
            assert p2.parse(text, partial_as_fail=True)[0] == v1


def make_run(tmp: Path, texts: list[str | None]) -> tuple[Path, Path, Path]:
    run, report = tmp / "run", tmp / "report"
    run.mkdir()
    report.mkdir()
    with (run / "results.jsonl").open("w", encoding="utf-8") as out:
        for i, text in enumerate(texts):
            label, _ = w.parse_verdict(text)
            out.write(json.dumps({"job_id": f"j{i}", "unit_id": f"u{i // 2}", "cluster_id": f"c{i // 4}",
                                  "task_id": f"t{i // 4}", "producer": w.PRODUCERS[i % 5],
                                  "split": "pilot" if i < 4 else "confirmation", "parser_version": w.PARSER_VERSION,
                                  "label": label, "status": "success" if label is not None else "unparseable",
                                  "raw_text": text}, ensure_ascii=False) + "\n")
    (report / "summary.json").write_text(json.dumps({"results_sha256": w.sha256_file(run / "results.jsonl")}))
    protocol = tmp / "protocol.md"
    protocol.write_text(f"## 14. earlier\n\ntext\n\n{p2.SECTION_HEADING}\n\nrules\n\n## 16. later\n", encoding="utf-8")
    return run, report, protocol


def test_relabel_writes_predictions_and_receipt_without_text():
    texts = ["Yes", "The constraint is not satisfied.", "The constraint is partially satisfied.", "No",
             "The constraint is satisfied.", "", "Sí, se cumple.", "Yes"]
    with tempfile.TemporaryDirectory() as d:
        run, report, protocol = make_run(Path(d), texts)
        receipt = p2.relabel(run, report, protocol, expected_rows=len(texts))
        rows = [json.loads(l) for l in (run / "results_parser_v2.jsonl").open(encoding="utf-8")]
        assert [r["label"] for r in rows] == [1, 0, None, 0, 1, None, 1, 1]
        assert [r["label_v1"] for r in rows] == [1, None, None, 0, None, None, None, 1]
        assert [r["label_partial_as_fail"] for r in rows] == [1, 0, 0, 0, 1, None, 1, 1]
        assert all("raw_text" not in r and r["parser_version"] == p2.PARSER_VERSION for r in rows)
        assert rows[1]["status"] == "success" and rows[2]["status"] == "unparseable"
        public = (report / "parser_v2_receipt.json").read_text(encoding="utf-8")
        assert "satisfied" not in public and "Sí" not in public
        assert receipt["labels"]["v2"] == {"fail": 2, "missing": 2, "pass": 4}
        assert receipt["source_results_sha256"] == w.sha256_file(run / "results.jsonl")
        assert receipt["output_sha256"] == w.sha256_file(run / "results_parser_v2.jsonl")
        assert receipt["protocol_section_sha256"] == p2.sha256_text(p2.protocol_section(protocol))
        frame = receipt["common_frame_preview"]
        assert frame["v1"]["tasks_complete"] == 0                  # both tasks have a v1-missing criterion
        assert frame["partial_as_fail"]["tasks_complete"] == 1     # task t0 completes; t1 keeps an empty answer
        assert receipt["reference_values_opened"] is False


def test_relabel_refuses_a_changed_export_or_label():
    texts = ["Yes", "The constraint is not satisfied."]
    with tempfile.TemporaryDirectory() as d:
        run, report, protocol = make_run(Path(d), texts)
        (report / "summary.json").write_text(json.dumps({"results_sha256": "0" * 64}))
        try:
            p2.relabel(run, report, protocol, expected_rows=2)
            raise AssertionError("hash mismatch was accepted")
        except w.StopRun:
            pass
    with tempfile.TemporaryDirectory() as d:
        run, report, protocol = make_run(Path(d), texts)
        rows = [json.loads(l) for l in (run / "results.jsonl").open()]
        rows[0]["label"] = 0                                        # stored label no longer matches its text
        (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (report / "summary.json").write_text(json.dumps({"results_sha256": w.sha256_file(run / "results.jsonl")}))
        try:
            p2.relabel(run, report, protocol, expected_rows=2)
            raise AssertionError("tampered label was accepted")
        except w.StopRun:
            pass
    with tempfile.TemporaryDirectory() as d:
        run, report, protocol = make_run(Path(d), texts)
        protocol.write_text("## 14. only\n", encoding="utf-8")
        try:
            p2.relabel(run, report, protocol, expected_rows=2)
            raise AssertionError("missing section 15 was accepted")
        except w.StopRun:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
