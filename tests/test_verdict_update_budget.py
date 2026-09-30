"""Guard inferential presentation errors in the saved-row figure builder."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from src.plot_verdict_update_budget import DEFAULT_SOURCE, ROOT, plot, source_record, summarize_frame


def frame():
    rows = []
    for seed, fp, ff in [(0, 6, 9), (1, 2, 0), (2, 0, 4)]:
        rows.append(dict(dataset="test", task_order="release", policy="random_criterion",
                         budget_share=.2, seed=seed, n_tasks=10, n_criteria=20, budget_cap=4,
                         actual_queries=4, residual_fp=2+fp, residual_ff=5-ff,
                         residual_errors=7+fp-ff, gated_residual_fp=2,
                         gated_residual_ff=5, gated_residual_errors=7,
                         introduced_fp=fp, delayed_ff_corrections=ff))
    return pd.DataFrame(rows)


def test_paired_mean_and_canonical_exclusion():
    r = summarize_frame(frame(), seeds=(1, 2)).iloc[0]
    assert (r.delta_fp_mean, r.delta_ff_mean, r.delta_total_mean) == (-1, 2, 1)
    assert r.seeds == 2
    assert (r.delta_total_negative, r.delta_total_zero, r.delta_total_positive) == (1, 0, 1)


def test_mismatched_components_rejected():
    f = frame()
    f.loc[f.seed.eq(1), "introduced_fp"] = 1
    with pytest.raises(ValueError, match="decomposition"):
        summarize_frame(f, seeds=(1, 2))


def test_missing_and_duplicate_seeds_rejected():
    with pytest.raises(ValueError, match="seed"):
        summarize_frame(frame().iloc[:2], seeds=(1, 2))
    with pytest.raises(ValueError, match="duplicate"):
        summarize_frame(pd.concat([frame(), frame().iloc[1:2]]), seeds=(1, 2))


def test_cap_is_not_actual_expenditure():
    f = frame()
    f["actual_queries"] = 2
    assert summarize_frame(f, seeds=(1, 2)).iloc[0].actual_queries_mean == 2
    f.loc[f.seed.eq(1), "actual_queries"] = 5
    with pytest.raises(ValueError, match="available cap"):
        summarize_frame(f, seeds=(1, 2))


@pytest.mark.parametrize("relative", ["artifacts/expected/new", "artifacts/figures/new", "data/new"])
def test_renderer_rejects_protected_outputs(relative):
    with pytest.raises(ValueError, match="protected"):
        plot(DEFAULT_SOURCE, Path(relative))


def test_external_output_renders_full_saved_grid(monkeypatch, tmp_path):
    """Exercise actual rendering outside the checkout without redoing acquisition."""
    frozen_hash = hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "external-figures"
    receipt = plot(DEFAULT_SOURCE, output)
    assert receipt["groups"] == 19089
    assert receipt["randomized_rows_checked"] == 591759
    assert len(receipt["all_policies_in_csv"]) == 7
    assert len(receipt["policies_in_figures"]) == 3
    assert receipt["source"]["path"] == DEFAULT_SOURCE.relative_to(ROOT).as_posix()
    assert receipt["renderer"]["path"] == "src/plot_verdict_update_budget.py"
    assert set(receipt["generated_artifacts"]) == {
        "verdict_update_budget_data.csv", "verdict_update_budget.pdf",
        "verdict_update_budget_all_cells.pdf", "verdict_update_budget.png"}
    for name, info in receipt["generated_artifacts"].items():
        path = output / name
        assert path.stat().st_size == info["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
    text = (output / "verdict_update_budget.json").read_text()
    assert str(tmp_path) not in text
    assert json.loads(text) == receipt
    assert hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest() == frozen_hash
    r = next(r for r in receipt["randomized_20_percent"]
             if r["dataset"] == "JB GPT-5.4 | binary_only"
             and r["policy"] == "disagreement_then_random")
    assert r["actual_queries_min"] == r["actual_queries_max"] == 4046
    assert r["delta_total_mean"] == pytest.approx(8.290322580645162)


def test_external_source_receipt_uses_basename(tmp_path):
    source = tmp_path / "private-input.csv"
    source.write_text("id,value\n1,2\n")
    record = source_record(source)
    assert record["path"] == source.name
    assert record["path_scope"] == "basename-only"
    assert str(tmp_path) not in json.dumps(record)
