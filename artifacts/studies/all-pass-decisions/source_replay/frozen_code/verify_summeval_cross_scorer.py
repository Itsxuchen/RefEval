"""Independently check crossed human scoring using four-bit failure sets."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.external_summeval_frame import DEFAULT_CACHE, DIMS, OUT as PRIOR, SOURCES


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/summeval_cross_scorer_20261126"
PROTOCOL = ROOT / "context/summeval_cross_scorer_20261126.md"
PROTOCOL_HASH = "a0b38a69236c769c05599ac4f68c378022ccdb3a47f7c55aee2dab179a5cb1df"
MEASURES = ("early", "endorsed", "eager_errors", "gated_errors")
MODELS = ("UniEval", "G-Eval GPT-4")
SHARES = tuple(range(0, 101, 10))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail_sets(passed):
    """Encode a row's four failed dimensions in the low four integer bits."""
    passed = np.asarray(passed)
    if passed.shape[-1] != 4 or not np.isin(passed, (0, 1)).all():
        raise ValueError("Expected four binary dimension labels")
    return np.sum((passed == 0) * np.array([1, 2, 4, 8]), axis=-1)


def independent_decisions(predicted_failures, auditor_failures, checked):
    """Derive both binary outputs by considering sets of outstanding failures."""
    predicted_failures = np.asarray(predicted_failures)
    auditor_failures = np.asarray(auditor_failures)
    checked = np.asarray(checked)
    revealed_failure = (auditor_failures & checked) != 0
    predicted_failures_remaining = (predicted_failures & (15 ^ checked)) != 0
    eager = ~(revealed_failure | predicted_failures_remaining)
    gated = ~revealed_failure & ((predicted_failures == 0) | (checked == 15))
    # Distinct algebra for the set on which the two decisions can differ.
    premature = ((predicted_failures != 0) & ~predicted_failures_remaining
                 & ~revealed_failure & (checked != 15))
    np.testing.assert_array_equal(eager & ~gated, premature)
    if np.any(gated & ~eager):
        raise AssertionError("Gated-only approval")
    return eager, gated, premature


def get_frames(cache):
    for name, (_, expected) in SOURCES.items():
        require(digest(cache / name) == expected, "Changed source: " + name)
    experts = {}
    with (cache / "experts.jsonl").open() as stream:
        for line in stream:
            data = json.loads(line)
            key = data["id"], data["model_id"]
            ratings = np.array([[r[d] for d in DIMS] for r in data["expert_annotations"]])
            require(key not in experts and ratings.shape == (3, 4), "Invalid expert row")
            require(np.isin(ratings, (1, 2, 3, 4, 5)).all(), "Invalid rating")
            experts[key] = ratings >= 4
    panels = {}
    held = set()
    for stage in ("held", "development"):
        path = PRIOR / f"{stage}_bits.json"
        metadata = json.loads((PRIOR / f"{stage}_manifest.json").read_text())
        require(digest(path) == metadata["panel_sha256"], "Changed historical judge panel")
        for row in json.loads(path.read_text())["panels"]["primary"]:
            key = tuple(row["unit"].rsplit("/", 1))
            require(key[0] == row["group"] and key not in panels, "Invalid join key")
            np.testing.assert_array_equal(experts[key].sum(axis=0) >= 2, row["reference"])
            panels[key] = tuple(row["predictions"][m] for m in MODELS)
            if stage == "held":
                held.add(key)
    require(set(experts) == set(panels) and len(held) == 1120 and len(experts) == 1600,
            "Invalid cohort")
    for frame, keys in (("held70", held), ("all100", set(experts))):
        ordered = sorted(keys)
        counts = Counter(k[0] for k in ordered)
        require(set(counts.values()) == {16}, "Unexpected article size")
        yield (frame, sorted(counts),
               np.asarray([panels[k] for k in ordered], dtype=int),
               fail_sets([experts[k] for k in ordered]))


def independently_rank_slots(pair_predictions, seed):
    flat_flags = (pair_predictions[:, 0] != pair_predictions[:, 1]).ravel()
    word = int.from_bytes(hashlib.sha256(b"criterion").digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([20260913, seed, word]))
    tie = rng.random(len(flat_flags))
    # Release-order DF prioritises flagged outputs in lexical order. Within each
    # flagged output it uses the seeded ties. Unflagged items are shuffled globally.
    def priority(i):
        return (0, i // 4, tie[i]) if flat_flags[i] else (1, 0, tie[i])

    order = sorted(range(len(flat_flags)), key=priority)
    return np.argsort(order).reshape(-1, 4)


def expected_articles(pair_predictions, experts):
    """Return integer event counts for every article and auditor/scorer position."""
    n = len(experts)
    predicted = fail_sets(pair_predictions).T
    reference = experts == 0
    counts = np.zeros((2, len(SHARES), n // 16, 3, 3, 4), dtype=np.int64)
    any_early = np.zeros((2, n, 3), dtype=bool)
    half_masks = set()
    seed_counts = [set(), set()]
    for seed in range(1, 32):
        ranks = independently_rank_slots(pair_predictions, seed)
        for b, pct in enumerate(SHARES):
            checked = ((ranks < (4 * n * pct // 100)) * [1, 2, 4, 8]).sum(axis=1)
            for m in range(2):
                eager, gated, early = independent_decisions(
                    predicted[m, :, None], experts, checked[:, None]
                )
                if pct == 50:
                    any_early[m] |= early
                    half_masks.add(checked.astype(np.uint8).tobytes())
                    outside_endorsements = reference.sum(axis=1)[:, None] - reference
                    seed_counts[m].add((int(early.sum()), int((early & reference).sum()),
                                        int((early * outside_endorsements).sum())))
                for a in range(3):
                    for s in range(3):
                        row = np.stack((early[:, a], early[:, a] & reference[:, s],
                                        eager[:, a] != reference[:, s],
                                        gated[:, a] != reference[:, s]), axis=-1)
                        counts[m, b, :, a, s] += row.reshape(-1, 16, 4).sum(axis=1)
    np.testing.assert_array_equal(counts[..., 3] - counts[..., 2],
                                  2 * counts[..., 1] - counts[..., 0])
    require(not counts[:, [0, -1], ..., 0].any(), "Early approvals at an endpoint")
    for a in range(3):
        require(not counts[:, -1, :, a, a, 2:].any(), "Own error after full checking")
    half_diagnostic = {
        "distinct_checked_masks": len(half_masks),
        "distinct_early_events": {
            model: {
                "distinct_summaries": int(any_early[m].any(axis=1).sum()),
                "distinct_articles": int(any_early[m].reshape(n // 16, -1).any(axis=1).sum()),
                "distinct_seed_early_and_endorsement_totals": len(seed_counts[m]),
            }
            for m, model in enumerate(MODELS)
        }
    }
    return counts, half_diagnostic


def check_summaries(frame, judge, pct, rates, draws, reported):
    # rates are raw integer counts per article summed over 31 orderings.
    n = 16 * len(rates)
    own = np.mean(rates[:, [0, 1, 2], [0, 1, 2]], axis=1) / 31
    external = np.mean(rates[:, [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], axis=1) / 31
    early_count = own[:, 0].sum()
    point = {"early_count": early_count, "early_per100": early_count * 100 / n}
    for name, values in (("own", own), ("other", external)):
        point[f"{name}_endorse_pct"] = (100 * values[:, 1].sum() / early_count
                                         if early_count else np.nan)
    own_diff = own[:, 3] - own[:, 2]
    ext_diff = external[:, 3] - external[:, 2]
    for name, difference in (("own_delta", own_diff), ("other_delta", ext_diff),
                              ("shift", ext_diff - own_diff)):
        point[name] = 100 * difference.sum() / n
        bounds = np.quantile(100 * (draws @ difference) / n, (.025, .975))
        point[name + "_lo"], point[name + "_hi"] = bounds
    for name, expected in point.items():
        got = reported[name]
        if np.isnan(expected):
            require(got in ("", "nan", "NA"), (frame, judge, pct, name, got))
        else:
            np.testing.assert_allclose(float(got), expected, atol=1e-7, rtol=0,
                                       err_msg=str((frame, judge, pct, name)))
    return len(point)


def verify(cache, report):
    require(digest(PROTOCOL) == PROTOCOL_HASH, "Protocol digest changed")
    with gzip.open(report / "article_counts.csv.gz", "rt", newline="") as stream:
        article = {}
        for r in csv.DictReader(stream):
            k = (r["frame"], r["judge"], int(r["budget_pct"]), r["article"],
                 int(r["auditor"]), int(r["scorer"]))
            require(k not in article and int(r["n_outputs"]) == 16, "Invalid saved article key")
            article[k] = [int(r[name]) for name in MEASURES]
    with (report / "summary.csv").open() as stream:
        summary = {(r["frame"], r["judge"], int(r["budget_pct"])): r
                   for r in csv.DictReader(stream)}
    checked_rows, checked_metrics = 0, 0
    half_diagnostics = {}
    for frame, ids, preds, experts in get_frames(cache):
        expected, half_diagnostics[frame] = expected_articles(preds, experts)
        n = len(ids)
        draws = np.random.default_rng(20261125).multinomial(n, np.ones(n) / n, size=1999)
        for m, judge in enumerate(MODELS):
            for b, pct in enumerate(SHARES):
                for i, article_id in enumerate(ids):
                    for a in range(3):
                        for s in range(3):
                            key = frame, judge, pct, article_id, a + 1, s + 1
                            require(article.pop(key) == list(expected[m, b, i, a, s]), key)
                            checked_rows += 1
                key = frame, judge, pct
                checked_metrics += check_summaries(
                    *key, expected[m, b], draws, summary.pop(key)
                )
    require(not article and not summary, "Unexpected exported results")
    verification = {
        "protocol_sha256": digest(PROTOCOL),
        "article_rows_reconstructed": checked_rows,
        "integer_count_cells_reconstructed": 4 * checked_rows,
        "point_and_interval_cells_reconstructed": checked_metrics,
        "post_run_descriptive_half_budget": half_diagnostics,
        "method": "Recomputed failure-set decisions from archived fixed judge bits and pinned individual ratings, independently ranked every criterion, and checked all nine roles and endpoint invariants.",
        "verified_files_sha256": {n: digest(report / n) for n in ("summary.csv", "article_counts.csv.gz")},
        "validator_sha256": digest(Path(__file__)),
    }
    (report / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    verify(args.cache, args.report)
