"""Replay a frozen, within-summary expert crossover with paired article uncertainty."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np

from src.conjunction_policy_study import make_order
from src.external_summeval_frame import (
    CUTOFF, DEFAULT_CACHE, DIMS, MODELS, OUT as PREVIOUS, ROOT, SOURCES,
    decode_grouped_subset, load_sources, partition, sha,
)

OUT = ROOT / "artifacts/reports/summeval_cross_scorer_20261126"
SEEDS = range(1, 32)
BUDGETS = range(0, 101, 10)
METRICS = ("early", "endorsed", "eager_errors", "gated_errors")
PROTOCOL_HASH = "a0b38a69236c769c05599ac4f68c378022ccdb3a47f7c55aee2dab179a5cb1df"


def check_freeze():
    receipt = (OUT / "design.sha256").read_text().splitlines()
    if not receipt or receipt[0].split()[0] != PROTOCOL_HASH:
        raise ValueError("Unexpected design freeze")
    for line in receipt:
        digest, path = line.split(maxsplit=1)
        if sha(ROOT / path) != digest:
            raise ValueError(f"Changed frozen input: {path}")


def load_frames(cache):
    sources = load_sources(cache)
    development, held = partition(sources["experts.jsonl"])
    decoded, missing, _ = decode_grouped_subset(sources, development | held)
    if missing or len(decoded) != 1600:
        raise ValueError(f"Incomplete decoded frame: {missing}")
    historical = {}
    for name, groups in (("development", development), ("held", held)):
        panel = json.loads((PREVIOUS / f"{name}_bits.json").read_text())["panels"]["primary"]
        if len(panel) != len(groups) * 16 or set(row["group"] for row in panel) != groups:
            raise ValueError("Historical split differs from the locked article partition")
        historical.update({row["unit"]: row for row in panel})
    keys, models, experts = [], [], []
    for key, _, scores in decoded:
        old = historical["/".join(key)]
        ratings = np.array([[r[d] for d in DIMS] for r in sources["experts.jsonl"][key]["expert_annotations"]])
        if ratings.shape != (3, 4) or not np.isin(ratings, np.arange(1, 6)).all():
            raise ValueError(f"Invalid individual rating on {key}")
        judge_bits = scores >= CUTOFF
        rater_bits = ratings >= 4
        if not np.array_equal(judge_bits, [old["predictions"][m] for m in MODELS]):
            raise ValueError(f"Judge bit mismatch on {key}")
        if not np.array_equal(rater_bits.sum(axis=0) >= 2, old["reference"]):
            raise ValueError(f"Majority mismatch on {key}")
        keys.append(key)
        models.append(judge_bits)
        experts.append(rater_bits)
    if keys != sorted(keys) or len(set(keys)) != 1600 or len(historical) != 1600:
        raise ValueError("Incomplete, duplicated or nonlexical summary keys")
    articles, systems = zip(*keys)
    if len(set(systems)) != 16 or any(articles.count(doc) != 16 for doc in set(articles)):
        raise ValueError("Expected the complete 100 by 16 frame")
    articles, models, experts = np.array(articles), np.array(models), np.array(experts)
    held_mask = np.isin(articles, sorted(held))
    return {"held70": (articles[held_mask], models[held_mask], experts[held_mask]),
            "all100": (articles, models, experts)}


def verdicts(primary, experts, checked):
    """Return eager and gated verdicts [summary, auditing expert], before scoring."""
    eager = np.where(checked[:, None, :], experts, primary[:, None, :]).all(axis=2)
    witnessed_fail = (checked[:, None, :] & ~experts).any(axis=2)
    gated = ~witnessed_fail & (primary.all(axis=1) | checked.all(axis=1))[:, None]
    return eager, gated


def article_counts(articles, models, experts):
    groups = np.unique(articles)
    n = len(articles)
    inputs = {"task_index": np.repeat(np.arange(n), 4), "task_ids": np.arange(n),
              "prediction": models[:, 0, :].ravel(), "secondary_predictions": models[:, 1, :].reshape(1, -1)}
    truth = experts.all(axis=2)
    # Dimensions are budget, judge, article, auditor slot, scorer slot, outcome.
    counts = np.zeros((len(BUDGETS), 2, len(groups), 3, 3, len(METRICS)), dtype=np.int64)
    for seed in SEEDS:
        order = make_order(inputs, "disagreement_then_random", "release", seed)
        ranks = np.empty(n * 4, dtype=int)
        ranks[order] = np.arange(n * 4)
        for b, pct in enumerate(BUDGETS):
            checked = ranks.reshape(n, 4) < n * 4 * pct // 100
            for j in range(len(MODELS)):
                eager, gated = verdicts(models[:, j], experts, checked)
                if (gated & ~eager).any():
                    raise AssertionError("Gated-only approval")
                if pct in (0, 100) and not np.array_equal(eager, gated):
                    raise AssertionError("Endpoints should coincide")
                if pct == 100 and not np.array_equal(eager, truth):
                    raise AssertionError("Full audit should reproduce its expert")
                early = eager & ~gated
                scoring_truth = truth[:, None, :]
                early_by_pair = np.broadcast_to(early[:, :, None], (n, 3, 3))
                outcomes = np.stack((early_by_pair, early_by_pair & scoring_truth,
                                     eager[:, :, None] != scoring_truth,
                                     gated[:, :, None] != scoring_truth), axis=-1)
                counts[b, j] += outcomes.reshape(len(groups), 16, 3, 3, 4).sum(axis=1)
    if not np.array_equal(counts[..., 3] - counts[..., 2], 2 * counts[..., 1] - counts[..., 0]):
        raise AssertionError("Early-approval accounting does not reproduce loss difference")
    return groups, counts


def percent_ratio(a, b):
    return 100 * a / b if b else None


def summarize(article, draws):
    # Each output has three self cells and six other-rater cells.
    same = np.eye(3, dtype=bool)
    own = article[:, same, :].mean(axis=1) / len(SEEDS)
    other = article[:, ~same, :].mean(axis=1) / len(SEEDS)
    own_d = own[:, 3] - own[:, 2]
    other_d = other[:, 3] - other[:, 2]
    if not np.array_equal(own[:, 0], other[:, 0]) or np.any(other_d - own_d > 1e-10):
        raise AssertionError("Role-average accounting failed")
    size = 16 * len(article)
    metrics = dict(n_outputs=size, n_articles=len(article), early_count=own[:, 0].sum(),
                   early_per100=100 * own[:, 0].sum() / size,
                   own_endorse_pct=percent_ratio(own[:, 1].sum(), own[:, 0].sum()),
                   other_endorse_pct=percent_ratio(other[:, 1].sum(), other[:, 0].sum()))
    for name, contributions in (("own_delta", own_d), ("other_delta", other_d), ("shift", other_d - own_d)):
        metrics[name] = 100 * contributions.sum() / size
        draws_per100 = 100 * (draws @ contributions) / size
        metrics[name + "_lo"], metrics[name + "_hi"] = np.percentile(draws_per100, [2.5, 97.5])
    return metrics


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(cache, out=OUT):
    check_freeze()
    frames = load_frames(cache)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, role_rows = [], []
    with (out / "article_counts.csv.gz").open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed, io.TextIOWrapper(compressed, newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "judge", "budget_pct", "article", "auditor", "scorer", "n_outputs", *METRICS])
        for frame, inputs in frames.items():
            groups, counts = article_counts(*inputs)
            draws = np.random.default_rng(20261125).multinomial(len(groups), np.ones(len(groups)) / len(groups), size=1999)
            for b, pct in enumerate(BUDGETS):
                for j, judge in enumerate(MODELS):
                    key = dict(frame=frame, judge=judge, budget_pct=pct)
                    cells = counts[b, j]
                    summary_rows.append({**key, **summarize(cells, draws)})
                    for p in range(3):
                        for s in range(3):
                            writer.writerows([frame, judge, pct, group, p + 1, s + 1, 16, *values]
                                             for group, values in zip(groups, cells[:, p, s]))
                            total = cells[:, p, s].sum(axis=0)
                            role_rows.append({**key, "auditor": p + 1, "scorer": s + 1,
                                              "n_outputs": len(inputs[0]), "n_orders": len(SEEDS),
                                              **dict(zip(METRICS, map(int, total)))})
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "role_counts.csv", role_rows)
    manifest = dict(protocol_sha256=PROTOCOL_HASH, source_sha256={k: d for k, (_, d) in SOURCES.items()},
                    runner_sha256=sha(Path(__file__)), outputs={name: sha(out / name) for name in
                    ("summary.csv", "role_counts.csv", "article_counts.csv.gz")})
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.cache, args.out)
