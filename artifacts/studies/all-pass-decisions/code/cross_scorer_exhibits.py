"""Rebuild the all-rater table and pointwise budget curves from cluster counts."""
from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs"
OUTPUT = ROOT / "generated"
MODELS = ("UniEval", "G-Eval GPT-4")
CASES = (("held70", MODELS[0]), ("JudgmentBench", "GPT-5.4"), ("held70", MODELS[1]))
LABELS = ("SummEval / UniEval", "JudgmentBench / GPT-5.4", "SummEval / G-Eval")
BUDGETS = tuple(range(0, 101, 10))
METRICS = ("early", "own_endorsed", "other_endorsed", "own_eager_error",
           "own_gated_error", "other_eager_error", "other_gated_error")


def summ_clusters():
    role_rows = defaultdict(dict)
    group_sizes = {}
    with gzip.open(INPUT / "cross_scorer_article_counts.csv.gz", "rt", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["frame"], r["judge"], int(r["budget_pct"]), r["article"])
            pair = (int(r["auditor"]), int(r["scorer"]))
            size = int(r["n_outputs"])
            if size != 16 or pair in role_rows[key]:
                raise ValueError("Duplicate role or unexpected article size")
            group_sizes[key] = size
            role_rows[key][pair] = [int(r[k]) for k in ("early", "endorsed", "eager_errors", "gated_errors")]
    counts = defaultdict(dict)
    for (frame, judge, budget, article), pairs in role_rows.items():
        if set(pairs) != {(a, b) for a in (1, 2, 3) for b in (1, 2, 3)}:
            raise ValueError("Incomplete expert rotation")
        same, other = (np.mean([row for (a, b), row in pairs.items() if (a == b) == own],
                               axis=0) / 31 for own in (True, False))
        if same[0] != other[0]:
            raise ValueError("Changing scorer changed the early approvals")
        vector = [same[0], same[1], other[1], same[2], same[3], other[2], other[3]]
        counts[frame, judge, budget][article] = (group_sizes[frame, judge, budget, article], vector)
    return counts


def legal_clusters(counts):
    with (INPUT / "cross_scorer_legal_clusters.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            key = "JudgmentBench", "GPT-5.4", round(100 * float(r["budget_share"]))
            if r["base_task"] in counts[key]:
                raise ValueError("Duplicate legal task")
            counts[key][r["base_task"]] = (int(r["n_outputs"]), [float(r[k]) for k in METRICS])


def summarize(groups, draws):
    sizes, matrix = zip(*(groups[k] for k in sorted(groups)))
    early, own_yes, other_yes, eo, go, ex, gx = np.asarray(matrix).T
    contrasts = np.stack((go - eo, gx - ex, (gx - ex) - (go - eo)), axis=1)
    np.testing.assert_allclose(contrasts[:, 0], 2 * own_yes - early, atol=1e-12)
    np.testing.assert_allclose(contrasts[:, 1], 2 * other_yes - early, atol=1e-12)
    n = sum(sizes)
    values = 100 * contrasts.sum(axis=0) / n
    boot = 100 * (draws @ contrasts) / (draws @ np.asarray(sizes))[:, None]
    intervals = np.percentile(boot, [2.5, 97.5], axis=0)
    total_early = early.sum()
    record = dict(n_outputs=n, n_groups=len(sizes), early_count=total_early,
                  early_per100=100 * total_early / n,
                  own_endorse_pct=100 * own_yes.sum() / total_early if total_early else "",
                  other_endorse_pct=100 * other_yes.sum() / total_early if total_early else "")
    for idx, name in enumerate(("own_delta", "other_delta", "shift")):
        record[name] = values[idx]
        record[name + "_lo"], record[name + "_hi"] = intervals[:, idx]
    return record


def table(results):
    rows = []
    for case, label in zip(CASES, LABELS):
        r = results[*case, 50]
        rows.append(" & ".join([label, f"{r['early_per100']:.2f}", f"{r['own_endorse_pct']:.1f}",
                                f"{r['other_endorse_pct']:.1f}", f"${r['own_delta']:+.2f}$",
                                f"${r['other_delta']:+.2f}$",
                                f"${r['shift']:+.2f}\\;[{r['shift_lo']:+.2f},{r['shift_hi']:+.2f}]$"]) + r" \\")
    (OUTPUT / "crossover.tex").write_text(r"\newcommand{\crossrows}{%" + "\n" + "\n".join(rows) + "\n}\n")


def draw(results):
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.9), layout="constrained")
    for ax, case, title in zip(axes, CASES, LABELS):
        grid = [results[*case, b] for b in BUDGETS]
        ax.axhline(0, color=".35", lw=.7)
        ax.axvline(50, color=".65", lw=.8, ls=":")
        for metric, color, marker, desc in (("own_delta", "#116285", "o", "Auditing expert"),
                                           ("other_delta", "#b64b15", "s", "Other expert")):
            ax.fill_between(BUDGETS, [r[metric + "_lo"] for r in grid], [r[metric + "_hi"] for r in grid],
                            color=color, alpha=.14, lw=0)
            ax.plot(BUDGETS, [r[metric] for r in grid], color=color, marker=marker,
                    lw=1.4, ms=2.5, label=desc)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Budget (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Gated − eager errors per 100", fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2,
               frameon=False, fontsize=8)
    fig.savefig(OUTPUT / "crossover.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def run():
    data = summ_clusters()
    legal_clusters(data)
    panels = {**{("held70", m): (70, 1120) for m in MODELS},
              **{("all100", m): (100, 1600) for m in MODELS}, ("JudgmentBench", "GPT-5.4"): (30, 206)}
    if set(data) != {(*panel, budget) for panel in panels for budget in BUDGETS}:
        raise ValueError("Missing or unexpected analysis condition")
    results = {}
    for panel, (n, n_outputs) in panels.items():
        draws = np.random.default_rng(20261125).multinomial(n, np.ones(n) / n, size=1999)
        for budget in BUDGETS:
            key = (*panel, budget)
            r = summarize(data[key], draws)
            if (r["n_groups"], r["n_outputs"]) != (n, n_outputs):
                raise ValueError("Unexpected cluster or output count")
            if budget in (0, 100) and (r["early_count"] or r["own_delta"] or r["other_delta"]):
                raise ValueError("Updates disagree at a common endpoint")
            results[key] = r
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "crossover_curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "judge", "budget_pct", *next(iter(results.values()))])
        writer.writerows([*k, *v.values()] for k, v in results.items())
    table(results)
    draw(results)


if __name__ == "__main__":
    run()
