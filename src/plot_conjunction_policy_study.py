"""Export scientific figures from saved policy-study rows (no experiment rerun)."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "reference-label-value-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS = ROOT / "artifacts/reproduced/conjunction_policy_study"
OUT = ROOT / "artifacts/figures/conjunction_policy_study"
POLICIES = ["random_criterion", "disagreement_then_random", "sc_judge", "sc_disagreement"]
LABELS = ["Random criteria", "Disagreement + random", "Judge-first short circuit", "Disagreement-first short circuit"]
COLORS = ["#78848C", "#BD741B", "#235B91", "#A4517C"]
STYLES = [":", "--", "-", "-."]
DATASETS = ["DR Gemini 3.1 Pro", "JB GPT-5.4", "JB GPT-5.4-mini"]
NAMES = ["RuVerBench DR / Gemini", "JudgmentBench / GPT-5.4", "JudgmentBench / GPT-5.4-mini"]


def plot(reports: Path = REPORTS, output: Path = OUT) -> None:
    global REPORTS, OUT
    REPORTS, OUT = reports, output
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#263640", "text.color": "#263640",
                         "axes.edgecolor": "#A5ABB0", "savefig.facecolor": "white"})
    values = defaultdict(list)
    with gzip.open(REPORTS / "budget_curves.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            if row["dataset"] not in DATASETS or row["policy"] not in POLICIES or row["task_order"] != "release" or int(row["seed"]) == 0:
                continue
            n, e = int(row["n_tasks"]), int(row["primary_criterion_errors"])
            values[(row["dataset"], row["policy"], float(row["budget_share"]))].append(
                [100*int(row["criterion_errors_found"])/e, 100*int(row["certified_tasks"])/n,
                 100*int(row["residual_errors"])/n])
    curve_data = []
    for key, rows in values.items():
        assert len(rows) == 31
        quantiles = np.quantile(np.asarray(rows), [.05, .5, .95], axis=0)
        curve_data.append(dict(dataset=key[0], policy=key[1], budget_share=key[2],
                               p05=quantiles[0].tolist(), median=quantiles[1].tolist(), p95=quantiles[2].tolist()))
    for limit, suffix in ((1, "full"), (.4, "early")):
        fig, axes = plt.subplots(3, 3, figsize=(13.4, 10.0), sharex=True)
        for i, ds in enumerate(DATASETS):
            for j in range(3):
                ax = axes[i, j]
                for policy, label, color, style in zip(POLICIES, LABELS, COLORS, STYLES):
                    seq = sorted([r for r in curve_data if r["dataset"] == ds and r["policy"] == policy], key=lambda r:r["budget_share"])
                    x = 100*np.array([r["budget_share"] for r in seq])
                    median = np.array([r["median"][j] for r in seq])
                    low = np.array([r["p05"][j] for r in seq]); high = np.array([r["p95"][j] for r in seq])
                    ax.plot(x, median, label=label, color=color, linestyle=style, linewidth=1.8)
                    ax.fill_between(x, low, high, color=color, alpha=.10, linewidth=0)
                ax.set_xlim(0, 100*limit); ax.set_ylim(bottom=0)
                if j < 2:
                    ax.set_ylim(0, 102)
                ax.grid(axis="y", color="#D8DDE0", linewidth=.6, alpha=.8)
                ax.set_axisbelow(True)
                if i == 0:
                    ax.set_title(["Criterion-error recall (%)", "Certified annotations / tasks (%)", "Residual task errors (%)"][j], pad=12, fontsize=11)
                if j == 0:
                    ax.set_ylabel(NAMES[i], fontsize=10, labelpad=8)
                if i == 2:
                    ax.set_xlabel("Available reference-query budget (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.52,.065))
        fig.suptitle("Query-policy outcomes under fixed published references" + (" — early-budget view" if suffix == "early" else ""), y=.98, fontsize=15)
        fig.text(.065, .034, "Lines: medians; bands: 5th–95th percentiles over 31 randomized tie orders, not confidence intervals. Task order: release.", fontsize=9)
        fig.text(.065, .016, "Budgets are caps; policies may stop early. Disagreement policies use two auxiliary DR judges / one JB judge; their inference cost is separate.", fontsize=9)
        fig.subplots_adjust(left=.10, right=.98, top=.925, bottom=.17, wspace=.26, hspace=.28)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"policy_frontiers_{suffix}.{ext}", dpi=200)
        plt.close(fig)

    study = json.loads((REPORTS / "study.json").read_text())
    names = list(study["datasets"])
    orders = ["release", "predicted_fail_count", "random"]
    mat = np.full((len(names), 3), np.nan); counts = {}
    for r in study["pairwise_equal_actual"]:
        if r["a"] == "sc_judge" and r["b"] == "disagreement_then_random":
            denominator = r["equal_actual_active_points"]
            value = r.get("certification_vs_bit_discovery_conflict", 0)
            mat[names.index(r["dataset"]), orders.index(r["task_order"])] = 100*value/denominator if denominator else np.nan
            counts[(r["dataset"], r["task_order"])] = (value, denominator)
    short = {"DR Gemini 3.1 Pro":"DR · Gemini", "DR GPT-5.4 low":"DR · GPT-5.4", "DR Qwen-plus":"DR · Qwen",
             "AC Qwen-plus":"AC · Qwen", "AC DeepSeek v4-pro":"AC · DeepSeek",
             "JB GPT-5.4":"JB · GPT-5.4 · full", "JB GPT-5.4-mini":"JB · mini · full",
             "JB GPT-5.4 | binary_only":"JB · GPT-5.4 · binary only", "JB GPT-5.4-mini | binary_only":"JB · mini · binary only"}
    fig, ax = plt.subplots(figsize=(10, 6.8))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(3), ["Release tasks", "Few predicted FAIL first", "Random tasks"])
    ax.set_yticks(range(len(names)), [short[n] for n in names])
    ax.tick_params(length=0)
    for i, name in enumerate(names):
        for j, order in enumerate(orders):
            count, total = counts[(name,order)]
            ax.text(j, i, f"{mat[i,j]:.1f}%\n{count}/{total}", ha="center", va="center",
                    fontsize=10, color="white" if mat[i,j]>55 else "#263640")
    cbar=fig.colorbar(im,ax=ax,shrink=.8,pad=.03);cbar.set_label("Tested comparisons with opposite ordering (%)")
    fig.suptitle("Criterion discovery vs certification: conditional tradeoffs", fontsize=14,y=.98)
    ax.set_title("Judge-first short circuit vs disagreement + random fallback",fontsize=10,pad=15)
    fig.text(.045,.06,"Denominator: interior budget points with equal actual queries; 1 canonical + 31 randomized orders. Counts are dependent comparisons.",fontsize=8.8)
    fig.text(.045,.034,"This is not a conflict probability or confidence interval. Nine cells come from two families; binary-only changes the JB scoring target.",fontsize=8.8)
    fig.subplots_adjust(left=.31,right=.95,top=.865,bottom=.14)
    for ext in ("png","pdf"):
        fig.savefig(OUT / f"conditional_tradeoffs.{ext}",dpi=200)
    plt.close(fig)
    payload={"curves":curve_data,"heatmap_counts":[dict(dataset=k[0],task_order=k[1],conflicts=v[0],active_comparisons=v[1]) for k,v in counts.items()],
             "source_sha256":{os.path.relpath(p, ROOT):hashlib.sha256(p.read_bytes()).hexdigest() for p in [REPORTS/"study.json",REPORTS/"budget_curves.csv.gz",Path(__file__)]}}
    (OUT/"plot_data.json").write_text(json.dumps(payload,indent=2))
    print(f"Saved 3 figures as PNG/PDF plus plotting values in {OUT}")


if __name__ == "__main__":
    plot()
