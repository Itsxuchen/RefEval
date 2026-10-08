"""Render two retrospective mechanism figures from saved, checked analysis tables.

This script does not rerun policies or fit a model. The update curves condition on
the complete fixed reference. Allocation components are arithmetic means of 31
paired seed differences, not differences between marginal medians.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/refeval-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts/expected/conjunction_mechanism"
FIGURES = ROOT / "artifacts/figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
DARK = "#243746"
GRAY = "#7B8790"
FIELDS = [
    "certified_tasks_delta", "certified_initial_correct_delta",
    "certified_initial_wrong_delta", "residual_errors_delta",
    "noncertified_ff_repair_delta", "introduced_fp_delta",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_sources():
    names = ["update_expected_grid.csv", "update_structures.json", "update_manifest.json",
             "allocation_summary.csv", "allocation_rows.csv.gz", "allocation_manifest.json"]
    hashes = {os.path.relpath(REPORTS / name, ROOT): sha256(REPORTS / name) for name in names}
    update = pd.read_csv(REPORTS / names[0])
    structures = json.loads((REPORTS / names[1]).read_text())
    um = json.loads((REPORTS / names[2]).read_text())
    summary = pd.read_csv(REPORTS / names[3])
    rows = pd.read_csv(REPORTS / names[4])
    am = json.loads((REPORTS / names[5]).read_text())
    for name in ("update_expected_grid.csv", "update_structures.json"):
        require(sha256(REPORTS / name) == um["output_sha256"][name], f"Changed frozen input: {name}")
    require(len(update) == um["analytic_rows"] == 1818, "Expected 9 x 2 x 101 update rows")
    require(len(summary) == am["summary_rows"] == 162, "Expected 9 x 3 x 6 allocation summaries")
    require(len(rows) == am["paired_rows"] == 5022, "Expected 162 x 31 paired allocation rows")
    require(um["seeds_compared"] == list(range(1, 32)) == am["config"]["seeds"], "Seeds must be 1..31")
    require(am["config"]["difference_direction"] == "mix minus SRS", "Unexpected allocation sign")
    require(set(um["policies"]) == {"random_criterion", "disagreement_then_random"}, "Update policy scope changed")
    require(set(am["config"]["policies"]) == {"random_criterion", "sc50_then_srs"}, "Allocation policy scope changed")
    require(set(update.dataset) == set(summary.dataset) == set(structures) and len(structures) == 9,
            "Dataset sets differ")
    require(set(update.task_order) == {"release"}, "Update requires release task order")
    require(not update.duplicated(["dataset", "policy", "budget_share"]).any(), "Duplicate update rows")
    require(not summary.duplicated(["dataset", "task_order", "budget_share"]).any(), "Duplicate summaries")
    require(not rows.duplicated(["dataset", "task_order", "budget_share", "seed"]).any(), "Duplicate paired rows")
    require(np.isfinite(update.select_dtypes(include="number").to_numpy()).all(), "Nonfinite update inputs")
    require((update.randomized_orders == 31).all() and (summary.seeds == 31).all(), "Order counts differ")
    for _, group in update.groupby(["dataset", "policy"]):
        require(np.allclose(np.sort(group.budget_share), np.arange(101) / 100), "Incomplete budget grid")
    for field in ["expected_introduced_fp", "expected_delayed_ff"]:
        require((update[field] >= -1e-12).all(), f"Negative probability-sum component: {field}")
    update_identity = float(np.max(np.abs(update.expected_delta - update.expected_delayed_ff + update.expected_introduced_fp)))
    observed_identity = float(np.max(np.abs(update.observed_mean_delta - update.observed_mean_delayed_ff + update.observed_mean_introduced_fp)))
    require(max(update_identity, observed_identity) < 1e-10, "Update additive identity failed")
    keys = ["dataset", "task_order", "budget_share"]
    for _, group in rows.groupby(keys):
        require(sorted(group.seed.tolist()) == list(range(1, 32)), "A paired cell does not have seeds 1..31")
    require(np.isfinite(rows[FIELDS].to_numpy()).all(), "Nonfinite paired allocation counts")
    aggregated = rows.groupby(keys)[FIELDS].mean()
    reported = summary.set_index(keys).sort_index()
    max_mean_error = 0.0
    for field in FIELDS:
        error = float(np.max(np.abs(aggregated[field] - reported[field + "_mean"])))
        max_mean_error = max(max_mean_error, error)
    require(max_mean_error < 1e-10, "Saved allocation means differ from paired rows")
    cert_identity = float(np.max(np.abs(rows.certified_tasks_delta - rows.certified_initial_correct_delta - rows.certified_initial_wrong_delta)))
    residual_identity = float(np.max(np.abs(rows.residual_errors_delta + rows.certified_initial_wrong_delta + rows.noncertified_ff_repair_delta - rows.introduced_fp_delta)))
    require(cert_identity == residual_identity == 0, "Allocation row decomposition failed")
    for dataset, group in rows.groupby("dataset"):
        require(set(group.n_tasks) == {structures[dataset]["n_units"]}, "Unit-count mismatch")
        require(set(group.n_criteria) == {structures[dataset]["n_criteria"]}, "Criterion-count mismatch")
    checks = {"update_rows": len(update), "allocation_summary_rows": len(summary),
              "allocation_paired_rows": len(rows), "seed0_excluded": True,
              "update_identity_max_abs_error": update_identity,
              "update_observed_identity_max_abs_error": observed_identity,
              "allocation_mean_recomputation_max_abs_error": max_mean_error,
              "allocation_certification_row_identity_max_abs_error": cert_identity,
              "allocation_residual_row_identity_max_abs_error": residual_identity}
    return update, structures, summary, hashes, checks


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BAC2C8")
    ax.tick_params(color="#BAC2C8", labelcolor=DARK)
    ax.grid(axis="y", color="#E4E8EB", linewidth=.65)
    ax.set_axisbelow(True)
    ax.axhline(0, color=GRAY, linewidth=.8)


def save_figure(fig, stem):
    paths = [FIGURES / f"{stem}.{extension}" for extension in ("png", "pdf")]
    fig.savefig(paths[0], dpi=200, facecolor="white")
    fig.savefig(paths[1], facecolor="white", metadata={"Creator": "RefEval saved-table renderer"})
    plt.close(fig)
    return paths


def update_figure(update, structures):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), sharey=True)
    fig.subplots_adjust(left=.078, right=.985, top=.78, bottom=.245, wspace=.14)
    fig.suptitle("Why verdict gating can help or hurt at the same query budget", x=.078,
                 ha="left", y=.975, fontsize=16, fontweight="bold", color=DARK)
    fig.text(.078, .914, "Full-reference expectation  |  disagreement first, random continuation  |  release order",
             fontsize=10.5, color=DARK)
    focal = []
    for ax, dataset, title in zip(axes, ["JB GPT-5.4", "JB GPT-5.4 | binary_only"],
                                 ["(a) JudgmentBench · GPT-5.4 · full", "(b) JudgmentBench · GPT-5.4 · binary only"]):
        d = update.loc[(update.dataset == dataset) & (update.policy == "disagreement_then_random")].sort_values("budget_share")
        s = structures[dataset]
        x = 100 * d.budget_share.to_numpy()
        style_axes(ax)
        ax.plot(x, d.expected_delayed_ff, color=ORANGE, linestyle=(0, (5, 2)), linewidth=1.9,
                label="E[delayed false fails] (D)")
        ax.plot(x, -d.expected_introduced_fp, color=BLUE, linestyle=(0, (2, 1)), linewidth=1.9,
                label="−E[introduced false passes] (−A)")
        ax.plot(x, d.expected_delta, color=DARK, linewidth=2.3, label="E[D − A]: gated − eager errors")
        ax.plot(x[::5], d.observed_mean_delta.to_numpy()[::5], linestyle="none", marker="o",
                markersize=4.4, markerfacecolor="white", markeredgecolor=DARK, markeredgewidth=1,
                label="Observed mean over 31 orders")
        for share in [.20, .84]:
            row = d.loc[np.isclose(d.budget_share, share)].iloc[0]
            ax.axvline(100 * share, color=GRAY, linestyle=(0, (2, 3)), linewidth=.8, zorder=0)
            ax.text(100 * share, 1.015, f"{share:.0%}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color=GRAY)
            focal.append({"dataset": dataset, "budget_share": share,
                          "expected_delayed_ff": float(row.expected_delayed_ff),
                          "expected_introduced_fp": float(row.expected_introduced_fp),
                          "expected_delta": float(row.expected_delta),
                          "observed_mean_delta": float(row.observed_mean_delta)})
        selected = focal[-2:]
        annotation = "\n".join(f"{r['budget_share']:.0%} net: {r['expected_delta']:+.2f}  (observed {r['observed_mean_delta']:+.2f})" for r in selected)
        ax.text(.025, .965, annotation, transform=ax.transAxes, va="top", fontsize=9.2,
                linespacing=1.7, color=DARK, bbox={"facecolor":"white", "alpha":.94, "edgecolor":"none", "pad":3})
        ax.set_title(f"{title}\nN = {s['n_units']:,} annotations; M = {s['n_criteria']:,} criteria",
                     loc="left", fontsize=10.5, pad=35, color=DARK, linespacing=1.6)
        ax.set(xlim=(0, 100), xticks=np.arange(0, 101, 20), xlabel="Query budget cap (% of criterion labels)")
    axes[0].set_ylabel("Gated − eager error count (annotations)")
    axes[0].set_ylim(-35, 137)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.07, .102), ncol=2,
               frameon=False, fontsize=9.8, columnspacing=2.4, handlelength=3)
    fig.text(.078, .044, "Positive net = more errors under gating. Expectations condition on all released reference labels; they are retrospective, not deployable forecasts.",
             fontsize=8.8, color=DARK)
    fig.text(.078, .018, "Points average seeds 1–31 (seed 0 excluded), every 5 percentage points. These order means do not quantify new-task uncertainty.",
             fontsize=8.8, color=DARK)
    return save_figure(fig, "conjunction_update_mechanism"), focal


def component_bars(ax, values, labels, colors):
    style_axes(ax)
    bars = ax.bar(np.arange(len(values)), values, color=colors, width=.61, edgecolor="white", linewidth=.7)
    ax.set_xticks(np.arange(len(values)), labels, fontsize=9.2, linespacing=1.4)
    low, high = min(0, min(values)), max(0, max(values))
    span = max(high - low, .5)
    ax.set_ylim(low - .18 * span, high + .23 * span)
    ax.set_xlim(-.6, len(values) - .4)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:+.2f}" if value else "0.00",
                    (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 5 if value >= 0 else -5),
                    textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top",
                    fontsize=10, color=DARK, fontweight="bold")
    ax.axvline(len(values) - 1.5, color="#CBD2D7", linestyle=(0, (2, 3)), linewidth=.8)


def allocation_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    fig.subplots_adjust(left=.078, right=.982, top=.80, bottom=.14, wspace=.24, hspace=.62)
    fig.suptitle("More certified verdicts need not mean fewer verdict errors", x=.078,
                 ha="left", y=.975, fontsize=16, fontweight="bold", color=DARK)
    fig.text(.078, .923, "20% budget · release order · Fixed 50/50 allocation minus random criterion sampling",
             fontsize=10.5, color=DARK)
    fig.text(.078, .892, "Each bar is the mean of 31 paired seed differences at equal actual query counts. Additive counts; column scales differ.",
             fontsize=10, color=DARK)
    focal = []
    for column, dataset in enumerate(["DR Gemini 3.1 Pro", "JB GPT-5.4"]):
        row = summary.loc[(summary.dataset == dataset) & (summary.task_order == "release") & np.isclose(summary.budget_share, .2)].iloc[0]
        unit = "tasks" if column == 0 else "annotations"
        title = "RuVerBench DR · Gemini 3.1 Pro" if column == 0 else "JudgmentBench · GPT-5.4 · full"
        upper = [float(row.certified_initial_correct_delta_mean), float(row.certified_initial_wrong_delta_mean), float(row.certified_tasks_delta_mean)]
        lower = [-float(row.certified_initial_wrong_delta_mean), -float(row.noncertified_ff_repair_delta_mean), float(row.introduced_fp_delta_mean), float(row.residual_errors_delta_mean)]
        component_bars(axes[0, column], upper,
                       ["Initially correct\ncertifications", "Initially wrong\ncertifications", "Net certification\ngain"],
                       [BLUE, ORANGE, DARK])
        axes[0, column].set_title(f"({'a' if column == 0 else 'b'}) {title}\nN = {int(row.n_tasks):,} {unit}; {int(row.actual_queries):,} queried criteria",
                                  loc="left", fontsize=10.8, pad=20, color=DARK, linespacing=1.6)
        axes[0, column].set_ylabel(f"Certification change ({unit})")
        component_bars(axes[1, column], lower,
                       ["−Δ certified\ninitially wrong", "−Δ uncertified\nfalse-fail repairs", "+Δ introduced\nfalse passes", "Net residual\nerror change"],
                       [BLUE if v < 0 else ORANGE for v in lower[:-1]] + [DARK])
        axes[1, column].set_title(f"({'c' if column == 0 else 'd'}) Eager-update error decomposition", loc="left", fontsize=10.8, pad=13, color=DARK)
        axes[1, column].set_ylabel(f"Residual error change ({unit})")
        focal.append({"dataset": dataset, "budget_share": .2, "actual_queries": int(row.actual_queries),
                      "n_units": int(row.n_tasks), "unit": unit,
                      "certification_components_and_total": upper, "residual_components_and_total": lower})
    fig.text(.078, .047, "Negative residual change = fewer errors. Initial correctness, false passes and false fails are defined against the fixed released reference.",
             fontsize=9, color=DARK)
    fig.text(.078, .019, "Components add before rounding. These are paired arithmetic means (seeds 1–31), not marginal medians or population effect estimates.",
             fontsize=9, color=DARK)
    return save_figure(fig, "conjunction_allocation_mechanism"), focal


def main(reports=None, figures=None):
    global REPORTS, FIGURES
    if reports is not None: REPORTS = Path(reports).resolve()
    if figures is not None: FIGURES = Path(figures).resolve()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": DARK, "text.color": DARK,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    update, structures, summary, source_hashes, checks = check_sources()
    FIGURES.mkdir(parents=True, exist_ok=True)
    update_paths, update_focal = update_figure(update, structures)
    allocation_paths, allocation_focal = allocation_figure(summary)
    require(all(sha256(ROOT / path) == digest for path, digest in source_hashes.items()), "An input changed while rendering")
    manifest = {
        "renderer": str(Path(__file__).resolve().relative_to(ROOT)),
        "renderer_sha256": sha256(Path(__file__)), "source_sha256": source_hashes,
        "validation": checks,
        "update_scope": "Two JB GPT-5.4 cells, disagreement_then_random, release order, all 101 budget caps. Exact oracle full-reference expectation and descriptive 31-order means; no confidence intervals or deployment forecast.",
        "allocation_scope": "DR Gemini and full JB GPT-5.4, 20% cap, release order, sc50_then_srs minus random_criterion at equal actual queries. Means of seed-paired differences; not marginal-median differences or population estimates.",
        "matched_marginal_null_panels": "Not plotted; saved sensitivity scenarios remain in update_permutation_summary.csv.",
        "update_focal_values": update_focal, "allocation_focal_values": allocation_focal,
        "output_sha256": {os.path.relpath(path, ROOT): sha256(path) for path in update_paths + allocation_paths},
    }
    manifest_path = FIGURES / "conjunction_mechanism_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"outputs": list(manifest["output_sha256"]), "manifest": os.path.relpath(manifest_path, ROOT), "validation": checks}, indent=2))


if __name__ == "__main__":
    main()
