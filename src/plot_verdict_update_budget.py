"""Plot matched-query update effects from frozen rows; never rerun acquisition.

Run: python -m src.plot_verdict_update_budget
Defaults read frozen inputs and write only under artifacts/reproduced.
The CSV includes all seven policies; the figures select three named comparators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "reference-label-value-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts/expected/conjunction_policy_study/budget_curves.csv.gz"
DEFAULT_OUTPUT = ROOT / "artifacts/reproduced/figures/conjunction_policy_study"
KEYS = ["dataset", "task_order", "policy", "budget_share"]
DATASETS = ["DR Gemini 3.1 Pro", "DR Qwen-plus", "AC Qwen-plus", "AC DeepSeek v4-pro",
            "DR GPT-5.4 low", "JB GPT-5.4", "JB GPT-5.4-mini",
            "JB GPT-5.4 | binary_only", "JB GPT-5.4-mini | binary_only"]
ORDERS = ["release", "predicted_fail_count", "random"]
POLICIES = ["random_criterion", "disagreement_then_random", "sc_judge"]
ALL_POLICIES = POLICIES + ["sc_natural", "sc_random", "disagreement_only", "sc_disagreement"]
TITLES = ["Random criteria", "Disagreement + random", "Judge-first short circuit\n(expanded y scale)"]
COLORS = ["#1672A4", "#C86B12", "#202B38"]


def summarize_frame(frame: pd.DataFrame, seeds=tuple(range(1, 32))) -> pd.DataFrame:
    """Validate matched differences before averaging; exclude canonical seed zero."""
    f = frame.loc[frame.seed.ne(0)].copy()
    if f.empty or f.duplicated(KEYS + ["seed"]).any():
        raise ValueError("Empty randomized frame or duplicate paired row")
    if set(f.seed) != set(seeds):
        raise ValueError("Unexpected random seed set")
    grouped = f.groupby(KEYS, sort=True)
    if not grouped.seed.apply(lambda s: set(s) == set(seeds)).all():
        raise ValueError("Each budget group must contain every expected seed")
    for prefix in ("", "gated_"):
        if not f[prefix + "residual_errors"].eq(
                f[prefix + "residual_fp"] + f[prefix + "residual_ff"]).all():
            raise ValueError("Residual error components do not sum")
    f["delta_total"] = f.gated_residual_errors - f.residual_errors
    f["delta_fp"] = f.gated_residual_fp - f.residual_fp
    f["delta_ff"] = f.gated_residual_ff - f.residual_ff
    if not (f.delta_fp.eq(-f.introduced_fp).all()
            and f.delta_ff.eq(f.delayed_ff_corrections).all()
            and f.delta_total.eq(f.delta_fp + f.delta_ff).all()
            and f.introduced_fp.ge(0).all() and f.delayed_ff_corrections.ge(0).all()):
        raise ValueError("Matched-query decomposition is violated")
    if not ((f.actual_queries >= 0) & (f.actual_queries <= f.budget_cap)
            & (f.budget_cap <= f.n_criteria)).all():
        raise ValueError("Actual queries must lie within the available cap")
    grouped = f.groupby(KEYS, sort=True)
    constants = ["n_tasks", "n_criteria", "budget_cap"]
    if grouped[constants].nunique().ne(1).any().any():
        raise ValueError("Frame sizes or cap vary within a paired group")
    out = grouped[constants].first()
    out["seeds"] = grouped.size()
    for field in ["delta_total", "delta_fp", "delta_ff", "actual_queries"]:
        out[field + "_mean"] = grouped[field].mean()
        out[field + "_min"] = grouped[field].min()
        out[field + "_max"] = grouped[field].max()
    out["delta_total_q05"] = grouped.delta_total.quantile(.05)
    out["delta_total_q95"] = grouped.delta_total.quantile(.95)
    for name, mask in [("negative", f.delta_total.lt(0)), ("zero", f.delta_total.eq(0)),
                       ("positive", f.delta_total.gt(0))]:
        out["delta_total_" + name] = mask.groupby([f[k] for k in KEYS]).sum()
    return out.reset_index()


def validate_grid(summary: pd.DataFrame) -> None:
    expected = pd.MultiIndex.from_product(
        [DATASETS, ORDERS, ALL_POLICIES, np.arange(101) / 100], names=KEYS)
    actual = pd.MultiIndex.from_frame(summary[KEYS])
    if len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError("Missing or unexpected dataset/order/policy/budget grid")


def panel(ax, seq, is_sc=False, annotate=False):
    x = 100 * seq.budget_share.to_numpy()
    ax.fill_between(x, seq.delta_total_q05, seq.delta_total_q95,
                    color=COLORS[2], alpha=.14, linewidth=0,
                    label="Net: 5th–95th query-order percentiles")
    ax.plot(x, seq.delta_fp_mean, color=COLORS[0], lw=1.7, linestyle="--",
            label="FP change: prevented false passes")
    ax.plot(x, seq.delta_ff_mean, color=COLORS[1], lw=1.7, linestyle="-.",
            label="FF change: deferred repairs")
    ax.plot(x, seq.delta_total_mean, color=COLORS[2], lw=2,
            label="Net error change (mean)")
    ax.axhline(0, color="#87929D", lw=.7)
    ax.axvline(20, color="#87929D", lw=.7, linestyle=":")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#DEE3E7", lw=.55)
    ax.set_axisbelow(True)
    if is_sc:
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([-1, -.5, 0, .5, 1])
    if annotate:
        r = seq.loc[seq.budget_share.eq(.2)].iloc[0]
        ax.text(.96, .96, f"20% cap: net {r.delta_total_mean:+.2f}\n"
                f"{int(r.delta_total_negative)} improve / {int(r.delta_total_zero)} tie / "
                f"{int(r.delta_total_positive)} worsen", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5,
                bbox=dict(facecolor="white", alpha=.88, edgecolor="none", pad=3))


def decorate(fig, axes, title, subtitle, bottom=.20):
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [1, 2, 3, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.53, .078), fontsize=9)
    fig.suptitle(title, x=.53, y=.985, fontsize=15, fontweight="bold")
    fig.text(.53, .938, subtitle, ha="center", fontsize=10)
    fig.text(.055, .060, "Each point pairs the same queried criteria. Means over seeds 1–31; shaded percentiles are order variation, not confidence intervals.", fontsize=8.4)
    fig.text(.055, .037, "Negative net: fewer errors with gating. Positive net: more errors with gating. FP + FF = net for means; quantiles do not add.", fontsize=8.4)
    fig.text(.055, .014, "Budget is an available criterion-query cap; early stopping can leave it unused. Auxiliary judge inference is a separate cost. Fixed released references.", fontsize=8.4)
    fig.subplots_adjust(left=.10, right=.98, top=.865, bottom=bottom, wspace=.22, hspace=.31)


def source_record(path: Path) -> dict:
    """Identify a source without publishing a private absolute filesystem path."""
    resolved = path.resolve()
    try:
        label = resolved.relative_to(ROOT.resolve()).as_posix()
        scope = "repository-relative"
    except ValueError:
        label = resolved.name
        scope = "basename-only"
    return {"path": label, "path_scope": scope,
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def plot(source_csv: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict:
    """Render the full saved grid, protecting versioned inputs and figures."""
    from src.reproduce import validate_output
    output = validate_output(Path(output))
    source_csv = Path(source_csv)
    source_csv = (source_csv if source_csv.is_absolute() else ROOT / source_csv).resolve()
    raw = pd.read_csv(source_csv)
    summary = summarize_frame(raw)
    validate_grid(summary)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "verdict_update_budget_data.csv"
    summary.to_csv(csv_path, index=False, float_format="%.12g")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#263640", "axes.labelcolor": "#263640",
                         "axes.edgecolor": "#A5ABB0", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    selected = ["JB GPT-5.4", "JB GPT-5.4 | binary_only"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), sharex=True)
    for i, dataset in enumerate(selected):
        for j, policy in enumerate(POLICIES):
            seq = summary.loc[(summary.dataset == dataset) & (summary.task_order == "release")
                              & (summary.policy == policy)].sort_values("budget_share")
            panel(axes[i, j], seq, is_sc=j == 2, annotate=True)
            if j != 2:
                axes[i, j].set_ylim(-40, 140)
                axes[i, j].set_yticks([-40, 0, 40, 80, 120])
            if i == 0:
                axes[i, j].set_title(TITLES[j], fontsize=11, pad=13)
            else:
                axes[i, j].set_xlabel("Available criterion-query cap (%)")
        axes[i, 0].set_ylabel(("Full strict target\n" if i == 0 else "Binary-only target\n")
                              + "Gated − eager errors (annotations)", labelpad=9)
    decorate(fig, axes, "When does waiting for a certificate change verdict errors?",
             "JudgmentBench / GPT-5.4  •  same 1,539 annotations, changed scoring target  •  release task order")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"verdict_update_budget.{ext}", dpi=180)
    plt.close(fig)

    # All nine cells and all three task orders: no selected-budget-only appendix.
    with PdfPages(output / "verdict_update_budget_all_cells.pdf") as pdf:
        for dataset in DATASETS:
            fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
            block = summary.loc[(summary.dataset == dataset) & summary.policy.isin(POLICIES[:2])]
            lower = min(-1, float(block.delta_fp_min.min()), float(block.delta_total_min.min()))
            upper = max(1, float(block.delta_ff_max.max()), float(block.delta_total_max.max()))
            pad = .08 * (upper - lower)
            for i, order in enumerate(ORDERS):
                for j, policy in enumerate(POLICIES):
                    seq = summary.loc[(summary.dataset == dataset) & (summary.task_order == order)
                                      & (summary.policy == policy)].sort_values("budget_share")
                    panel(axes[i, j], seq, is_sc=j == 2)
                    if j != 2:
                        axes[i, j].set_ylim(lower-pad, upper+pad)
                    if i == 0:
                        axes[i, j].set_title(TITLES[j], fontsize=11, pad=13)
                    if i == 2:
                        axes[i, j].set_xlabel("Available criterion-query cap (%)")
                order_name = ["Release order", "Few predicted FAIL first", "Random task order"][i]
                axes[i, 0].set_ylabel(order_name + "\nGated − eager errors (units)")
            meta = summary.loc[summary.dataset.eq(dataset)].iloc[0]
            decorate(fig, axes, f"Update effects across all budgets — {dataset}",
                     f"N = {int(meta.n_tasks):,} analysis units; M = {int(meta.n_criteria):,} criteria  •  "
                     "shared scale for first two columns within this page; scales vary by page", bottom=.18)
            pdf.savefig(fig)
            plt.close(fig)

    canonical = raw.loc[raw.seed.eq(0) & raw.task_order.eq("release")
                        & raw.budget_share.eq(.2) & raw.policy.eq("disagreement_then_random")
                        & raw.dataset.isin(selected)]
    paths = [csv_path, output / "verdict_update_budget.pdf",
             output / "verdict_update_budget_all_cells.pdf", output / "verdict_update_budget.png"]
    receipt = {
        "kind": "saved-result plotting, not a new scientific run",
        "randomized_rows_checked": int(raw.seed.ne(0).sum()), "groups": len(summary),
        "seeds": list(range(1, 32)), "all_policies_in_csv": ALL_POLICIES,
        "policies_in_figures": POLICIES, "budget_caps": "0..100 percent, step 1 percent",
        "summary": "arithmetic means of matched within-seed differences",
        "identity": "delta_total = delta_fp + delta_ff = -introduced_fp + delayed_ff_corrections",
        "band": "5th–95th percentiles across 31 randomized orders; not a CI; need not contain mean",
        "limits": "SC expanded scale; same-source target sensitivity is not an independent dataset; task orders need not produce independent curves",
        "canonical_20_percent": canonical.to_dict(orient="records"),
        "randomized_20_percent": summary.loc[summary.dataset.isin(selected)
            & summary.task_order.eq("release") & summary.budget_share.eq(.2)
            & summary.policy.isin(POLICIES)].to_dict(orient="records"),
        "source": source_record(source_csv), "renderer": source_record(Path(__file__)),
        "generated_artifacts": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                         "bytes": p.stat().st_size} for p in paths},
    }
    (output / "verdict_update_budget.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Validated {receipt['randomized_rows_checked']:,} rows / {len(summary):,} groups; saved main figure, 9-page supplement, CSV and receipt.")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Full saved policy-grid CSV, optionally gzip-compressed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Unversioned output directory; published inputs/figures are protected")
    args = parser.parse_args()
    try:
        plot(args.source, args.output)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
