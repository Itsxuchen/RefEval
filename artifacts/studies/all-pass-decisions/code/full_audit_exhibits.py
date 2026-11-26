"""Rebuild complete-audit and early-release contrasts from weighted cluster counts."""
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FRAMES = {
    "JudgmentBench / GPT-5.4": (30, 206),
    "SummEval / UniEval": (70, 1120),
    "SummEval / G-Eval GPT-4": (70, 1120),
}
DRAWS = 1999
SEED = 20261125
EVENTS = ("early", "auditor_pass", "auditor_pass_judge_fail", "early_auditor_pass",
          "early_auditor_fail", "auditor_fail_judge_pass")


def percent(num, den):
    return np.divide(100 * num, den, out=np.full(np.shape(den), np.nan), where=den > 0)


def measures(c):
    """Accept paired totals for the original frame and for every resampled frame."""
    n = c["outputs"]
    values = {key + "_peer_pass_pct": percent(c[key + "_peer_pass"], c[key + "_n"]) for key in EVENTS}
    values["early_minus_full_judge_fail_pp"] = (values["early_peer_pass_pct"]
                                                - values["auditor_pass_judge_fail_peer_pass_pct"])
    values["final_verdict_kept_among_early_pct"] = percent(c["early_auditor_pass_n"], c["early_n"])
    for name in ("judge_pass", "expert_pass", "early_n", "judge_fp", "judge_ff", "full_fp", "full_ff"):
        values[name + "_per100"] = percent(c[name], n)
    for policy in ("judge", "full"):
        values[policy + "_error_per100"] = percent(c[policy + "_fp"] + c[policy + "_ff"], n)
    values["full_minus_judge_per100"] = values["full_error_per100"] - values["judge_error_per100"]
    values["full_minus_always_fail_per100"] = values["full_error_per100"] - values["expert_pass_per100"]

    # Peer support means a PASS for upward overrides and a FAIL for downward overrides.
    down = c["auditor_fail_judge_pass_n"]
    changed = c["auditor_pass_judge_fail_n"] + down
    supported = c["auditor_pass_judge_fail_peer_pass"] + down - c["auditor_fail_judge_pass_peer_pass"]
    values["complete_change_rate_per100"] = percent(changed, n)
    values["peer_endorses_completed_changes_pct"] = percent(supported, changed)
    np.testing.assert_allclose(percent(changed - 2 * supported, n), values["full_minus_judge_per100"], atol=1e-10)

    # Completing an early PASS changes this same decision only when its auditor ultimately says FAIL.
    timing = c["early_auditor_fail_n"] - 2 * c["early_auditor_fail_peer_pass"]
    values["early_minus_completed_on_same_cases_per100_outputs"] = percent(timing, n)
    values["early_minus_completed_per100_early_approvals"] = percent(timing, c["early_n"])
    return values


def compute_estimates():
    with (ROOT / "inputs/full_audit_cluster_counts.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        numeric = [k for k in reader.fieldnames if k not in ("frame", "cluster")]
        groups = {frame: {} for frame in FRAMES}
        for row in reader:
            clusters = groups[row["frame"]]
            i = int(row["cluster"])
            if i in clusters:
                raise ValueError(f"Duplicate cluster {row['frame']}, {i}")
            clusters[i] = [float(row[k]) for k in numeric]

    estimates = []
    for frame, (n_groups, n_outputs) in FRAMES.items():
        clusters = groups[frame]
        if sorted(clusters) != list(range(1, n_groups + 1)):
            raise ValueError(f"Unexpected cluster identities in {frame}")
        counts = np.array([clusters[i] for i in sorted(clusters)])
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError(f"Invalid counts in {frame}")
        totals = counts.sum(axis=0)
        if totals[numeric.index("outputs")] != n_outputs:
            raise ValueError(f"Unexpected population size in {frame}")
        draws = np.random.default_rng(SEED).multinomial(n_groups, np.ones(n_groups) / n_groups, size=DRAWS)
        # A shared draw gives each term in a contrast the same article or task weights.
        c = dict(zip(numeric, np.vstack([totals, draws @ counts]).T))
        np.testing.assert_allclose(c["full_fp"], c["full_ff"], atol=1e-10)
        np.testing.assert_allclose(c["early_n"], c["early_auditor_pass_n"] + c["early_auditor_fail_n"], atol=1e-10)
        for metric, values in measures(c).items():
            draws_valid = values[1:][np.isfinite(values[1:])]
            lo, hi = np.percentile(draws_valid, [2.5, 97.5]) if len(draws_valid) else (None, None)
            estimates.append(dict(frame=frame, metric=metric,
                                  estimate=float(values[0]) if np.isfinite(values[0]) else None,
                                  lower95=lo, upper95=hi, defined_draws=len(draws_valid), total_draws=DRAWS))
    return estimates


def signed(value):
    return f"{value:+.2f}" if abs(value) >= 0.005 else "0.00"


def interval(row):
    return "$" + signed(row["estimate"]) + r"\;[" + signed(row["lower95"]) + "," + signed(row["upper95"]) + "]$"


def write_tables(rows, path):
    by_key = {(r["frame"], r["metric"]): r for r in rows}
    with (path.parent / "crossover_curves.csv").open(newline="") as stream:
        stages = {(r["frame"], r["judge"], int(r["budget_pct"])): r for r in csv.DictReader(stream)}
    full, early, stage = [], [], []
    for frame in FRAMES:
        def point(metric, precision):
            return f"{by_key[frame, metric]['estimate']:.{precision}f}"

        label = frame.replace("G-Eval GPT-4", "G-Eval")
        dataset, judge = frame.split(" / ")
        panel = ("held70" if dataset == "SummEval" else dataset, judge)
        half = stages[*panel, 50]
        # The zero and complete budgets have common verdicts under both update rules.
        for budget, metric in ((0, "judge_error_per100"), (100, "full_error_per100")):
            for rule in ("eager", "gated"):
                np.testing.assert_allclose(float(stages[*panel, budget][f"peer_{rule}_error_per100"]),
                                           by_key[frame, metric]["estimate"], atol=1e-10)
        stage.append(" & ".join([label, point("judge_error_per100", 2),
                                 f"{float(half['peer_eager_error_per100']):.2f}",
                                 f"{float(half['peer_gated_error_per100']):.2f}",
                                 point("full_error_per100", 2)]) + r" \\")
        full.append(" & ".join([label, point("judge_pass_per100", 1), point("expert_pass_per100", 1),
                                point("judge_error_per100", 2), point("full_error_per100", 2),
                                point("peer_endorses_completed_changes_pct", 1),
                                interval(by_key[frame, "full_minus_judge_per100"])]) + r" \\")
        early.append(" & ".join([label, point("early_peer_pass_pct", 1),
                                 point("auditor_pass_judge_fail_peer_pass_pct", 1),
                                 interval(by_key[frame, "early_minus_full_judge_fail_pp"]),
                                 interval(by_key[frame, "early_minus_completed_on_same_cases_per100_outputs"])]) + r" \\")
    path.write_text("\n".join([r"\newcommand{\fullauditrows}{%", *full, "}",
                               r"\newcommand{\earlyauditrows}{%", *early, "}",
                               r"\newcommand{\stageauditrows}{%", *stage, "}", ""]))


if __name__ == "__main__":
    rows = compute_estimates()
    out = ROOT / "generated"
    out.mkdir(exist_ok=True)
    with (out / "full_audit_estimates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_tables(rows, out / "full_audit.tex")
