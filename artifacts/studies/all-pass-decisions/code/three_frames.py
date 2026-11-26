"""Rebuild the four manuscript rows with a common denominator and scoring convention."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WMT = "WMT23 / Sonnet"
PANELS = {
    WMT: ("WMT23 / Sonnet", 18, 739),
    "JudgmentBench / GPT-5.4": ("Legal / GPT-5.4", 30, 206),
    "SummEval / UniEval": ("SummEval / UniEval", 70, 1120),
    "SummEval / G-Eval GPT-4": ("SummEval / G-Eval", 70, 1120),
}
WMT_DIR = ROOT / "wmt/artifacts/reports/wmt_blind_crossover_20261126/corrected"
DRAWS, SEED = 1999, 20261125


def wmt_counts(row):
    return dict(outputs=row["N"], up=row["up"], up_yes=row["up_agree"],
                down=row["down"], down_yes=row["down_agree"],
                early=row["early"], withdrawn=row["withdrawn"],
                withdrawn_peer_pass=row["withdrawn_agree"], judge_pass=row["judge_pass"],
                expert_pass=row["auditor_pass"], judge_err=row["retain_fp"] + row["retain_ff"],
                full_err=row["full_fp"] + row["full_ff"])


def earlier_counts(row):
    down = row["auditor_fail_judge_pass_n"]
    return dict(outputs=row["outputs"], up=row["auditor_pass_judge_fail_n"],
                up_yes=row["auditor_pass_judge_fail_peer_pass"], down=down,
                down_yes=down - row["auditor_fail_judge_pass_peer_pass"],
                early=row["early_n"], withdrawn=row["early_auditor_fail_n"],
                withdrawn_peer_pass=row["early_auditor_fail_peer_pass"], judge_pass=row["judge_pass"],
                expert_pass=row["expert_pass"], judge_err=row["judge_fp"] + row["judge_ff"],
                full_err=row["full_fp"] + row["full_ff"])


def cluster_counts(wmt_dir):
    panels = {frame: {} for frame in PANELS}
    data = json.loads((wmt_dir / "document_totals.json").read_text())
    panels[WMT] = {cluster: wmt_counts(row) for cluster, row in data.items()}
    with (ROOT / "inputs/full_audit_cluster_counts.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            frame = row.pop("frame")
            cluster = int(row.pop("cluster"))
            if cluster in panels[frame]:
                raise ValueError(f"Duplicate cluster in {frame}: {cluster}")
            panels[frame][cluster] = earlier_counts({k: float(v) for k, v in row.items()})
    return panels


def score(c):
    """Give every ratio its own matching population, including inside each bootstrap draw."""
    n = c["outputs"]
    up, down, early = c["up"], c["down"], c["early"]
    adopt = c["full_err"] - c["judge_err"]
    same = c["withdrawn"] - 2 * c["withdrawn_peer_pass"]
    # A change earns minus one error for a peer endorsement and plus one otherwise.
    np.testing.assert_allclose(adopt, up + down - 2 * (c["up_yes"] + c["down_yes"]), atol=1e-8)
    np.testing.assert_allclose(up - down, c["expert_pass"] - c["judge_pass"], atol=1e-8)
    return {
        "up_per100_outputs": (up, n), "up_peer_endorses_pct": (c["up_yes"], up),
        "down_per100_outputs": (down, n), "down_peer_endorses_pct": (c["down_yes"], down),
        "all_changes_peer_endorses_pct": (c["up_yes"] + c["down_yes"], up + down),
        "early_per100_outputs": (early, n),
        "withdrawn_per100_early": (c["withdrawn"], early),
        "retain_disagreements_per100_outputs": (c["judge_err"], n),
        "complete_disagreements_per100_outputs": (c["full_err"], n),
        "complete_minus_retain_per100_outputs": (adopt, n),
        "early_minus_eventual_per100_outputs": (same, n),
        "early_minus_eventual_per100_early": (same, early),
    }


def summarize(frame, clusters):
    _, k, n = PANELS[frame]
    if len(clusters) != k:
        raise ValueError(f"Unexpected number of clusters in {frame}")
    ordered = [clusters[i] for i in sorted(clusters)]
    keys = tuple(ordered[0])
    counts = np.array([[row[key] for key in keys] for row in ordered])
    if not np.isfinite(counts).all() or np.any(counts < -1e-10):
        raise ValueError(f"Invalid counts in {frame}")
    # Check both signed identities in each actual cluster prior to resampling.
    score(dict(zip(keys, counts.T)))
    weights = np.random.default_rng(SEED).multinomial(k, np.full(k, 1 / k), size=DRAWS)
    c = dict(zip(keys, np.vstack([counts.sum(axis=0), weights @ counts]).T))
    if c["outputs"][0] != n:
        raise ValueError(f"Unexpected number of outputs in {frame}")
    estimates = {}
    for name, (numerator, denominator) in score(c).items():
        observed = denominator[1:] > 0
        reps = 100 * numerator[1:][observed] / denominator[1:][observed]
        lo, hi = np.percentile(reps, [2.5, 97.5])
        estimates[name] = dict(frame=frame, metric=name, outputs=n, clusters=k,
                               numerator_eq=float(numerator[0]), denominator_eq=float(denominator[0]),
                               estimate=float(100 * numerator[0] / denominator[0]),
                               lower95=float(lo), upper95=float(hi), defined_draws=len(reps), total_draws=DRAWS)
    return estimates


def signed(value):
    return f"{value:+.2f}" if abs(value) >= .005 else "0.00"


def interval(row):
    return f"${signed(row['estimate'])}\\;[{signed(row['lower95'])},{signed(row['upper95'])}]$"


def write(wmt_dir, out):
    clusters = cluster_counts(wmt_dir)
    rows = {frame: summarize(frame, clusters[frame]) for frame in PANELS}
    # Check the shared estimands against the locally sealed WMT analysis.
    source = json.loads((wmt_dir / "held.json").read_text())["estimates"]
    for metric, source_metric in {
            "up_peer_endorses_pct": "up_endorsement_pct", "down_peer_endorses_pct": "down_endorsement_pct",
            "early_per100_outputs": "early_per100", "complete_minus_retain_per100_outputs": "full_minus_retain_1to1_per100",
            "early_minus_eventual_per100_outputs": "same_early_minus_eventual_1to1_per100"}.items():
        local, ref = rows[WMT][metric], source[source_metric]
        np.testing.assert_allclose([local[k] for k in ("estimate", "lower95", "upper95")],
                                   [ref["point"], *ref["ci95"]], atol=1e-10, rtol=0)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "three_frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(rows[WMT].values()))), lineterminator="\n")
        writer.writeheader()
        for panel in rows.values():
            for row in panel.values():
                # Serialize at a common precision across BLAS implementations.
                writer.writerow({k: f"{v:.9f}" if isinstance(v, float) else v for k, v in row.items()})
    lines = [r"\newcommand{\peerframerows}{%"]
    for frame, metrics in rows.items():
        cells = [PANELS[frame][0],
                 f"{metrics['up_peer_endorses_pct']['estimate']:.1f}",
                 f"{metrics['down_peer_endorses_pct']['estimate']:.1f}",
                 f"{metrics['early_per100_outputs']['estimate']:.2f}",
                 interval(metrics["complete_minus_retain_per100_outputs"]),
                 interval(metrics["early_minus_eventual_per100_outputs"])]
        lines.append(" & ".join(cells) + r" \\")
    (out / "three_frames.tex").write_text("\n".join([*lines, "}", ""]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wmt-dir", type=Path, default=WMT_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    write(args.wmt_dir, args.out)
