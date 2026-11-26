"""Compare early and complete expert overrides on the published crossover frames."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import judgmentbench_cross_scorer_comparison as legal
from src import summeval_cross_scorer as news
from src.conjunction_policy_study import make_order
from src.external_summeval_frame import DEFAULT_CACHE, MODELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reports/full_audit_endorsement_20261126"
DRAWS = 1999
SEED = 20261125


def output_counts(judge, truth, early_frequency):
    """One output receives weight one, then each auditor and each peer equally."""
    truth = np.asarray(truth, dtype=bool)
    early = np.asarray(early_frequency, dtype=float)
    if len(truth) < 2 or early.shape != truth.shape or np.any((early < 0) | (early > 1)):
        raise ValueError("Expected one early-approval probability per auditor")
    if judge and np.any(early):
        raise ValueError("An initially passing output cannot be an eager-only approval")
    peer = (truth.sum() - truth.astype(int)) / (len(truth) - 1)
    counts = dict(outputs=1, judge_pass=int(judge), expert_pass=truth.mean(),
                  early_unique=int(np.any(early)), judge_fp=judge * (1-peer).mean(),
                  judge_ff=(not judge) * peer.mean(), full_fp=np.mean(truth * (1-peer)),
                  full_ff=np.mean(~truth * peer))
    events = {"early": early, "auditor_pass": truth, "auditor_pass_judge_fail": truth * (not judge),
              "early_auditor_pass": early * truth, "early_auditor_fail": early * ~truth,
              "auditor_fail_judge_pass": ~truth * judge}
    for name, weight in events.items():
        counts[name + "_n"] = np.mean(weight)
        counts[name + "_peer_pass"] = np.mean(weight * peer)
    return counts


def make_groups(rows, model_index):
    # Keep the original output and criterion order when generating the shared masks.
    lengths = [len(judges[0]) for _, judges, _ in rows]
    judges = np.concatenate([j for _, j, _ in rows], axis=1)
    layout = dict(task_ids=np.arange(len(rows)), task_index=np.repeat(np.arange(len(rows)), lengths),
                  prediction=judges[0], secondary_predictions=judges[1:])
    masks = np.zeros((31, sum(lengths)), dtype=bool)
    cap = masks.shape[1] // 2
    for seed in range(1, 32):
        indices = make_order(layout, legal.POLICY, "release", seed)
        masks[seed-1, indices[:cap]] = True
    groups = defaultdict(lambda: defaultdict(float))
    start = 0
    for (group, judges, experts), length in zip(rows, lengths):
        checked = masks[:, start:start+length]
        pred = judges[model_index]
        truth = experts.all(axis=1)
        # An initially failing output can be released early exactly when every
        # predicted failure was checked, no checked answer failed, and a check remains.
        clears_predictions = np.all(checked | pred, axis=1)
        no_failure_seen = np.all(~checked[:, None, :] | experts, axis=2)
        eligible = ~pred.all() & clears_predictions & ~checked.all(axis=1)
        early = (eligible[:, None] & no_failure_seen).mean(axis=0)
        for name, value in output_counts(pred.all(), truth, early).items():
            groups[group][name] += value
        start += length
    return groups


def ratios(counts):
    def percent(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, 100 * num / den, np.nan)
    n = counts["outputs"]
    result = {}
    for k in ("early", "auditor_pass", "auditor_pass_judge_fail", "early_auditor_pass",
              "early_auditor_fail", "auditor_fail_judge_pass"):
        result[k + "_peer_pass_pct"] = percent(counts[k + "_peer_pass"], counts[k + "_n"])
    result["early_minus_full_judge_fail_pp"] = (result["early_peer_pass_pct"]
                                                - result["auditor_pass_judge_fail_peer_pass_pct"])
    result["final_verdict_kept_among_early_pct"] = percent(counts["early_auditor_pass_n"], counts["early_n"])
    for k in ("judge_pass", "expert_pass", "early_n", "judge_fp", "judge_ff", "full_fp", "full_ff"):
        result[k + "_per100"] = percent(counts[k], n)
    result["judge_error_per100"] = result["judge_fp_per100"] + result["judge_ff_per100"]
    result["full_error_per100"] = result["full_fp_per100"] + result["full_ff_per100"]
    result["full_minus_judge_per100"] = result["full_error_per100"] - result["judge_error_per100"]
    result["full_minus_always_fail_per100"] = result["full_error_per100"] - result["expert_pass_per100"]
    upward = counts["auditor_pass_judge_fail_n"]
    downward = counts["auditor_fail_judge_pass_n"]
    endorsed_changes = counts["auditor_pass_judge_fail_peer_pass"] + downward - counts["auditor_fail_judge_pass_peer_pass"]
    result["complete_change_rate_per100"] = percent(upward + downward, n)
    result["peer_endorses_completed_changes_pct"] = percent(endorsed_changes, upward + downward)
    np.testing.assert_allclose(result["full_minus_judge_per100"],
                               percent(upward + downward - 2 * endorsed_changes, n), atol=1e-12)
    # Same output, auditor, scorer and mask. Among early approvals only, a final
    # auditor FAIL flips the action; it helps if the peer also says FAIL.
    difference = counts["early_auditor_fail_n"] - 2 * counts["early_auditor_fail_peer_pass"]
    result["early_minus_completed_on_same_cases_per100_outputs"] = percent(difference, n)
    result["early_minus_completed_per100_early_approvals"] = percent(difference, counts["early_n"])
    return result


def summarize(label, groups):
    names = list(next(iter(groups.values())))
    matrix = np.array([[groups[g][k] for k in names] for g in sorted(groups)])
    n = len(matrix)
    draws = np.random.default_rng(SEED).multinomial(n, np.ones(n)/n, size=DRAWS)
    totals = dict(zip(names, matrix.sum(axis=0)))
    points = ratios(totals)
    samples = ratios(dict(zip(names, (draws @ matrix).T)))
    rows = []
    for key, value in points.items():
        finite = samples[key][np.isfinite(samples[key])]
        lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (None, None)
        rows.append(dict(frame=label, metric=key, estimate=float(value) if np.isfinite(value) else None,
                         lower95=lo, upper95=hi, defined_draws=len(finite), total_draws=DRAWS))
    return rows, totals


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(cache, out):
    news.check_freeze()
    leg, _ = legal.load_panel()
    articles, judges, experts = news.load_frames(cache)["held70"]
    summaries = list(zip(articles, judges, experts))
    frames = [("JudgmentBench / GPT-5.4", [(g, j, e) for g, _, e, j, _ in leg], 0)]
    frames.extend(("SummEval / " + name, summaries, MODELS.index(name)) for name in ("UniEval", "G-Eval GPT-4"))
    estimates, cluster_counts = [], []
    for label, rows, j in frames:
        groups = make_groups(rows, j)
        calculated, totals = summarize(label, groups)
        if not np.isclose(totals["full_fp"], totals["full_ff"]):
            raise ValueError("Role symmetry violated")
        if not np.isclose(totals["early_n"], totals["early_auditor_pass_n"] + totals["early_auditor_fail_n"]):
            raise ValueError("Incomplete early-approval partition")
        estimates.extend(calculated)
        cluster_counts.extend({"frame": label, "cluster": i, **groups[g]} for i, g in enumerate(sorted(groups), 1))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "estimates.csv", estimates)
    write_csv(out / "cluster_counts.csv", cluster_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.cache, args.out)
