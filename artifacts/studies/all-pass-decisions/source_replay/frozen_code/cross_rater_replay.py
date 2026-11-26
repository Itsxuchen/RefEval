"""Score decisions made from one released human vector against another, offline.

The scoring reference enters only after the audit answers, certificate and both
decisions exist. Policies, seeds, identities and underlying labels come from
the original matched JudgmentBench reference study.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.conjunction_label_value import SHARES, hybrid_queries
from src.conjunction_policy_data import load_datasets
from src.conjunction_policy_study import make_order, replay
from src.conjunction_reference_sensitivity import GRID, POLICIES, build_frames

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "artifacts/reports/conjunction_robustness/reference"
OUT = ROOT / "artifacts/reports/cross_rater_20261125"
METRICS = ("cert_pass", "cert_fail", "wrong_cert_pass", "wrong_cert_fail",
           "eager_FP", "eager_FF", "gated_FP", "gated_FF", "eager_errors",
           "gated_errors", "gated_minus_eager", "eager_repairs", "eager_harms",
           "gated_repairs", "gated_harms")
KEY = ("dataset", "policy", "audit", "score", "budget_share", "budget_cap")
SHOWN = (0., .006, .05, .2, .5, .6, .84, .95, 1.)
MIX = "sc50_then_srs"
N_SEEDS = 31
N_DRAWS = 1999


def audit(frame, queries, caps):
    """Return actions for every prefix. No scorer labels are an argument."""
    pred = np.asarray(frame["prediction"], dtype=bool)
    gold = np.asarray(frame["gold"], dtype=bool)
    ti = np.asarray(frame["task_index"])
    q = np.asarray(queries, dtype=int)
    m = len(pred)
    if len(set(q)) != len(q) or np.any(q < 0) or np.any(q >= m):
        raise ValueError("Invalid audit transcript")
    if len(q) and q.ndim != 1:
        raise ValueError("Expected one-dimensional query indices")
    starts = np.r_[0, np.flatnonzero(np.diff(ti)) + 1]
    if not np.array_equal(ti[starts], np.arange(len(starts))) or len(gold) != m:
        raise ValueError("Expected contiguous aligned output vectors")
    rank = np.full(m, m + 1, dtype=int)
    rank[q] = np.arange(1, len(q) + 1)
    seen = rank[None, :] <= np.minimum(caps, len(q))[:, None]
    cert_fail = np.logical_or.reduceat(seen & ~gold, starts, axis=1)
    cert_pass = np.logical_and.reduceat(seen & gold, starts, axis=1)
    eager = np.logical_and.reduceat(np.where(seen, gold, pred), starts, axis=1)
    initial = np.logical_and.reduceat(pred, starts)
    gated = cert_pass | (~cert_fail & initial)
    if np.any((cert_fail | cert_pass) & (gated != eager)):
        raise AssertionError("Eager and gated differ on certified outputs")
    return starts, initial, cert_pass, cert_fail, eager, gated


def score(actions, reference):
    starts, initial, cert_pass, cert_fail, eager, gated = actions
    truth = np.logical_and.reduceat(np.asarray(reference, dtype=bool), starts)
    err0 = initial != truth
    ee, ge = eager != truth, gated != truth
    return np.stack((cert_pass, cert_fail, cert_pass & ~truth, cert_fail & truth,
                     eager & ~truth, ~eager & truth, gated & ~truth, ~gated & truth,
                     ee, ge, ge.astype(np.int8) - ee, err0 & ~ee, ~err0 & ee,
                     err0 & ~ge, ~err0 & ge), axis=-1)


def bootstrap_rates(counts, sizes, draws):
    """Resample original base-task contributions with a shared output denominator."""
    sampled_sizes = draws @ sizes
    point = 100 * counts.sum(axis=0) / sizes.sum()
    draws_per100 = 100 * (draws @ counts) / sampled_sizes[:, None]
    return point, *np.quantile(draws_per100, (.025, .975), axis=0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def draw_curves(rows, path):
    """Shared-scale timing contrasts for both judges and both registered targets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True)
    styles = (("A", "A", "#175385", "-"), ("A", "B", "#175385", "--"),
              ("B", "B", "#aa4515", "-"), ("B", "A", "#aa4515", "--"))
    for column, judge in enumerate(("GPT-5.4", "GPT-5.4-mini")):
        for row, suffix in enumerate(("", " | binary_only")):
            ax = axes[row, column]
            ax.axhline(0, color="#636363", linewidth=.7)
            for auditor, scorer, color, style in styles:
                data = [r for r in rows if r["dataset"] == f"JB {judge}{suffix}"
                        and r["policy"] == "disagreement_then_random"
                        and r["audit"] == auditor and r["score"] == scorer]
                ax.plot([100*r["budget_share"] for r in data],
                        [100*r["gated_minus_eager_mean"]/r["n_outputs"] for r in data],
                        color=color, linestyle=style, linewidth=2,
                        label=f"Audit {auditor}, score {scorer}")
            ax.set_title(f"{judge} · {'binary-only' if suffix else 'full rubric'}", loc="left")
            ax.grid(axis="y", alpha=.17)
            ax.spines[["right", "top"]].set_visible(False)
            if row:
                ax.set_xlabel("Criteria checked (%)")
            if not column:
                ax.set_ylabel("Gated − eager errors per 100")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(path, dpi=170)
    plt.close(figure)


def legacy_checks(path):
    """Compare every original diagonal replicate for timing and allocation."""
    with gzip.open(path, "rt") as f:
        new = {}
        for r in csv.DictReader(f):
            if r["audit"] != r["score"]:
                continue
            key = r["dataset"], r["policy"], r["audit"], int(r["seed"]), float(r["budget_share"])
            new[key] = tuple(int(r[m]) for m in
                             ("actual_queries", "cert_pass", "cert_fail", "eager_FP",
                              "eager_FF", "gated_FP", "gated_FF", "gated_minus_eager"))
    metrics = ("certified_pass", "certified_fail", "eager_FP", "eager_FF", "gated_FP",
               "gated_FF", "gated_minus_eager")
    checks = {}
    for name in ("rq1", "rq2"):
        checked = 0
        with gzip.open(ARCHIVE / f"{name}_rows.csv.gz", "rt") as f:
            for r in csv.DictReader(f):
                if r["acquisition_ref"] != r["evaluation_ref"]:
                    continue
                key = (r["acquisition_ref"], int(r["seed"]), float(r["budget_share"]))
                policies = [(r["policy"], "")] if name == "rq1" else [
                    ("random_criterion", "srs_"), (MIX, "mix_")]
                for pol, prefix in policies:
                    expected = (int(r["actual_queries"]), *(int(r[prefix + k]) for k in metrics))
                    if new[(r["dataset"], pol, *key)] != expected:
                        raise AssertionError((name, pol, key, expected))
                    checked += 1
        checks[name] = checked
    if checks != {"rq1": 75144, "rq2": 2976}:
        raise AssertionError(checks)
    return checks


def run():
    datasets = load_datasets(ROOT)
    frames, pairs, diagnostics = build_frames(datasets)
    archive_manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    for name in frames:
        if datasets[name]["source_hashes"] != archive_manifest["data_hashes"][name]:
            raise ValueError("Input hash mismatch for " + name)
    for name in ("pairing.csv", "rq1_rows.csv.gz", "rq2_rows.csv.gz", "frame_diagnostics.json"):
        if sha(ARCHIVE / name) != archive_manifest["output_hashes"][name]:
            raise ValueError("Archived comparator changed: " + name)
    with (ARCHIVE / "pairing.csv").open() as f:
        if [{k: str(v) for k,v in p.items()} for p in pairs] != list(csv.DictReader(f)):
            raise ValueError("Reconstructed panel/anchor identities differ from archive")
    if diagnostics != json.loads((ARCHIVE / "frame_diagnostics.json").read_text()):
        raise ValueError("Reconstructed frame diagnostics differ")
    base_ids = np.array([p["base_task_id"] for p in pairs])
    start = np.r_[0, np.flatnonzero(base_ids[1:] != base_ids[:-1]) + 1]
    if len(set(base_ids)) != len(start):
        raise ValueError("Base tasks must be consecutive")
    sizes = np.diff(np.r_[start, len(base_ids)])
    n = len(sizes)
    draws = np.random.default_rng(20261125).multinomial(n, np.ones(n)/n, size=N_DRAWS)
    # Every stored cluster contribution is a sum over the same paired 31 orders.
    cluster = {}
    actual = defaultdict(int)
    order_rows = 0
    OUT.mkdir(parents=True, exist_ok=True)
    filepath = OUT / "orders.csv.gz"
    with filepath.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped, io.TextIOWrapper(zipped, newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((*KEY, "actual_queries", "seed", *METRICS))
        for dataset, refs in frames.items():
            m = len(refs["A"]["gold"])
            for seed in range(1, N_SEEDS + 1):
                for pol in POLICIES:
                    # Nonadaptive priorities are exactly shared across the four cells.
                    candidates = make_order(refs["A"], pol, "release", seed)
                    for auditor, frame in refs.items():
                        q = replay(frame, pol, "release", seed, candidates)["indices"] if pol == "sc_judge" else candidates
                        shares = tuple(sorted(set(GRID) | {0.006}))
                        batches = [(pol, q, shares)]
                        if pol == "sc_judge":
                            for share in SHARES:
                                cap = round(m * share)
                                first, rest = hybrid_queries(frame, cap, "release", seed, {"indices": q})
                                mix = np.r_[first, rest]
                                if len(mix) != cap:
                                    raise ValueError("Unequal allocation budget")
                                batches.append((MIX, mix, [share]))
                        for policy, queries, shares in batches:
                            caps = np.array([round(m * b) for b in shares])
                            actions = audit(frame, queries, caps)
                            for scorer, score_frame in refs.items():
                                measures = score(actions, score_frame["gold"])
                                per_base = np.add.reduceat(measures, start, axis=1, dtype=np.int32)
                                totals = per_base.sum(axis=1)
                                if auditor == scorer and np.any(totals[:, 2:4]):
                                    raise AssertionError("Own-panel certificate mismatch")
                                for share, cap, by_base, row in zip(shares, caps, per_base, totals):
                                    key = dataset, policy, auditor, scorer, share, int(cap)
                                    if key not in cluster:
                                        cluster[key] = np.zeros_like(by_base)
                                    cluster[key] += by_base
                                    used = min(len(queries), cap)
                                    actual[key] += int(used)
                                    writer.writerow((*key, used, seed, *row))
                                    order_rows += 1
    checks = legacy_checks(filepath)
    curves, intervals, allocation, reference_contrasts, cluster_inputs = [], [], [], [], []
    for key in sorted(cluster):
        head = dict(zip(KEY, key))
        dataset, policy, auditor, scorer, share, cap = key
        by_base = cluster[key] / N_SEEDS
        if dataset == "JB GPT-5.4" and share == .5 and policy in (
                "disagreement_then_random", MIX, "random_criterion"):
            for i, (size, totals) in enumerate(zip(sizes, cluster[key]), 1):
                cluster_inputs.append({**head, "cluster_index": i, "n_outputs": size,
                    "seeds": N_SEEDS, **{m + "_seed_sum": v for m,v in zip(METRICS, totals)}})
        mean = by_base.sum(axis=0)
        curves.append({**head, "actual_queries_mean": actual[key]/N_SEEDS, "n_outputs": len(pairs),
                       **{m + "_mean": v for m, v in zip(METRICS, mean)}})
        if share not in SHOWN:
            continue
        def interval_rows(matrix, fields, metadata):
            indices = [METRICS.index(k) for k in fields]
            point, lo, hi = bootstrap_rates(matrix[:, indices], sizes, draws)
            return [{**metadata, "metric": name, "mean_per100": p, "p025_per100": l, "p975_per100": h}
                    for name, p, l, h in zip(fields, point, lo, hi)]
        fields = ("eager_errors", "gated_errors", "gated_minus_eager", "wrong_cert_pass", "wrong_cert_fail")
        intervals.extend(interval_rows(by_base, fields, head))
        if policy == MIX:
            srs_key = dataset, "random_criterion", auditor, scorer, share, cap
            if actual[key] != actual[srs_key] or actual[key] != N_SEEDS * cap:
                raise AssertionError("Unequal realized audit spend")
            diff = by_base - cluster[srs_key] / N_SEEDS
            allocation.extend(interval_rows(diff, fields + ("cert_pass", "cert_fail"), head))
        if scorer != auditor:
            # Only the external scorer changes; actions and acquired evidence agree.
            own_key = dataset, policy, auditor, auditor, share, cap
            diff = by_base - cluster[own_key] / N_SEEDS
            reference_contrasts.extend(interval_rows(diff, fields, head))
    write_csv(OUT / "curves.csv", curves)
    write_csv(OUT / "task_cluster_intervals.csv", intervals)
    write_csv(OUT / "mix_minus_srs.csv", allocation)
    write_csv(OUT / "external_minus_own.csv", reference_contrasts)
    write_csv(OUT / "cluster_inputs_full_gpt54_half.csv", cluster_inputs)
    draw_curves(curves, OUT / "df_timing.png")
    out_files = ("orders.csv.gz", "curves.csv", "task_cluster_intervals.csv",
                 "mix_minus_srs.csv", "external_minus_own.csv", "df_timing.png",
                 "cluster_inputs_full_gpt54_half.csv")
    evidence = dict(n_outputs=len(pairs), n_base_tasks=n, seeds=N_SEEDS,
                    cluster_draws=N_DRAWS, cluster_seed=20261125, order_rows=order_rows,
                    curve_rows=len(curves), diagonal_rows_matched=checks,
                    archive_manifest_sha256=sha(ARCHIVE / "manifest.json"),
                    raw_inputs=datasets[next(iter(frames))]["source_hashes"],
                    source_sha256={p:sha(ROOT / "src" / p) for p in (
                        "cross_rater_replay.py", "conjunction_policy_study.py", "conjunction_policy_data.py",
                        "conjunction_reference_sensitivity.py", "conjunction_label_value.py")},
                    output_sha256={p:sha(OUT / p) for p in out_files})
    (OUT / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k:v for k,v in evidence.items() if k not in ("raw_inputs","source_sha256","output_sha256")}, indent=2))


if __name__ == "__main__":
    run()
