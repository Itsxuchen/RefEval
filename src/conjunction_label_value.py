"""Offline label-allocation extension: estimation, correction and certification.

Uniform criterion sampling is an explicitly designed baseline. Exact intervals below are conditional randomization intervals for the
fixed population's micro agreement, not expert-reference uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from src.conjunction_policy_data import load_datasets
from src.conjunction_policy_study import replay, rng_for, TASK_ORDERS

ROOT = Path(__file__).resolve().parents[1]
SHARES = (0.006, 0.05, 0.20, 0.50, 0.60, 1.0)


@lru_cache(maxsize=100000)
def hypergeom_ci(M: int, n: int, x: int, alpha: float = .05) -> tuple[int, int]:
    """Invert two equal-tail hypergeometric tests for the unknown success total.

    Includes tail probabilities exactly equal to alpha/2; discrete coverage is
    at least 1-alpha at each fixed sampling size, not simultaneous over budgets.
    """
    if not (isinstance(M, (int, np.integer)) and isinstance(n, (int, np.integer))
            and isinstance(x, (int, np.integer)) and 0 <= x <= n <= M and 0 < alpha < 1):
        raise ValueError("invalid hypergeometric observation")
    if n == 0:
        return 0, M
    if n == M:
        return x, x
    lo, hi = x, M - n + x
    while lo < hi:
        mid = (lo + hi) // 2
        if hypergeom.sf(x - 1, M, mid, n) >= alpha / 2:
            hi = mid
        else:
            lo = mid + 1
    lower = lo
    lo, hi = x, M - n + x
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom.cdf(x, M, mid, n) >= alpha / 2:
            lo = mid
        else:
            hi = mid - 1
    return lower, lo


def accuracy_inference(M: int, qA: int, correctA: int, qR: int, correctR: int) -> dict:
    """Known adaptive labels plus a fresh SRS of the remaining criterion slots."""
    if not (M > 0 and 0 <= correctA <= qA <= M and 0 <= correctR <= qR <= M - qA):
        raise ValueError("invalid adaptive/SRS counts")
    U = M - qA
    if U == 0:
        estimate = correctA / M
        return dict(estimate=estimate, ci_lower=estimate, ci_upper=estimate)
    lower, upper = hypergeom_ci(U, qR, correctR)
    return dict(estimate=(correctA + U * correctR / qR) / M if qR else None,
                ci_lower=(correctA + lower) / M, ci_upper=(correctA + upper) / M)


def query_metrics(data: dict, indices: np.ndarray) -> dict:
    """Direct queried-set arithmetic, independent of the previous delta updater."""
    g, p, ti = (np.asarray(data[k]) for k in ("gold", "prediction", "task_index"))
    m, n = len(g), len(data["task_ids"])
    indices = np.asarray(indices, dtype=int)
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= m):
        raise ValueError("duplicate or invalid query")
    observed = np.zeros(m, bool); observed[indices] = True
    k = np.bincount(ti, minlength=n)
    if np.any(k == 0):
        raise ValueError("empty task")
    seen = np.bincount(ti[observed], minlength=n)
    known_fail = np.bincount(ti[observed & (g == 0)], minlength=n) > 0
    known_pass = (seen == k) & ~known_fail
    unknown = ~(known_fail | known_pass)
    updated = np.where(observed, g, p)
    verdict = np.bincount(ti, weights=1 - updated, minlength=n) == 0
    truth = np.bincount(ti, weights=1 - g, minlength=n) == 0
    correct = int((g[indices] == p[indices]).sum())
    return dict(actual_queries=len(indices), certified_tasks=int((~unknown).sum()),
                pass_lower=float(known_pass.mean()), pass_upper=float((known_pass | unknown).mean()),
                pass_width=float(unknown.mean()), accuracy_lower=correct / m,
                accuracy_upper=(correct + m - len(indices)) / m,
                residual_errors=int((verdict != truth).sum()),
                criterion_errors_found=len(indices) - correct,
                sample_agreement=correct / len(indices) if len(indices) else None)


def hybrid_queries(data: dict, cap: int, order: str, seed: int, trace: dict | None = None):
    """Frozen50/50 allocation; SC stopping surplus goes to fresh random sampling."""
    m = len(data["gold"])
    if not 0 <= cap <= m:
        raise ValueError("budget outside population")
    if trace is None:
        trace = replay(data, "sc_judge", order, seed)
    A = trace["indices"][:cap // 2]
    remaining = np.ones(m, bool); remaining[A] = False
    slots = np.flatnonzero(remaining)
    priorities = rng_for(seed, "label-value-fresh-remainder").random(m)
    R = slots[np.argsort(priorities[slots], kind="stable")][:cap - len(A)]
    assert len(A) + len(R) == cap
    return A, R


def record(data, ds, seed, order, share, policy, indices, A, R):
    g, p = data["gold"], data["prediction"]
    inf = accuracy_inference(len(g), len(A), int((g[A] == p[A]).sum()),
                             len(R), int((g[R] == p[R]).sum()))
    metrics = query_metrics(data, indices)
    truth = float((g == p).mean())
    return dict(dataset=ds, seed=seed, task_order=order, budget_share=share,
                budget_cap=int(round(share * len(g))), policy=policy,
                n_tasks=len(data["task_ids"]), n_criteria=len(g),
                stageA_queries=len(A), stageR_queries=len(R),
                accuracy_truth=truth, accuracy_estimate=inf["estimate"],
                accuracy_error=None if inf["estimate"] is None else inf["estimate"] - truth,
                accuracy_ci_lower=inf["ci_lower"], accuracy_ci_upper=inf["ci_upper"],
                accuracy_ci_width=inf["ci_upper"] - inf["ci_lower"],
                accuracy_ci_covers=inf["ci_lower"] <= truth <= inf["ci_upper"],
                inference_kind="exact_conditional_SRS" if len(R) else "logical_bounds_or_census",
                **metrics)


def run_cell(ds: str, data: dict, seeds: list[int]) -> list[dict]:
    rows = []
    empty = np.array([], dtype=int)
    for seed in seeds:
        random_trace = replay(data, "random_criterion", "release", seed)
        for order in TASK_ORDERS:
            sc = replay(data, "sc_judge", order, seed)
            for share in SHARES:
                cap = int(round(share * len(data["gold"])))
                if order == "release":
                    R = random_trace["indices"][:cap]
                    rows.append(record(data, ds, seed, order, share, "random_criterion", R, empty, R))
                A = sc["indices"][:cap]
                rows.append(record(data, ds, seed, order, share, "sc_judge", A, A, empty))
                A, R = hybrid_queries(data, cap, order, seed, sc)
                rows.append(record(data, ds, seed, order, share, "sc50_then_srs", np.r_[A, R], A, R))
    return rows


def summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["task_order"], row["policy"], row["budget_share"])].append(row)
    result = []
    for (ds, order, policy, share), rs in groups.items():
        r = dict(dataset=ds, task_order=order, policy=policy, budget_share=share,
                 seeds=len(rs), budget_cap=rs[0]["budget_cap"], n_tasks=rs[0]["n_tasks"], n_criteria=rs[0]["n_criteria"])
        r["inference_kind"] = rs[0]["inference_kind"]
        for key in ("actual_queries", "certified_tasks", "residual_errors", "pass_width",
                    "accuracy_ci_width", "sample_agreement"):
            values = [x[key] for x in rs if x[key] is not None]
            r[key + "_median"] = float(np.median(values)) if values else None
            r[key + "_p05"] = float(np.quantile(values, .05)) if values else None
            r[key + "_p95"] = float(np.quantile(values, .95)) if values else None
        errs = [x["accuracy_error"] for x in rs if x["accuracy_error"] is not None]
        r.update(accuracy_bias=float(np.mean(errs)) if errs else None,
                 accuracy_rmse=float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
                 accuracy_within_half_pp=sum(abs(x) <= .005 for x in errs) if errs else None,
                 accuracy_interval_covered=sum(x["accuracy_ci_covers"] for x in rs))
        result.append(r)
    return result


def f27_bridge_demo() -> dict:
    """A small executable interface bridge; no re-run/relabeling of F27 curves."""
    from src.information_disclosure_replay import disclose_reference, conjunction_bounds
    rows = [dict(task_id=f"t{i // 2}", slot=i % 2, k=2, prediction=1,
                 category="binary", gold=int(i < 2)) for i in range(4)]
    examples = []
    for known in (set(), {"t0"}):
        released = disclose_reference(rows, False, known)
        bound = conjunction_bounds(released)
        examples.append(dict(queried_tasks=sorted(known), f27_with_exact_margins=bound,
                             linked_only_pass_range=[.5, 1.] if known else [0., 1.]))
    return dict(scope="Synthetic interface check: exact margins are extra information, not free labels.",
                examples=examples)


def write_csv(path: Path, rows: list[dict]):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_experiment(datasets: dict, seeds: int, output: Path) -> dict:
    started = time.perf_counter()
    rows = []
    for ds, data in datasets.items():
        part = run_cell(ds, data, list(range(1, seeds + 1)))
        rows.extend(part)
        print(f"{ds}: {len(part)} rows; elapsed {time.perf_counter() - started:.2f}s", flush=True)
    out = Path(output)
    out.mkdir(exist_ok=True, parents=True)
    write_csv(out / "budget_estimation_rows.csv.gz", rows)
    write_csv(out / "budget_estimation_summary.csv", summaries(rows))
    from src.conjunction_task_sampling import run_all as task_samples
    task_rows = task_samples(datasets, seeds=list(range(1, seeds + 1)))
    for row in task_rows:
        lo, hi = hypergeom_ci(row["population_n"], row["sample_n"], row["sample_reference_pass"])
        row["task_pass_ci_lower"] = lo / row["population_n"]
        row["task_pass_ci_upper"] = hi / row["population_n"]
        row["task_pass_ci_covers"] = row["task_pass_ci_lower"] <= row["population_task_pass"] <= row["task_pass_ci_upper"]
    write_csv(out / "whole_task_sampling.csv.gz", task_rows)
    source_paths = [Path(__file__), ROOT / "src/conjunction_task_sampling.py",
                    ROOT / "src/conjunction_policy_data.py", ROOT / "src/conjunction_policy_study.py",
                    ROOT / "src/information_disclosure_replay.py", ROOT / "context/conjunction_label_value_protocol.md"]
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - started,
                   rows=len(rows), task_sampling_rows=len(task_rows), seeds=seeds, shares=SHARES,
                   source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                   data_hashes={ds: data["source_hashes"] for ds, data in datasets.items()},
                   f27_bridge=f27_bridge_demo(),
                   scope="Fixed released references; nine cells/two families; no humans or API calls. Fixed-budget intervals are pointwise, not confidence sequences;31 seeds do not establish coverage.")
    (out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("rows", "task_sampling_rows", "seconds")}), flush=True)

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=31)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reproduced/conjunction_label_value")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be positive")
    run_experiment(load_datasets(), args.seeds, args.output)


if __name__ == "__main__":
    main()
