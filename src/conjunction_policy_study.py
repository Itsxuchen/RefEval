"""Fixed-reference budget, error-flow and order-sensitivity experiments.

Policy ordering never reads gold. Only observed FAIL answers trigger short circuit.
Unobserved gold is used by the evaluator, not by the query-selection controller.
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("random_criterion", "sc_natural", "sc_random", "sc_judge",
            "disagreement_only", "disagreement_then_random", "sc_disagreement")
TASK_ORDERS = ("release", "predicted_fail_count", "random")
FIELDS = ("criterion_errors_found", "certified_tasks", "residual_fp", "residual_ff",
          "introduced_fp", "fixed_initial_fp", "fixed_initial_ff", "tasks_touched", "new_fp_events",
          "certified_initial_ff")
GRID = np.arange(101) / 100.0


def rng_for(seed: int, purpose: str) -> np.random.Generator:
    word = int.from_bytes(hashlib.sha256(purpose.encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([20260913, seed, word]))


def make_order(data: dict, policy: str, task_order: str, seed: int) -> np.ndarray:
    """Observable-data-only candidate priorities; no reference access."""
    if policy not in POLICIES or task_order not in TASK_ORDERS:
        raise ValueError("unknown policy/task order")
    pred = np.asarray(data["prediction"])
    ti = np.asarray(data["task_index"])
    m, n = len(pred), len(data["task_ids"])
    slots = np.arange(m)
    # Random priorities stay fixed when a task is removed from a transcript.
    item_random = rng_for(seed, "criterion").random(m)
    task_random = rng_for(seed, "task").random(n)
    ties = slots if seed == 0 else item_random
    fail_count = np.bincount(ti, weights=1 - pred, minlength=n)
    if task_order == "release":
        task_rank = np.arange(n)
    else:
        if task_order == "random":
            tasks = np.argsort(task_random, kind="stable")
        else:
            tt = np.arange(n) if seed == 0 else task_random
            tasks = np.lexsort((tt, fail_count))
        task_rank = np.empty(n, dtype=int)
        task_rank[tasks] = np.arange(n)
    secondary = np.asarray(data["secondary_predictions"])
    flags = np.any(secondary != pred[None, :], axis=0) if len(secondary) else np.zeros(m, bool)
    if policy == "random_criterion":
        return np.argsort(item_random, kind="stable")
    if policy.startswith("disagreement_"):
        flagged = slots[flags]
        flagged = flagged[np.lexsort((ties[flagged], task_rank[ti[flagged]]))]
        if policy == "disagreement_only":
            return flagged
        remaining = slots[~flags]
        return np.concatenate((flagged, remaining[np.argsort(item_random[remaining], kind="stable")]))
    if policy == "sc_natural":
        return np.lexsort((slots, task_rank[ti]))
    if policy == "sc_random":
        return np.lexsort((item_random, task_rank[ti]))
    tier = pred if policy == "sc_judge" else np.where(flags, 0, np.where(pred == 0, 1, 2))
    return np.lexsort((ties, tier, task_rank[ti]))


def initial_state(data: dict) -> dict:
    ti, g, p = (np.asarray(data[k]) for k in ("task_index", "gold", "prediction"))
    n = len(data["task_ids"])
    k = np.bincount(ti, minlength=n).astype(int)
    if np.any(k == 0) or len(g) != len(p) or not np.isin(g, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("nonempty binary tasks required")
    gf = np.bincount(ti, weights=1 - g, minlength=n).astype(int)
    pf = np.bincount(ti, weights=1 - p, minlength=n).astype(int)
    correct_fail = np.bincount(ti, weights=((g == 0) & (p == 0)), minlength=n).astype(int)
    fp = (gf > 0) & (pf == 0)
    ff = (gf == 0) & (pf > 0)
    base = np.zeros((n, len(FIELDS)), dtype=int)
    base[:, 2], base[:, 3] = fp, ff
    return {"k": k, "gf": gf, "pf": pf, "initial_fp": fp, "initial_ff": ff,
            "baseline": base, "susceptible": (gf > 0) & (pf > 0) & (correct_fail == 0)}


def replay(data: dict, policy: str, task_order: str, seed: int,
           candidate_order: np.ndarray | None = None) -> dict:
    """Return actual query transcript and additive per-query evaluator changes."""
    order = make_order(data, policy, task_order, seed) if candidate_order is None else np.asarray(candidate_order)
    ti, gold, pred = (np.asarray(data[k]) for k in ("task_index", "gold", "prediction"))
    if len(set(order.tolist())) != len(order) or np.any(order < 0) or np.any(order >= len(pred)):
        raise ValueError("invalid or repeated candidate")
    st = initial_state(data)
    n = len(st["k"])
    pf, seen = st["pf"].copy(), np.zeros(n, dtype=int)
    observed_fail, certified = np.zeros(n, bool), np.zeros(n, bool)
    queries, deltas = [], []
    for index in order:
        t = int(ti[index])
        # This is the only reference-dependent selection, using an observed answer.
        if policy.startswith("sc_") and observed_fail[t]:
            continue
        g, p = int(gold[index]), int(pred[index])
        truth_pass = st["gf"][t] == 0
        before_pass = pf[t] == 0
        old_fp, old_ff = (before_pass and not truth_pass), (not before_pass and truth_pass)
        first_touch = seen[t] == 0
        seen[t] += 1
        pf[t] += p - g
        observed_fail[t] |= g == 0
        now_cert = bool(observed_fail[t] or seen[t] == st["k"][t])
        after_pass = pf[t] == 0
        fp, ff = (after_pass and not truth_pass), (not after_pass and truth_pass)
        if now_cert and (fp or ff):
            raise AssertionError("certified task is wrong")
        if ff and not old_ff:
            raise AssertionError("perfect replacement cannot introduce false fail")
        initial_fp, initial_ff = bool(st["initial_fp"][t]), bool(st["initial_ff"][t])
        d = [int(g != p), int(now_cert and not certified[t]), int(fp) - int(old_fp), int(ff) - int(old_ff),
             (int(fp) - int(old_fp)) if not initial_fp else 0,
             (int(old_fp) - int(fp)) if initial_fp else 0,
             (int(old_ff) - int(ff)) if initial_ff else 0,
             int(first_touch), int(fp and not old_fp), int(now_cert and not certified[t] and initial_ff)]
        queries.append(int(index)); deltas.append(d)
        certified[t] = now_cert
    arr = np.asarray(deltas, dtype=np.int64).reshape(-1, len(FIELDS))
    return {"indices": np.asarray(queries, dtype=int), "deltas": arr,
            "baseline_by_task": st["baseline"], "susceptible": st["susceptible"]}


def points_from_trace(data: dict, trace: dict, shares: np.ndarray = GRID,
                      removed_base: str | None = None) -> list[dict]:
    ti = np.asarray(data["task_index"])
    keep_tasks = np.ones(len(data["task_ids"]), bool)
    if removed_base is not None:
        keep_tasks = np.asarray(data["base_task_ids"]) != removed_base
    keep_items = keep_tasks[ti]
    keep_queries = keep_items[trace["indices"]]
    changes = trace["deltas"][keep_queries]
    start = trace["baseline_by_task"][keep_tasks].sum(axis=0)
    cumulative = np.vstack((start, start + np.cumsum(changes, axis=0)))
    m, n = int(keep_items.sum()), int(keep_tasks.sum())
    if not m or not n:
        return []
    initial_errors = int(start[2] + start[3])
    criterion_errors = int(((data["gold"] != data["prediction"]) & keep_items).sum())
    rows = []
    for share in shares:
        cap = int(round(float(share) * m))
        used = min(cap, len(changes))
        row = dict(zip(FIELDS, map(int, cumulative[used])))
        row.update(budget_share=float(share), budget_cap=cap, actual_queries=used,
                   n_tasks=n, n_criteria=m, primary_criterion_errors=criterion_errors,
                   initial_task_errors=initial_errors, residual_errors=row["residual_fp"] + row["residual_ff"],
                   uncertified_tasks=n - row["certified_tasks"], unused_budget=cap - used)
        row["gated_residual_fp"] = int(start[2]) - row["fixed_initial_fp"]
        row["gated_residual_ff"] = int(start[3]) - row["certified_initial_ff"]
        row["gated_residual_errors"] = row["gated_residual_fp"] + row["gated_residual_ff"]
        row["delayed_ff_corrections"] = row["gated_residual_ff"] - row["residual_ff"]
        rows.append(row)
    return rows


def trace_summary(data: dict, trace: dict) -> dict:
    n = len(data["task_ids"])
    values = trace["baseline_by_task"].sum(axis=0) + np.cumsum(trace["deltas"], axis=0)
    all_cert = np.flatnonzero(values[:, 1] == n)
    witnesses = []
    for pos in np.flatnonzero(trace["deltas"][:, 8] > 0)[:5]:
        index = int(trace["indices"][pos]); t = int(data["task_index"][index])
        witnesses.append({"query_number": int(pos + 1), "task_id": data["task_ids"][t],
                          "base_task_id": data["base_task_ids"][t], "criterion_id": data["criterion_ids"][index],
                          "category": data["category"][index]})
    return {"total_queries": len(trace["indices"]),
            "first_all_certified": int(all_cert[0] + 1) if len(all_cert) else None,
            "susceptible_masked_tasks": int(trace["susceptible"].sum()),
            "new_fp_events_total": int(trace["deltas"][:, 8].sum()),
            "maximum_simultaneous_introduced_fp": int(values[:, 4].max()) if len(values) else 0,
            "new_fp_witnesses": witnesses}


def metadata(data: dict) -> dict:
    st = initial_state(data)
    return {"family": data["family"], "target": data["target"],
            "tasks": len(data["task_ids"]), "base_tasks": len(set(data["base_task_ids"])),
            "criteria": len(data["gold"]), "reference_pass": int((st["gf"] == 0).sum()),
            "judge_pass": int((st["pf"] == 0).sum()), "false_pass": int(st["initial_fp"].sum()),
            "false_fail": int(st["initial_ff"].sum()), "susceptible_masked_tasks": int(st["susceptible"].sum()),
            "certificate_lower_bound": int(np.where(st["gf"] > 0, 1, st["k"]).sum()),
            "extra_judge_runs_for_disagreement": len(data["secondaries"]),
            "secondaries": data["secondaries"], "source_hashes": data["source_hashes"]}


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def pairwise_summary(rows: list[dict]) -> list[dict]:
    """Complete ordered-policy comparisons; finite grid/seed counts, not p-values."""
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r["dataset"], r["task_order"], r["seed"], r["budget_share"])][r["policy"]] = r
    result = defaultdict(lambda: defaultdict(int))
    for (ds, order, seed, share), pols in grouped.items():
        if not 0 < share < 1:
            continue
        for i, a in enumerate(POLICIES):
            for b in POLICIES[i+1:]:
                x, y = pols[a], pols[b]
                z = result[(ds, order, a, b)]
                z["shared_cap_points"] += 1
                equal = x["actual_queries"] == y["actual_queries"] == x["budget_cap"]
                z["equal_actual_active_points"] += int(equal)
                if not equal:
                    continue
                dc = x["certified_tasks"] - y["certified_tasks"]
                de = x["residual_errors"] - y["residual_errors"]
                db = x["criterion_errors_found"] - y["criterion_errors_found"]
                z["certification_vs_residual_conflict"] += int(dc * de > 0)
                z["certification_vs_bit_discovery_conflict"] += int(dc * db < 0)
                z["a_more_certified"] += int(dc > 0)
                z["a_fewer_residual"] += int(de < 0)
                z["a_more_bit_errors_found"] += int(db > 0)
                z["a_weakly_dominates_three"] += int(dc >= 0 and de <= 0 and db >= 0 and (dc > 0 or de < 0 or db > 0))
                z["b_weakly_dominates_three"] += int(dc <= 0 and de >= 0 and db <= 0 and (dc < 0 or de > 0 or db < 0))
    return [dict(dataset=k[0], task_order=k[1], a=k[2], b=k[3], **v) for k, v in result.items()]


def run_study(datasets: dict, seeds: int, output: Path, loto: bool = True) -> dict:
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True)
    ends, anchors = [], []
    row_count = deletion_count = 0
    pairs = defaultdict(lambda: defaultdict(int))
    with gzip.open(output / "budget_curves.csv.gz", "wt", encoding="utf-8", newline="") as cf, \
         gzip.open(output / "leave_one_base_task_out.csv.gz", "wt", encoding="utf-8", newline="") as lf:
        cw = lw = None
        for name, data in datasets.items():
            for seed in range(seeds):
                for order in TASK_ORDERS:
                    comparison_rows = []
                    for policy in POLICIES:
                        tr = replay(data, policy, order, seed)
                        base = {"dataset": name, "task_order": order, "policy": policy, "seed": seed,
                                "extra_judge_runs": len(data["secondaries"]) if "disagreement" in policy else 0}
                        pts = [dict(**base, **r) for r in points_from_trace(data, tr)]
                        if cw is None:
                            cw = csv.DictWriter(cf, fieldnames=list(pts[0])); cw.writeheader()
                        cw.writerows(pts); row_count += len(pts)
                        comparison_rows.extend(pts)
                        if seed == 0:
                            anchors.extend(r for r in pts if r["budget_share"] == .2)
                        ends.append(dict(**base, **trace_summary(data, tr)))
                        if loto and seed == 0 and order != "random":
                            for group in sorted(set(data["base_task_ids"])):
                                dropped = [dict(**base, removed_base=group, **r)
                                           for r in points_from_trace(data, tr, removed_base=group)]
                                if not dropped:
                                    continue
                                if lw is None:
                                    lw = csv.DictWriter(lf, fieldnames=list(dropped[0])); lw.writeheader()
                                lw.writerows(dropped); deletion_count += len(dropped)
                    for r in pairwise_summary(comparison_rows):
                        key = tuple(r[k] for k in ("dataset", "task_order", "a", "b"))
                        for k, v in r.items():
                            if k not in ("dataset", "task_order", "a", "b"):
                                pairs[key][k] += v
            print(f"completed {name}: {seeds} seeds; elapsed {time.perf_counter()-started:.2f}s", flush=True)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(__file__), ROOT / "src/conjunction_policy_data.py", ROOT / "context/conjunction_policy_protocol.md"]}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "seeds": seeds,
              "seed_interpretation": "seed0 canonical ties; seeds1+ randomized ties, not confidence intervals",
              "grid": GRID.tolist(), "policies": list(POLICIES), "task_orders": list(TASK_ORDERS),
              "source_hashes": hashes, "datasets": {k: metadata(v) for k, v in datasets.items()},
              "counts": {"trajectories": len(ends), "budget_rows": row_count, "deletion_rows": deletion_count},
              "endpoints": ends,
              "pairwise_equal_actual": [dict(dataset=k[0], task_order=k[1], a=k[2], b=k[3], **v) for k,v in pairs.items()],
              "canonical_20pct": anchors,
              "elapsed_seconds": time.perf_counter()-started}
    (output / "study.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    from src.conjunction_policy_data import load_datasets
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--no-loto", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reproduced/conjunction_policy_study")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be positive")
    data = load_datasets(ROOT)
    if args.dataset:
        data = {k: data[k] for k in args.dataset}
    r = run_study(data, args.seeds, args.output, not args.no_loto)
    print(json.dumps({"output": str(args.output), **r["counts"], "elapsed_seconds": r["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
