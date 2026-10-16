"""Preparatory WildIFEval policies and reference-conditional accounting.

The masked preflight failed; no empirical policy comparison has been run.

This module accepts already-authorized binary units. It performs no file reads,
downloads, score parsing, splitting or unmasking. Selection uses P/Q and observed
answers only. Evaluation may read complete reference vectors after authorization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections import defaultdict
from typing import Iterable

import numpy as np

from src.finite_population_intervals import hypergeom_ci

SEEDS = tuple(range(20261011, 20261112))
GRID = tuple(i / 100 for i in range(101))
POLICIES = ("srs", "mix50", "serial", "disagreement")
ALLOCATION_FIELDS = ("C", "CW", "D", "A")


@dataclass
class Frame:
    units: list[dict]
    p: np.ndarray
    q: np.ndarray | None
    reference: np.ndarray
    sizes: np.ndarray
    offsets: np.ndarray
    task_index: np.ndarray
    criterion_ids: list[str]
    task_rank: np.ndarray
    reference_key: str

    @property
    def M(self):
        return len(self.p)

    @property
    def N(self):
        return len(self.units)


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def prepare_frame(units: Iterable[dict], reference_key="r") -> Frame:
    """Canonicalize within one producer; require complete, nonempty bit vectors."""
    units = sorted(list(units), key=lambda x: str(x["unit_id"]))
    if not units or len({str(u["unit_id"]) for u in units}) != len(units):
        raise ValueError("nonempty frame with unique unit IDs required")
    if len({u["producer"] for u in units}) != 1:
        raise ValueError("budgets must be calculated separately for each producer")
    ps, qs, rs, sizes, ids = [], [], [], [], []
    have_q = [unit.get("q") is not None for unit in units]
    if any(have_q) and not all(have_q):
        raise ValueError("auxiliary Q must be complete or entirely unavailable")
    for unit in units:
        p, r = (list(unit[key]) for key in ("p", reference_key))
        q = list(unit["q"]) if all(have_q) else None
        if not p or len(p) != len(r) or (q is not None and len(q) != len(p)):
            raise ValueError("incomplete or empty unit")
        if any(x not in (0, 1) for row in (p, r) + ((q,) if q is not None else ()) for x in row):
            raise ValueError("binary labels required; missing is never FAIL")
        cids = unit.get("criterion_ids", [f'{unit["unit_id"]}::{j}' for j in range(len(p))])
        if len(cids) != len(p):
            raise ValueError("criterion ID length mismatch")
        ps.extend(p); qs.extend(q or []); rs.extend(r); sizes.append(len(p))
        ids.extend(map(str, cids))
    if len(set(ids)) != len(ids):
        raise ValueError("criterion IDs must be unique within a producer frame")
    sizes = np.asarray(sizes, dtype=np.int64)
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    order = sorted(range(len(units)), key=lambda i: (
        _digest(f'refeval-task-order-v1|{units[i]["cluster_id"]}|{units[i]["unit_id"]}'),
        str(units[i]["unit_id"])))
    task_rank = np.empty(len(units), dtype=np.int64)
    task_rank[order] = np.arange(len(units))
    return Frame(units, np.asarray(ps, dtype=np.int8), np.asarray(qs, dtype=np.int8) if all(have_q) else None,
                 np.asarray(rs, dtype=np.int8), sizes, offsets,
                 np.repeat(np.arange(len(units)), sizes), ids, task_rank, reference_key)


def _priorities(frame, seed, purpose):
    # Separate random namespaces protect the fresh-remainder SRS from the
    # reference-dependent short-circuit stage. Seeds couple paired policies.
    return np.asarray([_digest(f"refeval-{purpose}-v1|{seed}|{cid}")
                       for cid in frame.criterion_ids], dtype="S32")


def candidate_orders(frame: Frame, seed: int) -> dict[str, np.ndarray]:
    """All candidate orders depend only on immutable IDs and observable P/Q."""
    uniform = _priorities(frame, seed, "uniform")
    ties = _priorities(frame, seed, "criterion-tie")
    task_rank = frame.task_rank[frame.task_index]
    srs = np.argsort(uniform, kind="stable")
    serial_candidates = np.lexsort((ties, frame.p, task_rank))
    result = dict(srs=srs, serial_candidates=serial_candidates)
    if frame.q is not None:
        flagged = np.flatnonzero(frame.p != frame.q)
        flagged = flagged[np.lexsort((ties[flagged], task_rank[flagged]))]
        remaining = srs[frame.p[srs] == frame.q[srs]]
        result["disagreement"] = np.r_[flagged, remaining]
    return result


def serial_transcript(frame: Frame, candidates: np.ndarray) -> np.ndarray:
    """Query each task in order, stopping immediately on an observed FAIL."""
    ordered_sizes = frame.sizes[np.argsort(frame.task_rank)]
    offsets = np.r_[0, np.cumsum(ordered_sizes)[:-1]]
    positions = np.arange(frame.M)
    first_fail = np.minimum.reduceat(
        np.where(frame.reference[candidates] == 0, positions, frame.M), offsets)
    keep = positions <= np.repeat(first_fail, ordered_sizes)
    return candidates[keep]


def mixed_transcript(serial: np.ndarray, uniform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nested union A_m + SRS(complement A_m), |A_m|=min(floor(m/2),|SC|).

    At every cap the fresh remainder is the first m-|A_m| eligible entries of
    the independent uniform permutation. Unspent certification capacity goes
    to SRS. Each larger integer cap adds exactly one new queried criterion.
    """
    M = len(uniform)
    if len(set(map(int, uniform))) != M or set(map(int, uniform)) != set(range(M)):
        raise ValueError("uniform order must be a population permutation")
    if len(set(map(int, serial))) != len(serial) or np.any(serial < 0) or np.any(serial >= M):
        raise ValueError("invalid serial transcript")
    in_a, in_r = np.zeros(M, bool), np.zeros(M, bool)
    indices, counts = [], np.zeros(M + 1, dtype=np.int64)
    a_count = r_count = pointer = 0
    for cap in range(1, M + 1):
        target_a = min(cap // 2, len(serial))
        if a_count < target_a:
            index = int(serial[a_count])
            in_a[index] = True
            if in_r[index]:
                in_r[index] = False
                r_count -= 1
            else:
                indices.append(index)
            a_count += 1
        target_r = cap - a_count
        while r_count < target_r:
            index = int(uniform[pointer]); pointer += 1
            if in_a[index]:
                continue
            in_r[index] = True
            indices.append(index)
            r_count += 1
        counts[cap] = a_count
        if len(indices) != cap:
            raise AssertionError("mixed query unions must grow by one")
    return np.asarray(indices, dtype=np.int64), counts


def make_transcripts(frame: Frame, seed: int) -> dict[str, dict]:
    orders = candidate_orders(frame, seed)
    serial = serial_transcript(frame, orders["serial_candidates"])
    mixed, phase_a = mixed_transcript(serial, orders["srs"])
    result = {
        "srs": dict(indices=orders["srs"], phase_a=np.zeros(frame.M + 1, dtype=int)),
        "mix50": dict(indices=mixed, phase_a=phase_a, adaptive_indices=serial),
        "serial": dict(indices=serial),
    }
    if "disagreement" in orders:
        result["disagreement"] = dict(indices=orders["disagreement"])
    return result


def _interval_counts(start, end, length):
    start, end = np.asarray(start, dtype=int), np.asarray(end, dtype=int)
    valid = (start <= end) & (start <= length) & (end >= 0)
    start, end = np.maximum(start[valid], 0), np.minimum(end[valid], length)
    delta = np.bincount(start, minlength=length + 2) - np.bincount(end + 1, minlength=length + 2)
    return np.cumsum(delta[:-1])


def prefix_accounting(frame: Frame, indices: np.ndarray) -> dict[str, np.ndarray]:
    """All queried-set outcomes in O(M), independently of selection logic.

    Eager PASS occupies at most one interval along any fixed transcript. Its
    lower endpoint is when all erroneous P-FAIL bits have been replaced; its
    upper endpoint precedes the first discovered reference FAIL. A shared
    P/R FAIL prevents eager PASS throughout. Difference arrays aggregate these
    intervals without looping over budgets or inspecting labels in selection.
    """
    indices = np.asarray(indices, dtype=np.int64)
    L = len(indices)
    if len(np.unique(indices)) != L or np.any(indices < 0) or np.any(indices >= frame.M):
        raise ValueError("invalid/repeated transcript")
    ranks = np.full(frame.M, L + 1, dtype=np.int64)
    ranks[indices] = np.arange(1, L + 1)
    red_min = lambda x: np.minimum.reduceat(x, frame.offsets)
    red_max = lambda x: np.maximum.reduceat(x, frame.offsets)
    truth = red_min(frame.reference) == 1
    initial = red_min(frame.p) == 1
    wrong = truth != initial
    first = red_min(ranks)
    cert = np.where(truth, red_max(ranks),
                    red_min(np.where(frame.reference == 0, ranks, L + 1)))
    shared_fail = red_max((frame.p == 0) & (frame.reference == 0))
    left = red_max(np.where((frame.p == 0) & (frame.reference == 1), ranks, 0))
    left = np.where(shared_fail, L + 1, left)
    right = red_min(np.where((frame.p == 1) & (frame.reference == 0), ranks, L + 1)) - 1
    end = np.full(frame.N, L, dtype=int)
    counts = {
        "C": _interval_counts(cert, end, L),
        "CW": _interval_counts(cert[wrong], end[wrong], L),
        "certified_pass": _interval_counts(cert[truth], end[truth], L),
        "D": _interval_counts(left[truth & ~initial], cert[truth & ~initial] - 1, L),
        "A": _interval_counts(left[~truth & ~initial], right[~truth & ~initial], L),
        "eager_fp": _interval_counts(left[~truth], right[~truth], L),
        "eager_ff": _interval_counts(np.zeros(int(truth.sum()), int), left[truth] - 1, L),
        "U": _interval_counts(first, cert - 1, L),
        "tasks_touched": _interval_counts(first, end, L),
    }
    counts["certified_fail"] = counts["C"] - counts["certified_pass"]
    counts["E0"] = np.full(L + 1, int(wrong.sum()), dtype=int)
    counts["eager_errors"] = counts["eager_fp"] + counts["eager_ff"]
    counts["gate_errors"] = counts["E0"] - counts["CW"]
    counts["gate_fp"] = int((~truth & initial).sum()) - _interval_counts(
        cert[~truth & initial], end[~truth & initial], L)
    counts["gate_ff"] = counts["gate_errors"] - counts["gate_fp"]
    counts["update_delta"] = counts["gate_errors"] - counts["eager_errors"]
    counts["correct_queried"] = np.r_[0, np.cumsum(frame.p[indices] == frame.reference[indices])]
    counts["criterion_errors_found"] = np.arange(L + 1) - counts["correct_queried"]
    if not np.array_equal(counts["update_delta"], counts["D"] - counts["A"]):
        raise AssertionError("update identity failed")
    if not np.array_equal(counts["eager_errors"], counts["E0"] - counts["CW"] - counts["D"] + counts["A"]):
        raise AssertionError("allocation identity failed")
    if np.any(np.abs(counts["update_delta"]) > counts["U"]):
        raise AssertionError("partial-task bound failed")
    return counts


def _accuracy(M, a_count, a_correct, r_count, r_correct, exact_interval):
    remaining = M - a_count
    estimate = ((a_correct + remaining * r_correct / r_count) / M
                if r_count else a_correct / M if not remaining else None)
    result = dict(accuracy_estimate=estimate, accuracy_ci_lower=None,
                  accuracy_ci_upper=None, accuracy_ci_width=None)
    if exact_interval:
        lo, hi = hypergeom_ci(remaining, r_count, r_correct)
        result.update(accuracy_ci_lower=(a_correct + lo) / M,
                      accuracy_ci_upper=(a_correct + hi) / M,
                      accuracy_ci_width=(hi - lo) / M)
    return result


def rows_from_transcript(frame, transcript, policy, seed, shares=GRID,
                         interval_shares=(.2,), acquisition_reference=None,
                         explicit_caps=()):
    indices = transcript["indices"]
    counts = prefix_accounting(frame, indices)
    acquisition_reference = acquisition_reference or frame.reference_key
    same_reference = acquisition_reference == frame.reference_key
    if policy == "serial" and same_reference:
        if np.any(np.abs(counts["update_delta"]) > 1):
            raise AssertionError("serial update difference exceeds one")
        if counts["C"][-1] != frame.N or counts["eager_errors"][-1] or counts["gate_errors"][-1]:
            raise AssertionError("completed serial transcript must certify all tasks")
    if len(indices) == frame.M and (counts["eager_errors"][-1] or counts["gate_errors"][-1]):
        raise AssertionError("census errors must be zero")
    if policy == "mix50":
        adaptive = transcript["adaptive_indices"]
        a_correct = np.r_[0, np.cumsum(frame.p[adaptive] == frame.reference[adaptive])]
    else:
        a_correct = None
    rows = []
    points = [(float(b), int(round(float(b) * frame.M)), False) for b in shares]
    points += [(min(int(m), frame.M) / frame.M, min(int(m), frame.M), True) for m in explicit_caps]
    for share, cap, explicit in points:
        if not 0 <= cap <= frame.M:
            raise ValueError("budget outside population")
        used = min(cap, len(indices))
        row = {name: int(values[used]) for name, values in counts.items()}
        correct = row.pop("correct_queried")
        row.update(producer=frame.units[0]["producer"], reference=frame.reference_key,
                   acquisition_reference=acquisition_reference, policy=policy, seed=int(seed),
                   budget_share=share, budget_cap=cap, explicit_cap=explicit,
                   actual_queries=used, unused_budget=cap-used, n_units=frame.N,
                   n_criteria=frame.M, accuracy_truth=float(np.mean(frame.p == frame.reference)),
                   accuracy_logical_lower=correct/frame.M,
                   accuracy_logical_upper=(correct+frame.M-used)/frame.M,
                   task_pass_logical_lower=row["certified_pass"]/frame.N,
                   task_pass_logical_upper=1-row["certified_fail"]/frame.N)
        if policy in ("srs", "mix50"):
            a = int(transcript["phase_a"][cap])
            ac = int(a_correct[a]) if a_correct is not None else 0
            r, rc = used-a, correct-ac
            calculate_ci = explicit or any(abs(share-b) < 1e-12 for b in interval_shares)
            row.update(stage_a_queries=a, stage_srs_queries=r,
                       inference_kind="exact_conditional_srs" if calculate_ci else "srs_point_estimate_only",
                       **_accuracy(frame.M, a, ac, r, rc, calculate_ci))
        else:
            row.update(stage_a_queries=used, stage_srs_queries=0,
                       inference_kind="logical_bounds_only", accuracy_estimate=None,
                       accuracy_ci_lower=None, accuracy_ci_upper=None, accuracy_ci_width=None)
        rows.append(row)
    return rows


def replay_cell(units, reference_key="r", *, evaluation_reference_keys=None,
                seeds=SEEDS, shares=GRID, policies=POLICIES,
                interval_shares=(.2,), explicit_caps=(), include_transcripts=False):
    """Reacquire under reference_key; optionally reread identical queries under R2.

    Crossed evaluation never reruns selection. A serial bound is checked only
    for the same reference that generated stopping decisions. Query-order
    variability is not a task-composition confidence interval.
    """
    shares, policies = tuple(shares), tuple(policies)
    interval_shares, explicit_caps = tuple(interval_shares), tuple(explicit_caps)
    acquisition = prepare_frame(units, reference_key)
    evaluations = {key: prepare_frame(acquisition.units, key)
                   for key in (evaluation_reference_keys or (reference_key,))}
    rows, receipts = [], []
    for seed in seeds:
        transcripts = make_transcripts(acquisition, int(seed))
        for policy in policies:
            if policy not in POLICIES:
                raise ValueError("unknown policy")
            if policy not in transcripts:
                raise ValueError("disagreement continuation requires an observed auxiliary Q")
            trace = transcripts[policy]
            for frame in evaluations.values():
                rows.extend(rows_from_transcript(frame, trace, policy, seed, shares,
                            interval_shares, reference_key, explicit_caps))
            if include_transcripts:
                ids = [acquisition.criterion_ids[i] for i in trace["indices"]]
                receipts.append(dict(producer=acquisition.units[0]["producer"],
                                     acquisition_reference=reference_key, policy=policy, seed=int(seed),
                                     query_ids=ids,
                                     sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()))
    return dict(rows=rows, transcripts=receipts)


def allocation_forecasts(pilot_units, *, references=("r", "r2"), seeds=SEEDS,
                         minimum_clusters=20):
    """Fit both references before confirmation unmasking; return JSON-safe forecasts.

    Outputs are per 100 confirmation units and can be scaled without reading
    confirmation reference labels. The certificate-only control uses pilot
    predicted certificate change, never realized confirmation certification.
    """
    pilot_units, references, seeds = list(pilot_units), tuple(references), tuple(seeds)
    if not seeds:
        raise ValueError("at least one priority seed required")
    groups = defaultdict(list)
    for unit in pilot_units:
        groups[unit["producer"]].append(unit)
    if not groups:
        raise ValueError("empty pilot")
    predictions, endpoints = [], []
    for reference in references:
        cell_fits = {}
        for producer, units in sorted(groups.items()):
            rows = replay_cell(units, reference, seeds=seeds, shares=(.2,),
                               policies=("srs", "mix50", "serial"), interval_shares=())["rows"]
            endpoints.extend(rows)
            by_seed = defaultdict(dict)
            for row in rows:
                by_seed[row["seed"]][row["policy"]] = row
            deltas = {name: float(np.mean([p["mix50"][name] - p["srs"][name]
                                           for p in by_seed.values()])) for name in ALLOCATION_FIELDS}
            sample = rows[0]
            cell_fits[producer] = dict(n_units=len(units), E0=sample["E0"],
                                      clusters=len({u["cluster_id"] for u in units}), deltas=deltas)
        for producer, fit in cell_fits.items():
            pooled = fit["clusters"] < minimum_clusters
            source = list(cell_fits.values()) if pooled else [fit]
            n = sum(x["n_units"] for x in source)
            delta = {name: 100*sum(x["deltas"][name] for x in source)/n for name in ALLOCATION_FIELDS}
            error_fraction = sum(x["E0"] for x in source)/n
            predictions.append(dict(producer=producer, reference=reference, budget_share=.2,
                pilot_units=n, pilot_unique_clusters=len({u["cluster_id"] for u in pilot_units}) if pooled else fit["clusters"],
                pooling="all_producers" if pooled else "producer", seeds=len(seeds),
                **{"delta_"+key+"_per100": value for key, value in delta.items()},
                repair_forecast_per100=-delta["CW"]-delta["D"]+delta["A"],
                certificate_only_forecast_per100=-error_fraction*delta["C"],
                pilot_initial_error_fraction=error_fraction))
    return dict(schema_version="wildifeval-allocation-v1", predictions=predictions,
                pilot_endpoint_rows=endpoints)


def no_pilot_controls(confirmation_units, pilot_units, *, reference_key="r",
                      shares=(.2, .5, .8), seeds=SEEDS, policies=POLICIES):
    """Confirmation-only controls at the same charged pilot+confirmation cost.

    Budgets are per producer; no-pilot spend is capped at its confirmation
    census. Both requested charged cost and actual queries are retained.
    """
    shares, seeds, policies = tuple(shares), tuple(seeds), tuple(policies)
    groups, pilot_cost = defaultdict(list), defaultdict(int)
    for unit in confirmation_units:
        groups[unit["producer"]].append(unit)
    for unit in pilot_units:
        pilot_cost[unit["producer"]] += len(unit["p"])
    output = []
    for producer, units in sorted(groups.items()):
        M = sum(len(u["p"]) for u in units)
        for share in shares:
            charged = pilot_cost[producer] + round(share*M)
            rows = replay_cell(units, reference_key, seeds=seeds, shares=(), policies=policies,
                               explicit_caps=(min(charged, M),))["rows"]
            for row in rows:
                row.update(control="no_pilot_same_charged_budget", focal_confirmation_share=share,
                           pilot_reference_bits=pilot_cost[producer], requested_total_cost=charged,
                           no_pilot_cap=min(charged, M), cost_capped_at_census=charged > M)
            output.extend(rows)
    return output
