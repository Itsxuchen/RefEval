"""Preparatory pilot forecasts for a prospective WildIFEval validation.

No empirical design freeze or external forecast has been executed; the masked
preflight failed. This module has been tested only on synthetic inputs.

This module has no file, network, or model access. ``fit_forecaster`` consumes
pilot references; ``predict_frame`` reads only observable confirmation metadata.
The expectation is exact over a uniform fixed-size criterion query set. The
bootstrap conditions on the fitted pilot and resamples confirmation clusters.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import math
from typing import Iterable, Mapping

import numpy as np


PREDICTORS = ("task_linked", "matched_margin", "producer_margin", "zero")
DEFAULT_SHARES = (.2, .5, .8)


def _bits(values: Iterable[int]) -> tuple[int, ...]:
    bits = tuple(values)
    if not bits or any(x not in (0, 1) for x in bits):
        raise ValueError("A nonempty vector of binary labels is required")
    return tuple(int(x) for x in bits)


@lru_cache(maxsize=300000)
def event_probability(M: int, m: int, required: int, forbidden: int) -> float:
    """Probability that a uniform m-of-M subset includes/excludes given items."""
    if any(not isinstance(x, (int, np.integer)) for x in (M, m, required, forbidden)):
        raise ValueError("Counts must be integers")
    if min(M, m, required, forbidden) < 0 or m > M:
        raise ValueError("Invalid finite-population query counts")
    if required + forbidden > M or required > m or forbidden > M - m:
        return 0.0
    value = 1.0
    for j in range(required):
        value *= (m - j) / (M - j)
    for j in range(forbidden):
        value *= (M - m - j) / (M - required - j)
    return value


def pattern_counts(prediction: Iterable[int], reference: Iterable[int]) -> tuple[int, int, int, int]:
    p, r = _bits(prediction), _bits(reference)
    if len(p) != len(r):
        raise ValueError("Prediction and reference lengths differ")
    return len(p), p.count(0), r.count(0), sum(a == b == 0 for a, b in zip(p, r))


@lru_cache(maxsize=300000)
def events_from_counts(k: int, pf: int, rf: int, common_fail: int,
                       M: int, m: int) -> tuple[float, float]:
    """Return (A, D): introduced false passes and delayed false-fail repairs."""
    if not (1 <= k <= M and 0 <= pf <= k and 0 <= rf <= k
            and max(0, pf + rf - k) <= common_fail <= min(pf, rf)):
        raise ValueError("Invalid task pattern counts")
    event_probability(M, m, 0, 0)  # Validate even structural zero cases.
    if pf == 0 or pf == k:
        return 0.0, 0.0
    A = event_probability(M, m, pf, rf) if rf > 0 and common_fail == 0 else 0.0
    D = (event_probability(M, m, pf, 0) - event_probability(M, m, k, 0)) if rf == 0 else 0.0
    if D < -1e-12:
        raise ArithmeticError("Negative delayed-repair probability")
    return A, max(0.0, D)


def exact_task_events(prediction: Iterable[int], reference: Iterable[int], M: int, m: int) -> dict:
    A, D = events_from_counts(*pattern_counts(prediction, reference), M, m)
    return {"A": A, "D": D, "delta": D - A}


def _observable(unit: Mapping) -> tuple[str, str, int, int]:
    # Intentionally never iterate through all mapping values: callers may use a
    # restricted mapping whose reference access raises an exception.
    p = _bits(unit["p"])
    k = int(unit["k"])
    if k != len(p):
        raise ValueError("Metadata k does not equal prediction vector length")
    return str(unit["producer"]), str(unit["cluster_id"]), k, p.count(0)


def _key(values: Iterable) -> str:
    import json
    return json.dumps(list(values), separators=(",", ":"))


def _pool_record(rows: list[dict]) -> dict:
    patterns = Counter(tuple(row["pattern"]) for row in rows)
    pp_total = sum(row["pattern"][0] - row["pattern"][1] for row in rows)
    pf_total = sum(row["pattern"][1] for row in rows)
    pp_rfail = sum(row["pattern"][2] - row["pattern"][3] for row in rows)
    pf_rfail = sum(row["pattern"][3] for row in rows)
    return {
        "n_units": len(rows),
        "n_clusters": len({row["cluster"] for row in rows}),
        "patterns": [dict(k=k, pf=pf, rf=rf, common_fail=cf, count=count)
                     for (k, pf, rf, cf), count in sorted(patterns.items())],
        "p_ref_fail_given_pred_pass": (pp_rfail + .5) / (pp_total + 1),
        "p_ref_fail_given_pred_fail": (pf_rfail + .5) / (pf_total + 1),
        "margin_counts": {"pred_pass": pp_total, "pred_fail": pf_total,
                          "ref_fail_pred_pass": pp_rfail, "ref_fail_pred_fail": pf_rfail},
    }


def fit_forecaster(pilot_units: Iterable[Mapping], reference_key: str = "r",
                   min_clusters: int = 20) -> dict:
    """Fit serializable donor pools from pilot only, omitting structural zeros.

    Fallback sequence: producer/k/pf; k/pf; k/any-pf; k; all mixed-P pilot.
    The third and fourth levels coincide because fitting excludes pf=0 and pf=k;
    both are retained explicitly to expose that predeclared design property.
    """
    if min_clusters < 1:
        raise ValueError("min_clusters must be positive")
    grouped: dict[str, list[dict]] = defaultdict(list)
    producers, unit_ids = set(), set()
    total = structural = 0
    for unit in pilot_units:
        producer, cluster, k, pf = _observable(unit)
        uid = str(unit["unit_id"])
        if uid in unit_ids:
            raise ValueError("Duplicate pilot unit_id")
        unit_ids.add(uid)
        producers.add(producer)
        pattern = pattern_counts(unit["p"], unit[reference_key])
        total += 1
        if pf in (0, k):
            structural += 1
            continue
        row = {"cluster": cluster, "pattern": pattern}
        keys = (("producer_k_pf", producer, k, pf), ("k_pf", k, pf),
                ("k_any_pf", k, True), ("k", k), ("all",), ("producer", producer))
        for key in keys:
            grouped[_key(key)].append(row)
    pools = {key: _pool_record(rows) for key, rows in sorted(grouped.items())}
    return {"schema_version": 1, "reference_key": reference_key,
            "min_clusters": int(min_clusters), "pilot_units": total,
            "pilot_structural_zero_units": structural, "pilot_mixed_units": total - structural,
            "producers": sorted(producers), "pools": pools,
            "pseudocount": .5,
            "fit_rule": "fixed mixed-P pilot pools; no confirmation references"}


def select_pool(fit: Mapping, producer: str, k: int, pf: int,
                producer_only: bool = False) -> tuple[dict | None, dict]:
    """Select a fixed donor pool and return an auditable support description."""
    if pf in (0, k):
        return None, {"pool_level": "structural_zero", "pool_key": None,
                      "n_clusters": 0, "n_units": 0, "below_minimum": False}
    keys = (("producer", producer), ("all",)) if producer_only else (
        ("producer_k_pf", producer, k, pf), ("k_pf", k, pf),
        ("k_any_pf", k, True), ("k", k), ("all",))
    for index, key in enumerate(keys):
        pool = fit["pools"].get(_key(key))
        if pool is not None and (pool["n_clusters"] >= fit["min_clusters"] or index == len(keys) - 1):
            return pool, {"pool_level": key[0], "pool_key": _key(key),
                          "n_clusters": pool["n_clusters"], "n_units": pool["n_units"],
                          "below_minimum": pool["n_clusters"] < fit["min_clusters"]}
    raise ValueError("Mixed-P confirmation target has no mixed-P pilot support; stop primary analysis")


def support_manifest(fit: Mapping, confirmation_metadata: Iterable[Mapping]) -> list[dict]:
    counts = Counter((producer, k, pf) for producer, _, k, pf in map(_observable, confirmation_metadata))
    return [{"producer": producer, "k": k, "predicted_failures": pf, "confirmation_units": n,
             **select_pool(fit, producer, k, pf)[1],
             "producer_margin_support": select_pool(fit, producer, k, pf, True)[1]}
            for (producer, k, pf), n in sorted(counts.items())]


def _linked_events(pool: Mapping, M: int, m: int) -> tuple[float, float]:
    A = D = 0.0
    for p in pool["patterns"]:
        a, d = events_from_counts(p["k"], p["pf"], p["rf"], p["common_fail"], M, m)
        A += p["count"] * a
        D += p["count"] * d
    return A / pool["n_units"], D / pool["n_units"]


def independent_margin_events(k: int, pf: int, q_pass: float, q_fail: float,
                              M: int, m: int) -> tuple[float, float]:
    """Enumerate independent R-pattern probabilities, collapsed by symmetry.

    A requires no R failures among P failures and at least one among P passes.
    Enumerating the binomial count g is exactly equivalent to enumerating all
    2**k bit patterns; tests independently enumerate those bit patterns.
    """
    if not (0 <= q_pass <= 1 and 0 <= q_fail <= 1 and 0 <= pf <= k <= M):
        raise ValueError("Invalid independent-margin parameters")
    event_probability(M, m, 0, 0)
    if pf in (0, k):
        return 0.0, 0.0
    pp = k - pf
    all_pred_fails_ref_pass = (1 - q_fail) ** pf
    ref_all_pass = all_pred_fails_ref_pass * (1 - q_pass) ** pp
    D = ref_all_pass * (event_probability(M, m, pf, 0) - event_probability(M, m, k, 0))
    A = all_pred_fails_ref_pass * sum(
        math.comb(pp, rf) * q_pass ** rf * (1 - q_pass) ** (pp - rf)
        * event_probability(M, m, pf, rf) for rf in range(1, pp + 1))
    return A, max(0.0, D)


def _frame_counts(units: Iterable[Mapping], reference_key: str | None = None,
                  weights: Mapping[str, int] | None = None) -> tuple[dict, dict]:
    observables: dict[str, Counter] = defaultdict(Counter)
    actual: dict[str, Counter] = defaultdict(Counter)
    for unit in units:
        producer, cluster, k, pf = _observable(unit)
        weight = 1 if weights is None else int(weights.get(cluster, 0))
        if weight < 0:
            raise ValueError("Cluster weights must be nonnegative")
        if not weight:
            continue
        observables[producer][(k, pf)] += weight
        if reference_key is not None:
            actual[producer][pattern_counts(unit["p"], unit[reference_key])] += weight
    return dict(observables), dict(actual)


def _predict_counts(fit: Mapping, observables: Mapping, shares: Iterable[float]) -> list[dict]:
    rows = []
    for producer, counts in sorted(observables.items()):
        n = sum(counts.values())
        M = sum(k * count for (k, _), count in counts.items())
        for share in shares:
            if not 0 <= share <= 1:
                raise ValueError("Budget share outside [0,1]")
            m = int(round(float(share) * M))
            totals = {name: [0.0, 0.0] for name in PREDICTORS}
            for (k, pf), count in counts.items():
                if pf in (0, k):
                    continue
                pool, _ = select_pool(fit, producer, k, pf)
                coarse, _ = select_pool(fit, producer, k, pf, True)
                linked = _linked_events(pool, M, m)
                margin = independent_margin_events(k, pf, pool["p_ref_fail_given_pred_pass"],
                                                    pool["p_ref_fail_given_pred_fail"], M, m)
                simple = independent_margin_events(k, pf, coarse["p_ref_fail_given_pred_pass"],
                                                    coarse["p_ref_fail_given_pred_fail"], M, m)
                for name, (A, D) in zip(PREDICTORS[:3], (linked, margin, simple)):
                    totals[name][0] += count * A
                    totals[name][1] += count * D
            for predictor, (A, D) in totals.items():
                rows.append({"producer": producer, "budget_share": float(share), "budget_cap": m,
                             "M": M, "n_units": n, "predictor": predictor,
                             "expected_A_per100": 100 * A / n,
                             "expected_D_per100": 100 * D / n,
                             "forecast_per100": 100 * (D - A) / n})
    return rows


def predict_frame(fit: Mapping, confirmation_metadata: Iterable[Mapping],
                  shares: Iterable[float] = DEFAULT_SHARES) -> list[dict]:
    """Forecast from fixed pilot fit and unlabeled target P/k/producer metadata."""
    observable, _ = _frame_counts(confirmation_metadata)
    return _predict_counts(fit, observable, tuple(shares))


def _outcome_counts(actual: Mapping, shares: Iterable[float]) -> list[dict]:
    rows = []
    for producer, counts in sorted(actual.items()):
        n = sum(counts.values())
        M = sum(pattern[0] * count for pattern, count in counts.items())
        for share in shares:
            if not 0 <= share <= 1:
                raise ValueError("Budget share outside [0,1]")
            m = int(round(float(share) * M))
            A = D = 0.0
            for pattern, count in counts.items():
                a, d = events_from_counts(*pattern, M, m)
                A += count * a
                D += count * d
            rows.append({"producer": producer, "budget_share": float(share), "budget_cap": m,
                         "M": M, "n_units": n, "expected_A_per100": 100 * A / n,
                         "expected_D_per100": 100 * D / n, "observed_per100": 100 * (D - A) / n})
    return rows


def exact_frame_outcomes(units: Iterable[Mapping], reference_key: str = "r",
                         shares: Iterable[float] = DEFAULT_SHARES) -> list[dict]:
    _, actual = _frame_counts(units, reference_key)
    return _outcome_counts(actual, tuple(shares))


def _comparison(predictions: list[dict], outcomes: list[dict]) -> tuple[list[dict], dict]:
    observed = {(row["producer"], row["budget_share"]): row for row in outcomes}
    rows, errors = [], defaultdict(list)
    for pred in predictions:
        actual = observed[(pred["producer"], pred["budget_share"])]
        error = pred["forecast_per100"] - actual["observed_per100"]
        rows.append({**pred, "observed_per100": actual["observed_per100"],
                     "forecast_error_per100": error, "absolute_error_per100": abs(error)})
        errors[pred["predictor"]].append(abs(error))
    maes = {name: float(np.mean(values)) for name, values in errors.items()}
    return rows, {"mae_per100": maes, "d_per100": maes["task_linked"] - maes["matched_margin"]}


def compare_forecasts(fit: Mapping, confirmation_units: Iterable[Mapping], reference_key: str = "r",
                      shares: Iterable[float] = DEFAULT_SHARES, replicates: int = 999,
                      seed: int = 20261010) -> dict:
    """Score frozen forecasts and paired whole-cluster composition intervals.

    Fitted pilot parameters never change. Each resampled frame receives newly
    computed M, m, predictions, and exact outcomes. Producers are jointly
    resampled; the point statistic equally weights producer-by-budget cells.
    """
    if fit["reference_key"] != reference_key:
        raise ValueError("Fit and outcome reference roles differ")
    if replicates < 1:
        raise ValueError("At least one bootstrap replicate is required")
    units, shares = list(confirmation_units), tuple(shares)
    if not units:
        raise ValueError("Confirmation frame is empty")
    observable, actual = _frame_counts(units, reference_key)
    point_rows, point_summary = _comparison(_predict_counts(fit, observable, shares), _outcome_counts(actual, shares))
    clusters = sorted({str(unit["cluster_id"]) for unit in units})
    producers = sorted(observable)
    # Pre-aggregate each cluster so bootstrap never reconstructs duplicated text
    # or label rows, and never refits the pilot.
    cluster_observable = {cluster: defaultdict(Counter) for cluster in clusters}
    cluster_actual = {cluster: defaultdict(Counter) for cluster in clusters}
    for unit in units:
        producer, cluster, k, pf = _observable(unit)
        cluster_observable[cluster][producer][(k, pf)] += 1
        cluster_actual[cluster][producer][pattern_counts(unit["p"], unit[reference_key])] += 1
    rng = np.random.default_rng(seed)
    bootstrap, cell_actual = [], defaultdict(list)
    for rep in range(replicates):
        multiplicities = np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))
        obs = {producer: Counter() for producer in producers}
        truth = {producer: Counter() for producer in producers}
        for cluster, weight in zip(clusters, multiplicities):
            if not weight:
                continue
            for producer, counts in cluster_observable[cluster].items():
                obs[producer].update({key: int(weight) * count for key, count in counts.items()})
            for producer, counts in cluster_actual[cluster].items():
                truth[producer].update({key: int(weight) * count for key, count in counts.items()})
        if any(not counts for counts in obs.values()):
            raise ValueError("A cluster resample removed a producer cell; common-frame assumption failed")
        predicted, observed = _predict_counts(fit, obs, shares), _outcome_counts(truth, shares)
        _, scored = _comparison(predicted, observed)
        bootstrap.append({"replicate": rep, "d_per100": scored["d_per100"],
                          **{f"mae_{key}": value for key, value in scored["mae_per100"].items()},
                          "criterion_counts": {producer: sum(k * n for (k, _), n in counts.items())
                                               for producer, counts in obs.items()}})
        for row in observed:
            cell_actual[(row["producer"], row["budget_share"])].append(row["observed_per100"])
    lo, hi = map(float, np.quantile([row["d_per100"] for row in bootstrap], [.025, .975]))
    decision = "predictive_advantage" if point_summary["d_per100"] < 0 and hi < 0 else (
        "predictive_advantage_contradicted" if lo > 0 else "unresolved")
    for row in point_rows:
        qlo, qhi = map(float, np.quantile(cell_actual[(row["producer"], row["budget_share"])], [.025, .975]))
        value, observed = row["forecast_per100"], row["observed_per100"]
        row.update(observed_ci_low=qlo, observed_ci_high=qhi,
                   forecast_magnitude_class="near_zero" if abs(value) < 1 else "material",
                   opposite_sign=bool(abs(value) >= 1 and value * observed < 0),
                   clear_directional_failure=bool(abs(value) >= 1 and ((value > 0 and qhi < 0) or (value < 0 and qlo > 0))))
    return {"reference_key": reference_key, "point_rows": point_rows,
            "summary": {**point_summary, "d_ci_low": lo, "d_ci_high": hi, "decision": decision,
                        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
                        "n_confirmation_clusters": len(clusters),
                        "interval_scope": "conditional confirmation-cluster composition; fixed pilot fit",
                        "quantile_method": "numpy linear percentile 2.5%,97.5%"},
            "bootstrap_rows": bootstrap, "support_manifest": support_manifest(fit, units)}
