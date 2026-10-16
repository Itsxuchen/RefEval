"""Outcome-blind checks: all inputs are constructed, never released Wild labels."""
from copy import deepcopy
import itertools
import json
import math

import numpy as np
import pytest

from src.wildifeval_forecast import (
    compare_forecasts, event_probability, exact_frame_outcomes, exact_task_events,
    fit_forecaster, independent_margin_events, predict_frame, select_pool, support_manifest,
)


def unit(uid, p, r=None, producer="a", cluster=None):
    return {"unit_id": uid, "producer": producer, "cluster_id": cluster or uid,
            "k": len(p), "p": tuple(p), **({"r": tuple(r)} if r is not None else {})}


def literal_events(p, r, queried):
    initial, truth = all(p), all(r)
    eager = all(r[j] if j in queried else p[j] for j in range(len(p)))
    certified = all(j in queried for j in range(len(p))) or any(j in queried and not r[j] for j in range(len(p)))
    gate = truth if certified else initial
    A = int(not truth and not initial and eager)
    D = int(truth and not initial and eager and not certified)
    assert int(gate != truth) - int(eager != truth) == D - A
    return A, D


def test_exact_events_against_every_small_binary_pattern_and_query_subset():
    for k in range(1, 5):
        M = k + 2  # Include unrelated criteria: drawing is global, not task-local.
        for p in itertools.product((0, 1), repeat=k):
            for r in itertools.product((0, 1), repeat=k):
                for m in range(M + 1):
                    enumerated = [literal_events(p, r, set(q)) for q in itertools.combinations(range(M), m)]
                    expected = np.mean(enumerated, axis=0)
                    actual = exact_task_events(p, r, M, m)
                    assert actual["A"] == pytest.approx(expected[0], abs=1e-13)
                    assert actual["D"] == pytest.approx(expected[1], abs=1e-13)
                    assert actual["delta"] == pytest.approx(expected[1] - expected[0], abs=1e-13)


def test_probability_against_combinatorial_ratio_and_input_validation():
    for M in range(8):
        for m in range(M + 1):
            for r in range(M + 1):
                for f in range(M - r + 1):
                    possible = 0 <= m-r <= M-r-f
                    exact = math.comb(M-r-f, m-r) / math.comb(M,m) if possible else 0
                    assert event_probability(M,m,r,f) == pytest.approx(exact, abs=1e-13)
    with pytest.raises(ValueError):
        event_probability(5,6,1,1)
    with pytest.raises(ValueError):
        event_probability(5,2.5,1,1)
    with pytest.raises(ValueError):
        exact_task_events([0,1],[1],10,3)


@pytest.mark.parametrize("qpass,qfail", [(0,0),(0,1),(1,0),(1,1),(.23,.61)])
def test_independent_margin_collapsing_matches_full_reference_enumeration(qpass,qfail):
    for k in range(1,9):
        for pf in range(k+1):
            p = (0,)*pf + (1,)*(k-pf)
            for m in (0,3,7,10):
                A = D = 0.
                for r in itertools.product((0,1),repeat=k):
                    probability = math.prod((qfail if pp == 0 else qpass) if rr == 0 else
                                            (1-qfail if pp == 0 else 1-qpass) for pp,rr in zip(p,r))
                    events = exact_task_events(p,r,10,m)
                    A += probability*events["A"]
                    D += probability*events["D"]
                assert independent_margin_events(k,pf,qpass,qfail,10,m) == pytest.approx((A,D),abs=1e-13)


def test_pools_count_unique_clusters_and_both_structural_zeros_are_excluded():
    pilot = [unit(f"u{i}",[0,1],[1,1],cluster="same") for i in range(25)]
    pilot += [unit(f"v{i}",[0,1],[1,0],producer="b") for i in range(20)]
    pilot += [unit("zero-pass",[1,1],[0,0]),unit("zero-fail",[0,0],[1,1])]
    fit = fit_forecaster(pilot)
    pool, support = select_pool(fit,"a",2,1)
    assert support["pool_level"] == "k_pf"
    assert support["n_clusters"] == 21
    assert pool["n_units"] == 45
    assert fit["pilot_mixed_units"] == 45
    assert fit["pilot_structural_zero_units"] == 2
    # Jeffreys counts use criterion counts, not an average of unit proportions.
    assert pool["p_ref_fail_given_pred_pass"] == pytest.approx(20.5/46)
    assert pool["p_ref_fail_given_pred_fail"] == pytest.approx(.5/46)
    for pf in (0,2):
        assert select_pool(fit,"a",2,pf)[0] is None
    for row in predict_frame(fit,[unit("a",[0,0]),unit("b",[1,1])]):
        assert row["forecast_per100"] == 0


def test_fallback_and_no_support_are_explicit():
    fit = fit_forecaster([unit(f"u{i}",[0,1,1],[1,1,1]) for i in range(20)])
    assert select_pool(fit,"new",3,2)[1]["pool_level"] == "k_any_pf"
    assert select_pool(fit,"new",4,1)[1]["pool_level"] == "all"
    small = fit_forecaster([unit("u",[0,1],[1,0])])
    assert select_pool(small,"a",2,1)[1]["below_minimum"]
    no_mixed = fit_forecaster([unit("u",[0,0],[1,1])])
    with pytest.raises(ValueError,match="no mixed-P pilot support"):
        predict_frame(no_mixed,[unit("v",[0,1])])
    # Structural-zero targets do not invent or need reference-pattern support.
    assert all(row["forecast_per100"] == 0 for row in predict_frame(no_mixed,[unit("v",[1,1])]))


class PoisonReferences(dict):
    def __getitem__(self,key):
        if key in {"r","r2","q"}:
            raise AssertionError("Unlabeled forecast tried to read a hidden reference or auxiliary label")
        return super().__getitem__(key)

    def values(self):
        raise AssertionError("Do not traverse hidden dictionary values")

    def items(self):
        raise AssertionError("Do not traverse hidden dictionary items")


def test_prediction_reads_only_observable_fields_and_json_fit_roundtrip():
    pilot = [unit(f"u{i}",[0,1],[1,1] if i<10 else [1,0]) for i in range(20)]
    fit = fit_forecaster(pilot)
    metadata = [PoisonReferences(unit("a",[0,1],[0,0])),PoisonReferences(unit("b",[0,1],[1,1]))]
    original = deepcopy(fit)
    result = predict_frame(fit,metadata)
    assert result == predict_frame(json.loads(json.dumps(fit)),metadata)
    assert fit == original
    assert len(support_manifest(fit,metadata)) == 1


def test_pattern_preserving_training_reproduces_exact_reference_frame():
    pilot = [unit("a",[0,1],[1,1]),unit("b",[0,1],[1,0])]
    target = [unit("c",[0,1],[1,1]),unit("d",[0,1],[1,0])]
    fit = fit_forecaster(pilot,min_clusters=1)
    predicted = [r for r in predict_frame(fit,target) if r["predictor"] == "task_linked"]
    observed = exact_frame_outcomes(target)
    for p,r in zip(predicted,observed):
        assert p["forecast_per100"] == pytest.approx(r["observed_per100"])
    rows = support_manifest(fit,target)
    assert rows[0]["pool_level"] == "producer_k_pf"


def test_per_producer_budgets_use_each_cell_criterion_frame():
    pilot = [unit("p",[0,1],[1,1]),unit("q",[0,1,1],[1,0,1],producer="b")]
    target = [unit("v",[0,1]),unit("w",[0,1,1],producer="b"),unit("x",[0,1,1],producer="b")]
    fit = fit_forecaster(pilot,min_clusters=1)
    rows = predict_frame(fit,target,shares=(.2,))
    assert {(r["producer"],r["M"],r["budget_cap"]) for r in rows} == {("a",2,0),("b",6,1)}


def test_bootstrap_holds_pilot_fixed_resamples_clusters_jointly_and_recomputes_M():
    pilot = [unit(f"p{i}",[0,1],[1,1] if i<10 else [1,0]) for i in range(20)]
    units=[]
    for producer in ("a","b"):
        units.extend([unit(f"{producer}c1",[0,1],[1,1],producer,"c1"),
                      unit(f"{producer}c2",[0,1,1],[1,0,1],producer,"c2"),
                      unit(f"{producer}c3",[0,1,1,1],[1,1,1,1],producer,"c3")])
    fit = fit_forecaster(pilot,min_clusters=1)
    original = deepcopy(fit)
    result = compare_forecasts(fit,units,replicates=31)
    assert fit == original
    assert result == compare_forecasts(fit,units,replicates=31)
    counts = [r["criterion_counts"] for r in result["bootstrap_rows"]]
    assert all(c["a"] == c["b"] for c in counts)
    assert len({c["a"] for c in counts}) > 1
    assert result["summary"]["n_confirmation_clusters"] == 3
    assert len(result["point_rows"]) == 2*3*4
    assert result["summary"]["d_ci_low"] <= result["summary"]["d_ci_high"]
    assert result["summary"]["d_per100"] == pytest.approx(
        result["summary"]["mae_per100"]["task_linked"]-result["summary"]["mae_per100"]["matched_margin"])
    with pytest.raises(ValueError,match="reference roles differ"):
        compare_forecasts(fit,units,reference_key="r2",replicates=2)


def test_alternative_reference_is_fitted_separately_before_prediction():
    pilot = [dict(unit(f"u{i}",[0,1],[1,1]),r2=(1,0)) for i in range(20)]
    target = [unit("v",[0,1]),unit("w",[0,1])]
    primary = predict_frame(fit_forecaster(pilot,"r"),target)
    alternative = predict_frame(fit_forecaster(pilot,"r2"),target)
    get = lambda rows: [r["forecast_per100"] for r in rows if r["predictor"]=="task_linked"]
    assert get(primary) != get(alternative)
    assert all(x>=0 for x in get(primary))
    assert all(x<=0 for x in get(alternative))
