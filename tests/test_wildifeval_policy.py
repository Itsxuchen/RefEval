"""Independent brute-force synthetic checks; never load external study labels."""
from itertools import combinations, permutations, product

import numpy as np
import pytest

from src import wildifeval_policy as policy


def unit(p, r, uid="a", *, q=None, r2=None, producer="producer", cluster=None):
    return dict(unit_id=uid, cluster_id=cluster or uid, producer=producer,
                p=list(p), q=list(q if q is not None else p), r=list(r),
                r2=list(r2 if r2 is not None else r),
                criterion_ids=[f"{uid}::{j}" for j in range(len(p))])


def brute(frame, indices):
    seen = set(map(int, indices))
    totals = dict(C=0, CW=0, D=0, A=0, E0=0, eager_fp=0, eager_ff=0,
                  gate_errors=0, gate_fp=0, gate_ff=0, U=0, certified_pass=0,
                  certified_fail=0, tasks_touched=0, criterion_errors_found=0)
    for i, offset in enumerate(frame.offsets):
        slots = list(range(offset, offset + frame.sizes[i]))
        p, r = [int(frame.p[j]) for j in slots], [int(frame.reference[j]) for j in slots]
        q = [j in seen for j in slots]
        truth, initial = all(r), all(p)
        certified = any(s and not v for s, v in zip(q, r)) or all(q)
        eager = all(rv if s else pv for s, rv, pv in zip(q, r, p))
        gate = truth if certified else initial
        vals = dict(C=certified, CW=certified and initial != truth,
                    D=truth and not initial and eager and not certified,
                    A=not truth and not initial and eager,
                    E0=initial != truth, eager_fp=eager and not truth,
                    eager_ff=not eager and truth, gate_errors=gate != truth,
                    gate_fp=gate and not truth, gate_ff=not gate and truth,
                    U=any(q) and not certified, certified_pass=certified and truth,
                    certified_fail=certified and not truth, tasks_touched=any(q),
                    criterion_errors_found=sum(s and pv != rv for s, pv, rv in zip(q, p, r)))
        for key, val in vals.items():
            totals[key] += int(val)
    totals["eager_errors"] = totals["eager_fp"]+totals["eager_ff"]
    totals["update_delta"] = totals["gate_errors"]-totals["eager_errors"]
    totals["correct_queried"] = len(seen)-totals["criterion_errors_found"]
    return totals


def test_all_binary_patterns_and_subsets_match_literal_task_definitions():
    for k in range(1, 5):
        for p in product((0, 1), repeat=k):
            for r in product((0, 1), repeat=k):
                frame = policy.prepare_frame([unit(p, r)])
                for n in range(k+1):
                    for subset in combinations(range(k), n):
                        indices = np.asarray(subset, dtype=int)
                        counts = policy.prefix_accounting(frame, indices)
                        expected = brute(frame, subset)
                        assert {key: int(v[-1]) for key, v in counts.items()} == expected


def test_multitask_every_prefix_and_truncated_transcripts():
    units = [unit([0, 1], [1, 0], "a"), unit([0, 1], [1, 1], "b"),
             unit([1], [0], "c"), unit([1], [1], "d")]
    frame = policy.prepare_frame(units)
    for permutation in permutations(range(frame.M)):
        counts = policy.prefix_accounting(frame, np.asarray(permutation))
        for n in range(frame.M+1):
            expected = brute(frame, permutation[:n])
            assert {key: int(v[n]) for key, v in counts.items()} == expected


def test_mixed_union_matches_independent_complement_selection_at_every_cap():
    M = 5
    for uniform in permutations(range(M)):
        for serial in ([3, 1, 4, 0, 2], [2, 4], []):
            transcript, counts = policy.mixed_transcript(np.asarray(serial, int), np.asarray(uniform))
            for cap in range(M+1):
                A = set(serial[:cap//2])
                R = [i for i in uniform if i not in A][:cap-len(A)]
                assert set(transcript[:cap]) == A | set(R)
                assert counts[cap] == len(A)
                assert len(transcript[:cap]) == len(set(transcript[:cap])) == cap


def test_serial_uses_observed_fail_only_and_completes_each_task():
    units = [unit([0, 1, 0], [1, 0, 1], "a"), unit([0, 1], [1, 1], "b"),
             unit([1, 1], [0, 0], "c")]
    frame = policy.prepare_frame(units)
    candidates = policy.candidate_orders(frame, 12)["serial_candidates"]
    observed_failed, literal = set(), []
    for i in candidates:
        t = frame.task_index[i]
        if t in observed_failed:
            continue
        literal.append(i)
        if not frame.reference[i]:
            observed_failed.add(t)
    actual = policy.serial_transcript(frame, candidates)
    assert list(actual) == literal
    counts = policy.prefix_accounting(frame, actual)
    assert np.max(np.abs(counts["update_delta"])) <= 1
    assert counts["C"][-1] == len(units)


def test_candidate_priority_is_reference_blind_and_input_order_invariant():
    units = [unit([0, 1], [1, 0], "a", q=[1, 1]), unit([1, 0, 1], [0, 1, 1], "b")]
    alternate = [{**u, "r": [1-x for x in u["r"]]} for u in reversed(units)]
    a, b = policy.prepare_frame(units), policy.prepare_frame(alternate)
    aa, bb = policy.candidate_orders(a, 44), policy.candidate_orders(b, 44)
    for key in aa:
        assert np.array_equal(aa[key], bb[key])
    assert not np.array_equal(policy._priorities(a, 44, "uniform"),
                              policy._priorities(a, 44, "criterion-tie"))


def test_same_transcript_alternative_reference_is_distinct_from_reacquisition():
    units = [unit([0, 1, 1], [0, 1, 1], "a", r2=[1, 1, 1]),
             unit([0, 1, 1], [0, 1, 1], "b", r2=[1, 1, 1])]
    crossed = policy.replay_cell(units, "r", evaluation_reference_keys=("r", "r2"),
                                 policies=("serial",), seeds=(42,), shares=(1,), include_transcripts=True)
    reacquired = policy.replay_cell(units, "r2", policies=("serial",), seeds=(42,), shares=(1,),
                                    include_transcripts=True)
    assert [row["actual_queries"] for row in crossed["rows"]] == [2, 2]
    assert crossed["rows"][1]["C"] == 0
    assert reacquired["rows"][0]["actual_queries"] == 6
    assert reacquired["rows"][0]["C"] == 2


def test_accuracy_interval_is_design_based_and_pure_serial_is_not_ci():
    units = [unit([1, 0, 1], [1, 1, 0], "a"), unit([0, 1], [0, 1], "b")]
    rows = policy.replay_cell(units, seeds=(20,), shares=(.2, 1.))['rows']
    for row in rows:
        if row["policy"] in ("serial", "disagreement"):
            assert row["accuracy_estimate"] is None
            assert row["accuracy_ci_lower"] is None
            assert row["inference_kind"] == "logical_bounds_only"
        elif row["budget_share"] == .2:
            assert row["accuracy_ci_lower"] is not None
        else:
            assert row["accuracy_estimate"] == row["accuracy_truth"]
            assert row["accuracy_ci_lower"] is None  # full-grid CI is not requested


def test_conditional_estimate_unbiased_given_adaptive_set_and_complete_remainder():
    correctness = [1, 0, 1, 1, 0]
    A = [1, 2]
    remaining = [i for i in range(5) if i not in A]
    estimates = []
    for R in combinations(remaining, 2):
        inference = policy._accuracy(5, len(A), sum(correctness[i] for i in A),
                                     len(R), sum(correctness[i] for i in R), True)
        estimates.append(inference["accuracy_estimate"])
    assert np.mean(estimates) == pytest.approx(np.mean(correctness))
    assert policy._accuracy(5, 5, 3, 0, 0, True)["accuracy_ci_lower"] == .6


def test_pilot_forecasts_are_per100_and_certificate_control_uses_predicted_C():
    units = [unit([0, 1], [1, 0], str(i), r2=[1, 1], producer="x" if i < 3 else "y")
             for i in range(6)]
    result = policy.allocation_forecasts(units, seeds=(10, 11), minimum_clusters=4)
    assert len(result["predictions"]) == 4
    for row in result["predictions"]:
        assert row["pooling"] == "all_producers"
        assert row["pilot_units"] == 6
        assert row["repair_forecast_per100"] == pytest.approx(
            -row["delta_CW_per100"]-row["delta_D_per100"]+row["delta_A_per100"])
        assert row["certificate_only_forecast_per100"] == pytest.approx(
            -row["pilot_initial_error_fraction"]*row["delta_C_per100"])


def test_no_pilot_controls_charge_actual_pilot_bits_per_producer():
    pilot = [unit([1, 1, 0], [1, 1, 1], "pilot")]
    heldout = [unit([1, 0], [0, 1], "heldout")]
    rows = policy.no_pilot_controls(heldout, pilot, seeds=(7,), shares=(.2,), policies=("srs",))
    assert len(rows) == 1
    assert rows[0]["requested_total_cost"] == 3
    assert rows[0]["actual_queries"] == 2
    assert rows[0]["cost_capped_at_census"]


def test_auxiliary_absence_does_not_block_non_disagreement_policies():
    units = [unit([1, 0], [1, 1])]
    units[0].pop("q")
    result = policy.replay_cell(units, seeds=(2,), policies=("srs", "mix50", "serial"), shares=(.2, 1.))
    assert len(result["rows"]) == 6
    with pytest.raises(ValueError, match="requires an observed auxiliary"):
        policy.replay_cell(units, seeds=(2,), policies=("disagreement",), shares=(.2,))


@pytest.mark.parametrize("bad", [[], [unit([], [])], [unit([1], [None])],
                                  [unit([1], [1]), unit([1], [1])]])
def test_invalid_units_rejected(bad):
    with pytest.raises(ValueError):
        policy.prepare_frame(bad)
