"""Independent behavioral checks for finite-reference query experiments.

The reference implementation below reads each observed query set directly. It
does not use the production delta updater to compute expected task verdicts.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src import conjunction_policy_study as study


def data_world(gold, prediction, sizes=None, secondary=None, bases=None):
    gold, prediction = np.asarray(gold, dtype=int), np.asarray(prediction, dtype=int)
    sizes = list(sizes or [len(gold)])
    assert sum(sizes) == len(gold) == len(prediction)
    n, m = len(sizes), len(gold)
    return {
        "gold": gold, "prediction": prediction,
        "task_index": np.repeat(np.arange(n), sizes),
        "task_ids": [f"t{i}" for i in range(n)],
        "base_task_ids": list(bases or [f"b{i}" for i in range(n)]),
        "criterion_ids": [f"c{i}" for i in range(m)],
        "category": ["binary"] * m,
        "secondary_predictions": np.asarray([1 - prediction] if secondary is None else secondary, dtype=int).reshape(-1, m),
        "secondaries": ["secondary"], "family": "synthetic", "target": "all_pass", "source_hashes": {},
    }


def naive_metrics(data, indices):
    queried = set(map(int, indices))
    assert len(queried) == len(indices)
    ti, g, p = data["task_index"], data["gold"], data["prediction"]
    result = {field: 0 for field in study.FIELDS}
    for t in range(len(data["task_ids"])):
        slots = set(np.flatnonzero(ti == t).tolist())
        observed = slots & queried
        truth = all(g[i] for i in slots)
        initial = all(p[i] for i in slots)
        current = all(g[i] if i in observed else p[i] for i in slots)
        certified = any(g[i] == 0 for i in observed) or observed == slots
        result["criterion_errors_found"] += sum(g[i] != p[i] for i in observed)
        result["certified_tasks"] += certified
        result["residual_fp"] += current and not truth
        result["residual_ff"] += truth and not current
        result["introduced_fp"] += current and not truth and not initial
        result["fixed_initial_fp"] += initial and not truth and not current
        result["fixed_initial_ff"] += truth and not initial and current
        result["tasks_touched"] += bool(observed)
        result["certified_initial_ff"] += truth and not initial and certified
        if certified:
            assert current == truth
        # Count transitions independently in the actual order, not by endpoint.
        values = {i: int(p[i]) for i in slots}
        before = initial
        for i in indices:
            if i in slots:
                values[int(i)] = int(g[i])
                after = all(values.values())
                result["new_fp_events"] += after and not before and not truth
                before = after
    return {key: int(value) for key, value in result.items()}


def assert_naive_prefixes(data, trace):
    m = len(data["gold"])
    points = study.points_from_trace(data, trace, np.arange(m + 1) / m)
    for cap, point in enumerate(points):
        used = min(cap, len(trace["indices"]))
        assert point["budget_cap"] == cap
        assert point["actual_queries"] == used
        assert point["unused_budget"] == cap - used
        expected = naive_metrics(data, trace["indices"][:used])
        for field, value in expected.items():
            assert point[field] == value, (field, cap, point[field], value)
        gated_fp = gated_ff = 0
        queried = set(trace["indices"][:used].tolist())
        for t in range(len(data["task_ids"])):
            slots = set(np.flatnonzero(data["task_index"] == t).tolist())
            truth = all(data["gold"][i] for i in slots)
            certified = any(data["gold"][i] == 0 for i in slots & queried) or slots <= queried
            gated = truth if certified else all(data["prediction"][i] for i in slots)
            gated_fp += gated and not truth
            gated_ff += truth and not gated
        assert point["gated_residual_fp"] == gated_fp
        assert point["gated_residual_ff"] == gated_ff
        assert point["gated_residual_errors"] == gated_fp + gated_ff
        assert point["residual_fp"] - point["gated_residual_fp"] == point["introduced_fp"]
        assert point["gated_residual_ff"] - point["residual_ff"] == point["delayed_ff_corrections"]
    for field in ("criterion_errors_found", "certified_tasks", "tasks_touched", "new_fp_events"):
        values = [point[field] for point in points]
        assert values == sorted(values)
    false_fails = [point["residual_ff"] for point in points]
    assert false_fails == sorted(false_fails, reverse=True)
    for field in ("gated_residual_fp", "gated_residual_ff", "gated_residual_errors"):
        values = [point[field] for point in points]
        assert values == sorted(values, reverse=True)


def test_exhaustive_small_worlds_match_independent_prefix_metrics():
    # Every gold/prediction configuration through three criteria, every query
    # policy and task-order mode, canonical and randomized ties.
    for k in (1, 2, 3):
        for gold in itertools.product((0, 1), repeat=k):
            for prediction in itertools.product((0, 1), repeat=k):
                data = data_world(gold, prediction)
                for policy, order, seed in itertools.product(study.POLICIES, study.TASK_ORDERS, (0, 11)):
                    assert_naive_prefixes(data, study.replay(data, policy, order, seed))


def test_multiple_tasks_shared_budget_prefixes_and_certification():
    rng = np.random.default_rng(971)
    for _ in range(12):
        data = data_world(rng.integers(0, 2, 8), rng.integers(0, 2, 8), [1, 3, 4],
                          secondary=rng.integers(0, 2, (2, 8)))
        for policy, order, seed in itertools.product(study.POLICIES, study.TASK_ORDERS, (0, 7)):
            trace = study.replay(data, policy, order, seed)
            assert_naive_prefixes(data, trace)
            if policy.startswith("sc_"):
                point = study.points_from_trace(data, trace, np.asarray([1.0]))[0]
                assert point["certified_tasks"] == 3
                assert point["residual_errors"] == 0
                # Each completed task is certified before moving to the next;
                # an eager-update regression can affect only the current task.
                values = np.cumsum(trace["deltas"][:, study.FIELDS.index("introduced_fp")])
                assert values.max(initial=0) <= 1
            for pos in np.flatnonzero(trace["deltas"][:, study.FIELDS.index("new_fp_events")]):
                t = data["task_index"][trace["indices"][pos]]
                assert trace["susceptible"][t]


def test_query_selection_cannot_read_unobserved_gold():
    for prediction in itertools.product((0, 1), repeat=3):
        worlds = [data_world(g, prediction, [1, 2]) for g in itertools.product((0, 1), repeat=3)]
        for policy, order, seed in itertools.product(study.POLICIES, study.TASK_ORDERS, (0, 5)):
            traces = [study.replay(data, policy, order, seed)["indices"].tolist() for data in worlds]
            visible = {key: value for key, value in worlds[0].items() if key != "gold"}
            assert np.array_equal(study.make_order(visible, policy, order, seed),
                                  study.make_order(worlds[-1], policy, order, seed))
            for a, b in itertools.combinations(range(len(worlds)), 2):
                # Until an observed answer differs, the next selected query (or
                # stopping decision) must coincide in both possible gold worlds.
                for position in range(max(len(traces[a]), len(traces[b])) + 1):
                    qa = traces[a][position] if position < len(traces[a]) else None
                    qb = traces[b][position] if position < len(traces[b]) else None
                    assert qa == qb
                    if qa is None or worlds[a]["gold"][qa] != worlds[b]["gold"][qb]:
                        break


def test_candidate_exhaustion_is_not_full_budget_use_and_fallback_continues():
    data = data_world([1, 0, 1, 1], [0, 1, 1, 1], [2, 2], secondary=[[1, 1, 1, 1]])
    only = study.replay(data, "disagreement_only", "release", 0)
    fallback = study.replay(data, "disagreement_then_random", "release", 0)
    assert only["indices"].tolist() == [0]
    assert fallback["indices"].tolist()[:1] == [0]
    assert set(fallback["indices"].tolist()) == set(range(4))
    only_point = study.points_from_trace(data, only, np.asarray([1.0]))[0]
    fallback_point = study.points_from_trace(data, fallback, np.asarray([1.0]))[0]
    assert (only_point["budget_cap"], only_point["actual_queries"], only_point["unused_budget"]) == (4, 1, 3)
    assert only_point["residual_fp"] == 1
    assert fallback_point["residual_errors"] == 0


def test_empty_disagreement_trace_preserves_uncertified_initial_verdicts():
    data = data_world([1, 0], [0, 1], secondary=[[0, 1]])
    trace = study.replay(data, "disagreement_only", "release", 0)
    assert len(trace["indices"]) == 0
    assert_naive_prefixes(data, trace)
    assert study.trace_summary(data, trace)["first_all_certified"] is None


def test_single_label_repair_can_create_then_remove_a_false_pass():
    data = data_world([1, 0], [0, 1], secondary=[[1, 1]])
    trace = study.replay(data, "disagreement_then_random", "release", 0)
    points = study.points_from_trace(data, trace, np.asarray([0, .5, 1]))
    assert [point["residual_errors"] for point in points] == [0, 1, 0]
    assert [point["criterion_errors_found"] for point in points] == [0, 1, 2]
    assert [point["introduced_fp"] for point in points] == [0, 1, 0]
    assert [point["new_fp_events"] for point in points] == [0, 1, 1]
    assert [point["certified_tasks"] for point in points] == [0, 0, 1]


def test_certificate_gate_prevents_new_errors_but_can_delay_a_correct_update():
    data = data_world([1, 1], [0, 1], secondary=[[1, 1]])
    trace = study.replay(data, "disagreement_only", "release", 0)
    final = study.points_from_trace(data, trace, np.asarray([1.0]))[0]
    assert final["actual_queries"] == 1
    assert final["residual_errors"] == 0
    assert final["certified_tasks"] == 0
    assert final["gated_residual_errors"] == final["delayed_ff_corrections"] == 1


def drop_base(data, base):
    keep_tasks = [i for i, b in enumerate(data["base_task_ids"]) if b != base]
    keep_items = np.flatnonzero(np.isin(data["task_index"], keep_tasks))
    task_map = {old: new for new, old in enumerate(keep_tasks)}
    item_map = {old: new for new, old in enumerate(keep_items)}
    subset = dict(data)
    for key in ("gold", "prediction"):
        subset[key] = data[key][keep_items]
    subset["task_index"] = np.asarray([task_map[int(data["task_index"][i])] for i in keep_items])
    subset["secondary_predictions"] = data["secondary_predictions"][:, keep_items]
    for key in ("task_ids", "base_task_ids"):
        subset[key] = [data[key][i] for i in keep_tasks]
    for key in ("criterion_ids", "category"):
        subset[key] = [data[key][i] for i in keep_items]
    return subset, item_map


def test_base_task_deletion_matches_reexecution_with_frozen_candidate_priorities():
    data = data_world([1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1, 0], [2, 2, 3],
                      bases=["shared", "shared", "other"], secondary=[[1, 1, 0, 0, 1, 1, 1]])
    shares = np.arange(101) / 100
    for policy, order, seed, base in itertools.product(study.POLICIES, study.TASK_ORDERS, (0, 19), ("shared", "other")):
        full_trace = study.replay(data, policy, order, seed)
        sub, mapping = drop_base(data, base)
        original_candidates = study.make_order(data, policy, order, seed)
        candidates = np.asarray([mapping[i] for i in original_candidates if i in mapping], dtype=int)
        rerun = study.replay(sub, policy, order, seed, candidate_order=candidates)
        filtered = study.points_from_trace(data, full_trace, shares, removed_base=base)
        assert filtered == study.points_from_trace(sub, rerun, shares)


@pytest.mark.parametrize("candidates", ([0, 0], [-1], [2]))
def test_replay_rejects_repeated_or_out_of_range_queries(candidates):
    data = data_world([1, 0], [0, 1])
    with pytest.raises(ValueError, match="invalid or repeated"):
        study.replay(data, "random_criterion", "release", 0, np.asarray(candidates))


def test_random_criterion_baseline_is_global_and_task_order_invariant():
    data = data_world([1] * 12, [1] * 12, [4, 4, 4])
    for seed in (0, 1, 8):
        orders = [study.make_order(data, "random_criterion", order, seed) for order in study.TASK_ORDERS]
        assert all(np.array_equal(orders[0], order) for order in orders[1:])
        assert len(set(data["task_index"][orders[0][:4]].tolist())) > 1
