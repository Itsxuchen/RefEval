"""Exhaustive literal worlds/query sets, independent of vector metric helpers."""
import itertools

import numpy as np
import pytest

from src.conjunction_allocation_mechanism import (
    account_query_set, compare_allocations, pair_units, prepare_data, summarize,
)


def frame(g, p, sizes=(2, 1)):
    ids = [f't{i}' for i in range(len(sizes))]
    return dict(gold=np.array(g), prediction=np.array(p), task_index=np.repeat(np.arange(len(sizes)), sizes),
                task_ids=ids, base_task_ids=ids)


def direct(data, q):
    observed = set(q)
    total = dict(cert=0, err=0, fixed_fp=0, fixed_ff=0, new_fp=0, gated=0)
    for t in range(len(data['task_ids'])):
        ix = np.flatnonzero(data['task_index'] == t).tolist()
        truth = all(data['gold'][i] == 1 for i in ix)
        initial = all(data['prediction'][i] == 1 for i in ix)
        values = [data['gold'][i] if i in observed else data['prediction'][i] for i in ix]
        end = all(x == 1 for x in values)
        cert = any(data['gold'][i] == 0 for i in observed.intersection(ix)) or all(i in observed for i in ix)
        total['cert'] += cert
        total['err'] += end != truth
        total['fixed_fp'] += initial and not truth and not end
        total['fixed_ff'] += not initial and truth and end
        total['new_fp'] += not initial and not truth and end
        total['gated'] += initial != truth and not cert
    return total


def test_all_three_bit_worlds_all_query_sets_and_equal_budget_pairs():
    sets = [tuple(i for i, x in enumerate(bits) if x) for bits in itertools.product((0, 1), repeat=3)]
    checked = 0
    for g in itertools.product((0, 1), repeat=3):
        for p in itertools.product((0, 1), repeat=3):
            data = frame(g, p); prepared = prepare_data(data)
            accounts = {}
            for q in sets:
                a = account_query_set(data, q, prepared); b = direct(data, q)
                for field, key in [('certified_tasks','cert'), ('residual_errors','err'), ('fixed_initial_fp','fixed_fp'),
                                   ('fixed_initial_ff','fixed_ff'), ('introduced_fp','new_fp'), ('gated_residual_errors','gated')]:
                    assert a['totals'][field] == b[key]
                accounts[q] = a
                checked += 1
            for a in sets:
                for b in sets:
                    if len(a) != len(b): continue
                    rows = pair_units(accounts[a]['units'], accounts[b]['units'])
                    assert sum(r['certified_tasks_delta'] for r in rows) == accounts[b]['totals']['certified_tasks'] - accounts[a]['totals']['certified_tasks']
                    assert sum(r['residual_errors_delta'] for r in rows) == accounts[b]['totals']['residual_errors'] - accounts[a]['totals']['residual_errors']
    assert checked == 512


def test_extra_certification_can_fail_both_majority_and_error_improvement():
    # SRS repairs an uncertified false fail; mix certifies initially-correct TN.
    data = frame([1,1,0], [0,1,0])
    c = compare_allocations(data, [0], [2])
    assert c['paired']['certified_tasks_delta'] == 1
    assert c['paired']['residual_errors_delta'] == 1
    assert c['paired']['cert_gain_with_error_increase'] == 1
    assert c['srs']['totals']['noncertified_ff_repair'] == 1
    # Extra certificate entirely initially wrong: majority-correct hypothesis fails.
    d = frame([1,1,0], [1,1,1])
    c = compare_allocations(d, [0], [2])
    assert c['paired']['mix_only_initial_correct_fraction'] == 0
    assert c['paired']['net_cert_gain_initial_correct_share'] == 0
    reverse = compare_allocations(d, [2], [0])
    assert reverse['paired']['net_cert_gain_initial_correct_share'] is None


def test_partial_correction_introduces_unobserved_false_pass():
    data = frame([1,0,1], [0,1,1])
    row = account_query_set(data, [0])['units'][0]
    assert row['initial_type'] == 'TN'
    assert row['introduced_fp'] == row['noncertified_introduced_fp'] == 1
    assert row['queried_false_fail_bits'] == row['unobserved_false_pass_bits'] == 1
    assert row['eager_pass'] == 1 and row['gated_pass'] == 0


def test_invalid_queries_and_unequal_budgets_fail():
    data = frame([1,0,1], [0,1,1])
    for q in [[1,1], [-1], [3]]:
        with pytest.raises(ValueError): account_query_set(data,q)
    with pytest.raises(ValueError): compare_allocations(data,[0],[0,1])
