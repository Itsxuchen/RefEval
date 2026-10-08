"""Retrospective paired accounting for frozen 50/50 SC + SRS allocation.

Hidden-reference task types explain realized outcomes; they are not deployable
features or a new allocation policy. Positive differences always mean mix - SRS.
Independent task arithmetic reuses only the existing query selectors; saved
microaccuracy intervals are cross-checked using their existing inference helper.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.conjunction_policy_data import load_datasets
from src.conjunction_policy_study import TASK_ORDERS, replay
from src.conjunction_label_value import SHARES, accuracy_inference, hybrid_queries

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'artifacts/expected/conjunction_label_value/budget_estimation_rows.csv.gz'
TYPES = ('TP', 'TN', 'FP', 'FF')
UNIT_FIELDS = ('certified_tasks', 'certified_pass', 'certified_fail',
               'certified_initial_correct', 'certified_initial_wrong',
               'residual_errors', 'residual_fp', 'residual_ff', 'gated_residual_errors',
               'fixed_initial_fp', 'fixed_initial_ff', 'introduced_fp', 'introduced_ff',
               'noncertified_ff_repair', 'noncertified_introduced_fp',
               'queried_false_fail_bits', 'unobserved_false_pass_bits')


def prepare_data(data):
    g, p = np.asarray(data['gold']), np.asarray(data['prediction'])
    ti = np.asarray(data['task_index'])
    if len(g) != len(p) or len(g) != len(ti) or not np.isin(g, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError('invalid binary frame')
    units = []
    for t, task in enumerate(data['task_ids']):
        slots = np.flatnonzero(ti == t).tolist()
        if not slots:
            raise ValueError('empty task')
        gold, pred = [bool(g[i]) for i in slots], [bool(p[i]) for i in slots]
        truth, initial = all(gold), all(pred)
        kind = 'TP' if truth and initial else 'FF' if truth else 'FP' if initial else 'TN'
        units.append(dict(task_id=task, base_task_id=data['base_task_ids'][t], slots=slots,
                          gold=gold, prediction=pred, truth_pass=truth, initial_pass=initial,
                          initial_type=kind))
    if sum(len(u['slots']) for u in units) != len(g):
        raise ValueError('task frame omits slots')
    return units


def account_query_set(data, indices, prepared=None):
    """Literal task loop, independent of production metric/delta helpers."""
    ids = [int(i) for i in indices]
    m = len(data['gold'])
    observed = set(ids)
    if len(ids) != len(observed) or any(i < 0 or i >= m for i in ids):
        raise ValueError('invalid or repeated query')
    states = []
    for unit in prepared if prepared is not None else prepare_data(data):
        slots, gold, pred = unit['slots'], unit['gold'], unit['prediction']
        seen = [i in observed for i in slots]
        cert_fail = any(s and not g for s, g in zip(seen, gold))
        cert_pass = all(seen) and all(gold)
        certified = cert_pass or cert_fail
        truth, initial, kind = unit['truth_pass'], unit['initial_pass'], unit['initial_type']
        eager = all(g if s else p for s, g, p in zip(seen, gold, pred))
        gated = truth if certified else initial
        introduced_fp = kind == 'TN' and eager
        introduced_ff = kind == 'TP' and not eager
        row = dict(task_id=unit['task_id'], base_task_id=unit['base_task_id'], initial_type=kind,
                   truth_pass=int(truth), initial_pass=int(initial), eager_pass=int(eager), gated_pass=int(gated),
                   n_criteria=len(slots), queried=sum(seen),
                   criterion_errors_found=sum(s and g != p for s, g, p in zip(seen, gold, pred)),
                   certified_tasks=int(certified), certified_pass=int(cert_pass), certified_fail=int(cert_fail),
                   certified_initial_correct=int(certified and truth == initial),
                   certified_initial_wrong=int(certified and truth != initial),
                   residual_errors=int(eager != truth), residual_fp=int(eager and not truth),
                   residual_ff=int(not eager and truth), gated_residual_errors=int(gated != truth),
                   fixed_initial_fp=int(kind == 'FP' and not eager),
                   fixed_initial_ff=int(kind == 'FF' and eager), introduced_fp=int(introduced_fp),
                   introduced_ff=int(introduced_ff), noncertified_ff_repair=int(kind == 'FF' and eager and not certified),
                   noncertified_introduced_fp=int(introduced_fp and not certified),
                   queried_false_fail_bits=sum(s and g and not p for s, g, p in zip(seen, gold, pred)),
                   unobserved_false_pass_bits=sum(not s and not g and p for s, g, p in zip(seen, gold, pred)))
        assert not introduced_ff
        assert not certified or eager == truth
        assert row['residual_errors'] == int(initial != truth) - row['fixed_initial_fp'] - row['fixed_initial_ff'] + row['introduced_fp']
        if introduced_fp:
            assert not certified and row['queried_false_fail_bits'] > 0 and row['unobserved_false_pass_bits'] > 0
            assert all(s for s, p in zip(seen, pred) if not p)
            assert all(not s for s, g in zip(seen, gold) if not g)
        if row['fixed_initial_fp']:
            assert cert_fail
        assert row['fixed_initial_ff'] == int(certified and kind == 'FF') + row['noncertified_ff_repair']
        states.append(row)
    totals = {field: sum(row[field] for row in states) for field in UNIT_FIELDS}
    totals.update(actual_queries=len(ids), n_tasks=len(states), n_criteria=m,
                  criterion_errors_found=sum(row['criterion_errors_found'] for row in states),
                  initial_task_errors=sum(row['initial_type'] in ('FP', 'FF') for row in states))
    for kind in TYPES:
        totals['initial_' + kind] = sum(row['initial_type'] == kind for row in states)
        totals['certified_' + kind] = sum(row['certified_tasks'] for row in states if row['initial_type'] == kind)
    n, correct = len(states), len(ids) - totals['criterion_errors_found']
    totals.update(pass_lower=totals['certified_pass'] / n,
                  pass_upper=1 - totals['certified_fail'] / n,
                  pass_width=1 - totals['certified_tasks'] / n,
                  accuracy_lower=correct / m, accuracy_upper=(correct + m - len(ids)) / m,
                  sample_agreement=correct / len(ids) if ids else None)
    return dict(units=states, totals=totals)


def pair_units(srs, mix):
    """Signed additive differences and disjoint certification-set composition."""
    if len(srs) != len(mix):
        raise ValueError('unaligned task frames')
    rows = []
    for s, h in zip(srs, mix):
        if (s['task_id'], s['initial_type']) != (h['task_id'], h['initial_type']):
            raise ValueError('unaligned task identity/truth')
        row = dict(task_id=s['task_id'], base_task_id=s['base_task_id'], initial_type=s['initial_type'])
        for field in UNIT_FIELDS:
            row[field + '_delta'] = h[field] - s[field]
        ho, so = h['certified_tasks'] and not s['certified_tasks'], s['certified_tasks'] and not h['certified_tasks']
        row.update(mix_only_cert=int(ho), srs_only_cert=int(so), both_cert=int(h['certified_tasks'] and s['certified_tasks']))
        for group in ('correct', 'wrong'):
            match = (s['initial_type'] in ('TP', 'TN')) == (group == 'correct')
            row['mix_only_cert_initial_' + group] = int(ho and match)
            row['srs_only_cert_initial_' + group] = int(so and match)
        for kind in TYPES:
            row['mix_only_cert_' + kind] = int(ho and s['initial_type'] == kind)
            row['srs_only_cert_' + kind] = int(so and s['initial_type'] == kind)
            row['certified_' + kind + '_delta'] = (int(ho) - int(so)) * (s['initial_type'] == kind)
        assert row['certified_tasks_delta'] == row['mix_only_cert'] - row['srs_only_cert']
        assert row['residual_errors_delta'] == -row['fixed_initial_fp_delta'] - row['fixed_initial_ff_delta'] + row['introduced_fp_delta']
        assert row['gated_residual_errors_delta'] == -row['certified_initial_wrong_delta']
        rows.append(row)
    return rows


def sum_pairs(rows):
    fields = [k for k in rows[0] if k not in ('task_id', 'base_task_id', 'initial_type')]
    return {k: sum(r[k] for r in rows) for k in fields}


def compare_allocations(data, srs, mix, prepared=None):
    a, b = account_query_set(data, srs, prepared), account_query_set(data, mix, prepared)
    if a['totals']['actual_queries'] != b['totals']['actual_queries']:
        raise ValueError('unequal actual query budgets')
    units = pair_units(a['units'], b['units'])
    totals = sum_pairs(units)
    gain, gross = totals['certified_tasks_delta'], totals['mix_only_cert']
    totals.update(mix_only_initial_correct_fraction=totals['mix_only_cert_initial_correct'] / gross if gross > 0 else None,
                  net_cert_gain_initial_correct_share=totals['certified_initial_correct_delta'] / gain if gain > 0 else None,
                  cert_gain_without_error_decrease=int(gain > 0 and totals['residual_errors_delta'] >= 0),
                  cert_gain_with_error_increase=int(gain > 0 and totals['residual_errors_delta'] > 0))
    return dict(srs=a, mix=b, units=units, paired=totals)


def saved_record(data, ds, order, seed, share, policy, account, A, R):
    """Independent task/micro counts; old CI helper retained and identified."""
    g, p = data['gold'], data['prediction']
    inf = accuracy_inference(len(g), len(A), int((g[A] == p[A]).sum()), len(R), int((g[R] == p[R]).sum()))
    totals = account['totals']
    truth = sum(bool(a) == bool(b) for a, b in zip(g, p)) / len(g)
    keys = ('actual_queries', 'certified_tasks', 'pass_lower', 'pass_upper', 'pass_width',
            'accuracy_lower', 'accuracy_upper', 'residual_errors', 'criterion_errors_found', 'sample_agreement')
    return dict(dataset=ds, seed=seed, task_order=order, budget_share=share,
                budget_cap=round(share * len(g)), policy=policy, n_tasks=len(data['task_ids']), n_criteria=len(g),
                stageA_queries=len(A), stageR_queries=len(R), accuracy_truth=truth,
                accuracy_estimate=inf['estimate'], accuracy_error=None if inf['estimate'] is None else inf['estimate'] - truth,
                accuracy_ci_lower=inf['ci_lower'], accuracy_ci_upper=inf['ci_upper'],
                accuracy_ci_width=inf['ci_upper'] - inf['ci_lower'],
                accuracy_ci_covers=inf['ci_lower'] <= truth <= inf['ci_upper'],
                inference_kind='exact_conditional_SRS' if len(R) else 'logical_bounds_or_census',
                **{k: totals[k] for k in keys})


def assert_saved(record, saved):
    if set(record) != set(saved):
        raise AssertionError('saved scalar schema changed')
    for k, value in record.items():
        actual = saved[k]
        if value is None:
            okay = actual == ''
        elif isinstance(value, bool):
            okay = actual == str(value)
        elif isinstance(value, (int, float)):
            okay = math.isclose(float(value), float(actual), rel_tol=1e-12, abs_tol=1e-12)
        else:
            okay = str(value) == actual
        if not okay:
            raise AssertionError(f"saved mismatch {record['dataset']}/{record['seed']}/{record['task_order']}/{record['budget_share']}/{record['policy']}/{k}: {value} != {actual}")


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['task_order'], row['budget_share'])].append(row)
    output = []
    for (ds, order, share), rs in groups.items():
        r = dict(dataset=ds, task_order=order, budget_share=share, seeds=len(rs), actual_queries=rs[0]['actual_queries'], n_tasks=rs[0]['n_tasks'])
        fields = [k for k in rs[0] if k.endswith('_delta') or k.startswith(('mix_only_cert', 'srs_only_cert')) or k == 'both_cert']
        for k in fields:
            vals = [x[k] for x in rs]
            r[k + '_mean'] = float(np.mean(vals))
            r[k + '_paired_median'] = float(np.median(vals))
        for k in ('certified_tasks', 'residual_errors', 'gated_residual_errors'):
            r['srs_' + k + '_mean'] = float(np.mean([x['srs_' + k] for x in rs]))
            r['mix_' + k + '_mean'] = float(np.mean([x['mix_' + k] for x in rs]))
            r[k + '_difference_of_marginal_medians'] = float(np.median([x['mix_' + k] for x in rs]) - np.median([x['srs_' + k] for x in rs]))
        gross = sum(x['mix_only_cert'] for x in rs)
        gain = sum(x['certified_tasks_delta'] for x in rs)
        r['gross_mix_only_initial_correct_fraction'] = sum(x['mix_only_cert_initial_correct'] for x in rs) / gross if gross > 0 else None
        r['net_positive_gain_initial_correct_share'] = sum(x['certified_initial_correct_delta'] for x in rs) / gain if gain > 0 else None
        r['gross_majority_initial_correct'] = r['gross_mix_only_initial_correct_fraction'] > .5 if gross > 0 else None
        r['net_majority_initial_correct'] = r['net_positive_gain_initial_correct_share'] > .5 if gain > 0 else None
        r['cert_gain_seeds'] = sum(x['certified_tasks_delta'] > 0 for x in rs)
        r['cert_gain_without_error_decrease_seeds'] = sum(x['cert_gain_without_error_decrease'] for x in rs)
        r['cert_gain_with_error_increase_seeds'] = sum(x['cert_gain_with_error_increase'] for x in rs)
        r['residual_error_improve_seeds'] = sum(x['residual_errors_delta'] < 0 for x in rs)
        r['residual_error_tie_seeds'] = sum(x['residual_errors_delta'] == 0 for x in rs)
        r['residual_error_worsen_seeds'] = sum(x['residual_errors_delta'] > 0 for x in rs)
        output.append(r)
    return output


def write_csv(path, rows):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(output, seeds=31):
    from src.reproduce import validate_output
    output = validate_output(Path(output))
    started = time.perf_counter()
    data = load_datasets()
    with gzip.open(SAVED, 'rt') as stream:
        old = {(r['dataset'], int(r['seed']), r['task_order'], float(r['budget_share']), r['policy']): r for r in csv.DictReader(stream)}
    rows, focal, base_rows, error_paths, checked = [], [], [], [], set()
    empty = np.array([], dtype=int)
    for ds, cell in data.items():
        prepared = prepare_data(cell)
        for seed in range(1, seeds + 1):
            random = replay(cell, 'random_criterion', 'release', seed)['indices']
            srs_cache = {}
            for share in SHARES:
                cap = round(share * len(cell['gold']))
                srs_cache[share] = account_query_set(cell, random[:cap], prepared)
            for order in TASK_ORDERS:
                trace = replay(cell, 'sc_judge', order, seed)
                for share in SHARES:
                    cap = round(share * len(cell['gold']))
                    A, R = hybrid_queries(cell, cap, order, seed, trace)
                    a, b = srs_cache[share], account_query_set(cell, np.r_[A, R], prepared)
                    assert a['totals']['actual_queries'] == b['totals']['actual_queries'] == cap
                    units = pair_units(a['units'], b['units']); paired = sum_pairs(units)
                    gain, gross = paired['certified_tasks_delta'], paired['mix_only_cert']
                    paired.update(mix_only_initial_correct_fraction=paired['mix_only_cert_initial_correct'] / gross if gross > 0 else None,
                                  net_cert_gain_initial_correct_share=paired['certified_initial_correct_delta'] / gain if gain > 0 else None,
                                  cert_gain_without_error_decrease=int(gain > 0 and paired['residual_errors_delta'] >= 0),
                                  cert_gain_with_error_increase=int(gain > 0 and paired['residual_errors_delta'] > 0))
                    for policy, o, account, aa, rr in [('random_criterion', 'release', a, empty, random[:cap]), ('sc50_then_srs', order, b, A, R)]:
                        rec = saved_record(cell, ds, o, seed, share, policy, account, aa, rr)
                        key = (ds, seed, o, share, policy)
                        assert_saved(rec, old[key]); checked.add(key)
                    row = dict(dataset=ds, task_order=order, seed=seed, budget_share=share, actual_queries=cap,
                               n_tasks=len(prepared), n_criteria=len(cell['gold']), **paired)
                    for prefix, account in [('srs', a), ('mix', b)]:
                        for k in ('certified_tasks', 'residual_errors', 'gated_residual_errors', 'certified_initial_correct', 'certified_initial_wrong'):
                            row[prefix + '_' + k] = account['totals'][k]
                    rows.append(row)
                    if share == .2 and order == 'release':
                        focal.append(row)
                        for su, hu in zip(a['units'], b['units']):
                            if any(u['introduced_fp'] or u['noncertified_ff_repair'] for u in (su, hu)):
                                detail = dict(dataset=ds, seed=seed, task_id=su['task_id'], base_task_id=su['base_task_id'], initial_type=su['initial_type'])
                                for prefix, u in [('srs', su), ('mix', hu)]:
                                    detail.update({prefix+'_'+k:v for k,v in u.items() if k not in ('task_id','base_task_id','initial_type')})
                                error_paths.append(detail)
                        by_base = defaultdict(list)
                        for unit in units:
                            by_base[unit['base_task_id']].append(unit)
                        for base, us in by_base.items():
                            part = sum_pairs(us)
                            base_rows.append(dict(dataset=ds, seed=seed, base_task_id=base, n_units=len(us), **part,
                                loo_certified_tasks_delta=paired['certified_tasks_delta'] - part['certified_tasks_delta'],
                                loo_residual_errors_delta=paired['residual_errors_delta'] - part['residual_errors_delta'],
                                loo_cert_gain_without_error_decrease=int(paired['certified_tasks_delta'] - part['certified_tasks_delta'] > 0 and paired['residual_errors_delta'] - part['residual_errors_delta'] >= 0)))
        print(f'{ds}: completed; {time.perf_counter()-started:.1f}s', flush=True)
    summary = summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'allocation_rows.csv.gz', rows)
    write_csv(output / 'allocation_summary.csv', summary)
    write_csv(output / 'allocation_focal_seeds.csv', focal)
    write_csv(output / 'allocation_focal_base_tasks.csv.gz', base_rows)
    if error_paths: write_csv(output / 'allocation_focal_error_paths.csv.gz', error_paths)
    paths = [Path(__file__), ROOT / 'src/conjunction_policy_data.py', ROOT / 'src/conjunction_policy_study.py', ROOT / 'src/conjunction_label_value.py', SAVED]
    protocol = ROOT / 'context/conjunction_mechanism_protocol.md'
    if protocol.exists(): paths.append(protocol)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    units_receipt = {ds: dict(n_units=len(c['task_ids']), n_base_tasks=len(set(c['base_task_ids'])), n_criteria=len(c['gold']),
        unit_frame_sha256=hashlib.sha256(json.dumps(list(zip(c['task_ids'], c['base_task_ids'])), ensure_ascii=False).encode()).hexdigest(),
        data_hashes=c['source_hashes']) for ds, c in data.items()}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        paired_rows=len(rows), summary_rows=len(summary), focal_seed_rows=len(focal), focal_base_rows=len(base_rows), focal_error_path_rows=len(error_paths),
        saved_unique_rows_matched=len(checked), saved_fields_per_row=len(next(iter(old.values()))),
        datasets=units_receipt, source_hashes=source_hashes,
        config=dict(seeds=list(range(1,seeds+1)), task_orders=list(TASK_ORDERS), shares=list(SHARES),
                    policies=['random_criterion','sc50_then_srs'], difference_direction='mix minus SRS'),
        identities=['certified_delta = net initially-correct certification + net initially-wrong certification',
                    'residual_error_delta = -fixed_initial_FP_delta - fixed_initial_FF_delta + introduced_FP_delta',
                    'gated_residual_error_delta = -certified_initial_wrong_delta',
                    'residual_error_delta = -certified_initial_wrong_delta - noncertified_FF_repair_delta + introduced_FP_delta',
                    'introduced_FP implies uncertified, at least one repaired predicted-fail/reference-pass bit, and an unobserved predicted-pass/reference-fail bit; introduced_FF=0'],
        scope=['Fixed-reference retrospective task accounting; hidden truth/type is not a deployable feature.',
               'Query selectors and random generators reused exactly; task arithmetic independently implemented; inference helper reused only to reconcile saved scalar endpoints.',
               '31 paired seed differences describe randomized order variation, not independent datasets or population inference.',
               'Means of additive paired components add; medians of paired differences need not equal differences of marginal medians.',
               'Gross mix-only composition excludes shared certificates; signed net contributions subtract SRS-only certificates. Net shares only reported when total net gain>0 and may lie outside[0,1].',
               'Base-task deletion removes contributions from the fixed acquired sets; it does not rerun or rebudget the allocation.',
               'Current six fixed caps and nine cells only; no policy tuning, new labels, API calls, or human labor claims.'])
    (output / 'allocation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/reports/conjunction_mechanism')
    parser.add_argument('--seeds', type=int, default=31)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 31: parser.error('seeds must be between1and31')
    print(json.dumps(run(args.output, args.seeds), indent=2))


if __name__ == '__main__':
    main()
