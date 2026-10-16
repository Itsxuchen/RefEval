"""Independent, metadata-only verification of the WildIFEval feasibility stop.

No production ingestion helpers are imported. JSON numeric literals become
validity sentinels during parsing; they are never retained as numeric scores,
thresholded, returned, or logged. Outputs contain hashes and aggregate counts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
from pathlib import Path
import unicodedata


PRODUCERS = ('deepseek-v3', 'llama3.1-405b', 'llama3.3-70b', 'mistral-large', 'qwen2.5-72b')
JUDGES = {'p': 'llama3.1-70b', 'r': 'deepseek-v3', 'q': 'qwen2.5-72b', 'r2': 'gpt-4o-2024-08-06'}
PINNED_REVISIONS = {'benchmark': 'fc5f47b2b1be19ac95986624a4c0d8fd25dd316c',
                    'predictions': 'e34bc0fcc01b7f1d49032878563e8c32393c9d59'}


def expected_source_paths():
    """Independent inventory: three metadata files plus five files per producer."""
    paths = {'benchmark/data/test-00000-of-00001.parquet', 'benchmark/README.md', 'predictions/README.md'}
    for producer in PRODUCERS:
        paths.add(f'predictions/model_predictions/{producer}-0shot-wild-if-eval.json')
        for role, judge in JUDGES.items():
            subdir = 'other_judges/' if role in ('q', 'r2') else ''
            paths.add(f'predictions/llm_aaj_scores/{subdir}llm-aaj-{judge}.{producer}-0shot-wild-if-eval.json')
    return paths


def validate_inventory(manifest):
    entries = manifest.get('files', ())
    paths = [entry['path'] for entry in entries]
    if len(paths) != 28 or len(set(paths)) != 28 or set(paths) != expected_source_paths():
        raise AssertionError('Source manifest must contain exactly the 28 unique expected paths')
    if manifest['revisions'] != PINNED_REVISIONS:
        raise AssertionError('Unexpected upstream snapshot revision')
    return len(paths)


class NumberMask(Enum):
    PRESENT = 'finite_number_in_unit_interval'
    INVALID = 'invalid_number'


def numeric_mask(literal: str) -> NumberMask:
    """Range validation only; the Decimal and its textual value are discarded."""
    try:
        value = Decimal(literal)
        valid = value.is_finite() and Decimal(0) <= value <= Decimal(1)
    except InvalidOperation:
        valid = False
    return NumberMask.PRESENT if valid else NumberMask.INVALID


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            # Do not place the potentially private key in the error message.
            raise ValueError('Duplicate JSON key: independent verification stops')
        result[key] = value
    return result


def masked_json(path: Path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle, parse_float=numeric_mask, parse_int=numeric_mask,
                         parse_constant=lambda _: NumberMask.INVALID,
                         object_pairs_hook=unique_object)


def sha_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def read_score_metadata(path):
    raw = masked_json(path)
    if not isinstance(raw, dict):
        raise ValueError('Score document must be an object')
    result = {}
    for prompt, row in raw.items():
        scores = row.get('scores') if isinstance(row, dict) else None
        if not isinstance(scores, dict):
            result[prompt] = {'keys': frozenset(), 'valid_keys': frozenset(), 'valid_object': False}
            continue
        result[prompt] = {'keys': frozenset(scores), 'valid_keys': frozenset(
            key for key, value in scores.items() if value is NumberMask.PRESENT), 'valid_object': True}
    return result


def read_output_metadata(path):
    raw = masked_json(path)
    if not isinstance(raw, dict):
        raise ValueError('Output document must be an object')
    if 'predictions_key' in raw:
        wrapper = raw['predictions_key']
        raw = raw[wrapper] if isinstance(wrapper, str) and wrapper in raw else wrapper
        if not isinstance(raw, dict):
            raise ValueError('Unsupported output wrapper')
    result = {}
    for prompt, messages in raw.items():
        if not isinstance(messages, list):
            result[prompt] = {'valid': False, 'output_hash': '', 'normalized_prompt_hash': ''}
            continue
        users = [x.get('content') for x in messages if isinstance(x, dict) and x.get('role') == 'user']
        answers = [x.get('content') for x in messages if isinstance(x, dict) and x.get('role') == 'assistant']
        valid = len(users) == 1 and users[0] == prompt and len(answers) == 1 and isinstance(answers[0], str) and bool(answers[0].strip())
        result[prompt] = {'valid': valid,
                          'output_hash': sha_text(answers[0]) if len(answers) == 1 and isinstance(answers[0], str) else '',
                          'normalized_prompt_hash': sha_text(normalized(prompt))}
    return result


def key_diagnostics(primary, reference):
    shared = primary.keys() & reference.keys()
    exact = whitespace = different_count = 0
    pairs = Counter()
    for prompt in shared:
        pkeys, rkeys = primary[prompt]['keys'], reference[prompt]['keys']
        exact += pkeys == rkeys
        whitespace += {normalized(x) for x in pkeys} == {normalized(x) for x in rkeys}
        different_count += len(pkeys) != len(rkeys)
        pairs[f'{len(pkeys)},{len(rkeys)}'] += 1
    return {'shared_task_text_keys': len(shared), 'same_exact_constraint_set': exact,
            'same_whitespace_normalized_constraint_set': whitespace,
            'different_constraint_count': different_count, 'criterion_count_pairs': dict(sorted(pairs.items()))}


def reconstruct_metadata(benchmark, outputs, scores, producers=PRODUCERS):
    """Independent relational-set join and graph component reconstruction."""
    tasks, signature_to_tasks = {}, defaultdict(set)
    for row in benchmark:
        conv, criteria = row['conversation_id'], tuple(row['decomposition'])
        if not isinstance(conv, str) or not all(isinstance(c, str) for c in criteria):
            raise ValueError('Invalid benchmark metadata schema')
        tid = sha_text('wildifeval-task|' + conv)
        if tid in tasks:
            raise ValueError('Repeated conversation identifier')
        tasks[tid] = criteria
        signature_to_tasks[tuple(sorted(criteria))].add(tid)
    # A prompt may be linked only if every exact-set association identifies the
    # same benchmark task. No ordering or semantic/fuzzy matching is allowed.
    prompt_candidates = defaultdict(set)
    for producer in producers:
        for role in JUDGES:
            for prompt, meta in scores[producer][role].items():
                prompt_candidates[prompt].update(signature_to_tasks.get(tuple(sorted(meta['keys'])), ()))
    task_prompts = defaultdict(set)
    for prompt, tids in prompt_candidates.items():
        if len(tids) == 1:
            task_prompts[next(iter(tids))].add(prompt)
    token_tasks, rows = defaultdict(set), {}
    for tid, criteria in tasks.items():
        prompts = task_prompts[tid]
        prompt = next(iter(prompts)) if len(prompts) == 1 else None
        unique_task = len(signature_to_tasks[tuple(sorted(criteria))]) == 1
        proper = unique_task and len(criteria) == len(set(criteria)) and 2 <= len(criteria) <= 8 and prompt is not None
        for producer in producers:
            out = outputs[producer].get(prompt, {'valid': False, 'output_hash': '', 'normalized_prompt_hash': ''})
            prompt_hash = sha_text(prompt) if prompt is not None else ''
            normal_hash = out['normalized_prompt_hash'] or (sha_text(normalized(prompt)) if prompt is not None else '')
            for kind, value in [('prompt', normal_hash), ('output', out['output_hash'])]:
                if value:
                    token_tasks[(kind, value)].add(tid)
            uid = sha_text('wildifeval-unit|' + tid + '|' + producer)
            hashes = [sha_text(c) for c in criteria]
            result = {'unit_id': uid, 'producer': producer, 'task_id': tid, 'k': str(len(criteria)),
                      'prompt_hash': prompt_hash, 'normalized_prompt_hash': normal_hash,
                      'output_hash': out['output_hash'], 'constraint_hashes': json.dumps(hashes),
                      'criterion_ids': json.dumps([sha_text(uid + '|' + str(j) + '|' + h) for j, h in enumerate(hashes)])}
            role_valid = {}
            for role in JUDGES:
                rec = scores[producer][role].get(prompt)
                role_valid[role] = bool(proper and out['valid'] and rec is not None and rec['valid_object']
                                        and rec['keys'] == set(criteria) and rec['valid_keys'] == set(criteria))
                result[role + '_missing_positions'] = json.dumps([j for j, c in enumerate(criteria)
                                                                 if rec is None or c not in rec['valid_keys']])
            result['eligible_pr'] = str(int(role_valid['p'] and role_valid['r']))
            result['eligible_four_role'] = str(int(all(role_valid.values())))
            rows[uid] = result
    # Graph search is intentionally independent of the production union-find.
    neighbors = {tid: set() for tid in tasks}
    for tids in token_tasks.values():
        if len(tids) > 1:
            anchor = min(tids)
            neighbors[anchor].update(tids - {anchor})
            for tid in tids - {anchor}:
                neighbors[tid].add(anchor)
    unseen, components, clusters = set(tasks), [], {}
    while unseen:
        pending = [unseen.pop()]
        component = set(pending)
        while pending:
            current = pending.pop()
            new = neighbors[current] & unseen
            unseen.difference_update(new)
            component.update(new)
            pending.extend(new)
        components.append(component)
        cid = sha_text('wildifeval-cluster|' + '|'.join(sorted(component)))
        for tid in component:
            clusters[tid] = cid
    for row in rows.values():
        row['cluster_id'] = clusters[row['task_id']]
    summary = {'benchmark_rows': len(tasks), 'unique_constraint_sets': len(signature_to_tasks),
               'ambiguous_constraint_sets': sum(len(v) > 1 for v in signature_to_tasks.values()),
               'ambiguous_constraint_set_rows': sum(len(v) for v in signature_to_tasks.values() if len(v) > 1),
               'total_clusters_before_complete_case': len(components),
               'largest_cluster_before_complete_case': max(map(len, components), default=0), 'frames': {}}
    for frame in ('pr', 'four_role'):
        eligible = {producer: {r['task_id'] for r in rows.values() if r['producer'] == producer and r['eligible_' + frame] == '1'}
                    for producer in producers}
        common = set.intersection(*eligible.values())
        common_clusters = {clusters[t] for t in common}
        ordered = sorted(common_clusters, key=lambda c: sha_text('refeval-external-v1|' + c))
        pilot = set(ordered[:len(ordered) // 5])
        for row in rows.values():
            row['common_' + frame] = str(int(row['task_id'] in common))
            row['split_' + frame] = ('pilot' if row['cluster_id'] in pilot else 'confirmation') if row['task_id'] in common else ''
        summary['frames'][frame] = {'common_tasks': len(common), 'units': len(common) * len(producers),
                                   'clusters': len(common_clusters), 'pilot_clusters': len(pilot),
                                   'confirmation_clusters': len(common_clusters) - len(pilot),
                                   'eligible_tasks_before_common_intersection': {p: len(v) for p, v in eligible.items()}}
    summary['p_r_constraint_key_comparison'] = {p: key_diagnostics(scores[p]['p'], scores[p]['r']) for p in producers}
    return rows, summary


def verify(sources: Path, preflight: Path, output: Path):
    sources, preflight, output = Path(sources), Path(preflight), Path(output)
    report_path = preflight / 'report.json'
    report = json.loads(report_path.read_text())
    manifest_path = sources / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if sha_file(manifest_path) != report['sources_manifest_sha256']:
        raise ValueError('Published preflight and source-manifest hashes differ')
    validate_inventory(manifest)
    checked_files = []
    for entry in manifest['files']:
        rel = Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Invalid manifest path')
        if sha_file(sources / rel) != entry['sha256']:
            raise ValueError('Pinned source hash differs')
        checked_files.append(str(rel))
    for name, key in [('masked_units.csv', 'masked_units_sha256'), ('excluded_units.csv', 'excluded_units_sha256')]:
        if sha_file(preflight / name) != report[key]:
            raise ValueError('Published masked artifact hash differs')
    import pyarrow.parquet as pq
    benchmark = pq.read_table(sources / 'benchmark/data/test-00000-of-00001.parquet', columns=['conversation_id', 'decomposition']).to_pylist()
    scores, outputs = {}, {}
    for producer in PRODUCERS:
        scores[producer] = {}
        outputs[producer] = read_output_metadata(sources / f'predictions/model_predictions/{producer}-0shot-wild-if-eval.json')
        for role, judge in JUDGES.items():
            subdir = 'other_judges/' if role in ('q', 'r2') else ''
            scores[producer][role] = read_score_metadata(sources / f'predictions/llm_aaj_scores/{subdir}llm-aaj-{judge}.{producer}-0shot-wild-if-eval.json')
    rows, independent = reconstruct_metadata(benchmark, outputs, scores)
    scalar_checks = 0
    for key, value in independent.items():
        if key == 'frames':
            for frame, fields in value.items():
                for field, observed in fields.items():
                    if report['frames'][frame][field] != observed:
                        raise AssertionError('Independent frame count mismatch: ' + frame + '/' + field)
                    scalar_checks += 1
        else:
            if report[key] != value:
                raise AssertionError('Independent metadata mismatch: ' + key)
            scalar_checks += 1
    csv_checks, seen = 0, set()
    with (preflight / 'masked_units.csv').open() as handle:
        for published in csv.DictReader(handle):
            uid = published['unit_id']
            if uid in seen or uid not in rows:
                raise AssertionError('Duplicate or unrecognized published unit ID')
            seen.add(uid)
            for field, value in rows[uid].items():
                if published[field] != value:
                    raise AssertionError('Independent masked-row mismatch in field: ' + field)
                csv_checks += 1
    if seen != set(rows):
        raise AssertionError('Missing independently reconstructed unit IDs')
    receipt = {'status': 'PASS', 'schema_version': 1, 'metadata_only': True,
               'numeric_scores_retained': False, 'pass_fail_threshold_applied': False,
               'production_join_helpers_used': False, 'sources_manifest_sha256': sha_file(manifest_path),
               'preflight_report_sha256': sha_file(report_path), 'masked_units_sha256': sha_file(preflight / 'masked_units.csv'),
               'verifier_source_sha256': sha_file(Path(__file__)), 'source_files_hash_checked': len(checked_files),
               'report_field_checks': scalar_checks, 'masked_units_compared': len(seen),
               'masked_cell_checks': csv_checks, 'independent_counts': independent,
               'scope': 'Independent exact criterion-set joins, validity masks, IDs, eligibility, common-frame membership, missing positions, duplicate-linked clusters and split; every source hash. No outcome analysis.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Preserve an existing independent receipt; choose another output path')
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def verify_saved(saved: Path):
    """Offline CI check of public compressed metadata and the independent receipt.

    This verifies saved-artifact consistency. Re-running the independent source
    join requires ``verify`` and the pinned local download cache.
    """
    saved = Path(saved)
    report_path = saved / 'preflight_report.json'
    report = json.loads(report_path.read_text())
    manifest = json.loads((saved / 'source_manifest.json').read_text())
    receipt = json.loads((saved / 'independent_verification.json').read_text())
    expected_file_count = validate_inventory(manifest)
    if receipt.get('source_files_hash_checked') != expected_file_count:
        raise AssertionError('Independent receipt source-file count differs from exact inventory')
    if receipt.get('verifier_source_sha256') != sha_file(Path(__file__)):
        raise AssertionError('Independent receipt verifier-source hash differs from current code')
    if manifest['revisions'] != PINNED_REVISIONS or sha_file(saved / 'source_manifest.json') != report['sources_manifest_sha256']:
        raise AssertionError('Saved source snapshot identity mismatch')
    if not (report['status'] == 'masked_preflight' and report['score_values_exposed'] is False
            and receipt['metadata_only'] is True and receipt['pass_fail_threshold_applied'] is False
            and receipt['numeric_scores_retained'] is False and receipt['status'] == 'PASS'):
        raise AssertionError('Saved artifacts do not describe a masked-only verification')
    if report.get('feasible') != report['frames']['four_role']['feasible']:
        raise AssertionError('Top-level feasibility differs from the original four-role gate')
    if receipt['sources_manifest_sha256'] != report['sources_manifest_sha256'] or receipt['preflight_report_sha256'] != sha_file(report_path):
        raise AssertionError('Independent receipt does not match saved sources/report')
    tables = {}
    for name in ('masked_units', 'excluded_units'):
        compressed = saved / (name + '.csv.gz')
        h = hashlib.sha256()
        with gzip.open(compressed, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != report[name + '_sha256']:
            raise AssertionError('Uncompressed saved CSV hash mismatch: ' + name)
        with gzip.open(compressed, 'rt', newline='') as handle:
            tables[name] = list(csv.DictReader(handle))
    units = tables['masked_units']
    ids = {r['unit_id'] for r in units}
    if len(ids) != len(units) or len(ids) != receipt['masked_units_compared']:
        raise AssertionError('Saved masked unit count or uniqueness mismatch')
    if receipt['masked_units_sha256'] != report['masked_units_sha256']:
        raise AssertionError('Saved independent receipt differs from masked rows')
    tasks = {r['task_id'] for r in units}
    if len(tasks) != report['benchmark_rows']:
        raise AssertionError('Saved benchmark task count mismatch')
    if {r['unit_id'] for r in tables['excluded_units']} != {r['unit_id'] for r in units if r['common_four_role'] != '1'}:
        raise AssertionError('Saved exclusion list mismatch')
    count_checks = 0
    for frame in ('pr', 'four_role'):
        summary = report['frames'][frame]
        common = {r['task_id'] for r in units if r['common_' + frame] == '1'}
        actual_eligible = {p: {r['task_id'] for r in units if r['producer'] == p and r['eligible_' + frame] == '1'} for p in PRODUCERS}
        if common != set.intersection(*actual_eligible.values()):
            raise AssertionError('Saved common frame differs from producer intersection')
        rows = [r for r in units if r['common_' + frame] == '1']
        clusters = {r['cluster_id'] for r in rows}
        pilot = {r['cluster_id'] for r in rows if r['split_' + frame] == 'pilot'}
        confirmation = {r['cluster_id'] for r in rows if r['split_' + frame] == 'confirmation'}
        if pilot & confirmation or pilot | confirmation != clusters:
            raise AssertionError('Saved cluster split overlap or missing partition')
        ordered = sorted(clusters, key=lambda c: sha_text('refeval-external-v1|' + c))
        if pilot != set(ordered[:len(ordered) // 5]):
            raise AssertionError('Saved split differs from fixed hash rule')
        counts = {'common_tasks': len(common), 'units': len(rows), 'clusters': len(clusters),
                  'pilot_clusters': len(pilot), 'confirmation_clusters': len(confirmation),
                  'eligible_tasks_before_common_intersection': {p: len(v) for p, v in actual_eligible.items()}}
        for key, value in counts.items():
            if summary[key] != value or receipt['independent_counts']['frames'][frame][key] != value:
                raise AssertionError('Saved count mismatch: ' + frame + '/' + key)
            count_checks += 1
        retention = len(common) / len(tasks)
        gates = {'clusters_at_least_500': len(clusters) >= 500, 'pilot_clusters_at_least_100': len(pilot) >= 100,
                 'confirmation_clusters_at_least_300': len(confirmation) >= 300,
                 'retention_each_at_least_70pct': retention >= .7}
        if summary['gates'] != gates or summary['feasible'] != all(gates.values()):
            raise AssertionError('Saved feasibility decision differs from fixed gates')
        if any(value != retention for value in summary['retention_by_producer'].values()):
            raise AssertionError('Saved retention differs from common-frame denominator')
    return {'status': 'PASS', 'mode': 'saved_only', 'network_or_source_access': False,
            'masked_units': len(units), 'frame_count_checks': count_checks,
            'scope': 'Saved metadata/hash/gate consistency; source join is attested by the separate independent receipt'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--saved-only', type=Path, metavar='DIRECTORY')
    args = parser.parse_args()
    if args.saved_only:
        print(json.dumps(verify_saved(args.saved_only), indent=2))
        return
    if args.sources is None or args.preflight is None or args.output is None:
        parser.error('Full verification requires --sources, --preflight and --output')
    result = verify(args.sources, args.preflight, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'metadata_only', 'source_files_hash_checked', 'report_field_checks', 'masked_units_compared', 'masked_cell_checks')}, indent=2))


if __name__ == '__main__':
    main()
