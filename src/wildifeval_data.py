"""Masked ingestion and staged access for the pinned WildIFEval release.

No prompt, output, explanation, or numerical score is emitted by preflight.
The source JSON cache is local only: it is not a redistributable artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PRODUCERS = ('deepseek-v3', 'llama3.1-405b', 'llama3.3-70b', 'mistral-large', 'qwen2.5-72b')
ROLES = {'p': 'llama3.1-70b', 'r': 'deepseek-v3', 'r2': 'gpt-4o-2024-08-06', 'q': 'qwen2.5-72b'}
REVISIONS = {'benchmark': 'fc5f47b2b1be19ac95986624a4c0d8fd25dd316c',
             'predictions': 'e34bc0fcc01b7f1d49032878563e8c32393c9d59'}
FRAME_ROLES = {'four_role': ('p', 'q', 'r', 'r2'), 'pr': ('p', 'r')}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize_prompt(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).split())


class UniqueMap(dict):
    """Keep the first key and record collisions instead of silently overwriting."""
    def __init__(self, pairs):
        super().__init__()
        self.duplicates = set()
        for key, value in pairs:
            if key in self:
                self.duplicates.add(key)
            else:
                self[key] = value


def read_json(path: Path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f, object_pairs_hook=UniqueMap)


def score_file(producer: str, role: str) -> str:
    sub = 'other_judges/' if role in ('q', 'r2') else ''
    return f'predictions/llm_aaj_scores/{sub}llm-aaj-{ROLES[role]}.{producer}-0shot-wild-if-eval.json'


def output_file(producer: str) -> str:
    return f'predictions/model_predictions/{producer}-0shot-wild-if-eval.json'


def source_paths():
    paths = ['benchmark/data/test-00000-of-00001.parquet', 'benchmark/README.md', 'predictions/README.md']
    for producer in PRODUCERS:
        paths.append(output_file(producer))
        paths.extend(score_file(producer, role) for role in ROLES)
    return paths


def _expected_entries(expected_manifest: Path | None):
    if expected_manifest is None:
        candidate = Path('artifacts/expected/wildifeval_external/source_manifest.json')
        if not candidate.exists():
            return None
    else:
        candidate = Path(expected_manifest)
    expected = read_json(candidate)
    if expected.get('revisions') != REVISIONS:
        raise ValueError('Expected source revisions differ from pinned revisions')
    entries = {entry['path']: entry for entry in expected['files']}
    if len(entries) != len(expected['files']) or set(entries) != set(source_paths()):
        raise ValueError('Expected source manifest has a different or duplicate file inventory')
    return entries


def _check_expected_file(path: Path, rel: str, entries):
    if entries is not None:
        expected = entries[rel]
        if path.stat().st_size != expected['bytes'] or file_sha256(path) != expected['sha256']:
            raise ValueError('Source differs from published expected manifest: ' + rel)


def download_sources(sources_dir: Path, expected_manifest: Path | None = None):
    """Fetch immutable URLs and verify published expected hashes before acceptance.

    If present, the default expected manifest is
    artifacts/expected/wildifeval_external/source_manifest.json. A bad existing
    cache or downloaded temporary file is refused, never silently relabelled.
    """
    sources_dir = Path(sources_dir)
    sources_dir.mkdir(parents=True, exist_ok=True)
    expected = _expected_entries(expected_manifest)
    if (sources_dir / 'manifest.json').exists():
        validate_sources(sources_dir)
    for rel in source_paths():
        family, name = rel.split('/', 1)
        repo = 'wild-if-eval' + ('-predictions' if family == 'predictions' else '')
        path = sources_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix(path.suffix + '.partial')
            url = f'https://huggingface.co/datasets/gililior/{repo}/resolve/{REVISIONS[family]}/{name}'
            subprocess.run(['curl', '-fsSL', '--retry', '2', '--connect-timeout', '20', '--max-time', '300', url, '-o', str(tmp)], check=True)
            _check_expected_file(tmp, rel, expected)
            tmp.rename(path)
        _check_expected_file(path, rel, expected)
    return write_source_manifest(sources_dir, expected_manifest)


def write_source_manifest(sources_dir: Path, expected_manifest: Path | None = None):
    expected = _expected_entries(expected_manifest)
    entries = []
    for rel in source_paths():
        path = Path(sources_dir) / rel
        family, name = rel.split('/', 1)
        _check_expected_file(path, rel, expected)
        repo = 'gililior/wild-if-eval' + ('-predictions' if family == 'predictions' else '')
        entries.append({'path': rel, 'repo_id': repo, 'revision': REVISIONS[family], 'upstream_path': name,
                        'bytes': path.stat().st_size, 'sha256': file_sha256(path)})
    manifest = {'schema_version': 1, 'revisions': REVISIONS, 'files': entries,
                'distribution': 'local sealed cache; do not redistribute text or score JSON'}
    out = Path(sources_dir) / 'manifest.json'
    if out.exists() and json.loads(out.read_text()) != manifest:
        raise ValueError('Existing source manifest differs; preserve it and use a new snapshot directory')
    out.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def validate_sources(sources_dir: Path):
    manifest = read_json(Path(sources_dir) / 'manifest.json')
    for entry in manifest['files']:
        if file_sha256(Path(sources_dir) / entry['path']) != entry['sha256']:
            raise ValueError('Source hash mismatch: ' + entry['path'])
    return manifest


def masked_score_status(value):
    """Only validity/missingness, never threshold classification, in preflight."""
    if isinstance(value, bool):
        return 'unparseable'
    if isinstance(value, (int, float)):
        return 'present' if math.isfinite(value) and 0 <= value <= 1 else 'unparseable'
    if value == 'ERR':
        return 'error'
    return 'missing' if value is None else 'unparseable'


def masked_score_records(path: Path):
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError('Score JSON is not an object: ' + path.name)
    records = {}
    types = Counter()
    for prompt, row in raw.items():
        reasons = []
        if prompt in raw.duplicates:
            reasons.append('duplicate_task_key')
        if not isinstance(row, dict) or not isinstance(row.get('scores'), dict):
            records[prompt] = {'keys': (), 'masks': {}, 'reasons': reasons + ['missing_score_object']}
            continue
        if 'scores' in row.duplicates:
            reasons.append('duplicate_score_object')
        scores = row['scores']
        if scores.duplicates:
            reasons.append('duplicate_constraint_key')
        masks = {key: masked_score_status(value) for key, value in scores.items()}
        types.update(type(value).__name__ for value in scores.values())
        records[prompt] = {'keys': tuple(scores), 'masks': masks, 'reasons': reasons}
    return records, {'task_entries': len(raw), 'duplicate_task_keys': len(raw.duplicates),
                     'score_value_types': dict(sorted(types.items()))}


def masked_output_records(path: Path):
    raw = read_json(path)
    if isinstance(raw, dict) and 'predictions_key' in raw:
        # Source loader permits an object directly below this named wrapper.
        wrapped = raw['predictions_key']
        if isinstance(wrapped, str) and wrapped in raw:
            raw = raw[wrapped]
        elif isinstance(wrapped, dict):
            raw = wrapped
        else:
            raise ValueError('Unsupported predictions_key wrapper')
    if not isinstance(raw, dict):
        raise ValueError('Predictions JSON is not an object')
    records = {}
    for prompt, messages in raw.items():
        reasons = []
        if prompt in raw.duplicates:
            reasons.append('duplicate_output_task_key')
        if not isinstance(messages, list):
            records[prompt] = {'reasons': reasons + ['invalid_output_messages']}
            continue
        if any(isinstance(x, UniqueMap) and ({'role', 'content'} & x.duplicates) for x in messages):
            reasons.append('duplicate_output_message_field')
        users = [x.get('content') for x in messages if isinstance(x, dict) and x.get('role') == 'user']
        assistants = [x.get('content') for x in messages if isinstance(x, dict) and x.get('role') == 'assistant']
        if len(users) != 1 or users[0] != prompt:
            reasons.append('output_prompt_mismatch')
        if len(assistants) != 1 or not isinstance(assistants[0], str) or not assistants[0].strip():
            reasons.append('missing_or_ambiguous_output')
        records[prompt] = {'reasons': reasons,
                           'output_hash': digest(assistants[0]) if len(assistants) == 1 and isinstance(assistants[0], str) else '',
                           'prompt_hash': digest(prompt), 'normalized_prompt_hash': digest(normalize_prompt(prompt))}
    return records, {'task_entries': len(raw), 'duplicate_task_keys': len(raw.duplicates)}


class UnionFind:
    def __init__(self, keys):
        self.parents = {k: k for k in keys}
    def find(self, key):
        if self.parents[key] != key:
            self.parents[key] = self.find(self.parents[key])
        return self.parents[key]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def split_clusters(cluster_ids):
    ordered = sorted(set(cluster_ids), key=lambda c: digest('refeval-external-v1|' + c))
    pilot = set(ordered[:math.floor(.2 * len(ordered))])
    return {c: 'pilot' if c in pilot else 'confirmation' for c in ordered}


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_masked_frame(benchmark_rows, outputs, scores):
    """Pure masked join; score records must be produced by masked_score_records.

    The benchmark has no task-text column. Its complete exact constraint set is
    matched to score-object keys. Any non-unique decomposition or prompt mapping
    is excluded, never resolved from numerical labels or dictionary order.
    """
    signature_ids = defaultdict(list)
    tasks = {}
    invalid = {}
    for i, row in enumerate(benchmark_rows):
        conv = row.get('conversation_id')
        constraints = row.get('decomposition')
        if not isinstance(conv, str) or not isinstance(constraints, (list, tuple)) or not all(isinstance(c, str) for c in constraints):
            raise ValueError('Invalid benchmark row schema')
        task_id = digest('wildifeval-task|' + conv)
        if task_id in tasks:
            raise ValueError('Duplicate benchmark conversation ID')
        tasks[task_id] = {'task_id': task_id, 'conversation_id_hash': digest(conv), 'constraints': tuple(constraints), 'k': len(constraints)}
        signature_ids[tuple(sorted(constraints))].append(task_id)
        if not 2 <= len(constraints) <= 8:
            invalid.setdefault(task_id, []).append('out_of_scope_criterion_count')
        if len(set(constraints)) != len(constraints):
            invalid.setdefault(task_id, []).append('duplicate_benchmark_constraint')
    candidates = defaultdict(set)
    for producer in PRODUCERS:
        for role in ROLES:
            for prompt, record in scores[producer][role].items():
                candidates[prompt].update(signature_ids.get(tuple(sorted(record['keys'])), ()))
    prompt_by_task = defaultdict(set)
    for prompt, ids in candidates.items():
        if len(ids) == 1:
            prompt_by_task[next(iter(ids))].add(prompt)
    for task_id, task in tasks.items():
        if len(signature_ids[tuple(sorted(task['constraints']))]) != 1:
            invalid.setdefault(task_id, []).append('ambiguous_benchmark_decomposition')
        if len(prompt_by_task[task_id]) != 1:
            invalid.setdefault(task_id, []).append('missing_or_ambiguous_prompt_link')
    uf = UnionFind(tasks)
    cluster_tokens = {}
    units = []
    for task_id, task in sorted(tasks.items()):
        prompts = prompt_by_task[task_id]
        prompt = next(iter(prompts)) if len(prompts) == 1 else None
        constraint_hashes = [digest(c) for c in task['constraints']]
        for producer in PRODUCERS:
            unit_id = digest('wildifeval-unit|' + task_id + '|' + producer)
            base_reasons = list(invalid.get(task_id, ()))
            out = outputs[producer].get(prompt)
            if out is None:
                out = {};base_reasons.append('missing_output')
            base_reasons.extend(out.get('reasons', ()))
            row = {'unit_id': unit_id, 'producer': producer, 'task_id': task_id,
                   'conversation_id_hash': task['conversation_id_hash'], 'k': task['k'],
                   'prompt_hash': out.get('prompt_hash', digest(prompt) if prompt is not None else ''),
                   'normalized_prompt_hash': out.get('normalized_prompt_hash', digest(normalize_prompt(prompt)) if prompt is not None else ''),
                   'output_hash': out.get('output_hash', ''),
                   'constraint_hashes': json.dumps(constraint_hashes),
                   'criterion_ids': json.dumps([digest(unit_id + '|' + str(j) + '|' + h) for j, h in enumerate(constraint_hashes)])}
            for token in [('prompt', row['normalized_prompt_hash']), ('output', row['output_hash'])]:
                if token[1]:
                    if token in cluster_tokens:
                        uf.union(task_id, cluster_tokens[token])
                    else:
                        cluster_tokens[token] = task_id
            all_reasons = {}
            for role in ROLES:
                rec = scores[producer][role].get(prompt)
                reasons = list(base_reasons)
                positions = []
                if rec is None:
                    reasons.append('missing_task_scores');positions = list(range(task['k']))
                else:
                    reasons.extend(rec['reasons'])
                    if set(rec['keys']) != set(task['constraints']):
                        reasons.append('constraint_set_mismatch')
                    for j, criterion in enumerate(task['constraints']):
                        state = rec['masks'].get(criterion, 'missing')
                        if state != 'present':
                            positions.append(j);reasons.append('score_' + state)
                row[role + '_missing_positions'] = json.dumps(positions)
                all_reasons[role] = sorted(set(reasons))
            row['missing_reasons'] = json.dumps(all_reasons, sort_keys=True)
            for frame, roles in FRAME_ROLES.items():
                row['eligible_' + frame] = int(not any(all_reasons[r] for r in roles))
            units.append(row)
    groups = defaultdict(list)
    for task_id in tasks:
        groups[uf.find(task_id)].append(task_id)
    cluster_map = {task_id: digest('wildifeval-cluster|' + '|'.join(sorted(members))) for members in groups.values() for task_id in members}
    for row in units:
        row['cluster_id'] = cluster_map[row['task_id']]
    by_producer = {p: [r for r in units if r['producer'] == p] for p in PRODUCERS}
    frames = {}
    for frame in FRAME_ROLES:
        eligible_sets = {p: {r['task_id'] for r in by_producer[p] if r['eligible_' + frame]} for p in PRODUCERS}
        common = set.intersection(*eligible_sets.values())
        clusters = {cluster_map[t] for t in common}
        split = split_clusters(clusters)
        for row in units:
            row['common_' + frame] = int(row['task_id'] in common)
            row['split_' + frame] = split.get(row['cluster_id'], '') if row['task_id'] in common else ''
        n_pilot = sum(v == 'pilot' for v in split.values())
        retention = {p: len(common) / len(tasks) for p in PRODUCERS}
        gates = {'clusters_at_least_500': len(clusters) >= 500, 'pilot_clusters_at_least_100': n_pilot >= 100,
                 'confirmation_clusters_at_least_300': len(clusters) - n_pilot >= 300,
                 'retention_each_at_least_70pct': all(v >= .7 for v in retention.values())}
        frames[frame] = {'common_tasks': len(common), 'units': len(common) * len(PRODUCERS), 'clusters': len(clusters),
                         'pilot_clusters': n_pilot, 'confirmation_clusters': len(clusters) - n_pilot,
                         'eligible_tasks_before_common_intersection': {p: len(s) for p, s in eligible_sets.items()},
                         'retention_by_producer': retention, 'gates': gates, 'feasible': all(gates.values()),
                         'largest_cluster_tasks': max((sum(t in common for t in members) for members in groups.values()), default=0)}
    excluded = []
    for row in units:
        if not row['common_four_role']:
            excluded.append({k: row[k] for k in ('unit_id', 'producer', 'task_id', 'missing_reasons', 'eligible_four_role', 'common_four_role')})
    role_complete_counts = {p: {role: sum(not json.loads(row['missing_reasons'])[role] for row in by_producer[p]) for role in ROLES} for p in PRODUCERS}
    role_constraint_comparisons = {}
    for producer in PRODUCERS:
        p_records, r_records = scores[producer]['p'], scores[producer]['r']
        shared = sorted(set(p_records) & set(r_records))
        k_pairs = Counter((len(p_records[t]['keys']), len(r_records[t]['keys'])) for t in shared)
        role_constraint_comparisons[producer] = {
            'shared_task_text_keys': len(shared),
            'same_exact_constraint_set': sum(set(p_records[t]['keys']) == set(r_records[t]['keys']) for t in shared),
            'same_whitespace_normalized_constraint_set': sum({normalize_prompt(c) for c in p_records[t]['keys']} == {normalize_prompt(c) for c in r_records[t]['keys']} for t in shared),
            'different_constraint_count': sum(len(p_records[t]['keys']) != len(r_records[t]['keys']) for t in shared),
            'criterion_count_pairs': {str(a) + ',' + str(b): n for (a, b), n in sorted(k_pairs.items())},
        }
    reason_counts = {}
    for producer in PRODUCERS:
        reason_counts[producer] = {}
        for role in ROLES:
            counts = Counter(reason for r in by_producer[producer] for reason in json.loads(r['missing_reasons'])[role])
            reason_counts[producer][role] = dict(sorted(counts.items()))
    report = {'schema_version': 1, 'status': 'masked_preflight', 'feasible': frames['four_role']['feasible'],
              'benchmark_rows': len(tasks), 'benchmark_k_counts': dict(sorted(Counter(t['k'] for t in tasks.values()).items())),
              'unique_constraint_sets': len(signature_ids), 'ambiguous_constraint_sets': sum(len(v) > 1 for v in signature_ids.values()),
              'ambiguous_constraint_set_rows': sum(len(v) for v in signature_ids.values() if len(v) > 1),
              'join_method': 'exact complete constraint-set keys uniquely identify benchmark conversation; no text-key overwrite',
              'judge_output_link_limit': 'Released judge files identify producer and task text, but do not embed the evaluated output hash',
              'frames': frames, 'exclusion_reason_counts': reason_counts,
              'complete_benchmark_units_by_role': role_complete_counts,
              'p_r_constraint_key_comparison': role_constraint_comparisons,
              'total_clusters_before_complete_case': len(groups), 'largest_cluster_before_complete_case': max(map(len, groups.values()), default=0),
              'score_values_exposed': False}
    return units, excluded, report


def preflight(sources_dir: Path, output_dir: Path):
    sources_dir, output_dir = Path(sources_dir), Path(output_dir)
    validate_sources(sources_dir)
    import pyarrow.parquet as pq  # optional external-validation dependency
    benchmark = pq.read_table(sources_dir / 'benchmark/data/test-00000-of-00001.parquet').to_pylist()
    scores, outputs, inventory = {}, {}, {}
    for producer in PRODUCERS:
        scores[producer] = {}
        outputs[producer], inventory[output_file(producer)] = masked_output_records(sources_dir / output_file(producer))
        for role in ROLES:
            scores[producer][role], inventory[score_file(producer, role)] = masked_score_records(sources_dir / score_file(producer, role))
    units, excluded, report = build_masked_frame(benchmark, outputs, scores)
    report['inventory'] = inventory
    report['sources_manifest_sha256'] = file_sha256(sources_dir / 'manifest.json')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('masked_units.csv', 'excluded_units.csv', 'report.json'):
        if (output_dir / name).exists():
            raise ValueError('Preflight outputs already exist; do not overwrite: ' + name)
    _write_csv(output_dir / 'masked_units.csv', units, list(units[0]))
    _write_csv(output_dir / 'excluded_units.csv', excluded, ['unit_id', 'producer', 'task_id', 'missing_reasons', 'eligible_four_role', 'common_four_role'])
    report['masked_units_sha256'] = file_sha256(output_dir / 'masked_units.csv')
    report['excluded_units_sha256'] = file_sha256(output_dir / 'excluded_units.csv')
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def _validate_receipt(receipt_path: Path, project_root: Path, expected_stage: str):
    receipt = read_json(receipt_path)
    if receipt.get('stage') != expected_stage:
        raise ValueError('Wrong access-receipt stage')
    key = 'frozen_files' if expected_stage == 'design_freeze' else 'prediction_files'
    if not receipt.get(key):
        raise ValueError('Receipt does not freeze files')
    for rel, expected in receipt[key].items():
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Receipt paths must be project-relative')
        if file_sha256(Path(project_root) / path) != expected:
            raise ValueError('Receipt file hash mismatch: ' + rel)
    return receipt


def load_stage(sources_dir: Path, preflight_dir: Path, design_receipt: Path, *, stage: str,
               project_root: Path, prediction_receipt: Path | None = None):
    """Reveal only the requested partition after validating the immutable gates.

    Return lower-case p/q/r/r2 tuples. Confirmation metadata includes p/q only.
    Alternative frames require a dated, explicit amendment named in the receipt.
    """
    if stage not in ('pilot', 'confirmation_metadata', 'confirmation'):
        raise ValueError('Unknown access stage')
    sources_dir, preflight_dir = Path(sources_dir), Path(preflight_dir)
    receipt = _validate_receipt(Path(design_receipt), Path(project_root), 'design_freeze')
    report = read_json(preflight_dir / 'report.json')
    if file_sha256(preflight_dir / 'report.json') != receipt.get('preflight_sha256'):
        raise ValueError('Preflight receipt mismatch')
    if file_sha256(sources_dir / 'manifest.json') != receipt.get('sources_manifest_sha256'):
        raise ValueError('Source-manifest receipt mismatch')
    for name, key in [('masked_units.csv', 'masked_units_sha256'), ('excluded_units.csv', 'excluded_units_sha256')]:
        if file_sha256(preflight_dir / name) != report[key]:
            raise ValueError('Masked frame hash mismatch')
    frame = receipt.get('analysis_frame', 'four_role')
    if frame not in FRAME_ROLES or not report['frames'][frame]['feasible']:
        raise ValueError('Feasibility gate failed for selected frame')
    if frame != 'four_role' and (not receipt.get('amendment_path') or receipt['amendment_path'] not in receipt['frozen_files']):
        raise ValueError('Alternative frame requires a frozen pre-outcome amendment')
    if stage == 'confirmation':
        if prediction_receipt is None:
            raise ValueError('Confirmation reference remains sealed without prediction receipt')
        prediction = _validate_receipt(Path(prediction_receipt), Path(project_root), 'prediction_freeze')
        if prediction.get('design_receipt_sha256') != file_sha256(Path(design_receipt)):
            raise ValueError('Prediction receipt does not match design freeze')
    validate_sources(sources_dir)
    wanted_split = 'pilot' if stage == 'pilot' else 'confirmation'
    with (preflight_dir / 'masked_units.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['common_' + frame] == '1' and r['split_' + frame] == wanted_split]
    units = []
    for row in rows:
        units.append({'unit_id': row['unit_id'], 'producer': row['producer'], 'cluster_id': row['cluster_id'], 'k': int(row['k']),
                      'criterion_ids': tuple(json.loads(row['criterion_ids'])), '_prompt_hash': row['prompt_hash'],
                      '_constraint_hashes': tuple(json.loads(row['constraint_hashes']))})
    roles = [r for r in FRAME_ROLES[frame] if r in ('p', 'q') or stage != 'confirmation_metadata']
    for producer in PRODUCERS:
        producer_units = [u for u in units if u['producer'] == producer]
        selected = {u['_prompt_hash']: u for u in producer_units}
        if len(selected) != len(producer_units):
            raise ValueError('Duplicate frozen prompt hash; refusing overwrite')
        for role in roles:
            raw = read_json(sources_dir / score_file(producer, role))
            for prompt, record in raw.items():
                unit = selected.get(digest(prompt))
                if unit is None:
                    continue
                scores = {digest(c): value for c, value in record['scores'].items()}
                values = [scores[h] for h in unit['_constraint_hashes']]
                if any(masked_score_status(value) != 'present' for value in values):
                    raise ValueError('Previously complete score became missing')
                unit[role] = tuple(int(value >= .5) for value in values)
            if any(role not in unit for unit in selected.values()):
                raise ValueError('A frozen unit is absent from source')
    for unit in units:
        del unit['_prompt_hash'], unit['_constraint_hashes']
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('download', 'preflight'))
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-manifest', type=Path, help='Published source hashes; uses artifacts/expected/wildifeval_external/source_manifest.json when present')
    args = parser.parse_args()
    if args.command == 'download':
        result = download_sources(args.sources, args.expected_manifest)
        print(json.dumps({'files': len(result['files']), 'revisions': result['revisions']}))
    else:
        if args.output is None:
            parser.error('--output is required for preflight')
        report = preflight(args.sources, args.output)
        print(json.dumps({'status': report['status'], 'feasible': report['feasible'], 'frames': report['frames']}, indent=2))


if __name__ == '__main__':
    main()
