"""Build deterministic new-judge jobs without opening saved reference scores.

Private JSONL contains prompt/response text and must not be redistributed. The
public-safe manifest records only hashes, source identities and structural counts.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from src.verify_wildifeval_preflight import PINNED_REVISIONS, PRODUCERS, validate_inventory

SOURCE_MANIFEST_SHA256 = '8454bb1e69b29565dd44f8591d583bb7ccd6a708214cc86292d07505b94551da'
MASKED_UNITS_SHA256 = '5d55fc361840ee6f198143b38494585dc9817c1ee4ee6ab192c85b17eb0ad9e1'
TEMPLATE_FILE_SHA256 = '2864bbed5efbf0ba4263e47617f1cac35e3dfd456987d27de5bc84a75c68ddac'
TEMPLATE_COMMIT = '82ebff7068b9faeadc5468d55935d62b1601d89c'


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key in job source')
        result[key] = value
    return result


def read_template(path):
    """Read the literal prompt without importing or executing upstream code."""
    tree = ast.parse(Path(path).read_text())
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROMPT_EVAL' for t in node.targets):
            value = node.value
            strip = (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                     and value.func.attr == 'strip' and not value.args and not value.keywords)
            value = ast.literal_eval(value.func.value if strip else value)
            if not isinstance(value, str):
                raise ValueError('Upstream prompt is not a literal string')
            found.append(value.strip() if strip else value)
    if len(found) != 1:
        raise ValueError('Missing or ambiguous upstream prompt template')
    # Formatting checks field names, while preserving the literal upstream text.
    found[0].format(instruction='instruction', response='response', constraint='constraint')
    return found[0]


def select_common_units(masked_rows, producers=PRODUCERS):
    """Select the five-producer complete saved-R frame from missingness only."""
    rows = list(masked_rows)
    if {r['producer'] for r in rows} != set(producers):
        raise ValueError('Masked frame has unexpected producer identities')
    ids = [r['unit_id'] for r in rows]
    pairs = [(r['producer'], r['task_id']) for r in rows]
    if len(ids) != len(set(ids)) or len(pairs) != len(set(pairs)):
        raise ValueError('Duplicate unit identity in masked frame')
    eligible = {p: set() for p in producers}
    for row in rows:
        reasons = json.loads(row['missing_reasons'])
        if not isinstance(reasons.get('r'), list):
            raise ValueError('Reference completeness mask missing')
        if reasons['r'] == []:
            eligible[row['producer']].add(row['task_id'])
    common = set.intersection(*(eligible[p] for p in producers))
    selected = [dict(r) for r in rows if r['task_id'] in common]
    task_clusters = defaultdict(set)
    for row in selected:
        task_clusters[row['task_id']].add(row['cluster_id'])
    if any(len(v) != 1 for v in task_clusters.values()):
        raise ValueError('Same task appears in inconsistent clusters')
    ordered = sorted({r['cluster_id'] for r in selected}, key=lambda c: sha_text('refeval-external-v1|' + c))
    pilot = set(ordered[:len(ordered) // 5])
    for row in selected:
        row['split'] = 'pilot' if row['cluster_id'] in pilot else 'confirmation'
    return sorted(selected, key=lambda r: (r['producer'], r['unit_id']))


def make_unit_jobs(row, criteria, prompt, response, template):
    """Validate every content link and render one immutable job per full criterion."""
    criteria = tuple(criteria)
    k = int(row['k'])
    if not 2 <= k <= 8 or len(criteria) != k or len(set(criteria)) != k:
        raise ValueError('Original complete criterion vector required')
    if not isinstance(prompt, str) or not isinstance(response, str) or not response.strip():
        raise ValueError('Missing prompt or saved response')
    if sha_text(prompt) != row['prompt_hash'] or sha_text(response) != row['output_hash']:
        raise ValueError('Saved prompt/output hash differs from masked frame')
    hashes = [sha_text(c) for c in criteria]
    if hashes != json.loads(row['constraint_hashes']):
        raise ValueError('Ordered criterion hashes differ from masked frame')
    expected_ids = [sha_text(row['unit_id'] + '|' + str(j) + '|' + h) for j, h in enumerate(hashes)]
    if expected_ids != json.loads(row['criterion_ids']):
        raise ValueError('Criterion identities differ from masked frame')
    expected_unit = sha_text('wildifeval-unit|' + row['task_id'] + '|' + row['producer'])
    if expected_unit != row['unit_id']:
        raise ValueError('Unit identity differs from task/producer link')
    jobs = []
    for j, criterion in enumerate(criteria):
        rendered = template.format(instruction=prompt, response=response, constraint=criterion)
        prompt_hash = sha_text(rendered)
        job_id = sha_text('wildifeval-new-p-v1|' + expected_ids[j] + '|' + prompt_hash)
        jobs.append({'job_id': job_id, 'unit_id': row['unit_id'], 'task_id': row['task_id'],
                     'cluster_id': row['cluster_id'], 'producer': row['producer'], 'criterion_index': j,
                     'criterion_id': expected_ids[j], 'k': k, 'split': row['split'], 'prompt': rendered,
                     'prompt_sha256': prompt_hash, 'output_hash': row['output_hash'], 'criterion_hash': hashes[j]})
    unit = {'unit_id': row['unit_id'], 'task_id': row['task_id'], 'cluster_id': row['cluster_id'],
            'producer': row['producer'], 'k': k, 'split': row['split'], 'criterion_ids': expected_ids,
            'criterion_hashes': hashes, 'job_ids': [j['job_id'] for j in jobs],
            'prompt_sha256': row['prompt_hash'], 'output_hash': row['output_hash']}
    return unit, jobs


def checked_inputs(sources, masked_csv, template_path, preflight_report=None):
    sources, masked_csv, template_path = Path(sources), Path(masked_csv), Path(template_path)
    manifest_path = sources / 'manifest.json'
    if sha_file(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError('Unexpected source manifest content')
    manifest = json.loads(manifest_path.read_text(), object_pairs_hook=unique_object)
    validate_inventory(manifest)
    if sha_file(masked_csv) != MASKED_UNITS_SHA256:
        raise ValueError('Masked unit table differs from verified frozen metadata')
    if sha_file(template_path) != TEMPLATE_FILE_SHA256:
        raise ValueError('Unexpected upstream grading template file')
    by_path = {e['path']: e for e in manifest['files']}
    consumed = ['benchmark/data/test-00000-of-00001.parquet'] + [f'predictions/model_predictions/{p}-0shot-wild-if-eval.json' for p in PRODUCERS]
    checked = {}
    for name in consumed:
        path = sources / name
        if path.stat().st_size != by_path[name]['bytes'] or sha_file(path) != by_path[name]['sha256']:
            raise ValueError('Consumed source differs from manifest: ' + name)
        checked[name] = by_path[name]['sha256']
    if preflight_report is not None:
        report = json.loads(Path(preflight_report).read_text())
        if (report['sources_manifest_sha256'] != SOURCE_MANIFEST_SHA256
                or report['masked_units_sha256'] != MASKED_UNITS_SHA256 or report['score_values_exposed'] is not False):
            raise ValueError('Preflight receipt does not match input metadata')
    return {'source_revisions': PINNED_REVISIONS, 'source_manifest_sha256': SOURCE_MANIFEST_SHA256,
            'masked_units_sha256': MASKED_UNITS_SHA256, 'template_file_sha256': TEMPLATE_FILE_SHA256,
            'consumed_source_sha256': checked,
            'preflight_report_sha256': sha_file(preflight_report) if preflight_report is not None else None}


def build_jobs(sources_dir, masked_csv, template_path, private_dir, manifest_path, preflight_report=None):
    sources, private_dir, manifest_path = Path(sources_dir), Path(private_dir), Path(manifest_path)
    # Existing artifacts are immutable. Interrupted partial files are also kept.
    targets = [private_dir / 'jobs.jsonl', private_dir / 'units.jsonl', manifest_path]
    if any(p.exists() or p.with_suffix(p.suffix + '.partial').exists() for p in targets):
        raise ValueError('Preserve existing job artifacts; choose a new run directory')
    inputs = checked_inputs(sources, masked_csv, template_path, preflight_report)
    template = read_template(template_path)
    import pyarrow.parquet as pq
    raw_benchmark = pq.read_table(sources / 'benchmark/data/test-00000-of-00001.parquet', columns=['conversation_id', 'decomposition']).to_pylist()
    benchmark = {}
    for row in raw_benchmark:
        task_id = sha_text('wildifeval-task|' + row['conversation_id'])
        if task_id in benchmark:
            raise ValueError('Duplicate benchmark task identity')
        benchmark[task_id] = tuple(row['decomposition'])
    with Path(masked_csv).open(newline='') as f:
        selected = select_common_units(csv.DictReader(f))
    if not selected:
        raise ValueError('No complete common saved-reference frame')
    private_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_tmp, units_tmp = [p.with_suffix(p.suffix + '.partial') for p in targets[:2]]
    counts = Counter(); producers = defaultdict(Counter)
    sets = {s: {'clusters': set(), 'tasks': set()} for s in ('pilot', 'confirmation')}
    job_ids, unit_ids = set(), set()
    with jobs_tmp.open('x', encoding='utf-8') as job_file, units_tmp.open('x', encoding='utf-8') as unit_file:
        for producer in sorted(PRODUCERS):
            source = sources / f'predictions/model_predictions/{producer}-0shot-wild-if-eval.json'
            raw = json.loads(source.read_text(), object_pairs_hook=unique_object)
            if 'predictions_key' in raw:
                wrapper = raw['predictions_key']
                raw = raw[wrapper] if isinstance(wrapper, str) and wrapper in raw else wrapper
            if not isinstance(raw, dict):
                raise ValueError('Unsupported producer-output wrapper')
            output_index = {}
            for prompt, messages in raw.items():
                key = sha_text(prompt)
                if key in output_index:
                    raise ValueError('Duplicate hashed prompt key')
                output_index[key] = (prompt, messages)
            for row in selected:
                if row['producer'] != producer:
                    continue
                prompt, messages = output_index[row['prompt_hash']]
                users = [m.get('content') for m in messages if isinstance(m, dict) and m.get('role') == 'user']
                responses = [m.get('content') for m in messages if isinstance(m, dict) and m.get('role') == 'assistant']
                if len(users) != 1 or users[0] != prompt or len(responses) != 1:
                    raise ValueError('Producer message roles do not identify one prompt/response')
                unit, jobs = make_unit_jobs(row, benchmark[row['task_id']], prompt, responses[0], template)
                if unit['unit_id'] in unit_ids or any(j['job_id'] in job_ids for j in jobs):
                    raise ValueError('Duplicate generated job or unit identifier')
                unit_ids.add(unit['unit_id']);job_ids.update(j['job_id'] for j in jobs)
                unit_file.write(json.dumps(unit, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
                for job in jobs:
                    job_file.write(json.dumps(job, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
                split = row['split']
                counts[split + '_units'] += 1;counts[split + '_criteria'] += len(jobs)
                producers[producer]['units'] += 1;producers[producer]['criteria'] += len(jobs)
                producers[producer][split + '_units'] += 1;producers[producer][split + '_criteria'] += len(jobs)
                sets[split]['clusters'].add(row['cluster_id']);sets[split]['tasks'].add(row['task_id'])
    if sets['pilot']['clusters'] & sets['confirmation']['clusters']:
        raise ValueError('Cluster leakage between pilot and confirmation')
    totals = {'jobs': len(job_ids), 'units': len(unit_ids),
              'tasks': len(sets['pilot']['tasks'] | sets['confirmation']['tasks']),
              'clusters': len(sets['pilot']['clusters'] | sets['confirmation']['clusters'])}
    if totals != {'jobs': 134485, 'units': 37090, 'tasks': 7418, 'clusters': 6847}:
        raise ValueError('Pinned complete-reference frame count differs from independently verified inventory')
    jobs_tmp.rename(targets[0]);units_tmp.rename(targets[1])
    splits = {s: {'clusters': len(sets[s]['clusters']), 'tasks': len(sets[s]['tasks']),
                  'units': counts[s + '_units'], 'criteria': counts[s + '_criteria']} for s in sets}
    result = {'schema_version': 1, 'status': 'reference_value_free_jobs_built', 'counts': totals, 'splits': splits,
              'producers': dict(producers), 'reference_files_opened': False, 'reference_values_accessed': False,
              'new_model_calls': 0, 'generation_results_included': False,
              'scope': 'All five producer outputs on the complete saved-DeepSeek-reference common frame; new judge completeness is not yet known',
              'source_inputs': inputs, 'template_commit': TEMPLATE_COMMIT, 'template_name': 'PROMPT_EVAL',
              'rendered_template_sha256': sha_text(template), 'builder_source_sha256': sha_file(Path(__file__)),
              'job_order': 'lexicographic producer, unit_id, then original criterion position',
              'split_rule': 'First floor(0.20 G) clusters sorted by SHA256(refeval-external-v1|cluster_id) are pilot',
              'job_id_rule': 'SHA256(wildifeval-new-p-v1|criterion_id|rendered_prompt_sha256)',
              'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha_file(p)} for p in targets[:2]},
              'distribution': 'Do not publish jobs.jsonl: it contains upstream prompts and outputs. This manifest and unit hashes contain no source text or reference values.'}
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--masked', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--preflight-report', type=Path)
    args = parser.parse_args()
    result = build_jobs(args.sources, args.masked, args.template, args.private_dir, args.manifest, args.preflight_report)
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'splits': result['splits']}, indent=2))


if __name__ == '__main__':
    main()
