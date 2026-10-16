"""Masking, joins and staged access use synthetic labels only."""
import csv
import json
from pathlib import Path

import pytest

from src.wildifeval_data import (
    PRODUCERS, ROLES, build_masked_frame, digest, file_sha256, load_stage,
    masked_output_records, masked_score_records, masked_score_status,
    normalize_prompt, score_file, split_clusters,
)


def fixture_frame():
    benchmark = [{'conversation_id': f'c{i}', 'decomposition': [f'a{i}', f'b{i}']} for i in range(3)]
    outputs = {}
    scores = {}
    for p in PRODUCERS:
        outputs[p] = {}
        scores[p] = {role: {} for role in ROLES}
        for i, row in enumerate(benchmark):
            prompt = f'task {i}'
            outputs[p][prompt] = {'reasons': [], 'prompt_hash': digest(prompt),
                                  'normalized_prompt_hash': digest(prompt), 'output_hash': digest(p + str(i))}
            for role in ROLES:
                scores[p][role][prompt] = {'keys': tuple(row['decomposition']),
                                          'masks': {c: 'present' for c in row['decomposition']}, 'reasons': []}
    return benchmark, outputs, scores


def test_mask_status_never_collapses_error_to_fail():
    assert masked_score_status(0.0) == masked_score_status(1.0) == 'present'
    for v in [float('nan'), float('inf'), -1, 2, True, '0.8']:
        assert masked_score_status(v) == 'unparseable'
    assert masked_score_status('ERR') == 'error'
    assert masked_score_status(None) == 'missing'


def test_duplicate_json_keys_are_not_silently_overwritten(tmp_path):
    p = tmp_path / 'scores.json'
    p.write_text('{"task":{"scores":{"a":0.1,"a":0.9,"b":"ERR"}},"task":{"scores":{"a":0.9}}}')
    records, counts = masked_score_records(p)
    assert counts['duplicate_task_keys'] == 1
    assert set(records['task']['reasons']) == {'duplicate_task_key', 'duplicate_constraint_key'}
    assert records['task']['masks'] == {'a': 'present', 'b': 'error'}
    assert '0.1' not in json.dumps(records) and '0.9' not in json.dumps(records)


def test_output_requires_the_same_prompt_and_one_response(tmp_path):
    p = tmp_path / 'outputs.json'
    p.write_text(json.dumps({'predictions_key': {'a': [{'role': 'user', 'content': 'b'}, {'role': 'assistant', 'content': 'answer'}]}}))
    records, _ = masked_output_records(p)
    assert records['a']['reasons'] == ['output_prompt_mismatch']
    assert 'answer' not in json.dumps(records)


def test_missing_extra_judge_excludes_whole_task_but_not_pr_frame():
    benchmark, outputs, scores = fixture_frame()
    scores[PRODUCERS[0]]['r2']['task 0']['masks']['a0'] = 'error'
    units, excluded, report = build_masked_frame(benchmark, outputs, scores)
    assert report['frames']['four_role']['common_tasks'] == 2
    assert report['frames']['pr']['common_tasks'] == 3
    assert len(excluded) == 5  # one producer's failure excludes the common task from all cells
    assert all(r['k'] == 2 for r in units)  # criterion dropping is forbidden
    assert report['score_values_exposed'] is False


def test_ambiguous_decomposition_is_excluded_without_arbitrary_match():
    benchmark, outputs, scores = fixture_frame()
    benchmark.append({'conversation_id': 'another', 'decomposition': ['a0', 'b0']})
    units, _, report = build_masked_frame(benchmark, outputs, scores)
    assert report['ambiguous_constraint_set_rows'] == 2
    assert report['frames']['pr']['common_tasks'] == 2
    assert sum(r['common_pr'] for r in units) == 10


def test_complete_constraint_set_required_even_when_other_role_links():
    benchmark, outputs, scores = fixture_frame()
    scores[PRODUCERS[0]]['p']['task 0']['keys'] = ('a0',)
    scores[PRODUCERS[0]]['p']['task 0']['masks'] = {'a0': 'present'}
    units, _, report = build_masked_frame(benchmark, outputs, scores)
    assert report['frames']['pr']['common_tasks'] == 2
    row = next(r for r in units if r['producer'] == PRODUCERS[0] and r['prompt_hash'] == digest('task 0'))
    assert json.loads(row['p_missing_positions']) == [1]


def test_duplicate_outputs_cluster_across_producers_and_tasks():
    benchmark, outputs, scores = fixture_frame()
    outputs[PRODUCERS[0]]['task 0']['output_hash'] = 'same'
    outputs[PRODUCERS[-1]]['task 1']['output_hash'] = 'same'
    units, _, report = build_masked_frame(benchmark, outputs, scores)
    cluster0 = {r['cluster_id'] for r in units if r['prompt_hash'] == digest('task 0')}
    cluster1 = {r['cluster_id'] for r in units if r['prompt_hash'] == digest('task 1')}
    assert cluster0 == cluster1
    assert report['largest_cluster_before_complete_case'] == 2
    assert report['frames']['four_role']['clusters'] == 2


def test_split_is_order_invariant_and_exact_twenty_percent():
    keys = [f'cluster{i}' for i in range(17)]
    assert split_clusters(keys) == split_clusters(list(reversed(keys)))
    assert list(split_clusters(keys).values()).count('pilot') == 3
    assert normalize_prompt(' A\n  b ') == 'A b'
    assert normalize_prompt('A') != normalize_prompt('a')


def staged_fixture(tmp_path):
    sources = tmp_path / 'sources'; sources.mkdir()
    preflight = tmp_path / 'preflight';preflight.mkdir()
    rows = []
    files = []
    for producer in PRODUCERS:
        for split, prompt in [('pilot', 'pilot task'), ('confirmation', 'sealed task')]:
            rows.append({'unit_id': producer + '-' + split, 'producer': producer, 'cluster_id': split, 'k': 2,
                         'criterion_ids': '["id1","id2"]', 'constraint_hashes': json.dumps([digest('a'), digest('b')]),
                         'prompt_hash': digest(prompt), 'common_four_role': 1, 'split_four_role': split})
        for role in ROLES:
            path = sources / score_file(producer, role);path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'pilot task': {'scores': {'a': .8, 'b': .1}},
                                        'sealed task': {'scores': {'a': .1, 'b': .8}}}))
            files.append({'path': str(path.relative_to(sources)), 'sha256': file_sha256(path)})
    (sources / 'manifest.json').write_text(json.dumps({'files': files}))
    with (preflight / 'masked_units.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (preflight / 'excluded_units.csv').write_text('unit_id\n')
    report = {'frames': {'four_role': {'feasible': True}},
              'masked_units_sha256': file_sha256(preflight / 'masked_units.csv'),
              'excluded_units_sha256': file_sha256(preflight / 'excluded_units.csv')}
    (preflight / 'report.json').write_text(json.dumps(report))
    (tmp_path / 'protocol.md').write_text('synthetic fixed protocol')
    receipt = {'stage': 'design_freeze', 'preflight_sha256': file_sha256(preflight / 'report.json'),
               'sources_manifest_sha256': file_sha256(sources / 'manifest.json'),
               'frozen_files': {'protocol.md': file_sha256(tmp_path / 'protocol.md')}}
    design = tmp_path / 'design.json';design.write_text(json.dumps(receipt))
    return sources, preflight, design


def test_stage_loader_never_returns_confirmation_reference_without_prediction_freeze(tmp_path):
    sources, preflight, design = staged_fixture(tmp_path)
    pilot = load_stage(sources, preflight, design, stage='pilot', project_root=tmp_path)
    assert all(u['cluster_id'] == 'pilot' and u['r'] == (1, 0) for u in pilot)
    metadata = load_stage(sources, preflight, design, stage='confirmation_metadata', project_root=tmp_path)
    assert all(u['cluster_id'] == 'confirmation' and 'r' not in u and 'r2' not in u for u in metadata)
    with pytest.raises(ValueError, match='remains sealed'):
        load_stage(sources, preflight, design, stage='confirmation', project_root=tmp_path)
    (tmp_path / 'predictions.json').write_text('{"frozen": true}')
    pred = {'stage': 'prediction_freeze', 'design_receipt_sha256': file_sha256(design),
            'prediction_files': {'predictions.json': file_sha256(tmp_path / 'predictions.json')}}
    pred_path = tmp_path / 'prediction_receipt.json';pred_path.write_text(json.dumps(pred))
    confirmed = load_stage(sources, preflight, design, stage='confirmation', project_root=tmp_path, prediction_receipt=pred_path)
    assert all(u['r'] == (0, 1) for u in confirmed)
    (tmp_path / 'predictions.json').write_text('{"edited_after_freeze": true}')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_stage(sources, preflight, design, stage='confirmation', project_root=tmp_path, prediction_receipt=pred_path)


def test_loader_rejects_changed_masked_frame(tmp_path):
    sources, preflight, design = staged_fixture(tmp_path)
    with (preflight / 'masked_units.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='Masked frame hash'):
        load_stage(sources, preflight, design, stage='pilot', project_root=tmp_path)


def test_loader_rejects_failed_gate_before_labels(tmp_path):
    sources, preflight, design = staged_fixture(tmp_path)
    report = json.loads((preflight / 'report.json').read_text())
    report['frames']['four_role']['feasible'] = False
    (preflight / 'report.json').write_text(json.dumps(report))
    receipt = json.loads(design.read_text());receipt['preflight_sha256'] = file_sha256(preflight / 'report.json')
    design.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='Feasibility gate'):
        load_stage(sources, preflight, design, stage='pilot', project_root=tmp_path)


def test_out_of_scope_condition_counts_are_excluded():
    benchmark, outputs, scores = fixture_frame()
    benchmark[0]['decomposition'] = ['single']
    for p in PRODUCERS:
        for role in ROLES:
            scores[p][role]['task 0'] = {'keys': ('single',), 'masks': {'single': 'present'}, 'reasons': []}
    units, _, report = build_masked_frame(benchmark, outputs, scores)
    assert report['frames']['pr']['common_tasks'] == 2
    row = next(r for r in units if r['prompt_hash'] == digest('task 0'))
    assert 'out_of_scope_criterion_count' in json.loads(row['missing_reasons'])['p']


def test_criterion_version_diagnostic_uses_keys_not_score_values():
    benchmark, outputs, scores = fixture_frame()
    for p in PRODUCERS:
        scores[p]['p']['task 0'] = {'keys': ('rewritten a', 'rewritten b', 'extra'),
                                   'masks': {'rewritten a': 'present', 'rewritten b': 'present', 'extra': 'present'}, 'reasons': []}
    _, _, report = build_masked_frame(benchmark, outputs, scores)
    diagnostic = report['p_r_constraint_key_comparison'][PRODUCERS[0]]
    assert diagnostic['same_exact_constraint_set'] == 2
    assert diagnostic['different_constraint_count'] == 1
    assert diagnostic['criterion_count_pairs'] == {'2,2': 2, '3,2': 1}


def expected_download_fixture(tmp_path, monkeypatch):
    import src.wildifeval_data as wd
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wd, 'source_paths', lambda: ['benchmark/README.md'])
    sources = tmp_path / 'sources'; (sources / 'benchmark').mkdir(parents=True)
    file = sources / 'benchmark/README.md';file.write_text('pinned document')
    expected = {'revisions': wd.REVISIONS,
                'files': [{'path': 'benchmark/README.md', 'bytes': file.stat().st_size, 'sha256': file_sha256(file)}]}
    manifest = tmp_path / 'expected.json';manifest.write_text(json.dumps(expected))
    return wd, sources, file, manifest


def test_download_refuses_modified_existing_cache_against_expected(tmp_path, monkeypatch):
    wd, sources, file, manifest = expected_download_fixture(tmp_path, monkeypatch)
    file.write_text('modified document')
    monkeypatch.setattr(wd.subprocess, 'run', lambda *a, **kw: pytest.fail('Should not fetch when existing cache is bad'))
    with pytest.raises(ValueError, match='published expected manifest'):
        wd.download_sources(sources, manifest)
    assert not (sources / 'manifest.json').exists()


def test_download_uses_default_expected_manifest_and_accepts_exact_cache(tmp_path, monkeypatch):
    wd, sources, file, manifest = expected_download_fixture(tmp_path, monkeypatch)
    default = tmp_path / 'artifacts/expected/wildifeval_external/source_manifest.json'
    default.parent.mkdir(parents=True);default.write_bytes(manifest.read_bytes())
    monkeypatch.setattr(wd.subprocess, 'run', lambda *a, **kw: pytest.fail('Exact cache must not be fetched again'))
    result = wd.download_sources(sources)
    assert result['files'][0]['sha256'] == file_sha256(file)
    assert (sources / 'manifest.json').exists()


def test_download_checks_temporary_file_before_accepting_it(tmp_path, monkeypatch):
    wd, sources, file, manifest = expected_download_fixture(tmp_path, monkeypatch)
    file.unlink()
    def fake_fetch(args, **kwargs):
        Path(args[-1]).write_text('incorrect remote content')
    monkeypatch.setattr(wd.subprocess, 'run', fake_fetch)
    with pytest.raises(ValueError, match='published expected manifest'):
        wd.download_sources(sources, manifest)
    assert not file.exists()
    assert not (sources / 'manifest.json').exists()
