"""All fixtures are synthetic; no real outcome values are opened."""
import copy
import json
from pathlib import Path

import pytest

from src.wildifeval_alternative_preflight import (
    PINNED_REVISIONS, PRODUCERS, discover_channels, survey_metadata, coverage_gates,
    _validate_extra_manifest,
)
from src.verify_wildifeval_preflight import sha_file, sha_text, read_score_metadata


def fixture(n=3):
    names = [('main/deepseek-v3', 'deepseek-v3', 'main'), ('main/llama3.1-70b', 'llama3.1-70b', 'main'),
             ('agreement/deepseek-v3', 'deepseek-v3', 'agreement'), ('agreement/llama3.3-70b', 'llama3.3-70b', 'agreement'),
             ('agreement/qwen2.5-72b', 'qwen2.5-72b', 'agreement'), ('agreement/gpt-4o-2024-08-06', 'gpt-4o-2024-08-06', 'agreement')]
    channels = {name: {'judge': judge, 'run': run, 'files': {}} for name, judge, run in names}
    benchmark = [{'conversation_id': str(i), 'decomposition': [f'a{i}', f'b{i}']} for i in range(n)]
    outputs, scores = {}, {}
    for producer in PRODUCERS:
        outputs[producer] = {}
        scores[producer] = {name: {} for name in channels}
        for i, row in enumerate(benchmark):
            prompt = 'prompt ' + str(i)
            outputs[producer][prompt] = {'valid': True, 'output_hash': sha_text(producer + '/' + str(i)),
                                         'normalized_prompt_hash': sha_text(prompt)}
            for name in channels:
                scores[producer][name][prompt] = {'valid_object': True, 'keys': frozenset(row['decomposition']),
                                                 'valid_keys': frozenset(row['decomposition'])}
    return benchmark, outputs, scores, channels


def test_enumerates_all_channel_combinations_and_distinguishes_same_judge_runs():
    report = survey_metadata(*fixture())
    assert report['saved_channels'] == 6 and report['distinct_judges'] == 5
    assert len(report['combinations']) == 50  # C(6,2)+C(6,3)+C(6,4)
    assert len(report['raw_pair_diagnostics']) == 15
    duplicate = next(c for c in report['combinations'] if c['channels'] == ['agreement/deepseek-v3', 'main/deepseek-v3'])
    assert duplicate['distinct_judge_identities'] is False
    assert duplicate['usable_under_original_coverage_gates'] is False
    assert report['outcome_analysis_performed'] is False


def test_exact_full_current_criteria_required_and_not_fuzzy_joined():
    benchmark, outputs, scores, channels = fixture()
    channel = 'main/llama3.1-70b'
    for producer in PRODUCERS:
        scores[producer][channel]['prompt 0'] = {'valid_object': True, 'keys': frozenset(['paraphrase a', 'paraphrase b']),
                                               'valid_keys': frozenset(['paraphrase a', 'paraphrase b'])}
    report = survey_metadata(benchmark, outputs, scores, channels)
    assert report['channel_coverage'][channel]['common_tasks'] == 2
    assert report['channel_coverage']['main/deepseek-v3']['common_tasks'] == 3
    pair = next(x for x in report['raw_pair_diagnostics'] if x['channels'] == ['main/deepseek-v3', channel])
    assert pair['by_producer'][PRODUCERS[0]]['same_exact_constraint_set'] == 2


def test_missing_scores_and_producer_intersection_exclude_entire_task():
    benchmark, outputs, scores, channels = fixture()
    scores[PRODUCERS[-1]]['agreement/qwen2.5-72b']['prompt 0']['valid_keys'] = frozenset(['a0'])
    report = survey_metadata(benchmark, outputs, scores, channels)
    assert report['channel_coverage']['agreement/qwen2.5-72b']['common_tasks'] == 2
    assert report['channel_coverage']['agreement/qwen2.5-72b']['criteria_per_producer_in_common_frame'] == 4
    assert all(c['common_tasks'] == 2 for c in report['combinations'] if 'agreement/qwen2.5-72b' in c['channels'])


def test_clustering_keeps_duplicate_outputs_together_across_producers():
    benchmark, outputs, scores, channels = fixture()
    outputs[PRODUCERS[0]]['prompt 0']['output_hash'] = 'duplicate'
    outputs[PRODUCERS[1]]['prompt 1']['output_hash'] = 'duplicate'
    report = survey_metadata(benchmark, outputs, scores, channels)
    assert report['total_clusters_before_complete_case'] == 2
    assert report['largest_cluster_before_complete_case'] == 2
    assert all(c['clusters'] == 2 for c in report['combinations'])


def test_fixed_gates_keep_retention_denominator_full_benchmark():
    result = coverage_gates(set(map(str, range(500))), set(map(str, range(500))), 7523)
    assert result['gates']['clusters_at_least_500'] is True
    assert result['gates']['pilot_clusters_at_least_100'] is True
    assert result['gates']['confirmation_clusters_at_least_300'] is True
    assert result['gates']['retention_each_at_least_70pct'] is False
    assert result['coverage_feasible'] is False
    assert coverage_gates(set(map(str, range(500))), set(map(str, range(500))), 700)['coverage_feasible'] is True


def test_inventory_discovers_every_run_separately_and_ignores_other_producers():
    entries = []
    for run, judge in [('', 'deepseek-v3'), ('other_judges/', 'deepseek-v3'), ('other_judges/', 'llama3.3-70b')]:
        for producer in PRODUCERS:
            entries.append({'rfilename': f'llm_aaj_scores/{run}llm-aaj-{judge}.{producer}-0shot-wild-if-eval.json'})
    entries.append({'rfilename': 'llm_aaj_scores/llm-aaj-deepseek-v3.OtherProducer-0shot-wild-if-eval.json'})
    inventory = {'sha': PINNED_REVISIONS['predictions'], 'siblings': entries}
    channels = discover_channels(inventory)
    assert set(channels) == {'main/deepseek-v3', 'agreement/deepseek-v3', 'agreement/llama3.3-70b'}
    assert all(len(c['files']) == 5 for c in channels.values())
    inventory['sha'] = 'wrong'
    with pytest.raises(ValueError, match='revision'):
        discover_channels(inventory)


def test_manifest_tamper_and_duplicate_files_rejected(tmp_path):
    path = tmp_path / 'one.json';path.write_text('metadata')
    entry = {'path': path.name, 'sha256': sha_file(path), 'bytes': path.stat().st_size}
    manifest = {'revision': PINNED_REVISIONS['predictions'], 'files': [entry]}
    _validate_extra_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match='Duplicate'):
        _validate_extra_manifest(tmp_path, {**manifest, 'files': [entry, entry]})
    path.write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        _validate_extra_manifest(tmp_path, manifest)


def test_alternative_parser_uses_presence_only(tmp_path):
    path = tmp_path / 'scores.json'
    path.write_text(json.dumps({'task': {'scores': {'a': .01, 'b': .99, 'c': 'ERR'}}}))
    metadata = read_score_metadata(path)
    assert metadata['task']['valid_keys'] == frozenset(['a', 'b'])
    assert set(metadata['task']) == {'keys', 'valid_keys', 'valid_object'}
    assert '.01' not in str(metadata) and '.99' not in str(metadata)
