"""Synthetic input preparation; never API calls or real reference labels."""
import copy
import json
from pathlib import Path

import pytest

from src.wildifeval_jobs import (
    PRODUCERS, sha_text, select_common_units, make_unit_jobs, read_template,
    checked_inputs,
)


def row(task='t0', producer=PRODUCERS[0], prompt='task', response='output', criteria=('a', 'b'), cluster='c0'):
    unit = sha_text('wildifeval-unit|' + task + '|' + producer)
    hashes = [sha_text(c) for c in criteria]
    ids = [sha_text(unit + '|' + str(j) + '|' + h) for j, h in enumerate(hashes)]
    return {'unit_id': unit, 'task_id': task, 'cluster_id': cluster, 'producer': producer,
            'k': str(len(criteria)), 'prompt_hash': sha_text(prompt), 'output_hash': sha_text(response),
            'constraint_hashes': json.dumps(hashes), 'criterion_ids': json.dumps(ids),
            'missing_reasons': json.dumps({'p': ['irrelevant_old_P_mismatch'], 'r': [], 'q': ['missing']}),
            'split': 'pilot'}


def test_jobs_bind_exact_content_and_contain_no_reference_values():
    r = row(prompt='Task {untrusted}\n中文', response='Answer {text}')
    unit, jobs = make_unit_jobs(r, ('a', 'b'), 'Task {untrusted}\n中文', 'Answer {text}', '{instruction}\n{response}\n{constraint}')
    assert unit['criterion_ids'] == json.loads(r['criterion_ids'])
    assert unit['prompt_sha256'] == r['prompt_hash']
    assert len(jobs) == 2 and len(set(j['job_id'] for j in jobs)) == 2
    for j, job in enumerate(jobs):
        assert job['criterion_index'] == j
        assert job['prompt_sha256'] == sha_text(job['prompt'])
        assert job['job_id'] == sha_text('wildifeval-new-p-v1|' + unit['criterion_ids'][j] + '|' + job['prompt_sha256'])
        assert not ({'r', 'reference', 'score', 'label', 'p', 'q'} & set(job))
    assert 'prompt' not in unit
    assert unit['job_ids'] == [j['job_id'] for j in jobs]


def test_job_ids_change_with_rendering_not_with_run_clock():
    r = row()
    _, first = make_unit_jobs(r, ('a', 'b'), 'task', 'output', '{instruction}{response}{constraint}')
    _, repeat = make_unit_jobs(r, ('a', 'b'), 'task', 'output', '{instruction}{response}{constraint}')
    _, changed = make_unit_jobs(r, ('a', 'b'), 'task', 'output', '{instruction}\n{response}{constraint}')
    assert first == repeat
    assert first[0]['job_id'] != changed[0]['job_id']


@pytest.mark.parametrize('field,value,error', [
    ('prompt_hash', 'bad', 'prompt/output hash'),
    ('output_hash', 'bad', 'prompt/output hash'),
    ('constraint_hashes', '["bad","other"]', 'criterion hashes'),
    ('criterion_ids', '["bad","other"]', 'Criterion identities'),
    ('unit_id', 'bad', 'Criterion identities'),
])
def test_changed_input_links_fail_closed(field, value, error):
    r = row();r[field] = value
    with pytest.raises(ValueError, match=error):
        make_unit_jobs(r, ('a', 'b'), 'task', 'output', '{instruction}{response}{constraint}')


def test_reordering_or_dropping_original_conditions_is_rejected():
    r = row()
    with pytest.raises(ValueError, match='Ordered criterion'):
        make_unit_jobs(r, ('b', 'a'), 'task', 'output', '{instruction}{response}{constraint}')
    with pytest.raises(ValueError, match='complete criterion'):
        make_unit_jobs(r, ('a',), 'task', 'output', '{instruction}{response}{constraint}')


def test_common_frame_uses_only_reference_missingness_and_fixed_clusters():
    rows = [row(task=f't{i}', producer=p, cluster=f'c{i}') for i in range(10) for p in PRODUCERS]
    rows[0]['missing_reasons'] = json.dumps({'r': ['error']})
    selected = select_common_units(list(reversed(rows)))
    assert len(selected) == 45 and not any(r['task_id'] == 't0' for r in selected)
    pilot = {r['cluster_id'] for r in selected if r['split'] == 'pilot'}
    confirm = {r['cluster_id'] for r in selected if r['split'] == 'confirmation'}
    assert len(pilot) == 1 and len(confirm) == 8 and not pilot & confirm
    expected = min((f'c{i}' for i in range(1, 10)), key=lambda c: sha_text('refeval-external-v1|' + c))
    assert pilot == {expected}
    assert selected == select_common_units(rows)


def test_inconsistent_cluster_or_duplicate_units_rejected():
    rows = [row(producer=p) for p in PRODUCERS]
    with pytest.raises(ValueError, match='Duplicate unit'):
        select_common_units(rows + [rows[0]])
    rows[0]['cluster_id'] = 'different'
    with pytest.raises(ValueError, match='inconsistent clusters'):
        select_common_units(rows)


def test_template_is_literal_and_never_executes_source(tmp_path):
    p = tmp_path / 'template.py'
    p.write_text("raise RuntimeError('must not execute')\nPROMPT_EVAL = '''  {instruction} {response} {constraint}  '''.strip()")
    assert read_template(p) == '{instruction} {response} {constraint}'
    p.write_text("PROMPT_EVAL = dangerous_function()")
    with pytest.raises((ValueError, TypeError)):
        read_template(p)


def test_manifest_hash_verified_before_any_source_or_score_access(tmp_path):
    sources = tmp_path / 'sources';sources.mkdir()
    (sources / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='source manifest content'):
        checked_inputs(sources, tmp_path/'missing_masked', tmp_path/'missing_template')
