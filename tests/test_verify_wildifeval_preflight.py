"""Synthetic checks of the independent metadata-only verifier."""
import json
import csv
import gzip
import io

import pytest

from src.verify_wildifeval_preflight import (
    JUDGES, PRODUCERS, NumberMask, key_diagnostics, masked_json, numeric_mask,
    read_output_metadata, read_score_metadata, reconstruct_metadata, sha_text, sha_file,
    PINNED_REVISIONS, verify_saved, expected_source_paths, validate_inventory,
)


def test_numeric_literals_are_irreversibly_masked_before_json_values_exist(tmp_path):
    path = tmp_path / 'synthetic.json'
    path.write_text('{"scores":{"a":0.001,"b":0.999,"c":0,"d":1,"e":-1,"f":NaN,"g":"ERR","h":true}}')
    raw = masked_json(path)['scores']
    assert all(raw[k] is NumberMask.PRESENT for k in 'abcd')
    assert all(raw[k] is NumberMask.INVALID for k in 'ef')
    assert not any(type(v) in (int,float) for v in raw.values())
    for value in ('Infinity','-Infinity','2','-0.1'):
        assert numeric_mask(value) is NumberMask.INVALID


def test_duplicate_key_stops_without_printing_the_private_key(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"private prompt":{"scores":{"criterion":0.1,"criterion":0.9}}}')
    with pytest.raises(ValueError,match='Duplicate JSON key') as error:
        read_score_metadata(path)
    assert 'criterion' not in str(error.value)
    assert 'private prompt' not in str(error.value)


def test_score_metadata_preserves_error_masks_but_no_values_or_explanations(tmp_path):
    path = tmp_path / 'scores.json'
    path.write_text(json.dumps({'task': {'scores': {'a': .1234, 'b': 'ERR', 'c': True, 'd': None},
                                         'explanation': 'secret answer'}}))
    result = read_score_metadata(path)
    assert result['task']['valid_keys'] == {'a'}
    assert result['task']['keys'] == {'a','b','c','d'}
    assert 'secret answer' not in repr(result) and '.1234' not in repr(result)


def test_output_metadata_hashes_and_validates_same_task(tmp_path):
    path = tmp_path / 'outputs.json'
    path.write_text(json.dumps({'predictions_key': {'task': [{'role':'user','content':'wrong task'},
                                                           {'role':'assistant','content':'secret output'}]}}))
    result = read_output_metadata(path)
    assert result['task']['valid'] is False
    assert result['task']['output_hash'] == sha_text('secret output')
    assert 'secret output' not in repr(result)


def fixture():
    benchmark = [{'conversation_id': f'c{i}', 'decomposition': [f'a{i}',f'b{i}']} for i in range(3)]
    outputs,scores = {},{}
    for producer in PRODUCERS:
        outputs[producer] = {}
        scores[producer] = {role:{} for role in JUDGES}
        for i,row in enumerate(benchmark):
            prompt = f'prompt {i}'
            outputs[producer][prompt] = {'valid':True,'output_hash':sha_text(producer+prompt),
                                         'normalized_prompt_hash':sha_text(prompt)}
            for role in JUDGES:
                scores[producer][role][prompt] = {'keys':frozenset(row['decomposition']),
                                                  'valid_keys':frozenset(row['decomposition']),'valid_object':True}
    return benchmark,outputs,scores


def test_independent_exact_set_join_excludes_mismatch_and_invalid_label_whole_tasks():
    benchmark,outputs,scores = fixture()
    scores[PRODUCERS[0]]['p']['prompt 0']['keys'] = frozenset(['a0','different'])
    scores[PRODUCERS[-1]]['q']['prompt 1']['valid_keys'] = frozenset(['a1'])
    rows,result = reconstruct_metadata(benchmark,outputs,scores)
    assert result['frames']['pr']['common_tasks'] == 2
    assert result['frames']['four_role']['common_tasks'] == 1
    assert len(rows) == 15
    mismatch = next(r for r in rows.values() if r['producer']==PRODUCERS[0] and r['prompt_hash']==sha_text('prompt 0'))
    assert mismatch['eligible_pr'] == '0'
    assert all(r['k'] == '2' for r in rows.values())


def test_ambiguous_decomposition_rejected_and_output_links_connect_clusters():
    benchmark,outputs,scores = fixture()
    outputs[PRODUCERS[0]]['prompt 0']['output_hash']='same'
    outputs[PRODUCERS[1]]['prompt 1']['output_hash']='same'
    rows,result = reconstruct_metadata(benchmark,outputs,scores)
    assert result['frames']['pr']['clusters']==2
    assert result['largest_cluster_before_complete_case']==2
    benchmark.append({'conversation_id':'duplicate','decomposition':['a0','b0']})
    rows,result = reconstruct_metadata(benchmark,outputs,scores)
    assert result['ambiguous_constraint_set_rows']==2
    assert result['frames']['pr']['common_tasks']==2


def test_key_diagnostic_counts_are_independent_of_validity_and_do_not_fuzzy_match():
    make=lambda keys:{'keys':frozenset(keys),'valid_keys':frozenset(),'valid_object':True}
    p={'one':make(['a','b']),'two':make(['a  b','c']),'three':make(['old','list','extra'])}
    r={'one':make(['b','a']),'two':make(['a b','c']),'three':make(['new','list'])}
    result=key_diagnostics(p,r)
    assert result['shared_task_text_keys']==3
    assert result['same_exact_constraint_set']==1
    assert result['same_whitespace_normalized_constraint_set']==2
    assert result['different_constraint_count']==1


def saved_fixture(folder):
    benchmark,outputs,scores = fixture()
    rows,report = reconstruct_metadata(benchmark,outputs,scores)
    for frame,counts in report['frames'].items():
        gates={'clusters_at_least_500':False,'pilot_clusters_at_least_100':False,
               'confirmation_clusters_at_least_300':False,'retention_each_at_least_70pct':True}
        counts.update(gates=gates,feasible=False,retention_by_producer={p:1.0 for p in PRODUCERS})
    report.update(status='masked_preflight',score_values_exposed=False,feasible=False)
    source=folder/'source_manifest.json';source.write_text(json.dumps({'revisions':PINNED_REVISIONS,
        'files':[{'path':path} for path in sorted(expected_source_paths())]}))
    report['sources_manifest_sha256']=sha_file(source)
    records=list(rows.values())
    for name,table in [('masked_units',records),('excluded_units',[])]:
        out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=list(records[0]) if table else ['unit_id'])
        writer.writeheader();writer.writerows(table)
        data=out.getvalue().encode()
        with gzip.open(folder/(name+'.csv.gz'),'wb') as handle:handle.write(data)
        import hashlib
        report[name+'_sha256']=hashlib.sha256(data).hexdigest()
    reportpath=folder/'preflight_report.json';reportpath.write_text(json.dumps(report))
    receipt={'status':'PASS','metadata_only':True,'pass_fail_threshold_applied':False,
             'numeric_scores_retained':False,'sources_manifest_sha256':sha_file(source),
             'preflight_report_sha256':sha_file(reportpath),'masked_units_compared':len(rows),
             'masked_units_sha256':report['masked_units_sha256'],'independent_counts':report,
             'source_files_hash_checked':28}
    import src.verify_wildifeval_preflight as module
    receipt['verifier_source_sha256']=sha_file(module.__file__)
    (folder/'independent_verification.json').write_text(json.dumps(receipt))


def test_saved_only_mode_checks_hashes_counts_fixed_split_and_feasibility(tmp_path):
    saved_fixture(tmp_path)
    result=verify_saved(tmp_path)
    assert result['status']=='PASS'
    assert result['network_or_source_access'] is False
    assert result['masked_units']==15
    with gzip.open(tmp_path/'masked_units.csv.gz','ab') as handle:handle.write(b'corruption')
    with pytest.raises(AssertionError,match='CSV hash mismatch'):
        verify_saved(tmp_path)


def test_saved_only_mode_recomputes_the_gate_instead_of_trusting_report(tmp_path):
    saved_fixture(tmp_path)
    reportpath=tmp_path/'preflight_report.json'
    report=json.loads(reportpath.read_text());report['frames']['pr']['feasible']=True
    reportpath.write_text(json.dumps(report))
    receiptpath=tmp_path/'independent_verification.json'
    receipt=json.loads(receiptpath.read_text());receipt['preflight_report_sha256']=sha_file(reportpath)
    receiptpath.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError,match='feasibility decision'):
        verify_saved(tmp_path)


def test_manifest_inventory_rejects_missing_duplicate_or_unexpected_sources():
    entries=[{'path':path} for path in sorted(expected_source_paths())]
    manifest={'revisions':PINNED_REVISIONS,'files':entries}
    assert validate_inventory(manifest)==28
    for changed in (entries[:-1], entries[:-1]+[entries[0]], entries[:-1]+[{'path':'unexpected.json'}]):
        with pytest.raises(AssertionError,match='28 unique expected paths'):
            validate_inventory({**manifest,'files':changed})


@pytest.mark.parametrize('field,value,pattern',[
    ('verifier_source_sha256','incorrect','verifier-source hash'),
    ('source_files_hash_checked',27,'source-file count'),
])
def test_saved_only_mode_rejects_stale_verifier_or_source_count(tmp_path,field,value,pattern):
    saved_fixture(tmp_path)
    path=tmp_path/'independent_verification.json'
    receipt=json.loads(path.read_text());receipt[field]=value
    path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError,match=pattern):
        verify_saved(tmp_path)


def test_saved_only_mode_requires_consistent_top_level_feasibility(tmp_path):
    saved_fixture(tmp_path)
    reportpath=tmp_path/'preflight_report.json'
    report=json.loads(reportpath.read_text());report['feasible']=True
    reportpath.write_text(json.dumps(report))
    receiptpath=tmp_path/'independent_verification.json'
    receipt=json.loads(receiptpath.read_text());receipt['preflight_report_sha256']=sha_file(reportpath)
    receiptpath.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError,match='Top-level feasibility'):
        verify_saved(tmp_path)
