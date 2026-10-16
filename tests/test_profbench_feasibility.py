"""Synthetic-only checks of reference masking and metadata counts."""
import json
import pytest
from src.profbench_feasibility import masked_object,profile,MODELS,analyze


def synthetic(value):
    criterion={'criterion_description':'Include required result','criterion_type':['Reasoning'],'criterion_weight':'Major'}
    criterion.update({m+'_fulfilment':value for m in MODELS})
    row={'task_id':'one','domain':'Physics PhD','prompt':'task','rubrics':[criterion]}
    row.update({m+'_response':'response '+m for m in MODELS})
    return json.loads(json.dumps(row),object_pairs_hook=masked_object)


def test_reference_true_and_false_have_identical_mask_and_metadata():
    a,b=synthetic(True),synthetic(False)
    assert a==b
    assert profile([a],'{response} {criterion_description}')==profile([b],'{response} {criterion_description}')
    assert all(a['rubrics'][0][m+'_fulfilment']=={'valid_binary':True,'source_type':'bool'} for m in MODELS)


def test_counts_use_three_outputs_and_one_base_task_without_outcome_values():
    result=profile([synthetic(True)],'{response} {criterion_description}')
    assert result['base_tasks']==1
    assert result['output_units']==3
    assert result['criterion_definitions']==1
    assert result['response_criterion_pairs']==3
    assert result['complete_reference_output_units']==3
    assert result['released_annotator_id_fields']==[]
    assert result['released_judge_prediction_fields']==[]


def test_missing_reference_masks_do_not_change_target_length():
    result=profile([synthetic(None)],'{response} {criterion_description}')
    assert result['criterion_definitions']==1
    assert result['complete_reference_output_units']==0
    assert result['invalid_reference_fields_by_producer']=={m:1 for m in MODELS}


def test_duplicate_keys_and_base_tasks_stop():
    with pytest.raises(ValueError,match='Duplicate JSON key'):
        json.loads('{"a":1,"a":2}',object_pairs_hook=masked_object)
    with pytest.raises(ValueError,match='Duplicate base-task'):
        profile([synthetic(True),synthetic(False)],'{response} {criterion_description}')


def test_source_tampering_stops_before_json_or_reference_parse(tmp_path):
    (tmp_path/'test.jsonl').write_text('intentionally not valid JSON')
    with pytest.raises(ValueError,match='input hash mismatch: test.jsonl'):
        analyze(tmp_path,tmp_path/'report.json')
    assert not (tmp_path/'report.json').exists()
