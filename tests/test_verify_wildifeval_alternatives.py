"""Synthetic privacy and replay-contract checks of the independent verifier."""
import json
import pytest
from src.verify_wildifeval_alternatives import load,num,VALID,INVALID,verify


def test_numeric_parser_does_not_keep_pass_or_fail_values(tmp_path):
    path=tmp_path/'labels.json'
    path.write_text('{"a":0.0,"b":1.0,"c":0.37,"d":0.89,"e":-1,"f":NaN}')
    parsed=load(path)
    assert all(parsed[k] is VALID for k in 'abcd')
    assert all(parsed[k] is INVALID for k in 'ef')
    assert not any(isinstance(v,(int,float)) for v in parsed.values())


def test_duplicate_keys_stop_instead_of_changing_associations(tmp_path):
    path=tmp_path/'duplicates.json';path.write_text('{"task":{"a":0,"a":1}}')
    with pytest.raises(ValueError,match='Duplicate key'):
        load(path)


def test_manifest_binding_checked_before_sources_are_parsed(tmp_path):
    sources=tmp_path/'sources';sources.mkdir()
    alternatives=tmp_path/'alternatives';alternatives.mkdir()
    (sources/'manifest.json').write_text('{}')
    report=tmp_path/'report.json';report.write_text(json.dumps({'source_manifest_sha256':'wrong'}))
    with pytest.raises(ValueError,match='does not bind'):
        verify(sources,alternatives,report,tmp_path/'receipt.json')
    assert not (tmp_path/'receipt.json').exists()
