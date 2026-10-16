"""Synthetic checks for the offline prospective cost inventory."""
import json

import pytest

from src.external_validation_costs import discard_reference_fields, read_literal_template, summarize


def test_reference_fields_removed_recursively():
    raw = json.loads('{"task_id": "fixture", "rubrics": [{"criterion_description": "C", "o3_fulfilment": true, "r1-0528_fulfilment": false}]}', object_pairs_hook=discard_reference_fields)
    assert raw == {"task_id": "fixture", "rubrics": [{"criterion_description": "C"}]}


def test_literal_template_without_executing_source(tmp_path):
    source = tmp_path / "template.py"
    source.write_text('raise RuntimeError("must not execute")\nPROMPT = " x {response} ".strip()\n')
    assert read_literal_template(source, "PROMPT") == "x {response}"


def test_request_and_cost_arithmetic_independent_of_tokenizer():
    rows = [dict(task_id="a", cluster_id="c", producer="p", k=2,
                 criterion_tokens={"fixture": [10, 20]}, joint_tokens={"fixture": 25})]
    result = summarize(rows, ("fixture",))
    assert result["criterion_requests_per_new_judge"] == 2
    assert result["token_counts"]["fixture"]["criterion_input_with_8_to_16_wrapper_tokens"] == [46, 62]
    scenario = next(x for x in result["cost_scenarios"] if x["model"] == "gpt-6-luna_standard_short" and x["output_tokens_per_criterion"] == 32)
    assert scenario["minimum_encoding_proxy_usd"] == pytest.approx((46 * .10 + 64 * .50)/1e6)
    assert scenario["maximum_encoding_proxy_usd"] == pytest.approx((62 * .10 + 64 * .50)/1e6)
