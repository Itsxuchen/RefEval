"""Offline prompt-token inventory for prospective external validation.

Reads benchmark constraints, producer outputs and masked join metadata only.
Never opens WildIFEval score JSON or accesses ProfBench fulfilment values.
Token counts are local encoding counts, not API billing measurements.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PRICING = {
    "gpt-6-luna_standard_short": {"input_per_million": .10, "output_per_million": .50},
    "gpt-6.1-sol_standard_short": {"input_per_million": 2.00, "output_per_million": 10.00},
}
PRICE_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICE_DATE = "2026-10-15"
WILD_CODE_COMMIT = "82ebff7068b9faeadc5468d55935d62b1601d89c"
PROF_CODE_COMMIT = "b06a29cda4d1433e9a9aad8171e4086299083e94"
COMPATIBILITY_SOURCE = "https://developers.openai.com/api/docs/guides/latest-model#gpt-6-astra-update-api-and-model-parameters"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discard_reference_fields(pairs):
    """Remove label fields while constructing JSON objects; retain prompt text."""
    return {key: value for key, value in pairs if not key.endswith("_fulfilment")}


def read_literal_template(path, name):
    """Read a static string assignment, optionally .strip(), without executing code."""
    module = ast.parse(Path(path).read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = node.value
            strip = isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "strip" and not value.args and not value.keywords
            value = ast.literal_eval(value.func.value if strip else value)
            if not isinstance(value, str):
                raise ValueError("Template is not a literal string")
            return value.strip() if strip else value
    raise ValueError("Template not found")


def stats(values):
    values = np.asarray(values, dtype=np.int64)
    return dict(count=len(values), total=int(values.sum()), minimum=int(values.min()),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)), maximum=int(values.max()))


def summarize(rows, encodings):
    N = len(rows)
    requests = sum(r["k"] for r in rows)
    result = dict(tasks=len({r["task_id"] for r in rows}), clusters=len({r["cluster_id"] for r in rows}),
                  output_units=N, criterion_requests_per_new_judge=requests,
                  joint_requests_per_new_judge=N, by_producer={}, token_counts={})
    for producer in sorted({r["producer"] for r in rows}):
        cell = [r for r in rows if r["producer"] == producer]
        result["by_producer"][producer] = dict(units=len(cell), criteria=sum(r["k"] for r in cell))
    for encoding in encodings:
        a = [token for r in rows for token in r["criterion_tokens"][encoding]]
        b = [r["joint_tokens"][encoding] for r in rows]
        result["token_counts"][encoding] = dict(
            criterion_prompt_text=stats(a), joint_prompt_text=stats(b),
            criterion_input_with_8_to_16_wrapper_tokens=[sum(a)+8*requests, sum(a)+16*requests],
            joint_input_with_8_to_16_wrapper_tokens=[sum(b)+8*N, sum(b)+16*N])
    # Local tokenizer spread is descriptive, not a bound on another model's encoder.
    result["cost_scenarios"] = []
    for model, rate in PRICING.items():
        for output_tokens_per_criterion in (1, 32, 128, 1000):
            inputs = [result["token_counts"][enc]["criterion_input_with_8_to_16_wrapper_tokens"][i]
                      for enc in encodings for i in (0, 1)]
            output_tokens = requests * output_tokens_per_criterion
            costs = [(t*rate["input_per_million"]+output_tokens*rate["output_per_million"])/1e6 for t in inputs]
            result["cost_scenarios"].append(dict(model=model, output_tokens_per_criterion=output_tokens_per_criterion,
                minimum_encoding_proxy_usd=min(costs), maximum_encoding_proxy_usd=max(costs),
                rate_basis="standard short context; all input at noncached input rate; no retries/tools/taxes"))
    return result


def wild_inventory(sources, masked_csv, template, encoders):
    """Cost new judges only on complete current-criterion/saved-R common tasks."""
    import pyarrow.parquet as pq
    sources = Path(sources)
    masked = list(csv.DictReader(Path(masked_csv).open()))
    producers = sorted({r["producer"] for r in masked})
    eligible = {p: {r["task_id"] for r in masked if r["producer"] == p and not json.loads(r["missing_reasons"])["r"]}
                for p in producers}
    common = set.intersection(*eligible.values())
    selected = [r for r in masked if r["task_id"] in common]
    by_task = {sha("wildifeval-task|"+r["conversation_id"]): list(r["decomposition"])
               for r in pq.read_table(sources / "benchmark/data/test-00000-of-00001.parquet").to_pylist()}
    results = []
    hashes = {"masked_csv": file_sha(masked_csv), "benchmark_parquet": file_sha(sources / "benchmark/data/test-00000-of-00001.parquet")}
    for producer in producers:
        output_path = sources / "predictions/model_predictions" / f"{producer}-0shot-wild-if-eval.json"
        hashes["producer_outputs/"+producer] = file_sha(output_path)
        raw = json.loads(output_path.read_text())
        if "predictions_key" in raw:
            wrapper = raw["predictions_key"]
            raw = raw[wrapper] if isinstance(wrapper, str) else wrapper
        outputs = {sha(prompt): (prompt, messages) for prompt, messages in raw.items()}
        for row in selected:
            if row["producer"] != producer:
                continue
            prompt, messages = outputs[row["prompt_hash"]]
            responses = [m["content"] for m in messages if m.get("role") == "assistant"]
            if len(responses) != 1 or sha(responses[0]) != row["output_hash"]:
                raise ValueError("Output hash no longer matches masked frame")
            response, constraints = responses[0], by_task[row["task_id"]]
            if [sha(c) for c in constraints] != json.loads(row["constraint_hashes"]):
                raise ValueError("Criterion vector no longer matches masked frame")
            prompts = [template.format(instruction=prompt, response=response, constraint=c) for c in constraints]
            # Different hypothetical judging protocol, shown only to expose
            # the cost of repeating the response. It is not an interchangeable
            # implementation of the original per-criterion grader.
            joint = ("Evaluate each numbered constraint against the task instruction and response. "
                     "Return one Yes or No for each constraint in order.\n\nTask instruction:\n"+prompt+
                     "\n\nResponse:\n"+response+"\n\nConstraints:\n"+
                     "\n".join(f"{i+1}. {c}" for i,c in enumerate(constraints))+"\n\nAnswers:\n")
            result = {key: row[key] for key in ("task_id", "cluster_id", "producer")}
            result.update(k=len(constraints), criterion_tokens={}, joint_tokens={})
            for name, encoder in encoders.items():
                result["criterion_tokens"][name] = [len(encoder.encode(text, disallowed_special=())) for text in prompts]
                result["joint_tokens"][name] = len(encoder.encode(joint, disallowed_special=()))
            results.append(result)
    scenarios = {"full_aligned_saved_R_frame": summarize(results, encoders)}
    clusters = sorted({r["cluster_id"] for r in results}, key=lambda c: sha("refeval-cost-subsample-v1|"+c))
    for count in (1250, 1500):
        chosen = set(clusters[:count])
        scenarios[f"new_prospective_{count}_cluster_cohort"] = summarize([r for r in results if r["cluster_id"] in chosen], encoders)
        scenarios[f"new_prospective_{count}_cluster_cohort"]["scope"] = "cost illustration only; a new cohort amendment, not a rescue of the original 70% retention gate"
    split = sorted(clusters, key=lambda c: sha("refeval-external-v1|"+c))
    pilot = set(split[:len(split)//5])
    scenarios["full_aligned_saved_R_frame"]["pilot_cost_bits"] = sum(r["k"] for r in results if r["cluster_id"] in pilot)
    scenarios["full_aligned_saved_R_frame"]["pilot_clusters"] = len(pilot)
    scenarios["full_aligned_saved_R_frame"]["confirmation_clusters"] = len(clusters)-len(pilot)
    scenarios["full_aligned_saved_R_frame"]["benchmark_retention"] = len(common)/len(by_task)
    return dict(scenarios=scenarios, input_hashes=hashes, benchmark_tasks=len(by_task),
                complete_benchmark_criterion_requests=5*sum(map(len, by_task.values())),
                score_files_opened=False, stored_reference_values_accessed=False)


def prof_inventory(data_path, template, encoders):
    results = []
    for line in Path(data_path).read_text().splitlines():
        raw = json.loads(line, object_pairs_hook=discard_reference_fields)
        task = str(raw["task_id"])
        # Extract only public prompts needed for new grading. Existing
        # *_fulfilment values are never accessed or thresholded.
        constraints = [r["criterion_description"] for r in raw["rubrics"]]
        for producer in ("o3", "grok4", "r1-0528"):
            response = raw[producer+"_response"]
            prompts = [template.format(response=response, criterion_description=c) for c in constraints]
            joint = "Response:\n\n"+response+"\n\nFor each numbered criterion return Yes or No in order:\n"+"\n".join(f"{j+1}. {c}" for j,c in enumerate(constraints))
            row = dict(task_id=task, cluster_id=task, producer=producer, k=len(constraints), criterion_tokens={}, joint_tokens={})
            for name, encoder in encoders.items():
                row["criterion_tokens"][name] = [len(encoder.encode(p, disallowed_special=())) for p in prompts]
                row["joint_tokens"][name] = len(encoder.encode(joint, disallowed_special=()))
            results.append(row)
    return dict(inventory=summarize(results, encoders), input_sha256=file_sha(data_path), reference_values_accessed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wild-sources", type=Path, required=True)
    parser.add_argument("--wild-masked", type=Path, required=True)
    parser.add_argument("--wild-template", type=Path, required=True)
    parser.add_argument("--prof-data", type=Path, required=True)
    parser.add_argument("--prof-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite an existing cost inventory")
    import tiktoken
    encoders = {name: tiktoken.get_encoding(name) for name in ("o200k_base", "cl100k_base")}
    payload = dict(status="prospective_cost_estimate_no_api_calls", pricing_checked=PRICE_DATE,
        pricing=PRICING, pricing_source=PRICE_SOURCE, tokenizer_version=tiktoken.__version__,
        tokenization_scope="Exact text counts in named local encodings; neither a validated GPT-6 tokenizer nor an API billing measurement. Chat-wrapper allowance 8–16 tokens/request is a sensitivity assumption, not a guarantee.",
        output_scope="1/32/128/1000 output-token scenarios, not observed generations. Reasoning consumes the output-token allowance; retries and cache-write surcharges are excluded.",
        model_compatibility={
            "source": COMPATIBILITY_SOURCE,
            "gpt-6-luna": "reasoning=none is a candidate for the temperature/logprobs interface; no API fixture or account access check has run",
            "gpt-6.1-sol": "does not support reasoning=none; non-none reasoning requires removing temperature/logprobs/top_logprobs, so this is a price tier only, not a drop-in replacement for the released Wild grader",
            "new_protocol_required": "Hard Yes/No labels or joint-criterion prompts change the grader and must be frozen as a prospective design before revealing validation outcomes",
        },
        source_commits=dict(wild=WILD_CODE_COMMIT, prof=PROF_CODE_COMMIT),
        template_hashes=dict(wild=file_sha(args.wild_template), prof=file_sha(args.prof_template)),
        wild=wild_inventory(args.wild_sources, args.wild_masked, read_literal_template(args.wild_template, "PROMPT_EVAL"), encoders),
        prof=prof_inventory(args.prof_data, read_literal_template(args.prof_template, "QUESTION_PROMPT_YES_NO"), encoders))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"wild_requests": payload["wild"]["scenarios"]["full_aligned_saved_R_frame"]["criterion_requests_per_new_judge"],
                      "prof_requests": payload["prof"]["inventory"]["criterion_requests_per_new_judge"], "model_calls":0}))


if __name__ == "__main__":
    main()
