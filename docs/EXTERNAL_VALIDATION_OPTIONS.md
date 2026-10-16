# Next external-validation route

**Follow-up,2026-10-16:** The author has adopted the proposed two-role scope. See [execution preparation](WILDIFEVAL_EXECUTION.md) for the current stage. The dated comparison below records the decision's basis; no external outcomes have been produced.


2026-10-15. **Recommendation: generate one fixed, aligned primary judge on WildIFEval and use the saved DeepSeek reference for a two-role mechanism-forecast test.** ProfBench is a useful subsequent human-reference replication. This is a prospective recommendation: no new model call, amended design freeze, pilot or held-out outcome has been executed.

## Other saved WildIFEval judges

The pinned inventory contains six grading channels for five judge identities across the five preselected producing models. We checked every pair, triple and four-channel set: 50 combinations, of which 39 contain distinct judge identities. The other 11 contain both DeepSeek runs; those are not treated as independent judges or pooled together. All combinations fail the original feasibility gates.

| Distinct-judge pair | Common tasks | Clusters | Hypothetical pilot / held-out clusters | Retention |
|---|---:|---:|---:|---:|
| DeepSeek-main + Qwen | 337 | 329 | 65 / 264 | 4.48% |
| DeepSeek-main + GPT-4o | 335 | 328 | 65 / 263 | 4.45% |
| DeepSeek-main + Llama-3.3 | 332 | 324 | 64 / 260 | 4.41% |
| Original DeepSeek-main + Llama-3.1 | 40 | 40 | 8 / 32 | 0.53% |

The extra agreement-study graders largely share their criterion sets; their coverage is too small for the planned confirmation. The search covers all grading channels in the pinned release for the five fixed producers, not every historical version or private asset. It did not inspect pass rates, accuracies or policy effects.

DeepSeek-main alone supplies **7,418 complete common tasks, 6,847 clusters and 134,485 criterion records** across 37,090 producer-output units. Retention is 98.60%. Successful new P scores could therefore support the original size gates without reducing the benchmark denominator. Recheck completeness after generation; these counts do not guarantee a successful new judge run. The saved R files lack evaluated-output hashes, so using R still relies on release provenance for producer/task-to-output association. Fresh P will record exact output and criterion hashes; removing the historical R assumption would require regenerating R too.

[All combinations](../artifacts/expected/external_route_feasibility/alternatives.json) · [Independent verification of 39 source hashes, six channels and 50 combinations](../artifacts/expected/external_route_feasibility/alternatives_independent_verification.json).

## Why not replace the main study with ProfBench

The pinned [ProfBench release](https://huggingface.co/datasets/nvidia/ProfBench/tree/f5c471cf2f277c264621e4b5c0cad5bb466f9260) contains 40 base tasks, 120 saved outputs, 1,162 criterion definitions and 3,486 valid human-reference fields. All outputs have complete reference vectors. There are no released independent judge predictions, individual rater IDs or repeated-rating vectors in the inspected data/code inventory. The full paper's 7,347 pairs are not the public subset count.

ProfBench uses a positive importance-weighted score. All-pass would be a constructed target, with 15–59 criteria per task (median 27). Strict passes could be rare; we have not computed their prevalence. The [paper](https://arxiv.org/html/2510.18941v1) reports additional expert validation on a subset, which does not make every public label adjudicated truth.

Forty task clusters give only eight pilot and 32 held-out clusters under a 20/80 split. Prefer using all 40 tasks in a separately prespecified human-reference replication; keep the three outputs together for task-level uncertainty. It cannot replace the large native-all-pass forecast test. The official grading prompt needs only the saved response and one criterion, so no new report generation is needed. A new runner must omit human-reference fields from jobs and use immutable content IDs: the upstream loader carries human annotations in its records and assigns indices after shuffling, which should not be copied into our hidden-reference/resume workflow.

[Masked ProfBench inventory and source hashes](../artifacts/expected/external_route_feasibility/profbench_feasibility.json).

## Cost comparison

Offline reconstruction of the original per-criterion prompts gives 79–81 million input text tokens for WildIFEval and 5.6–5.7 million for ProfBench in two named local encodings. These are tokenizer proxies, not model-specific billing counts. The following includes an assumed 8–16 message-wrapper tokens per request and **32–128 total output tokens per request**, at [standard short-context prices](https://developers.openai.com/api/docs/pricing) checked on 2026-10-15. It excludes cache-write surcharges, retries, taxes and tools; it assumes no cache discount.

| One new judge | Criterion requests | GPT-6 Luna price tier | GPT-6.1 Sol price tier |
|---|---:|---:|---:|
| WildIFEval full aligned R frame | 134,485 | $10–17 | $204–338 |
| ProfBench public frame | 3,486 | $0.62–0.80 | $12–16 |

At 1,000 output tokens per call, the WildIFEval estimates rise to about $76 and $1,510 respectively. The output scenarios include reasoning tokens and are not measured generations. At an assumed sustained completion throughput of 1–5 requests/second, WildIFEval would take roughly 7.5–37 hours and ProfBench 12–58 minutes; account limits and actual latency have not been tested.

**Interface compatibility matters.** Luna with `reasoning=none` is a candidate for the released Wild grader's temperature/logprobs interface; no account/API fixture has verified it. Sol does not support `none`, and the [current parameter guidance](https://developers.openai.com/api/docs/guides/latest-model) requires removing temperature/logprobs/top_logprobs for non-none reasoning. Its cost row is a price illustration, not a drop-in grader. Hard Yes/No or joint-rubric judging would require an explicitly frozen new protocol. Joint judging changes within-task error dependence as well as price.

Fresh Luna P at 32 output tokens plus fresh Sol R at 128 output tokens would cost about $343–348 under these assumptions and require a revised grading protocol. It removes the historical output-version association gap but still produces model references. Neither route measures expert labor or establishes adjudicated truth.

[Token counts, all price scenarios and source hashes](../artifacts/expected/external_route_feasibility/costs.json).

## Proposed sequence and falsification

1. Amend to a two-role P/R main test before new grading. Retain the original five producers, 500/100/300 cluster and 70% retention gates, fixed split, budgets and primary forecast comparison. Keep SRS, mixed allocation and serial certification. Auxiliary-judge disagreement policies and complete alternative-reference sensitivity are outside this reduced design.
2. Check the candidate interface on unrelated synthetic inputs, then freeze checkpoint, prompt, probability/label parser, retry rule and a cash cap before dataset calls. A $100 hard stopping budget is a planning suggestion for the Luna route, not an invoice prediction or authorization to spend. Hash the exact output, rubric and request. Invalid calls remain missing, not FAIL.
3. Recheck the completed new-P frame and freeze its IDs. Reveal pilot references only. On the current complete-R frame the pilot would cost 26,375 reference bits, not exactly 20% of all bits. At a 20% held-out query cap the combined primary-reference budget would be 47,995 bits; compare with a no-pilot control at the same cost. Track API generation dollars separately.
4. Save every predictor's five-producer, three-budget predictions before held-out reference access. Test whether task-linked pilot information reduces MAE relative to the matched-stratum marginal model. A paired interval crossing zero is unresolved; an interval entirely above zero contradicts the proposed advantage. Report opposite directions, tiny effects and structural zeros without changing the judge, target or headline budget.
5. Add ProfBench later as a small human-reference replication under its own prospective protocol. Do not relabel it a large pilot-training study or a native conjunctive benchmark.

The manuscript's three research questions and all previous results remain unchanged. This proposal is not a new operational policy-selector claim.

## Reproduce the feasibility checks

Install optional readers with `python -m pip install ".[external]"`. The [original audit instructions](WILDIFEVAL.md#reproduce-the-preflight) download the pinned original sources to `artifacts/wildifeval-cache/`. Expanded checking downloads about 50 MB more:

```bash
python -m src.wildifeval_alternative_preflight download --sources artifacts/wildifeval-cache --alternatives artifacts/wildifeval-alternatives
python -m src.wildifeval_alternative_preflight survey --sources artifacts/wildifeval-cache --alternatives artifacts/wildifeval-alternatives --output artifacts/external-route-recheck/alternatives.json
python -m src.verify_wildifeval_alternatives --sources artifacts/wildifeval-cache --alternatives artifacts/wildifeval-alternatives --report artifacts/external-route-recheck/alternatives.json --output artifacts/external-route-recheck/independent.json
```

The expanded manifest and inventory are included beside the saved report. Existing reports are not overwritten. The source-level survey reads only criterion keys, validity masks and grouping metadata. For ProfBench, the offline `profbench_feasibility` module takes `--sources` and `--output`; the saved report lists the pinned HF/GitHub versions, consumed filenames and hashes. It checks the data and template hashes before parsing. `external_validation_costs --help` lists its source/cache/template inputs; it removes reference fields and reconstructs prompts without invoking a model. Every new module has synthetic tests. Raw source texts, outputs, human labels and judge-score files are excluded from this package.
