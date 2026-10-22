# External validation design: task structure, update effects, and allocation

Design date: 2026-10-10. **2026-10-16 status: the author adopted the two-role new-P/saved-R route; Unrelated API calibration and the160-request fee canary have returned. The evidence-only compatibility repair was checked and full scoring is now running; no completed P dataset exists. No pilot or confirmation reference has been opened for analysis.** Section 12 records the execution amendment. Sections 10–11 preserve the failed original preflight and subsequent feasibility comparison. This document is a prospective protocol, not an executed external validation or external registration. The manuscript's three RQs remain unchanged.

**2026-10-18 status:** Section 13 fixes the ProfBench analysis plan before any ProfBench reference label is read. The WildIFEval generation of Section 12 is paused.

## 1. Decision and the claim being tested

Start with **WildIFEval**, using its saved scores to test transfer to a native strict all-pass evaluation without new model inference. Its reference is an explicitly named LLM, not adjudicated human truth. Use **ProfBench** as the strongest human-reference follow-up if a new frozen judge run is feasible. Neither route requires Mercor. AGENTIF is a promising programmatic-reference extension, but its all-code complete-task subset has not been counted.

The identities `E_gate − E_eager = D − A` and `E_eager = E0 − C_W − D + A` follow from definitions. Reproducing them on another dataset checks the implementation; it does not independently confirm an empirical explanation. The falsifiable extension is:

> Can task-linked error configurations observed in a limited pilot predict the direction and size of the update effect on withheld tasks more accurately than criterion margins alone?

We will not predict that gating must be worse at high budgets in every environment. The existing paper already implies regimes in which delay is absent or offset by avoided false passes. A transport test must permit these cases to contradict a forecast.

## 2. Original description-level screen (superseded by Section 10 for WildIFEval)

This initial screen did not establish full record-level feasibility. In particular, the WildIFEval same-output association was inferred from release filenames, and complete criterion compatibility had not been checked. Section 10 supersedes that initial recommendation. The screen inspected official descriptions, code, directory inventories and limited schema/headers. It did not compute the proposed outcomes. Search of current project source, progress, decisions and analysis inventory found no prior execution on the selected candidates. WildIFEval and AGENTIF were read as literature; IFEval and SWE-bench were actually analyzed earlier and cannot be relabeled unused confirmation data.

| Candidate | Verified usable structure | Main gap and role |
|---|---|---|
| WildIFEval | Current release: 7,523 tasks, 2–8 constraints; native strict conjunction; saved outputs and same-output scores from several judges | Best no-inference mechanism test, conditional on machine reference; exact joins/missingness still require a masked preflight |
| ProfBench | 40 public base tasks, 3 saved responses per task, task–response–criterion links to human fulfilment labels | Human-reference transfer; official score is weighted, so strict conjunction is constructed. Saved judge predictions were not found; a frozen judge run is needed |
| AGENTIF | 707 instructions, native all-constraint ISR; code/LLM/hybrid evaluation; 24 saved legacy model-result files located | Count fully program-checkable tasks before choosing this route; code reference and an independent judge must be separate. `None` can mean inapplicable or execution failure |
| InFoBench | Public expert CSV and separate GPT-4 grading archives; human validation covers 50 base instructions across 5 producing models | Output versions differ between expert and judge folders; instruction ID is insufficient for joining. Resolve output hashes and reference-column semantics first. Native DRFR is a mean |
| IFBench | Official checkers and example outputs; 300 tasks, but 256 have one listed checker and only 44 have two | Small programmatic control; strict/loose checkers are different scoring rules, not independent judge/reference labels |
| ComplexBench / FollowBench | Multi-constraint tasks and evaluation code; ComplexBench also publishes saved responses | Missing matched independent reference/prediction vectors; ComplexBench includes dependencies beyond a single conjunction. Lower priority |
| HealthBench meta-eval | Binary physician judgments and rater identifiers | Need complete multi-criterion output vectors and new judge predictions; selected-condition conjunction is not native clinical success |
| BiGGen / RubricBench / HelpSteer3 / RewardBench 2 | Useful scalar, preference or principle data | Their published objects do not directly provide the required human-referenced multi-bit rubric frame. Do not manufacture one by renaming preference labels |

Sources: [WildIFEval benchmark](https://huggingface.co/datasets/gililior/wild-if-eval), [saved-score card](https://huggingface.co/datasets/gililior/wild-if-eval-predictions/blob/main/README.md), [score implementation](https://github.com/gililior/wild-if-eval-code/blob/82ebff7068b9faeadc5468d55935d62b1601d89c/scripts/data_analysis/plots_for_paper.py); [ProfBench code and data links](https://github.com/NVlabs/ProfBench), [explicit human-label loader](https://github.com/NVlabs/ProfBench/blob/main/run_llm_judge_on_provided_reports.py), [paper §4.1](https://arxiv.org/html/2510.18941v1); [AGENTIF data](https://huggingface.co/datasets/THU-KEG/AgentIF), [saved results](https://github.com/agentif/agentif.github.io/tree/25c2a886df763aa42a451bb563139c5e3b254974/docs/results), [evaluation code](https://github.com/THU-KEG/AgentIF/blob/76662306ba5b7fa61a4f8c81ef590bfd4432e490/code4eval/1.evaluation_api.py); [InFoBench release](https://github.com/qinyiwei/InfoBench), [paper §§3–4](https://aclanthology.org/2024.findings-acl.772.pdf); [IFBench frozen tree](https://github.com/allenai/IFBench/tree/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d).

Additional schema sources: [ComplexBench](https://github.com/thu-coai/ComplexBench), [FollowBench](https://github.com/YJiangcm/FollowBench), [HealthBench meta-evaluation](https://github.com/openai/simple-evals/blob/main/healthbench_meta_eval.py), [BiGGen results](https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results), [RubricBench](https://github.com/planepig/rubricbench), [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3), [RewardBench 2](https://huggingface.co/datasets/allenai/reward-bench-2).

## 3. WildIFEval frame fixed before outcome access

**Producing models:** use the five models for which the official agreement study supplies the extra graders: `deepseek-v3`, `llama3.1-405b`, `llama3.3-70b`, `mistral-large`, `qwen2.5-72b`. Include all five and report every cell. The list follows availability of a common reference-sensitivity design, not leaderboard performance.

**Roles:** initial prediction P = saved `llama3.1-70b`; primary reference R = saved `deepseek-v3`; alternative reference R2 = saved `gpt-4o-2024-08-06`; auxiliary Q = saved `qwen2.5-72b`. Hold P and Q fixed when changing R. The original benchmark's primary/secondary naming does not establish which judge is correct. Score >= 0.5 maps to PASS, as in the source implementation. Reference roles are frozen, never selected by the results they produce.

**Unit:** one prompt–producing-model output, with its entire original list of conditions. Cluster all outputs from a conversation together. Exact duplicate normalized prompts and duplicate output hashes are connected into the same cluster; this grouping is based on text/IDs, never scores. Do not silently overwrite duplicated task-text or constraint-text keys.

**Preflight may expose only:** snapshot IDs and hashes, filenames, model identifiers, join keys, field types, constraint counts, missing/error masks and cluster membership. The parser keeps numerical score values out of logs and analytical tables. Freeze the complete eligible-ID list and five-model common frame before revealing any reference values. Missing data determine a complete-case target, not a claim about the full original benchmark.

**Missingness:** `ERR`, absent entries, duplicate ambiguous keys and unparseable values are missing, never FAIL. Require complete vectors for P, Q, R and R2 for the primary common-frame analysis. Report every exclusion by cause and producing model; do not drop only the offending condition and change the task target. Retain the excluded-ID manifest. An all-missing-as-FAIL replay matching the upstream convention may be reported separately as an implementation sensitivity, not ground truth.

**Minimum feasibility:** at least 500 unique eligible clusters in the common frame, at least 100 pilot clusters and 300 confirmation clusters after grouping, and at least 70% retention in each producer cell. These are advance feasibility guardrails, not a power guarantee. If they fail, stop the confirmatory plan and report the shortfall. A revised design must be dated before any new outcome inspection; do not quietly substitute a more favorable subset or dataset.

## 4. Split, budget and exposed information

Assign clusters by sorting SHA-256 of `refeval-external-v1|canonical_cluster_id`. The first floor(0.20 G) clusters form the pilot; the remaining clusters form confirmation. Keep all descendant outputs, constraints and repeated records together. The split is performed once, with no reshuffling to improve balance or significance. Each producing-model cell has its own finite criterion frame and budget; all five cells share the cluster split and are resampled together. They are not independent datasets.

Reveal complete reference vectors only for pilot clusters. These are real reference acquisitions in the simulated information budget: pilot cost is the actual number M_pilot of revealed bits, not 20% by assumption. Evaluating a confirmation protocol at share b costs `M_pilot + round(b M_confirmation)` primary-reference bits. Also report the no-pilot control at this same total cost, capped at the confirmation census. Alternative-reference and validation-only reveals are recorded separately. Computationally public labels are hidden to the decision rule, not claimed to require expert work.

The forecast can use all frozen P/Q predictions and unlabeled task metadata, as in the original study, plus pilot R values. Confirmation R/R2 values remain inaccessible until the prediction file and source hashes have been saved. Synthetic fixtures and existing RuVerBench/JudgmentBench may be used to debug the pipeline; the confirmation labels may not be used to tune it.

## 5. Primary falsifiable test: predict matched-query update effects

The primary outcome is the **exact order-expected gated-minus-eager disagreement difference per 100 confirmation units under uniform criterion sampling**, at b = 0.20, 0.50 and 0.80. Use equal FP/FF costs. Exact expectation removes query-order Monte Carlo noise from this test. It does not remove uncertainty about transfer between tasks.

Build two forecasts from the **same paid pilot labels**:

1. **Task-linked forecast.** For each pilot unit, retain the joint P/R bit pattern. Evaluate its exact A and D event probabilities at the confirmation frame's M and m using the manuscript's hypergeometric formula. Average these per-unit contributions within predeclared observable strata, and multiply by the corresponding confirmation stratum counts. Strata start at (producing model, exact k, number of P-FAIL bits). If fewer than 20 unique pilot clusters contribute, pool in this fixed sequence: drop producing model; replace P-FAIL count by any-P-FAIL; use k alone; use the whole pilot. Tasks with zero P-FAIL bits or with every bit predicted FAIL have exactly zero update contrast. Enforce both cases before any pooling or bootstrap; fit the transport model only for 0 < number of P-FAIL bits < k. Publish the pooling choice and support count for every stratum. This is a transport model; its accuracy is not guaranteed by the event identity.
2. **Matched-stratum margin forecast (primary control).** Use exactly the same observable strata, pilot-unit pools, minimum cluster count and fallback rules as the task-linked forecast. Within each selected pool, estimate Pr(R=FAIL | P=PASS) and Pr(R=FAIL | P=FAIL) using Jeffreys 1/2 pseudocounts. Apply these probabilities independently to the known P bits of each confirmation task; enumerate possible R patterns (at most 2^8) and average their exact A/D probabilities. Enforce the same structural zero cases. This gives both forecasts the same observable stratification; the comparison does not reward the task-linked model merely for conditioning on k or predicted-failure count. Report a producer-only margin model and a zero-effect forecast as simpler controls. The comparison tests the utility of linked configurations relative to this matched control, not a causal intervention on dependence.

Before opening either confirmation reference, fit on pilot R and pilot R2 separately and save all 30 cell-by-budget-by-reference predictions for every predictor. Freeze allocation forecasts for both references at the same time. The primary statistic is the equally weighted mean absolute forecast error over the five producing-model cells and three fixed budgets. Let d be MAE(task-linked) minus MAE(matched-stratum margin), measured in disagreements per 100 units.

**Decision rule:** evidence for useful predictive transfer requires d < 0 and a 95% paired cluster-bootstrap upper limit below zero. If the interval spans zero, predictive advantage is unresolved. If its lower limit is above zero, the task-linked transport forecast is worse and the proposed predictive advantage is contradicted. A negative point estimate alone is not confirmation.

**Directional failures:** for any pre-unmasking forecast with magnitude >= 1 disagreement/100, report whether the realized contrast has the opposite sign. A wholly opposite-signed task-composition interval is a clear failed directional prediction. Forecasts within +/-1 are labelled near-zero rather than turned into favorable sign wins. The 1/100 threshold is an advance operational tolerance, not a detected scientific boundary. Report the entire error table, including cells in which both forecasts fail. No requirement that gating win or lose is imposed after seeing data.

Use 999 cluster bootstrap replicates, seeded 20261010, holding fitted pilot rules and parameters fixed and resampling whole confirmation clusters jointly across producing-model cells. Recompute M, budgets, forecasts from the frozen fitted rule, and exact outcome expectations for each replicate. Use percentile 2.5% and 97.5% endpoints. These intervals describe conditional task-composition stability given the pilot; a separate nested pilot/confirmation resample is secondary and cannot replace the primary result.

## 6. Secondary tests: allocation and query structure

At the fixed 20% confirmation cap compare criterion SRS, fixed 50/50 judge-first certification plus SRS, and pure judge-first short circuit. Use 101 priority seeds (20261011–20261111); the same seed couples paired policies. Each producer has its own M_c and m_c = round(b M_c), and total cost sums the five cells. Fix task order by SHA-256 of `refeval-task-order-v1|cluster_id|unit_id`; within-task priority ties and uniform global priorities use seed-specific hashes of immutable criterion IDs. Complete clustering/ID canonicalization before forming this order. Run the full 0–100% grid for display, but do not choose a new headline budget after seeing the maximum. Pure certification can stop below its cap; report both actual queries and outcomes. The SRS and mixed designs use the same actual total, transferring unspent phase-one budget to SRS.

Before unmasking confirmation references, replay these policies on the pilot and record per-100 predictions of delta C, delta C_W, delta D and delta A at 20%. Extrapolate the pilot per-unit means separately for each producing-model cell. Pool across cells only if a cell has fewer than 20 pilot clusters, as declared in the prediction file. The event-based repair forecast is `−delta C_W − delta D + delta A`. Compare its absolute confirmation error with a certificate-only forecast `−pilot_initial_error_fraction × predicted_delta_C_from_pilot`. The latter is a heuristic control, not a claim about prior literature or an optimal method.

This tests whether locating certificates and accounting for early repair improves prediction beyond certificate counts. Failure to improve, or a confident opposite-sign repair outcome when the pilot predicts a material gain, weakens the proposed external usefulness of this explanation. Post-hoc recomputation of the identity using all confirmation labels cannot rescue a failed forecast. These allocation analyses are secondary; their 95% intervals are descriptive, without a confirmatory multiple-claim success declaration.

Also run disagreement-first continuation using the frozen auxiliary Q, and serial judge-first querying, on the same confirmation frame. Serial `|E_gate − E_eager| <= 1`, census zero and the accounting identities are mandatory correctness checks. A failure stops analysis for debugging; it is not evidence against a mathematical result. Disagreement-first effects are fully reported but are not substituted for the uniform-sampling primary endpoint.

Accuracy-estimation precision, certificate counts and residual disagreements are reported together. Pure certification has no design-based accuracy interval; its logical feasible range must not be displayed as a confidence interval. Count validation-only labels and auxiliary predictions separately from the operational query budget.

## 7. Reference sensitivity and what would change our conclusion

Keep the common frame and P/Q vectors fixed. First re-evaluate the acquired R transcripts using R2 at the same positions. Then rerun adaptive policies using R2 responses. These distinguish reference interpretation from acquisition changes. For the primary uniform forecast, use the R2 model fitted and frozen alongside R before either confirmation set was opened, and compare against confirmation R2; do not pool R/R2 as independent tasks.

| Observation | Consequence for the paper |
|---|---|
| Task-linked prediction beats the margin model under both references | Supports transport of useful task-structure information to this native all-pass frame |
| Exact identities hold, but prediction does not improve | Keep the contribution as retrospective accounting; do not claim prospective guidance |
| Large forecasted delay is absent or has the opposite sign | Reject that external directional forecast; inspect only as an explicitly post-hoc explanation |
| More certificates bring no repair benefit | Report the outcome; check whether it was predicted by certificate placement rather than calling certification a proxy for repair |
| Main advantage depends on which LLM is treated as reference | Restrict the claim to reference-conditional behavior; do not describe model disagreement as verified error |
| All relevant effects are tiny | Evidence of limited practical importance in this frame, even if equations are exact |
| Matching/missingness gates fail | Feasibility failure, not a negative mechanism result and not permission to choose a favorable replacement |

The primary pass/fail decision is fixed in section 5. Secondary forecasts and reference checks qualify its scope; a favorable subset cannot overturn a failed primary endpoint. This study by itself does not establish expert-time savings or a deployment-optimal selector.

## 8. Preventing result selection and preserving a real holdout

Before any reference-unmasking: publish a dated protocol revision, upstream snapshot hashes, join/exclusion counts, cluster split, model-role manifest, code commit and analysis command. Then publish a hashed pilot prediction file before confirmation access. Save an exposure log: this screening read dataset documentation and earlier papers; some official pages displayed aggregate leaderboards or sample previews automatically, but no proposed policy effect was computed or used for selection. Do not claim the researchers have never seen any public result on these benchmarks.

Preserve every attempt, including parser failures, insufficient matches, missingness exclusions and zero effects. Amendments after pilot inspection are allowed only as labelled development changes, followed by a new prediction freeze while confirmation stays sealed. If confirmation has been opened, revisions become exploratory and require genuinely new confirmation data for stronger claims. No new budget, producer, reference or target may replace the primary result because it looks better. Cluster splitting does not guarantee absence from model training; this study tests analyst-unseen protocol outcomes, not model-training contamination.

No pilot-based protocol selector will be added to the manuscript as a completed result. If one is later studied, its loss, candidate set, total charged cost and no-pilot controls require an additional frozen specification. This plan's primary contribution is prediction of a mechanism, not a new RQ4.

## 9. Human/programmatic follow-ups and implementation readiness

ProfBench can test human-reference transfer with all 40 public tasks and all three supplied outputs. Use complete released fulfilment vectors and explicitly construct an all-pass target. Since only 40 task clusters are available, treat it as a small external replication with clustered uncertainty, not a powerful pilot-training benchmark. Before running any judge, freeze a checkpoint/prompt/inference configuration and cost cap; no model invocation has been authorized or performed by this design document. A publicly available local judge is possible, but compute is not costless. Report raw weighted outcomes separately without mixing their estimand with strict success.

For AGENTIF, first count tasks whose entire original condition set has executable independent programmatic checks. Run a separate judge on those same saved outputs if needed. Dropping semantic criteria from mixed tasks creates a new target and must not be presented as replication of native ISR. Resolve `None` provenance before any target is defined. IFBench's 44 multi-checker tasks are a small check, not the primary external sample. IFEval/SWE remain previously exposed controls.

WildIFEval assets cite Apache-2.0, but underlying LMSYS prompt terms restrict redistribution. Publish download instructions, version hashes, analysis code and non-text derived aggregates; do not rehost raw prompts or outputs without applicable permission. [Underlying dataset terms](https://huggingface.co/datasets/lmsys/lmsys-chat-1m/blob/main/README.md). ProfBench uses the NVIDIA Evaluation Dataset License for data and MIT for code; AGENTIF's data card specifies CC-BY-NC-4.0. InFoBench's code is MIT; tie the expert archive to its actual terms before redistributing it.

**Completed now:** source/schema screen, candidate decision, hypotheses, predictors, rejection criteria, split/cost/missingness rules and outcome-access plan. **Required before execution:** exact HF snapshot IDs and file hashes; masked full joins/counts; parser/synthetic probability checks; implementation/code freeze; pilot prediction freeze. No external score matrix, new policy result, judge call, human annotation or prospective success claim has been produced in this turn.


## 10. Masked execution check — 2026-10-15

The user authorized a masked full join and missingness check, followed by design freezing, pilot and confirmation only if usable. We downloaded 28 pinned source files (1,094,111,058 bytes) and checked identifiers, complete criterion sets, output metadata, numerical-value validity masks and duplicate connections. No score was thresholded for an external outcome, no pilot fit was produced, and no confirmation result was calculated. Numerical scores were excluded from the analyst-visible preflight outputs. The raw cache is not an encrypted holdout; stage controls prevent passing its reference values into analysis before the recorded gates.

Pinned benchmark: `fc5f47b2b1be19ac95986624a4c0d8fd25dd316c`; pinned predictions/scores: `e34bc0fcc01b7f1d49032878563e8c32393c9d59`.

| Planned check | Four-role common frame | P/R-only diagnostic |
|---|---:|---:|
| Complete current-benchmark tasks across all five producers | 4 | 40 |
| Producer-output units | 20 | 200 |
| Task clusters | 4 | 40 |
| Hash-split pilot / confirmation clusters | 0 / 4 | 8 / 32 |
| Retention relative to 7,523 benchmark tasks | 0.0532% | 0.5317% |
| Original feasibility thresholds | Failed | Failed |

The P/R-only frame was inspected as a metadata-only rescue diagnostic. It was not adopted as a new confirmatory design. No thresholds were lowered and no alternative model roles or criterion subsets were substituted.

The decisive problem is criterion alignment, beyond the sparse auxiliary/alternative-reference coverage. P and R share 9,779 task-text keys in each producer's files, but only 212 complete criterion sets agree exactly; whitespace normalization still yields 212. In 5,835 shared tasks the criterion counts differ. Requiring the complete current benchmark criterion set yields only 40 valid P vectors per producer, compared with 7,418–7,419 R vectors. Q and R2 each cover a small additional-grader subset. These are source-association findings, not performance comparisons or evidence that one judge is wrong.

The benchmark release supplies conversation IDs and decompositions, without the original task-text column. The join uses unique exact complete criterion sets to link these IDs to saved task-text keys. Thirty-one repeated sets involving 84 benchmark rows are excluded as ambiguous. Score artifacts identify the producing model and task text but omit an evaluated-output hash; exact output identity across grading runs remains a source-provenance assumption. Neither fuzzy semantic matching nor deleting unmatched criteria preserves the proposed native target.

**Decision: STOP before design/label-access freeze.** Section 3's gates fail, so there is no pilot, prediction-freeze file or held-out mechanism result. A coverage count alone initially suggested P/R could support a revised frame; the full criterion join rejected that possibility before any amendment or outcome access. Obtaining correctly aligned saved scores, or making a new fixed judge run on the exact released criteria, is necessary before reconsidering this route. Any such input or role change requires a new dated protocol revision and masked checks before labels are used.

The execution report, masked identity/exclusion tables, source hashes and independent verification are published with the code. The forecast and policy modules were checked only on synthetic fixtures; their presence does not mean this external study ran. This feasibility failure neither confirms nor refutes the manuscript's empirical mechanisms.

Continuation requires both compatible criterion sets and adequate coverage. Repairing P alone does not make the original four-role plan feasible: Q and R2 remain sparse. A narrower P/R-only study would require a pre-outcome amendment that explicitly drops auxiliary-query and alternative-reference claims, followed by the same feasibility checks. No such amendment has been adopted.


## 11. Expanded feasibility comparison — 2026-10-15 (recommendation only)

A metadata-only survey checked all six saved grading channels for the five fixed producers: 50 pairs/triples/four-channel combinations, 39 with distinct judge identities. Every combination fails the unchanged size/retention gates. The best distinct pair, DeepSeek-main plus Qwen, retains 337 tasks and 329 clusters (65 pilot, 264 confirmation), or 4.48% of the fixed benchmark denominator. The two DeepSeek runs are not independent judges. An independent parser reproduced 39 source hashes, six channel summaries and all 50 combination summaries. No accuracy, pass rate or policy outcome informed selection.

DeepSeek-main alone has 7,418 common complete tasks, 6,847 clusters and 134,485 criterion records across five producers (98.60% retention). We recommend generating one fixed new P on those exact outputs and full rubrics, then rechecking feasibility. This would require a separately dated two-role amendment: retain the original gates, split, budgets and forecast endpoint, but omit auxiliary-disagreement policies and complete R2 sensitivity. **This section records the recommendation, not its adoption or a label-access freeze.** Existing R lacks output hashes; its association still relies on release provenance. New P hashes do not repair historical R provenance.

The public ProfBench subset has 40 tasks, 120 outputs, 1,162 criterion definitions and 3,486 complete human-reference fields. It has no released judge predictions or repeated-rater vectors in the inspected inventory. Its native score is weighted; an all-pass target would be constructed. It is a subsequent small human-reference replication, not a substitute for the large forecast test. Keep three outputs clustered by task and avoid a claim that eight pilot clusters support general policy learning.

Offline prompt-length and price scenarios compare both routes. At assumed 32–128 total output tokens per request, the Luna price tier gives approximately $10–17 for WildIFEval versus $0.62–0.80 for ProfBench. These are planning estimates, not billing counts or execution authorization; model compatibility, actual token use, retry limits and a cash cap must be checked before any paid grading. At 1,000 output tokens per request the Wild estimate rises to about $76. ProfBench jobs must exclude human references and use immutable content IDs instead of shuffled resume indices.

The existing R frame would charge 26,375 pilot reference bits; a 20% held-out query budget adds 21,620, for 47,995 total. API dollars and revealed-reference counts are separate budgets. No pilot, new prediction freeze, held-out outcome, model invocation or reference thresholding was performed. See the local route report and public `docs/EXTERNAL_VALIDATION_OPTIONS.md` for source pins, costs and prospective execution steps. All three manuscript RQs remain unchanged.


## 12. Author-adopted two-role execution amendment — 2026-10-16

The author approved the recommended WildIFEval route and reuse of an existing local OpenAI key. This adopts the scientific scope below before any new-P results or reference outcomes are inspected. It does not certify an untested API response, establish actual generation cost, or issue the later completed-data/analysis freeze. Existing four-role failures and all original results remain intact.

### Roles and scope

P is a newly generated fixed judge on the exact benchmark criteria and saved outputs. The candidate is `gpt-6-luna` with no reasoning, selected for the previously documented interface and price, before outcomes. R remains the pinned saved DeepSeek-main scores. Keep all five producing models, the native strict target, exact criterion association, and the original >=500 total / >=100 pilot / >=300 confirmation clusters and >=70% per-producer retention gates. Drop Q-dependent disagreement continuation and complete R2 sensitivity from this two-role execution. Missingness excludes the whole task across all five producers, never only the failed criterion. Reapply the original hash-based 20/80 cluster split to the final complete common frame before any R access. Report original and final frame counts and every exclusion.

The generated jobs already bind 134,485 criteria, 37,090 outputs, 7,418 tasks and 6,847 clusters to immutable prompt/output/criterion hashes. The present candidate split has 26,375 pilot reference bits and 108,110 confirmation bits. These are preparation counts; new-P failures may change them. All job text stays local. Saved R's historical association still relies on upstream producer/task provenance because the release lacks evaluated-output hashes.

### Generation, compatibility and cash accounting

Use the pinned upstream `PROMPT_EVAL`, one criterion per request, one user message, temperature 0, `reasoning_effort=none`, `logprobs=true`, `top_logprobs=5`, `max_completion_tokens=1000`, `service_tier=default`, and `store=false`. The 1000-token cap preserves the released grader's maximum. Reproduce its probability parser: inspect at most the first five generated positions and use the first position with more than 0.0001 combined Yes/No probability mass, returning Yes/(Yes+No). PASS means probability >=0.5. Refusal, truncation or unparseable success is missing, not FAIL, and is not regenerated to select a parseable answer.

The initial generation attempt did not produce a completion. After the author restored API access, two unrelated synthetic cases passed at the full production configuration, returning `gpt-6-luna`, valid top-five log probabilities, 32 input and 4 output tokens each, and zero reasoning tokens. The returned fingerprint was null, so fingerprint-level version stability cannot be verified. Both the requested model and returned identity are bound in the generation configuration. The two calls cost $0.0000104 under the reported-token standard rates ($0.000012 under the conservative cache-write accounting); these are priced usage, not a provider invoice or a dataset cost estimate. Do not silently switch models after a failure.

The author-authorized implementation uses a100USD total ceiling and a1USD calibration-phase ceiling. After the compatibility diagnostic below, the continuing generation ledger is reduced to99.50USD and0.50USD is reserved separately for that diagnostic; the combined ceiling remains100USD. Keep an explicit conservative reserve for any request whose billing outcome is unknown, including0.001USD for the unsuccessful initial probe; the ledger also includes the0.000012USD conservative charge for the two successful synthetic probes; this is a reserve, not a claimed charge. Each future dispatch must durably reserve worst-case model-token cost before transmission. Use the documented922,000-token maximum input at the long-context cache-write rate plus configured maximum output at the long-context output rate, rather than treating a local tokenizer as a guaranteed API bound. Known completed calls may settle at conservative actual-usage rates. Only known usage can release reservations; timeouts and lost responses retain their reserve. Unexpected model, service tier, usage boundary or fingerprint changes stop the run. All totals exclude taxes and unrelated account activity; this is a client-side token-cost ceiling under the frozen rate table, not an organization-wide billing lock.

At current standard rates, short-context input/cached-input/cache-write/output are0.10/0.01/0.125/0.50USD per million; long-context rates are0.20/0.02/0.25/0.75. Save the applicable dated pricing source. Record actual returned usage, conservative priced usage and unresolved reservations separately; neither is presented as a finalized provider invoice. Existing offline $10–17 estimates remain scenarios, not observed costs.

A generation freeze binds exact jobs, prompt template, config, code and successful synthetic calibration. A small, preselected subset of new-P jobs then measures real prompt/completion usage under those same settings; its rows are retained and reused in full generation. It does not reveal R and is separate from the statistical pilot. The cost-calibration selection is the 32 smallest SHA-256 hashes of `fee-calibration-v1|job_id` within each producer, giving160requests selected without scores. Its manifest and cumulative1USD phase cap are fixed before those requests. Full generation requires a separate passed fee-calibration receipt on the same durable ledger; an output-directory change cannot reset the budget. Resume by immutable IDs, preserve successful calls, and allow at most three attempts for transient transport/rate errors. Quota, authentication, access or invalid-configuration errors stop globally; they are not ordinary retryable429s. No secret is stored in manifests or public artifacts.

### Analysis freeze, pilot and confirmation

After every planned job has reached a terminal state and new-P completeness passes all unchanged gates, issue the completed-data design freeze binding source/config/code/P-output hashes, eligible IDs, exclusions and final split. Only then open pilot R. Fit the task-linked and matched-stratum margin predictors plus the producer-margin and zero controls; freeze all15producer-by-budget predictions for every predictor and the allocation predictions before opening confirmation R. In this two-role amendment there are15primary cells rather than the original30including R2.

Retain section5's equally weighted primary MAE difference, budgets0.20/0.50/0.80, 999confirmation-cluster bootstrap draws, seed20261010 and fixed pilot parameters. Recompute the criterion universe, query budget, predictions and exact outcomes within every draw. An upper95% endpoint below zero supports the stated forecast advantage; an interval spanning zero leaves it unresolved; a lower endpoint above zero contradicts it. Preserve the1-per100directional-failure threshold, structural zeros and all cells. Do not replace the primary result with a favorable subgroup or budget.

Retain section6's SRS/mix50/serial comparisons,101seeds20261011–20261111, full0–100% display grid and20% headline allocation comparison. Report paired seed variability as query-order variability, not a task-composition confidence interval. No new allocation task-bootstrap success criterion is introduced in this amendment. Charge actual pilot reference bits and include no-pilot controls at matched total reference cost. Model-reference disagreements remain reference-conditional; the study does not estimate expert time or objective truth.

Publish this amendment and the code/metadata preparation record before reference access; publish the completed-data freeze and pilot prediction freeze at their respective stages. Current stage: reference-free jobs, offline tests, successful unrelated API calibration and a completed160-request fee canary. The canary returned152stored predictions and8terminal missing records; full scoring is running under the response-evidence repair below. No scoring dataset completion, final-data freeze, pilot fit, confirmation outcome, or support/refutation conclusion exists yet.


### Pre-reference compatibility correction after the fee canary — 2026-10-16

All160fixed canary requests returned known usage, costing0.014804375USD under conservative cache-write accounting. Including the initial reserve and synthetic probes, the same-ledger total was0.015816375USD. Stratifying the fixed canary by producer gives a full-generation usage projection of12.44455182421875USD before unrelated compatibility diagnostics. This projection does not guarantee the invoice or final coverage.

Eight responses had a usable Yes/No scoring position but failed the separate response-evidence serializer. The v1 serializer inspected all first-five positions, including positions after the score was already determined. In32unrelated synthetic probes, a valid scoring position0 was followed by a log probability of+0.000003814697265625 at position3. This reproduced the failure without retrying any dataset job or reading R. It establishes a real compatibility problem; it does not establish the exact malformed field in the eight dataset responses, whose unused raw positions were not saved. The diagnostic's conservative returned-usage cost was0.000484USD, within its separately reserved0.50USD.

The correction changes evidence serialization only: retain the prefix through the first valid scoring position, mark earlier invalid/skipped positions explicitly, and omit later positions that do not enter the score. The original probability calculation, threshold, model, prompts, API parameters, selected jobs and scientific plan stay fixed. Preserve v1's152successful results byte-for-byte and all8terminal missing jobs; none is regenerated. New records identify the evidence-serialization version.

Before continuing, archive v1 code/config/freeze, full exports and a consistent SQLite backup. Migrate the same ledger under an exclusive lock with no active requests; preserve all job states, attempts, recorded usage, cumulative charges and prior reserves. Bind predecessor artifacts, the diagnostic and the migration receipt in the new generation freeze. Lower the continuing ledger cap to99.50USD to reserve the external diagnostic allowance. Regenerate the fee receipt from the same160canary requests under the new configuration; do not select another canary. Analysis must verify both generations and the retained v1 successful/missing IDs before opening R. This is a pre-reference implementation correction, not a change chosen from mechanism outcomes.

## 13. ProfBench analysis plan before reference access — 2026-10-18

This section fixes the ProfBench analysis before any human fulfilment label is read. Section 9 named ProfBench as the human-reference follow-up, and this section carries out that route. It changes no endpoint, budget, split or decision rule in Sections 5, 6 and 12. ProfBench has 40 tasks, so this study is a small held-out replication of the manuscript's findings. It neither fits nor tests the pilot forecasts of Section 5. Notation follows the manuscript.

**Judge run.** Two Gemini judges scored every criterion through Vertex AI on 2026-10-18. The judges were `gemini-3.1-pro-preview` and `gemini-3.1-flash-lite` with thinking level LOW and temperature 0. We call them the Pro judge and the Flash-Lite judge. Each request sent the official ProfBench yes/no prompt for one criterion. The configuration, job manifest, source hashes and a 50 USD cash cap were frozen in `artifacts/reports/profbench_vertex/freeze.json` before the first request. An earlier freeze differed only in its 80 USD cap and was replaced before any request. All 6,972 requests returned a one-word Yes or No. The ledger records 27.37 USD of settled cost and 0.34 USD of unknown charges.

**Exposure before this plan.** No reference label value has been read. An earlier feasibility check read only the type of each fulfilment field. All 3,486 fields are Boolean and none is missing. The judge code drops every `_fulfilment` field while it parses the source file. Before drafting this section we viewed only rubric metadata and judge predictions. These included each judge's count of units predicted to pass and the number of criteria on which the judges disagree. Public ProfBench pages report aggregate model scores, but no outcome of this plan was computed from them.

### 13.1 Frame

**Units and clusters.** A unit is one released response to one task. Each of the 40 tasks has responses from o3, grok4 and r1-0528, so N = 120 units in 40 task clusters. A unit keeps every criterion of its task, and the frame has M = 3,486 criteria. Units follow release order. Tasks keep their order in the pinned `test.jsonl`, and the responses of each task follow the order o3, grok4, r1-0528.

**Reference.** The reference label of a criterion is the released Boolean fulfilment field for that response. The task's own annotator assigned these labels, and the release has one label per criterion. A missing or non-Boolean label removes the whole unit from every setting. Such a label never counts as FAIL, and every exclusion is reported with its cause.

**Predictions.** A prediction is PASS if the judge's stripped answer starts with "Yes". This is the rule of the official ProfBench parser. No prediction is regenerated or replaced.

### 13.2 Targets and settings

**Targets.** The full target requires every criterion of a unit to pass. It follows Section 9 and is the primary target. ProfBench gives each criterion one of four importance levels. The levels are Critical, Major, Minor and Additional, and their score weights are 4, 3, 2 and 1. The Critical-only target requires every Critical criterion of a unit to pass. We add it because full rubrics are long. Tasks have 15 to 59 criteria (median 27) but only 0 to 16 Critical criteria (median 7). A long conjunction leaves few reference passes and therefore few false fails to repair. Under the full target the two judges predict PASS for only 5 and 1 of the 120 units. We chose the second target from these label-free counts and the rubric metadata. Two tasks have no Critical criterion, so their six units leave this target. The Critical-only frame therefore has N = 114 and M = 834.

**Settings.** A setting pairs one judge with one target. The four settings share units and labels and are not independent replications.

**Disagreement flags.** Each judge supplies the flags for the other. A criterion is flagged if the other judge's prediction differs from that of the setting's judge. The judges disagree on 343 of the 3,486 criteria and on 73 of the 834 Critical criteria.

**Native score.** For each judge we also report the mean ProfBench weighted score under its predictions and under the reference. This estimand differs from all-pass success, so we report it separately (Section 9).

### 13.3 Analyses

A new reader packs ProfBench into the arrays of `src/conjunction_policy_data.py`. A driver applies the functions behind the manuscript tables to these arrays. Budgets, numbers of orders and seed schemes follow the manuscript, and the setting name keys each random stream. The analyses are the following:

1. **Initial verdicts (Table 1).** Units, criteria, reference passes, initial false passes and initial false fails.
2. **Update rules (Table 2).** The exact expectation 𝔼[Eᵍᵃᵗᵉ − Eᵉᵃᵍᵉʳ] = 𝔼[D] − 𝔼[A] under uniform and disagreement-first checking at every budget from 1% to 99%. The tables show 5%, 20%, 50%, 84% and 95%.
3. **Judge-first short circuit (Table 3).** Thirty-one orders with seeds 1 to 31. We report incorrect verdicts under eager updating for the three policies at 5% and 20%. We also report the mean share of criteria checked when every unit is certified.
4. **Threshold updating (Table B6).** Cost ratios ρ = 1, 3 and 9 with c = ρ/(1 + ρ). Flagged and unflagged criteria each have one miss rate with a Beta(1, 1) prior. Uniform and disagreement-first checking run at the 20 budgets of Table B6 with 200 orders. In the states where the two rules differ, we compare q̂ᵢ with the reference verdict. This comparison pools the four settings, both policies and the budgets 5%, 20%, 50% and 84%.
5. **Allocation (Tables 4 and C2).** The mix against criterion SRS at 5%, 20%, 50% and 60% over 31 paired orders. We report ΔC, ΔCᴡ, ΔD, ΔA, ΔEᵉᵃᵍᵉʳ and the RMSE of the agreement estimate.
6. **First-phase share (Appendix D).** Shares s = 0, 0.1, …, 0.9 at the same four budgets.
7. **Criterion-level reports (Table 5).** Each setting receives 64 rearrangements of its reference labels within D1 strata. They are evaluated under uniform and disagreement-first checking at the five budgets of Table 2. Each setting thus has 10 conditions. D1 strata share unit length, the setting's prediction and criterion category. The category is the unordered set of released criterion types. The integer program of Appendix F gives the D0 and D1 pass-rate bounds. The task-level counts give θ and θ̂ − θ (Proposition 4).
8. **Task composition (Table E3).** We draw 399 resamples of the tasks with replacement, and settings with the same target share each draw. Each draw keeps all three responses of a task and recomputes N, M and the budgets.

The alternative-reference analysis of Section 5.5 is omitted because ProfBench releases one label per criterion. Whole-unit sampling (Appendix D) is also omitted. It carries no transport claim, and its summary code is not among the frozen modules.

### 13.4 Transport claims

Five findings of the manuscript become transport claims with fixed classification rules. Each claim is classified on the released frame. The claims and the reasons for their thresholds are the following:

- **T1. Gating gains no ground at larger budgets.** In Section 5.2 every gating advantage at budgets above 28% was below 0.1 incorrect verdicts per 100 units. The quantity is 𝔼[Eᵍᵃᵗᵉ − Eᵉᵃᵍᵉʳ] at 50%, 84% and 95% under uniform and disagreement-first checking. These are 24 values across the four settings. A value is near zero if its absolute value is below one incorrect verdict per 100 units. T1 is consistent if no value is −1 or below and inconsistent otherwise. One verdict per 100 units is one percentage point of task-level accuracy. Section 5 uses the same tolerance for its directional forecasts. A threshold at zero would count negligible gating advantages as counterexamples. We also report how many of the 792 setting–policy–budget values are negative and the largest budget with a negative value.
- **T2. Threshold updating stays near the better fixed rule.** In Table B6 threshold updating had the smallest mean excess cost at ρ = 3 and ρ = 9. The quantity is the mean excess over the better fixed rule in the 160 cells of four settings, two policies and 20 budgets. T2 is consistent if threshold updating has the smallest mean excess at both ratios. Means within 0.01 per 100 units count as tied, and a tie for the smallest value counts as consistent. Table B6 reports these means to two decimals, so smaller differences count as ties. The excess at ρ = 1 is reported without a rule.
- **T3. Independence understates how often held units pass.** Section 5.2 found a mean q̂ᵢ of 0.53 in the states where the rules differ, whereas 77% of these units passed. The quantities are the mean q̂ᵢ and the reference pass share in the pooled states of item 4 of Section 13.3. T3 is consistent if the mean q̂ᵢ is below the pass share and inconsistent if it is above. It is uninformative if fewer than 20 distinct responses ever enter these states. With fewer than 20 units the standard error of a pass share can exceed 0.11. This is about half the gap of 0.24 found in Section 5.2. Section 5 also requires 20 clusters before it pools strata.
- **T4. Certificates cost precision.** In Section 5.3 the mix certified more units than criterion SRS and gave a less precise agreement estimate in all 36 setting–budget cells. The quantities are ΔC and the RMSE ratio of the mix to criterion SRS at 5%, 20%, 50% and 60%. T4 is consistent if ΔC > 0 and the ratio exceeds 1 in all 16 setting–budget cells. The manuscript found no exception, so a single exception counts against the claim. ΔEᵉᵃᵍᵉʳ has no rule because its sign differed between the two benchmarks.
- **T5. A closed form predicts the cost in precision.** Let s be the share of the budget actually spent in the first phase. In Appendix D the ratio √((1 − sb)/(1 − s)) was within 5% of the exact RMSE ratio at budgets of 5% and 20%. It was within 12% at 50% and 60%. The quantity is the largest relative deviation at each budget over the nine positive shares of item 6. T5 is consistent if these tolerances hold in all four settings. The tolerances are the accuracies that Appendix D reports for the retrospective data.

The remaining outputs are descriptive and carry no rule. Example 1 already shows that criterion-level reports cannot fix the preferred rule. We therefore report the number of reversals per setting. Section 5.4 counted reversals with both differences below one verdict separately, and we do the same. The D0 and D1 interval widths and the position of θ̂ relative to the D0 interval are also descriptive. The initial errors of Table 1 and the native weighted score complete the descriptive outputs.

### 13.5 Uncertainty

Forty task clusters give limited precision. For T1 we report the share of the 399 task resamples in which each value is −1 or below. For T4 we report the share in which ΔC ≤ 0 or the RMSE ratio is at most 1. These shares describe sensitivity to task composition. They are not confidence levels, and the classification uses the released frame only. T2, T3 and T5 are reported for the released frame only. No p-value is computed.

### 13.6 Execution order

The analysis proceeds in six steps:

1. Write the ProfBench reader and test that it rebuilds the frozen job manifest.
2. Run the full pipeline on synthetic reference labels that flip a seeded random 15% of the Pro judge's predictions. The run must reproduce Propositions 1 and 3 exactly and keep the serial bound of Proposition 2. Tests also check that the driver reproduces published manuscript values on RuVerBench and JudgmentBench.
3. Freeze the analysis. The file `artifacts/reports/profbench_vertex/analysis_freeze.json` records the SHA-256 of this section, of every analysis code file and of every input, together with the command.
4. Read the reference labels once through the frozen reader.
5. Run the frozen command and keep every output.
6. Report every setting, budget and claim regardless of direction.

A failed identity check or serial bound stops the analysis for debugging. Such a failure is not evidence about the claims. Changes after step 4 are exploratory and are labelled as such. No setting, target, budget or seed may replace a frozen one after step 4. The rules of Section 8 on preserving attempts and reporting every cell apply unchanged.

### 13.7 Consequences for the manuscript

The manuscript reports these results as a held-out replication whatever their direction. It reports all four settings with the full target first. A consistent claim supports transport to a human reference in new professional domains. An inconsistent claim is reported as a counterexample. The abstract and Section 1.1 then drop any generality that the counterexample contradicts. Near-zero results are reported as small effects in this frame. The judges may have seen the public ProfBench release during training. This replication therefore tests outcomes unseen by the analysts and not training contamination.

## 14. Primary-judge replacement before completion — 2026-10-21

### Reason

The Section 12 generation of P stopped before completion. Of 134,485 frozen jobs, 7,357 had stored predictions, 156 were terminal failures and 126,972 were pending. The ledger had reached its 99.50 USD cap. Known usage of the successful requests was 0.67 USD. The other 98.74 USD were reservations kept for 427 failed attempts, of which 426 returned HTTP 429. Section 12 released a reservation only on known usage, so rejected requests kept theirs. The OpenAI account then became unavailable to the author. No reference value R was opened. No P prediction was compared with any reference.

### Decision

P becomes `gemini-3.1-flash-lite` for all 134,485 frozen jobs. The job file is unchanged (SHA-256 `914683b47714464ed85fea1f7505cbadc5f91e3246b4c959052e5c62aa857cd3`). The 7,357 `gpt-6-luna` predictions and the 156 failures are archived and excluded from every analysis. P therefore comes from one model. A mixed P would depend on the order in which jobs ran. The new model is the Flash-Lite judge of Section 13 with the same configuration. Its frozen rates keep the projected cost within the cap. Section 12 forbids a silent switch after a failure. This section records the switch and its reason before any request is sent. The files of the stopped run under `luna_run/` stay unchanged.

### Request and parser

Each job is one request line of a Gemini API batch job, keyed by its job identifier. The line holds the job's pinned upstream `PROMPT_EVAL` text as one user message without a system instruction. The configuration is temperature 0, thinking level LOW and at most 2048 output tokens. Gemini 3 models return no log probabilities. The Section 12 probability parser therefore cannot be applied, and the answer text decides the label. The parser reads the first five alphabetic words of the answer and ignores thought parts. The first word equal to "yes" or "no", ignoring case, gives PASS or FAIL. An answer without such a word is missing, and so is an empty or blocked answer. A missing answer is not FAIL and is not regenerated.

R thresholds DeepSeek's probability Yes/(Yes+No) at 0.5. When the first answer token is Yes or No, greedy decoding returns the more probable of the two, which is the same rule. Vertex does not document temperature 0 as greedy decoding, and thinking tokens precede the answer. P and R therefore use closely related but different decision rules. The manuscript reports this difference.

### Batch execution and cost

Batch inference replaces online requests because its rate is half the standard rate. The requests go to the Gemini API, which serves the same model. The author's machine had no Google Cloud credentials for Vertex AI batch jobs. A Vertex AI online runner and a Vertex AI batch freeze were prepared first, and neither sent a dataset request. Two unrelated synthetic requests through Vertex AI on 2026-10-21 checked access and cost less than 0.001 USD. The frozen rates are 0.125 USD per million input tokens and 0.75 USD per million output tokens, with thinking tokens counted as output. These are the published Gemini API batch rates for this model.

Two unrelated synthetic online requests at the standard rate check access, the configuration and the returned model identity. The fee canary keeps the Section 12 selection, the 32 smallest SHA-256 hashes of `fee-calibration-v1|job_id` within each producer, giving 160 requests. It runs as one batch job, and its rows enter the final data. The remaining 134,325 jobs form 17 batch jobs in job-file order. Each batch job holds at most 10,000 requests and 28 MB of job records. This keeps each job below the Tier 1 limit of 10 million enqueued tokens. When the API refuses a new batch job at that limit, submission waits until a running job finishes. Each output line carries its job identifier as key. A line whose key matches no submitted job, or matches one twice, stops the run before that batch job is settled.

Each answered request settles at its returned usage and the batch rates. A request without a response is not billed. A batch job cannot stop at a cost boundary. The cap therefore governs submission. The canary is submitted only if its worst-case cost fits under the cap. This bound assumes two bytes per input token plus 50 tokens and twice the output cap for every request. The full run is submitted only if the producer-stratified canary projection times 1.25 is within the cap. Every later batch job is submitted only if the settled cost plus 1.25 times the projected cost of all uncollected batch jobs stays within the cap. If only the canary gate fails, the author may raise the cap with a recorded reason, because the canary reveals cost and no reference value.

A request that returns no response is resubmitted in a later batch job, at most twice. This resubmission depends on a missing response and never on an answer. A model identity other than `gemini-3.1-flash-lite` stops the run. A scenario based on the Section 13 Flash-Lite canary assumes about 124 output tokens per request and four characters per input token. It projects about 25 USD. This scenario is not an observed cost.

### Freeze, files and order

Requests use the author's Gemini API key. The script reads it at run time from an environment variable or an untracked local file. It never prints, logs or stores the key. The code is `src/wildifeval_gemini_batch.py`, with the shared parser, job index and prompt checks in `src/wildifeval_vertex_judge.py`. Private state and answer texts stay in `data/interim/wildifeval_external/flashlite_gemini_batch/`. Public receipts without text go to `artifacts/reports/wildifeval_external/flashlite_gemini_batch/`. The file `freeze.json` binds the job-file hash, the configuration, the parser version, both code hashes, the hash of this section, the canary list and the shard plan. Every run verifies them before it sends a request.

The execution proceeds in seven steps:

1. Write the freeze.
2. Send the two synthetic probes.
3. Run the canary batch job and apply the cost gate.
4. Submit and collect the 17 batch jobs, then resubmit requests without a response.
5. Export P with every missing record.
6. Adapt the Section 12 completed-data design freeze to this archive and test it on P alone.
7. Open pilot R.

Everything else in Section 12 is unchanged. This includes R, the five producers, the strict target, the retention gates, task-level missingness, the hash split, the MAE design, the budgets, the seeds and the allocation comparisons.

## 15. Parser amendment A1 before reference access — 2026-10-22

### Reason

Section 14 step 5 exported P on 2026-10-22. All 134,485 jobs returned an answer from `gemini-3.1-flash-lite`, at a settled cost of 29.84 USD. The section 14 parser labelled 91,984 answers PASS and 23,505 FAIL, and left 18,996 (14.1%) missing. The pinned upstream prompt ends with "Is the following constraint satisfied: ...? ###Answer:" and does not ask for Yes or No. Most missing answers state a verdict in words. A first diagnostic found that the first line of 10,923 says the constraint is not satisfied and the first line of 3,721 says it is satisfied. Of the rest, 1,785 are empty, 247 were blocked by the provider and 2,320 use other wording. Section 12 excludes a task across all five producers when any of its criteria is missing. Under the section 14 parser, 3,634 of 7,418 tasks (49.0%) stay complete, below the 70% retention gate of Section 3. The missing answers lean towards FAIL, so a complete-case frame would favour outputs that pass.

### What was inspected

Only P was inspected: its missingness masks, which Section 3 allows before reference access, and the stored answer texts of missing records. Draft versions of the rule below were run on those texts to check what each step matched. No reference value R was opened. No P prediction was compared with any reference, and no pilot or confirmation outcome exists. This is a labelled development change under Section 8. It precedes the completed-data design freeze of section 14 step 6.

### Rule

Parser A1 (`first-line-verdict-text-v2`) reads each stored answer once and applies these steps in order:

1. A label from the section 14 parser is kept unchanged. A1 never changes a PASS or FAIL of that parser.
2. An empty answer stays missing. This includes every answer blocked by the provider.
3. The decision window is the first non-empty line, lower-cased. Text inside double quotes, curly quotes or backticks is removed, so a quoted criterion cannot supply the verdict. Markdown emphasis characters are removed.
4. A first word followed directly by punctuation decides when it belongs to a closed list. The yes words are sí, sim, sì, ja, oui, да, ya, 是, 是的, はい, 네, 예, evet, بله, igen, kyllä and ναι. The no words are nein, nei, nej, nee, non, não, нет, ні, nie, tidak, 不, 不是, 否, いいえ, 아니요, 아니오, hayır, خیر, نه, nem, ei and όχι. Words that also open ordinary sentences in another language are left out, among them si, tak, так, da and ano. English yes and no are already handled in step 1.
5. A window that says the constraint is not applicable, not yet applicable, not evaluable, cannot be determined, evaluated, assessed, verified or judged, or is unable or impossible to determine it, stays missing. A window that says the constraint is partially satisfied stays missing.
6. Clauses opened by "if" or "whether" are dropped, as in "To determine if the constraint is satisfied, ...". A negative phrase is "not" or "n't", an optional "fully", "entirely", "completely", "clearly" or "successfully", then satisfied, met, fulfilled, satisfy, meet or fulfil(l); "fails to satisfy", "fails to meet" and "unsatisfied" also count. A positive phrase is "is", "was", "are", "were" or "been", the same optional adverb, then satisfied, met or fulfilled; "satisfies", "meets", "fulfils" and "fulfills" also count. A positive phrase is searched only after negative phrases are removed. Exactly one kind of phrase decides. A window with both kinds, or neither, stays missing.

Missing stays missing. It is never FAIL, and no answer is regenerated. The all-missing-as-FAIL replay of Section 3 remains an implementation sensitivity only.

### Roles of the parsers

Parser A1 gives P for the primary analysis and for every Section 3 gate. Two sensitivity versions of P are reported with every analysis that uses P. The first is the section 14 parser, with its own frame and its own gate result. The second is A1 with a partial verdict read as FAIL, because "satisfied" is a strict condition. If the section 14 parser fails a gate that A1 passes, the manuscript reports that failure. A1 widens the text rule for P. R still thresholds DeepSeek's probability Yes/(Yes+No) at 0.5. The manuscript reports this difference together with the one recorded in Section 14.

### Files and order

The code is `src/wildifeval_parser_v2.py`, with tests in `tests/test_wildifeval_parser_v2.py`. It reads the private export `results.jsonl` and checks its hash against `summary.json`. It also checks every stored label against the section 14 parser. The relabelled predictions go to `results_parser_v2.jsonl` beside the export. That file keeps every identifier and the section 14 label, and it contains no answer text. The public receipt `parser_v2_receipt.json` holds the counts per rule and producer, the labels under all three parsers and a P-only missingness preview. It binds the source hash, the output hash, the code hash and the hash of this section. This section, the code and the tests are committed to version control before `relabel` runs on the full export. Section 14 continues with step 6, using A1 for P.

## 16. Repair sensitivity R1 before reference access — 2026-10-22

### Reason and scope

After amendment A1, 4,430 of the 134,485 P labels are missing. A provider block (`PROHIBITED_CONTENT`) emptied 2,032 answers. The other 2,398 answers exist but give no PASS or FAIL that A1 reads. Of these, 1,221 abstain, 764 report partial satisfaction, 401 give no verdict in the first line and 12 contradict themselves. The pinned prompt never asks for Yes or No. The author approved asking these 2,398 jobs once more with that request. Blocked jobs are not re-sent. The provider gives no reason for a block, and this study does not try to get flagged content past the provider's filter. No reference value R was opened before this section.

### Request

Each repair job uses its frozen prompt with one added line, "Answer with Yes or No.", placed directly above the final "###Answer:". Everything else is the section 14 configuration: `gemini-3.1-flash-lite` in Gemini API Batch Mode, temperature 0, thinking level LOW, at most 2048 output tokens and the same batch rates. Each job is sent once. A request without a response is resubmitted at most twice, as in section 14. No answer is regenerated, whatever it says. Parser A1 reads the new answers, and an answer it cannot read stays missing.

### Role in the analysis

Repaired P keeps every A1 label and fills a missing A1 label with the repair label. It is a sensitivity analysis only. Its repair labels come from a different prompt than the rest of P, and the repair set was chosen by missingness. The primary analysis keeps A1. The manuscript reports the repair outcomes by A1 reason and the frame of repaired P beside the A1 frame. Repaired P does not change any Section 3 gate decision, which A1 already determines.

### Cost, files and order

The cap is 10 USD. A batch job is submitted only if the section 14 worst-case bound for its requests fits under the cap. The code is `src/wildifeval_repair_batch.py`, with tests in `tests/test_wildifeval_repair_batch.py`. It reuses the section 14 batch mechanics. Private state and answer texts stay in `data/interim/wildifeval_external/flashlite_gemini_repair/`. Public receipts without text go to `artifacts/reports/wildifeval_external/flashlite_gemini_repair/`. The file `freeze.json` binds the A1 predictions hash, the repair rules, the job list, the added line, the configuration, the code hashes, the hash of this section and the shard plan. Every run verifies it before it sends a request. This section, the code and the tests are committed before the freeze is written and before any request is sent.

## 17. Completed-data design freeze on A1 — 2026-10-22

### Adaptation of the section 12 freeze

Section 14 step 6 adapts the section 12 completed-data design freeze to the Gemini batch archive. The analysis code `src/wildifeval_external_run.py` keeps its whole-task missingness rule, its hash split of the final frame, its gates and its sealed pilot and confirmation stages unchanged. Only the verification of the generation archive differs. For a section 14 run freeze, `src/wildifeval_flashlite_design.py` replaces the OpenAI checks. It verifies the archive from the raw batch files to the final P metadata:

1. The job and unit files match the builder manifest. The saved reference files match the pinned source manifest by hash, and none of their values is read.
2. The run freeze binds the job file, both runner files, the configuration, the section 14 parser and the hash of section 14. The executing runner and parser files equal the archived ones.
3. Every request line carries its job's frozen prompt hash and the frozen request format and configuration. Every frozen job was sent. Every answer matches exactly one request and comes from `gemini-3.1-flash-lite`. The cost recomputed from returned usage equals the settled cost.
4. Every exported section 14 label and every A1 label reproduces from its raw answer. A1 changes no section 14 label. The A1 receipt binds the export, its output, the parser code and the hash of section 15.
5. The P metadata rebuilds exactly from the A1 labels through the unchanged section 12 frame construction, and the frame passes every gate.

The design freeze also binds the raw batch files, the configuration, this protocol and the repair receipts of section 16.

### Audit before the freeze

The archive audit above runs inside the freeze. A manual parser audit read samples of the answers without reference values. Sixty-one section 14 labels came from a yes/no word after the first word. Eight of them read a "no" inside the quoted criterion text. In three, the judge said the constraint was satisfied, and in five it said the constraint was not applicable. These eight labels stay as frozen, because section 15 forbids A1 from changing a section 14 label, and the manuscript reports them. Seven section 14 labels follow a judge that wrote "Yes, the constraint is not satisfied". The parser reads the first word as specified, and these labels are reported as judge self-contradictions. Random samples of 40 A1 verdict-phrase labels, 40 other-language first-word labels and 40 repair labels matched the judge's stated verdict in every case.

### Sensitivity versions of P

The split is recomputed on each final frame, so the section 14 parser, the partial-as-FAIL variant and repaired P each imply a different pilot set. A sensitivity frame could place in its pilot a cluster that the A1 frame assigns to confirmation. No sensitivity analysis therefore reads any reference value before the primary confirmation stage. Sensitivity analyses run after it. Their pilot is the A1 pilot clusters within their own frame, and every other cluster is confirmation. They are labelled as sensitivity results. The section 14 parser fails the retention gate, and this failure is reported.

### Order

This section, the adapter and its tests are committed before the design freeze is written. Section 8 requires the dated revision, the snapshot hashes, the join and exclusion counts, the cluster split, the role manifest, the code commit and the analysis command to be published before any reference value is unmasked. The pilot stage runs only after that publication.

## 18. Correction: the Vertex AI batch run completed — 2026-10-22

### What happened

Section 14 states that the Vertex AI batch freeze sent no dataset request. That statement is incorrect. The Vertex AI batch archive `flashlite_batch/` was frozen at 20:46 UTC on 2026-10-21 as `wildifeval-p-vertex-batch-v3`. It used the same job file, model, configuration and section 14 parser as the Gemini API run, under an earlier draft of section 14. At 21:20 UTC it sent the two synthetic probes and the 160-request fee canary, which returned 160 answers for 0.0366 USD. Between 21:28 and 21:30 UTC it submitted the other 134,325 requests as five batch jobs in Google Cloud project `gen-lang-client-0244811936`. The local process then stopped, and later attempts failed for lack of Google Cloud credentials. The Gemini API run was frozen at 21:33 UTC. A read-only status query on 2026-10-22 found that all six Vertex batch jobs had succeeded between 22:34 and 22:48 UTC on 2026-10-21, with 134,485 successful requests. Before this section, no output of the five full jobs had been downloaded or read. Neither ledger includes the Vertex charge. At the frozen batch rates it is about 30 USD, but the billing account has not been checked.

### Decision

P remains the Gemini API run labelled by A1, whatever the Vertex outputs show. The Vertex outputs are collected only for a replication check on P. Both parsers are applied to them unchanged. The check reports label agreement between the two runs overall, by producer and by A1 rule, and the task completeness of the Vertex run. It reads no reference value. A Vertex label never replaces, fills or filters a P label, and it never changes the frame.

### Design freeze

This section changes the protocol file, which the section 17 design freeze binds. That freeze (v1) is therefore superseded and archived unchanged. A new freeze (v2) is issued on the same P metadata and also binds the replication receipt. The audit of section 17 did not cover the Vertex archive. Everything else in section 17 is unchanged.
