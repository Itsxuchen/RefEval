# External validation design: task structure, update effects, and allocation

Date: 2026-10-10. Status: **design completed; metadata feasibility checked; execution and outcome access not started**. This is a prospective analysis plan for new-to-this-study outcomes, not a claim that an external validation has succeeded or that this draft has been externally preregistered. It does not change the current manuscript's three RQs.

## 1. Decision and the claim being tested

Start with **WildIFEval**, using its saved scores to test transfer to a native strict all-pass evaluation without new model inference. Its reference is an explicitly named LLM, not adjudicated human truth. Use **ProfBench** as the strongest human-reference follow-up if a new frozen judge run is feasible. Neither route requires Mercor. AGENTIF is a promising programmatic-reference extension, but its all-code complete-task subset has not been counted.

The identities `E_gate − E_eager = D − A` and `E_eager = E0 − C_W − D + A` follow from definitions. Reproducing them on another dataset checks the implementation; it does not independently confirm an empirical explanation. The falsifiable extension is:

> Can task-linked error configurations observed in a limited pilot predict the direction and size of the update effect on withheld tasks more accurately than criterion margins alone?

We will not predict that gating must be worse at high budgets in every environment. The existing paper already implies regimes in which delay is absent or offset by avoided false passes. A transport test must permit these cases to contradict a forecast.

## 2. Public-data feasibility screen

The screen inspected official descriptions, code, directory inventories and limited schema/headers. It did not compute the proposed outcomes. Search of current project source, progress, decisions and analysis inventory found no prior execution on the selected candidates. WildIFEval and AGENTIF were read as literature; IFEval and SWE-bench were actually analyzed earlier and cannot be relabeled unused confirmation data.

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
