# Methods

## Scope and analysis units

This is a retrospective finite-reference study. Each analysis defines its target relative to a specified released reference, and existing judge predictions remain fixed. A matched repeated-output extension changes the reference while fixing both primary and auxiliary predictions. Query policies are evaluated by replaying access to those labels. The study neither gathers new human judgments nor measures the time needed to produce a reference.

RuVerBench has 284 retained DeepResearch tasks and 210 AgenticCoding tasks. Five saved judge-by-domain runs cover these task sets. A task passes here if **all retained criteria** pass; this does not restore criteria removed by upstream filtering or establish complete original-task success.

JudgmentBench contributes 1,539 output-by-rater annotations over 30 base tasks, 1,314 distinct outputs, and 49 raters. A full strict outcome requires every positive binary item to receive full credit and every occurrence-count penalty to be zero. A binary-only sensitivity removes penalty items and therefore changes the target. The native benchmark's quality-evaluation purpose should not be conflated with this constructed conjunction. Two judges and two scoring targets produce four analysis cells, not four independent datasets.

The resulting nine cells belong to two data families. Repeated annotations, shared tasks, shared judges, and alternative targets are retained in the evidence rather than treated as independent replications. [Input provenance and retained fields](DATA_SOURCES.md)

## Four targets

For unit $i$, let $k_i$ be its number of retained criteria. Write $Y_{ij}\in\{0,1\}$ for the reference, $P_{ij}\in\{0,1\}$ for the primary prediction, $N$ for the number of units, and $M=\sum_i k_i$ for the number of criterion records. Reference and predicted task outcomes are $Z_i=\prod_jY_{ij}$ and $\widehat Z_i=\prod_jP_{ij}$.

| Target | Definition | Relevant evidence |
|---|---|---|
| Criterion microaccuracy | $A=M^{-1}\sum_{ij}\mathbf1(P_{ij}=Y_{ij})$ | Identifying agreement totals or a suitable probability-sampling estimator |
| Aggregate task success | $\theta=N^{-1}\sum_i Z_i$ | Identifying task information or a suitable unit-sampling estimator |
| Individual certification | A unit's outcome is fixed by the observed information | A reference certificate under the declared information model |
| Verdict repair | Movement of an existing prediction toward its fixed reference | A stated query set and update rule, with false passes and false fails tracked separately |

Criterion errors discovered among queried items are an additional policy diagnostic. They are neither an estimate of all errors without a sampling argument nor a count of corrected task verdicts. Category-balanced accuracy, when reported elsewhere for the judges, is different from the microaccuracy target $A$.

In JudgmentBench, task success weights each output-by-rater annotation equally and microaccuracy weights each retained criterion record equally. Neither weights base tasks equally. Fix the success event separately from the analytical objective; finite-frame SRS guarantees do not assert independent new tasks or raters.

## Queries and certificates

One query reveals one reference bit. Before choosing the next query, a policy may use predictions, disagreement flags, metadata, fixed priorities, and already revealed references. Unqueried references are used only by the retrospective outcome evaluator.

In the **query-only model**, unqueried bits are unrestricted and no trusted cross-task constraints are supplied. A unit is certified FAIL after one queried FAIL, or certified PASS after every bit has been queried and passed. An uncertified unit can still have a correct prediction. Certification and predictive correctness are separate outcomes.

For complete individual certification, the per-instance certificate-size bound is

$$
L(Y)=\sum_{i:Z_i=1} k_i+\#\{i:Z_i=0\}.
$$

This is classical conjunction-certificate logic. It allows a reference-dependent best certificate, whereas an implementable policy does not know failing locations beforehand. It is not a lower bound on the queries required to estimate $\theta$, a model of human effort, or a claim of a new optimal algorithm.

## Policy comparison

| Saved policy key | Selection and stopping rule |
|---|---|
| `random_criterion` | Shuffle all criterion records and query to the cap. |
| `sc_natural` | Visit contiguous units in the stated order; use published within-unit criterion order; stop that unit at the first observed FAIL. |
| `sc_random` | Use randomized within-unit criterion order with the same short-circuit rule. |
| `sc_judge` | Query predicted FAIL items before predicted PASS items within each unit; short-circuit on an observed FAIL. |
| `disagreement_only` | Query flagged disagreements in the shared priority order; stop when flags are exhausted. |
| `disagreement_then_random` | Complete the flagged phase, then query remaining records in global random order; no within-unit short circuit. |
| `sc_disagreement` | Within each unit, prioritize flags, then other predicted FAIL items, then predicted PASS items; short-circuit on observed FAIL. |

Disagreement methods use two additional complete saved judge runs in DeepResearch and one in AgenticCoding or JudgmentBench. Those predictions are additional information and should be reported separately from reference queries. The comparison does not convert them into free inference or a common monetary cost.

The full study crosses seven policies, three task orders, 101 criterion caps from 0% to 100%, and 32 seeds in nine cells. Task orders are release ID, ascending predicted-FAIL count, and random. Seed 0 preserves canonical ties for structured policies; seeds 1–31 randomize ties with shared priorities. Random policies are randomized at every seed. Some factor combinations produce identical trajectories and do not add observations.

Caps use `int(round(budget_share * M))`; the full and binary-only 20% caps are 4,697 and 4,046 respectively. Policies can spend less than the cap after exhaustion or complete certification; `actual_queries` records what was used. Compare policies at matched `actual_queries`; equal available caps alone do not establish matched spending. A stopped trajectory's plateau does not represent continued payment.

Leave-one-base-task-out analysis removes entire base-task blocks, including all related JudgmentBench annotations. It holds relative priorities fixed and recomputes the budget denominator. This is a finite-frame influence diagnostic, not unseen validation. The later robustness extension additionally reruns the fixed allocation after whole-base-task resampling; this has a different target from deleting contributions from fixed query sets.

## Updating is distinct from selection

**Eager updating** replaces each queried prediction with its correct reference and recomputes the task conjunction immediately. Correct replacements can expose previously masked errors. An initially correct FAIL is susceptible when every predicted FAIL is wrong, while at least one true FAIL remains hidden. Correcting the predicted FAILs before finding a true FAIL can temporarily create a false PASS.

Once any true FAIL is observed, the conjunction is settled. Perfect reference replacement cannot introduce a false FAIL on a truly passing unit. Cumulative transient events, maximum simultaneous false passes, and endpoint errors are different statistics.

**Certificate-gated updating** keeps the initial task verdict until the same queried labels certify an outcome, then updates it. Under correct fixed references it cannot introduce a new task error, but it can delay repairs of initially false FAILs. It changes the reporting rule, not the query set. The study reports prevented false passes, delayed repairs, and total residual errors rather than calling the rule a free improvement.

Along a fixed growing query transcript with correct references, gated absolute error cannot increase. Its difference from eager can nevertheless increase when eager repairs false fails earlier. Both rules agree on certified and untouched units. If $U$ is the number of partially queried, uncertified units, their total-error difference is bounded in absolute value by $U$; serial task-completing short circuit leaves at most one such unit. These are elementary properties of the rules, not new guarantees for noisy references.

### Comparison contracts

| Comparison | Held fixed | Varied and interpreted |
|---|---|---|
| Eager versus gated | Target, judge, policy, order, seed, acquired criterion set | Updating rule; endpoint FP/FF and residual-error consequences |
| Query allocation | Target, judge and stated actual-query resource | Acquired sets; certification, residual error and appropriately weighted precision |
| Whole-unit sampling | Fixed SRS unit count and sampled units | Full versus short-circuit label costs, which are random |
| Reference-link disclosure | Labels and scoring target, with stated trusted margins | Visible associations and finite feasible ranges; no new label acquisition |

The full-to-binary sensitivity also changes the success event, criterion universe, certificate lengths, priorities and absolute query count at a percentage cap. It does not isolate a penalty-type or rubric-length effect. Hidden references are available to the retrospective scorer, not to an online policy selecting queries. These comparisons do not establish a universally optimal deployment strategy or the prevalence of a mistaken industry practice.

## Estimation combined with certification

The label-value extension compares criterion SRS, pure judge-first short circuit, and `sc50_then_srs`. It uses 31 seeds, three task orders, and caps of 0.6%, 5%, 20%, 50%, 60%, and 100% in all nine cells. The small 0.6% cap is a design magnitude, not a reconstruction of any benchmark's auditing practice.

The fixed mixed allocation spends up to $\lfloor B/2\rfloor$ queries on certification. All remaining budget goes to an independent SRS without replacement from unqueried criterion records. The split is not optimized, and cap-specific mixed samples need not nest across budgets.

Let $C$ be the first-stage queried set, $R=M-|C|$ the number of remaining bits, and $S$ an SRS of size $n$ from the remainder. With $a_u=\mathbf1(P_u=Y_u)$,

$$
\widehat A=\frac{\sum_{u\in C}a_u+(R/n)\sum_{u\in S}a_u}{M}.
$$

Conditional on the first stage, this estimator is unbiased under the stated SRS design. Exact hypergeometric-tail inversion for the remaining correct-label total gives a conditional, pointwise interval with at least nominal 95% coverage at a fixed budget; discreteness can make it conservative. Census and empty-sample cases are handled explicitly. These are not simultaneous bands, confidence sequences, or intervals for the correctness of the reference itself. The design argument establishes the guarantee; 31 repetitions describe its observed performance. This guarantee refers to the specified mathematical interval construction. The implementation uses floating-point tails away from the threshold and exact integer/rational comparison near it. The inclusive-boundary defect is repaired, with no changes to 12,555 historical interval rows. See the [endpoint repair and validation](REPRODUCIBILITY.md#numerical-endpoint-repair-020).

Pure prioritized checking does not make naive sample agreement a design-unbiased estimate of $A$. Its saved `accuracy_ci_*` fields contain **logical bounds or a census value**, identified by `inference_kind=logical_bounds_or_census`; they must not be plotted as 95% confidence intervals.

### Whole-unit sampling

A separate control samples a fixed number of task or annotation units uniformly without replacement, then certifies each selected unit by either reading its complete rubric or short circuit. Both procedures observe the same sampled $Z_i$ and yield the same sample-mean estimator of $\theta$; short circuit can require fewer criterion queries.

This fixes the **number of units sampled**, while realized label cost varies with rubric length and failures. It is not an equal-actual-label-budget comparison with the allocation study. Full-rubric Horvitz–Thompson and difference estimates of criterion accuracy require all sampled bits and appropriate unequal-rubric weighting. Estimates outside $[0,1]$ are retained rather than clipped. Exact finite-frame design RMSE is an evaluation statistic based on the known population, not a deployable confidence interval.

## Information sufficiency and pairing

### Query-only identification

If $n_P$ units are certified PASS and $n_F$ certified FAIL, unrestricted completion gives

$$
\theta\in[n_P/N,\,1-n_F/N],\qquad
\operatorname{width}=1-(n_P+n_F)/N.
$$

This logical range is not a confidence interval. It can remain wide even when a valid sampling design provides a useful statistical estimate of the population mean. In the query-only, equally weighted model, more certified units strictly reduce its width, and full certification identifies the mean. With extra trusted constraints, a known count of one PASS among two units fixes the mean at one half without identifying either unit. The two-bit-margin example below has different implications: its unqueried second unit becomes logically certified from the extra constraint. Unqueried does not always mean uncertified.

### Reference-link disclosure: F27

The separate disclosure replay retains complete labels and prediction maps, hides reference-to-task links, and supplies exact reference-containing count summaries. D0 releases reference-success counts grouped by rubric length and predicted label. D1 further conditions on criterion category. Nested whole-task prefixes restore actual reference links across ten fixed random orders. No new labels are acquired.

For each release, binary-completion optimization finds the smallest and largest compatible reference conjunction rates. Its unknown criterion bits and task conjunctions obey exact disclosed counts, restored-label equalities, and conjunction constraints. Integer endpoint witnesses and solver results accompany the saved outputs. The intervals are finite feasible endpoint hulls, not confidence limits or claims that every continuous interior value is attainable. Task-uniform Mean Criteria is already identified by D0 and serves as a linear control.

The two-task bridge makes the distinction explicit. With two bits per task and a trusted total of two PASS bits, the aggregate pass rate initially lies in $[0,1/2]$. After the first task's two bits are observed PASS, it is exactly $1/2$ although the second task was not queried. Without that trusted total, the respective ranges are $[0,1]$ and $[1/2,1]$. Extra information changes identification; it does not improve a query policy at unchanged information.

The original public release already contains the links hidden in F27. This is a constructed disclosure experiment, not an identified reporting defect in that release.

### Optional pairing diagnostic: F28

JudgmentBench's constructed excellent and good arms are averaged within each of 30 base tasks and then task-weighted uniformly. Both marginal score multisets are known. Hiding their matching therefore leaves the finite-sample mean gap unchanged but makes covariance unknown. Restoring pairs restricts the possible matching; rearrangement bounds give sharp covariance and paired-standard-error extrema.

Mapping those standard errors to gap $\pm1.96\,SE$ and an illustrative 1 pp equivalence margin is an exploratory superpopulation diagnostic at $N=30$. It does not establish exact coverage, independently sampled raters, or a comparison between named solver systems. All six source-by-metric comparisons remain unresolved with complete pairing under that exploratory rule.

## Related work and operational use

The study concerns fixed-reference consequences of component queries and verdict updates. It does not introduce the general distinction between estimation and finding failures. [DeepSample](https://arxiv.org/html/2403.19271v1), its [LLM sentiment extension](https://repository.tudelft.nl/record/uuid:9cd84618-3d98-41a7-a120-6f772f2b0e1f), and a [code-model replication](https://arxiv.org/html/2606.27601v1) already compare objective-dependent sampling performance. The relevant additional question is what changes when a query reveals only part of a compound outcome and an existing verdict may be updated before that outcome is certified.

| Prior work | Unit and target | Intervention or guarantee | Relation to this study |
|---|---|---|---|
| DeepSample and its extensions | Test labels; estimation and failure exposure | Sampling choices, budget and context | Objective tradeoffs are established; this study measures component-query certification and updating consequences. |
| [Veneris and Hajj](https://www.eecg.toronto.edu/~veneris/tcad99.pdf), §V-C | Circuit errors and test vectors | Repair can expose a masked error | Error unmasking is established; here corrected reference bits can expose remaining errors in a conjunctive verdict. |
| [Correction and Corruption](https://arxiv.org/html/2604.18245v3), §§2.2,3.4,4.1 | Paired before/after task outcomes and fixed candidate outputs | Selective versus always-apply updates and correction/corruption accounting | Fixed-set controls and gating ideas already exist; our acquired criterion set and exact AND verdict are different objects. |
| [Boolean function evaluation](https://arxiv.org/abs/2111.08793v3) and [correlated certification](https://arxiv.org/html/2604.02611v1) | Queried bits and function values | Certificates and query-cost analysis under specified assumptions | Certificate logic is a foundation, not a new algorithm or optimality result here. |
| [Mind2Web 2](https://arxiv.org/html/2506.21506v2), Appendix D.2 | Rubric nodes and task scoring | Short circuit is disabled for complete node-level meta-evaluation | The operational distinction already exists; judge-call costs differ from fixed reference-bit queries. |
| [Prediction-powered evaluation](https://arxiv.org/html/2608.26638v2) and [limited-audit best-arm identification](https://arxiv.org/html/2601.21471v1) | Human scores or audited model outcomes | Population inference and system selection | These are legitimate different estimands; no efficiency victory follows from comparing their targets with certification coverage. |
| [Rao and Callison-Burch](https://arxiv.org/html/2606.00093v2) | Judgment protocols and agreement | Aggregation, missingness and finite bounds | F27 changes disclosed links under fixed labels and scoring; general reporting effects are established. |
| [Chen et al.](https://arxiv.org/html/2606.15031v2) | Repeated or named LLM reports | Calibrated partial identification of latent truth | Richer observations already have an identification role; F27 is a fixed empirical-reference application. |
| [Modular trajectory-risk certification](https://arxiv.org/html/2608.05199v1) | Per-stage certificates and joint audits | Statistical trajectory-risk bounds | Closely related composition/information theme; its risk certificate differs from an individual Boolean outcome certificate. |

Closer precedents constrain the general contribution claims:

| Prior work and reading location | Established result or design | Difference and limit relevant here |
|---|---|---|
| [Schreiber and Amsterdamer, ICDE2026 expanded v2](https://arxiv.org/html/2603.08612v2), §§2–3.2,5–6 | AND/OR verification, risky tuples and budgeted query-guided verification | Their worst-case MES and label-error probabilities differ from exact reference replacement and realized FP/FF. Their construction includes independent verification-error assumptions; neither mechanism nor downstream validation is claimed as new here. |
| [Krivosheev et al., CSCW2018](https://marcosbaez.com/assets/pdf/krivosheev2018combining.pdf), §§3–5; [2020 active screening](https://arxiv.org/html/2012.02297v1), §§2.1–2.2 | Item-predicate queries, finite-budget screening and stopping | Component querying already exists. This study uses fixed references and measures matched-query conjunctive updating consequences. |
| [DeepEST2021](https://arxiv.org/html/2102.04287v1), §§II–III; [Zhang et al., FSE2026](https://arxiv.org/html/2604.23342v1), §§3.3,5.3–5.4 | Failure-seeking with performance estimation; selection metrics across multiple objectives and shifts | Goal-dependent selection and lack of a universal winner are established. The fixed 50/50 allocation is not a new optimal method. |
| [ActiveClean2016](https://activeclean.github.io/files/activeclean-vldb16.pdf), §2.2 | Separate data selection and model updating; mixed clean/dirty data can harm downstream learning | The general separation and local-improvement warning are prior work; our exact AND replacement mechanism and observed consequences differ. |
| [Active Testing, ICML2018](https://proceedings.mlr.press/v80/nguyen18d.html), §§3.1–3.2,4.1–4.3 | Selective vetting with learned-posterior performance estimates | Not a design-unbiasedness or coverage guarantee under arbitrary adaptive sampling. Component labels alone do not distinguish this study. |
| [PRECISE, AAAI2026](https://ojs.aaai.org/index.php/AAAI/article/view/41427), [preprint v1 methods](https://arxiv.org/html/2601.18777v1) | Document-level judgments aggregated into query-level Precision@K with prediction-powered correction | Gold queries have complete top-K references under the stated query-sampling assumptions; this is population retrieval-quality inference, not arbitrary partial-bit AND certification. |
| [Jin and Chen2026](https://link.springer.com/article/10.1007/s10515-026-00638-5), §7; [GGC2026](https://arxiv.org/html/2607.28082v1) | Evidence-guided or learned selective correction | Their gate evidence differs from exact logical certificates. Selective correction itself is not our invention. |

These are bounded, source-specific comparisons, not a systematic review or an assertion that all neighbors have been found. The [bibliography](references.bib) also retains related active label cleaning and screening sources. The empirical contribution is the measured magnitude and conditional behavior of matched-query updates and common-resource allocation in the stated frames. A different combination of established ideas does not by itself prove novelty; no superiority over these methods is claimed without a common estimand and comparison.

Prediction-assisted inference and active testing also supply established estimation methods, including [control variates for language evaluation](https://aclanthology.org/P18-1060/), [PPI](https://doi.org/10.1126/science.adi6000), [Active Testing](https://proceedings.mlr.press/v139/kossen21a.html), and [Active Statistical Inference](https://proceedings.mlr.press/v235/zrnic24a.html). The fixed 50/50 design is a transparent comparison, not a claimed improvement over these estimators. Additional reading on robust sampling, PPAT, OPAL, rubric dependencies, graph aggregation, selective auditing and correlated judges is indexed in [the bibliography](references.bib). A literature match or a different combination of known concepts does not by itself settle novelty.

### What could change in an evaluation protocol?

| Intended claim or requirement | Design supported by the stated model | Cost and when not to switch |
|---|---|---|
| Estimate criterion accuracy | Criterion SRS, or independent remainder SRS after a fixed certification allocation, with correct weighting | Certification spending may widen intervals. Uncertified tasks alone do not invalidate population estimation. No superiority over all corrected active designs is established. |
| Estimate population task success | Fix a random sample of units and certify each sampled conjunction | Full versus short-circuit observation gives the same sampled task values with different label cost. Cost is random; this is not an equal-label-budget comparison. |
| Certify a specific outcome | Query until an observed FAIL or a complete observed PASS certificate | True PASS can require all retained bits. Universal certification is unnecessary when only a population mean is required. |
| Increase certification while retaining an accuracy estimate | A prespecified certification-plus-probability-sampling allocation | At the reported 20% cap, the mix improves both task metrics in JB, but DR has more certification and more residual errors. The intervals widen in both. The split is not optimized. |
| Avoid introducing errors through verdict revisions | Update a task only when the acquired references certify its outcome | This can delay false-fail repairs and can retain initial false passes. It does not make every displayed PASS certified. |

For the canonical JB `disagreement_then_random` set of 4,697 queries (release order, seed 0), eager updating leaves 28 false passes and 211 false fails; gating leaves 22 and 220. If nonnegative endpoint losses are $c_{FP}$ and $c_{FF}$, gating minus eager loss is $-6c_{FP}+9c_{FF}$. For positive $c_{FF}$, this **particular saved endpoint** favors gating above a loss ratio of 1.5 and eager below it; at equal weights gating leaves three more errors. The ratio is an algebraic illustration, not a fitted deployment threshold, measured monetary cost, or a trajectory-wide risk comparison. Costs must be specified by the user; unseen reference values cannot be used to select a deployed policy.

A hard requirement that every published PASS carry a reference certificate would additionally require withholding or distinguishing uncertified initial predictions. The implemented gating rule retains them; a publish-only-certified or abstention workflow was not evaluated. The empirical contribution candidate is the measured effect of these component-query and update choices, including contrary cases, under the stated finite reference frame.

## Interpretation limits

References are fixed, targets are explicitly constructed, and the data were explored before the final controls were frozen. The protocols document retrospective choices rather than preregistration or held-out confirmation. Order repetitions are not independent benchmark samples. All results concern reference bits and the declared update rules; they do not measure labor, economic utility, occupational competence, reference validity, or universal policy dominance.

See the [policy protocol](../context/conjunction_policy_protocol.md), [allocation protocol](../context/conjunction_label_value_protocol.md), and [result-specific counterexamples](RESULTS.md).


## Reference and composition sensitivity

The [robustness extension](ROBUSTNESS.md) uses 206 equally weighted repeated outputs, a fixed independent judge anchor and two hash-chosen whole rater vectors. Acquisition/evaluation references are crossed so that re-reading the same query set is distinguished from rerunning an adaptive selector. “Certification” concerns the reference supplying the queried labels. Whole-base-task resampling separately retains outputs/raters, recalculates budgets and reruns policies for 399 empirical compositions. The two sources of sensitivity are reported separately.

For criterion microagreement, proportional prediction strata and a charged-pilot Neyman control supplement SRS. Conditional on pilot observations, uniformly sample each remaining stratum, estimate its total, and add the known pilot correct count. Exact-tail stratum bounds with Bonferroni allocation yield a conservative conditional interval. Pilot costs and low-cap fallback are explicit in every row. The [protocol](../context/conjunction_robustness_protocol.md) specifies integer allocation and coverage conventions.

Three close antecedents clarify the study's contribution. [Ünlüyurt (2004)](https://doi.org/10.1016/j.dam.2002.08.001) reviews sequential component testing; its known-probability, independent-component setting gives the classical cost/failure-probability ordering for an AND outcome. Our judge-first heuristic and fixed global budgets measure several different outcomes. [Fisch et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract-Conference.html) develop StratPPI for efficient estimation; our finite-population stratified controls use a different interval construction. [Elangovan et al. (2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8798321486948322be2b4d658744ba72-Abstract-Conference.html) study human uncertainty in automatic-evaluator assessment; our matched-reference analysis measures its consequences for query paths, certification and verdict updating. These comparisons support the empirical positioning, not claims to have invented those principles.

The finite-budget mechanism model conditions on complete references to explain prevented false passes and delayed repairs. It is an explanatory calculation, not a free predictor available before checking. Its matched-margin rearrangement is a reporting-sufficiency example; it also changes task-pass rates. RuVerBench releases linked records, so the experiment concerns using only criterion summaries, rather than a defect in its release.
