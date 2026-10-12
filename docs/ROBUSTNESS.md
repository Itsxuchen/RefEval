# Reference choice, task composition and estimation controls

The extension tests three limits of the original results. It holds judge predictions fixed while changing the human reference, reruns policies after resampling entire base tasks, and compares estimation against two budget-accounted stratified controls. The original large JudgmentBench update effect persists in direction on the matched reference subset but changes in size. Some other update comparisons reverse. At the focal 20% budget, the allocation benefit in JudgmentBench survives the examined reference and task-composition changes; the small RuVerBench error increase does not survive task-composition changes consistently.

In the original full-target frame, GPT-5.4 predicts PASS for 31 of 1,539 annotations (2.0%), compared with 227 reference PASS annotations (14.7%); seven are recognized by the judge. Its predictions are strongly FAIL-dominated, which shapes the repair and certification results. The target itself is not constant.

These are retrospective checks on the existing releases. The [protocol](../context/conjunction_robustness_protocol.md) fixes their finite grids and estimands. “Error” below means disagreement with the named released reference. Full strict and binary-only define different success events; both are constructed conjunction targets. The references have not been adjudicated by this study.

## Reference choice on the same outputs

JudgmentBench contains 206 outputs with repeated human annotations. The original 431 annotations of those outputs are weighted by annotation. Our matched analysis instead gives each output one unit, chooses two distinct raters by an ID-hash rule, and fixes both primary and auxiliary predictions from an independently hash-chosen saved judge assignment. A and B are symmetric panel labels, with no temporal or quality ordering; the selected raters can differ across outputs. There are 206 units under either reference, with 3,098 full-target or 2,664 binary-only criterion records. Changing reference therefore does not silently change the judge run, rubric, output sample or unit weights. The [pair map](../artifacts/expected/conjunction_robustness/reference/pairing.csv) records the selections.

The [frame diagnostics](../artifacts/expected/conjunction_robustness/reference/frame_diagnostics.json) bridge three frames: all 1,539 original annotations, their 431 repeated-output annotations, and the new 206 equally weighted outputs. In the original repeated-annotation subset, 32 of 54 initial false fails have at least one peer rater who also says FAIL; only 22 of 56 reference-PASS annotations have all peers agree PASS. These are disagreement diagnostics, not known reference-error rates or percentages of the whole-frame update effect.

Holding the original query population and design fixed, the expected 108.216 delayed repairs at the 84% cap decompose as follows. This is a partition of **D**, not of the net effect **D − A**; it neither deletes annotations nor changes their references.

| Other raters on the same output | Initial false fails | Expected delayed repairs | Share of D |
|---|---:|---:|---:|
| At least one says FAIL | 32 | 15.455 | 14.3% |
| All say PASS | 22 | 8.700 | 8.0% |
| No other rater available | 166 | 84.061 | 77.7% |

Most of the delay therefore has no repeated-rating check. The observed 32/54 peer-disagreement proportion cannot be extrapolated to the 166 unreplicated annotations. The [manuscript numerical audit](../artifacts/manuscript/numerical_claim_audit.json), `claims.false_fail_diagnostics`, reconstructs this table directly from the original criterion bits and integrates the random continuation exactly.

### Which wrong rejections do disagreement flags reach?

Among the 220 full-target initial false fails, 79 have no flagged criterion, 80 have flags only on predicted-PASS items, 59 have some erroneous rejections flagged, and two have every erroneous rejection flagged. Thus 159/220 (72.3%) have no wrong rejection flagged. None has its entire rubric flagged. Erroneous penalty rejections are flagged in 21/274 instances (7.7%), compared with 163/221 binary instances (73.8%). The changed binary-only target has complete rejection coverage in 36/67 initial false fails. This describes the scoring target's error structure, without identifying model-family membership as its cause.

An exploratory repeated-rating check within these full-target false fails finds another rater rejecting 15/62 penalty instances jointly rejected by the judges, versus 6/11 analogous binary instances. Among flagged wrong rejections the counts are 0/3 penalty and 20/32 binary instances. These are annotation-criterion instances with shared outputs, not independent samples. The small selected groups do not establish that shared penalty rejections are the most unreliable references or adjudicate the raters. Counts are regenerated in the same audit under `exploratory_peer_items`.

For the fixed GPT-5.4 anchor in the 206-output full-target frame, reference A has 31 PASS outputs and reference B has 23; the judge has four under either reference. The two references disagree on 32 output verdicts. Their criterion agreement is 2,462/3,098 (79.47%). This selected pair statistic has a different weighting from the all-pairs or all-annotations diagnostics.

### Updating: size can change, and some directions reverse

For disagreement-first continuation, the same nonadaptive query sets support both reference evaluations. Entries are means over 31 query orders, in **gated minus eager residual disagreements**:

| Fixed GPT-5.4 frame | Budget | Reference A | Reference B | A / B per 100 outputs |
|---|---:|---:|---:|---:|
| Full strict | 20% | +0.26 | +0.23 | +0.13 / +0.11 |
| Full strict | 84% | +12.42 | +8.00 | +6.03 / +3.88 |
| Binary-only | 20% | −0.13 | −3.61 | −0.06 / −1.75 |
| Binary-only | 84% | +4.29 | +1.48 | +2.08 / +0.72 |

Thus the large high-budget penalty is reference-sensitive even where its direction persists. The low-budget advantage in the original 1,539-unit full frame is absent under both references in this matched subset. That difference also involves changing the sample, weights and anchor draw; it is not a pure reference-replacement effect. On the *same* 206 outputs, binary-only random criterion sampling at 20% changes from +1.00 with A to −1.00 with B. For the mini judge under binary-only disagreement-first continuation, the same endpoint changes from −1.48 to +0.94. The complete grid retains these contrary cases. At the smallest 0.6% allocation cap, some binary-only reference configurations lose certificates and increase disagreement; the 20% benefit is not a whole-grid dominance claim. [RQ1 table](../artifacts/expected/conjunction_robustness/reference/rq1_summary.csv)

### Allocation: distinguish re-reading from reacquiring

For adaptive acquisition, reference replacement can change which items get queried. We therefore cross the reference driving acquisition with the reference supplying both queried replacement labels and endpoint truth. An off-diagonal cell retains acquired positions but re-reads their labels. It is a counterfactual fixed-set evaluation, not an executable transcript in which one keeps A's answers while declaring B to be truth.

At 20% full-target budget, both policies spend 620 queries. Entries are paired **50/50 allocation minus SRS** means over 31 orders:

| Reference driving acquisition | Reference used to read and score | Extra certificates | Residual disagreement change |
|---|---|---:|---:|
| A | A | +27.65 | −11.68 |
| A | B | +14.90 | −4.52 |
| B | A | +13.58 | −7.26 |
| B | B | +32.13 | −8.71 |

All four mean contrasts increase certification and reduce disagreement. The cross-reference comparisons still contain individual contrary query orders: A/B increases disagreement in 2 of 31 orders, and B/A in 1 of 31. The two diagonal analyses retain a benefit, while holding A’s acquired set fixed exposes a larger reference-dependent change. These subset counts cannot be subtracted directly from the original −90.06 result. [RQ2 table](../artifacts/expected/conjunction_robustness/reference/rq2_summary.csv)

![Fixed-judge reference sensitivity](../artifacts/figures/conjunction_robustness_reference.png)

Top: disagreement-first continuation update differences on the same 206 outputs. Bottom: allocation differences; dashed curves hold queried positions from one reference and re-read using the other. Lines summarize 31 orders, without population confidence bands. The full and binary-only percentage caps correspond to different absolute query counts.

## Base-task composition sensitivity

We resample complete base-task blocks 399 times, preserving all their outputs, raters and criteria. The two JudgmentBench targets share the same base-task multiplicities. Every replicate has its own unit/criterion totals, rounded query budgets and rerun policies. RQ1 averages query orders analytically; RQ2 averages the same 31-seed panel within each replicate. This is sensitivity to the empirical task composition. Population confidence claims would additionally require a defensible sampling model for the curated tasks.

![Three frames over the complete budget](../artifacts/figures/conjunction_robustness_budget.png)

Disagreement-first continuation, 101 budget caps. Both rows use common vertical scales across frames. Shading is the pointwise 2.5th–97.5th percentile across compositions. Every curve is exactly zero at 0% and 100% budget. The original constant-width simultaneous bands remain in the source tables and plotting CSV but are omitted here: their constant radius obscures the known endpoints. Peaks are recomputed in every replicate. [All curves and summaries](../artifacts/expected/conjunction_robustness/cluster/summary.json)

For full JudgmentBench, the original order-expected gated-minus-eager difference at 20% is −0.13 per 100 annotations; its composition range is [−0.54, +0.39]. At 84%, it is +6.26, with range [+3.51, +9.44]. The high-budget burden and the small low-budget advantage thus have different stability for this query policy. This composition study retains the original reference and annotation weights; the separate 206-output A/B study does not provide a joint reference-by-composition interval. Original full and binary-only count curves retain their different scales, rather than being represented by a single favorable endpoint.

For the 20% mixed allocation:

| Frame | Original residual change per 100 units | 2.5th–97.5th composition range | Replicates with fewer disagreements |
|---|---:|---:|---:|
| JB GPT-5.4 full | −5.85 | [−7.99, −4.09] | 399 / 399 |
| JB GPT-5.4 binary-only | −3.81 | [−4.95, −2.05] | 399 / 399 |
| DR Gemini | +0.28 | [−0.91, +0.51] | 196 / 399 |

All three frames gain certificates in all 399 replicates. DR has 198 disagreement increases and five ties. Its original small increase is therefore an example of certification without dependable repair benefit, rather than a stable claim that the allocation makes DR worse. The table's original binary-only value must be read from the saved paired result, not from marginal medians. [Allocation resamples](../artifacts/expected/conjunction_robustness/cluster/rq2_frame_means.csv.gz)

## Budget-accounted stratified controls

The estimand is criterion-level microagreement with the fixed reference. SRS remains a transparent control. Proportional stratification uses the primary predicted PASS/FAIL label. A second design charges about 20% of its cap to a pilot, estimates within-stratum variability from those observed labels, then applies a capacity-constrained Neyman allocation to the remaining items. All pilot and subsequent labels count toward the same cap. Tiny caps use an explicitly recorded SRS fallback.

Conditional on the pilot, the estimator adds its known correct total to the estimated remaining totals. Each noncensused remaining stratum gets a hypergeometric count interval, with Bonferroni error allocation; their endpoints sum to a conservative conditional 95% interval. These are finite-population stratified controls, not a reimplementation of StratPPI. The latter is an essential related approach with different inferential machinery. [Fisch et al.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract-Conference.html)

The figure compares all nine analysis cells at the 20% label cap. Proportional stratification lowers exact design RMSE by **0.03–4.9%** in every cell. The charged-pilot Neyman design raises RMSE by **1.1–9.5%** in eight cells and lowers it by **0.9%** in DR Qwen. The finite sum integrates every possible pilot count outcome under the actual smoothed, integer allocation and conditional estimator. These results concern this implementation and budget.

![Equal-budget interval widths and exact design RMSE in all nine cells](../artifacts/figures/conjunction_robustness_estimation.png)

Left: mean conservative conditional 95% interval width over 31 orders. Right: exact design RMSE for fixed-frame criterion microagreement. Both axes use percentage points, and all pilot labels count toward the budget. The original empirical RMSE over 31 orders remains a Monte Carlo diagnostic in the archived simulation tables.

For example, proportional stratification changes AC DeepSeek RMSE from **1.84 to 1.77 pp**, DR Gemini from **1.31 to 1.28 pp**, and full-JB GPT-5.4 from **0.558 to 0.542 pp**. Their mean guarantee-based interval widths increase from 7.54 to 11.82, 5.45 to 8.18, and 2.21 to 3.47 pp. Small point-estimation gains coexist with wider intervals because the interval construction combines per-stratum bounds conservatively.

The fixed 50/50 mix has an estimated **33–39% higher RMSE** than SRS across all nine cells. Each of its 31 first-stage transcripts yields an exact conditional variance; their average estimates the overall variance. The square root of this average is the reported RMSE estimate, with a delta-method Monte Carlo SE. It does not exactly integrate the first-stage randomness. [Design risks](../artifacts/expected/conjunction_robustness/estimation/design_rmse.csv) and [conditional variances](../artifacts/expected/conjunction_robustness/estimation/mix_conditional_variances.csv) distinguish the two calculation types.

Neither stratified guarantee-based interval is narrower in the 45 noncensus cell×cap comparisons. This includes the conservatism of combining per-stratum bounds. Approximate normal intervals are separately recorded for all methods using the same finite-population variance construction. In full JB, proportional stratification narrows their mean width from 2.19 to 2.13 pp, but only 28/31 intervals cover the fixed truth in these repetitions. Width alone therefore cannot rank estimators, and these repetitions cannot establish a coverage guarantee. [Complete estimation results](../artifacts/expected/conjunction_robustness/estimation/stratified_summary.csv)

## Numerical repair and validation

The inclusive hypergeometric endpoint now returns [0,13] for M=16, n=2, x=0 at 95%. A threshold-near floating-point result triggers exact integer/rational tail comparison. Exact enumeration checks all observation acceptance sets and coverage for M≤24, with separate two-stage estimator and conditional-coverage tests. The regression compares 11,718 saved label-value intervals and 837 saved whole-unit intervals: **none of the 12,555 saved rows changes**. Original tables and their provenance remain historical artifacts. [Interval regression](../artifacts/expected/conjunction_robustness/estimation/interval_regression.json)

The reference implementation scores query masks directly and checks unchanged nonadaptive transcripts. The cluster implementation checks its vectorized short-circuit transcript against the original replay. Tests enumerate small worlds; a fresh public reproduction additionally compares all new scientific tables row by row. File hashes establish file integrity separately from those arithmetic checks. [Reproduction commands and receipts](REPRODUCIBILITY.md)

## What the evidence contributes

The measured question is how limited checks translate into estimates, certificates and changed verdicts. Delayed repairs depend on initial false fails, the locations of predicted failures and how likely the remaining checks are to finish. They do not equal a false-fail count multiplied by rubric length. Changing the reference or task composition can change that balance.

Sequential testing already supplies short-circuit logic and cost-sensitive check ordering. Human-uncertainty work already establishes that judge assessment depends on reference choice. The new evidence here measures the consequences jointly for fixed-label acquisition and conjunctive verdict updates, with counterexamples and sensitivity analyses on the released frames. [Ünlüyurt](https://doi.org/10.1016/j.dam.2002.08.001), [Elangovan et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8798321486948322be2b4d658744ba72-Abstract-Conference.html)

The earlier matched-margin rearrangements show that criterion confusion summaries can be compatible with different downstream checking outcomes. This is a concrete reporting-sufficiency example built on classical dependence facts. RuVerBench releases linked records, enabling our analysis; the issue is relying on its criterion summaries alone. The rearrangements also change task pass rates, so they do not isolate a causal correlation effect. [Mechanism tables](../artifacts/expected/conjunction_mechanism/update_permutation_summary.csv)


## Timing and sensitivity interpretations

Full JB at 50% budget has an original effect of +2.91 disagreements per 100 annotations, with pointwise task-composition quantiles [1.05, 5.20]; all 399 resamples are positive. At 84%, the range remains [3.51, 9.44]. Binary-only at 20% has range [−0.44, 1.50], with 328 positive, 65 negative and six zero resampled effects. These resamples retain within-task reference structure. Matched-margin label rearrangements deliberately alter it, so their reversals concern information sufficiency.

AC DeepSeek has 117/210 reference-PASS tasks. Disagreement-first continuation at 20% has expected A = 0.9011, D = 12.0536 and D−A = 11.1526 tasks (5.3107 per 100). This comparison lacks JB penalty items but remains a constructed conjunction. The coding supplement below supplies AC task-composition checks; no AC reference-panel experiment is available.

Full-JB delay at 84% splits into 77.8711 on 156 penalty-only-rejected false fails and 30.3450 on 64 false fails rejected by both item types. This partitions original D = 108.2161 without changing acquisition; it is not an intervention deleting penalties. Low flag coverage delays eager repair but does not itself increase the gap to certification. The last mistaken-rejection query and completion of the remaining rubric jointly determine that gap.

<a id="coding-task-sensitivity-supplement-local-rc4"></a>
## Coding-task sensitivity supplement

At a 20% cap (169 of 843 criteria), disagreement-first continuation yields +5.3734 tasks for AC Qwen and +11.1526 for AC DeepSeek, or 2.5588 and 5.3107 per 100 of the 210 tasks. Under uniform criterion sampling, the corresponding effects are 0.6755 and 1.1416 per 100. These are exact query-order expectations in the original frames.

| Judge / policy | Original per 100 | 2.5–97.5% composition range | Positive draws / 399 |
|---|---:|---:|---:|
| Qwen / uniform | 0.68 | [0.28, 1.30] | 399 |
| Qwen / disagreement-first continuation | 2.56 | [0.65, 4.95] | 396 |
| DeepSeek / uniform | 1.14 | [0.55, 1.83] | 399 |
| DeepSeek / disagreement-first continuation | 5.31 | [2.50, 9.13] | 399 |

Each draw resamples 210 complete base tasks, recomputes the criterion count and budget, and integrates over query-order randomness. Both judges share the AC task multiplicities. The retrospective protocol fixed both judges, both policies, all 101 budget points, seed 20261008 and 399 draws before this supplement was computed. These ranges describe task-composition sensitivity, not an external population or reference-adjudication interval. Three Qwen draws reverse the disagreement-first effect; all are retained.

Disagreement flags cover every wrong rejection in 6/10 Qwen and 12/16 DeepSeek initial false fails. None has every rubric item flagged. Of 843 criteria, 95 are flagged, followed by 74 random checks at the 20% cap. This contrasts with 2/220 full-JB false fails and explains earlier delayed-repair exposure in coding. In the existing 64 matched-margin DeepSeek configurations, the 20% disagreement-first effect ranges from 2.07 to 12.27 tasks and remains positive; those rearrangements are a different analysis from the 399 composition draws.

[Protocol, section D](../context/conjunction_robustness_protocol.md) · [Full-grid summary](../artifacts/expected/conjunction_coding_sensitivity/summary.json) · [Curves](../artifacts/expected/conjunction_coding_sensitivity/rq1_curves.csv.gz) · [Task multiplicities](../artifacts/expected/conjunction_coding_sensitivity/multiplicities.csv.gz)
