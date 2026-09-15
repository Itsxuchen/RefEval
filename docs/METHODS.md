# Methods

## Scope and analysis units

This is a retrospective finite-reference study. Released labels are treated as correct for the defined target, and existing judge predictions remain fixed. Query policies are evaluated by replaying access to those labels. The study neither gathers new human judgments nor measures the time needed to produce a reference.

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

Caps are rounded to criterion counts. Policies can spend less than the cap after exhaustion or complete certification; `actual_queries` records what was used. Equal-actual-query comparisons require both policies to have consumed the same cap. A stopped trajectory's plateau does not represent continued payment.

Leave-one-base-task-out analysis removes entire base-task blocks, including all related JudgmentBench annotations. It holds relative priorities fixed and recomputes the budget denominator. This is a finite-frame influence diagnostic, not unseen validation. The later fixed-allocation extension has no additional leave-one-base-task-out study.

## Updating is distinct from selection

**Eager updating** replaces each queried prediction with its correct reference and recomputes the task conjunction immediately. Correct replacements can expose previously masked errors. An initially correct FAIL is susceptible when every predicted FAIL is wrong, while at least one true FAIL remains hidden. Correcting the predicted FAILs before finding a true FAIL can temporarily create a false PASS.

Once any true FAIL is observed, the conjunction is settled. Perfect reference replacement cannot introduce a false FAIL on a truly passing unit. Cumulative transient events, maximum simultaneous false passes, and endpoint errors are different statistics.

**Certificate-gated updating** keeps the initial task verdict until the same queried labels certify an outcome, then updates it. Under correct fixed references it cannot introduce a new task error, but it can delay repairs of initially false FAILs. It changes the reporting rule, not the query set. The study reports prevented false passes, delayed repairs, and total residual errors rather than calling the rule a free improvement.

## Estimation combined with certification

The label-value extension compares criterion SRS, pure judge-first short circuit, and `sc50_then_srs`. It uses 31 seeds, three task orders, and caps of 0.6%, 5%, 20%, 50%, 60%, and 100% in all nine cells. The small 0.6% cap is a design magnitude, not a reconstruction of any benchmark's auditing practice.

The fixed mixed allocation spends up to $\lfloor B/2\rfloor$ queries on certification. All remaining budget goes to an independent SRS without replacement from unqueried criterion records. The split is not optimized, and cap-specific mixed samples need not nest across budgets.

Let $C$ be the first-stage queried set, $R=M-|C|$ the number of remaining bits, and $S$ an SRS of size $n$ from the remainder. With $a_u=\mathbf1(P_u=Y_u)$,

$$
\widehat A=\frac{\sum_{u\in C}a_u+(R/n)\sum_{u\in S}a_u}{M}.
$$

Conditional on the first stage, this estimator is unbiased under the stated SRS design. Exact hypergeometric-tail inversion for the remaining correct-label total gives a conditional, pointwise interval with at least nominal 95% coverage at a fixed budget; discreteness can make it conservative. Census and empty-sample cases are handled explicitly. These are not simultaneous bands, confidence sequences, or intervals for the correctness of the reference itself. The design argument establishes the guarantee; 31 repetitions describe its observed performance.

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

This logical range is not a confidence interval. It can remain wide even when a valid sampling design provides a useful statistical estimate of the population mean.

### Reference-link disclosure: F27

The separate disclosure replay retains complete labels and prediction maps, hides reference-to-task links, and supplies exact reference-containing count summaries. D0 releases reference-success counts grouped by rubric length and predicted label. D1 further conditions on criterion category. Nested whole-task prefixes restore actual reference links across ten fixed random orders. No new labels are acquired.

For each release, binary-completion optimization finds the smallest and largest compatible reference conjunction rates. Its unknown criterion bits and task conjunctions obey exact disclosed counts, restored-label equalities, and conjunction constraints. Integer endpoint witnesses and solver results accompany the saved outputs. The intervals are finite feasible endpoint hulls, not confidence limits or claims that every continuous interior value is attainable. Task-uniform Mean Criteria is already identified by D0 and serves as a linear control.

The two-task bridge makes the distinction explicit. With two bits per task and a trusted total of two PASS bits, the aggregate pass rate initially lies in $[0,1/2]$. After the first task's two bits are observed PASS, it is exactly $1/2$ although the second task was not queried. Without that trusted total, the respective ranges are $[0,1]$ and $[1/2,1]$. Extra information changes identification; it does not improve a query policy at unchanged information.

### Optional pairing diagnostic: F28

JudgmentBench's constructed excellent and good arms are averaged within each of 30 base tasks and then task-weighted uniformly. Both marginal score multisets are known. Hiding their matching therefore leaves the finite-sample mean gap unchanged but makes covariance unknown. Restoring pairs restricts the possible matching; rearrangement bounds give sharp covariance and paired-standard-error extrema.

Mapping those standard errors to gap $\pm1.96\,SE$ and an illustrative 1 pp equivalence margin is an exploratory superpopulation diagnostic at $N=30$. It does not establish exact coverage, independently sampled raters, or a comparison between named solver systems. All six source-by-metric comparisons remain unresolved with complete pairing under that exploratory rule.

## Interpretation limits

References are fixed, targets are explicitly constructed, and the data were explored before the final controls were frozen. The protocols document retrospective choices rather than preregistration or held-out confirmation. Order repetitions are not independent benchmark samples. All results concern reference bits and the declared update rules; they do not measure labor, economic utility, occupational competence, reference validity, or universal policy dominance.

See the [policy protocol](../context/conjunction_policy_protocol.md), [allocation protocol](../context/conjunction_label_value_protocol.md), and [result-specific counterexamples](RESULTS.md).
