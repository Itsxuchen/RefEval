# What Should Reference Labels Buy?

### Estimation and Certification in Conjunctive Evaluations

[![Reproduction checks](https://github.com/Itsxuchen/RefEval/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Itsxuchen/RefEval/actions/workflows/reproduce.yml)

Research code, compact analysis data, and saved results for a retrospective study of reference checking in rubric-based evaluation. A reference check can help estimate judge accuracy, certify an individual outcome, or repair an existing verdict. When every retained criterion must pass, these goals can favor different checking and updating protocols.

**Manuscript in preparation; no arXiv identifier has been assigned.**

The study uses fixed released references and saved predictions from **RuVerBench** and **JudgmentBench**. JudgmentBench's strict all-pass outcome is a researcher-defined stress test. The nine analysis cells come from two data families; they are not nine independent replications. One query reveals one existing reference bit. Query counts do not measure human time.

## Research questions

Under explicit evaluation targets and information conditions, how do reference-query allocation and verdict updating affect estimation precision, certification coverage, and residual error in conjunctive evaluations?

1. **Error propagation and updating:** When do criterion-level errors and partial corrections change conjunctive task verdicts, and how does the verdict-update rule affect these changes?
2. **Allocation of limited references:** How do sampling units and query allocations trade off estimation precision, certification coverage, and residual verdict error under stated reference-query resources?
3. **Evidence sufficiency:** Which reference information suffices to certify individual verdicts or identify aggregate success, and how does identification differ from statistical estimation?

RQ2 is the main empirical analysis. RQ1 examines the consequences of partial replacement and delayed updates; RQ3 supplies the information conditions needed to interpret the other results. Short-circuit evaluation, Boolean certificates, and probability sampling are established tools. The contribution is their controlled empirical comparison and its conditional findings.

Prior work already studies component-level screening, sampling tradeoffs, Boolean certification, error unmasking and selective updating. This study focuses on their measured consequences when reference queries resolve components of conjunctive outcomes and verdict updating is specified separately. See the [related-work comparison and operational implications](docs/METHODS.md#related-work-and-operational-use) for the scope of the empirical contribution.

## Main results

At a **20% criterion-query cap**, the table compares random criterion sampling with a fixed allocation that spends up to half the cap on judge-first short circuit, then samples independently from the remaining criteria. All shown policies use the same actual number of queries within each cell.

| Reference frame / allocation | Actual queries | Certified units | Residual verdict errors | Accuracy interval width (pp) |
|---|---:|---:|---:|---:|
| RuVerBench DR Gemini / random criterion SRS | 323 | 131 | 8 | 5.51 |
| RuVerBench DR Gemini / fixed 50/50 allocation | 323 | 169 | 9 | 7.62 |
| JudgmentBench GPT-5.4 / random criterion SRS | 4,697 | 798 | 223 | 2.21 |
| JudgmentBench GPT-5.4 / fixed 50/50 allocation | 4,697 | 1,035 | 133 | 2.95 |

Entries after query count are **marginal medians over seeds 1–31**, with release task order; they need not describe a single realized run. Accuracy is criterion-level microagreement with the fixed reference. Its intervals follow a hypergeometric-tail construction with conditional, pointwise 95% coverage under the stated sampling design; the [known floating-point boundary limitation](docs/REPRODUCIBILITY.md#numerical-boundary-caveat) is documented separately. [Exact result keys and additional comparisons](docs/RESULTS.md)

- **Allocation can improve task outcomes while reducing estimation precision.** In the JudgmentBench comparison above, the mixed allocation improves certification and residual errors in all 31 paired orders, but produces wider accuracy intervals. In DR it increases certification while leaving more residual errors in 18 of 31 orders. The fixed split is a comparison condition, not an optimal policy.
- **More criterion repairs need not produce fewer task errors.** In one JudgmentBench disagreement-only endpoint, 4,316 queries repair 1,217 criterion errors, but task errors change from 244 to 245: two false passes and two false fails are repaired, while five false passes are introduced. On a separate matched set of 4,697 queries (canonical seed 0, release task order), certificate-gated updating prevents six new false passes but delays nine repairs, changing residual errors from 239 to 242.
- **The contrary cases matter.** The criterion-error-discovery versus certification reversal survives all 31 randomized tie orders for DR Gemini and JudgmentBench mini, but none for the stronger JudgmentBench judge. Directly using the stronger primary also outperforms the weak-primary-plus-strong-auxiliary short-circuit policy on both task metrics in the tested release-order comparison. These results do not establish universal superiority of either a hybrid or a single judge.

![Policy outcomes over the first 40% of available criterion budget](artifacts/figures/conjunction_policy_study/policy_frontiers_early.png)

The figure shows four selected policies from the seven-policy study. Lines are medians and bands are 5–95% ranges across randomized orders, **not confidence intervals**. The horizontal axis is available budget; actual queries stop when a policy exhausts its candidates or finishes certification. The fixed 50/50 allocation is a separate extension summarized in the table above.

## Full-budget verdict-update effects

![Matched-query eager versus certificate-gated update effects](artifacts/figures/conjunction_policy_study/verdict_update_budget.png)

On the **same queried set**, gating prevents newly introduced false passes but can delay false-fail repairs. For disagreement-then-random at 20% cap, release task order, the mean gated-minus-eager error difference over seeds 1–31 is **−2.065** for full strict (4,697 queries) and **+8.290** for binary-only (4,046 queries). The canonical full-target example above has the opposite sign from its randomized mean. Effects change with budget; neither endpoint establishes universal superiority.

Lines are arithmetic means of within-seed differences; bands are 5th–95th order percentiles, **not confidence intervals**. The short-circuit column uses an explicitly expanded vertical scale. Changing scoring target also changes the criterion universe, certificate lengths and absolute budget, so it does not isolate a penalty-type effect. [Numerical details](docs/RESULTS.md#full-budget-matched-query-effects) · [Main PDF](artifacts/figures/conjunction_policy_study/verdict_update_budget.pdf) · [Nine-cell/order supplement](artifacts/figures/conjunction_policy_study/verdict_update_budget_all_cells.pdf) · [All-seven-policy plotting data](artifacts/figures/conjunction_policy_study/verdict_update_budget_data.csv).

## Get started

From the repository root, inside a Python environment:

```bash
python -m pip install -r requirements.lock.txt
python -m src.check_package
python -m pytest
python -m src.reproduce --mode smoke --output artifacts/smoke
```

For the complete reproduction and comparison against the included results:

```bash
python -m src.reproduce --mode full --output artifacts/reproduced
python -m src.verify_release --results artifacts/reproduced
```

Historical **v0.1.0** validation: **130 tests passed; complete offline replay in 169 seconds** in the recorded environment. The CI badge reports checks for current main; the dated release receipt is not a test count for later commits. [Execution receipts](docs/REPRODUCIBILITY.md#release-010-verification)

The scientific replay uses local inputs and requires no model API key. Smoke mode is a bounded execution check; use full mode for the complete declared analysis. Commands write to the supplied output directory, while archived numerical targets remain under `artifacts/expected/`. See [reproduction scope, output files, and validation](docs/REPRODUCIBILITY.md).

## Read the study

| Document | Contents |
|---|---|
| [Methods](docs/METHODS.md) | Targets, policies, estimators, information conditions, nearest-neighbor comparison and operational implications |
| [Reading bibliography](docs/references.bib) | 46 technical and data sources; inclusion does not imply every source is used by the implementation or an exhaustive search |
| [Results](docs/RESULTS.md) | Numerical anchors, full-grid coverage, counterexamples, and source keys |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Execution modes, expected results, and the limits of each check |
| [Data sources](docs/DATA_SOURCES.md) | Upstream provenance, included fields, transformations, and source terms |
| [Policy protocol](context/conjunction_policy_protocol.md) | Policy, budget, ordering, and sensitivity contract |
| [Label-value protocol](context/conjunction_label_value_protocol.md) | Fixed allocation, exact inference, and whole-unit sampling contract |

The information-disclosure analysis retains complete references while hiding and restoring their task links. It adds no labels. Its finite feasible ranges are distinct from sampling confidence intervals. An optional JudgmentBench pairing diagnostic holds both arm means fixed and studies covariance and standard-error uncertainty; it is not evidence of a new system ranking. [Details](docs/METHODS.md#information-sufficiency-and-pairing)

## Data, software, and citation

The included analysis inputs contain labels, scores, scoring metadata, and identifiers needed for replay. Source task texts, full generated responses, free-text comments, and recorded human-time fields are excluded. The package reproduces analysis conditional on the released references; it does not independently validate their semantic correctness.

Code licensing is specified in [LICENSE](LICENSE). Dataset-derived files retain the terms identified in [DATA_SOURCES](docs/DATA_SOURCES.md) and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md); the software license does not relicense upstream material.

Use [CITATION.cff](CITATION.cff) to cite this software and results package, and cite the upstream datasets when using their derived records. Until a manuscript identifier is assigned, describe the package as research software rather than citing a nonexistent preprint or publication.
