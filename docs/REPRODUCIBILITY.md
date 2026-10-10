# Reproducibility

## What the package reproduces

The package contains compact, source-derived labels and identity/scoring metadata, offline analysis code, tests, figures, and archived numerical results. It supports replay of query allocation and verdict updating conditional on those references. It does not rerun the original judge models or independently re-evaluate the source task texts.

The included inputs exclude source task texts, full generated answers, free-text comments, and recorded human-time fields. [DATA_SOURCES](DATA_SOURCES.md) documents the included projections and provenance; [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) describes source terms.

## Install and test

Use CPython 3.12. Create and activate a virtual environment, then run commands from the repository root:

```bash
python -m pip install -r requirements.lock.txt
python -m src.check_package
python -m pytest
```

Tests cover input identities and labels, query budgets and stopping, non-anticipating selection, certification, update transitions, small-world calculations, sampling estimators, and information-disclosure bounds. A test result describes the checks actually executed by that invocation; it should not be substituted for a full independent replication of every saved row.

## Bounded smoke execution

```bash
python -m src.reproduce --mode smoke --output artifacts/smoke
```

Smoke mode provides a bounded check that the packaged inputs and analysis entry points execute together. Its reduced execution is not the complete policy grid or the full stability analysis. Inspect its reported scope before using its outputs as evidence for a particular result.

## Complete execution

```bash
python -m src.reproduce --mode full --output artifacts/reproduced
python -m src.verify_release --results artifacts/reproduced
```

The reproduction command writes new outputs beneath the supplied directory. The archived targets remain in `artifacts/expected/`. Use the supplied output directory consistently when verifying a run. A results check compares reproduced scientific quantities with the archived targets; timestamps, elapsed time, and compressed-file headers are not scientific outcomes.

These commands use local analysis inputs and require no model API credentials. Installing Python packages may require network access. The historical policy study took approximately 107 seconds, and the label-value extension approximately 26 seconds in the original environment. Complete reproduction includes additional processing, so these figures are historical component timings, not a runtime promise for another machine.

## Rebuild the full-budget update figures

Full mode also renders the matched-query update figures from its regenerated
budget rows and compares every plotting-data row with the distributed CSV.
Smoke mode does not render this 31-random-order figure. To rebuild only the
figures from frozen rows, without rerunning query acquisition:

```bash
python -m src.plot_verdict_update_budget \
  --source artifacts/expected/conjunction_policy_study/budget_curves.csv.gz \
  --output artifacts/reproduced/figures/conjunction_policy_study
```

The renderer protects source files, inputs, frozen numerical targets and published
figures from output overlap. It writes a main PNG/PDF, a nine-page PDF supplement,
an all-seven-policy summary CSV and a source/artifact hash receipt. The main figure
selects two JB targets and three policies; the supplement selects nine cells and
three task orders for those policies. The CSV retains all 19,089 randomized groups.
Means are paired within seed, exclude canonical seed 0, and use seeds 1–31.
Order-percentile bands are not confidence intervals. Recreated PDF bytes can vary
with metadata or renderer versions; scientific summary values are compared with
declared numerical tolerances instead of requiring identical image bytes.

The public receipt records repository-relative source locations when available,
otherwise a basename, plus hashes and output filenames. Historical local-workspace
paths and the old renderer hash are not treated as current public-package provenance.

## Release 0.1.0 verification

The isolated CPython 3.12.12 environment on macOS arm64 completed the full
reproduction in **168.854 seconds** on 2026-09-15. The integrated suite passed
**130 tests**. This is a measured reference run, not a runtime guarantee.

The [execution receipt](../artifacts/validation/full_run_manifest.json) and
[numerical comparison](../artifacts/validation/full_verification.json) record
all-row comparisons for 610,848 policy-budget records, 1,968,288 deletion records,
11,718 allocation records, 837 whole-unit samples and their summaries. The
disclosure check includes 7,192 endpoint witnesses. Numerical comparisons retain
counts exactly and use declared floating tolerances; they do not equate runtime
metadata or nonunique optimal assignments. Independent arithmetic checks reuse
the selector, with their scope recorded separately.

[Source restoration](../artifacts/validation/source_restoration.json) verified
eight pinned-source JudgmentBench projections and two copied saved prediction
tables. [Input projection parity](../artifacts/validation/input_projection_parity.json)
checked all nine cell arrays and identities against the original study inputs.
[Environment and metadata checks](../artifacts/validation/environment_and_checks.json)
record the software versions and citation-schema validation.

## Archived result groups

| Location under `artifacts/expected/` | Scientific contents |
|---|---|
| `conjunction_policy_study/study.json` | Dataset frames, full-grid configuration, endpoints, canonical points, and original run provenance |
| `conjunction_policy_study/budget_curves.csv.gz` | 610,848 cell/policy/order/seed/budget rows, including actual queries and update-rule outcomes |
| `conjunction_policy_study/leave_one_base_task_out.csv.gz` | 1,968,288 canonical-order influence results with budgets recomputed after deletion |
| `conjunction_policy_study/analysis_summary.json` | Explicit contrast definitions, paired order counts, negative cases, and denominators |
| `conjunction_policy_study/error_flow.json` | Scoring-mode decompositions, susceptibility, and criterion-to-task error transitions |
| `conjunction_label_value/budget_estimation_rows.csv.gz` | 11,718 fixed-cap allocation rows with task outcomes and microaccuracy inference |
| `conjunction_label_value/budget_estimation_summary.csv` | 31-seed descriptive summaries, interval widths, RMSE, and actual query counts |
| `conjunction_label_value/whole_task_sampling.csv.gz` | 837 fixed-unit-count samples with variable criterion costs and task/accuracy estimates |
| `conjunction_label_value/manifest.json` | Original configuration and the executable two-task information-condition bridge |
| `information_disclosure_replay.json` | Reference-link disclosure results, reveal orders, integer endpoint witnesses, and the optional paired-score diagnostic |

Compact policy CSVs supply the exact rows behind the displayed tables. Scientific figures live in `artifacts/figures/`; their original plotting data and full-resolution files accompany the displayed PNGs when included.

## Reading the saved quantities correctly

- **Available cap versus actual queries:** compare `budget_cap` and `actual_queries`. A policy may exhaust disagreements or finish certification before using its cap.
- **Seeds:** policy seed 0 is the canonical tie order; randomized summaries use seeds 1–31. The label-value extension uses seeds 1–31. Repeated orders are not new independently sampled benchmarks.
- **Marginal medians:** separately summarized certification, error, and interval-width medians need not occur in the same run.
- **Microaccuracy inference:** `exact_conditional_SRS` marks hypergeometric-tail intervals under the sampling design. `logical_bounds_or_census` marks pure-SC ranges; their `accuracy_ci_*` column names do not make them confidence intervals.
- **Whole-unit samples:** `requested_task_share` determines a fixed number of units. Label cost varies; these rows are not equal-cost competitors to the fixed-cap allocation grid.
- **New errors:** endpoint introduced FP, maximum simultaneous introduced FP, and cumulative transient events have different meanings.
- **Disclosed information:** the reference-link replay already knows exact reference-containing margins. Its feasible ranges cannot be interpreted as the yield of acquiring new labels.
- **Pairing:** the optional diagnostic changes feasible covariance while holding marginal scores and their mean gap fixed.

## What the historical checks established

The original policy-study validation independently recomputed outcome arithmetic for **126 canonical real-data rows and 2,898 numerical fields**, while reusing the source loader and query selection. Small-world tests separately covered **84 label worlds, 3,528 trajectories, and 13,104 integer budget prefixes**. Those checks do not imply a second independent implementation of all 610,848 budget rows or all query policies.

The label-value validation reconciled **5,580 baseline rows in six fields** against the policy study, rechecked **837 whole-unit samples**, and independently scored **27 real mixed-allocation rows while reusing selection**. Exhaustive toy calculations support the stated estimator and interval properties. The 31 empirical sampling repetitions illustrate performance; they do not prove nominal coverage by themselves.

The disclosure implementation checks integer endpoint attainment, truth containment, nesting, and full recovery, including independently enumerable small label assignments. Pairing bounds are checked against small permutations. A stored feasible witness establishes attainability; optimality is supplied by the declared optimization calculation and its solver certificate, rather than by the witness alone.

These counts describe the historical scientific validation scope. Fresh package tests and reproduction checks have their own execution record. Source-file hashes from the original research run and hashes of the cleaned release serve different purposes: projection, relocation, or documentation changes can alter bytes while leaving declared numerical inputs and estimands unchanged. The package's provenance records distinguish those layers rather than assigning new dates to old test results.

### Numerical endpoint repair (0.2.0)

The shared hypergeometric implementation now includes exactly equal tail endpoints by recomputing threshold-near cases with integer/rational arithmetic. The diagnostic M=16, n=2, x=0 returns [0,13]. Exhaustive checks cover all M≤24 observation acceptance sets and finite-population coverage. A regression over 11,718 saved label-value and 837 saved task-pass rows changes none of their endpoints. The historical tables and release receipts remain intact; the corrected code is used for all new runs. [Regression receipt](../artifacts/expected/conjunction_robustness/estimation/interval_regression.json)

## Mechanism and robustness extension (0.2.0)

Run the original full command above **and** the following command for the complete published analysis:

```bash
python -m src.reproduce_extensions --output artifacts/reproduced-extensions
python -m src.reproduce_extensions --output artifacts/reproduced-extensions --verify-only
```

The extension runs the 64-permutation mechanism analysis, all 31-seed allocation decompositions, four matched-reference cells with crossed acquisition/scoring references, 399 whole-base-task composition resamples, and nine-cell budget-accounted estimation controls. It regenerates the mechanism and robustness figures. It then compares every extension scientific CSV row and four scientific JSON payloads against the versioned expected tables. Sources, protocols and output checks receive separate manifests. A new/empty output directory preserves earlier run receipts. The original `src.verify_release` command covers the original study; it does not claim to verify the extension.

The extension's interval tests and direct small-world arithmetic complement replay equality. Equality to saved tables alone cannot validate reference correctness, statistical transport or an untested procedure. Reference replacement keeps both judge draws fixed; its off-diagonal cells re-read queried labels as well as the endpoint truth. Task-composition sensitivity retains the observed raters. Normal approximate intervals are separate from guarantee-based comparisons; the old mix table has only its original conditional interval.


**0.2.0 validation:** 215 tests passed. Original complete replay: 169.8 seconds; mechanism/robustness replay: 304.9 seconds. All 25 extension scientific CSV tables (752,947 rows), four scientific JSON payloads, and 606 plotting rows matched. Timings describe this recorded environment. [Original-study receipt](../artifacts/validation/robustness_base_run_manifest.json) · [Extension receipt](../artifacts/validation/extension_run_manifest.json) · [Row comparisons](../artifacts/validation/extension_verification.json)

The 0.2.0 execution receipts are saved under `artifacts/validation/extension_*` and `robustness_*`. They describe that release's code and figures. Its complete [GitHub CI run](https://github.com/Itsxuchen/RefEval/actions/runs/37731851023) completed successfully at commit `7081eef`. Historical 0.1.0 counts and timings elsewhere on this page identify that release only.

### Presentation clarification after 0.2.0

The main budget figure now shows pointwise composition ranges, while the saved simultaneous-band values remain unchanged. The estimation figure displays both conservative interval widths and empirical RMSE in all nine cells. A/B descriptions use symmetric reference panels. These changes leave the tagged release and scientific outputs intact.

[Presentation verification](../artifacts/validation/presentation_verification.json) recomputes all 162 estimation summaries from the saved query-order rows, checks 1,674 matched-budget groups, and confirms that all 606 budget plotting rows and 64 data/result/protocol files match the 0.2.0 commit. All 215 tests passed again. The [current renderer manifest](../artifacts/figures/conjunction_robustness_render_manifest.json) identifies the revised figure inputs and outputs; the earlier full-run receipts retain their original hashes. This focused check does not claim a new full scientific replay.

## Scope of reproducibility

Reproducing these calculations checks a finite-reference analysis. It does not verify unobserved expert consensus, human labor efficiency, broad occupational validity, or a system-ranking decision. The package has a defined empirical scope; no new data collection is needed to execute it. Use [METHODS](METHODS.md) for the assumptions and [RESULTS](RESULTS.md) for the preserved contrary cases.

## Release metadata and paper linkage

`CITATION.cff` identifies the software version and repository maintainer, following
[GitHub's citation-file format](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files).
It does not invent a paper author list or manuscript identifier.
The tagged source archive, environment pins, scoped data notices and execution
receipts support a future paper's code-availability statement.

A public GitHub release is separate from submitting an article to arXiv.
[arXiv ancillary files](https://info.arxiv.org/help/ancillary_files.html) are tied to
an article version and can include supporting code and data. This repository is
not itself an arXiv submission. Venue-specific anonymous review packages, if
needed later, must follow that venue's instructions.


## Manuscript build and claim checks

The [complete paper](../artifacts/manuscript/paper.pdf) is an author draft with main text, tables, figures, references, mathematical derivations, and reproduction details. It has not been submitted to arXiv or assigned a paper identifier. The editable source is [paper.tex](../artifacts/manuscript/paper.tex), with [methods_appendix.tex](../artifacts/manuscript/methods_appendix.tex) and a [selected bibliography](../artifacts/manuscript/paper_references.bib). The broader [reading bibliography](references.bib) remains available.

Install [Tectonic](https://tectonic-typesetting.github.io/) (validated with version 0.17.0), then run from the repository root:

```bash
python -m src.build_paper
```

To select an executable or a separate output location:

```bash
python -m src.build_paper --tectonic /path/to/tectonic --output artifacts/paper-build
```

The compiler writes `paper.pdf` and diagnostic files in `artifacts/paper-build/`, leaving the versioned PDF and scientific files intact. Initial TeX package downloads require network access; cached subsequent builds use the installed bundle. The source uses standard LaTeX, `plainnat` and BibTeX and can also be built in a full TeX installation. PDF metadata may differ across builds; compare scientific values and rendered content, not PDF hashes alone.

Run the manuscript numerical audit with the pinned Python environment:

```bash
python -m src.verify_manuscript_claims --output artifacts/paper-checks/numerical_claim_audit.json
```

The [saved claim audit](../artifacts/manuscript/numerical_claim_audit.json) records source hashes, row selectors, units, and independently recomputed arithmetic and combinatorial expectations. It distinguishes analytical expectations, paired order means, marginal medians, composition ranges, and matched-reference subsets. The [citation audit](../artifacts/manuscript/citation_audit.md) records the primary sources and the claims they support. These checks complement the full scientific replay; they do not adjudicate reference correctness.
