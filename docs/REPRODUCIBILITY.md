# Reproducibility

## Current release: v0.3.0

The [v0.3.0 snapshot](https://github.com/Itsxuchen/RefEval/tree/v0.3.0) aligns the manuscript, portable sources, citation metadata and complete numerical package. It includes the task-order table, Appendix B.4 fixed-verdict example, and Appendix E.3 coding sensitivity. No arXiv identifier or Zenodo DOI has been assigned.

The original policy/disclosure analysis and the complete extension analysis were both rerun for this release. The extension comparison covers **29 CSV tables / 998,652 rows and five scientific JSON payloads**. Release receipts: [original run](../artifacts/validation/v030_original_run_manifest.json), [original comparisons](../artifacts/validation/v030_original_verification.json), [extension run](../artifacts/validation/v030_extension_run_manifest.json), and [extension comparisons](../artifacts/validation/v030_extension_verification.json). Tests, manuscript claims and PDF/source checks are recorded in [build validation](../artifacts/manuscript/build_validation.json). Earlier validation counts below are dated historical records.

The five author-run RuVerBench configurations, prompts, context truncation and recovery rules are documented in [DATA_SOURCES](DATA_SOURCES.md#generation-protocol-for-the-five-study-runs). The offline prompt module reconstructs inputs and parses responses; it does not call providers. Publication renderers now reset their own style within a scoped context. The regression check renders both call orders under different ambient colors and compares all five PNGs pixel for pixel, while also checking restoration after errors.

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

The extension runs the 64-permutation mechanism analysis, all 31-seed allocation decompositions, four matched-reference cells with crossed acquisition/scoring references, 399 whole-base-task composition resamples, and nine-cell budget-accounted estimation controls. It regenerates the mechanism and robustness figures. It then compares every extension scientific CSV row and five scientific JSON payloads (four in the original 0.2.0 extension) against the versioned expected tables. Sources, protocols and output checks receive separate manifests. A new/empty output directory preserves earlier run receipts. The original `src.verify_release` command covers the original study; it does not claim to verify the extension.

The extension's interval tests and direct small-world arithmetic complement replay equality. Equality to saved tables alone cannot validate reference correctness, statistical transport or an untested procedure. Reference replacement keeps both judge draws fixed; its off-diagonal cells re-read queried labels as well as the endpoint truth. Task-composition sensitivity retains the observed raters. Normal approximate intervals are separate from guarantee-based comparisons; the old mix table has only its original conditional interval.


**0.2.0 validation:** 215 tests passed. Original complete replay: 169.8 seconds; mechanism/robustness replay: 304.9 seconds. All 25 extension scientific CSV tables (752,947 rows), four scientific JSON payloads, and 606 plotting rows matched. Timings describe this recorded environment. [Original-study receipt](../artifacts/validation/robustness_base_run_manifest.json) · [Extension receipt](../artifacts/validation/extension_run_manifest.json) · [Row comparisons](../artifacts/validation/extension_verification.json)

The 0.2.0 execution receipts are saved under `artifacts/validation/extension_*` and `robustness_*`. They describe that release's code and figures. Its complete [GitHub CI run](https://github.com/Itsxuchen/RefEval/actions/runs/37731851023) completed successfully at commit `7081eef`. Historical 0.1.0 counts and timings elsewhere on this page identify that release only.

### Presentation clarification after 0.2.0

The main budget figure now shows pointwise composition ranges, while the saved simultaneous-band values remain unchanged. That presentation used conservative interval widths and empirical RMSE in all nine cells; the 2026-10-11 design-risk correction below supersedes its precision comparison. A/B descriptions use symmetric reference panels. These changes leave the tagged release and scientific outputs intact.

[Presentation verification](../artifacts/validation/presentation_verification.json) recomputes all 162 estimation summaries from the saved query-order rows, checks 1,674 matched-budget groups, and confirms that all 606 budget plotting rows and 64 data/result/protocol files match the 0.2.0 commit. All 215 tests passed again. The [updated renderer manifest](../artifacts/figures/conjunction_robustness_render_manifest.json) identifies the revised figure inputs and outputs; the earlier full-run receipts retain their original hashes. This focused check does not claim a new full scientific replay.

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


## Design RMSE correction at the 20% cap

```bash
python -m src.conjunction_design_rmse --output artifacts/design-risk-replay
```

This command recomputes exact SRS and proportional-stratum risk and integrates all pilot success-count combinations for the charged-pilot design. It also regenerates 31 release-order first-stage mix transcripts, analytically integrates their fresh SRS variance, and reports the remaining Monte Carlo SE. Outputs are `design_rmse.csv` (36 rows) and `mix_conditional_variances.csv` (279 rows). The full extension command also runs and verifies this stage. The published main precision comparison covers the 20% cap; the original six-cap simulation tables remain unchanged as historical diagnostics.

Small-population tests independently enumerate pilot identities and subsequent samples to check conditional unbiasedness and unconditional risk. The manuscript audit recomputes the complete risk tables from the inputs, reusing the allocation implementation. Input references, interval constructions, certification and repair outputs are unchanged by this correction.

**2026-10-11 verification:** 223 tests and 547 manuscript checks passed. The complete extension replay matched all 27 scientific CSV tables (753,262 rows) and four scientific JSON payloads in 299.8 seconds. [Current correction receipt](../artifacts/validation/design_rmse_replay.json). Original 0.2.0 receipts above remain historical.

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

The [saved claim audit](../artifacts/manuscript/numerical_claim_audit.json) records source hashes, row selectors, units, and independently recomputed arithmetic and combinatorial expectations. It distinguishes analytical expectations, paired order means, marginal medians, composition ranges, and matched-reference subsets. The [citation audit](../artifacts/manuscript/citation_audit.md) records the primary sources and the claims they support. The audit also reconstructs the fixed-query delay split by peer status, flag coverage by item type, and the exploratory repeated-rating item check. It compares every one of the 90 printed amplitude rows with the underlying configurations and stores full-precision rows in the receipt. These checks complement the full scientific replay; they do not adjudicate reference correctness.


### Portable manuscript sources

[arxiv_source.zip](../artifacts/manuscript/arxiv_source.zip) contains only the main TeX file, methods appendix, selected BibTeX bibliography and three cited PDF figures. Figure paths are adjusted for compilation from the extracted directory; the scientific text is unchanged. Extract into an empty directory and run Tectonic on paper.tex (or use a compatible full LaTeX installation). This archive is preparation for author review, not a submission or arXiv compilation receipt.

The numerical audit additionally verifies all 54 task-order table entries against raw trajectories, the AC 20% comparison, delay by rejected item type, the 50% composition ranges and the 792-query fixed-verdict construction. It distinguishes that construction from the empirical rearrangements, whose task PASS totals vary. Current checks are recorded in [build_validation.json](../artifacts/manuscript/build_validation.json). Frozen data and expected replay outputs remain unchanged.

The portable archive can be rebuilt with the same compiler invocation:

```bash
python -m src.build_paper --source-archive artifacts/manuscript/arxiv_source.zip
```

The archive places the main TeX file, appendix, selected bibliography, and the three referenced figures at paths that resolve from its root. It excludes compilation logs, the output PDF, and unused figures.

### Coding sensitivity supplement

The new supplement is saved separately in `artifacts/expected/conjunction_coding_sensitivity/`: `multiplicities.csv.gz`, `rq1_curves.csv.gz`, `summary.json`, and `manifest.json`. The protocol was extended before these resampling outcomes were computed. Both coding judges use the same 399 task-composition draws; both policies and all 101 budget shares are retained.

The standard `src.reproduce_extensions` command now includes this supplement. A separate replay is also available:

```bash
python -m src.conjunction_cluster_sensitivity --coding-only --output artifacts/coding-replayed
python -m src.conjunction_cluster_sensitivity --coding-only --output artifacts/coding-replayed --verify-against artifacts/expected/conjunction_coding_sensitivity
```

The historical rc.4 extension replay reproduced **29 CSV tables / 998,652 rows and five scientific JSON payloads**. See [run receipt](../artifacts/validation/rc4_extension_run_manifest.json) and [rowwise comparison](../artifacts/validation/rc4_extension_verification.json). That revision passed 233 tests and 607 claim checks using 35 source files. All 62 data/expected files that existed at rc.3 remain byte-identical; four new supplement files are added. The original policy/disclosure replay was not repeated for rc.4; v0.3.0 reruns both analyses as recorded above.

The portable source archive contains the two TeX files, the selected `.bib`, and three PDF figures. [arXiv's TeX instructions](https://info.arxiv.org/help/submit_tex.html) support `.bib` processing; [its compiler documentation](https://info.arxiv.org/help/faq/texlive.html) lists XeLaTeX, supported since November 2025. The local Tectonic build and an independent extracted-source build are checked; the eventual arXiv preview is a separate check. The rc.4 abstract had 1,488 plain-text characters, below the [1,920-character limit](https://info.arxiv.org/help/prep.html). Author affiliation was explicitly confirmed, and the fixed title-page date was removed.

The paper identifies package version 0.3.0 and links to its tag. The release commit identifies the exact snapshot; a future arXiv identifier or archive DOI can be added to citation metadata when obtained. The author-selected title is **Beyond Criterion Accuracy: The Limits of Reference Checking in Conjunctive LLM Evaluation**. Section 7 completes the full-JB timing explanation before introducing the contrasting coding timing. All release-level checks and their scope are recorded above and in the build receipt.

## WildIFEval preflight (2026-10-15; stopped before pilot)

[The masked audit](WILDIFEVAL.md) is separate from the v0.3.0 experiments. The planned frame fails the predeclared association/completeness gates, so no external pilot or held-out results exist. CI validates the included compressed metadata offline. Optional source-level reproduction downloads 28 pinned files, verifies their hashes and repeats both independent joins; the exact commands and scope are in that document.

The [expanded feasibility comparison](EXTERNAL_VALIDATION_OPTIONS.md) adds all saved grading-channel combinations, a masked ProfBench inventory and offline prompt-cost scenarios. Its independent source check verifies 39 hashes, six channels and 50 combinations. These are input checks and planning estimates, not external mechanism results. The optional `external` dependency installs pinned Parquet and tokenizer readers; no API client or credential is needed for these checks. The new modules have synthetic tests, and saved reports are covered by the package integrity manifest.
