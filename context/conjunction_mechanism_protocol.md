# Retrospective mechanism study — 2026-10-07

## Purpose and status

The user authorized beginning the mechanism work proposed after the writing/external-validity review. This document specifies the new analysis before its new outputs are generated. The released frames, original results and their qualitative reversals have already been examined extensively. This is a retrospective, hypothesis-directed extension, not a preregistration, unseen validation set, or a new deployment policy.

Questions: (1) which within-task error configurations and query priorities account for prevented false passes versus delayed false-fail repairs, and (2) whether allocation-induced extra certificates concern initially correct or initially wrong verdicts. Retain the existing RQ1–RQ3 study. No new API calls, human labels, reference adjudication, dataset acquisition or changes to AfterQuery work.

## Inputs and units

Use the strict existing loader in src/conjunction_policy_data.py and the existing frozen policy/label-value sampling implementations. Nine analysis cells arise from two data families; changed targets and judges sharing tasks are dependent. JudgmentBench units remain output-by-rater annotations, not independent base tasks. Preserve all raw data, original scientific sources, saved outputs, manifests and the public checkout.

Sources: artifacts/reports/conjunction_policy_study/budget_curves.csv.gz, artifacts/reports/conjunction_policy_study/error_flow.json, artifacts/reports/conjunction_label_value/budget_estimation_rows.csv.gz and their existing protocols. Hash inputs and analysis source for the new output manifest. Full references may be used by a retrospective evaluator, never as uncharged query-selection inputs.

## A. Finite-budget update mechanism

Let A_b count initially correct FAIL units that eager updating turns into false PASS at budget b. Let D_b count initially false FAIL units already repaired by eager updating but not yet certified. Then the paired net error difference, gated minus eager, is D_b − A_b. This exact decomposition is a check/definition, not the empirical hypothesis or a new theoretical contribution.

For an initially correct FAIL to be susceptible, every initial predicted failure must be reference-PASS, and at least one reference-FAIL must have been predicted PASS. An introduced false PASS requires all initial predicted failures queried and every true failure still unqueried. A delayed false-fail repair requires all initial predicted failures queried but some retained condition still unqueried. Existing error-flow analysis already supplies the susceptible condition and an ever-unmasking probability; the new object is the finite-budget endpoint expectation under specified query priorities.

For a prefix of uniformly permuted groups, all groups before the active group are observed, later groups unobserved, and h of G members of the active group observed uniformly. For r required-observed and f required-unobserved members of that group, compute

P = product((h-j)/(G-j), j=0..r-1) × product((G-h-j)/(G-r-j), j=0..f-1),

with probability zero for impossible counts, required elements in future groups or forbidden elements in completed groups. Empty events have probability one. Summing the relevant per-unit probabilities yields E[A_b] and E[D_b], without fitting to observed randomized trajectories.

Primary scope: all nine cells, release task order, random_criterion and disagreement_then_random, all 101 existing budget shares. Random criterion selection is one group. Disagreement selection has one flagged group per task in release order, followed by one global remaining group; the flagged tier is not globally randomized. Match rounded absolute caps to the existing implementation. Seeds1–31 describe random within-group orders; seed0 is a separate canonical order.

Compare analytical expectations against saved means and descriptive Monte Carlo standard errors. Agreement in expectation is a mathematical implication to check by enumeration and implementation, not independent scientific validation. Do not require every finite Monte Carlo mean to equal its expectation or use many pointwise deviations as independent tests.

### Matched-marginal counterfactual

Generate 64 fixed permutations with root seed20261007 for each cell. Shuffle reference labels within (task criterion count k, primary prediction, category/scoring mode), preserving the original task slots, length vector, predictions, auxiliary predictions and each stratum's reference/prediction confusion counts. Evaluate the same analytical model at caps .05, .20, .50, .84 and .95, for both primary policies. The .20 and .84 choices include previously observed focal/peak regions; their selection is retrospective.

Candidate sufficiency claim: these preserved criterion-level margins, lengths and observable priorities determine the sign/size of the expected update effect. A counterfactual with the same preserved quantities but a different sign disproves sign sufficiency in that frame; variation without a sign change disproves exact-size sufficiency but does not establish sign instability. Report zero variation and degenerate strata too. Report the full counterfactual range, median and order quantiles, without calling them confidence intervals or a random sample of real environments.

This shuffle changes task reference pass rates and the joint reference/prediction structure. It does not isolate an error-correlation effect while holding base pass rates/initial task errors fixed. All permutations reuse the same two source families. No external generalization or novel mechanism follows merely from a changed curve.

## B. Certification versus verdict repair

Reconstruct the original random_criterion and sc50_then_srs query sets at all nine cells × three task orders ×31seeds × six existing shares (.006,.05,.20,.50,.60,1). Reuse the release-order SRS baseline for all three mixed-policy orders; these are not three independent baselines. Match actual query counts and the original rounded caps, including early short-circuit stopping followed by fresh-remainder sampling.

For each unit, derive reference outcome, initial prediction, initial TP/TN/FP/FF class, observed PASS/FAIL certification, and eager/gated updated outcomes directly from labels and the query mask. Verify aggregate outputs against the saved label-value rows. Independently derive task certification/update counts; if an existing interval helper is reused, label that verification as a compatibility check, not independent interval validation.

Partition certified units into initially correct and initially wrong; additionally retain TP/TN/FP/FF. Compare both gross mixed-only and SRS-only certified sets and their signed difference. A greater total certificate count can coexist with loss of certificates on initially wrong units. For eager updating, record repairs of initially wrong verdicts separately from introduced false passes, and repairs of initial false FAILs that remain uncertified. Correct reference replacement cannot introduce false FAILs.

Candidate empirical explanations:

1. Extra certificates are predominantly on initially correct units. Evaluate this using the gross mixed-only certificate composition, not a ratio with a zero/negative net denominator. Report cases that fail the claim; report net components separately.
2. Losing certified initially wrong units or losing uncertified false-fail repairs can offset total certification gains. Quantify signed components in every paired contrast, not only favorable cells.
3. More certification consistently means less residual error. Count the actual contrary paired outcomes and explain them using the direct transition components. This is a candidate relationship to test, not an assumed law.

Use paired differences before aggregating. Means support additive decompositions; the median of a difference need not be the difference of medians. Retain the original marginal summaries for compatibility, clearly labeled. Give detailed results at20%release and base-task contributions there to identify concentration; base-task aggregation is descriptive, not fresh independent inference.

## Information and operational boundary

Task lengths, primary/auxiliary predictions and a declared query order are available before new reference acquisition. Actual FP/FF status, true failure locations, susceptible membership and reference outcome are full-reference diagnostics. Certification status and queried reference bits are available after the corresponding queries. A predictor requiring the full-reference quantities is an oracle explanatory model, not a free operational selector.

This extension does not train a policy-selection model. Operational prediction would require separately specified features obtainable from a charged pilot, estimation uncertainty, a frozen decision rule/loss and validation on evidence not used to develop that rule. A later pilot would count all pilot and subsequent queries toward the same total resources; the current retrospective analysis cannot report such costs as measured or claim deployment gains.

## Validation and deliverables

New implementation files: src/conjunction_update_mechanism.py and src/conjunction_allocation_mechanism.py; corresponding focused tests in tests/. Keep implementation separate because original sources are hash-bound by prior receipts. New outputs go under artifacts/reports/conjunction_mechanism/; plot PNG/PDF files go under existing artifacts/figures/. Update the current writing kit/English scaffold and project bookkeeping after results are checked, without replacing old evidence.

Validation: exact enumeration of small grouped-query examples; direct per-unit arithmetic and transition identities; source-table alignment; preserved permutation strata; no unobserved-reference dependence in acquisition; boundary cases at zero/census; matched-query comparison. Use an independent reviewer for definitions and headline claims. Run targeted tests plus relevant existing conjunction tests. Report discrepancies and failed candidate explanations rather than adjusting the grid until the preferred story holds.

Stop after this finite analysis, figures and interpretation. Do not silently expand into a new benchmark, reference-noise study, broad model-fitting search, or public release. If the results do not explain policy ordering beyond the observed frames, record that as the remaining research gap.
