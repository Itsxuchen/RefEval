# Conjunction policy study: frozen analysis contract

Public edition of the 2026-09-13 retrospective, fixed-reference analysis contract.
This contract is written before the new full-grid runs. RuVerBench/JudgmentBench
and the earlier20%examples have already been explored; this is not preregistration
or unseen validation. Editorial changes remove internal planning context; scoring and experimental
settings are unchanged.

## Targets and population

- Seven original run cells: four RuVerBench runs, new GPT-5.4 DR and two JB judges.
- Two additional JB binary-only cells explicitly change the strict-full-credit
  target by dropping occurrence-count penalties. Never pool them as unchanged-target
  interventions. Nine cells are two families, not nine independent replications.
- One reference query reveals one criterion bit perfectly. No reference validity,
  annotation minutes, model inference cost or solver-comparison claim is tested.
- JB records are output-by-human-rater annotations, clustered in30base tasks;
  RuVerBench base tasks are their retained task IDs. All joins/metadata must pass.

## Frozen policies and ordering

1. random_criterion: uniformly shuffled global criterion order; no certification stop.
2. sc_natural: contiguous tasks; published criterion order; stop task at observed FAIL.
3. sc_random: contiguous tasks; shuffled criteria; same stop.
4. sc_judge: contiguous tasks; predicted FAIL first; same stop.
5. disagreement_only: all flagged criteria, in shared task/tie order; stop when flags exhausted.
6. disagreement_then_random: same complete global flagged phase, then remaining
   criteria in global random order; no within-task short circuit.
7. sc_disagreement: within each task flags first, then remaining predicted FAIL,
   then predicted PASS; short circuit. This is different from global fallback.

Three task orders: release ID order, predicted FAIL count ascending (ID tie at seed0),
and random task order. Global random-criterion ignores this factor deliberately.
Seed0 preserves published ties for structured policies. Seeds1..31 randomize ties
using shared fixed priorities; sc_random/global random are randomized at every seed.
Within a dataset and seed, all policies receive the same task/criterion priorities.
No learned allocation. Selection sees predictions, disagreement, metadata and already
queried references only. Full reference data are evaluation-only.

## Budget and outcomes

Use every integer share0..100%, rounded to a criterion count. All policies have the
same available cap within a cell. Report actual queries separately after exhaustion
or full certification; call a comparison equal-actual only if both consumed the cap.
Disagreement exhaustion is an explicit plateau, with the fallback arm as its control.
Secondaries: two extra complete runs for DR, one for AC/JB; frozen existing predictions,
not additional paid calls. Counts of extra judges are separate from reference queries.

Three objectives: criterion errors discovered (maximize), certified tasks (maximize),
and backfilled task errors FP+FF (minimize). Certification requires an observed FAIL or
all criteria observed PASS. Uncertified does not mean wrong. Residual error is unknown
to a deployed policy and must not be used to choose online actions.
Track new false passes, corrected initial FP/FF and witnesses. All replacements are
reference-correct; newly introduced FF is impossible. Identify initially correct FAIL
tasks with no correctly predicted FAIL as susceptible to transient unmasking.

Design-review addition before full runs: evaluate a certificate-gated output contract
on the identical queries (keep the initial task verdict until certified, then update).
This costs no extra queries and cannot introduce errors under perfect references;
it can delay useful corrections of initial false FAIL. It is a reporting/commit-rule
sensitivity, not an eighth selection policy or a claimed new algorithm. Report the
remaining errors and delayed corrections rather than describing gating as free gain.

Report complete grids and raw per-seed counts; order quantiles are not confidence
intervals. Use full-certification query counts and shared-cap tradeoff tables; do not
rank incomparable criteria by an undeclared combined utility.

## Stability and scope

-32fixed seeds including canonical seed0; all judges/targets/task orders retained.
- Leave-one-base-task-out on canonical release and predicted-failure-count orders,
  every policy and cap. Remove entire JB base-task blocks, not single annotations.
  Hold priorities fixed. Filtering task-separable query transcripts is exactly the
  corresponding deletion replay; verify against literal reruns in tests. Recompute
  the cap and denominators after deletion. This is sensitivity, not held-out learning.
- Inspect criterion-type transitions and compare full vs binary-only targets.
- Preserve null/reversed/dominance results and distinguish paired metric conflicts
  at equal actual queries from shared-cap comparisons after early stopping.

## Validation and release

Before full runs: strict source checks, unit/invariant tests and measured small-run
timing with a quantified ETA. Independent tests enumerate small AND worlds, compare
prefix metrics to a naive evaluator, check non-anticipating selection/caps/exhaustion
and verify deletion replay. Independently reconstruct headline real-data rows.
Record input/source/contract hashes and run time. No original raw changes, API calls,
human review, GitHub push or paper submission. Produce a report even if anticipated
tradeoffs do not survive controls. Conference readiness is an evidence-based judgment,
not inferred from the number of tests, curves or datasets.
