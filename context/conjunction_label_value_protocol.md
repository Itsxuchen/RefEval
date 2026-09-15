# Label value extension — 2026-09-13

Public edition of the September 13, 2026 analysis contract. The extension was
designed after exploratory analyses and the completed seven-policy study; it is
retrospective, not a preregistration. Editorial changes remove internal planning
context; scoring, resource definitions and experimental settings are unchanged.

1. Uniform criterion SRS is a generic probability-sampling baseline. The 0.006
   budget fraction is an illustrative design setting, not an estimate of a
   particular benchmark provider's sampling practice.
2. Preserve nine existing cells, strict loader and fixed reference/target scope.
   Compare randomcriterion, pureSCjudge, and a frozen50/50SCthenfreshSRS control.
   Three taskorders, seeds1..31, caps0.6%,5%,20%,50%,60%,100%; existing full-grid
   policy/LOTO evidence is reused rather than presented as newly executed here.
3. First stage spends up to floor(B/2) actual SC queries. All leftover budget goes
   to an independent SRS without replacement from the unqueried criterion pool.
   No split tuning. Extra selected references are perfect answers in a replay.
   Each cap defines its own two-stage design; hybrid sets need not be nested across
   caps. It is not an anytime adaptive budget trajectory.
4. Report queried-set task certification, eager-backfill residual task errors,
   and logical pass-rate bounds. Width equals uncertified share and adds no new
   independent outcome. Uncertified does not mean wrong or statistically useless.
5. Accuracy means criterion-micro agreement, not balanced accuracy. SRS/hybrid
   estimate = (known correct + remaining population size × remainder sample mean)
   / total criterion count. Invert exact hypergeometric tails for a conditional95%
   pointwise interval; no confidence-sequence or reference-validity claim. Report
   bias/RMSE/coverage counts across seeds, not median proximity as proof of accuracy.
   Naive adaptive sample agreement is descriptive and generally selection-biased.
6. Whole-verdict-unit SRS control selects a fixed number of tasks before any labels,
   then compares complete-rubric versus short-circuit cost for the identical sample.
   Task pass sample mean and HT micro agreement (full rubric) are design-unbiased;
   realized label cost varies. It is not a same-cap comparison or a replication of a provider's audit.
7. F27 shares feasible-completion logic but includes exact population margins and
   restores existing label links. The new query design acquires labels without those
   margins. Connect via an executable small example; do not relabel old F27 curves.
8. Validate exact intervals/conditional unbiasedness and logical bounds by small
   exhaustive worlds, compare new queried-set arithmetic to existing results, record
   source/data hashes and measured preflight/full-run time. Preserve original sources.
