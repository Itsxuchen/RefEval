# When an expert revises a judge's verdict

Does another expert agree when an auditor changes a judge's verdict? On 206 legal outputs, retaining the judge produced 14.16 disagreements per 100 outputs with a peer. Adopting the complete audit produced 15.37. On 1,120 news summaries, adopting the full audit reduced peer disagreements from 37.08 to 30.83 for UniEval and from 37.44 to 30.83 for G-Eval. All comparisons give equal weight to each output. The paired 95% intervals for the changes were [−3.39, +4.90], [−9.32, −3.04] and [−9.37, −4.05], respectively. [Numerical exhibits](exhibits.pdf)

The half-budget replay follows PASS decisions issued before all conditions have been checked. In law, peers endorsed 39.8% of early approvals and 43.3% of complete approvals of initially rejected outputs. The difference was −3.54 percentage points [−9.73, +6.31]. Following the **same early cases** until the audit finished gave a different comparison: early approval added 0.79 peer disagreements per 100 **all** legal outputs [+0.02, +1.62]. The tables report both comparisons for each judge. SummEval uses final expert ratings recorded after the raters could see one another's scores.

## Reproduce

With Python 3.12, install the packages in requirements.txt and run the commands below. Tectonic is optional for numerical reproduction. Its first run can download LaTeX packages.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
./reproduce.sh
./reproduce.sh --pdf
```

The build writes TeX tables, vector figures and numerical controls to `generated/`. The PDF option compiles the standalone exhibits and, when the working-paper source is present, the manuscript. check_estimates.py independently rebuilds the legal cross-rater intervals from clustered counts and checks the RMSE, hypergeometric coverage and expected interval width for 432 hard-budget sampling designs. The compact inputs preserve the numerical observations needed by these computations. A fresh call to a hosted model can return a different answer.

| Input | Starting point for regeneration |
|---|---|
| [full_audit_cluster_counts.csv](inputs/full_audit_cluster_counts.csv) | Counts of peer verdicts, complete revisions and early approvals, averaged over the role rotations and query orders within each output. One row per legal task or news article per judge. |
| cross_scorer_article_counts.csv.gz | Article-by-auditor-by-scorer counts of early approvals, endorsements and errors across 31 priority orders, for both judges, 11 budgets and both overlapping summary frames. Article numbers preserve lexical order across the full set of 100 articles. |
| cross_scorer_legal_clusters.csv | Base-task contributions for the 206 legal outputs, averaged over all available auditor/scorer roles within each output and over the same 31 orders. |
| paired_bits.json | Per-source criteria, checker variants and three grading occasions for the fixed 62-response Multi-IF panel. Contains the ancestral four-model eligibility mask. |
| rater_orders.csv.gz and rater_groups.csv | Counts at each acquisition order and budget, and group-wise counts at the legal panel's half-budget point. The group file reproduces the 1,999 paired task reweightings. |
| cost_orders.csv.gz and pass_design.csv | 40,176 order/cost records and finite-population stratum counts for computing precision and coverage. |
| fold_forecasts.csv and forecast_validation.csv | Fold-specific development scores, held-out error counts and directional-rate forecasts on four historical frames. Ties use the recorded candidate order. |

## Cross-scoring the same decisions

The [locally frozen protocol](inputs/cross_scorer_protocol.md) fixes the 50% budget, initial judges, scoring rules, priority orders and statistics. The primary summary analysis uses 1,120 summaries of 70 articles and UniEval. The G-Eval judge swap and overlapping 1,600-summary frame were specified as sensitivities. Each of the three experts supplies the checked labels in turn. Each updated verdict is compared with that expert and separately with each of the other two. On the legal frame the same calculation rotates all distinct attorney pairs. Each output receives equal weight.

The complete-audit and timing comparisons were added after inspection of the original half-budget results. For a complete audit, peer endorsement counts a peer PASS on an upward change and a peer FAIL on a downward change. At half budget, early approvals are the cases on which immediate updating gives PASS while gating retains FAIL. A peer endorsement of an early approval always means a peer PASS. The direction of each reported error difference appears in its table heading.

The first three [exhibit tables](exhibits.pdf) compare completed revisions, early approvals and the absolute error counts under each policy. The [full-audit analysis](code/full_audit_exhibits.py) rebuilds 69 estimates and their pointwise intervals from the cluster counts. Its [complete estimates file](generated/full_audit_estimates.csv) records the number of defined bootstrap draws for each ratio. A blank estimate means there were no qualifying cases. The [cross-scoring analysis](code/cross_scorer_exhibits.py) reconstructs peer disagreements under immediate and gated updating from the original outcomes and checks the verdict-loss identity. Intervals use 1,999 paired whole-cluster resamples with fixed paths (70 articles or 30 legal tasks). Each resample preserves an output's expert rotations. The 31 query orders describe the audit design. The [machine-readable curves](generated/crossover_curves.csv) give errors for both rules, their differences and the paired change upon switching scorer at all 11 budget points.

For optional reproduction starting from the authors' expert and judge releases, see [source replay instructions](source_replay/README.md). The numerical route above is offline after installing the Python dependencies.

## Reference populations

The official Multi-IF English checker is the primary reference for the fresh panel. Literal suffix matching and deleting the ending are sensitivities chosen after the original panel, before the repeated calls. RuVerBench tasks group criterion verifications. JudgmentBench units in the large frame are output-by-rater annotations. The legal crossover uses 206 distinct outputs with repeated attorney ratings, giving 490 directed ordered pairs. An earlier A/B analysis in the exhibits uses two designated rating positions per output. ProfBench Critical requires all criteria designated Critical for that target. Derived labels are scored relative to their stated references. [Multi-IF](https://huggingface.co/datasets/facebook/Multi-IF) · [RuVerBench](https://github.com/THU-KEG/RuVerBench) · [JudgmentBench](https://arxiv.org/abs/2605.25240v5) · [ProfBench](https://arxiv.org/abs/2510.18941)

## Supporting experiments

SummEval contributes 480 development summaries from 30 articles and 1,120 held summaries from 70 disjoint articles. The evaluator scores were already available from G-Eval and UniEval. The protocol in inputs/external_protocol.md predeclares the four dimensions, thresholds, a development-calibrated sensitivity, loss weights and paired article bootstrap. external_seal.json fixes the development choices. external_groups.csv contains anonymized held article counts. Running reproduce.sh recalculates all six external rows, the key paired intervals and a post hoc constant-FAIL comparison. The study's target is a strict transformation of SummEval's original ordinal expert ratings. [SummEval](https://doi.org/10.1162/tacl_a_00373) · [Archived G-Eval scores](https://github.com/nlpyang/geval/tree/6f84404ce4503d2ba0cde0293c5d402ecf2976a8/results) · [Archived UniEval scores](https://github.com/maszhongming/UniEval/tree/d33e7b6cfebe97b2bafe435adbd818230d5a416a/reproduce/unieval_predict/summarization)

The [working-paper bibliography](related_work.bib) covers prior work on human disagreement, judging, selective verification and the relationship between item-level and verdict-level errors.

The offline route starts from saved model predictions and numerical audit outcomes. The optional source replay starts the crossover from the published records. New model calls would constitute additional observations.

The open-weight pilot files record development objectives, the development choices that were sealed, and their subsequent application to five held legal tasks. They contain model counts and hashes. Qwen2.5-0.5B and Gemma 3-1B scored 51 distinct inputs each on a private CPU runner. Provider fees were zero. The paid prompt-repeat experiment used 514 returned requests and cost $4.472320 at the pinned usage rates. The constant-FAIL comparisons are post hoc diagnostics. See DATA_NOTICES.md for attribution and distribution terms.

The selector verifier recomputes the three decisions from aggregate criterion counts, development-only rubric composition and task errors. It checks all 135 historical candidate forecasts, the choices in each historical fold and the two external development representations at all three error costs. It requires no article keys or individual human ratings.
