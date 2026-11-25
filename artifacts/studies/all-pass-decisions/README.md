# Beyond Criterion Agreement

This companion reproduces the quantitative exhibits for the combined working draft on judge selection, reference checking and all-pass task decisions. The release uses numerical records only. The original annotations and rubric text are available under each upstream benchmark's distribution terms.

## Reproduce

With Python 3.12, install the packages in requirements.txt and run the commands below. Tectonic is optional for numerical reproduction. Its first run can download LaTeX packages.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
./reproduce.sh
./reproduce.sh --pdf
```

The build writes TeX table rows and a vector figure to generated/. exhibits.tex incorporates these outputs directly and compiles into a standalone exhibit PDF. check_estimates.py independently rebuilds the legal cross-rater intervals from clustered counts and checks the RMSE, hypergeometric coverage and expected interval width for 432 hard-budget sampling designs. The compact inputs preserve the numerical observations needed by these computations. A fresh call to a hosted model can return a different answer.

| Input | Starting point for regeneration |
|---|---|
| paired_bits.json | Per-source criteria, checker variants and three grading occasions for the fixed 62-response Multi-IF panel. Contains the ancestral four-model eligibility mask. |
| rater_orders.csv.gz and rater_groups.csv | Counts at each acquisition order and budget, and group-wise counts at the legal panel's half-budget point. The group file reproduces the 1,999 paired task reweightings. |
| cost_orders.csv.gz and pass_design.csv | 40,176 order/cost records and finite-population stratum counts for computing precision and coverage. |
| fold_forecasts.csv and forecast_validation.csv | Fold-specific development scores, held-out error counts and directional-rate forecasts on four historical frames. Ties use the recorded candidate order. |

## Reference populations

The official Multi-IF English checker is the primary reference for the fresh panel. Literal suffix matching and deleting the ending are sensitivities chosen after the original panel, before the repeated calls. RuVerBench tasks group criterion verifications. JudgmentBench units in the large frame are output-by-rater annotations. The cross-rater experiment instead uses 206 distinct outputs with two designated raters per output. ProfBench Critical requires all criteria designated Critical for that target. Derived labels are scored relative to their stated references. [Multi-IF](https://huggingface.co/datasets/facebook/Multi-IF) · [RuVerBench](https://github.com/THU-KEG/RuVerBench) · [JudgmentBench](https://arxiv.org/abs/2605.25240v5) · [ProfBench](https://arxiv.org/abs/2510.18941)

## External selection test

SummEval contributes 480 development summaries from 30 articles and 1,120 held summaries from 70 disjoint articles. The evaluator scores were already available from G-Eval and UniEval. The protocol in inputs/external_protocol.md predeclares the four dimensions, thresholds, a development-calibrated sensitivity, loss weights and paired article bootstrap. external_seal.json fixes the development choices. external_groups.csv contains anonymized held article counts. Running reproduce.sh recalculates all six external rows, the key paired intervals and a post hoc constant-FAIL comparison. The study's target is a strict transformation of SummEval's original ordinal expert ratings. [SummEval](https://doi.org/10.1162/tacl_a_00373) · [Archived G-Eval scores](https://github.com/nlpyang/geval/tree/6f84404ce4503d2ba0cde0293c5d402ecf2976a8/results) · [Archived UniEval scores](https://github.com/maszhongming/UniEval/tree/d33e7b6cfebe97b2bafe435adbd818230d5a416a/reproduce/unieval_predict/summarization)

The two additional nearest-neighbor comparisons concern Hamming versus exact-vector loss and prevalence correction using sensitivity and specificity. An all-pass verdict pools many different item vectors into one FAIL outcome. Calibrating the prevalence of passing uses operating characteristics at the verdict level. Bibliographic details are in related_work.bib. [Dembczyński et al.](https://doi.org/10.1007/978-3-642-15880-3_24) · [Rogan and Gladen](https://doi.org/10.1093/oxfordjournals.aje.a112510)

This package starts from saved model predictions and numerical audit outcomes. Full regeneration of the original provider responses, authors' historical Paper A, upstream data collections and raw expert annotations falls outside its scope.

The open-weight pilot files record development objectives, the development choices that were sealed, and their subsequent application to five held legal tasks. They contain model counts and hashes. Qwen2.5-0.5B and Gemma 3-1B scored 51 distinct inputs each on a private CPU runner. Provider fees were zero. The paid prompt-repeat experiment used 514 returned requests and cost $4.472320 at the pinned usage rates. The constant-FAIL comparisons are post hoc diagnostics. See DATA_NOTICES.md for attribution and distribution terms.

The selector verifier recomputes the three decisions from aggregate criterion counts, development-only rubric composition and task errors. It checks all 135 historical candidate forecasts, the choices in each historical fold and the two external development representations at all three error costs. It requires no article keys or individual human ratings.
