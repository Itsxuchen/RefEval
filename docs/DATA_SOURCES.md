# Data sources and fixed-version provenance

This package reproduces numeric experiments from fixed released references and saved judge predictions. The required tables are included; normal reproduction does not download source datasets or call model APIs. The package studies estimation, certification, and verdict updates under stated reference-query budgets. It does not measure expert labor or establish that a reference is an error-free measure of professional quality.

[data/PROVENANCE.json](../data/PROVENANCE.json) records source-file SHA-256 hashes, packaged-file hashes, retained and removed columns, row counts, and transformations. A source hash identifies the original file before projection; a packaged hash identifies the distributed bytes. Licenses are scoped in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## RuVerBench

Source: THU-KEG, [RuVerBench](https://github.com/THU-KEG/RuVerBench/tree/0af0256870d6322490c7ab15eb6370779b898a37), revision `0af0256870d6322490c7ab15eb6370779b898a37`. Paper: Peng et al. (2026), [Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?](https://arxiv.org/abs/2606.29920v2).

The included files are this study's compact prediction tables, not an upstream release of predictions for every judge in the RuVerBench paper:

| Packaged file | Saved runs | Rows |
| --- | --- | ---: |
| `data/processed/public_judge_predictions.csv` | DeepResearch: Gemini 3.1 Pro and Qwen-plus; AgenticCoding: Qwen-plus and DeepSeek v4-pro | 4,916 |
| `data/processed/gpt54_low_dr_predictions.csv` | DeepResearch: GPT-5.4, low reasoning | 1,615 |

Both files retain `run`, `domain`, `task_id`, `criterion_id`, `criterion_index`, `category`, `gold`, `provider`, `model`, and `prediction`. Reference labels and taxonomy fields come from RuVerBench; prediction bits and run metadata record this study's judge runs. The two packaged files preserve the saved CSV bytes. They contain identifiers and scores, without full source prompts, rubric bodies, candidate outputs, or judge response prose. The four-run table covers 494 distinct tasks; its 4,916 rows are criterion judgments, not independent tasks. The extra GPT-5.4 run reuses the DeepResearch references.

The reference-label and taxonomy source files below were fetched from the pinned revision on 2026-09-15 and matched the local study inputs byte for byte. Paths are relative to `data/benchmark/` in the upstream repository.

| Source file | SHA-256 |
| --- | --- |
| `deepresearch_labels.json` | `879e4f7974a28e02db8f79801a42b824a04df7a5f7211506d4953667317d55a5` |
| `deepresearch_taxonomy.json` | `4d810ae986565956a79300c8a842bc7e0585eefd395953546a9531daf85875e7` |
| `agenticcoding_labels.json` | `69050ef6f844ec3d7ee623b2a91ca1ca3f4eb4d77b8653c8e4be9596c22a95ee` |
| `agenticcoding_taxonomy.json` | `3baf1bb19b0bdbb15594425a80a4b6ae5369f172c576b72b187296a7a185ee7e` |

RuVerBench filters and downsamples criteria. The study's conjunction means passing all **retained criteria** for a task, not passing every condition in the original upstream task. Its `gold` field is the released reference, not a new human annotation collected for this study. Task and criterion identifiers preserve joins; DeepResearch criterion indices follow the retained source order.

The [pinned data notice](https://github.com/THU-KEG/RuVerBench/blob/0af0256870d6322490c7ab15eb6370779b898a37/DATA_LICENSE.md) licenses RuVerBench-created annotations, taxonomy, normalization, and analysis artifacts under CC BY-NC 4.0. It expressly excludes upstream source content from that grant. The package retains the relevant source attribution in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md); its code license does not replace those terms.

## JudgmentBench

Source: [judgmentbench/JudgmentBench on Hugging Face](https://huggingface.co/datasets/judgmentbench/JudgmentBench/tree/945ff52f4c63c17006dc30ec35fefbddf3ccf58d), revision `945ff52f4c63c17006dc30ec35fefbddf3ccf58d`. Paper: Yang et al. (2026), [JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment](https://arxiv.org/abs/2605.25240v3). The [pinned dataset card](https://huggingface.co/datasets/judgmentbench/JudgmentBench/blob/945ff52f4c63c17006dc30ec35fefbddf3ccf58d/README.md) identifies the released dataset as MIT-licensed and states that the BigLaw Bench base tasks were included with the original rights holder's permission.

The package includes these numeric and identifier projections under `data/reference/judgmentbench/`:

| Upstream path at the pinned revision | Packaged filename | Rows | Removed source columns |
| --- | --- | ---: | --- |
| `base/rubric_items.csv` | `rubric_items.csv` | 457 | `section`, `label` |
| `human/annotations_rubric.csv` | `human_annotations_rubric.csv` | 1,539 | `time_spent_seconds`, `comment` |
| `human/rubric_item_scores.csv` | `human_rubric_item_scores.csv` | 23,487 | None |
| `autograders/gpt_5_4/annotations_rubric.csv` | `gpt_5_4_annotations_rubric.csv` | 1,539 | `comment` |
| `autograders/gpt_5_4/rubric_item_scores.csv` | `gpt_5_4_rubric_item_scores.csv` | 23,487 | None |
| `autograders/gpt_5_4_mini/annotations_rubric.csv` | `gpt_5_4_mini_annotations_rubric.csv` | 1,539 | `comment` |
| `autograders/gpt_5_4_mini/rubric_item_scores.csv` | `gpt_5_4_mini_rubric_item_scores.csv` | 23,487 | None |
| `outputs/outputs.csv` | `outputs_metadata.csv` | 2,274 | `quality_level_order`, `version_number`, `output_text` |

Projections retain source row order and field values. `outputs_metadata.csv` contains only `output_id`, `task_id`, and `quality_level`. Every source file in the table was independently fetched at the pinned revision on 2026-09-15 and matched the local pre-projection study input byte for byte, including the two autograder releases. Source and packaged hashes are recorded separately in [the manifest](../data/PROVENANCE.json).

The GPT-5.4 and GPT-5.4-mini item scores are released JudgmentBench autograder data; this package did not generate them. Human annotation identifiers, output identifiers, and base-task identifiers represent different units. Repeated or quality-varied outputs are not independent tasks, and the autograders are not additional human raters.

The primary strict conjunction is a researcher-defined stress-test target: full credit on retained binary criteria and no penalty occurrence. It is not JudgmentBench's native weighted rubric headline score. Scoring conversions and sensitivity targets are implemented in `src/conjunction_policy_data.py` and described in the study protocol.

## Optional source restoration

To independently rebuild the projections, run this optional command from the repository root:

```bash
python -m src.restore_source_projections --output artifacts/restored-inputs
```

The destination must be new and must not overlap protected package content. The script downloads the eight JudgmentBench source files to temporary storage, checks source hashes, applies the manifest's ordered columns, and checks each resulting hash and row count. It then writes numeric projections and `restore_report.json` to the destination; full source downloads are deleted. The two RuVerBench files are copied from the package after hash verification: this command cannot regenerate the authors' judge predictions from reference labels. Normal experiment reproduction uses the already included tables and does not need this download step.

For separate inspection of upstream context, use the revisions above instead of `main`. Exact source download URL patterns are:

```text
https://raw.githubusercontent.com/THU-KEG/RuVerBench/0af0256870d6322490c7ab15eb6370779b898a37/data/benchmark/<source-file>
https://huggingface.co/datasets/judgmentbench/JudgmentBench/resolve/945ff52f4c63c17006dc30ec35fefbddf3ccf58d/<upstream-path>
```

Keep any separately downloaded source context outside the bundled data directories and check its source hash. The restoration script uses the package's CSV serialization as well as the declared column projection, so its packaged-byte check verifies both data values and representation.

Full task prompts, source documents, rubric text, output prose, and annotation comments are outside this compact package. Their omission is a packaging choice, not a claim that all upstream redistribution is prohibited. Any redistribution of restored source content should follow the specific source notices. Fresh model runs, if conducted separately, can differ with provider version and execution date; they are a different replication target from replaying the included predictions.
