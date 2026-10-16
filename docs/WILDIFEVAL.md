# WildIFEval masked preflight: stopped before pilot

Date: 2026-10-15. Status: **input association failed; no external outcome analysis**.

The proposed four-judge frame cannot support the planned external validation from the inspected public snapshot. Two separate problems arise: extra judges cover a small subset, and the two broadly covered judges score different complete criterion sets on most shared task texts. Removing the extra-judge requirement does not repair the join.

| Frame | Common tasks / clusters | Producer-output units | Pilot / confirmation clusters | Retained benchmark tasks |
|---|---:|---:|---:|---:|
| Planned P, Q, R, R2 | 4 | 20 | 0 / 4 | 0.0532% |
| P/R-only diagnostic | 40 | 200 | 8 / 32 | 0.5317% |

The original gates require at least 500 eligible clusters, 100 pilot clusters, 300 confirmation clusters and 70% retention. Both frames fail every gate. Hash-split counts above describe a masked diagnostic assignment; none of those labels was released to an analysis stage.

For each of the five producing-model files, P/R share 9,779 task-text keys. Complete criterion sets agree on 212; whitespace normalization does not increase this count. Criterion counts differ on 5,835 shared prompts. On the current 7,523-task benchmark, complete valid P vectors exist for 40 tasks per producer; R covers 7,418–7,419, R2 339–343, and Q 342–347. The exclusion counts overlap and must not be added as disjoint missingness categories.

The source card's multi-judge availability did not establish criterion-level compatibility. Our earlier feasibility recommendation was premature. This execution checked the actual full associations before looking at performance and therefore stopped without a pilot, a fitted prediction or held-out outcome. It does not provide negative evidence against the mechanism.

## Evidence and checks

- [Canonical masked report](../artifacts/expected/wildifeval_external/preflight_report.json): counts, source inventory, duplicate diagnostics, role completeness, criterion-count matrix and feasibility gates.
- `artifacts/expected/wildifeval_external/masked_units.csv.gz`: all 37,615 candidate producer-output units, hashed identities, full criterion positions, missingness, exclusions and diagnostic splits; no score values or source text.
- `artifacts/expected/wildifeval_external/excluded_units.csv.gz`: excluded candidate identities and overlapping reason categories.
- [Independent verification](../artifacts/expected/wildifeval_external/independent_verification.json): a second parser and join, using validity sentinels rather than retaining numerical scores.
- [Protocol and stopping rule](../context/conjunction_external_validation_protocol.md).

Raw downloaded artifacts are local-only under a private local cache (the commands below use ignored `artifacts/wildifeval-cache/`). The public package distributes source hashes and compressed masked tables, with commands to redownload the pinned inputs. The benchmark snapshot is `fc5f47b2b1be19ac95986624a4c0d8fd25dd316c`; the predictions snapshot is `e34bc0fcc01b7f1d49032878563e8c32393c9d59`.

The benchmark omits task text, so the join maps only unique exact complete criterion sets to saved task keys. Thirty-one repeated sets involve 84 benchmark rows and are excluded. The source score files do not store evaluated-output hashes, so the producer/task filename link is a provenance assumption, not an independent proof of identical output versions. Scores were read only for schema/validity masks during preflight, never thresholded into reference PASS/FAIL outcomes.

## What is needed to continue

Obtain a correctly aligned saved primary-judge file, or generate a new fixed judge run on the exact benchmark rubric and saved output. Then repeat masked joins and freeze any revised model roles, source versions and protocol before pilot access. Do not use semantic approximate criterion matching, delete failed conditions, lower feasibility thresholds to fit the observed subset, or report these four/forty clusters as the intended external confirmation.

The staged access loader refuses failed frames. Prediction and policy code has been tested on synthetic fixtures for later use; no claim of real-data external validation follows from those tests. The current manuscript and v0.3.0 scientific outputs are unchanged.

## Reproduce the preflight

The included compressed tables contain only metadata. Check their hashes, common-frame counts and original stopping gates offline:

```bash
python -m src.verify_wildifeval_preflight --saved-only artifacts/expected/wildifeval_external
```

To regenerate from the pinned upstream files, install the optional Parquet reader and download about 1.09 GB. Downloads validate all 28 source hashes against the published manifest. Use a fresh output directory; the preflight refuses to overwrite an existing report.

```bash
python -m pip install ".[external]"
python -m src.wildifeval_data download --sources artifacts/wildifeval-cache
python -m src.wildifeval_data preflight --sources artifacts/wildifeval-cache --output artifacts/wildifeval-preflight
python -m src.verify_wildifeval_preflight --sources artifacts/wildifeval-cache --preflight artifacts/wildifeval-preflight --output artifacts/wildifeval-verification/independent_verification.json
```

The second implementation checks source hashes and independently reconstructs the join, clusters and splits without importing the production join helpers. The recorded run checked 37,615 rows and 752,300 metadata cells. Its JSON parser replaces numeric literals directly with range-validity sentinels. The production parser inspects score types/ranges but never thresholds them or exports their values during preflight. These are workflow access controls, not encryption: local source files still contain scores.

`wildifeval_forecast.py` and `wildifeval_policy.py` are preparatory components tested only on synthetic inputs. Their finite-population and prefix-count tests do not validate an external empirical forecast. The stage loader refuses reference access when feasibility fails. No outcome runner or design/prediction freeze receipt was issued.

## Bounded search for an aligned source

The pinned predictions inventory contains 64 files, including one Llama-3.1-70B grader file for each of 14 producers. For the selected five producers, this check found no alternative aligned primary-judge file. The official code at [revision 82ebff7](https://github.com/gililior/wild-if-eval-code/tree/82ebff7068b9faeadc5468d55935d62b1601d89c) points to the same predictions repository. Its agreement-analysis code uses DeepSeek, Llama-3.3 and Qwen against GPT-4o; the planned primary judge here was Llama-3.1. This search covered the pinned inventory and relevant download/evaluation/analysis code, not every historical revision or private asset.

## Local package checks

The update passed 291 tests (58 new metadata/synthetic tests plus the previous 233), 607 existing manuscript numerical checks, the independent source verification and the offline metadata check. All 148 existing tracked files under source, tests, data, expected results, figures and manuscript retain their v0.3.0 bytes. [Validation record](../artifacts/validation/wildifeval_preflight_20261015.json). These checks do not turn the failed preflight into empirical external validation.
