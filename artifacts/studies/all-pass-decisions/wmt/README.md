# WMT23 English-to-German news

On 739 held translations, a blind peer endorsed 50.3% of complete revisions from FAIL to PASS and 52.0% from PASS to FAIL. An early approval followed to the same auditor's final verdict added 0.81 peer disagreements per 100 translations, or 5.47 per 100 early approvals. The first rate describes the whole replayed workload; the second describes the approvals that were actually issued. Three professionals rated each translation from a blank interface in the first round. Ten distinct translators contributed across the 18 held news documents. [Expert annotation procedures](https://aclanthology.org/2026.acl-long.1553/) · [Machine evaluation procedures](https://aclanthology.org/2026.emnlp-main.2213/)

## Reproduce

Use Python 3.12 with the parent study's pinned NumPy version. Run from the parent study directory.

```sh
python wmt/replay.py
python wmt/replay.py --source --cache /tmp/refeval-wmt23-inputs
```

The first command works offline. It reconstructs all 43 held estimates and pointwise 95% intervals from [18 document-level count records](artifacts/reports/wmt_blind_crossover_20261126/corrected/document_totals.json) and checks them against the released [result](artifacts/reports/wmt_blind_crossover_20261126/corrected/held.json). Resampling preserves the expert rotations and averages over the 31 fixed query paths. Intervals describe variation across documents with the observed panel.

The second command downloads or verifies four upstream files (about 170 MiB) by pinned Git blob ID and SHA-256. It reruns the development gate with nine documents, applies the recorded seal, and recomputes the confirmation and nine follow-up outputs. The template audit fetches the pinned template as part of this route. No model API or credentials are required. To retain the results, add `--out /path/to/new/directory`. The comparison accepts numerical differences up to floating-point tolerance across platforms.

## Protocol and evidence

The [original protocol](context/wmt_blind_crossover_20261126.md) fixes the split, evidence order, criterion taxonomy, losses and primary contrast. Both [original](artifacts/reports/wmt_blind_crossover_20261126/seal.json) and [corrected](artifacts/reports/wmt_blind_crossover_20261126/corrected/seal.json) receipts are retained. After the first held run, the parser was corrected to exclude a source-only error under its released name. The equal-cost complete-minus-retain effect changed from −0.54 to −0.50 per hundred. The first run's [complete source and results](archive/wmt_blind_first_run_20261126.zip) and [correction note](context/wmt_blind_source_issue_correction_20261126.md) document that chronology. The receipts were timestamped locally.

The [follow-up directory](artifacts/reports/wmt_followup_20261126/) contains six role rotations, actual pair and leave-person-out summaries, the binary alt-test identity, span matching on withdrawn approvals, prior-stratified second-pass scores and a reconstruction of the recorded machine prompts. These analyses were specified after reading the confirmation. The all-pass target combines four project-defined major-error checks. Translators initially annotated whole segments; partial review simulates revealing individual categories.

## Sources and reuse

Human MQM ratings and re-annotation records come from Google's WMT release, pinned at commit 29acd6999aaea3586378c9dcb58d8cc4820cede3. Its repository includes an Apache 2.0 license. The news translations and source articles remain subject to their respective rights. The replay reads the files from the upstream publisher. [Human rating release](https://github.com/google/wmt-mqm-human-evaluation/tree/29acd6999aaea3586378c9dcb58d8cc4820cede3/generalMT2023) · [Repository license](https://github.com/google/wmt-mqm-human-evaluation/blob/29acd6999aaea3586378c9dcb58d8cc4820cede3/LICENSE) · [Freitag et al.](https://aclanthology.org/2021.tacl-1.87/)

The archived Sonnet 4.5 and Haiku 4.5 machine annotations and the prompt belong to the Span-Level Machine Translation Meta-Evaluation release, pinned at ca494e2626fd85f440ba3934404d347b281bdfcc. Its repository is licensed CC BY-NC 4.0. This companion distributes derived counts, study code and receipts; readers obtain the input files from their publishers. [Machine releases and paper](https://github.com/amazon-science/span-mt-metaeval/tree/ca494e2626fd85f440ba3934404d347b281bdfcc) · [License](https://github.com/amazon-science/span-mt-metaeval/blob/ca494e2626fd85f440ba3934404d347b281bdfcc/LICENSE)
