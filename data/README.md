# Analysis inputs

These are fixed-reference replay inputs. The complete field lists, row counts,
column projections and hashes are recorded in [PROVENANCE.json](PROVENANCE.json).
Source revisions and licenses are described in [DATA_SOURCES](../docs/DATA_SOURCES.md).

`processed/` contains study-generated judge prediction bits joined to released
RuVerBench reference labels. `gold=True` means that a retained criterion is met
under the released reference; `prediction=True` means the judge predicts it is met.
The task key is interpreted within its domain and run. Repeated judges do not add
new independent tasks. Both prediction tables retain their original bytes.

`reference/judgmentbench/` contains column projections of the public numeric and
identifier tables. Annotation, rater, output and base-task IDs have different
roles. Binary criteria are met at full credit; occurrence-count penalties are met
at zero occurrences. The strict conjunction is defined for this study, not the
benchmark's native weighted score. The output metadata table includes all released
output identities, of which a subset appears in the analyzed rubric annotations.

Projection removes source text, annotation comments and human timing while
preserving order, identity and numerical fields. Missing, duplicated, misaligned
or out-of-domain inputs are rejected by the strict loader. No imputation is used.

Third-party labels retain their original terms. The repository's software MIT
license does not apply as a replacement dataset license.
