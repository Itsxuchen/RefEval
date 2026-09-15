# Contributing

Please report reproducibility issues with the release tag or commit, Python and
dependency versions, operating system, command, and a minimal error log. Do not
include credentials or unrelated private data.

Changes to scoring targets, reference labels, sampling units, query costs or
update rules change the scientific question. Describe those changes explicitly
and add tests for the affected contract. Keep the expected numerical outputs
immutable within a release; a new scientific result requires a new release.

Run `python -m pytest -q` and the full reproduction commands in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before proposing a numerical change.

