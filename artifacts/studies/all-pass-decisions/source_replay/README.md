# Rebuild both cross-scorer experiments from the public ratings

The paper's artifact can build its table and budget curves from the checked-in aggregate counts. This optional workflow independently reconstructs those counts from the original expert and judge records. It uses 14 checksum-pinned public files, about 79 MiB in total. No model inference or API key is needed.

From the package root run:

```sh
python -m pip install -r source_replay/requirements.txt
python source_replay/replay.py --out build/raw-cross-scorer
```

Use Python 3.12 and a new output directory. Dependencies are NumPy, SciPy and Requests, with Matplotlib already included in the parent exhibit environment. The command stages original downloads in temporary storage, reconstructs the two frames, and retains the numerical summaries and validation records in the requested directory. Python's temporary directory setting controls where intermediate source data are processed.

## What is verified

1. SummEval uses all three individual expert vectors within each summary, the six pinned author files, the historical article split, the G-Eval response parser and both original fixed thresholds. The historical development and held majority-vote panels are recreated solely to check input alignment. Their SHA-256 values are matched to the original design receipt, then the crossover scores each expert's individual verdict. A separate four-bit implementation rebuilds every article-by-role count and checks the endpoints.
2. JudgmentBench downloads the pinned rubric items, human and two judge annotations/item scores, and output metadata. The original SHA-256 checks include each entire source file. The two-lawyer control pairing and fixed primary/auxiliary prediction anchor are rebuilt using the recorded identity hashes. All available attorneys then rotate through every auditor/peer role on each of the 206 repeatedly annotated outputs; the 120 previously published anchor controls are also verified.
3. The two results are compared to the numerical inputs used to produce the paper. The verifier checks 33,660 SummEval article-by-role rows (134,640 integer outcome counts) after matching the released article IDs to their global lexical ranks. It also checks all fields in 330 JudgmentBench base-task-by-budget rows, with a tolerance of 1e-10 on numeric values. `source_checks.json` additionally records whether six files have the exact historical byte representation. Different BLAS, compression libraries or floating-point serialization can affect this last diagnostic independently of the numerical checks.

The original frozen design is at [../inputs/cross_scorer_protocol.md](../inputs/cross_scorer_protocol.md). Its hash and its four historical input digests are in [controls/design.sha256](controls/design.sha256). [controls/expected.json](controls/expected.json) records all eight JudgmentBench URLs, input/output hashes and source revision. The six SummEval family URLs and hashes are in [controls/external_freeze.json](controls/external_freeze.json). The code under `frozen_code/` preserves the study implementations. Their paths are staged into a temporary historical layout at runtime so their own integrity checks can run. The three other shared source modules come from `../code/src/`. SummEval's earlier seal and protocol are reused from `../inputs/`.

## Sources and optional offline use

For the summary experiment, the source articles and generated summaries accompany the recorded scores in the authors' original files. The analysis consumes these files in temporary storage. The numerical package contains derived counts and source hashes. [SummEval](https://github.com/Yale-LILY/SummEval) · [G-Eval](https://github.com/nlpyang/geval/tree/6f84404ce4503d2ba0cde0293c5d402ecf2976a8/results) · [UniEval](https://github.com/maszhongming/UniEval/tree/d33e7b6cfebe97b2bafe435adbd818230d5a416a/reproduce)

For the legal experiment, original output text appears in the published output table that the loader uses to verify output identities. Human and automated annotations are from the JudgmentBench authors. See the source card for its terms. [JudgmentBench, pinned revision](https://huggingface.co/datasets/judgmentbench/JudgmentBench/tree/945ff52f4c63c17006dc30ec35fefbddf3ccf58d) · [Package data notices](../DATA_NOTICES.md)

To use source files already obtained from the original hosts, pass `--summeval-cache DIR` and/or `--judgmentbench-cache DIR`. These directories are read only. The first contains `experts.jsonl`, `summeval_result.json` and the four `gpt4_*_detailed.json` files. The second uses the eight dictionary keys of `judgmentbench_sources` in [controls/expected.json](controls/expected.json) as relative filenames, including `outputs/outputs.csv`. The source bytes are checked identically for cached and downloaded runs.
