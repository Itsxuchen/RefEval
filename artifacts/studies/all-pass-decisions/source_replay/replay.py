"""Reconstruct the cross-scorer study from the original public ratings."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import requests


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
CONTROLS = HERE / "controls"


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, expected, destination, cached=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if cached is not None:
        copy(cached, destination)
    else:
        with requests.get(url, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            with destination.open("wb") as stream:
                for chunk in response.iter_content(1 << 20):
                    stream.write(chunk)
    if digest(destination) != expected:
        raise ValueError(f"Source SHA-256 mismatch: {url}")


def build_pairing(root, destination):
    """Recreate historical two-rater and fixed-judge anchors from public IDs."""
    from src.conjunction_reference_sensitivity import identity_rank, write_csv
    file = root / "data/raw/judgmentbench/human_annotations_rubric.csv"
    grouped = defaultdict(list)
    with file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[row["output_id"]].append(row)
    pairs = []
    for output, rows in grouped.items():
        raters = {r["annotator_id"] for r in rows}
        if len(raters) < 2:
            continue
        if len(set(r["task_id"] for r in rows)) != 1 or len(raters) != len(rows):
            raise ValueError(f"Inconsistent or duplicated attorney records: {output}")
        def key(salt, r):
            return identity_rank(salt, output, r["annotator_id"], r["annotation_id"])
        first, second = sorted(rows, key=lambda r: key("reference-panel-v1", r))[:2]
        anchor = min(rows, key=lambda r: key("judge-anchor-v1", r))
        pair = dict(base_task_id=first["task_id"], output_id=output,
                    available_annotations=len(rows), available_raters=len(raters))
        for name, row in (("A", first), ("B", second), ("anchor", anchor)):
            pair[name + "_annotation_id"] = row["annotation_id"]
            pair[name + "_rater_id"] = row["annotator_id"]
        pairs.append(pair)
    write_csv(destination, sorted(pairs, key=lambda r: (r["base_task_id"], r["output_id"])))


def install_sources(work):
    for script in sorted((HERE / "frozen_code").glob("*.py")):
        copy(script, work / "src" / script.name)
    for stem in ("conjunction_policy_data", "conjunction_policy_study", "finite_population_intervals"):
        copy(PACKAGE / "code/src" / (stem + ".py"), work / "src" / (stem + ".py"))
    (work / "src/__init__.py").touch()
    sys.path.insert(0, str(work))


def compare_exhibit_inputs(result):
    """Reconcile every article/role integer and each legal cluster contribution."""
    with gzip.open(result / "article_counts.csv.gz", "rt") as f:
        article_ids = sorted({r["article"] for r in csv.DictReader(f)})
    ranks = {k: f"article_{i:03d}" for i, k in enumerate(article_ids, start=1)}
    first, second = result / "article_counts.csv.gz", PACKAGE / "inputs/cross_scorer_article_counts.csv.gz"
    with gzip.open(first, "rt") as left, gzip.open(second, "rt") as right:
        a, b = csv.DictReader(left), csv.DictReader(right)
        if a.fieldnames != b.fieldnames:
            raise ValueError("Article input columns changed")
        n = 0
        for source, expected in zip(a, b, strict=True):
            source["article"] = ranks[source["article"]]
            if source != expected:
                raise ValueError(f"Article counts differ at row {n + 1}")
            n += 1
    with (result / "jb_cluster_contributions.csv").open() as left, (PACKAGE / "inputs/cross_scorer_legal_clusters.csv").open() as right:
        a, b = csv.DictReader(left), csv.DictReader(right)
        if a.fieldnames != b.fieldnames:
            raise ValueError("Legal input columns changed")
        m = 0
        for source, expected in zip(a, b, strict=True):
            if source["base_task"] != expected["base_task"]:
                raise ValueError(f"Legal task mismatch: {source['base_task']}")
            for k in a.fieldnames:
                if k != "base_task":
                    np.testing.assert_allclose(float(source[k]), float(expected[k]), rtol=0, atol=1e-10)
            m += 1
    if (n, m) != (33660, 330):
        raise ValueError(f"Incomplete exhibit inputs: {(n,m)}")
    return n, m


def compare_full_audit_table(actual, published, keys, expected_rows):
    """Compare keyed statistics, including undefined estimates and valid draw counts."""
    def read(path):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            value_columns = set(reader.fieldnames) - set(keys)
            records = {}
            for row in reader:
                key = tuple(row[k] for k in keys)
                if key in records:
                    raise ValueError(f"Duplicate key in {path.name}: {key}")
                records[key] = row
        return value_columns, records

    columns, records = read(actual)
    expected_columns, expected = read(published)
    if columns != expected_columns or records.keys() != expected.keys() or len(records) != expected_rows:
        raise ValueError(f"Incompatible rows or columns in {actual.name}")
    for key, row in records.items():
        for col in columns:
            value, other = row[col], expected[key][col]
            if value == "" or other == "":
                if value != other:
                    raise ValueError(f"Undefined statistic differs at {key}, {col}")
            elif not np.isclose(float(value), float(other), atol=1e-10, rtol=0):
                raise ValueError(f"Statistic differs at {key}, {col}: {value} versus {other}")
    return len(records)


def reproduce(work, out, sum_cache, jb_cache):
    install_sources(work)
    from src import external_summeval_frame as previous
    from src import summeval_cross_scorer as summaries
    from src import judgmentbench_cross_scorer_comparison as legal
    from src import verify_summeval_cross_scorer as independent
    from src import full_audit_endorsement

    expected = json.loads((CONTROLS / "expected.json").read_text())
    se_source = work / "summeval-downloads"
    for name, (url, checksum) in previous.SOURCES.items():
        fetch(url, checksum, se_source / name, None if sum_cache is None else sum_cache / name)
    for name, item in expected["judgmentbench_sources"].items():
        fetch(item["url"], item["sha256"], work / item["path"], None if jb_cache is None else jb_cache / name)

    result = summaries.OUT
    copy(PACKAGE / "inputs/cross_scorer_protocol.md", work / "context/summeval_cross_scorer_20261126.md")
    copy(CONTROLS / "design.sha256", result / "design.sha256")
    copy(PACKAGE / "inputs/external_protocol.md", previous.OUT / "pre_outcome_protocol.md")
    copy(CONTROLS / "external_freeze.json", previous.OUT / "pre_outcome_freeze_v2.json")
    previous.run("development", se_source)
    previous.run("held", se_source, PACKAGE / "inputs/external_seal.json")
    summaries.run(se_source, result)
    independent.verify(se_source, result)

    legal.REFERENCE.mkdir(parents=True)
    build_pairing(work, legal.REFERENCE / "pairing.csv")
    historical = "cluster_inputs_full_gpt54_half.csv"
    copy(CONTROLS / "anchored_legal_counts.csv", legal.CROSSED / historical)
    archived = expected["jb_inputs"]
    # The original functions consume these four manifest fields.
    (legal.REFERENCE / "manifest.json").write_text(json.dumps({
        "data_hashes": {"JB GPT-5.4": archived},
        "output_hashes": {"pairing.csv": archived[str((legal.REFERENCE / "pairing.csv").relative_to(work))]},
    }))
    (legal.CROSSED / "verification.json").write_text(json.dumps({
        "output_sha256": {historical: archived[str((legal.CROSSED / historical).relative_to(work))]},
        "source_sha256": expected["jb_code"],
    }))
    legal.run(result)
    articles, groups = compare_exhibit_inputs(result)
    full_result = work / "full-audit"
    full_audit_endorsement.run(se_source, full_result)
    full_groups = compare_full_audit_table(
        full_result / "cluster_counts.csv", PACKAGE / "inputs/full_audit_cluster_counts.csv",
        ("frame", "cluster"), 170)
    full_estimates = compare_full_audit_table(
        full_result / "estimates.csv", PACKAGE / "generated/full_audit_estimates.csv",
        ("frame", "metric"), 69)
    copy_files = ["summary.csv", "role_counts.csv", "jb_summary.csv", "verification.json",
                  "run_manifest.json", "jb_verification.json"]
    # Platform BLAS/NumPy and gzip/zlib versions can affect serialized floating
    # values or compressed bytes. The independent integer and tolerance checks
    # above are the portable evidence. Keep byte-equality as a diagnostic.
    matched = {name: digest(result / name) == checksum
               for name, checksum in expected["crossover_outputs"].items()}
    with tempfile.TemporaryDirectory(dir=out.parent, prefix="cross-scorer-copy-") as staging:
        stage = Path(staging) / "verified"
        for name in copy_files:
            copy(result / name, stage / name)
        for name in ("cluster_counts.csv", "estimates.csv"):
            copy(full_result / name, stage / "full_audit" / name)
        (stage / "source_checks.json").write_text(json.dumps({
            "summaries": 1600, "summaries_in_primary": 1120, "legal_outputs": 206,
            "source_files_verified": len(previous.SOURCES) + len(expected["judgmentbench_sources"]),
            "article_role_rows_compared": articles, "base_task_budget_rows_compared": groups,
            "full_audit_cluster_rows_compared": full_groups, "full_audit_estimates_compared": full_estimates,
            "historical_float_and_compression_byte_agreement": matched,
        }, indent=2) + "\n")
        stage.rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory for results and verification receipts")
    parser.add_argument("--summeval-cache", type=Path, help="Read the six original source files here instead of downloading")
    parser.add_argument("--judgmentbench-cache", type=Path, help="Read the eight files under the documented cache layout")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error("--out must be a new path")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="refeval-source-replay-") as d:
        reproduce(Path(d).resolve(), out, args.summeval_cache, args.judgmentbench_cache)
    print(f"Verified source replay: {out}")


if __name__ == "__main__":
    main()
