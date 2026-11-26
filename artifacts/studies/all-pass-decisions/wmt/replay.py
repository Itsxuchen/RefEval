"""Recompute the WMT held estimates and check the post-hoc analyses against the release."""
import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src import wmt_blind_crossover as study
from src import wmt_blind_inputs as inputs

ROOT = Path(__file__).resolve().parent
FOLLOWUP = ROOT / "artifacts/reports/wmt_followup_20261126"


def compare(actual, expected, where):
    """Allow floating-point evaluation order to vary across BLAS implementations."""
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"Different fields in {where}")
        for key, value in expected.items():
            compare(actual[key], value, f"{where}/{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Different lengths in {where}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{where}/{index}")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-12):
            raise ValueError(f"{where}: reproduced {actual}, expected {expected}")
    elif actual != expected:
        raise ValueError(f"{where}: reproduced {actual}, expected {expected}")


def load_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def verify_files(directory, reference, names):
    for name in names:
        method = load_rows if name.endswith(".csv") else study.read
        compare(method(directory/name), method(reference/name), name)


def offline():
    held = study.read(study.OUT / "held.json")
    seal = study.read(study.OUT / "seal.json")
    compare(study.code_hashes(), seal["code"], "frozen code and protocol")
    compare(study.digest(study.INITIAL / "seal.json"), seal["original_seal_sha256"], "initial seal")
    compare(study.digest(study.OUT / "seal.json"), held["seal_sha256"], "correction receipt")
    compare(study.digest(study.OUT / "development.json"), seal["development_sha256"], "development")
    counts = study.read(study.OUT / "document_totals.json")
    docs = sorted(counts)
    compare(docs, seal["split"]["held"], "held documents")
    compare(sum(int(counts[d]["N"]) for d in docs), held["outputs"], "held outputs")
    # Each frozen row is one document's totals; analyze resamples intact documents.
    contributions = {k: np.array([counts[d][k] for d in docs]) for k in counts[docs[0]]}
    estimates, _ = study.analyze([SimpleNamespace(doc=d) for d in docs], contributions)
    compare(estimates, held["estimates"], "document bootstrap")
    endpoints = study.read(FOLLOWUP / "pairs_alt_identity.json")
    machine = endpoints["llm_strict"]
    human = endpoints["human_strict"]
    total = machine + human + endpoints["ties"]
    compare(total, 3*held["outputs"], "candidate rotations")
    compare(100*(machine-human)/total, estimates["full_minus_retain_1to1_per100"]["point"],
            "signed advantage")
    early = estimates["early_per100"]["point"]
    timing = estimates["same_early_minus_eventual_1to1_per100"]["point"]
    print(f"Verified {len(estimates)} estimates and intervals from {len(docs)} documents; "
          f"same-case discrepancies per 100 early releases = {100*timing/early:.4f}.", flush=True)


def run_module(module, *args):
    subprocess.run([sys.executable, "-B", "-m", "src."+module, *map(str, args)],
                   cwd=ROOT, check=True)


def source(cache, out):
    inputs.prepare(cache)
    seal = study.read(study.OUT / "seal.json")
    for name, expected in seal["source_sha256"].items():
        compare(study.digest(cache/name), expected, name)
    confirmation = out/"confirmation"
    confirmation.mkdir()
    run_module("wmt_blind_crossover", "develop", "--cache", cache, "--out", confirmation)
    verify_files(confirmation, study.OUT, ["development.json"])
    shutil.copyfile(study.OUT/"seal.json", confirmation/"seal.json")
    run_module("wmt_blind_crossover", "held", "--cache", cache, "--out", confirmation)
    verify_files(confirmation, study.OUT, ["held.json", "document_totals.json"])

    sensitivity = out/"sensitivity"
    sensitivity.mkdir()
    run_module("wmt_peer_pairs", "--cache", cache, "--out", sensitivity)
    run_module("wmt_withdrawn_spans", "--cache", cache, "--out", sensitivity/"spans.json")
    run_module("wmt_prompt_provenance", "--cache", cache, "--out", sensitivity/"prompts.json")
    expected = sorted(p.name for p in FOLLOWUP.iterdir() if p.suffix in (".json", ".csv"))
    verify_files(sensitivity, FOLLOWUP, expected)
    print(f"Verified the development and held results and {len(expected)} sensitivity files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="store_true", help="also replay four archived upstream inputs")
    parser.add_argument("--cache", type=Path, help="download or reuse the pinned upstream files here")
    parser.add_argument("--out", type=Path, help="save new replay outputs in an unused directory")
    args = parser.parse_args()
    if args.source and args.cache is None:
        parser.error("--source requires --cache")
    if not args.source and (args.out or args.cache):
        parser.error("--out and --cache belong to --source")
    offline()
    if args.source:
        if args.out is not None:
            out = args.out.resolve()
            out.mkdir(parents=True, exist_ok=False)
            source(args.cache.resolve(), out)
        else:
            with tempfile.TemporaryDirectory() as temporary:
                source(args.cache.resolve(), Path(temporary))
