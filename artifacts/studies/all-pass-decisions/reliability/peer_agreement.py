"""Compare expert agreement on one criterion and on the verdict for the same output.

Each output has unit weight. Within an output all ordered pairs of different
experts have equal weight. The criterion estimand also draws one of the
retained criteria uniformly. Kappa uses the pooled binary marginal for this
exchangeable role assignment; it is not a mean of fixed-rater kappas.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from urllib.request import urlopen

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "code" / "src"))
sys.path.insert(0, str(PACKAGE / "source_replay" / "frozen_code"))
sys.path.insert(0, str(PACKAGE / "wmt" / "src"))

from conjunction_policy_data import read_judgmentbench
from external_summeval_frame import DIMS, partition
import wmt_blind_inputs as wmt


def checked_file(path, url, digest):
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cached source hash differs: {path.name}")
        return path
    with urlopen(url, timeout=90) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Downloaded source hash differs: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def legal_frame(root):
    refs = json.loads((PACKAGE / "source_replay/controls/expected.json").read_text())
    for source in refs["judgmentbench_sources"].values():
        checked_file(root / source["path"], source["url"], source["sha256"])
    _, labels, annotations, _ = read_judgmentbench(root)
    grouped = defaultdict(list)
    for aid, annotation in annotations.items():
        grouped[annotation["output_id"]].append(aid)
    for aids in grouped.values():
        if len(aids) < 2:
            continue
        atts = [annotations[aid] for aid in aids]
        if len({a["annotator_id"] for a in atts}) != len(aids):
            raise ValueError("Duplicate attorney on the same output")
        tasks = {a["task_id"] for a in atts}
        items = sorted(labels["human"][aids[0]])
        if len(tasks) != 1 or any(set(labels["human"][aid]) != set(items) for aid in aids):
            raise ValueError("Different rubrics on the same output")
        yield tasks.pop(), [[labels["human"][aid][j] for j in items] for aid in aids]


def summary_frame(cache):
    refs = json.loads((PACKAGE / "source_replay/controls/external_freeze.json").read_text())
    path = checked_file(cache / "experts.jsonl", *refs["sources"]["experts.jsonl"])
    experts = {}
    with path.open() as stream:
        for line in stream:
            r = json.loads(line)
            key = r["id"], r["model_id"]
            if key in experts or len(r["expert_annotations"]) != 3:
                raise ValueError("Unexpected summary panel")
            experts[key] = [[e[d] >= 4 for d in DIMS] for e in r["expert_annotations"]]
    _, held = partition(experts)
    for (article, _), bits in experts.items():
        if article in held:
            yield article, bits


def translation_frame(cache):
    wmt.prepare(cache)
    seal = PACKAGE / "wmt/artifacts/reports/wmt_blind_crossover_20261126/corrected/seal.json"
    for unit in wmt.load_units(cache, json.loads(seal.read_text())["split"]["held"]):
        yield unit.doc, unit.experts


def summarize(name, units, expected_shape):
    marginal = [Fraction(), Fraction()]
    disagreement = [Fraction(), Fraction()]
    groups = set()
    outputs = 0
    for group, bits in units:
        groups.add(group)
        outputs += 1
        m, k = len(bits), len(bits[0])
        if m < 2 or k < 1 or any(len(e) != k for e in bits):
            raise ValueError("Invalid expert by criterion matrix")
        # For m raters and x passes, 2*x*(m-x) ordered pairs disagree.
        positive = [sum(col) for col in zip(*bits)]
        for level, counts in enumerate([positive, [sum(all(e) for e in bits)]]):
            marginal[level] += Fraction(sum(counts), m * len(counts))
            offdiag = sum(2 * x * (m - x) for x in counts)
            disagreement[level] += Fraction(offdiag, m * (m - 1) * len(counts))
    if (outputs, len(groups)) != expected_shape:
        raise ValueError(f"Frame changed: {name}, {outputs}, {len(groups)}")
    for j, level in enumerate(["criterion", "verdict"]):
        pass_rate = marginal[j] / outputs
        observed = 1 - disagreement[j] / outputs
        chance = pass_rate ** 2 + (1 - pass_rate) ** 2
        kappa = (observed - chance) / (1 - chance)
        yield [name, outputs, len(groups), level, *[f"{float(x):.9f}" for x in (pass_rate, observed, kappa)]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wmt-cache", type=Path)
    p.add_argument("--summary-cache", type=Path)
    p.add_argument("--legal-root", type=Path, help="Directory containing data/raw/judgmentbench")
    p.add_argument("--write", type=Path, help="Write a new table instead of checking the shipped one")
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["dataset", "outputs", "clusters", "level", "expert_pass", "agreement", "pooled_kappa"])
        writer.writerows(summarize("WMT23", translation_frame(args.wmt_cache or root / "wmt"), (739, 18)))
        writer.writerows(summarize("JudgmentBench", legal_frame(args.legal_root or root), (206, 30)))
        writer.writerows(summarize("SummEval", summary_frame(args.summary_cache or root / "summary"), (1120, 70)))
    table = buf.getvalue()
    if args.write:
        with args.write.open("x") as stream:
            stream.write(table)
    elif table != Path(__file__).with_name("peer_agreement.csv").read_text():
        raise ValueError("Matched-panel results differ from the shipped summary")
    print(table, end="")


if __name__ == "__main__":
    main()
