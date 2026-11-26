"""Rotate all distinct attorneys per output on the historical full-target audit paths."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.conjunction_policy_data import read_csv, read_judgmentbench
from src.conjunction_policy_study import make_order
from src.cross_rater_replay import bootstrap_rates, write_csv

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/reports/conjunction_robustness/reference"
CROSSED = ROOT / "artifacts/reports/cross_rater_20261125"
OUT = ROOT / "artifacts/reports/summeval_cross_scorer_20261126"
POLICY = "disagreement_then_random"
SHARES = np.arange(0, 101, 10) / 100.0
METRICS = ("early", "own_endorsed", "other_endorsed", "own_eager_error",
           "own_gated_error", "other_eager_error", "other_gated_error",
           "own_gated_minus_eager", "other_gated_minus_eager", "other_minus_own_delta")
BOOTSTRAP_SEED = 20261125


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(paths, pairing, historical):
    archive = json.loads((REFERENCE / "manifest.json").read_text())
    expected = archive["data_hashes"]["JB GPT-5.4"]
    for path in paths:
        if sha(path) != expected[str(path.relative_to(ROOT))]:
            raise ValueError(f"Raw input changed: {path}")
    if sha(pairing) != archive["output_hashes"]["pairing.csv"]:
        raise ValueError("Published anchored output list changed")
    receipt = json.loads((CROSSED / "verification.json").read_text())
    if sha(historical) != receipt["output_sha256"][historical.name]:
        raise ValueError("Historical cross-scoring controls changed")
    if sha(ROOT / "src/conjunction_policy_study.py") != receipt["source_sha256"]["conjunction_policy_study.py"]:
        raise ValueError("Historical ranking implementation changed")


def load_panel():
    items, labels, annotations, paths = read_judgmentbench(ROOT)
    pairing = REFERENCE / "pairing.csv"
    control = CROSSED / "cluster_inputs_full_gpt54_half.csv"
    verify_sources(paths, pairing, control)
    panel = read_csv(pairing)
    members = defaultdict(list)
    for aid, row in annotations.items():
        members[row["output_id"]].append(aid)
    eligible = {oid for oid, aids in members.items() if len({annotations[a]["annotator_id"] for a in aids}) > 1}
    if len(panel) != len(eligible) or {r["output_id"] for r in panel} != eligible:
        raise ValueError("Unmatched repeated-output frame")
    rows = []
    for row in panel:
        oid, anchor = row["output_id"], row["anchor_annotation_id"]
        aids = sorted(members[oid])
        if len({annotations[a]["annotator_id"] for a in aids}) != len(aids):
            raise ValueError(f"Repeated ratings by the same attorney on {oid}")
        if any(annotations[a]["task_id"] != row["base_task_id"] for a in aids):
            raise ValueError(f"Incompatible base task on {oid}")
        rubric = sorted(labels["human"][anchor], key=lambda i: int(items[i]["item_order"]))
        if any(set(labels["human"][a]) != set(rubric) for a in aids):
            raise ValueError(f"Incompatible rubric on {oid}")
        bits = np.array([[labels["human"][a][i] for i in rubric] for a in aids], dtype=bool)
        judges = np.array([[labels[name][anchor][i] for i in rubric]
                           for name in ("gpt_5_4", "gpt_5_4_mini")], dtype=bool)
        anchors = [aids.index(row[role + "_annotation_id"]) for role in ("A", "B")]
        rows.append((row["base_task_id"], oid, bits, judges, anchors))
    return rows, (paths + [pairing, control])


def actions(prediction, ratings, ranks, cap):
    """Compute actions for each order and auditor before introducing any scorer."""
    seen = ranks[:, None, :] < cap
    observed = ratings[None, :, :]
    eager = np.where(seen, observed, prediction).all(axis=2)
    gated = np.all(seen & observed, axis=2) | (
        prediction.all() & ~np.any(seen & ~observed, axis=2))
    if np.any(gated & ~eager):
        raise AssertionError("Unexpected gated-only PASS")
    return eager, gated


def per_output(eager, gated, truth):
    """Uniform auditors, then uniform other attorneys conditional on the auditor."""
    other_pass = (truth.sum() - truth.astype(int)) / (len(truth) - 1)
    early = eager & ~gated
    own_eager, own_gated = eager != truth, gated != truth
    other_eager = np.where(eager, 1 - other_pass, other_pass)
    other_gated = np.where(gated, 1 - other_pass, other_pass)
    own_delta = own_gated.astype(int) - own_eager.astype(int)
    other_delta = other_gated - other_eager
    np.testing.assert_allclose(own_delta, early * (2 * truth.astype(int) - 1))
    np.testing.assert_allclose(other_delta, early * (2 * other_pass - 1), atol=1e-15)
    if np.any(np.mean(other_delta - own_delta, axis=1) > 1e-14):
        raise AssertionError("Symmetric own-to-other shift unexpectedly positive")
    values = (early, early & truth, early * other_pass, own_eager, own_gated,
              other_eager, other_gated, own_delta, other_delta, other_delta - own_delta)
    return np.array([x.mean() for x in values])


def cross_check(group_controls):
    """Historical FP and FF totals give independent checks for all 31 orders in aggregate."""
    historical = read_csv(CROSSED / "cluster_inputs_full_gpt54_half.csv")
    checked = 0
    for row in historical:
        if row["policy"] != POLICY or row["dataset"] != "JB GPT-5.4" or float(row["budget_share"]) != 0.5:
            continue
        key = int(row["cluster_index"]), row["audit"], row["score"]
        expected = [int(row[k + "_seed_sum"]) for k in ("eager_FP", "eager_FF", "gated_FP", "gated_FF")]
        np.testing.assert_array_equal(group_controls[key], expected)
        checked += 1
    if checked != 120:
        raise ValueError(f"Historical cluster control coverage {checked}")
    return checked


def run(out):
    panel, paths = load_panel()
    output_ids = [x[1] for x in panel]
    n_raters = [len(x[2]) for x in panel]
    lengths = [x[3].shape[1] for x in panel]
    judges = np.concatenate([x[3] for x in panel], axis=1)
    frame = dict(task_ids=output_ids, task_index=np.repeat(np.arange(len(panel)), lengths),
                 prediction=judges[0], secondary_predictions=judges[1:])
    orders = [make_order(frame, POLICY, "release", seed) for seed in range(1, 32)]
    ranks = np.argsort(orders, axis=1)
    group_names = sorted({x[0] for x in panel})
    groups = {name: i for i, name in enumerate(group_names)}
    n_groups = len(groups)
    sizes = np.bincount([groups[x[0]] for x in panel], minlength=n_groups)
    n_outputs, n_items = len(panel), sum(lengths)
    if (n_outputs, n_items, n_groups) != (206, 3098, 30):
        raise ValueError("Unexpected historical frame")
    caps = [round(b * n_items) for b in SHARES]
    contributions = np.zeros((len(caps), n_groups, len(METRICS)))
    ever_early = np.zeros(len(caps), dtype=int)
    controls = defaultdict(lambda: np.zeros(4, dtype=int))
    pair_details = []
    start = 0
    for base, oid, ratings, models, anchors in panel:
        stop = start + models.shape[1]
        truth = ratings.all(axis=1)
        for c, cap in enumerate(caps):
            eager, gated = actions(models[0], ratings, ranks[:, start:stop], cap)
            if cap in (0, n_items):
                expected = truth[None, :] if cap == n_items else models[0].all()
                if np.any(eager != expected) or np.any(gated != expected):
                    raise AssertionError(f"Endpoint violation on {oid}")
            ever_early[c] += np.any(eager & ~gated)
            contributions[c, groups[base]] += per_output(eager, gated, truth)
            if cap == n_items // 2:
                for auditor, own in enumerate(truth):
                    early = eager[:, auditor] & ~gated[:, auditor]
                    for scorer, other in enumerate(truth):
                        if scorer == auditor:
                            continue
                        pair_details.append(dict(base_task=base, output=oid,
                            auditor_slot=auditor + 1, scorer_slot=scorer + 1,
                            within_output_weight=1/(len(truth)*(len(truth)-1)),
                            n_orders=len(early), early_sum=int(early.sum()),
                            own_pass=int(own), other_pass=int(other),
                            own_eager_errors=int(np.sum(eager[:, auditor] != own)),
                            own_gated_errors=int(np.sum(gated[:, auditor] != own)),
                            other_eager_errors=int(np.sum(eager[:, auditor] != other)),
                            other_gated_errors=int(np.sum(gated[:, auditor] != other))))
                for ai, a in zip(("A", "B"), anchors):
                    for si, s in zip(("A", "B"), anchors):
                        e, g, y = eager[:, a], gated[:, a], truth[s]
                        controls[groups[base] + 1, ai, si] += [
                            sum(e & ~y), sum(~e & y), sum(g & ~y), sum(~g & y)]
        start = stop
    n_checked = cross_check(controls)
    draws = np.random.default_rng(BOOTSTRAP_SEED).multinomial(n_groups, np.ones(n_groups)/n_groups, size=1999)
    curves, clusters = [], []
    for c, cap in enumerate(caps):
        matrix = contributions[c]
        rates, lower, upper = bootstrap_rates(matrix, sizes, draws)
        row = dict(budget_share=SHARES[c], actual_queries=cap,
                   mean_early_outputs=matrix[:, 0].sum(),
                   distinct_outputs_ever_early=int(ever_early[c]))
        for j, name in enumerate(METRICS):
            row[name + "_per100"] = rates[j]
            row[name + "_lo"] = lower[j]
            row[name + "_hi"] = upper[j]
        for j, name in [(1, "own"), (2, "other")]:
            row[name + "_endorsement_pct"] = 100 * rates[j] / rates[0] if rates[0] else ""
        curves.append(row)
        for i, name in enumerate(group_names):
            clusters.append(dict(budget_share=SHARES[c], base_task=name, n_outputs=sizes[i],
                                 **dict(zip(METRICS, matrix[i]))))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "jb_summary.csv", curves)
    write_csv(out / "jb_cluster_contributions.csv", clusters)
    write_csv(out / "jb_half_pairs.csv", pair_details)
    receipt = dict(output_weighting="equal by output, by auditor within output and by peer within auditor",
                   n_outputs=n_outputs, n_base_tasks=n_groups, n_criteria=n_items,
                   rater_histogram=dict(Counter(n_raters)), n_directed_pairs=sum(r*(r-1) for r in n_raters),
                   orders=list(range(1,32)), n_cluster_draws=1999, cluster_seed=BOOTSTRAP_SEED,
                   historical_four_count_cells_verified=n_checked, inputs_sha256={
                       str(p.relative_to(ROOT)): sha(p) for p in paths},
                   code_sha256={p:sha(ROOT / "src" / p) for p in (
                       Path(__file__).name, "conjunction_policy_study.py", "conjunction_policy_data.py",
                       "cross_rater_replay.py")},
                   outputs_sha256={p:sha(out / p) for p in ("jb_summary.csv", "jb_cluster_contributions.csv", "jb_half_pairs.csv")})
    (out / "jb_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()
    result = next(row for row in run(arguments.out) if row["budget_share"] == .5)
    print(json.dumps(result, indent=2))
