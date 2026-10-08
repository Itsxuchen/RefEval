"""Base-task composition sensitivity with complete resampling and policy replay.

The bootstrap distribution concerns an empirical composition perturbation. It is
not automatic population coverage for curated tasks, reference uncertainty, or
new raters. RQ1 averages query order analytically; RQ2 conditions on a fixed
31-seed panel and reruns acquisition at each resampled frame's actual budget.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from src.conjunction_label_value import hybrid_queries, query_metrics
from src.conjunction_policy_data import load_datasets
from src.conjunction_policy_study import make_order, replay
from src.conjunction_update_mechanism import expected_curve


ROOT = Path(__file__).resolve().parents[1]
CELLS = ("JB GPT-5.4", "JB GPT-5.4 | binary_only", "DR Gemini 3.1 Pro")
POLICIES = ("random_criterion", "disagreement_then_random")
SHARES = tuple(np.arange(101) / 100)
SEEDS = tuple(range(1, 32))
ALLOCATION_SHARE = .20


def base_order(data: dict) -> list[str]:
    return list(dict.fromkeys(data["base_task_ids"]))


def composition_draws(base_ids: list[str], replicates: int, seed: int,
                      family: str) -> np.ndarray:
    """One shared multinomial multiplicity vector per family and replicate."""
    if replicates < 1 or not base_ids or len(set(base_ids)) != len(base_ids):
        raise ValueError("positive replicates and unique nonempty base IDs required")
    word = int.from_bytes(hashlib.sha256(family.encode()).digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, word]))
    return rng.multinomial(len(base_ids), np.full(len(base_ids), 1 / len(base_ids)), size=replicates)


def resample_frame(data: dict, multiplicities: dict[str, int]) -> dict:
    """Clone whole base tasks while retaining the original relative unit order.

    Each original annotation is followed by its copies. A base/copy identifier
    is shared across every annotation belonging to that copied base task; rater
    identities remain the observed raters rather than new independent people.
    """
    bases = base_order(data)
    if set(multiplicities) != set(bases):
        raise ValueError("multiplicities must cover exactly the source base tasks")
    if any(not isinstance(v, (int, np.integer)) or v < 0 for v in multiplicities.values()):
        raise ValueError("multiplicities must be nonnegative integers")
    if not sum(multiplicities.values()):
        raise ValueError("empty resampled frame")
    ti = np.asarray(data["task_index"])
    n = len(data["task_ids"])
    k = np.bincount(ti, minlength=n)
    if np.any(k == 0) or not np.all(np.diff(ti) >= 0):
        raise ValueError("nonempty contiguous source units required")
    offsets = np.r_[0, np.cumsum(k)]
    parents, copies = [], []
    for unit, base in enumerate(data["base_task_ids"]):
        for copy in range(int(multiplicities[base])):
            parents.append(unit)
            copies.append(copy)
    parent = np.asarray(parents, dtype=int)
    slots = np.concatenate([np.arange(offsets[t], offsets[t + 1]) for t in parent])
    clone_k = k[parent]
    clone_ti = np.repeat(np.arange(len(parent)), clone_k)
    clone = {**data}
    for field in ("gold", "prediction"):
        clone[field] = np.asarray(data[field])[slots].copy()
    clone["secondary_predictions"] = np.asarray(data["secondary_predictions"])[:, slots].copy()
    clone["task_index"] = clone_ti
    for field in ("criterion_ids", "category"):
        clone[field] = [data[field][i] for i in slots]
    clone["task_ids"] = [f"{data['task_ids'][t]}::clone_{c}" for t, c in zip(parent, copies)]
    clone["base_task_ids"] = [f"{data['base_task_ids'][t]}::clone_{c}" for t, c in zip(parent, copies)]
    clone["output_ids"] = [f"{data['output_ids'][t]}::clone_{c}" for t, c in zip(parent, copies)]
    clone["rater_ids"] = [data["rater_ids"][t] for t in parent]
    clone["source_unit_index"] = parent
    clone["source_base_task_ids"] = [data["base_task_ids"][t] for t in parent]
    clone["clone_copy_index"] = np.asarray(copies, dtype=int)
    # These are checked on every reconstructed frame, not just in toy tests.
    expected_n = sum(multiplicities[b] for b in data["base_task_ids"])
    expected_m = sum(int(kk) * multiplicities[b] for kk, b in zip(k, data["base_task_ids"]))
    assert len(parent) == expected_n and len(slots) == expected_m
    assert len(set(clone["task_ids"])) == len(parent)
    assert len(set(clone["base_task_ids"])) == sum(multiplicities.values())
    assert np.array_equal(np.bincount(clone_ti), clone_k)
    assert np.all(np.diff(parent) >= 0)
    for field in ("gold", "prediction"):
        assert np.array_equal(clone[field], np.asarray(data[field])[slots])
    assert np.array_equal(clone["secondary_predictions"], np.asarray(data["secondary_predictions"])[:, slots])
    for field in ("gold", "prediction", "task_index", "secondary_predictions"):
        clone[field].flags.writeable = False
    return clone


def query_indices(data: dict, policy: str, seed: int) -> np.ndarray:
    """Vectorized query transcript, equivalent to existing release-order replay.

    The SC controller reads successive answers until the first FAIL per unit.
    The vector expression identifies exactly that stopping position; it neither
    changes the pre-answer candidate ordering nor prioritizes by hidden labels.
    """
    if policy not in ("random_criterion", "sc_judge"):
        raise ValueError("only the two allocation policies are supported")
    order = make_order(data, policy, "release", seed)
    if policy == "random_criterion":
        return order
    position = np.arange(len(order))
    tasks = np.asarray(data["task_index"])[order]
    first_fail = np.full(len(data["task_ids"]), len(order), dtype=int)
    np.minimum.at(first_fail, tasks, np.where(np.asarray(data["gold"])[order] == 0, position, len(order)))
    return order[position <= first_fail[tasks]]


def validate_query_equivalence(data: dict, seeds=SEEDS) -> int:
    checked = 0
    for seed in seeds:
        for policy in ("random_criterion", "sc_judge"):
            np.testing.assert_array_equal(query_indices(data, policy, seed),
                                          replay(data, policy, "release", seed)["indices"])
            checked += 1
    return checked


def allocation_panel(data: dict, seeds=SEEDS) -> tuple[list[dict], dict]:
    """Replay the fixed seed panel under the resampled frame's 20% query cap."""
    if not seeds:
        raise ValueError("at least one query-order seed required")
    m, n = len(data["gold"]), len(data["task_ids"])
    cap = int(round(ALLOCATION_SHARE * m))
    rows = []
    for seed in seeds:
        srs = query_indices(data, "random_criterion", seed)[:cap]
        sc = {"indices": query_indices(data, "sc_judge", seed)}
        A, R = hybrid_queries(data, cap, "release", seed, sc)
        random_metrics = query_metrics(data, srs)
        mix_metrics = query_metrics(data, np.r_[A, R])
        assert random_metrics["actual_queries"] == mix_metrics["actual_queries"] == cap
        row = dict(query_seed=int(seed), budget_share=ALLOCATION_SHARE,
                   budget_cap=cap, n_units=n, n_criteria=m,
                   stageA_queries=len(A), stageR_queries=len(R))
        for field in ("certified_tasks", "residual_errors"):
            row["srs_" + field] = random_metrics[field]
            row["mix_" + field] = mix_metrics[field]
            delta = mix_metrics[field] - random_metrics[field]
            row[field + "_delta"] = delta
            row[field + "_delta_per100"] = 100 * delta / n
        rows.append(row)
    mean = dict(budget_share=ALLOCATION_SHARE, budget_cap=cap, n_units=n,
                n_criteria=m, query_seeds=len(seeds))
    for key in rows[0]:
        if key not in ("query_seed", "budget_share", "budget_cap", "n_units", "n_criteria"):
            mean[key + "_mean"] = float(np.mean([row[key] for row in rows]))
    return rows, mean


def curve_summary(original: np.ndarray, draws: np.ndarray) -> dict:
    """Percentile sensitivity ranges and a centered sup-deviation grid band."""
    original, draws = np.asarray(original, float), np.asarray(draws, float)
    if draws.ndim != 2 or original.shape != (draws.shape[1],) or not len(draws):
        raise ValueError("one original grid and a nonempty replicate-by-grid matrix required")
    point = np.quantile(draws, [.025, .5, .975], axis=0)
    sup = np.max(np.abs(draws - original[None, :]), axis=1)
    radius = float(np.quantile(sup, .95))
    return dict(original=original.tolist(), pointwise_q025=point[0].tolist(),
                pointwise_median=point[1].tolist(), pointwise_q975=point[2].tolist(),
                sup_deviation_q95=radius, simultaneous_grid_lower=(original - radius).tolist(),
                simultaneous_grid_upper=(original + radius).tolist(),
                empirical_band_containment=float(np.mean(sup <= radius)))


def peak_record(curve: list[dict]) -> dict:
    """Recompute extrema on every full curve; report plateaus explicitly."""
    value = np.asarray([x["expected_delta"] for x in curve])
    hi, lo = float(value.max()), float(value.min())
    mx = np.flatnonzero(np.isclose(value, hi, rtol=0, atol=1e-12))
    mn = np.flatnonzero(np.isclose(value, lo, rtol=0, atol=1e-12))
    n = curve[0]["n_units"]
    return dict(max_delta=hi, max_delta_per100=100 * hi / n,
                max_budget_first=curve[int(mx[0])]["budget_share"],
                max_budget_last=curve[int(mx[-1])]["budget_share"], max_tied_grid_points=len(mx),
                min_delta=lo, min_delta_per100=100 * lo / n,
                min_budget_first=curve[int(mn[0])]["budget_share"],
                min_budget_last=curve[int(mn[-1])]["budget_share"], min_tied_grid_points=len(mn))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_output_path(root: Path, output: Path) -> Path:
    """Reject protected trees and their ancestors after resolving symlinks."""
    root, output = Path(root).resolve(), Path(output).resolve()
    protected = [root / name for name in (
        "src", "data", "tests", "context", "docs", ".git", "validation", "figures",
        "artifacts/expected", "artifacts/figures", "artifacts/validation",
        "artifacts/reports/conjunction_policy_study", "artifacts/reports/conjunction_label_value",
        "artifacts/reports/conjunction_mechanism",
    )]
    if root.is_relative_to(output) or any(
        output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected
    ):
        raise ValueError("output overlaps a protected input/source/reference tree or its ancestor")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty; preserve earlier runs")
    return output


class CsvSink:
    def __init__(self, stream):
        self.stream, self.writer, self.rows = stream, None, 0

    def append(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.rows += 1


def run(root: Path, output: Path, protocol: Path, replicates=399, seed=20261008) -> dict:
    """Write only the caller-selected new output folder; load data from root."""
    root, protocol = Path(root).resolve(), Path(protocol).resolve()
    output = validate_output_path(root, output)
    if not protocol.is_file():
        raise ValueError("a saved frozen protocol is required before generating results")
    if replicates < 1:
        raise ValueError("positive replicate count required")
    started = time.perf_counter()
    all_data = load_datasets(root)
    datasets = {name: all_data[name] for name in CELLS}
    assert base_order(datasets[CELLS[0]]) == base_order(datasets[CELLS[1]])
    sources = [Path(__file__), root / "src/conjunction_policy_data.py",
               root / "src/conjunction_policy_study.py", root / "src/conjunction_label_value.py",
               root / "src/conjunction_update_mechanism.py", protocol]
    preserved = {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): _sha(p) for p in sources}
    for data in datasets.values():
        preserved.update(data["source_hashes"])
    families = {name: "JB" if name.startswith("JB ") else "DR" for name in CELLS}
    draws, base_ids = {}, {}
    for name, data in datasets.items():
        family = families[name]
        if family not in draws:
            base_ids[family] = base_order(data)
            draws[family] = composition_draws(base_ids[family], replicates, seed, family)
    equivalent_queries_checked = 0
    for name, data in datasets.items():
        equivalent_queries_checked += validate_query_equivalence(data)
        family = families[name]
        first_clone = resample_frame(data, dict(zip(base_ids[family], draws[family][0])))
        equivalent_queries_checked += validate_query_equivalence(first_clone, seeds=(1, 7, 31))
    output.mkdir(parents=True, exist_ok=True)
    matrices, originals, peaks, allocation = defaultdict(list), {}, defaultdict(list), defaultdict(list)
    allocation_original = {}
    with ExitStack() as stack:
        filenames = ("multiplicities.csv.gz", "rq1_curves.csv.gz", "rq1_peaks.csv.gz",
                     "rq2_seed_pairs.csv.gz", "rq2_frame_means.csv.gz")
        sinks = {name: CsvSink(stack.enter_context(gzip.open(output / name, "wt", newline="", encoding="utf-8"))) for name in filenames}
        for family, matrix in draws.items():
            for rep, counts in enumerate(matrix):
                assert int(counts.sum()) == len(base_ids[family])
                for base, count in zip(base_ids[family], counts):
                    sinks["multiplicities.csv.gz"].append(dict(family=family, replicate=rep, base_task_id=base, multiplicity=int(count)))
        for name, original in datasets.items():
            family = families[name]
            for rep in range(-1, replicates):
                data = original if rep == -1 else resample_frame(original, dict(zip(base_ids[family], draws[family][rep])))
                n, m = len(data["task_ids"]), len(data["gold"])
                for policy in POLICIES:
                    curve = [dict(dataset=name, replicate=rep, n_units=n, n_criteria=m, **r)
                             for r in expected_curve(data, policy, SHARES)]
                    values = np.asarray([r["expected_delta"] for r in curve])
                    for row in curve:
                        for key in ("expected_introduced_fp", "expected_delayed_ff", "expected_delta"):
                            row[key + "_per100"] = 100 * row[key] / n
                        sinks["rq1_curves.csv.gz"].append(row)
                    extrema = dict(dataset=name, replicate=rep, policy=policy, n_units=n, n_criteria=m, **peak_record(curve))
                    sinks["rq1_peaks.csv.gz"].append(extrema)
                    if rep == -1:
                        originals[(name, policy)] = dict(count=values, per100=100 * values / n,
                                                        n_units=n, n_criteria=m, extrema=extrema)
                    else:
                        matrices[(name, policy, "count")].append(values)
                        matrices[(name, policy, "per100")].append(100 * values / n)
                        peaks[(name, policy)].append(extrema)
                pairs, mean = allocation_panel(data)
                for pair in pairs:
                    sinks["rq2_seed_pairs.csv.gz"].append(dict(dataset=name, replicate=rep, **pair))
                sinks["rq2_frame_means.csv.gz"].append(dict(dataset=name, replicate=rep, **mean))
                if rep == -1:
                    allocation_original[name] = mean
                else:
                    allocation[name].append(mean)
                if rep == -1 or (rep + 1) % 25 == 0 or rep == replicates - 1:
                    print(f"{name}: replicate {rep + 1}/{replicates}; elapsed {time.perf_counter() - started:.1f}s", flush=True)
        output_counts = {name: sink.rows for name, sink in sinks.items()}
    summary = dict(kind="base-task composition bootstrap sensitivity", replicates=replicates,
                   budget_shares=list(SHARES), rq1=[], rq2=[])
    for (name, policy), original in originals.items():
        item = dict(dataset=name, policy=policy, n_units=original["n_units"], n_criteria=original["n_criteria"],
                    original_extrema=original["extrema"])
        for unit in ("count", "per100"):
            item[unit] = curve_summary(original[unit], np.asarray(matrices[(name, policy, unit)]))
        item["peak_replicate_quantiles"] = {
            key: dict(zip(("q025", "median", "q975"), map(float, np.quantile([r[key] for r in peaks[(name, policy)]], [.025, .5, .975]))))
            for key in ("max_delta", "max_delta_per100", "max_budget_first", "min_delta", "min_delta_per100", "min_budget_first")}
        summary["rq1"].append(item)
    for name, rows in allocation.items():
        fields = ("certified_tasks_delta_mean", "residual_errors_delta_mean",
                  "certified_tasks_delta_per100_mean", "residual_errors_delta_per100_mean")
        summary["rq2"].append(dict(dataset=name, original=allocation_original[name],
            quantiles={key: dict(zip(("q025", "median", "q975"), map(float, np.quantile([r[key] for r in rows], [.025, .5, .975])))) for key in fields},
            cert_gain_replicates=sum(r["certified_tasks_delta_mean"] > 0 for r in rows),
            error_reduction_replicates=sum(r["residual_errors_delta_mean"] < 0 for r in rows),
            error_increase_replicates=sum(r["residual_errors_delta_mean"] > 0 for r in rows)))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert all(_sha(root / path if not Path(path).is_absolute() else Path(path)) == value for path, value in preserved.items())
    manifest = dict(replicates=replicates, root_seed=seed, cells=list(CELLS), query_seed_panel=list(SEEDS),
        family_base_order=base_ids, shared_JB_multiplicities=True, rq1_policies=list(POLICIES),
        rq1_budget_shares=list(SHARES), rq2_budget_share=ALLOCATION_SHARE,
        bootstrap_interpretation="Empirical base-task composition sensitivity. Curated source tasks are not established IID population draws; no new-rater or reference-validity CI.",
        query_order_interpretation="RQ1 uses exact conditional expectations over randomized priorities. RQ2 reruns a fixed 31-seed panel in every frame; composition quantiles do not isolate or estimate its finite-panel Monte Carlo error.",
        band_interpretation="Approximate centered bootstrap sup-absolute-deviation 95% bands separately for each cell/policy over the fixed 101-point grid. Not joint across six curves, exact design coverage, an anytime band, or continuous-budget coverage. Per100-unit effects are the primary scale; counts retained separately.",
        peak_interpretation="Extrema recalculated per replicate across the full grid; plateau bounds and first maximizer retained. Peak-location quantiles describe empirical sensitivity, not regularity-guaranteed argmax CIs.",
        quantile_method="linear (NumPy default)",
        resampling_order="Original release units in original relative order, each repeated by its base multiplicity; shared base/copy identifiers, original rater identities, and all primary/secondary/reference bits retained. Actual caps recomputed from each new criterion count.",
        source_hashes=preserved, preserved_sources_unchanged=True, output_rows=output_counts,
        vectorized_query_equality_checks=equivalent_queries_checked,
        elapsed_seconds=time.perf_counter() - started)
    manifest["output_sha256"] = {p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--replicates", type=int, default=399)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    output = args.output or args.root / "artifacts/reports/conjunction_robustness/cluster"
    protocol = args.protocol or args.root / "context/conjunction_robustness_protocol.md"
    print(json.dumps(run(args.root, output, protocol, args.replicates, args.seed), indent=2))


if __name__ == "__main__":
    main()
