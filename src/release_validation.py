"""Compare scientific outputs with frozen references and inspect MILP witnesses.

Timestamps, provenance digests and nonunique optimal assignments are not numeric
targets. Scientific CSV fields are compared in full. Small smoke runs compare a
declared subset and cannot be presented as complete reproduction.
"""
from __future__ import annotations

import csv
import gzip
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OMIT = {"created_utc", "generated_utc", "seconds", "elapsed_seconds", "source_hashes",
        "source_sha256", "data_hashes", "provenance", "prediction_provenance",
        "solver_certificates"}


def assert_equal(actual, expected, location="root") -> int:
    """Compare a scientific payload, retaining identities and finite counts."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        ak, ek = set(actual) - OMIT, set(expected) - OMIT
        if ak != ek:
            raise AssertionError(f"{location}: key mismatch {sorted(ak ^ ek)}")
        return sum(assert_equal(actual[k], expected[k], f"{location}.{k}") for k in sorted(ak))
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{location}: length {len(actual)} != {len(expected)}")
        return sum(assert_equal(a, b, f"{location}[{i}]") for i, (a, b) in enumerate(zip(actual, expected)))
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(actual, int) and isinstance(expected, int):
            okay = actual == expected
        else:
            okay = math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8)
        if not okay:
            raise AssertionError(f"{location}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{location}: {actual!r} != {expected!r}")
    return 1


def _csv(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", newline="", encoding="utf-8")


def _value(value: str):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value


def _row(actual, expected, number):
    if actual == expected:
        return len(actual)
    if set(actual) != set(expected):
        raise AssertionError(f"CSV row {number}: columns differ")
    for field in actual:
        if actual[field] != expected[field]:
            assert_equal(_value(actual[field]), _value(expected[field]), f"CSV row {number}/{field}")
    return len(actual)


def compare_csv(actual: Path, expected: Path, *, subset: bool = False,
                key_fields: tuple[str, ...] = ()) -> dict:
    count = fields = 0
    with _csv(actual) as af, _csv(expected) as ef:
        ar, er = csv.DictReader(af), csv.DictReader(ef)
        if ar.fieldnames != er.fieldnames:
            raise AssertionError(f"CSV columns differ: {actual.name}")
        if subset:
            selected = {}
            for row in ar:
                key = tuple(row[k] for k in key_fields)
                if key in selected:
                    raise AssertionError("duplicate subset row identity")
                selected[key] = row
            for row in er:
                key = tuple(row[k] for k in key_fields)
                if key in selected:
                    fields += _row(selected.pop(key), row, count)
                    count += 1
            if selected:
                raise AssertionError(f"{len(selected)} smoke rows absent from expected table")
        else:
            for count, (a, e) in enumerate(itertools.zip_longest(ar, er), 1):
                if a is None or e is None:
                    raise AssertionError(f"CSV row count differs: {actual.name}")
                fields += _row(a, e, count)
    if count == 0:
        raise AssertionError(f"empty comparison: {actual.name}")
    return {"rows": count, "fields": fields, "scope": "subset" if subset else "all rows"}


def verify_disclosure_witnesses(result: dict) -> dict:
    """Literal independent assignment checks, without calling an optimizer."""
    source = ROOT / "data/processed/public_judge_predictions.csv"
    runs = defaultdict(lambda: defaultdict(list))
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            runs[row["run"]][f"{row['domain']}/{row['task_id']}"].append(row)
    checked = links = margins = 0
    for run in result["ruverbench"]:
        tasks = runs[run["run"]]
        for rows in tasks.values():
            rows.sort(key=lambda r: int(r["criterion_index"]))
        ids = sorted(tasks)
        orders = {r["order"]: r["task_order"] for r in run["orders"]}
        truth = sum(all(r["gold"] == "True" for r in rows) for rows in tasks.values())
        by_k = {k: [r for t in ids if len(tasks[t]) == k for r in tasks[t]]
                for k in {len(rows) for rows in tasks.values()}}
        for release in [run["baseline"], run["category_disclosure"], *run["task_link_rows"]]:
            selected = set(orders.get(release["order"], [])[:math.floor(release["requested_fraction"] * len(ids))])
            category = release["stage"] != "D0_k_confusion"
            totals = {"lower": 0, "upper": 0}
            seen = set()
            for cert in release["solver_certificates"]:
                k, endpoint = cert["k"], cert["endpoint"]
                if (k, endpoint) in seen:
                    raise AssertionError("duplicate witness stratum")
                seen.add((k, endpoint))
                bits = [int(v) for v in cert["reference_label_witness_bits"]]
                if len(bits) != len(by_k[k]) or set(bits) - {0, 1}:
                    raise AssertionError("malformed reference witness")
                observed, expected = defaultdict(int), defaultdict(int)
                for row, bit in zip(by_k[k], bits):
                    group = (row["prediction"], row["category"] if category else "all")
                    observed[group] += bit
                    expected[group] += row["gold"] == "True"
                    if f"{row['domain']}/{row['task_id']}" in selected:
                        links += 1
                        if bit != int(row["gold"] == "True"):
                            raise AssertionError("witness violates restored reference link")
                if observed != expected:
                    raise AssertionError("witness violates disclosed marginal counts")
                margins += len(observed)
                passes = [int(all(bits[i:i+k])) for i in range(0, len(bits), k)]
                if "".join(map(str, passes)) != cert["task_pass_witness_bits"] or sum(passes) != cert["pass_count"]:
                    raise AssertionError("witness conjunction mismatch")
                if cert["constraint_violation"] > 1e-8 or cert["objective_gap"] > 1e-7:
                    raise AssertionError("solver did not certify its endpoint")
                totals[endpoint] += sum(passes)
                checked += 1
            if seen != {(k, e) for k in by_k for e in totals}:
                raise AssertionError("incomplete witness coverage")
            assert_equal([100 * totals[e] / len(ids) for e in ("lower", "upper")], release["pass_rate_bounds_pp"])
            if not totals["lower"] <= truth <= totals["upper"]:
                raise AssertionError("true assignment excluded")
    return {"endpoint_witnesses": checked, "restored_links_checked": links,
            "marginal_constraints_checked": margins,
            "scope": "Direct endpoint feasibility; global optimality additionally uses the original solver's closed objective bounds and exhaustive small-world tests."}


def verify_outputs(directory: Path, expected: Path, mode: str) -> dict:
    full = mode == "full"
    csv_checks = {}
    common_keys = ("dataset", "task_order", "policy", "seed", "budget_share")
    tables = [
        ("conjunction_policy_study/budget_curves.csv.gz", common_keys),
        ("conjunction_label_value/budget_estimation_rows.csv.gz", common_keys),
        ("conjunction_label_value/whole_task_sampling.csv.gz", ("dataset", "seed", "requested_task_share")),
    ]
    if full:
        tables += [("conjunction_policy_study/leave_one_base_task_out.csv.gz", ()),
                   ("conjunction_label_value/budget_estimation_summary.csv", ()),
                   ("conjunction_policy_study/canonical_20pct.csv", ())]
    for relative, keys in tables:
        csv_checks[relative] = compare_csv(directory / relative, expected / relative, subset=not full, key_fields=keys)
    json_checks = {}
    flow = "conjunction_policy_study/error_flow.json"
    json_checks[flow] = {"scientific_leaves_checked": assert_equal(
        json.loads((directory / flow).read_text()), json.loads((expected / flow).read_text()), flow)}
    if full:
        for relative in ("conjunction_policy_study/study.json", "conjunction_policy_study/analysis_summary.json",
                         "conjunction_label_value/manifest.json",
                         "information_disclosure_replay.json"):
            a = json.loads((directory / relative).read_text())
            e = json.loads((expected / relative).read_text())
            json_checks[relative] = {"scientific_leaves_checked": assert_equal(a, e, relative)}
    disclosure = json.loads((directory / "information_disclosure_replay.json").read_text())
    if not full:
        reference = json.loads((expected / "information_disclosure_replay.json").read_text())
        leaves = assert_equal(disclosure["population"], reference["population"])
        for a, e in zip(disclosure["ruverbench"], reference["ruverbench"]):
            excluded = {"task_link_curve", "task_link_rows", "orders"}
            leaves += assert_equal({k:v for k,v in a.items() if k not in excluded},
                                   {k:v for k,v in e.items() if k not in excluded})
            leaves += assert_equal(a["orders"], e["orders"][:len(a["orders"])])
            orders = {row["order"] for row in a["orders"]}
            leaves += assert_equal(a["task_link_rows"], [row for row in e["task_link_rows"] if row["order"] in orders])
        a, e = disclosure["judgmentbench"], reference["judgmentbench"]
        leaves += assert_equal(a["n_tasks"], e["n_tasks"])
        leaves += assert_equal(a["n_blocks"], e["n_blocks"])
        leaves += assert_equal(a["orders"], e["orders"][:len(a["orders"])])
        for ac, ec in zip(a["comparisons"], e["comparisons"]):
            excluded = {"pair_link_curve", "pair_link_rows"}
            leaves += assert_equal({k:v for k,v in ac.items() if k not in excluded},
                                   {k:v for k,v in ec.items() if k not in excluded})
            orders = {row["order"] for row in ac["pair_link_rows"]}
            leaves += assert_equal(ac["pair_link_rows"], [row for row in ec["pair_link_rows"] if row["order"] in orders])
        json_checks["information_disclosure_replay.json"] = {"scientific_leaves_checked": leaves, "scope": "smoke order subset"}
    return {"passed": True, "mode": mode, "complete_reproduction": full,
            "csv_checks": csv_checks, "json_checks": json_checks,
            "disclosure_witnesses": verify_disclosure_witnesses(disclosure),
            "comparison_tolerance": {"absolute": 1e-8, "relative": 1e-9},
            "scope": "All frozen scientific tables and selected JSON payloads" if full else "Bounded real-data subsets and witness checks; not the complete experiment grid"}
