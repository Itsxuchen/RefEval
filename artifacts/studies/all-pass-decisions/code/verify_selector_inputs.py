"""Recompute development selections using anonymous sufficient counts and the standard library.

For historical folds, hold one block out and pool the other four to fit each
selector. For SummEval, fit each representation using development counts.
The archived model forecasts and decision seal serve only as verification targets.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from math import fsum, isclose
from pathlib import Path


INPUTS = Path(__file__).resolve().parents[1] / "inputs"
RULES = ("criterion", "empirical_task", "independent_task")
COSTS = ((1, 1), (3, 1), (1, 3))
ERRORS = ("criterion_false_pass", "criterion_false_fail", "task_false_pass", "task_false_fail")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_number(value, reference, field):
    check(isclose(value, float(reference), abs_tol=1e-10, rel_tol=1e-11),
          f"{field} differs: computed {value}, archived {reference}")


def read_counts(filename, collection):
    data = json.loads((INPUTS / filename).read_text())
    check(data["kt_columns"] == ["criteria", "reference_failures", "number_of_units"],
          "unexpected reference histogram schema")
    check(tuple(data["error_columns"]) == ERRORS, "unexpected error count schema")
    return data[collection]


def reference_table(filename):
    with (INPUTS / filename).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = {(r["frame"], int(r["fold"]), r["cost"], r["model"]): r for r in rows}
    check(len(records) == len(rows), "duplicate reference row")
    return records


def pool(blocks):
    histogram = Counter()
    groups = 0
    models = sorted(blocks[0]["errors"])
    totals = {model: [0, 0, 0, 0] for model in models}
    for block in blocks:
        check(set(block["errors"]) == set(models), "candidate mismatch")
        check(type(block["groups"]) is int and block["groups"] > 0, "invalid group count")
        groups += block["groups"]
        seen = set()
        for k, t, units in block["kt"]:
            check(all(type(x) is int for x in (k,t,units)) and 0 <= t <= k and k > 0 and units > 0,
                  "invalid histogram row")
            check((k,t) not in seen, "duplicate histogram cell within block")
            histogram[k,t] += units
            seen.add((k,t))
        for model, counts in block["errors"].items():
            check(len(counts) == 4 and all(type(v) is int and v >= 0 for v in counts), "invalid errors")
            totals[model] = [x + y for x,y in zip(totals[model], counts)]
    units = sum(histogram.values())
    fail_bits = sum(t*n for (k,t),n in histogram.items())
    pass_bits = sum((k-t)*n for (k,t),n in histogram.items())
    pass_units = sum(n for (_,t),n in histogram.items() if t == 0)
    for fp,ff,tfp,tff in totals.values():
        check(fp <= fail_bits and ff <= pass_bits and tfp <= units-pass_units and tff <= pass_units,
              "error total exceeds its reference denominator")
    return dict(histogram=histogram, groups=groups, units=units, pass_bits=pass_bits, fail_bits=fail_bits,
                models=models, counts=totals)


def objectives(dev, cost):
    cf, ce = cost
    result = {}
    for model in dev["models"]:
        fp,ff,tfp,tff = dev["counts"][model]
        alpha = (ff + 0.5) / (dev["pass_bits"] + 1)
        beta = (fp + 0.5) / (dev["fail_bits"] + 1)
        expected_fp = []
        expected_ff = []
        for (k,t), n in dev["histogram"].items():
            predicted_pass = (1-alpha)**(k-t) * beta**t
            if t:
                expected_fp.append(n * predicted_pass)
            else:
                expected_ff.append(n * (1-predicted_pass))
        fp_rate = fsum(expected_fp) / dev["units"]
        ff_rate = fsum(expected_ff) / dev["units"]
        result[model] = dict(criterion=(fp+ff)/(dev["fail_bits"]+dev["pass_bits"]),
            empirical_task=(cf*tfp+ce*tff)/dev["units"], independent_task=cf*fp_rate+ce*ff_rate,
            alpha=alpha, beta=beta, predicted_FP_rate=fp_rate, predicted_FF_rate=ff_rate)
    return result


def tied_winners(scores, rule):
    best = min(s[rule] for s in scores.values())
    return sorted(model for model,s in scores.items() if abs(s[rule]-best) <= 1e-12)


def historical(frames):
    archived = reference_table("fold_forecasts.csv")
    validation = reference_table("forecast_validation.csv")
    used = set()
    held_totals = defaultdict(Counter)
    for frame, folds in frames.items():
        check(set(folds) == set("01234"), "five grouped folds expected")
        for key, block in folds.items():
            dev = pool([b for k,b in folds.items() if k != key])
            held = pool([block])
            for cf,ce in COSTS:
                cost = f"{cf}:{ce}"
                score = objectives(dev, (cf,ce))
                published = {}
                for model, metrics in score.items():
                    index = (frame, int(key), cost, model)
                    used.add(index)
                    old = archived[index]
                    val = validation[index]
                    published[model] = dict(criterion=float(old["criterion_dev_error"]),
                        empirical_task=float(old["empirical_dev_loss"]), independent_task=float(old["predicted_dev_loss"]))
                    for rule in RULES:
                        check_number(metrics[rule], published[model][rule], rule)
                    for field in ("alpha","beta","predicted_FP_rate","predicted_FF_rate"):
                        check_number(metrics[field], old[field], field)
                    for field, actual in zip(("train_criterion_FP","train_criterion_FF","train_task_FP","train_task_FF"),
                                             dev["counts"][model]):
                        check_number(actual,val[field],field)
                    held_fp,held_ff = held["counts"][model][2:]
                    actual_loss = (cf*held_fp + ce*held_ff)/held["units"]
                    for field,actual in (("dev_groups",dev["groups"]),("test_groups",held["groups"]),
                        ("dev_units",dev["units"]),("test_units",held["units"]),("train_reference_pass_bits",dev["pass_bits"]),
                        ("train_reference_fail_bits",dev["fail_bits"]),("test_task_FP",held_fp),("test_task_FF",held_ff),
                        ("dev_predicted_loss_per100",100*metrics["independent_task"]),
                        ("dev_observed_loss_per100",100*metrics["empirical_task"]),("heldout_observed_loss_per100",100*actual_loss),
                        ("predicted_minus_heldout_per100",100*(metrics["independent_task"] - actual_loss))):
                        check_number(actual,val[field],field)
                for rule in RULES:
                    ties = tied_winners(score, rule)
                    check(ties == tied_winners(published, rule), f"changed tie set {frame}/{cost}/{key}/{rule}")
                    winner = ties[0]
                    fp, ff = held["counts"][winner][2:]
                    held_totals[frame,cost][rule] += cf*fp + ce*ff
                held_totals[frame,cost]["units"] += held["units"]
    check(used == archived.keys() == validation.keys(), "historical forecast table coverage mismatch")
    return held_totals, len(used)


def external(panels):
    seal = json.loads((INPUTS/"external_seal.json").read_text())["panels"]
    check(set(panels) == set(seal), "external panel mismatch")
    for name, block in panels.items():
        dev = pool([block])
        saved = seal[name]
        for field in ("groups","units"):
            check(dev[field]==saved[field], f"external {field} mismatch")
        for cf,ce in COSTS:
            cost = f"{cf}:{ce}"
            score = objectives(dev,(cf,ce))
            old = saved["costs"][cost]
            for rule in RULES:
                ties = tied_winners(score, rule)
                check(ties == old["ties"][rule] and ties[0] == old["choices"][rule], "external choice mismatch")
                for model, metrics in score.items():
                    check_number(metrics[rule],old["objectives"][rule][model],f"{name}/{cost}/{rule}/{model}")
            for model, metrics in score.items():
                for field in ("predicted_FP_rate","predicted_FF_rate"):
                    check_number(metrics[field],old[field.replace("predicted_","predicted_task_")][model],field)


def main():
    frames = read_counts("fold_kt.json", "frames")
    external_panels = read_counts("external_dev_counts.json", "panels")
    check(len(frames)==4 and len(external_panels)==2, "unexpected number of frames")
    totals, n = historical(frames)
    external(external_panels)
    for frame in sorted(frames):
        row = totals[frame,"1:1"]
        values = " / ".join(str(row[r]) for r in RULES)
        print(f"{frame}: {values} errors on {row['units']} units (criterion / empirical / independent)")
    n_historical = len(frames)*5*len(COSTS)*len(RULES)
    n_external = len(external_panels)*len(COSTS)*len(RULES)
    print(f"Verified {n} historical candidate forecasts, {n_historical} historical selector choices "
          f"and {n_external} external selector choices from anonymous counts.")


if __name__ == "__main__":
    main()
