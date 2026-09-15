"""Scoring-type and masked-error analysis on fixed published reference labels."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.conjunction_policy_data import load_datasets
from src.conjunction_policy_study import initial_state, make_order, replay

ROOT = Path(__file__).resolve().parents[1]


def analyze(data: dict) -> dict:
    st = initial_state(data)
    ti, gold, pred = data["task_index"], data["gold"], data["prediction"]
    modes = np.asarray(data["category"])
    rows, modes_count = [], {}
    for mode in sorted(set(modes)):
        ix = modes == mode
        modes_count[mode] = {"criteria": int(ix.sum()),
                            "criterion_fp": int(((gold == 0) & (pred == 1) & ix).sum()),
                            "criterion_ff": int(((gold == 1) & (pred == 0) & ix).sum())}
    ff_types = Counter()
    for t in range(len(data["task_ids"])):
        idx = np.flatnonzero(ti == t)
        ff_modes = sorted(set(modes[idx[(gold[idx] == 1) & (pred[idx] == 0)]]))
        if st["initial_ff"][t]:
            ff_types["+".join(ff_modes)] += 1
        if st["susceptible"][t]:
            f, q = int(st["gf"][t]), int(st["pf"][t])
            # All q initially predicted failures are false. A new task FP occurs
            # iff all q are repaired before any of f real failures is queried.
            rows.append({"task_id": data["task_ids"][t], "base_task_id": data["base_task_ids"][t],
                         "reference_fail_count": f, "false_fail_count": q,
                         "false_fail_modes": ff_modes,
                         "reference_fail_modes": sorted(set(modes[idx[gold[idx] == 0]])),
                         "uniform_random_unmask_probability": 1.0 / math.comb(f + q, q)})
    result = {"susceptible_tasks": len(rows), "susceptible_base_tasks": len(set(r["base_task_id"] for r in rows)),
              "susceptible_rows": rows, "criterion_mode_errors": modes_count,
              "initial_task_false_fail_rejection_modes": dict(ff_types),
              "uniform_random_expected_ever_unmasked": sum(r["uniform_random_unmask_probability"] for r in rows),
              "canonical_policy_transitions": {}}
    for policy in ("random_criterion", "disagreement_only", "disagreement_then_random", "sc_judge", "sc_disagreement"):
        tr = replay(data, policy, "release", 0)
        new_positions = np.flatnonzero(tr["deltas"][:, 8])
        query_ids = tr["indices"][new_positions]
        new_tasks = ti[query_ids]
        assert st["susceptible"][new_tasks].all()
        assert len(set(new_tasks.tolist())) == len(new_tasks)
        types = Counter(data["category"][i] for i in query_ids)
        by_base = Counter(data["base_task_ids"][t] for t in new_tasks)
        outcome = {"ever_new_fp_tasks": len(new_tasks), "last_repaired_criterion_mode": dict(types),
                   "new_fp_base_task_counts": dict(by_base), "base_tasks_with_new_fp": len(by_base)}
        if policy == "sc_judge":
            assert len(new_tasks) == len(rows), "judge-fail-first must unmask every susceptible task before its fail certificate"
        if policy == "disagreement_only":
            # Independent direct final-label check for transition identities.
            q = np.zeros(len(gold), bool); q[tr["indices"]] = True
            after = np.where(q, gold, pred)
            ap = np.bincount(ti, weights=1-after, minlength=len(st["k"])) == 0
            rp, pp = st["gf"] == 0, st["pf"] == 0
            outcome["final_transition_counts"] = {
                "fixed_initial_fp": int((~rp & pp & ~ap).sum()),
                "fixed_initial_ff": int((rp & ~pp & ap).sum()),
                "new_fp": int((~rp & ~pp & ap).sum()),
                "new_ff": int((rp & pp & ~ap).sum()),
                "before_task_errors": int((rp != pp).sum()), "after_task_errors": int((rp != ap).sum())}
            outcome["final_new_fp_tasks"] = [data["task_ids"][t] for t in np.flatnonzero(~rp & ~pp & ap)]
        result["canonical_policy_transitions"][policy] = outcome
    return result


def check_random_formula() -> int:
    checked = 0
    for f in range(1, 4):
        for q in range(1, 4):
            good = total = 0
            for order in itertools.permutations(range(f+q)):
                # 0..q-1 false failing labels, q..q+f-1 reference failures.
                positions = {i: pos for pos, i in enumerate(order)}
                good += max(positions[i] for i in range(q)) < min(positions[i] for i in range(q, f+q))
                total += 1
            assert math.isclose(good/total, 1/math.comb(f+q, q))
            checked += 1
    return checked


def run_experiment(datasets: dict, output: Path) -> dict:
    res = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "scope": "finite-label scoring-type/transition analysis; changed binary-only target is not an intervention",
           "random_formula_small_cases_verified": check_random_formula(),
           "random_formula": "sum over susceptible tasks of 1/binomial(reference_fail_count+false_fail_count,false_fail_count)",
           "datasets": {name: analyze(data) for name, data in datasets.items()},
           "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__), ROOT/"src/conjunction_policy_data.py", ROOT/"src/conjunction_policy_study.py"]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(res, indent=2))
    print(json.dumps({name: {k:r[k] for k in ["susceptible_tasks", "susceptible_base_tasks", "uniform_random_expected_ever_unmasked"]}
                      for name,r in res["datasets"].items()}, indent=2))

    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/reproduced/conjunction_policy_study/error_flow.json")
    args = p.parse_args()
    run_experiment(load_datasets(ROOT), args.output)


if __name__ == "__main__":
    main()
