"""Read-only verification of regenerated results against frozen scientific targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.release_validation import verify_outputs
from src.reproduce import EXPECTED, ROOT, configuration, validate_output
from src.verify_conjunction_policy_study import verify as verify_policy
from src.verify_conjunction_label_value import verify as verify_label


def verify(results: Path, mode: str = "full") -> dict:
    results = validate_output(results)
    manifest = json.loads((results / "run_manifest.json").read_text())
    if manifest["configuration"] != configuration(mode):
        raise AssertionError("saved execution schedule does not match requested verification mode")
    if not manifest["passed"]:
        raise AssertionError("saved reproduction did not complete")
    policy = verify_policy(ROOT, results / "conjunction_policy_study")
    if not policy["passed"]:
        raise AssertionError("independent policy arithmetic or source digests failed")
    label = verify_label(results / "conjunction_label_value", results / "conjunction_policy_study")
    result = verify_outputs(results, EXPECTED, mode)
    result["direct_arithmetic"] = {
        "policy_canonical_rows": policy["canonical_rows_checked"],
        "label_hybrid_rows": label["independently_scored_hybrid_rows"],
        "whole_task_rows": label["raw_rechecked_whole_task_rows"]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("artifacts/reproduced"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    print(json.dumps(verify(args.results, args.mode), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
