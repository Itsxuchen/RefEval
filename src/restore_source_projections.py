"""Optionally rebuild numeric projections from pinned public source files.

This downloads JudgmentBench sources into temporary storage and retains only
verified projections. RuVerBench prediction tables are copied from the package:
the authors' saved model predictions cannot be regenerated from source labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Callable
import urllib.request


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
JUDGMENTBENCH_REVISION = "945ff52f4c63c17006dc30ec35fefbddf3ccf58d"
JUDGMENTBENCH_PREFIX = (
    "https://huggingface.co/datasets/judgmentbench/JudgmentBench/resolve/"
)
RUVER_FILES = {
    "data/processed/public_judge_predictions.csv",
    "data/processed/gpt54_low_dr_predictions.csv",
}
PROTECTED = (
    "data", "src", "tests", "docs", "licenses", "context", ".git",
    "artifacts/expected", "artifacts/figures",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe manifest path: {value!r}")
    return path


def validate_output(root: Path, output: Path) -> Path:
    """Reject existing paths and overlap with immutable package inputs."""
    root = root.resolve()
    requested = output if output.is_absolute() else root / output
    if requested.exists() or requested.is_symlink():
        raise ValueError(f"Output must be a new directory: {requested}")
    destination = requested.resolve()
    if destination == root or root.is_relative_to(destination):
        raise ValueError(f"Output overlaps the package root: {destination}")
    for relative in PROTECTED:
        protected = (root / relative).resolve()
        if (destination == protected or destination.is_relative_to(protected)
                or protected.is_relative_to(destination)):
            raise ValueError(f"Output overlaps protected package content: {destination}")
    return destination


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "reference-label-value/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream)


def project_csv(source: Path, destination: Path, columns: list[str]) -> int:
    """Preserve row order and values with the package's CSV serialization."""
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(sys.maxsize)
    try:
        with source.open("r", newline="", encoding="utf-8") as incoming:
            reader = csv.DictReader(incoming)
            header = reader.fieldnames or []
            if len(header) != len(set(header)) or not set(columns).issubset(header):
                raise ValueError("Source CSV has duplicate headers or missing projected columns")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", newline="", encoding="utf-8") as outgoing:
                writer = csv.DictWriter(outgoing, fieldnames=columns)
                writer.writeheader()
                count = 0
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("Source CSV row does not match its header")
                    writer.writerow({column: row[column] for column in columns})
                    count += 1
            return count
    finally:
        csv.field_size_limit(previous_limit)


def check_table(path: Path, entry: dict) -> int:
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Packaged SHA-256 mismatch: {entry['path']}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != entry["columns"]:
            raise ValueError(f"Packaged columns mismatch: {entry['path']}")
        count = sum(1 for _ in reader)
    if count != entry["rows"]:
        raise ValueError(f"Packaged row count mismatch: {entry['path']}")
    return count


def validate_entry(entry: dict) -> None:
    path = safe_relative_path(entry["path"])
    for field in ("source_sha256", "sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", entry[field]):
            raise ValueError(f"Invalid {field}: {path}")
    columns = entry["columns"]
    if (not columns or len(columns) != len(set(columns))
            or any(not isinstance(column, str) or not column for column in columns)
            or not isinstance(entry["rows"], int) or entry["rows"] < 0):
        raise ValueError(f"Invalid table schema: {path}")
    if str(path) in RUVER_FILES:
        return
    if path.parent != PurePosixPath("data/reference/judgmentbench"):
        raise ValueError(f"Unsupported manifest input: {path}")
    source_path = safe_relative_path(entry.get("source_path", ""))
    revision = entry.get("source_revision")
    if revision != JUDGMENTBENCH_REVISION:
        raise ValueError(f"JudgmentBench source revision is missing or unexpected: {path}")
    expected_url = f"{JUDGMENTBENCH_PREFIX}{revision}/{source_path}"
    if entry.get("source_url") != expected_url:
        raise ValueError(f"JudgmentBench source URL is not the pinned manifest path: {path}")


def restore_sources(
    root: Path,
    output: Path,
    *,
    fetch: Callable[[str, Path], None] = download_file,
) -> dict:
    """Validate every input before publishing a new numeric-only output tree."""
    root = root.resolve()
    destination = validate_output(root, output)
    manifest = json.loads((root / "data/PROVENANCE.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    if not entries or len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("Manifest contains no inputs or duplicate output paths")
    for entry in entries:
        validate_entry(entry)

    results = []
    with tempfile.TemporaryDirectory(prefix="reference-label-value-restore-") as temporary:
        working = Path(temporary)
        staged = working / "projected"
        staged.mkdir()
        for index, entry in enumerate(entries):
            relative = entry["path"]
            target = staged / relative
            if relative in RUVER_FILES:
                source = (root / relative).resolve()
                if not source.is_relative_to(root / "data"):
                    raise ValueError(f"Saved prediction input escapes package data: {relative}")
                if sha256(source) != entry["source_sha256"]:
                    raise ValueError(f"Saved prediction SHA-256 mismatch: {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                action = "copied_saved_predictions_not_regenerated"
            else:
                source = working / f"source-{index}.csv"
                fetch(entry["source_url"], source)
                if sha256(source) != entry["source_sha256"]:
                    raise ValueError(f"Source SHA-256 mismatch: {entry['source_path']}")
                project_csv(source, target, entry["columns"])
                source.unlink()
                action = "downloaded_pinned_source_and_projected"
            count = check_table(target, entry)
            result = {"path": relative, "status": "verified", "action": action,
                      "rows": count, "sha256": entry["sha256"]}
            if relative not in RUVER_FILES:
                result.update(source_url=entry["source_url"],
                              source_sha256=entry["source_sha256"])
            results.append(result)

        report = {"status": "verified", "files": results,
                  "raw_sources_retained": False,
                  "ruver_predictions_regenerated": False,
                  "note": "RuVerBench saved predictions are copied, not regenerated from reference labels."}
        (staged / "restore_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        # mkdir is the final exclusive reservation: an intervening existing path
        # causes failure rather than replacing even an empty user directory.
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(exist_ok=False)
        try:
            shutil.copytree(staged, destination, dirs_exist_ok=True)
        except BaseException:
            shutil.rmtree(destination)
            raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/restored-inputs"),
                        help="New output directory; relative paths are resolved from the package root")
    args = parser.parse_args()
    try:
        report = restore_sources(PACKAGE_ROOT, args.output)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"Restoration failed: {error}\n")
    print(json.dumps({"output": str((PACKAGE_ROOT / args.output).resolve()), **report}, indent=2))


if __name__ == "__main__":
    main()
