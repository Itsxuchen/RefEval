"""Check versioned files, projected-input schemas and fixed result provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE_MANIFEST.json"
ROOT_FILES = {
    "README.md", "LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md", "CHANGELOG.md", "pyproject.toml", "requirements.txt",
    "requirements.lock.txt", ".gitignore", ".gitattributes",
}
TREES = ("src", "tests", "context", "docs", "licenses", "data",
         "artifacts/expected", "artifacts/figures", "artifacts/validation", ".github")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_files(root: Path) -> list[Path]:
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in TREES:
        paths.extend(p for p in (root / name).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix not in {".pyc", ".log"} and p.name != ".DS_Store")
    return sorted(paths)


def write_manifest(root: Path) -> dict:
    manifest = {
        "schema_version": 1, "software_version": "0.1.0",
        "scope": "Current release files; historical scientific output provenance is separate.",
        "files": {str(p.relative_to(root)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in package_files(root)},
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def checked_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Manifest paths must remain inside the repository")
    return path


def verify(root: Path) -> dict:
    manifest = json.loads((root / MANIFEST).read_text())
    actual = {str(p.relative_to(root)) for p in package_files(root)}
    if actual != set(manifest["files"]):
        raise ValueError("Release file inventory differs from manifest")
    for name, info in manifest["files"].items():
        path = checked_path(root, name)
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError(f"Release file changed: {name}")
    inputs = json.loads((root / "data/PROVENANCE.json").read_text())
    row_count = 0
    for item in inputs["files"]:
        path = checked_path(root, item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Input hash mismatch: {item['path']}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != item["columns"]:
                raise ValueError(f"Input schema mismatch: {item['path']}")
            rows = sum(1 for _ in reader)
        if rows != item["rows"]:
            raise ValueError(f"Input row count mismatch: {item['path']}")
        row_count += rows
    expected = json.loads((root / "artifacts/expected/PROVENANCE.json").read_text())
    for item in expected["files"]:
        if sha256(checked_path(root, item["path"])) != item["sha256"]:
            raise ValueError(f"Expected scientific output changed: {item['path']}")
    return {"passed": True, "release_files": len(actual),
            "input_files": len(inputs["files"]), "input_rows": row_count,
            "expected_files": len(expected["files"]),
            "scope": "File integrity and published-input schemas; not a scientific rerun or source-label adjudication."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true",
                        help="Maintainer operation: freeze the current release inventory")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest(ROOT)
    print(json.dumps(verify(ROOT), indent=2))


if __name__ == "__main__":
    main()
