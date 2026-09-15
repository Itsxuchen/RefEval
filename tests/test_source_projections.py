"""Source restoration contracts; all downloads are local test doubles."""

import hashlib
import json
from pathlib import Path

import pytest

from src.restore_source_projections import (
    JUDGMENTBENCH_PREFIX, JUDGMENTBENCH_REVISION, project_csv,
    restore_sources, validate_output,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixture_package(tmp_path):
    root = tmp_path / "package"
    (root / "data/processed").mkdir(parents=True)
    saved = b"gold,prediction\r\nTrue,False\r\n"
    relative = "data/processed/public_judge_predictions.csv"
    (root / relative).write_bytes(saved)
    source = b'id,comment,score\r\nsecond,"ignored, text",0.00\r\nfirst,"multi\nline",1\r\n'
    expected = b"score,id\r\n0.00,second\r\n1,first\r\n"
    entry = {
        "path": "data/reference/judgmentbench/fixture.csv",
        "source_sha256": digest(source), "sha256": digest(expected),
        "columns": ["score", "id"], "rows": 2,
        "source_path": "human/fixture.csv", "source_revision": JUDGMENTBENCH_REVISION,
        "source_url": f"{JUDGMENTBENCH_PREFIX}{JUDGMENTBENCH_REVISION}/human/fixture.csv",
    }
    manifest = {"files": [
        {"path": relative, "source_sha256": digest(saved), "sha256": digest(saved),
         "columns": ["gold", "prediction"], "rows": 1}, entry,
    ]}
    (root / "data/PROVENANCE.json").write_text(json.dumps(manifest))
    return root, source, expected, manifest


def test_restoration_downloads_only_source_and_retains_only_verified_projection(tmp_path):
    root, source, expected, manifest = fixture_package(tmp_path)
    downloaded = []

    def fetch(url, destination):
        downloaded.append((url, destination))
        destination.write_bytes(source)

    report = restore_sources(root, Path("artifacts/restored-inputs"), fetch=fetch)
    output = root / "artifacts/restored-inputs"
    assert (output / manifest["files"][1]["path"]).read_bytes() == expected
    assert (output / manifest["files"][0]["path"]).read_bytes() == (root / manifest["files"][0]["path"]).read_bytes()
    assert len(downloaded) == 1
    assert downloaded[0][0] == manifest["files"][1]["source_url"]
    assert not downloaded[0][1].parent.exists()
    assert report["status"] == "verified"
    assert report["raw_sources_retained"] is False
    assert report["ruver_predictions_regenerated"] is False
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == {
        *(entry["path"] for entry in manifest["files"]), "restore_report.json",
    }


@pytest.mark.parametrize("failure", ["source_hash", "projected_hash", "rows", "saved_prediction"])
def test_failed_verification_publishes_no_output_and_cleans_temporary_sources(tmp_path, failure):
    root, source, _, manifest = fixture_package(tmp_path)
    if failure == "projected_hash":
        manifest["files"][1]["sha256"] = "0" * 64
    elif failure == "rows":
        manifest["files"][1]["rows"] = 3
    elif failure == "saved_prediction":
        (root / manifest["files"][0]["path"]).write_bytes(b"corrupt\n")
    (root / "data/PROVENANCE.json").write_text(json.dumps(manifest))
    downloaded = []

    def fetch(url, destination):
        downloaded.append(destination)
        destination.write_bytes(b"corrupt\n" if failure == "source_hash" else source)

    with pytest.raises(ValueError, match="mismatch"):
        restore_sources(root, Path("artifacts/restored-inputs"), fetch=fetch)
    assert not (root / "artifacts/restored-inputs").exists()
    assert all(not path.parent.exists() for path in downloaded)


@pytest.mark.parametrize("relative", [
    "data/new", "src/new", "tests/new", "docs/new", "licenses/new", "context/new",
    ".git/new", "artifacts/expected/new", "artifacts/figures/new", "artifacts", ".", "..",
])
def test_protected_output_paths_are_rejected(tmp_path, relative):
    root, _, _, _ = fixture_package(tmp_path)
    with pytest.raises(ValueError, match="Output"):
        validate_output(root, Path(relative))


def test_existing_and_symlinked_outputs_are_rejected_without_modification(tmp_path):
    root, _, _, _ = fixture_package(tmp_path)
    existing = root / "artifacts/existing"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="new directory"):
        validate_output(root, existing)
    alias = root / "data-alias"
    alias.symlink_to(root / "data", target_is_directory=True)
    with pytest.raises(ValueError, match="protected"):
        validate_output(root, alias / "new")
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("field,value", [
    ("path", "../outside.csv"),
    ("source_path", "../outside.csv"),
    ("source_revision", "main"),
    ("source_url", "https://example.org/not-the-source.csv"),
])
def test_manifest_paths_and_source_pins_are_checked_before_downloading(tmp_path, field, value):
    root, _, _, manifest = fixture_package(tmp_path)
    manifest["files"][1][field] = value
    (root / "data/PROVENANCE.json").write_text(json.dumps(manifest))

    def forbidden_fetch(url, destination):
        pytest.fail("Invalid manifest must fail before download")

    with pytest.raises(ValueError):
        restore_sources(root, Path("artifacts/restored-inputs"), fetch=forbidden_fetch)


def test_projection_preserves_quoted_fields_empty_strings_and_newlines(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "projection.csv"
    source.write_bytes(b'id,keep,remove\r\n2,"a,b",unused\r\n1,"two\nlines",unused\r\n3,,unused\r\n')
    assert project_csv(source, output, ["keep", "id"]) == 3
    assert output.read_bytes() == b'keep,id\r\n"a,b",2\r\n"two\nlines",1\r\n,3\r\n'


@pytest.mark.parametrize("source", [b"id,id\r\n1,2\r\n", b"wrong\r\n1\r\n", b"id,keep\r\n1\r\n"])
def test_projection_rejects_ambiguous_or_incomplete_csv(tmp_path, source):
    path = tmp_path / "source.csv"
    path.write_bytes(source)
    with pytest.raises(ValueError):
        project_csv(path, tmp_path / "out.csv", ["id", "keep"])
