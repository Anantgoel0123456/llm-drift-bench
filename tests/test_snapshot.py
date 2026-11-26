"""Tests for offline baseline snapshot integrity manifests and tamper verification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from llm_drift_bench.snapshot import (
    SnapshotManifest,
    create_snapshot_manifest,
    format_snapshot_verification_text,
    verify_snapshot_manifest,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_create_and_verify_snapshot_manifest_detects_tampering_and_missing(
    tmp_path: Path,
) -> None:
    for fname in ("baseline.jsonl", "extraction-suite.json", "drift.toml"):
        shutil.copy2(EXAMPLES / fname, tmp_path / fname)

    manifest_file = tmp_path / "snapshot.manifest.json"
    manifest = create_snapshot_manifest(
        tmp_path / "baseline.jsonl",
        suite_path=tmp_path / "extraction-suite.json",
        config_path=tmp_path / "drift.toml",
        output_path=manifest_file,
    )
    assert manifest.baseline_model == "demo-reference"
    assert manifest.sample_count == 2
    assert [a.role for a in manifest.artifacts] == ["baseline", "suite", "policy"]

    ok = verify_snapshot_manifest(manifest_file)
    assert ok.valid is True
    assert ok.checked == 3
    assert "VALID (3 artifact(s) verified)" in format_snapshot_verification_text(ok)

    # Tamper with drift.toml and confirm mismatch is detected immediately.
    (tmp_path / "drift.toml").write_text("# tampered\n", encoding="utf-8")
    tampered = verify_snapshot_manifest(manifest_file)
    assert tampered.valid is False
    assert len(tampered.mismatches) == 1
    assert "policy (drift.toml)" in tampered.mismatches[0]
    assert "[MODIFIED]" in format_snapshot_verification_text(tampered)

    # Delete suite and confirm missing file is flagged.
    (tmp_path / "extraction-suite.json").unlink()
    missing_res = verify_snapshot_manifest(manifest_file)
    assert missing_res.valid is False
    assert len(missing_res.missing) == 1
    assert "[MISSING ]" in format_snapshot_verification_text(missing_res)
    assert missing_res.to_dict()["valid"] is False


def test_snapshot_rejects_path_traversal_and_outside_base_dir(tmp_path: Path) -> None:
    outside = tmp_path / "outside.jsonl"
    shutil.copy2(EXAMPLES / "baseline.jsonl", outside)
    nested = tmp_path / "nested"
    nested.mkdir()

    with pytest.raises(ValueError, match="outside base directory"):
        create_snapshot_manifest(outside, output_path=nested / "manifest.json")

    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(
        json.dumps({
            "version": 1,
            "created_at": "2026-11-23T00:00:00Z",
            "tool_version": "0.3.0",
            "baseline_model": "m",
            "sample_count": 1,
            "prompt_ids_sha256": "a" * 64,
            "artifacts": [
                {"role": "baseline", "path": "../secret.jsonl", "sha256": "a" * 64, "size_bytes": 1}
            ],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not escape"):
        verify_snapshot_manifest(bad_manifest)

    with pytest.raises(ValueError, match="Unsupported snapshot manifest version"):
        SnapshotManifest.from_dict({"version": True})


def test_snapshot_validation_base_dir_and_prompt_id_digest_mismatch(tmp_path: Path) -> None:
    from llm_drift_bench.snapshot import SnapshotArtifact

    with pytest.raises(ValueError, match="non-empty string 'role'"):
        SnapshotArtifact.from_dict(
            {"role": "", "path": "a.jsonl", "sha256": "a" * 64, "size_bytes": 1}
        )
    with pytest.raises(ValueError, match="relative POSIX path"):
        SnapshotArtifact.from_dict(
            {"role": "baseline", "path": "a\\b.jsonl", "sha256": "a" * 64, "size_bytes": 1}
        )
    with pytest.raises(ValueError, match="invalid sha256"):
        SnapshotArtifact.from_dict(
            {"role": "baseline", "path": "a.jsonl", "sha256": "not-hex", "size_bytes": 1}
        )
    with pytest.raises(ValueError, match="size_bytes"):
        SnapshotArtifact.from_dict(
            {"role": "baseline", "path": "a.jsonl", "sha256": "a" * 64, "size_bytes": -1}
        )

    valid_base = {
        "version": 1,
        "created_at": "2026-11-25T00:00:00Z",
        "tool_version": "0.3.0",
        "baseline_model": "m",
        "sample_count": 1,
        "prompt_ids_sha256": "a" * 64,
        "artifacts": [
            {"role": "baseline", "path": "a.jsonl", "sha256": "a" * 64, "size_bytes": 1}
        ],
    }
    with pytest.raises(ValueError, match="missing non-empty string"):
        SnapshotManifest.from_dict({**valid_base, "baseline_model": ""})
    with pytest.raises(ValueError, match="sample_count"):
        SnapshotManifest.from_dict({**valid_base, "sample_count": 0})
    with pytest.raises(ValueError, match="non-empty 'artifacts'"):
        SnapshotManifest.from_dict({**valid_base, "artifacts": []})

    # Mixed models in baseline are rejected
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "m1", "output": "a"}) + "\n"
        + json.dumps({"prompt_id": "p2", "model_name": "m2", "output": "b"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mixes multiple model names"):
        create_snapshot_manifest(mixed)

    # Explicit base_dir without output_path, plus prompt_ids_sha256 mismatch detection
    sub = tmp_path / "sub"
    sub.mkdir()
    shutil.copy2(EXAMPLES / "baseline.jsonl", sub / "baseline.jsonl")
    manifest = create_snapshot_manifest(sub / "baseline.jsonl", base_dir=tmp_path)
    assert manifest.artifacts[0].path == "sub/baseline.jsonl"

    tampered_meta = manifest.to_dict()
    tampered_meta["prompt_ids_sha256"] = "0" * 64
    meta_file = tmp_path / "bad_ids.json"
    meta_file.write_text(json.dumps(tampered_meta), encoding="utf-8")
    ver = verify_snapshot_manifest(meta_file, base_dir=tmp_path)
    assert ver.valid is False and "prompt ID digest" in ver.mismatches[0]

    non_obj = tmp_path / "list.json"
    non_obj.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        verify_snapshot_manifest(non_obj)

    # create_snapshot_manifest without base_dir or output_path resolves commonpath cleanly
    m_auto = create_snapshot_manifest(sub / "baseline.jsonl")
    assert m_auto.artifacts[0].path == "baseline.jsonl"

    # Empty baseline capture and empty prompt suite rejected explicitly
    empty_cap = tmp_path / "empty_cap.jsonl"
    empty_cap.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        create_snapshot_manifest(empty_cap)

    empty_suite = tmp_path / "empty_suite.json"
    empty_suite.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        create_snapshot_manifest(sub / "baseline.jsonl", suite_path=empty_suite)


