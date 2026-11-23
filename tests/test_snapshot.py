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
