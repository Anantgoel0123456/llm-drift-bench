"""Offline baseline snapshot integrity manifests (SHA-256 verification for golden fixtures)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any

from llm_drift_bench import __version__
from llm_drift_bench._file_io import parse_source_json, read_utf8
from llm_drift_bench._matching import require_same_ids
from llm_drift_bench.engine import load_model_samples, load_prompt_cases
from llm_drift_bench.policy import load_policy

MANIFEST_VERSION = 1


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _sha256_prompt_ids(prompt_ids: set[str]) -> str:
    payload = "\n".join(sorted(prompt_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_rel_posix(raw_path: str) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path.strip() or "\\" in raw_path:
        raise ValueError(f"Snapshot artifact path must be a relative POSIX path: {raw_path!r}.")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or ":" in raw_path or ".." in pure.parts:
        raise ValueError(f"Snapshot artifact path must not escape base directory: {raw_path!r}.")
    return pure


@dataclass(frozen=True)
class SnapshotArtifact:
    """Integrity record for one file referenced by a baseline snapshot manifest."""

    role: str
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotArtifact:
        for key in ("role", "path", "sha256"):
            val = data.get(key)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"Snapshot artifact requires non-empty string {key!r}.")
        _safe_rel_posix(data["path"])
        if len(data["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in data["sha256"]):
            raise ValueError(f"Snapshot artifact has invalid sha256 digest: {data['sha256']!r}.")
        size = data.get("size_bytes")
        if type(size) is not int or size < 0:
            raise ValueError("Snapshot artifact 'size_bytes' must be a non-negative integer.")
        return cls(
            role=data["role"],
            path=data["path"],
            sha256=data["sha256"],
            size_bytes=size,
        )


@dataclass(frozen=True)
class SnapshotManifest:
    """Auditable SHA-256 manifest pinning a golden baseline capture, suite, and policy."""

    version: int
    created_at: str
    tool_version: str
    baseline_model: str
    sample_count: int
    prompt_ids_sha256: str
    artifacts: tuple[SnapshotArtifact, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "tool_version": self.tool_version,
            "baseline_model": self.baseline_model,
            "sample_count": self.sample_count,
            "prompt_ids_sha256": self.prompt_ids_sha256,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotManifest:
        version = data.get("version")
        if type(version) is not int or version != MANIFEST_VERSION:
            raise ValueError(f"Unsupported snapshot manifest version: {version!r}.")
        for key in ("created_at", "tool_version", "baseline_model", "prompt_ids_sha256"):
            val = data.get(key)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"Snapshot manifest missing non-empty string {key!r}.")
        count = data.get("sample_count")
        if type(count) is not int or count < 1:
            raise ValueError("Snapshot manifest 'sample_count' must be a positive integer.")
        raw_artifacts = data.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ValueError("Snapshot manifest requires a non-empty 'artifacts' list.")
        artifacts = tuple(
            SnapshotArtifact.from_dict(item)
            if isinstance(item, dict)
            else (_ for _ in ()).throw(ValueError("Snapshot artifact entry must be an object."))
            for item in raw_artifacts
        )
        return cls(
            version=version,
            created_at=data["created_at"],
            tool_version=data["tool_version"],
            baseline_model=data["baseline_model"],
            sample_count=count,
            prompt_ids_sha256=data["prompt_ids_sha256"],
            artifacts=artifacts,
        )


@dataclass(frozen=True)
class SnapshotVerification:
    """Result of verifying a SnapshotManifest against files on disk."""

    valid: bool
    checked: int
    mismatches: tuple[str, ...]
    missing: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked": self.checked,
            "mismatches": list(self.mismatches),
            "missing": list(self.missing),
        }


def _resolve_root(
    paths: list[Path],
    base_dir: str | Path | None,
    output_path: str | Path | None,
) -> Path:
    if base_dir is not None:
        return Path(base_dir).resolve()
    if output_path is not None:
        return Path(output_path).resolve().parent
    return Path(os.path.commonpath([str(p.resolve().parent) for p in paths])).resolve()


def _atomic_write(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


def create_snapshot_manifest(
    baseline_path: str | Path,
    *,
    suite_path: str | Path | None = None,
    config_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    created_at: str | None = None,
) -> SnapshotManifest:
    """Validate and hash golden baseline artifacts into a portable SHA-256 manifest."""
    base_file = Path(baseline_path).resolve()
    samples = load_model_samples(base_file)
    if not samples:
        raise ValueError("Baseline capture cannot be empty.")
    models = sorted({s.model_name for s in samples})
    if len(models) != 1:
        raise ValueError(f"Baseline capture mixes multiple model names: {models!r}.")
    base_ids = {s.prompt_id for s in samples}

    tagged_files: list[tuple[str, Path]] = [("baseline", base_file)]
    if suite_path is not None:
        suite_file = Path(suite_path).resolve()
        cases = load_prompt_cases(suite_file)
        if not cases:
            raise ValueError("Prompt suite cannot be empty.")
        require_same_ids(base_ids, {c.prompt_id for c in cases}, "Prompt suite")
        tagged_files.append(("suite", suite_file))
    if config_path is not None:
        cfg_file = Path(config_path).resolve()
        load_policy(cfg_file)
        tagged_files.append(("policy", cfg_file))

    root = _resolve_root([p for _, p in tagged_files], base_dir, output_path)
    artifacts: list[SnapshotArtifact] = []
    for role, path in tagged_files:
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Snapshot artifact {path} is outside base directory {root}; "
                "specify --base-dir covering all referenced files."
            ) from exc
        digest, size = _sha256_file(path)
        artifacts.append(SnapshotArtifact(role=role, path=rel, sha256=digest, size_bytes=size))

    stamp = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = SnapshotManifest(
        version=MANIFEST_VERSION,
        created_at=stamp,
        tool_version=__version__,
        baseline_model=models[0],
        sample_count=len(samples),
        prompt_ids_sha256=_sha256_prompt_ids(base_ids),
        artifacts=tuple(artifacts),
    )
    if output_path is not None:
        _atomic_write(Path(output_path), manifest.to_json(indent=2))
    return manifest


def verify_snapshot_manifest(
    manifest_path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> SnapshotVerification:
    """Verify every file hash and baseline prompt-ID digest recorded in a snapshot manifest."""
    source = Path(manifest_path).resolve()
    parsed = parse_source_json(read_utf8(source), source)
    if not isinstance(parsed, dict):
        raise ValueError(f"{source}: snapshot manifest must be a JSON object.")
    manifest = SnapshotManifest.from_dict(parsed)

    root = Path(base_dir).resolve() if base_dir is not None else source.parent
    missing: list[str] = []
    mismatches: list[str] = []

    for artifact in manifest.artifacts:
        rel = _safe_rel_posix(artifact.path)
        target = (root / Path(*rel.parts)).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Snapshot artifact escapes base directory: {artifact.path!r}.")
        if not target.is_file():
            missing.append(f"{artifact.role} ({artifact.path}): file not found")
            continue
        actual_sha, actual_size = _sha256_file(target)
        if actual_sha != artifact.sha256 or actual_size != artifact.size_bytes:
            mismatches.append(
                f"{artifact.role} ({artifact.path}): digest mismatch "
                f"(expected {artifact.sha256[:12]}..., got {actual_sha[:12]}...)"
            )
            continue
        if artifact.role == "baseline":
            samples = load_model_samples(target)
            actual_ids_sha = _sha256_prompt_ids({s.prompt_id for s in samples})
            if (
                len(samples) != manifest.sample_count
                or actual_ids_sha != manifest.prompt_ids_sha256
            ):
                mismatches.append(
                    f"baseline ({artifact.path}): prompt ID digest or sample count mismatch"
                )

    return SnapshotVerification(
        valid=not missing and not mismatches,
        checked=len(manifest.artifacts),
        mismatches=tuple(mismatches),
        missing=tuple(missing),
    )


def format_snapshot_verification_text(result: SnapshotVerification) -> str:
    """Format a SnapshotVerification summary for terminal display."""
    if result.valid:
        return f"Snapshot verification: VALID ({result.checked} artifact(s) verified)."
    lines = [
        f"Snapshot verification: INVALID "
        f"({len(result.mismatches)} modified, {len(result.missing)} missing)"
    ]
    for item in result.missing:
        lines.append(f"  [MISSING ] {item}")
    for item in result.mismatches:
        lines.append(f"  [MODIFIED] {item}")
    return "\n".join(lines)
