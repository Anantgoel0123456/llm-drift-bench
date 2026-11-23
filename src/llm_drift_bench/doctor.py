"""Pre-flight environment, policy, schema, and capture alignment checks (doctor)."""

from __future__ import annotations

import importlib.util
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from llm_drift_bench import __version__
from llm_drift_bench._matching import require_same_ids
from llm_drift_bench.engine import load_model_samples, load_prompt_cases
from llm_drift_bench.policy import load_policy
from llm_drift_bench.prompt_lint import lint_prompt_suite
from llm_drift_bench.redaction import redact_samples
from llm_drift_bench.scorers import list_scorers


@dataclass(frozen=True)
class DoctorCheck:
    """One diagnostic check emitted by run_doctor."""

    name: str
    status: Literal["OK", "WARN", "FAIL"]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DoctorReport:
    """Aggregate pre-flight health report."""

    version: str
    python: str
    ok: bool
    scorers: tuple[str, ...]
    extras: dict[str, bool]
    checks: tuple[DoctorCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "python": self.python,
            "ok": self.ok,
            "scorers": list(self.scorers),
            "extras": dict(self.extras),
            "checks": [c.to_dict() for c in self.checks],
        }


def run_doctor(
    *,
    config_path: str | Path | None = None,
    suite_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
) -> DoctorReport:
    """Run non-destructive pre-flight checks on environment, config, suite, and captures."""
    extras = {
        "pydantic": importlib.util.find_spec("pydantic") is not None,
        "rich": importlib.util.find_spec("rich") is not None,
        "openvino_genai": importlib.util.find_spec("openvino_genai") is not None,
        "opik": importlib.util.find_spec("opik") is not None,
    }
    scorers = tuple(list_scorers())
    checks: list[DoctorCheck] = [
        DoctorCheck(
            "runtime",
            "OK",
            f"llm-drift-bench {__version__} on Python {platform.python_version()}",
        ),
        DoctorCheck(
            "scorers",
            "OK",
            f"{len(scorers)} scorer(s) available: {', '.join(scorers)}",
        ),
    ]

    if config_path is not None:
        try:
            policy = load_policy(config_path)
            checks.append(
                DoctorCheck(
                    "config",
                    "OK",
                    f"Loaded policy from {config_path} "
                    f"({len(policy.field_rules)} field rule(s), "
                    f"max_drift={policy.thresholds.max_composite_drift})",
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(DoctorCheck("config", "FAIL", str(exc)))

    id_sets: dict[str, set[str]] = {}
    secret_counts: Counter[str] = Counter()
    captures_scanned = 0

    if suite_path is not None:
        try:
            cases = load_prompt_cases(suite_path)
            if not cases:
                raise ValueError("Prompt suite is empty.")
            with_schema = sum(1 for case in cases if case.schema is not None)
            lint_findings = lint_prompt_suite(cases)
            lint_errors = [item for item in lint_findings if item.severity == "error"]
            lint_warnings = [item for item in lint_findings if item.severity == "warning"]
            if lint_errors:
                preview = "; ".join(
                    f"{item.prompt_id or 'suite'} ({item.code}): {item.message}"
                    for item in lint_errors[:3]
                )
                extra_n = f" (+{len(lint_errors) - 3} more)" if len(lint_errors) > 3 else ""
                checks.append(
                    DoctorCheck("suite_schemas", "FAIL", f"{preview}{extra_n}")
                )
            else:
                id_sets["Prompt suite"] = {c.prompt_id for c in cases}
                checks.append(
                    DoctorCheck(
                        "suite",
                        "OK",
                        f"{len(cases)} prompt case(s) valid ({with_schema} with JSON Schema)",
                    )
                )
            if lint_warnings:
                warn_preview = "; ".join(
                    f"{item.prompt_id or 'suite'} ({item.code})" for item in lint_warnings[:3]
                )
                extra_w = f" (+{len(lint_warnings) - 3} more)" if len(lint_warnings) > 3 else ""
                checks.append(
                    DoctorCheck(
                        "suite_hygiene",
                        "WARN",
                        f"{len(lint_warnings)} lint warning(s): {warn_preview}{extra_w}",
                    )
                )
        except (OSError, ValueError) as exc:
            checks.append(DoctorCheck("suite", "FAIL", str(exc)))

    for label, target in (
        ("Baseline capture", baseline_path),
        ("Candidate capture", candidate_path),
    ):
        if target is None:
            continue
        key = "baseline" if label.startswith("Baseline") else "candidate"
        try:
            samples = load_model_samples(target)
            if not samples:
                raise ValueError(f"{label} is empty.")
            models = sorted({s.model_name for s in samples})
            if len(models) > 1:
                raise ValueError(f"{label} mixes multiple model names: {models!r}.")
            missing_lat = sum(1 for s in samples if s.latency_ms is None)
            id_sets[label] = {s.prompt_id for s in samples}
            captures_scanned += 1
            _, red_summary = redact_samples(samples, include_pii=False)
            secret_counts.update(red_summary.counts_by_kind)
            status: Literal["OK", "WARN"] = "WARN" if 0 < missing_lat < len(samples) else "OK"
            if missing_lat == 0:
                lat_note = "complete timings"
            elif missing_lat == len(samples):
                lat_note = "no timings"
            else:
                lat_note = f"{missing_lat} missing latency"
            checks.append(
                DoctorCheck(
                    key,
                    status,
                    f"{len(samples)} sample(s) for model {models[0]!r} ({lat_note})",
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(DoctorCheck(key, "FAIL", str(exc)))

    if captures_scanned > 0:
        if secret_counts:
            breakdown = ", ".join(f"{k}={v}" for k, v in sorted(secret_counts.items()))
            checks.append(
                DoctorCheck(
                    "secrets_scan",
                    "WARN",
                    f"Detected {sum(secret_counts.values())} potential credential(s) "
                    f"({breakdown}); run 'redact' before committing captures.",
                )
            )
        else:
            checks.append(
                DoctorCheck("secrets_scan", "OK", "No credentials detected in loaded captures.")
            )

    if len(id_sets) >= 2:
        items = list(id_sets.items())
        ref_label, ref_ids = items[0]
        aligned = True
        for other_label, other_ids in items[1:]:
            try:
                require_same_ids(ref_ids, other_ids, other_label)
            except ValueError as exc:
                checks.append(DoctorCheck("id_alignment", "FAIL", str(exc)))
                aligned = False
        if aligned:
            checks.append(
                DoctorCheck(
                    "id_alignment",
                    "OK",
                    f"Exact prompt_id alignment ({len(ref_ids)} IDs) across "
                    + ", ".join(id_sets),
                )
            )

    overall_ok = all(check.status != "FAIL" for check in checks)
    return DoctorReport(
        version=__version__,
        python=platform.python_version(),
        ok=overall_ok,
        scorers=scorers,
        extras=extras,
        checks=tuple(checks),
    )


def format_doctor_text(report: DoctorReport) -> str:
    """Render DoctorReport as a compact terminal diagnostic checklist."""
    verdict = "OK" if report.ok else "FAILED"
    extras_str = ", ".join(
        f"{name}={'yes' if present else 'no'}" for name, present in report.extras.items()
    )
    lines = [
        f"llm-drift-bench doctor [{verdict}]",
        f"  Optional/runtime packages: {extras_str}",
        "-" * 72,
    ]
    for item in report.checks:
        lines.append(f" [{item.status:<4}] {item.name:<14} | {item.detail}")
    lines.append("-" * 72)
    return "\n".join(lines)
