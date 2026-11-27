"""Pre-flight environment, policy, schema, and capture alignment checks (doctor)."""

from __future__ import annotations

import importlib.util
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from llm_drift_bench import __version__
from llm_drift_bench._file_io import read_utf8
from llm_drift_bench._matching import require_same_ids
from llm_drift_bench.adapters import load_adapted_tool_events
from llm_drift_bench.consumer import load_consumer_decisions
from llm_drift_bench.conversation import load_conversations
from llm_drift_bench.engine import load_model_samples, load_prompt_cases
from llm_drift_bench.history import HISTORY_SCHEMA_VERSION
from llm_drift_bench.models import REPORT_SCHEMA_VERSION
from llm_drift_bench.policy import load_policy
from llm_drift_bench.prompt_lint import lint_prompt_suite
from llm_drift_bench.redaction import redact_samples
from llm_drift_bench.scorers import list_scorers
from llm_drift_bench.tool_events import check_no_secret_in_tool_args
from llm_drift_bench.tool_log import load_tool_events


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
    report_schema_version: int = REPORT_SCHEMA_VERSION
    history_schema_version: int = HISTORY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "python": self.python,
            "ok": self.ok,
            "report_schema_version": self.report_schema_version,
            "history_schema_version": self.history_schema_version,
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
    baseline_decisions_path: str | Path | None = None,
    candidate_decisions_path: str | Path | None = None,
    baseline_conversations_path: str | Path | None = None,
    candidate_conversations_path: str | Path | None = None,
    baseline_trace_path: str | Path | None = None,
    candidate_trace_path: str | Path | None = None,
    trace_adapter: Literal["normalized", "openai", "anthropic", "gemini", "otel"] = "normalized",
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
            f"llm-drift-bench {__version__} on Python {platform.python_version()} "
            f"(report_schema=v{REPORT_SCHEMA_VERSION}, "
            f"history_schema=v{HISTORY_SCHEMA_VERSION})",
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
            blank_outputs = sum(1 for s in samples if not s.output.strip())
            id_sets[label] = {s.prompt_id for s in samples}
            captures_scanned += 1
            _, red_summary = redact_samples(samples, include_pii=False)
            secret_counts.update(red_summary.counts_by_kind)
            status: Literal["OK", "WARN"] = (
                "WARN" if (0 < missing_lat < len(samples) or blank_outputs > 0) else "OK"
            )
            if missing_lat == 0:
                lat_note = "complete timings"
            elif missing_lat == len(samples):
                lat_note = "no timings"
            else:
                lat_note = f"{missing_lat} missing latency"
            blank_note = f", {blank_outputs} blank output(s)" if blank_outputs else ""
            checks.append(
                DoctorCheck(
                    key,
                    status,
                    f"{len(samples)} sample(s) for model {models[0]!r} ({lat_note}{blank_note})",
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(DoctorCheck(key, "FAIL", str(exc)))

    for label, key, target in (
        ("Baseline decisions", "baseline_decisions", baseline_decisions_path),
        ("Candidate decisions", "candidate_decisions", candidate_decisions_path),
    ):
        if target is None:
            continue
        try:
            decisions = load_consumer_decisions(target)
            rejected = sum(1 for d in decisions if not d.accepted)
            id_sets[label] = {d.prompt_id for d in decisions}
            checks.append(
                DoctorCheck(
                    key,
                    "OK",
                    f"{len(decisions)} decision(s) loaded ({rejected} rejected)",
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

    conv_id_sets: dict[str, set[str]] = {}
    for label, key, target in (
        ("Baseline conversations", "baseline_conversations", baseline_conversations_path),
        ("Candidate conversations", "candidate_conversations", candidate_conversations_path),
    ):
        if target is None:
            continue
        try:
            records = load_conversations(target)
            assistant_turns = sum(len(rec.assistant_turns()) for rec in records)
            conv_id_sets[label] = {rec.conversation_id for rec in records}
            conv_status: Literal["OK", "WARN"] = "WARN" if assistant_turns == 0 else "OK"
            checks.append(
                DoctorCheck(
                    key,
                    conv_status,
                    f"{len(records)} conversation(s) loaded "
                    f"({assistant_turns} assistant turn(s))",
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(DoctorCheck(key, "FAIL", str(exc)))

    if len(conv_id_sets) == 2:
        conv_items = list(conv_id_sets.items())
        ref_conv_label, ref_conv_ids = conv_items[0]
        other_conv_label, other_conv_ids = conv_items[1]
        try:
            require_same_ids(ref_conv_ids, other_conv_ids, other_conv_label)
            checks.append(
                DoctorCheck(
                    "conversation_alignment",
                    "OK",
                    f"Exact conversation_id alignment ({len(ref_conv_ids)} IDs) across "
                    f"{ref_conv_label}, {other_conv_label}",
                )
            )
        except ValueError as exc:
            checks.append(DoctorCheck("conversation_alignment", "FAIL", str(exc)))

    trace_texts: list[str] = []
    for key, target in (
        ("baseline_trace", baseline_trace_path),
        ("candidate_trace", candidate_trace_path),
    ):
        if target is None:
            continue
        try:
            events = (
                load_tool_events(target)
                if trace_adapter == "normalized"
                else load_adapted_tool_events(target, adapter=trace_adapter)
            )
            reads = sum(1 for e in events if e.kind == "read")
            edits = sum(1 for e in events if e.kind == "edit")
            failed = sum(1 for e in events if not e.success)
            trace_status: Literal["OK", "WARN"] = "WARN" if not events else "OK"
            checks.append(
                DoctorCheck(
                    key,
                    trace_status,
                    f"{len(events)} tool event(s) ({trace_adapter}: "
                    f"{reads} read, {edits} edit, {failed} failed)",
                )
            )
            trace_texts.append(read_utf8(target))
        except (OSError, ValueError) as exc:
            checks.append(DoctorCheck(key, "FAIL", str(exc)))

    if trace_texts:
        sec_res = check_no_secret_in_tool_args(trace_texts, trace_complete=True)
        if sec_res.status == "FAIL":
            checks.append(
                DoctorCheck(
                    "trace_secrets",
                    "WARN",
                    f"{sec_res.reason} Scrub credentials before committing traces.",
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    "trace_secrets", "OK", "No credentials detected in loaded tool traces."
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
        report_schema_version=REPORT_SCHEMA_VERSION,
        history_schema_version=HISTORY_SCHEMA_VERSION,
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
