"""Standalone deterministic semantic format checks for JSON fields (date, email, uuid, uri, ...).

Kept intentionally separate from validator.py so the strict JSON Schema subset contract
remains unchanged while users can opt into common semantic format checks on exact or [*]
wildcard JSONPaths.
"""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from llm_drift_bench._matching import index_records
from llm_drift_bench._report_views import markdown_cell
from llm_drift_bench.metrics import flatten_json_paths
from llm_drift_bench.models import ModelSample
from llm_drift_bench.policy import _compile_path_pattern
from llm_drift_bench.validator import extract_json_payload

FormatKind = Literal["date", "date-time", "email", "uri", "uuid", "ipv4", "iso-currency"]

SUPPORTED_FORMATS: tuple[FormatKind, ...] = (
    "date",
    "date-time",
    "email",
    "uri",
    "uuid",
    "ipv4",
    "iso-currency",
)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})?$"
)
_EMAIL_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class FormatIssue:
    """Single semantic format violation found in a captured sample."""

    prompt_id: str
    path: str
    expected_format: FormatKind
    actual_preview: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "path": self.path,
            "expected_format": self.expected_format,
            "actual_preview": self.actual_preview,
            "message": self.message,
        }


@dataclass(frozen=True)
class FormatReport:
    """Aggregate result of semantic field format checks across a capture set."""

    total_samples: int
    checked_values: int
    failed_values: int
    issues: tuple[FormatIssue, ...]

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_samples": self.total_samples,
            "checked_values": self.checked_values,
            "failed_values": self.failed_values,
            "issues": [item.to_dict() for item in self.issues],
        }


def validate_format_value(value: Any, fmt: FormatKind) -> tuple[bool, str | None]:
    """Validate a single scalar JSON value against one supported semantic format."""
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}. Choose from {SUPPORTED_FORMATS}.")
    if not isinstance(value, str):
        return False, f"Expected string for format {fmt!r}, got {type(value).__name__}."
    text = value.strip()
    if not text:
        return False, f"Empty string is not a valid {fmt}."

    if fmt == "date":
        if not _DATE_RE.fullmatch(text):
            return False, "Expected ISO-8601 calendar date YYYY-MM-DD."
        try:
            date.fromisoformat(text)
        except ValueError as exc:
            return False, f"Invalid calendar date: {exc}."
        return True, None

    if fmt == "date-time":
        if not _DATETIME_RE.fullmatch(text):
            return False, "Expected ISO-8601 date-time (e.g. 2026-11-25T10:30:00Z)."
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            datetime.fromisoformat(normalized)
        except ValueError as exc:
            return False, f"Invalid ISO-8601 date-time: {exc}."
        return True, None

    if fmt == "email":
        if len(text) > 254 or ".." in text or not _EMAIL_RE.fullmatch(text):
            return False, "Expected valid email address (local@domain.tld)."
        return True, None

    if fmt == "uri":
        if any(ch.isspace() for ch in text):
            return False, "URI must not contain whitespace."
        parsed = urllib.parse.urlparse(text)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False, "Expected absolute http:// or https:// URI with host."
        return True, None

    if fmt == "uuid":
        if not _UUID_RE.fullmatch(text):
            return False, "Expected canonical hyphenated UUID (8-4-4-4-12 hex)."
        return True, None

    if fmt == "ipv4":
        try:
            ipaddress.IPv4Address(text)
        except ValueError:
            return False, "Expected dotted-quad IPv4 address."
        return True, None

    # iso-currency
    if not _CURRENCY_RE.fullmatch(text):
        return False, "Expected 3-letter uppercase ISO-4217 currency code (e.g. USD, EUR)."
    return True, None


def parse_format_rule_spec(raw: str) -> tuple[str, FormatKind]:
    """Parse a CLI rule 'JSONPATH=FORMAT' (e.g. '$.email=email' or '$.items[*].date=date')."""
    if "=" not in raw:
        raise ValueError(f"Format rule must use 'JSONPATH=FORMAT' syntax: {raw!r}.")
    path, fmt = raw.split("=", 1)
    path = path.strip()
    fmt_clean = fmt.strip().lower()
    if not path.startswith("$"):
        raise ValueError(f"Format rule path must start with '$', got {path!r}.")
    if fmt_clean not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format {fmt_clean!r} for {path!r}. Choose from {SUPPORTED_FORMATS}."
        )
    return path, fmt_clean


def _preview_value(value: Any) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= 48 else rendered[:45] + "..."


def check_field_formats(
    samples: Sequence[ModelSample],
    field_formats: Mapping[str, FormatKind],
    *,
    allow_null: bool = False,
) -> FormatReport:
    """Validate specified JSONPath fields (exact or [*] wildcard) against semantic formats."""
    if not samples:
        raise ValueError("Sample collection cannot be empty.")
    if not field_formats:
        raise ValueError("At least one JSONPath=format rule is required.")

    compiled_rules: list[tuple[str, FormatKind, re.Pattern[str] | None]] = []
    for path, fmt in sorted(field_formats.items()):
        if not isinstance(path, str) or not path.startswith("$"):
            raise ValueError(f"Field format path must start with '$', got {path!r}.")
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format {fmt!r} for {path!r}. Choose from {SUPPORTED_FORMATS}."
            )
        pattern = _compile_path_pattern(path, allow_subtree=False) if "[*]" in path else None
        compiled_rules.append((path, fmt, pattern))

    sample_map = index_records(samples, "samples")
    checked = 0
    issues: list[FormatIssue] = []

    for prompt_id in sorted(sample_map):
        ok, payload, _ = extract_json_payload(sample_map[prompt_id].output)
        if not ok:
            continue
        flat = flatten_json_paths(payload)
        for concrete_path in sorted(flat):
            val = flat[concrete_path]
            for rule_path, fmt, pattern in compiled_rules:
                matched = (
                    pattern.match(concrete_path) is not None
                    if pattern is not None
                    else concrete_path == rule_path
                )
                if not matched:
                    continue
                if allow_null and val is None:
                    continue
                checked += 1
                valid, err = validate_format_value(val, fmt)
                if not valid:
                    issues.append(
                        FormatIssue(
                            prompt_id=prompt_id,
                            path=concrete_path,
                            expected_format=fmt,
                            actual_preview=_preview_value(val),
                            message=err or f"Invalid {fmt} value.",
                        )
                    )

    return FormatReport(
        total_samples=len(sample_map),
        checked_values=checked,
        failed_values=len(issues),
        issues=tuple(issues),
    )


def format_check_report_text(report: FormatReport) -> str:
    """Format a FormatReport as a clear human-readable text summary."""
    status = "PASSED" if report.passed else "FAILED"
    lines = [
        f"Semantic Field Format Check: {status} "
        f"({report.checked_values} value(s) checked across {report.total_samples} sample(s), "
        f"{report.failed_values} failed)",
    ]
    for issue in report.issues:
        lines.append(
            f"  {issue.prompt_id} {issue.path} [{issue.expected_format}]: "
            f"{issue.message} (got {issue.actual_preview})"
        )
    return "\n".join(lines) + "\n"


def format_check_report_markdown(report: FormatReport) -> str:
    """Format a FormatReport as GitHub-Flavored Markdown."""
    verdict = "PASS" if report.passed else "FAIL"
    lines = [
        f"## Semantic Field Format Check ({verdict})",
        "",
        f"Checked **{report.checked_values}** value(s) across **{report.total_samples}** "
        f"sample(s); **{report.failed_values}** failed.",
    ]
    if report.issues:
        lines += [
            "",
            "| Prompt ID | JSONPath | Expected | Got | Message |",
            "| :--- | :--- | :---: | :--- | :--- |",
        ]
        for issue in report.issues:
            lines.append(
                f"| `{markdown_cell(issue.prompt_id)}` | `{markdown_cell(issue.path)}` | "
                f"`{issue.expected_format}` | `{markdown_cell(issue.actual_preview)}` | "
                f"{markdown_cell(issue.message)} |"
            )
    return "\n".join(lines) + "\n"
