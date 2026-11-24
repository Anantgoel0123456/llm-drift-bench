"""Deterministic PII and credential redaction for captures and prompt suites."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm_drift_bench.models import ModelSample

_SECRET_REDACTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    )),
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("OPENAI_KEY", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}\b")),
    ("GITHUB_PAT", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("BEARER_TOKEN", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{24,}\b")),
    (
        "JWT",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    (
        "DATABASE_URL",
        re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^/\s:@]*:[^/\s@]+@[^\s\"']+"
        ),
    ),
)

_PII_REDACTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IPV4", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|1?\d\d?)\b"
    )),
)

_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_valid(digits: str) -> bool:
    if not (13 <= len(digits) <= 19) or not digits.isdigit():
        return False
    total = 0
    for idx, ch in enumerate(reversed(digits)):
        num = int(ch)
        if idx % 2 == 1:
            num *= 2
            if num > 9:
                num -= 9
        total += num
    return total % 10 == 0


@dataclass(frozen=True)
class RedactionReport:
    """Summary of redactions applied across a capture collection."""

    total_samples: int
    redacted_samples: int
    total_replacements: int
    counts_by_kind: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "redacted_samples": self.redacted_samples,
            "total_replacements": self.total_replacements,
            "counts_by_kind": dict(self.counts_by_kind),
        }


_UNSAFE_CUSTOM_REGEX_RE = re.compile(
    r"\\[1-9]|\(\?[=!<]|\)[\s]*[*+?{]|[*+?}]\s*[*+?{]"
)


def _compile_custom_patterns(
    custom_patterns: Mapping[str, str] | None,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    if not custom_patterns:
        return ()
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for label, raw_pattern in sorted(custom_patterns.items()):
        if not isinstance(label, str) or not label.strip() or not re.fullmatch(
            r"[A-Za-z0-9_-]+", label.strip()
        ):
            raise ValueError(
                f"Custom redaction label must be alphanumeric/underscore/dash: {label!r}."
            )
        if not isinstance(raw_pattern, str) or not raw_pattern:
            raise ValueError(f"Custom redaction pattern for {label!r} must be a non-empty string.")
        if len(raw_pattern) > 256:
            raise ValueError(f"Custom redaction pattern for {label!r} exceeds 256 characters.")
        if _UNSAFE_CUSTOM_REGEX_RE.search(raw_pattern):
            raise ValueError(
                f"Unsafe custom redaction pattern for {label!r}: "
                "nested/group quantifiers, backreferences, and lookarounds are disallowed."
            )
        try:
            regex = re.compile(raw_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid custom redaction pattern for {label!r}: {exc}.") from exc
        compiled.append((label.strip().upper(), regex))
    return tuple(compiled)


def redact_text(
    text: str,
    *,
    include_pii: bool = True,
    custom_patterns: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, int]]:
    """Redact credentials (and optionally PII or safe custom regex rules) in text."""
    compiled_custom = _compile_custom_patterns(custom_patterns)
    return _redact_text_compiled(text, include_pii=include_pii, custom_rules=compiled_custom)


def _redact_text_compiled(
    text: str,
    *,
    include_pii: bool,
    custom_rules: tuple[tuple[str, re.Pattern[str]], ...],
) -> tuple[str, dict[str, int]]:
    counts: Counter[str] = Counter()
    result = text

    for kind, pattern in _SECRET_REDACTION_RULES:
        result, n = pattern.subn(f"[REDACTED:{kind}]", result)
        if n:
            counts[kind] += n

    for kind, pattern in custom_rules:
        result, n = pattern.subn(f"[REDACTED:{kind}]", result)
        if n:
            counts[kind] += n

    if include_pii:
        def _replace_card(match: re.Match[str]) -> str:
            raw = match.group(0)
            digits = "".join(ch for ch in raw if ch.isdigit())
            if _luhn_valid(digits):
                counts["CREDIT_CARD"] += 1
                return "[REDACTED:CREDIT_CARD]"
            return raw

        result = _CARD_CANDIDATE_RE.sub(_replace_card, result)
        for kind, pattern in _PII_REDACTION_RULES:
            result, n = pattern.subn(f"[REDACTED:{kind}]", result)
            if n:
                counts[kind] += n

    return result, dict(sorted(counts.items()))


def _redact_value(
    value: Any,
    *,
    include_pii: bool,
    custom_rules: tuple[tuple[str, re.Pattern[str]], ...],
    counts: Counter[str],
) -> Any:
    if isinstance(value, str):
        cleaned, found = _redact_text_compiled(
            value, include_pii=include_pii, custom_rules=custom_rules
        )
        counts.update(found)
        return cleaned
    if isinstance(value, dict):
        return {
            k: _redact_value(
                v, include_pii=include_pii, custom_rules=custom_rules, counts=counts
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _redact_value(
                item, include_pii=include_pii, custom_rules=custom_rules, counts=counts
            )
            for item in value
        ]
    return value


def redact_samples(
    samples: Sequence[ModelSample],
    *,
    include_pii: bool = True,
    custom_patterns: Mapping[str, str] | None = None,
) -> tuple[list[ModelSample], RedactionReport]:
    """Redact secrets and optional PII/custom patterns across ModelSample outputs and metadata."""
    custom_rules = _compile_custom_patterns(custom_patterns)
    overall: Counter[str] = Counter()
    redacted_list: list[ModelSample] = []
    redacted_samples = 0

    for sample in samples:
        sample_counts: Counter[str] = Counter()
        clean_out, out_found = _redact_text_compiled(
            sample.output, include_pii=include_pii, custom_rules=custom_rules
        )
        sample_counts.update(out_found)
        clean_meta = _redact_value(
            sample.metadata,
            include_pii=include_pii,
            custom_rules=custom_rules,
            counts=sample_counts,
        )
        if sample_counts:
            redacted_samples += 1
            overall.update(sample_counts)
        redacted_list.append(
            ModelSample(
                prompt_id=sample.prompt_id,
                output=clean_out,
                model_name=sample.model_name,
                latency_ms=sample.latency_ms,
                tokens_generated=sample.tokens_generated,
                metadata=clean_meta if isinstance(clean_meta, dict) else {},
            )
        )

    report = RedactionReport(
        total_samples=len(samples),
        redacted_samples=redacted_samples,
        total_replacements=sum(overall.values()),
        counts_by_kind=dict(sorted(overall.items())),
    )
    return redacted_list, report
