"""Deterministic PII and credential redaction for captures and prompt suites."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
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


def redact_text(text: str, *, include_pii: bool = True) -> tuple[str, dict[str, int]]:
    """Redact credentials (and optionally PII) in text, returning (redacted, counts)."""
    counts: Counter[str] = Counter()
    result = text

    for kind, pattern in _SECRET_REDACTION_RULES:
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


def _redact_value(value: Any, *, include_pii: bool, counts: Counter[str]) -> Any:
    if isinstance(value, str):
        cleaned, found = redact_text(value, include_pii=include_pii)
        counts.update(found)
        return cleaned
    if isinstance(value, dict):
        return {
            k: _redact_value(v, include_pii=include_pii, counts=counts)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, include_pii=include_pii, counts=counts) for item in value]
    return value


def redact_samples(
    samples: Sequence[ModelSample],
    *,
    include_pii: bool = True,
) -> tuple[list[ModelSample], RedactionReport]:
    """Redact secrets and optional PII across ModelSample outputs and metadata."""
    overall: Counter[str] = Counter()
    redacted_list: list[ModelSample] = []
    redacted_samples = 0

    for sample in samples:
        sample_counts: Counter[str] = Counter()
        clean_out, out_found = redact_text(sample.output, include_pii=include_pii)
        sample_counts.update(out_found)
        clean_meta = _redact_value(sample.metadata, include_pii=include_pii, counts=sample_counts)
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
