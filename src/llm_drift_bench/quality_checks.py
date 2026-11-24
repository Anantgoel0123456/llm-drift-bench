"""Deterministic quantization & upgrade quality regression checks.

Detects failure modes commonly introduced by quantization or model upgrades:
- degenerate repetition loops (repeated lines or dense word n-gram loops)
- refusal flips (candidate refuses a prompt the baseline answered, or vice versa)
- sudden length truncation or verbosity blowup relative to baseline
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from llm_drift_bench._matching import index_records, require_same_ids
from llm_drift_bench.models import ModelSample

CheckKind = Literal["repetition_loop", "refusal_flip", "length_truncation", "length_blowup"]
Severity = Literal["warning", "error"]

_WORD_RE = re.compile(r"[a-z0-9']+")
_WHITESPACE_RE = re.compile(r"\s+")

_REFUSAL_PHRASES: tuple[str, ...] = (
    "i'm sorry, i cannot",
    "i'm sorry, but i cannot",
    "i'm sorry, i can't",
    "i'm sorry, but i can't",
    "i apologize, but i cannot",
    "i apologize, but i can't",
    "i cannot fulfill",
    "i can't fulfill",
    "i cannot assist with",
    "i can't assist with",
    "i cannot help with",
    "i can't help with",
    "i cannot provide",
    "i can't provide",
    "i cannot comply",
    "i am unable to",
    "i'm unable to",
    "as an ai language model",
    "as an ai, i cannot",
    "as an ai, i can't",
)


@dataclass(frozen=True)
class QualityFinding:
    """Single deterministic quality regression detected on a prompt pair."""

    prompt_id: str
    check: CheckKind
    severity: Severity
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "check": self.check,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class QualityReport:
    """Aggregate quality-regression check summary across a capture pair."""

    total_samples: int
    flagged_prompts: int
    findings: tuple[QualityFinding, ...]
    counts_by_check: dict[str, int]

    @property
    def passed(self) -> bool:
        return len(self.findings) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_samples": self.total_samples,
            "flagged_prompts": self.flagged_prompts,
            "counts_by_check": dict(self.counts_by_check),
            "findings": [item.to_dict() for item in self.findings],
        }


def detect_repetition_loop(
    text: str,
    *,
    ngram_words: int = 5,
    min_repeats: int = 4,
) -> tuple[bool, str | None]:
    """Detect degenerate consecutive line loops or dense word n-gram repetition loops."""
    if isinstance(ngram_words, bool) or ngram_words < 2:
        raise ValueError("ngram_words must be an integer >= 2.")
    if isinstance(min_repeats, bool) or min_repeats < 3:
        raise ValueError("min_repeats must be an integer >= 3.")

    # 1. Consecutive identical non-empty lines (>= 12 chars).
    prev_line: str | None = None
    streak = 0
    for raw_line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", raw_line.strip())
        if len(line) < 12:
            prev_line = None
            streak = 0
            continue
        if line == prev_line:
            streak += 1
            if streak >= min_repeats:
                preview = line[:60] + ("..." if len(line) > 60 else "")
                return True, f"Repeated line {streak}x consecutively: {preview!r}"
        else:
            prev_line = line
            streak = 1

    # 2. Dense repeated word n-grams.
    normalized = text.replace("\u2019", "'").lower()
    words = _WORD_RE.findall(normalized)
    if len(words) < ngram_words * min_repeats:
        return False, None

    positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx in range(len(words) - ngram_words + 1):
        gram = tuple(words[idx : idx + ngram_words])
        positions[gram].append(idx)

    best_gram: tuple[str, ...] | None = None
    best_count = 0
    for gram, locs in positions.items():
        if len(locs) < min_repeats:
            continue
        # Non-overlapping count (skip overlapping positions inside the same phrase).
        non_overlapping: list[int] = []
        for pos in locs:
            if not non_overlapping or pos >= non_overlapping[-1] + ngram_words:
                non_overlapping.append(pos)
        count = len(non_overlapping)
        if count < min_repeats:
            continue
        span = non_overlapping[-1] - non_overlapping[0]
        dense_burst = span <= count * ngram_words * 3
        if (dense_burst or count >= min_repeats + 2) and count > best_count:
            best_count = count
            best_gram = gram

    if best_gram is not None:
        phrase = " ".join(best_gram)
        return True, f"Repeated {ngram_words}-word phrase {best_count}x: {phrase!r}"
    return False, None


def is_likely_refusal(text: str) -> bool:
    """Return True if the opening of text matches a standard model refusal template."""
    normalized = _WHITESPACE_RE.sub(" ", text.replace("\u2019", "'").strip().lower())
    opening = normalized[:240]
    return any(phrase in opening for phrase in _REFUSAL_PHRASES)


def run_quality_checks(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
    *,
    max_length_ratio: float = 3.0,
    min_length_ratio: float = 0.25,
    min_baseline_chars: int = 40,
    ngram_words: int = 5,
    min_repeats: int = 4,
) -> QualityReport:
    """Run deterministic regression checks (loops, refusal flips, truncation, blowup)."""
    if not baseline_samples or not candidate_samples:
        raise ValueError("Both baseline and candidate sample collections must be non-empty.")
    if (
        isinstance(max_length_ratio, bool)
        or not math.isfinite(max_length_ratio)
        or max_length_ratio <= 1.0
    ):
        raise ValueError("max_length_ratio must be a finite number > 1.0.")
    if (
        isinstance(min_length_ratio, bool)
        or not math.isfinite(min_length_ratio)
        or not 0.0 < min_length_ratio < 1.0
    ):
        raise ValueError("min_length_ratio must be a finite number strictly between 0 and 1.")
    if isinstance(min_baseline_chars, bool) or min_baseline_chars < 1:
        raise ValueError("min_baseline_chars must be an integer >= 1.")

    base_map = index_records(baseline_samples, "baseline samples")
    cand_map = index_records(candidate_samples, "candidate samples")
    require_same_ids(set(base_map), set(cand_map), "Candidate")

    findings: list[QualityFinding] = []
    flagged_ids: set[str] = set()
    counts: Counter[str] = Counter()

    for prompt_id in sorted(base_map):
        base_text = base_map[prompt_id].output
        cand_text = cand_map[prompt_id].output

        base_loop, _ = detect_repetition_loop(
            base_text, ngram_words=ngram_words, min_repeats=min_repeats
        )
        cand_loop, cand_detail = detect_repetition_loop(
            cand_text, ngram_words=ngram_words, min_repeats=min_repeats
        )
        if cand_loop and not base_loop:
            findings.append(
                QualityFinding(
                    prompt_id=prompt_id,
                    check="repetition_loop",
                    severity="error",
                    detail=cand_detail or "Degenerate repetition loop detected.",
                )
            )
            flagged_ids.add(prompt_id)
            counts["repetition_loop"] += 1

        base_ref = is_likely_refusal(base_text)
        cand_ref = is_likely_refusal(cand_text)
        if not base_ref and cand_ref:
            findings.append(
                QualityFinding(
                    prompt_id=prompt_id,
                    check="refusal_flip",
                    severity="error",
                    detail="Candidate refused prompt where baseline answered.",
                )
            )
            flagged_ids.add(prompt_id)
            counts["refusal_flip"] += 1
        elif base_ref and not cand_ref:
            findings.append(
                QualityFinding(
                    prompt_id=prompt_id,
                    check="refusal_flip",
                    severity="warning",
                    detail="Candidate answered prompt where baseline refused.",
                )
            )
            flagged_ids.add(prompt_id)
            counts["refusal_flip"] += 1

        base_len = len(base_text.strip())
        cand_len = len(cand_text.strip())
        if base_len >= min_baseline_chars:
            ratio = cand_len / base_len
            if ratio < min_length_ratio:
                findings.append(
                    QualityFinding(
                        prompt_id=prompt_id,
                        check="length_truncation",
                        severity="error",
                        detail=(
                            f"Candidate output shrank to {cand_len} chars vs baseline "
                            f"{base_len} chars ({ratio:.2f}x < {min_length_ratio:.2f}x)."
                        ),
                    )
                )
                flagged_ids.add(prompt_id)
                counts["length_truncation"] += 1
            elif ratio > max_length_ratio:
                findings.append(
                    QualityFinding(
                        prompt_id=prompt_id,
                        check="length_blowup",
                        severity="warning",
                        detail=(
                            f"Candidate output grew to {cand_len} chars vs baseline "
                            f"{base_len} chars ({ratio:.2f}x > {max_length_ratio:.2f}x)."
                        ),
                    )
                )
                flagged_ids.add(prompt_id)
                counts["length_blowup"] += 1

    ordered_counts = {
        key: counts[key]
        for key in ("repetition_loop", "refusal_flip", "length_truncation", "length_blowup")
        if counts[key] > 0
    }
    return QualityReport(
        total_samples=len(base_map),
        flagged_prompts=len(flagged_ids),
        findings=tuple(findings),
        counts_by_check=ordered_counts,
    )


def format_quality_report_text(report: QualityReport) -> str:
    """Format a QualityReport as a clear human-readable text summary."""
    status = "PASSED (no regressions found)" if report.passed else "ISSUES DETECTED"
    lines = [
        f"Quality Regression Checks: {status} "
        f"({report.flagged_prompts}/{report.total_samples} prompts flagged)",
    ]
    if report.counts_by_check:
        summary = ", ".join(f"{k}={v}" for k, v in report.counts_by_check.items())
        lines.append(f"Counts: {summary}")
    if report.findings:
        lines.append("")
        for finding in report.findings:
            lines.append(
                f"  [{finding.severity.upper()}] {finding.prompt_id} ({finding.check}): "
                f"{finding.detail}"
            )
    return "\n".join(lines) + "\n"
