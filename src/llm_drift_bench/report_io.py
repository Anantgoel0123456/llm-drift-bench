"""Read saved reports without translating one scoring system into another."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from llm_drift_bench._file_io import parse_json, read_utf8
from llm_drift_bench.models import (
    BenchmarkResult,
    DriftBenchmarkReport,
    DriftThresholds,
    SampleDriftResult,
    SchemaValidationResult,
)

# Saved scores are rounded to four decimals, so recomputed aggregates can differ slightly.
_TOLERANCE = 5e-4
# Saved latency percentiles are rounded to two decimals (half a hundredth either way).
_LATENCY_ROUNDING_MS = 0.005


def _bad(message: str) -> ValueError:
    return ValueError(f"Saved report {message}")


def _check_unit(label: str, value: float | None, *, optional: bool = False) -> None:
    """Require a finite similarity, rate or drift score within [0, 1]."""
    if value is None:
        if optional:
            return
        raise _bad(f"{label} is required.")
    if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise _bad(f"{label} must be between 0 and 1.")


def _check_non_negative(label: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise _bad(f"{label} must be a finite non-negative number.")


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= _TOLERANCE


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _check_mean(label: str, actual: float, values: Sequence[float]) -> None:
    if not _close(actual, _mean(values)):
        raise _bad(f"{label} does not match its per-prompt results.")


def _check_optional_mean(label: str, actual: float | None, values: Sequence[float]) -> None:
    if not values:
        if actual is not None:
            raise _bad(f"{label} is present but no per-prompt result supplies it.")
        return
    if actual is None:
        raise _bad(f"{label} is missing although per-prompt results supply it.")
    _check_mean(label, actual, values)


def _check_validation(label: str, validation: SchemaValidationResult, raw: Any) -> None:
    if validation.is_schema_conformant and not validation.is_valid_json:
        raise _bad(f"{label} claims schema conformity for invalid JSON.")
    if validation.is_valid_json and validation.is_schema_conformant == bool(validation.issues):
        raise _bad(f"{label} conformity contradicts its validation issues.")
    if not validation.is_valid_json and not validation.issues:
        raise _bad(f"{label} reports invalid JSON without an issue.")
    if isinstance(raw, dict) and "passed" in raw and raw["passed"] is not validation.passed:
        raise _bad(f"{label} passed flag contradicts its validity fields.")


def _check_sample(
    result: SampleDriftResult, raw: Any, thresholds: DriftThresholds | None,
) -> None:
    where = f"result {result.prompt_id!r}"
    for name in (
        "token_jaccard_similarity", "ngram_similarity", "levenshtein_similarity",
        "composite_drift_score",
    ):
        _check_unit(f"{where} {name}", getattr(result, name))
    for name in ("json_structural_similarity", "json_exact_field_match", "json_numeric_drift"):
        _check_unit(f"{where} {name}", getattr(result, name), optional=True)
    raw_dict = raw if isinstance(raw, dict) else {}
    baseline, candidate = result.baseline_validation, result.candidate_validation
    _check_validation(f"{where} baseline validation", baseline, raw_dict.get("baseline_validation"))
    _check_validation(
        f"{where} candidate validation", candidate, raw_dict.get("candidate_validation"),
    )

    comparable = baseline.is_valid_json and candidate.is_valid_json
    if (result.json_structural_similarity is not None) != comparable or (
        (result.json_exact_field_match is not None) != comparable
    ):
        raise _bad(f"{where} JSON comparison scores contradict its validation results.")
    if result.json_numeric_drift is not None and not comparable:
        raise _bad(f"{where} numeric drift requires two valid JSON outputs.")

    regressed = baseline.passed and not candidate.passed
    if regressed and not result.drifted:
        raise _bad(f"{where} lost schema conformity but is not marked drifted.")
    if thresholds is None:
        return
    numeric = result.json_numeric_drift
    if not result.drifted and (
        result.composite_drift_score > thresholds.max_composite_drift + _TOLERANCE
        or (numeric is not None and numeric > thresholds.max_numeric_relative_drift + _TOLERANCE)
    ):
        raise _bad(f"{where} exceeds its drift thresholds but is not marked drifted.")
    if result.drifted and not (
        regressed
        or result.composite_drift_score >= thresholds.max_composite_drift - _TOLERANCE
        or (numeric is not None and numeric >= thresholds.max_numeric_relative_drift - _TOLERANCE)
    ):
        raise _bad(f"{where} is marked drifted without exceeding any drift threshold.")


def _allowed_rates(passed: int, valid: int, total: int) -> list[float]:
    """Rates the engine could have stored for this many passes.

    The denominator is every sample with a schema or valid JSON. The saved report does
    not record which prompts had schemas, so any denominator between the number of
    samples with valid JSON and the total is possible. No JSON expectation means 1.0.
    """
    rates = [passed / size for size in range(max(valid, 1), total + 1)]
    if valid == 0:
        rates.append(1.0)
    return rates


def _check_rate(label: str, rate: float, passed: int, valid: int, total: int) -> None:
    if not any(_close(rate, allowed) for allowed in _allowed_rates(passed, valid, total)):
        raise _bad(f"{label} does not match its per-prompt validation results.")


def _check_latency_summary(report: DriftBenchmarkReport) -> None:
    sides = (
        ("baseline", report.baseline_p50_latency_ms, report.baseline_p95_latency_ms),
        ("candidate", report.candidate_p50_latency_ms, report.candidate_p95_latency_ms),
    )
    for side, p50, p95 in sides:
        _check_non_negative(f"{side} P50 latency", p50)
        _check_non_negative(f"{side} P95 latency", p95)
        if (p50 is None) != (p95 is None):
            raise _bad(f"{side} latency percentiles must be present together.")
        if p50 is not None and p95 is not None and p50 > p95:
            raise _bad(f"{side} P50 latency exceeds its P95 latency.")
    complete = all(value is not None for _, p50, p95 in sides for value in (p50, p95))
    timed = all(result.latency_delta_ms is not None for result in report.results)
    if complete != timed:
        raise _bad("latency summary contradicts per-prompt latency deltas.")


def _check_pass_status(report: DriftBenchmarkReport) -> None:
    """A passing report cannot violate the thresholds it records."""
    thresholds = report.thresholds
    if not report.passed or thresholds is None:
        return
    if report.mean_composite_drift > thresholds.max_composite_drift + _TOLERANCE:
        raise _bad("passes although mean composite drift exceeds its threshold.")
    if report.candidate_json_conformity_rate < thresholds.min_json_conformity_rate - _TOLERANCE:
        raise _bad("passes although candidate JSON conformity is below its minimum.")
    numeric = report.mean_json_numeric_drift
    if numeric is not None and numeric > thresholds.max_numeric_relative_drift + _TOLERANCE:
        raise _bad("passes although mean numeric drift exceeds its threshold.")
    base, cand = report.baseline_p95_latency_ms, report.candidate_p95_latency_ms
    if base is not None and cand is not None and base > 0:
        floor_base = base + _LATENCY_ROUNDING_MS
        lowest = ((cand - _LATENCY_ROUNDING_MS - floor_base) / floor_base) * 100.0
        if lowest > thresholds.max_latency_regression_pct:
            raise _bad("passes although P95 latency regression exceeds its threshold.")


def _validate_engine_report(report: DriftBenchmarkReport, raw: dict[str, Any]) -> None:
    results = report.results
    if report.total_samples <= 0 or report.total_samples != len(results):
        raise ValueError("Saved report sample count must match its nonempty results.")
    ids = [result.prompt_id for result in results]
    if any(not pid.strip() for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Saved report prompt IDs must be nonempty and unique.")
    if report.drifted_samples != sum(result.drifted for result in results):
        raise ValueError("Saved report drifted sample count is inconsistent.")
    if report.passed != (not report.failure_reasons) or (report.passed and report.drifted_samples):
        raise ValueError("Saved report pass/fail status contradicts its failure evidence.")

    for name in (
        "baseline_json_conformity_rate", "candidate_json_conformity_rate",
        "mean_composite_drift", "max_composite_drift", "mean_token_jaccard",
        "mean_levenshtein_similarity",
    ):
        _check_unit(name, getattr(report, name))
    for name in ("mean_json_structural_similarity", "mean_json_numeric_drift"):
        _check_unit(name, getattr(report, name), optional=True)

    raw_results = raw.get("results")
    raw_items: list[Any] = raw_results if isinstance(raw_results, list) else []
    for index, result in enumerate(results):
        raw_item = raw_items[index] if index < len(raw_items) else None
        _check_sample(result, raw_item, report.thresholds)

    composites = [result.composite_drift_score for result in results]
    _check_mean("mean_composite_drift", report.mean_composite_drift, composites)
    if not _close(report.max_composite_drift, max(composites)):
        raise _bad("max_composite_drift does not match its per-prompt results.")
    _check_mean(
        "mean_token_jaccard", report.mean_token_jaccard,
        [result.token_jaccard_similarity for result in results],
    )
    _check_mean(
        "mean_levenshtein_similarity", report.mean_levenshtein_similarity,
        [result.levenshtein_similarity for result in results],
    )
    structural = [
        r.json_structural_similarity for r in results if r.json_structural_similarity is not None
    ]
    _check_optional_mean(
        "mean_json_structural_similarity", report.mean_json_structural_similarity, structural,
    )
    numeric = [r.json_numeric_drift for r in results if r.json_numeric_drift is not None]
    _check_optional_mean("mean_json_numeric_drift", report.mean_json_numeric_drift, numeric)

    valid = sum(
        result.baseline_validation.is_valid_json or result.candidate_validation.is_valid_json
        for result in results
    )
    total = len(results)
    _check_rate(
        "baseline_json_conformity_rate", report.baseline_json_conformity_rate,
        sum(result.baseline_validation.passed for result in results), valid, total,
    )
    _check_rate(
        "candidate_json_conformity_rate", report.candidate_json_conformity_rate,
        sum(result.candidate_validation.passed for result in results), valid, total,
    )
    _check_latency_summary(report)
    _check_pass_status(report)


def _validate_legacy_report(result: BenchmarkResult) -> None:
    metrics = result.metrics
    if result.total_prompts <= 0 or result.total_prompts != len(metrics):
        raise ValueError("Saved report prompt count must match its nonempty metrics.")
    ids = [metric.prompt_id for metric in metrics]
    if any(not pid.strip() for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Saved report prompt IDs must be nonempty and unique.")
    for metric in metrics:
        for label, delta, left, right in (
            ("latency", metric.latency_delta_ms, metric.candidate_latency_ms,
             metric.baseline_latency_ms),
            ("tokens per second", metric.tokens_per_sec_delta, metric.candidate_tokens_per_sec,
             metric.baseline_tokens_per_sec),
        ):
            expected = None if left is None or right is None else left - right
            if (expected is None) != (delta is None) or (
                expected is not None and delta is not None
                and not math.isclose(delta, expected, rel_tol=1e-6, abs_tol=1e-6)
            ):
                raise _bad(f"metric {metric.prompt_id!r} {label} delta is inconsistent.")
    for name, actual, values in (
        ("exact_match_rate", result.exact_match_rate, [float(m.exact_match) for m in metrics]),
        ("baseline_json_adherence_rate", result.baseline_json_adherence_rate,
         [float(m.baseline_valid_json) for m in metrics]),
        ("candidate_json_adherence_rate", result.candidate_json_adherence_rate,
         [float(m.format_adherence) for m in metrics]),
        ("mean_semantic_similarity", result.mean_semantic_similarity,
         [m.semantic_similarity for m in metrics]),
        ("mean_token_diff_ratio", result.mean_token_diff_ratio,
         [m.token_diff_ratio for m in metrics]),
        ("mean_drift_score", result.mean_drift_score, [m.drift_score for m in metrics]),
    ):
        _check_mean(name, actual, values)
    for name, optional_actual, optional_values in (
        ("mean_baseline_latency_ms", result.mean_baseline_latency_ms,
         [m.baseline_latency_ms for m in metrics]),
        ("mean_candidate_latency_ms", result.mean_candidate_latency_ms,
         [m.candidate_latency_ms for m in metrics]),
        ("mean_baseline_tokens_per_sec", result.mean_baseline_tokens_per_sec,
         [m.baseline_tokens_per_sec for m in metrics]),
        ("mean_candidate_tokens_per_sec", result.mean_candidate_tokens_per_sec,
         [m.candidate_tokens_per_sec for m in metrics]),
    ):
        measured = [value for value in optional_values if value is not None]
        complete = len(measured) == len(optional_values)
        if (optional_actual is None) == complete:
            raise _bad(f"{name} availability contradicts its per-prompt metrics.")
        if optional_actual is not None and not math.isclose(
            optional_actual, _mean(measured), rel_tol=1e-6, abs_tol=1e-6,
        ):
            raise _bad(f"{name} does not match its per-prompt metrics.")


def load_report(path: str | Path) -> DriftBenchmarkReport | BenchmarkResult:
    """Load complete engine or legacy evaluator reports, preserving their model.

    Older complete engine reports may omit thresholds; these remain unknown.
    Incomplete or unrecognized reports are rejected instead of filling in scores.
    Legacy BenchmarkResult reports retain their historical scoring semantics.

    Every per-prompt metric is range-checked and every summary value, pass/fail
    status and conformity rate must agree with the per-prompt results, so an edited
    file cannot carry a passing verdict that its own evidence contradicts.
    """
    text = read_utf8(path)
    raw = parse_json(text)
    if not isinstance(raw, dict):
        raise ValueError("A saved report must be a JSON object.")
    engine = "total_samples" in raw and "results" in raw
    legacy = "total_prompts" in raw and "metrics" in raw
    if engine == legacy:
        raise ValueError("Unrecognized or ambiguous saved report format.")
    if legacy:
        legacy_result = BenchmarkResult.model_validate_json(text, strict=True)
        _validate_legacy_report(legacy_result)
        return legacy_result
    adapter = TypeAdapter(DriftBenchmarkReport)
    report = adapter.validate_json(text, strict=True)
    _validate_engine_report(report, raw)
    return report
