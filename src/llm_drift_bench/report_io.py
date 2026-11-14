"""Read saved reports without translating one scoring system into another."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from llm_drift_bench._file_io import parse_json, read_utf8
from llm_drift_bench.models import BenchmarkResult, DriftBenchmarkReport


def load_report(path: str | Path) -> DriftBenchmarkReport | BenchmarkResult:
    """Load complete engine or legacy evaluator reports, preserving their model.

    Older complete engine reports may omit thresholds; these remain unknown.
    Incomplete or unrecognized reports are rejected instead of filling in scores.
    Legacy BenchmarkResult reports retain their historical scoring semantics.
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
        return BenchmarkResult.model_validate_json(text, strict=True)
    adapter = TypeAdapter(DriftBenchmarkReport)
    report = adapter.validate_json(text, strict=True)
    if report.total_samples <= 0 or report.total_samples != len(report.results):
        raise ValueError("Saved report sample count must match its nonempty results.")
    ids = [result.prompt_id for result in report.results]
    if any(not pid.strip() for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Saved report prompt IDs must be nonempty and unique.")
    if report.drifted_samples != sum(result.drifted for result in report.results):
        raise ValueError("Saved report drifted sample count is inconsistent.")
    if report.passed != (not report.failure_reasons) or (report.passed and report.drifted_samples):
        raise ValueError("Saved report pass/fail status contradicts its failure evidence.")
    for name in (
        "baseline_json_conformity_rate", "candidate_json_conformity_rate",
        "mean_composite_drift", "max_composite_drift", "mean_token_jaccard",
        "mean_levenshtein_similarity", "mean_json_structural_similarity",
        "mean_json_numeric_drift",
    ):
        value = getattr(report, name)
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"Saved report {name} must be between 0 and 1.")
    return report
