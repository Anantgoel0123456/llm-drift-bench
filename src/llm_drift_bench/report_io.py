"""Read saved reports without translating one scoring system into another."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import TypeAdapter

from llm_drift_bench.models import BenchmarkResult, DriftBenchmarkReport


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number is not allowed in reports: {value}.")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Saved report numbers must be finite.")
    return number


def load_report(path: str | Path) -> DriftBenchmarkReport | BenchmarkResult:
    """Load complete engine or legacy evaluator reports, preserving their model.

    Older complete engine reports may omit thresholds; these remain unknown.
    Incomplete or unrecognized reports are rejected instead of filling in scores.
    Legacy BenchmarkResult reports retain their historical scoring semantics.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    raw = json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
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
    return report
