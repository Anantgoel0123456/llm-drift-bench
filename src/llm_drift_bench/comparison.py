"""Compare candidates evaluated against the same reference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llm_drift_bench._matching import index_records, require_same_ids
from llm_drift_bench.models import DriftBenchmarkReport


@dataclass(frozen=True)
class CandidateSummary:
    """Recorded metrics for one candidate in a comparison."""

    candidate: str
    passed: bool
    samples: int
    drifted: int
    mean_drift: float


def compare_reports(reports: Sequence[DriftBenchmarkReport]) -> list[CandidateSummary]:
    """Order candidates by increasing mean composite drift, then by model name."""
    if len(reports) < 2:
        raise ValueError("Provide at least two engine reports.")
    reference = reports[0]
    expected = index_records(reference.results, "reference report")
    rows = []
    for report in reports:
        if report.baseline_model != reference.baseline_model:
            raise ValueError("The baseline model must match across reports.")
        actual = index_records(report.results, report.candidate_model)
        require_same_ids(set(expected), set(actual), report.candidate_model)
        if any(row.category != expected[pid].category for pid, row in actual.items()):
            raise ValueError("Prompt categories differ across reports.")
        rows.append(CandidateSummary(
            report.candidate_model, report.passed, report.total_samples, report.drifted_samples,
            report.mean_composite_drift,
        ))
    return sorted(rows, key=lambda row: (row.mean_drift, row.candidate))
