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
    max_drift: float
    p95_ms: float | None
    conformity: float
    worst_prompt_id: str | None


def compare_reports(reports: Sequence[DriftBenchmarkReport]) -> list[CandidateSummary]:
    """Order candidates by increasing mean composite drift, then by model name."""
    if len(reports) < 2:
        raise ValueError("Provide at least two engine reports.")
    reference = reports[0]
    expected = index_records(reference.results, "reference report")
    names: set[str] = set()
    rows = []
    for report in reports:
        if report.baseline_model != reference.baseline_model:
            raise ValueError("The baseline model must match across reports.")
        actual = index_records(report.results, report.candidate_model)
        require_same_ids(set(expected), set(actual), report.candidate_model)
        if any(row.category != expected[pid].category for pid, row in actual.items()):
            raise ValueError("Prompt categories differ across reports.")
        if report.thresholds is None or report.thresholds != reference.thresholds:
            raise ValueError("Reports must record identical drift thresholds.")
        if any(row.baseline_validation != expected[pid].baseline_validation
               for pid, row in actual.items()):
            raise ValueError("Baseline validation differs for a shared prompt.")
        if report.candidate_model in names:
            raise ValueError(f"Duplicate candidate name: {report.candidate_model!r}.")
        names.add(report.candidate_model)
        worst = min(actual.values(), key=lambda item: (-item.composite_drift_score, item.prompt_id),
                    default=None)
        rows.append(CandidateSummary(
            report.candidate_model, report.passed, report.total_samples, report.drifted_samples,
            report.mean_composite_drift,
            report.max_composite_drift,
            report.candidate_p95_latency_ms,
            report.candidate_json_conformity_rate,
            worst.prompt_id if worst is not None else None,
        ))
    return sorted(rows, key=lambda row: (row.mean_drift, row.candidate))


@dataclass(frozen=True)
class PromptChange:
    """A prompt whose individual drift verdict flipped between two candidates."""

    candidate: str
    prompt_id: str
    category: str
    state: str
    reference_drift: float
    candidate_drift: float


def prompt_changes(
    reports: Sequence[DriftBenchmarkReport], against: str,
) -> list[PromptChange]:
    """Return regressions and recoveries relative to a named candidate."""
    _, changes = compare_with_changes(reports, against)
    return changes


def compare_with_changes(
    reports: Sequence[DriftBenchmarkReport], against: str,
) -> tuple[list[CandidateSummary], list[PromptChange]]:
    """Compute rankings and prompt transitions with a single compatibility check."""
    ranking = compare_reports(reports)
    by_name = {report.candidate_model: report for report in reports}
    if against not in by_name:
        raise ValueError(f"Unknown reference candidate: {against!r}.")
    reference = index_records(by_name[against].results, against)
    changes = []
    for name, report in sorted(by_name.items()):
        if name == against:
            continue
        for row in sorted(report.results, key=lambda item: item.prompt_id):
            if reference[row.prompt_id].drifted == row.drifted:
                continue
            state = "regressed" if row.drifted else "recovered"
            changes.append(PromptChange(name, row.prompt_id, row.category, state,
                                        reference[row.prompt_id].composite_drift_score,
                                        row.composite_drift_score))
    return ranking, changes
