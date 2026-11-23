"""Compare candidates evaluated against the same reference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from llm_drift_bench._matching import index_records, require_same_ids
from llm_drift_bench.models import DriftBenchmarkReport

SortKey = Literal["drift", "passed", "conformity", "latency"]
_VALID_SORT_KEYS = ("drift", "passed", "conformity", "latency")


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


def _sort_rows(rows: list[CandidateSummary], sort_by: SortKey) -> list[CandidateSummary]:
    if sort_by not in _VALID_SORT_KEYS:
        raise ValueError(f"Unsupported sort_by {sort_by!r}; choose from {_VALID_SORT_KEYS}.")
    if sort_by == "passed":
        return sorted(rows, key=lambda row: (not row.passed, row.mean_drift, row.candidate))
    if sort_by == "conformity":
        return sorted(rows, key=lambda row: (-row.conformity, row.mean_drift, row.candidate))
    if sort_by == "latency":
        return sorted(
            rows,
            key=lambda row: (
                row.p95_ms is None,
                row.p95_ms if row.p95_ms is not None else float("inf"),
                row.mean_drift,
                row.candidate,
            ),
        )
    return sorted(rows, key=lambda row: (row.mean_drift, row.candidate))


def compare_reports(
    reports: Sequence[DriftBenchmarkReport],
    *,
    sort_by: SortKey = "drift",
) -> list[CandidateSummary]:
    """Order candidates by the selected metric (default: increasing mean drift, then name)."""
    if sort_by not in _VALID_SORT_KEYS:
        raise ValueError(f"Unsupported sort_by {sort_by!r}; choose from {_VALID_SORT_KEYS}.")
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
    return _sort_rows(rows, sort_by)


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
    reports: Sequence[DriftBenchmarkReport],
    against: str,
    *,
    sort_by: SortKey = "drift",
) -> tuple[list[CandidateSummary], list[PromptChange]]:
    """Compute rankings and prompt transitions with a single compatibility check."""
    ranking = compare_reports(reports, sort_by=sort_by)
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


@dataclass(frozen=True)
class PromptHotspot:
    """Frequency and peak drift among candidates that failed this prompt."""

    prompt_id: str
    category: str
    drifted_candidates: int
    max_drift: float


def _hotspots(reports: Sequence[DriftBenchmarkReport], limit: int) -> list[PromptHotspot]:
    if isinstance(limit, bool) or limit < 1:
        raise ValueError("--hotspots must be positive.")
    counts: dict[str, PromptHotspot] = {}
    for report in reports:
        for row in report.results:
            if row.drifted:
                previous = counts.get(row.prompt_id)
                counts[row.prompt_id] = PromptHotspot(
                    row.prompt_id, row.category,
                    1 + (previous.drifted_candidates if previous else 0),
                    max(previous.max_drift, row.composite_drift_score)
                    if previous else row.composite_drift_score,
                )
    return sorted(counts.values(), key=lambda h: (
        -h.drifted_candidates, -h.max_drift, h.prompt_id,
    ))[:limit]


@dataclass(frozen=True)
class PromptInspection:
    """Candidate-specific measurements and validation paths for a selected prompt."""

    prompt_id: str
    category: str
    candidate: str
    drifted: bool
    composite_drift: float
    numeric_drift: float | None
    latency_delta_ms: float | None
    valid_json: bool
    issues: tuple[tuple[str, str], ...]


def _inspect_prompts(
    reports: Sequence[DriftBenchmarkReport], prompt_ids: Sequence[str],
) -> list[PromptInspection]:
    requested = set(prompt_ids)
    inspections = []
    for report in reports:
        for row in report.results:
            if row.prompt_id in requested:
                validation = row.candidate_validation
                inspections.append(PromptInspection(
                    row.prompt_id, row.category, report.candidate_model, row.drifted,
                    row.composite_drift_score, row.json_numeric_drift, row.latency_delta_ms,
                    validation.is_valid_json,
                    tuple((issue.path, issue.rule) for issue in validation.issues),
                ))
    missing = requested.difference(row.prompt_id for row in inspections)
    if missing:
        raise ValueError(f"Unknown prompt IDs: {sorted(missing)!r}.")
    return sorted(inspections, key=lambda row: (row.prompt_id, -row.composite_drift, row.candidate))
