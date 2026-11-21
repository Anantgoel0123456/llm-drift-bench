"""Optional breakdowns for a validated multi-candidate comparison."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from llm_drift_bench.models import DriftBenchmarkReport, SampleDriftResult


@dataclass(frozen=True)
class CategorySummary:
    candidate: str
    category: str
    samples: int
    drifted: int
    mean_drift: float


def category_summary(reports: Sequence[DriftBenchmarkReport]) -> list[CategorySummary]:
    groups: dict[tuple[str, str], list[SampleDriftResult]] = defaultdict(list)
    for report in reports:
        for row in report.results:
            groups[report.candidate_model, row.category].append(row)
    return [
        CategorySummary(candidate, category, len(rows), sum(row.drifted for row in rows),
                        fmean(row.composite_drift_score for row in rows))
        for (candidate, category), rows in sorted(groups.items())
    ]


@dataclass(frozen=True)
class RuleCount:
    candidate: str
    rule: str
    affected_prompts: int
    occurrences: int


def rule_counts(reports: Sequence[DriftBenchmarkReport]) -> list[RuleCount]:
    counts: Counter[tuple[str, str]] = Counter()
    prompts: Counter[tuple[str, str]] = Counter()
    for report in reports:
        for row in report.results:
            rules = Counter(issue.rule for issue in row.candidate_validation.issues)
            for rule, count in rules.items():
                key = (report.candidate_model, rule)
                counts[key] += count
                prompts[key] += 1
    return [RuleCount(*key, prompts[key], counts[key])
            for key in sorted(counts, key=lambda k: (-prompts[k], -counts[k], k))]


@dataclass(frozen=True)
class LatencyIncrease:
    candidate: str
    prompt_id: str
    delta_ms: float


def latency_increases(reports: Sequence[DriftBenchmarkReport], limit: int) -> list[LatencyIncrease]:
    if limit < 1:
        raise ValueError("--slowdowns must be positive.")
    rows = [LatencyIncrease(report.candidate_model, row.prompt_id, row.latency_delta_ms)
            for report in reports for row in report.results
            if row.latency_delta_ms is not None and row.latency_delta_ms > 0]
    return sorted(rows, key=lambda item: (-item.delta_ms, item.candidate, item.prompt_id))[:limit]
