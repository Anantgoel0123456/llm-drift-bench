"""Optional breakdowns for a validated multi-candidate comparison."""

from __future__ import annotations

from collections import defaultdict
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
