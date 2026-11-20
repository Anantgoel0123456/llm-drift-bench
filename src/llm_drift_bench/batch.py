"""Evaluate multiple labeled captures with one baseline and one threshold policy."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from llm_drift_bench.engine import DriftBenchmarkEngine
from llm_drift_bench.models import DriftBenchmarkReport, DriftThresholds, ModelSample, PromptCase


def evaluate_many(
    baseline: Sequence[ModelSample],
    candidates: Iterable[tuple[str, Sequence[ModelSample]]],
    suite: Sequence[PromptCase] | None = None,
    thresholds: DriftThresholds | None = None,
) -> list[DriftBenchmarkReport]:
    """Evaluate each capture once, preserving its caller-supplied comparison label."""
    engine = DriftBenchmarkEngine(thresholds)
    reports = []
    for label, samples in candidates:
        report = engine.evaluate(baseline, samples, suite)
        reports.append(replace(report, candidate_model=label))
    if len(reports) < 2:
        raise ValueError("Provide at least two labeled candidate captures.")
    return reports
