"""Evaluate multiple labeled captures with one baseline and one threshold policy."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from llm_drift_bench.engine import DriftBenchmarkEngine
from llm_drift_bench.models import DriftBenchmarkReport, DriftThresholds, ModelSample, PromptCase
from llm_drift_bench.policy import DriftPolicy


def evaluate_many(
    baseline: Sequence[ModelSample],
    candidates: Iterable[tuple[str, Sequence[ModelSample]]],
    suite: Sequence[PromptCase] | None = None,
    thresholds: DriftThresholds | None = None,
    *,
    policy: DriftPolicy | None = None,
) -> list[DriftBenchmarkReport]:
    """Evaluate each capture once, preserving its caller-supplied comparison label."""
    if len({sample.model_name for sample in baseline}) > 1:
        raise ValueError("Baseline capture contains multiple model names.")
    engine = DriftBenchmarkEngine(thresholds, policy=policy)
    labels: set[str] = set()
    reports = []
    for label, samples in candidates:
        if not label.strip() or label in labels:
            raise ValueError(f"Candidate labels must be non-empty and unique: {label!r}.")
        labels.add(label)
        if len({sample.model_name for sample in samples}) > 1:
            raise ValueError(f"Candidate {label!r} contains multiple model names.")
        try:
            report = engine.evaluate(baseline, samples, suite)
        except ValueError as exc:
            raise ValueError(f"Candidate {label!r}: {exc}") from exc
        reports.append(replace(report, candidate_model=label))
    if len(reports) < 2:
        raise ValueError("Provide at least two labeled candidate captures.")
    return reports
