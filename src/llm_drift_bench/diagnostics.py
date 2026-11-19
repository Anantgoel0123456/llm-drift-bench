"""Explain the recorded verdict of an offline benchmark report."""

from __future__ import annotations

from dataclasses import dataclass

from llm_drift_bench.models import DriftBenchmarkReport


@dataclass(frozen=True)
class Finding:
    """One observed condition and a suggested next check."""

    code: str
    prompt_id: str | None
    evidence: str
    suggestion: str


def diagnose_report(report: DriftBenchmarkReport) -> list[Finding]:
    """Return actionable observations without recomputing the benchmark verdict."""
    findings: list[Finding] = []
    for result in report.results:
        if not result.drifted:
            continue
        findings.append(Finding(
            "drift", result.prompt_id, f"Composite drift {result.composite_drift_score:.4f}",
            "Compare the saved baseline and candidate responses for this prompt.",
        ))
    return findings
