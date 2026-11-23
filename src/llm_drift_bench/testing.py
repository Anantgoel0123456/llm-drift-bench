"""Framework-independent assertions for captured-output regression tests."""

from __future__ import annotations

from collections.abc import Sequence

from llm_drift_bench.comparison import prompt_changes
from llm_drift_bench.models import DriftBenchmarkReport


def assert_no_drift(report: DriftBenchmarkReport) -> None:
    """Raise AssertionError if a canonical engine report fails its gates.

    Use with pytest, unittest, or ordinary Python. This function does not
    recompute scores, change thresholds, or include captured response bodies
    in its error message. A passing report can still have unavailable latency
    measurements; it is not a certification of model quality or performance.
    """
    if not isinstance(report, DriftBenchmarkReport):
        raise TypeError(
            "assert_no_drift requires a canonical DriftBenchmarkReport, "
            f"got {type(report).__name__}."
        )
    if report.total_samples <= 0:
        raise ValueError("Cannot assert drift on a report with no evaluated samples.")
    if report.passed:
        return

    lines = [
        f"LLM drift check failed: {report.drifted_samples}/{report.total_samples} "
        "samples drifted.",
        *[f"- {reason}" for reason in report.failure_reasons],
    ]
    failures = sorted(
        (result for result in report.results if result.drifted),
        key=lambda result: result.prompt_id,
    )
    for result in failures[:5]:
        lines.append(
            f"- Prompt {result.prompt_id!r}: drift={result.composite_drift_score:.4f}, "
            f"candidate schema passed={result.candidate_validation.passed}."
        )
    if len(failures) > 5:
        lines.append(f"- {len(failures) - 5} additional drifted samples; inspect report.results.")
    raise AssertionError("\n".join(lines))


def assert_no_regression(
    reports: Sequence[DriftBenchmarkReport],
    *,
    against: str,
) -> None:
    """Raise AssertionError if any prompt newly regressed relative to candidate `against`."""
    changes = prompt_changes(reports, against=against)
    regressions = [ch for ch in changes if ch.state == "regressed"]
    if not regressions:
        return
    lines = [
        f"LLM regression check failed: {len(regressions)} prompt(s) newly regressed "
        f"relative to {against!r}.",
    ]
    for item in regressions[:5]:
        lines.append(
            f"- Candidate {item.candidate!r} prompt {item.prompt_id!r}: "
            f"{item.reference_drift:.4f} -> {item.candidate_drift:.4f}"
        )
    if len(regressions) > 5:
        lines.append(f"- {len(regressions) - 5} additional regressions.")
    raise AssertionError("\n".join(lines))
