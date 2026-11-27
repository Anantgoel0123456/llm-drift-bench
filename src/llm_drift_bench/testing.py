"""Framework-independent assertions for captured-output regression tests."""

from __future__ import annotations

from collections.abc import Sequence

from llm_drift_bench.comparison import prompt_changes
from llm_drift_bench.conversation import ConversationDriftReport
from llm_drift_bench.embeddings import EmbeddingDriftReport
from llm_drift_bench.gates import ReleaseVerdict
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


def assert_no_conversation_drift(report: ConversationDriftReport) -> None:
    """Raise AssertionError if a multi-turn ConversationDriftReport fails its gates."""
    if not isinstance(report, ConversationDriftReport):
        raise TypeError(
            "assert_no_conversation_drift requires a ConversationDriftReport, "
            f"got {type(report).__name__}."
        )
    if report.passed:
        return
    lines = [
        f"LLM conversation trajectory drift failed: "
        f"{report.drifted_conversations}/{report.total_conversations} conversation(s) drifted.",
        *[f"- {reason}" for reason in report.failure_reasons],
    ]
    failed = [c for c in report.conversations if c.drifted]
    for item in failed[:5]:
        lines.append(
            f"- Conversation {item.conversation_id!r}: max_drift={item.max_turn_drift:.4f}, "
            f"first_div={item.first_divergence_turn}, refusal_flips={len(item.refusal_flip_turns)}."
        )
    if len(failed) > 5:
        lines.append(f"- {len(failed) - 5} additional drifted conversation(s).")
    raise AssertionError("\n".join(lines))


def assert_no_embedding_drift(report: EmbeddingDriftReport) -> None:
    """Raise AssertionError if an EmbeddingDriftReport fails its similarity thresholds."""
    if not isinstance(report, EmbeddingDriftReport):
        raise TypeError(
            "assert_no_embedding_drift requires an EmbeddingDriftReport, "
            f"got {type(report).__name__}."
        )
    if report.passed:
        return
    lines = [
        f"LLM embedding similarity check failed (model={report.model!r}, "
        f"comparable={report.comparable_samples}/{report.total_samples}).",
        *[f"- {reason}" for reason in report.failure_reasons],
    ]
    raise AssertionError("\n".join(lines))


def assert_release_approved(verdict: ReleaseVerdict) -> None:
    """Raise AssertionError if a composite ReleaseVerdict is blocked (NO-GO)."""
    if not isinstance(verdict, ReleaseVerdict):
        raise TypeError(
            "assert_release_approved requires a ReleaseVerdict, "
            f"got {type(verdict).__name__}."
        )
    if verdict.approved:
        return
    lines = [
        f"LLM release gate NO-GO: {len(verdict.blocking_failures)} blocking failure(s).",
        *[f"- {reason}" for reason in verdict.blocking_failures],
    ]
    raise AssertionError("\n".join(lines))
