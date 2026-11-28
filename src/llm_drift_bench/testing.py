"""Framework-independent assertions for captured-output regression tests."""

from __future__ import annotations

import math
from collections.abc import Sequence

from llm_drift_bench.audit import AuditReport
from llm_drift_bench.comparison import prompt_changes
from llm_drift_bench.consumer import ConsumerComparison
from llm_drift_bench.conversation import ConversationDriftReport
from llm_drift_bench.cost import CostComparison, check_cost_budget
from llm_drift_bench.embeddings import EmbeddingDriftReport
from llm_drift_bench.formats import FormatReport
from llm_drift_bench.gates import ReleaseVerdict
from llm_drift_bench.logprobs import LogprobDriftSummary
from llm_drift_bench.models import DriftBenchmarkReport
from llm_drift_bench.quality_checks import QualityReport
from llm_drift_bench.rolling import RollingWindowReport
from llm_drift_bench.tool_events import ToolTraceDiff


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


def assert_no_tool_trace_drift(diff: ToolTraceDiff) -> None:
    """Raise AssertionError if an agent ToolTraceDiff fails its gates."""
    if not isinstance(diff, ToolTraceDiff):
        raise TypeError(
            f"assert_no_tool_trace_drift requires a ToolTraceDiff, got {type(diff).__name__}."
        )
    if diff.passed:
        return
    lines = [
        f"LLM agent tool-trace drift failed (calls {diff.baseline_calls}->{diff.candidate_calls}, "
        f"failed {diff.baseline_failed}->{diff.candidate_failed}, "
        f"jaccard={diff.action_jaccard:.4f}).",
        *[f"- {reason}" for reason in diff.failure_reasons],
    ]
    raise AssertionError("\n".join(lines))


def assert_audit_passed(report: AuditReport) -> None:
    """Raise AssertionError if a unified offline AuditReport failed."""
    if not isinstance(report, AuditReport):
        raise TypeError(
            f"assert_audit_passed requires an AuditReport, got {type(report).__name__}."
        )
    if report.passed:
        return
    lines = [
        f"LLM unified audit failed ({len(report.failure_reasons)} failure reason(s); "
        f"mean_drift={report.benchmark.mean_composite_drift:.4f}).",
        *[f"- {reason}" for reason in report.failure_reasons],
    ]
    raise AssertionError("\n".join(lines))


def assert_no_consumer_regression(comparison: ConsumerComparison) -> None:
    """Raise AssertionError if a downstream ConsumerComparison failed its rejection gates."""
    if not isinstance(comparison, ConsumerComparison):
        raise TypeError(
            "assert_no_consumer_regression requires a ConsumerComparison, "
            f"got {type(comparison).__name__}."
        )
    if comparison.passed:
        return
    lines = [
        f"LLM downstream consumer check failed "
        f"(rejected {comparison.baseline_rejection_rate:.1%} -> "
        f"{comparison.candidate_rejection_rate:.1%}, "
        f"delta {comparison.rejection_rate_delta:+.1%}).",
        *[f"- {reason}" for reason in comparison.failure_reasons],
    ]
    raise AssertionError("\n".join(lines))


def assert_no_quality_regressions(report: QualityReport) -> None:
    """Raise AssertionError if a QualityReport flagged any loop, refusal, or length anomaly."""
    if not isinstance(report, QualityReport):
        raise TypeError(
            "assert_no_quality_regressions requires a QualityReport, "
            f"got {type(report).__name__}."
        )
    if report.passed:
        return
    summary = ", ".join(f"{k}={v}" for k, v in report.counts_by_check.items())
    lines = [
        f"LLM quality regression check failed: "
        f"{report.flagged_prompts}/{report.total_samples} prompt(s) flagged ({summary}).",
    ]
    for finding in report.findings[:5]:
        lines.append(f"- Prompt {finding.prompt_id!r} ({finding.check}): {finding.detail}")
    if len(report.findings) > 5:
        lines.append(f"- {len(report.findings) - 5} additional quality finding(s).")
    raise AssertionError("\n".join(lines))


def assert_no_rolling_drift(report: RollingWindowReport) -> None:
    """Raise AssertionError if a RollingWindowReport exceeded drift or jitter thresholds."""
    if not isinstance(report, RollingWindowReport):
        raise TypeError(
            "assert_no_rolling_drift requires a RollingWindowReport, "
            f"got {type(report).__name__}."
        )
    if report.passed:
        return
    lines = [
        f"LLM rolling-baseline distribution check failed "
        f"({report.window_count} prior window(s), "
        f"max_tvd={report.overall.max_tvd:.4f}, "
        f"max_w1={report.overall.max_normalized_wasserstein:.4f}).",
        *[f"- {reason}" for reason in report.failure_reasons],
    ]
    raise AssertionError("\n".join(lines))


def assert_formats_valid(report: FormatReport) -> None:
    """Raise AssertionError if a FormatReport contains semantic field format violations."""
    if not isinstance(report, FormatReport):
        raise TypeError(
            f"assert_formats_valid requires a FormatReport, got {type(report).__name__}."
        )
    if report.passed:
        return
    lines = [
        f"LLM semantic format check failed: "
        f"{report.failed_values}/{report.checked_values} value(s) failed "
        f"({len(report.issues)} issue(s)).",
    ]
    for issue in report.issues[:5]:
        lines.append(
            f"- Prompt {issue.prompt_id!r} field {issue.path} "
            f"({issue.expected_format}): {issue.message}"
        )
    if len(report.issues) > 5:
        lines.append(f"- {len(report.issues) - 5} additional format issue(s).")
    raise AssertionError("\n".join(lines))


def assert_logprobs_within_budget(
    summary: LogprobDriftSummary,
    *,
    max_entropy_delta: float | None = None,
    max_top_k_jsd: float | None = None,
) -> None:
    """Raise AssertionError if token entropy or top-k JSD exceeds configured limits."""
    if not isinstance(summary, LogprobDriftSummary):
        raise TypeError(
            "assert_logprobs_within_budget requires a LogprobDriftSummary, "
            f"got {type(summary).__name__}."
        )
    if max_entropy_delta is not None and (
        isinstance(max_entropy_delta, bool)
        or not isinstance(max_entropy_delta, (int, float))
        or not math.isfinite(max_entropy_delta)
        or max_entropy_delta < 0.0
    ):
        raise ValueError("max_entropy_delta must be a finite non-negative number.")
    if max_top_k_jsd is not None and (
        isinstance(max_top_k_jsd, bool)
        or not isinstance(max_top_k_jsd, (int, float))
        or not math.isfinite(max_top_k_jsd)
        or not (0.0 <= max_top_k_jsd <= 1.0)
    ):
        raise ValueError("max_top_k_jsd must be a finite number in [0.0, 1.0].")
    if max_entropy_delta is None and max_top_k_jsd is None:
        return

    failures: list[str] = []
    if not summary.complete:
        failures.append(
            f"Incomplete logprobs coverage ({summary.comparable_samples}/{summary.total_samples})."
        )
    if (
        max_entropy_delta is not None
        and summary.mean_entropy_delta_bits is not None
        and abs(summary.mean_entropy_delta_bits) > max_entropy_delta
    ):
        failures.append(
            f"abs(mean_entropy_delta_bits) {abs(summary.mean_entropy_delta_bits):.4f} "
            f"exceeded limit {max_entropy_delta:.4f}."
        )
    if (
        max_top_k_jsd is not None
        and summary.max_top_k_jsd is not None
        and summary.max_top_k_jsd > max_top_k_jsd
    ):
        failures.append(
            f"max_top_k_jsd {summary.max_top_k_jsd:.4f} exceeded limit {max_top_k_jsd:.4f}."
        )
    if failures:
        lines = [
            f"LLM logprobs drift budget failed "
            f"({summary.comparable_samples}/{summary.total_samples} comparable):",
            *(f"- {msg}" for msg in failures),
        ]
        raise AssertionError("\n".join(lines))


def assert_cost_within_budget(
    comparison: CostComparison,
    *,
    max_token_increase_pct: float | None = None,
    max_cost_increase_pct: float | None = None,
) -> None:
    """Raise AssertionError if explicit output token or USD cost growth exceeds budget."""
    ok, detail = check_cost_budget(
        comparison,
        max_token_increase_pct=max_token_increase_pct,
        max_cost_increase_pct=max_cost_increase_pct,
    )
    if not ok:
        raise AssertionError(f"LLM cost/token budget failed: {detail}")
