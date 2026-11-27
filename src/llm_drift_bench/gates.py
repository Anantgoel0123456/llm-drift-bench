"""Composite GO/NO-GO release readiness gate combining drift, trends, stats, and tool checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from llm_drift_bench.consumer import ConsumerComparison
from llm_drift_bench.conversation import ConversationDriftReport
from llm_drift_bench.distribution import SegmentedDistributionReport
from llm_drift_bench.embeddings import EmbeddingDriftReport
from llm_drift_bench.history import HistoryTrendCheck
from llm_drift_bench.models import DriftBenchmarkReport
from llm_drift_bench.stats import McNemarResult, PairedSignificance
from llm_drift_bench.tool_events import ToolTraceDiff
from llm_drift_bench.tool_log import ToolCheckRecord


@dataclass(frozen=True)
class GateCheck:
    """Single named decision inside an aggregated release verdict."""

    name: str
    passed: bool
    blocking: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseVerdict:
    """Auditable release GO / NO-GO summary across independent evidence sources."""

    approved: bool
    checks: tuple[GateCheck, ...]
    blocking_failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "checks": [c.to_dict() for c in self.checks],
            "blocking_failures": list(self.blocking_failures),
            "warnings": list(self.warnings),
        }


def evaluate_release_gates(
    report: DriftBenchmarkReport,
    *,
    history_check: HistoryTrendCheck | None = None,
    significance: PairedSignificance | None = None,
    mcnemar: McNemarResult | None = None,
    tool_checks: Sequence[ToolCheckRecord] = (),
    segmented_distribution: SegmentedDistributionReport | None = None,
    consumer_comparison: ConsumerComparison | None = None,
    embedding_report: EmbeddingDriftReport | None = None,
    conversation_report: ConversationDriftReport | None = None,
    tool_trace_diff: ToolTraceDiff | None = None,
    fail_on_history_regression: bool = True,
    fail_on_significant_degradation: bool = False,
    require_tool_checks_pass: bool = False,
    fail_on_widespread_shift: bool = False,
    fail_on_consumer_rejection: bool = True,
    fail_on_embedding_drift: bool = True,
    fail_on_conversation_drift: bool = True,
    fail_on_tool_trace_drift: bool = True,
) -> ReleaseVerdict:
    """Aggregate canonical drift, trend, statistical, and tool evidence into one verdict.

    Never mutates or overrides the underlying DriftBenchmarkReport scores or verdict.
    """
    checks: list[GateCheck] = [
        GateCheck(
            name="benchmark_drift",
            passed=report.passed,
            blocking=True,
            detail=(
                f"Passed ({report.drifted_samples}/{report.total_samples} drifted, "
                f"mean drift {report.mean_composite_drift:.4f})"
                if report.passed
                else "; ".join(report.failure_reasons) or "Benchmark thresholds failed."
            ),
        )
    ]

    if history_check is not None:
        checks.append(
            GateCheck(
                name="history_trend",
                passed=not history_check.regressed,
                blocking=fail_on_history_regression,
                detail=history_check.reason,
            )
        )

    if significance is not None:
        degraded = significance.significant and significance.observed_diff > 0
        checks.append(
            GateCheck(
                name="statistical_drift_shift",
                passed=not degraded,
                blocking=fail_on_significant_degradation,
                detail=(
                    f"diff={significance.observed_diff:+.4f}, "
                    f"d={significance.effect_size_d:+.2f}, p={significance.p_value:.4f}"
                ),
            )
        )

    if mcnemar is not None:
        conf_degraded = mcnemar.significant and mcnemar.b_regressed > mcnemar.c_recovered
        checks.append(
            GateCheck(
                name="mcnemar_conformity",
                passed=not conf_degraded,
                blocking=fail_on_significant_degradation,
                detail=(
                    f"regressed={mcnemar.b_regressed}, recovered={mcnemar.c_recovered}, "
                    f"p={mcnemar.p_value:.4f}"
                ),
            )
        )

    if tool_checks:
        non_pass = [tc for tc in tool_checks if tc.result.status != "PASS"]
        checks.append(
            GateCheck(
                name="tool_workflow_checks",
                passed=not non_pass,
                blocking=require_tool_checks_pass,
                detail=(
                    f"All {len(tool_checks)} tool check(s) passed."
                    if not non_pass
                    else f"{len(non_pass)}/{len(tool_checks)} tool check(s) not PASS "
                    f"({', '.join(f'{tc.resource}:{tc.result.status}' for tc in non_pass[:3])})."
                ),
            )
        )

    if segmented_distribution is not None:
        widespread = segmented_distribution.shift_scope == "widespread"
        checks.append(
            GateCheck(
                name="distribution_blast_radius",
                passed=not widespread,
                blocking=fail_on_widespread_shift,
                detail=(
                    f"scope={segmented_distribution.shift_scope} "
                    f"({len(segmented_distribution.shifted_segments)}/"
                    f"{len(segmented_distribution.segments)} segments >= "
                    f"{segmented_distribution.shift_threshold:.2f}, "
                    f"worst={segmented_distribution.worst_segment_name or 'n/a'})"
                ),
            )
        )

    if consumer_comparison is not None:
        checks.append(
            GateCheck(
                name="consumer_rejection",
                passed=consumer_comparison.passed,
                blocking=fail_on_consumer_rejection,
                detail=(
                    f"Passed (rejected {consumer_comparison.baseline_rejection_rate:.1%} -> "
                    f"{consumer_comparison.candidate_rejection_rate:.1%}, "
                    f"delta {consumer_comparison.rejection_rate_delta:+.1%})"
                    if consumer_comparison.passed
                    else "; ".join(consumer_comparison.failure_reasons)
                    or "Downstream consumer rejection threshold exceeded."
                ),
            )
        )

    if embedding_report is not None:
        mean_s = (
            "n/a"
            if embedding_report.mean_similarity is None
            else f"{embedding_report.mean_similarity:.4f}"
        )
        checks.append(
            GateCheck(
                name="embedding_similarity",
                passed=embedding_report.passed,
                blocking=fail_on_embedding_drift,
                detail=(
                    f"Passed (model={embedding_report.model}, mean_sim={mean_s}, "
                    f"{embedding_report.comparable_samples}/{embedding_report.total_samples})"
                    if embedding_report.passed
                    else "; ".join(embedding_report.failure_reasons)
                    or "Embedding similarity gate failed."
                ),
            )
        )

    if conversation_report is not None:
        checks.append(
            GateCheck(
                name="conversation_trajectory",
                passed=conversation_report.passed,
                blocking=fail_on_conversation_drift,
                detail=(
                    f"Passed ({conversation_report.total_conversations} conversation(s), "
                    f"mean_drift={conversation_report.mean_trajectory_drift:.4f})"
                    if conversation_report.passed
                    else "; ".join(conversation_report.failure_reasons)
                    or "Multi-turn conversation trajectory gate failed."
                ),
            )
        )

    if tool_trace_diff is not None:
        checks.append(
            GateCheck(
                name="tool_trace_drift",
                passed=tool_trace_diff.passed,
                blocking=fail_on_tool_trace_drift,
                detail=(
                    f"Passed (calls {tool_trace_diff.baseline_calls}->"
                    f"{tool_trace_diff.candidate_calls}, "
                    f"jaccard={tool_trace_diff.action_jaccard:.4f})"
                    if tool_trace_diff.passed
                    else "; ".join(tool_trace_diff.failure_reasons)
                    or "Agent tool-trace drift check failed."
                ),
            )
        )

    blocking_failures = tuple(
        f"{c.name}: {c.detail}" for c in checks if c.blocking and not c.passed
    )
    warnings = tuple(
        f"{c.name}: {c.detail}" for c in checks if not c.blocking and not c.passed
    )
    return ReleaseVerdict(
        approved=not blocking_failures,
        checks=tuple(checks),
        blocking_failures=blocking_failures,
        warnings=warnings,
    )
