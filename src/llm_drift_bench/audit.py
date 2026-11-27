"""Unified offline audit combining drift, distributions, quality, logprobs, and cost."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from llm_drift_bench._report_views import markdown_cell, measurement
from llm_drift_bench.consumer import ConsumerComparison
from llm_drift_bench.conversation import ConversationDriftReport
from llm_drift_bench.cost import CostComparison, ModelPricing, estimate_cost_drift
from llm_drift_bench.distribution import (
    DistributionReport,
    SegmentedDistributionReport,
    analyze_field_distributions,
    analyze_segmented_distributions,
)
from llm_drift_bench.embeddings import EmbeddingDriftReport
from llm_drift_bench.engine import DriftBenchmarkEngine
from llm_drift_bench.logprobs import LogprobDriftSummary, analyze_logprobs_drift
from llm_drift_bench.models import (
    DriftBenchmarkReport,
    DriftThresholds,
    ModelSample,
    PromptCase,
)
from llm_drift_bench.policy import DriftPolicy
from llm_drift_bench.quality_checks import QualityReport, run_quality_checks
from llm_drift_bench.tool_events import ToolTraceDiff


def _validate_optional_unit(name: str, val: float | None) -> None:
    if val is not None and (
        isinstance(val, bool) or not math.isfinite(val) or not 0.0 <= val <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between 0.0 and 1.0.")


@dataclass(frozen=True)
class AuditReport:
    """Comprehensive offline regression audit combining all five deterministic checks."""

    benchmark: DriftBenchmarkReport
    distribution: DistributionReport
    quality: QualityReport
    logprobs: LogprobDriftSummary
    cost: CostComparison
    passed: bool
    failure_reasons: tuple[str, ...]
    segmented_distribution: SegmentedDistributionReport | None = None
    consumer: ConsumerComparison | None = None
    embeddings: EmbeddingDriftReport | None = None
    conversations: ConversationDriftReport | None = None
    tool_trace: ToolTraceDiff | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "benchmark": self.benchmark.to_dict(),
            "distribution": self.distribution.to_dict(),
            "quality": self.quality.to_dict(),
            "logprobs": self.logprobs.to_dict(),
            "cost": self.cost.to_dict(),
        }
        if self.segmented_distribution is not None:
            payload["segmented_distribution"] = self.segmented_distribution.to_dict()
        if self.consumer is not None:
            payload["consumer"] = self.consumer.to_dict()
        if self.embeddings is not None:
            payload["embeddings"] = self.embeddings.to_dict()
        if self.conversations is not None:
            payload["conversations"] = self.conversations.to_dict()
        if self.tool_trace is not None:
            payload["tool_trace"] = self.tool_trace.to_dict()
        return payload


def run_full_audit(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
    *,
    prompt_cases: Sequence[PromptCase] | None = None,
    policy: DriftPolicy | None = None,
    thresholds: DriftThresholds | None = None,
    max_tvd: float | None = None,
    max_jsd: float | None = None,
    max_wasserstein: float | None = None,
    max_entropy_delta: float | None = None,
    max_top_k_jsd: float | None = None,
    fail_on_quality_issues: bool = True,
    baseline_pricing: ModelPricing | None = None,
    candidate_pricing: ModelPricing | None = None,
    group_by: str | None = None,
    shift_threshold: float = 0.10,
    fail_on_widespread_shift: bool = False,
    consumer_comparison: ConsumerComparison | None = None,
    embedding_report: EmbeddingDriftReport | None = None,
    conversation_report: ConversationDriftReport | None = None,
    tool_trace_diff: ToolTraceDiff | None = None,
) -> AuditReport:
    """Run canonical drift, field distribution, quality, logprobs, and cost checks in one pass."""
    _validate_optional_unit("max_tvd", max_tvd)
    _validate_optional_unit("max_jsd", max_jsd)
    _validate_optional_unit("max_wasserstein", max_wasserstein)
    _validate_optional_unit("max_top_k_jsd", max_top_k_jsd)
    if max_entropy_delta is not None and (
        isinstance(max_entropy_delta, bool)
        or not math.isfinite(max_entropy_delta)
        or max_entropy_delta < 0.0
    ):
        raise ValueError("max_entropy_delta must be a finite non-negative number.")

    engine = DriftBenchmarkEngine(thresholds=thresholds, policy=policy)
    benchmark = engine.evaluate(
        baseline_samples=baseline_samples,
        candidate_samples=candidate_samples,
        prompt_cases=prompt_cases,
    )
    segmented: SegmentedDistributionReport | None = None
    if group_by is not None:
        segmented = analyze_segmented_distributions(
            baseline_samples,
            candidate_samples,
            group_by=group_by,
            prompt_cases=prompt_cases,
            policy=policy,
            shift_threshold=shift_threshold,
        )
        distribution = segmented.overall
    else:
        distribution = analyze_field_distributions(
            baseline_samples,
            candidate_samples,
            policy=policy,
        )
    quality = run_quality_checks(baseline_samples, candidate_samples)
    logprobs = analyze_logprobs_drift(baseline_samples, candidate_samples)
    cost = estimate_cost_drift(
        baseline_samples,
        candidate_samples,
        baseline_pricing=baseline_pricing,
        candidate_pricing=candidate_pricing,
    )

    failures: list[str] = []
    if not benchmark.passed:
        failures.extend(benchmark.failure_reasons)
    if max_tvd is not None and distribution.max_tvd > max_tvd:
        failures.append(
            f"Distribution max_tvd ({distribution.max_tvd:.4f}) exceeded limit ({max_tvd:.4f})."
        )
    if max_jsd is not None and distribution.max_jsd > max_jsd:
        failures.append(
            f"Distribution max_jsd ({distribution.max_jsd:.4f}) exceeded limit ({max_jsd:.4f})."
        )
    if max_wasserstein is not None and distribution.max_normalized_wasserstein > max_wasserstein:
        failures.append(
            f"Distribution max_normalized_wasserstein "
            f"({distribution.max_normalized_wasserstein:.4f}) exceeded limit "
            f"({max_wasserstein:.4f})."
        )
    if fail_on_quality_issues and not quality.passed:
        summary = ", ".join(f"{k}={v}" for k, v in quality.counts_by_check.items())
        failures.append(
            f"Quality regressions flagged on {quality.flagged_prompts} prompt(s) ({summary})."
        )
    lp_limited = max_entropy_delta is not None or max_top_k_jsd is not None
    if lp_limited and not logprobs.complete:
        failures.append(
            f"Logprobs coverage incomplete ({logprobs.comparable_samples}/{logprobs.total_samples})"
            " for configured logprobs threshold."
        )
    if (
        max_entropy_delta is not None
        and logprobs.mean_entropy_delta_bits is not None
        and abs(logprobs.mean_entropy_delta_bits) > max_entropy_delta
    ):
        failures.append(
            f"Logprobs |mean_entropy_delta_bits| ({abs(logprobs.mean_entropy_delta_bits):.4f}) "
            f"exceeded limit ({max_entropy_delta:.4f})."
        )
    if (
        max_top_k_jsd is not None
        and logprobs.max_top_k_jsd is not None
        and logprobs.max_top_k_jsd > max_top_k_jsd
    ):
        failures.append(
            f"Logprobs max_top_k_jsd ({logprobs.max_top_k_jsd:.4f}) "
            f"exceeded limit ({max_top_k_jsd:.4f})."
        )
    if (
        fail_on_widespread_shift
        and segmented is not None
        and segmented.shift_scope == "widespread"
    ):
        failures.append(
            f"Widespread distribution shift across {len(segmented.shifted_segments)}/"
            f"{len(segmented.segments)} segments ({', '.join(segmented.shifted_segments)})."
        )
    if consumer_comparison is not None and not consumer_comparison.passed:
        failures.extend(
            consumer_comparison.failure_reasons
            or ("Downstream consumer validation check failed.",)
        )
    if embedding_report is not None and not embedding_report.passed:
        failures.extend(
            embedding_report.failure_reasons
            or ("Embedding similarity drift check failed.",)
        )
    if conversation_report is not None and not conversation_report.passed:
        failures.extend(
            conversation_report.failure_reasons
            or ("Multi-turn conversation trajectory check failed.",)
        )
    if tool_trace_diff is not None and not tool_trace_diff.passed:
        failures.extend(
            tool_trace_diff.failure_reasons or ("Agent tool-trace drift check failed.",)
        )

    return AuditReport(
        benchmark=benchmark,
        distribution=distribution,
        quality=quality,
        logprobs=logprobs,
        cost=cost,
        passed=len(failures) == 0,
        failure_reasons=tuple(failures),
        segmented_distribution=segmented,
        consumer=consumer_comparison,
        embeddings=embedding_report,
        conversations=conversation_report,
        tool_trace=tool_trace_diff,
    )


def format_audit_text(report: AuditReport) -> str:
    """Render a concise executive text summary of an AuditReport."""
    b = report.benchmark
    d = report.distribution
    q = report.quality
    lp = report.logprobs
    c = report.cost
    state = "PASSED" if report.passed else "FAILED"
    tok_line = (
        f"{c.baseline_tokens} -> {c.candidate_tokens} ({c.token_delta:+d})"
        if c.complete and c.token_delta is not None
        else "n/a (incomplete explicit tokens_generated)"
    )
    lp_line = (
        f"entropy delta={measurement(lp.mean_entropy_delta_bits, '+.4f')} bits, "
        f"max top-k JSD={measurement(lp.max_top_k_jsd)}"
        if lp.comparable_samples > 0
        else "n/a (no logprobs metadata)"
    )
    lines = [
        f"LLM Drift Bench Unified Audit [{state}]",
        f"Models       : {b.baseline_model} -> {b.candidate_model} ({b.total_samples} prompt(s))",
        f"Per-prompt   : {'PASS' if b.passed else 'FAIL'} | "
        f"mean drift={b.mean_composite_drift:.4f}, max={b.max_composite_drift:.4f}, "
        f"JSON conformity={b.candidate_json_conformity_rate:.1%}",
        f"Distribution : max TVD={d.max_tvd:.4f}, max JSD={d.max_jsd:.4f}, "
        f"max norm W1={d.max_normalized_wasserstein:.4f} "
        f"({d.comparable_json_prompts} JSON prompts)",
        f"Quality      : {'PASS' if q.passed else 'ISSUES'} "
        f"({q.flagged_prompts}/{q.total_samples} prompts flagged)",
        f"Logprobs     : {lp_line}",
        f"Token volume : {tok_line}",
    ]
    if report.segmented_distribution is not None:
        seg = report.segmented_distribution
        lines.append(
            f"Blast radius : {seg.shift_scope} "
            f"({len(seg.shifted_segments)}/{len(seg.segments)} segments "
            f">= {seg.shift_threshold:.2f}, group_by={seg.group_by})"
        )
    if report.consumer is not None:
        cons = report.consumer
        lines.append(
            f"Consumer     : {'PASS' if cons.passed else 'FAIL'} | "
            f"rejected {cons.baseline_rejection_rate:.1%} -> {cons.candidate_rejection_rate:.1%} "
            f"({cons.rejection_rate_delta:+.1%})"
        )
    if report.embeddings is not None:
        emb = report.embeddings
        lines.append(
            f"Embeddings   : {'PASS' if emb.passed else 'FAIL'} | "
            f"mean cosine={measurement(emb.mean_similarity, '.4f')}, "
            f"min={measurement(emb.min_similarity, '.4f')} "
            f"({emb.comparable_samples}/{emb.total_samples})"
        )
    if report.conversations is not None:
        conv = report.conversations
        lines.append(
            f"Conversations: {'PASS' if conv.passed else 'FAIL'} | "
            f"drifted {conv.drifted_conversations}/{conv.total_conversations}, "
            f"mean drift={conv.mean_trajectory_drift:.4f}, "
            f"refusal flips={conv.total_refusal_flip_turns}"
        )
    if report.tool_trace is not None:
        tt = report.tool_trace
        lines.append(
            f"Tool trace   : {'PASS' if tt.passed else 'FAIL'} | "
            f"calls {tt.baseline_calls} -> {tt.candidate_calls} ({tt.call_delta:+d}), "
            f"failed {tt.baseline_failed} -> {tt.candidate_failed}, "
            f"jaccard={tt.action_jaccard:.4f}"
        )
    if report.failure_reasons:
        lines.append("")
        lines.append("Failure reasons:")
        for reason in report.failure_reasons:
            lines.append(f"  - {reason}")
    return "\n".join(lines) + "\n"


def format_audit_markdown(report: AuditReport) -> str:
    """Render a GitHub-Flavored Markdown summary of an AuditReport."""
    b = report.benchmark
    d = report.distribution
    q = report.quality
    lp = report.logprobs
    c = report.cost
    state = "PASSED" if report.passed else "FAILED"
    tok_val = (
        f"{c.baseline_tokens} -> {c.candidate_tokens} ({c.token_delta:+d})"
        if c.complete and c.token_delta is not None
        else "n/a"
    )
    lp_val = (
        f"dH={measurement(lp.mean_entropy_delta_bits, '+.4f')}b, "
        f"max JSD={measurement(lp.max_top_k_jsd)}"
        if lp.comparable_samples > 0
        else "n/a"
    )
    lines = [
        f"# Unified Drift Audit: {state}",
        "",
        f"Baseline: `{markdown_cell(b.baseline_model)}` | "
        f"Candidate: `{markdown_cell(b.candidate_model)}` | Samples: {b.total_samples}",
        "",
        "| Check | Status | Key Metrics |",
        "| --- | --- | --- |",
        f"| Per-prompt drift | {'PASS' if b.passed else 'FAIL'} | "
        f"mean={b.mean_composite_drift:.4f}, max={b.max_composite_drift:.4f}, "
        f"conformity={b.candidate_json_conformity_rate:.1%} |",
        f"| Field distribution | INFO | max TVD={d.max_tvd:.4f}, max JSD={d.max_jsd:.4f}, "
        f"max norm W1={d.max_normalized_wasserstein:.4f} |",
        f"| Quality checks | {'PASS' if q.passed else 'FAIL'} | "
        f"{q.flagged_prompts}/{q.total_samples} prompt(s) flagged |",
        f"| Token logprobs | {'COMPLETE' if lp.complete else 'N/A'} | {lp_val} |",
        f"| Token volume | {'COMPLETE' if c.complete else 'N/A'} | {tok_val} |",
    ]
    if report.segmented_distribution is not None:
        seg = report.segmented_distribution
        lines.append(
            f"| Blast radius (`{markdown_cell(seg.group_by)}`) | {markdown_cell(seg.shift_scope)} "
            f"| {len(seg.shifted_segments)}/{len(seg.segments)} segments &gt;= "
            f"{seg.shift_threshold:.2f} |"
        )
    if report.consumer is not None:
        cons = report.consumer
        lines.append(
            f"| Downstream consumer | {'PASS' if cons.passed else 'FAIL'} "
            f"| rejected {cons.baseline_rejection_rate:.1%} -&gt; "
            f"{cons.candidate_rejection_rate:.1%} ({cons.rejection_rate_delta:+.1%}) |"
        )
    if report.embeddings is not None:
        emb = report.embeddings
        lines.append(
            f"| Embedding similarity | {'PASS' if emb.passed else 'FAIL'} "
            f"| mean={measurement(emb.mean_similarity, '.4f')}, "
            f"min={measurement(emb.min_similarity, '.4f')} "
            f"({emb.comparable_samples}/{emb.total_samples}) |"
        )
    if report.conversations is not None:
        conv = report.conversations
        lines.append(
            f"| Multi-turn conversations | {'PASS' if conv.passed else 'FAIL'} "
            f"| drifted {conv.drifted_conversations}/{conv.total_conversations}, "
            f"mean={conv.mean_trajectory_drift:.4f}, "
            f"refusal flips={conv.total_refusal_flip_turns} |"
        )
    if report.tool_trace is not None:
        tt = report.tool_trace
        lines.append(
            f"| Agent tool trace | {'PASS' if tt.passed else 'FAIL'} "
            f"| calls {tt.baseline_calls} -&gt; {tt.candidate_calls} ({tt.call_delta:+d}), "
            f"jaccard={tt.action_jaccard:.4f}, "
            f"new edits={len(tt.added_edit_resources)} |"
        )
    if report.failure_reasons:
        lines += ["", "## Failure reasons", ""]
        lines.extend(f"- {markdown_cell(r)}" for r in report.failure_reasons)
    if q.findings:
        lines += [
            "",
            "## Quality findings",
            "",
            "| Prompt ID | Check | Severity | Detail |",
            "| --- | --- | --- | --- |",
        ]
        for finding in q.findings:
            lines.append(
                "| "
                + " | ".join(
                    (
                        markdown_cell(finding.prompt_id),
                        markdown_cell(finding.check),
                        markdown_cell(finding.severity),
                        markdown_cell(finding.detail),
                    )
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def export_audit_html(
    report: AuditReport,
    output_path: str | Path | None = None,
) -> str:
    """Export a standalone, script-free offline HTML executive audit dashboard."""
    b = report.benchmark
    d = report.distribution
    q = report.quality
    lp = report.logprobs
    c = report.cost
    state = "PASSED" if report.passed else "FAILED"
    accent = "#6ee7b7" if report.passed else "#fda4af"

    prompt_rows = "".join(
        "<tr>"
        f"<td>{escape(item.prompt_id)}</td>"
        f"<td>{escape(item.category)}</td>"
        f"<td>{'YES' if item.candidate_validation.passed else 'NO'}</td>"
        f"<td>{item.composite_drift_score:.4f}</td>"
        f"<td>{'DRIFT' if item.drifted else 'OK'}</td>"
        "</tr>"
        for item in b.results
    )
    qual_rows = "".join(
        "<tr>"
        f"<td>{escape(f.prompt_id)}</td>"
        f"<td>{escape(f.check)}</td>"
        f"<td>{escape(f.severity)}</td>"
        f"<td>{escape(f.detail)}</td>"
        "</tr>"
        for f in q.findings
    )
    qual_section = (
        "<h2>Quality Regressions</h2><div class='scroll'><table><thead><tr>"
        "<th>Prompt ID</th><th>Check</th><th>Severity</th><th>Detail</th>"
        f"</tr></thead><tbody>{qual_rows}</tbody></table></div>"
        if qual_rows
        else (
            "<h2>Quality Regressions</h2>"
            "<p class='muted'>No loops, refusal flips, or length anomalies.</p>"
        )
    )
    failure_html = (
        "<h2>Failure Reasons</h2><ul>"
        + "".join(f"<li>{escape(reason)}</li>" for reason in report.failure_reasons)
        + "</ul>"
        if report.failure_reasons
        else ""
    )
    tok_card = (
        f"{c.baseline_tokens} &rarr; {c.candidate_tokens} ({c.token_delta:+d})"
        if c.complete and c.token_delta is not None
        else "n/a"
    )
    lp_card = (
        f"{measurement(lp.mean_entropy_delta_bits, '+.3f')}b / "
        f"JSD {measurement(lp.max_top_k_jsd, '.3f')}"
        if lp.comparable_samples > 0
        else "n/a"
    )
    extra_cards = ""
    if report.embeddings is not None:
        emb = report.embeddings
        extra_cards += (
            '  <div class="card"><div class="label">Embedding Mean Cos</div>'
            f'<div class="val">{measurement(emb.mean_similarity, ".4f")} '
            f'({"PASS" if emb.passed else "FAIL"})</div></div>\n'
        )
    if report.conversations is not None:
        conv = report.conversations
        extra_cards += (
            '  <div class="card"><div class="label">Conversations Drifted</div>'
            f'<div class="val">{conv.drifted_conversations} / {conv.total_conversations} '
            f'({"PASS" if conv.passed else "FAIL"})</div></div>\n'
        )
    if report.tool_trace is not None:
        tt = report.tool_trace
        extra_cards += (
            '  <div class="card"><div class="label">Agent Tool Calls</div>'
            f'<div class="val">{tt.baseline_calls} &rarr; {tt.candidate_calls} '
            f'({"PASS" if tt.passed else "FAIL"})</div></div>\n'
        )
    conv_section = ""
    if report.conversations is not None:
        conv_rows = "".join(
            "<tr>"
            f"<td>{escape(item.conversation_id)}</td>"
            f"<td>{item.compared_turns}</td>"
            f"<td>{item.mean_turn_drift:.4f}</td>"
            f"<td>{item.max_turn_drift:.4f}</td>"
            f"<td>{item.first_divergence_turn or 'none'}</td>"
            f"<td>{len(item.refusal_flip_turns)}</td>"
            f"<td>{'DRIFT' if item.drifted else 'OK'}</td>"
            "</tr>"
            for item in report.conversations.conversations
        )
        conv_section = (
            "<h2>Multi-Turn Conversations</h2><div class='scroll'><table><thead><tr>"
            "<th>Conversation ID</th><th>Turns</th><th>Mean Drift</th><th>Max Drift</th>"
            "<th>First Div</th><th>Refusal Flips</th><th>State</th>"
            f"</tr></thead><tbody>{conv_rows}</tbody></table></div>"
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>LLM Drift Bench Unified Audit: {state}</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#0b1220;color:#e8eef8;
font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:40px 24px}}
h1{{font-size:2rem;line-height:1.2}}h2{{margin-top:32px}}.status{{font-weight:700;color:{accent}}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
gap:14px;margin:24px 0}}
.card{{background:#111e32;border:1px solid #30405a;border-radius:8px;padding:14px}}
.card .label{{font-size:.82rem;color:#a8bad2}}.card .val{{font-size:1.25rem;font-weight:700}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:#111e32}}
th,td{{padding:10px 12px;border-bottom:1px solid #30405a;text-align:left}}th{{color:#7dd3fc}}
td{{white-space:pre-wrap;overflow-wrap:anywhere}}.muted{{color:#a8bad2}}
</style></head><body><main>
<p class="muted">llm-drift-bench / unified offline audit</p>
<h1>Audit Verdict: <span class="status">{state}</span></h1>
<p>Baseline: {escape(b.baseline_model)} &rarr; Candidate: {escape(b.candidate_model)}
 ({b.total_samples} prompt(s))</p>
<div class="cards">
  <div class="card"><div class="label">Mean Composite Drift</div>
    <div class="val">{b.mean_composite_drift:.4f}</div></div>
  <div class="card"><div class="label">JSON Conformity</div>
    <div class="val">{b.candidate_json_conformity_rate:.1%}</div></div>
  <div class="card"><div class="label">Max Dist TVD / W1</div>
    <div class="val">{d.max_tvd:.3f} / {d.max_normalized_wasserstein:.3f}</div></div>
  <div class="card"><div class="label">Quality Flagged</div>
    <div class="val">{q.flagged_prompts} / {q.total_samples}</div></div>
  <div class="card"><div class="label">Logprobs dH / JSD</div>
    <div class="val">{lp_card}</div></div>
  <div class="card"><div class="label">Output Tokens</div>
    <div class="val">{tok_card}</div></div>
{extra_cards}</div>
{failure_html}
<h2>Per-Prompt Drift</h2>
<div class="scroll"><table><thead><tr>
<th>Prompt ID</th><th>Category</th><th>Schema OK</th><th>Drift</th><th>State</th>
</tr></thead><tbody>{prompt_rows}</tbody></table></div>
{conv_section}
{qual_section}
</main></body></html>
"""
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    return html


def export_audit_pr_comment(report: AuditReport) -> str:
    """Render a compact GitHub Pull Request comment summarizing an AuditReport."""
    b = report.benchmark
    d = report.distribution
    q = report.quality
    verdict = "PASS" if report.passed else "FAIL"
    lines = [
        "<!-- llm-drift-bench:audit -->",
        f"### LLM Drift Bench Audit: **{verdict}**",
        "",
        f"`{markdown_cell(b.baseline_model)}` -> `{markdown_cell(b.candidate_model)}` "
        f"({b.total_samples} prompt(s))",
        "",
        "| Check | Status | Metric |",
        "| :--- | :---: | :--- |",
        f"| Per-prompt drift | {'PASS' if b.passed else 'FAIL'} | "
        f"mean `{b.mean_composite_drift:.4f}`, "
        f"conformity `{b.candidate_json_conformity_rate:.1%}` |",
        f"| Field distribution | INFO | "
        f"max TVD `{d.max_tvd:.4f}`, max W1 `{d.max_normalized_wasserstein:.4f}` |",
        f"| Quality checks | {'PASS' if q.passed else 'FAIL'} | "
        f"`{q.flagged_prompts}/{q.total_samples}` flagged |",
    ]
    if report.embeddings is not None:
        emb = report.embeddings
        lines.append(
            f"| Embeddings | {'PASS' if emb.passed else 'FAIL'} | "
            f"mean cosine `{measurement(emb.mean_similarity, '.4f')}` |"
        )
    if report.conversations is not None:
        conv = report.conversations
        lines.append(
            f"| Conversations | {'PASS' if conv.passed else 'FAIL'} | "
            f"`{conv.drifted_conversations}/{conv.total_conversations}` drifted, "
            f"mean `{conv.mean_trajectory_drift:.4f}` |"
        )
    if report.tool_trace is not None:
        tt = report.tool_trace
        lines.append(
            f"| Agent tool trace | {'PASS' if tt.passed else 'FAIL'} | "
            f"calls `{tt.baseline_calls}->{tt.candidate_calls}` ({tt.call_delta:+d}), "
            f"jaccard `{tt.action_jaccard:.4f}` |"
        )
    if report.failure_reasons:
        lines.append("")
        lines.append(
            f"<details><summary>Failure reasons ({len(report.failure_reasons)})</summary>"
        )
        lines.append("")
        for reason in report.failure_reasons:
            lines.append(f"- {markdown_cell(reason)}")
        lines.append("")
        lines.append("</details>")
    return "\n".join(lines) + "\n"
