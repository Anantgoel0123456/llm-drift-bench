"""Offline, script-free shields-style SVG drift status badge generator."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

from llm_drift_bench._file_io import parse_source_json, read_utf8
from llm_drift_bench.audit import AuditReport
from llm_drift_bench.consumer import ConsumerComparison
from llm_drift_bench.conversation import ConversationDriftReport
from llm_drift_bench.models import DriftBenchmarkReport
from llm_drift_bench.report_io import load_engine_report
from llm_drift_bench.rolling import RollingWindowReport
from llm_drift_bench.tool_events import ToolTraceDiff


def _text_width(text: str) -> int:
    """Estimate compact SVG pill width in pixels without external font metrics."""
    return max(36, len(text) * 7 + 16)


def _render_badge_svg(
    *,
    label: str,
    value: str,
    passed: bool,
    output_path: str | Path | None = None,
) -> str:
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string.")
    clean_label = label.strip()
    color = "#2ea44f" if passed else "#d73a49"
    left_w = _text_width(clean_label)
    right_w = _text_width(value)
    total_w = left_w + right_w
    left_x = left_w / 2.0
    right_x = left_w + right_w / 2.0

    esc_label = escape(clean_label, quote=True)
    esc_value = escape(value, quote=True)
    aria = escape(f"{clean_label}: {value}", quote=True)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20" '
        f'role="img" aria-label="{aria}">'
        f"<title>{aria}</title>"
        '<linearGradient id="s" x2="0" y2="100%">'
        '<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        '<stop offset="1" stop-opacity=".1"/>'
        "</linearGradient>"
        f'<clipPath id="r"><rect width="{total_w}" height="20" rx="3" fill="#fff"/></clipPath>'
        '<g clip-path="url(#r)">'
        f'<rect width="{left_w}" height="20" fill="#555"/>'
        f'<rect x="{left_w}" width="{right_w}" height="20" fill="{color}"/>'
        f'<rect width="{total_w}" height="20" fill="url(#s)"/>'
        "</g>"
        '<g fill="#fff" text-anchor="middle" '
        'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" '
        'text-rendering="geometricPrecision" font-size="11">'
        f'<text x="{left_x:.1f}" y="15" fill="#010101" fill-opacity=".3">{esc_label}</text>'
        f'<text x="{left_x:.1f}" y="14">{esc_label}</text>'
        f'<text x="{right_x:.1f}" y="15" fill="#010101" fill-opacity=".3">{esc_value}</text>'
        f'<text x="{right_x:.1f}" y="14">{esc_value}</text>'
        "</g>"
        "</svg>\n"
    )
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(svg, encoding="utf-8")
    return svg


def export_drift_badge_svg(
    report: DriftBenchmarkReport,
    output_path: str | Path | None = None,
    *,
    label: str = "llm-drift",
) -> str:
    """Render an offline static SVG status badge summarizing a DriftBenchmarkReport."""
    if not isinstance(report, DriftBenchmarkReport):
        raise TypeError("export_drift_badge_svg requires a DriftBenchmarkReport instance.")
    state = "PASS" if report.passed else "FAIL"
    return _render_badge_svg(
        label=label,
        value=f"{state} {report.mean_composite_drift:.2f}",
        passed=report.passed,
        output_path=output_path,
    )


def export_audit_badge_svg(
    report: AuditReport,
    output_path: str | Path | None = None,
    *,
    label: str = "llm-audit",
) -> str:
    """Render an offline static SVG status badge summarizing a unified AuditReport."""
    if not isinstance(report, AuditReport):
        raise TypeError("export_audit_badge_svg requires an AuditReport instance.")
    state = "PASS" if report.passed else "FAIL"
    value = (
        f"{state} {report.benchmark.mean_composite_drift:.2f}"
        if report.passed
        else f"{state} ({len(report.failure_reasons)} issue(s))"
    )
    return _render_badge_svg(
        label=label,
        value=value,
        passed=report.passed,
        output_path=output_path,
    )


def export_conversation_badge_svg(
    report: ConversationDriftReport,
    output_path: str | Path | None = None,
    *,
    label: str = "conversations",
) -> str:
    """Render an offline static SVG status badge summarizing a ConversationDriftReport."""
    if not isinstance(report, ConversationDriftReport):
        raise TypeError("export_conversation_badge_svg requires a ConversationDriftReport.")
    state = "PASS" if report.passed else "FAIL"
    return _render_badge_svg(
        label=label,
        value=f"{state} {report.mean_trajectory_drift:.2f}",
        passed=report.passed,
        output_path=output_path,
    )


def export_tool_trace_badge_svg(
    diff: ToolTraceDiff,
    output_path: str | Path | None = None,
    *,
    label: str = "tool-trace",
) -> str:
    """Render an offline static SVG status badge summarizing an agent ToolTraceDiff."""
    if not isinstance(diff, ToolTraceDiff):
        raise TypeError("export_tool_trace_badge_svg requires a ToolTraceDiff instance.")
    state = "PASS" if diff.passed else "FAIL"
    return _render_badge_svg(
        label=label,
        value=f"{state} {diff.action_jaccard:.2f}",
        passed=diff.passed,
        output_path=output_path,
    )


def export_consumer_badge_svg(
    comparison: ConsumerComparison,
    output_path: str | Path | None = None,
    *,
    label: str = "consumer",
) -> str:
    """Render an offline static SVG status badge summarizing a ConsumerComparison."""
    if not isinstance(comparison, ConsumerComparison):
        raise TypeError("export_consumer_badge_svg requires a ConsumerComparison instance.")
    state = "PASS" if comparison.passed else "FAIL"
    return _render_badge_svg(
        label=label,
        value=f"{state} {comparison.candidate_rejection_rate:.1%}",
        passed=comparison.passed,
        output_path=output_path,
    )


def export_rolling_badge_svg(
    report: RollingWindowReport,
    output_path: str | Path | None = None,
    *,
    label: str = "rolling-dist",
) -> str:
    """Render an offline static SVG status badge summarizing a RollingWindowReport."""
    if not isinstance(report, RollingWindowReport):
        raise TypeError("export_rolling_badge_svg requires a RollingWindowReport instance.")
    state = "PASS" if report.passed else "FAIL"
    return _render_badge_svg(
        label=label,
        value=f"{state} TVD {report.overall.max_tvd:.2f}",
        passed=report.passed,
        output_path=output_path,
    )


def _require_finite_metric(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Saved report field {field!r} must be a finite number.")
    return float(value)


def export_badge_from_json(
    path: str | Path,
    output_path: str | Path | None = None,
    *,
    label: str | None = None,
) -> str:
    """Render an SVG status badge from a saved engine, audit, or conversation JSON report."""
    source = Path(path)
    payload = parse_source_json(read_utf8(source), source)
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: report JSON must be an object.")

    if "benchmark" in payload and "distribution" in payload and "passed" in payload:
        passed = payload["passed"]
        bench = payload["benchmark"]
        reasons = payload.get("failure_reasons", [])
        if not isinstance(passed, bool) or not isinstance(bench, dict) or not isinstance(
            reasons, list
        ):
            raise ValueError(f"{source}: malformed saved AuditReport JSON.")
        mean_drift = _require_finite_metric(
            bench.get("mean_composite_drift"), "benchmark.mean_composite_drift"
        )
        state = "PASS" if passed else "FAIL"
        value = f"{state} {mean_drift:.2f}" if passed else f"{state} ({len(reasons)} issue(s))"
        return _render_badge_svg(
            label="llm-audit" if label is None else label,
            value=value,
            passed=passed,
            output_path=output_path,
        )

    if "conversations" in payload and "mean_trajectory_drift" in payload and "passed" in payload:
        passed = payload["passed"]
        if not isinstance(passed, bool):
            raise ValueError(f"{source}: malformed saved ConversationDriftReport JSON.")
        mean_traj = _require_finite_metric(
            payload.get("mean_trajectory_drift"), "mean_trajectory_drift"
        )
        state = "PASS" if passed else "FAIL"
        return _render_badge_svg(
            label="conversations" if label is None else label,
            value=f"{state} {mean_traj:.2f}",
            passed=passed,
            output_path=output_path,
        )

    if (
        "baseline_calls" in payload
        and "candidate_calls" in payload
        and "action_jaccard" in payload
        and "passed" in payload
    ):
        passed = payload["passed"]
        if not isinstance(passed, bool):
            raise ValueError(f"{source}: malformed saved ToolTraceDiff JSON.")
        jaccard = _require_finite_metric(payload.get("action_jaccard"), "action_jaccard")
        state = "PASS" if passed else "FAIL"
        return _render_badge_svg(
            label="tool-trace" if label is None else label,
            value=f"{state} {jaccard:.2f}",
            passed=passed,
            output_path=output_path,
        )

    if (
        "baseline_rejection_rate" in payload
        and "candidate_rejection_rate" in payload
        and "passed" in payload
    ):
        passed = payload["passed"]
        if not isinstance(passed, bool):
            raise ValueError(f"{source}: malformed saved ConsumerComparison JSON.")
        cand_rate = _require_finite_metric(
            payload.get("candidate_rejection_rate"), "candidate_rejection_rate"
        )
        state = "PASS" if passed else "FAIL"
        return _render_badge_svg(
            label="consumer" if label is None else label,
            value=f"{state} {cand_rate:.1%}",
            passed=passed,
            output_path=output_path,
        )

    if "window_count" in payload and "overall" in payload and "passed" in payload:
        passed = payload["passed"]
        overall = payload["overall"]
        if not isinstance(passed, bool) or not isinstance(overall, dict):
            raise ValueError(f"{source}: malformed saved RollingWindowReport JSON.")
        max_tvd = _require_finite_metric(overall.get("max_tvd"), "overall.max_tvd")
        state = "PASS" if passed else "FAIL"
        return _render_badge_svg(
            label="rolling-dist" if label is None else label,
            value=f"{state} TVD {max_tvd:.2f}",
            passed=passed,
            output_path=output_path,
        )

    report = load_engine_report(source)
    return export_drift_badge_svg(
        report,
        output_path=output_path,
        label="llm-drift" if label is None else label,
    )
