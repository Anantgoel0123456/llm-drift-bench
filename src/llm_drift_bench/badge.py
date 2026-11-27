"""Offline, script-free shields-style SVG drift status badge generator."""

from __future__ import annotations

from html import escape
from pathlib import Path

from llm_drift_bench.audit import AuditReport
from llm_drift_bench.conversation import ConversationDriftReport
from llm_drift_bench.models import DriftBenchmarkReport


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
