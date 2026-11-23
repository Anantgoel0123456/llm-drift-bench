"""Console (Rich) tabular output and Markdown report exporter for llm-drift-bench."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from llm_drift_bench._report_views import (
    DETAIL_HEADERS,
    canonical_html,
    canonical_markdown,
    detail_rows,
    report_notes,
    summary_rows,
)
from llm_drift_bench._visuals import (
    canonical_pr_comment,
    comparison_html,
    render_html_diffs_section,
)
from llm_drift_bench.comparison import SortKey
from llm_drift_bench.models import DriftBenchmarkReport
from llm_drift_bench.tool_log import ToolCheckRecord

if TYPE_CHECKING:
    from rich.console import Console

    from llm_drift_bench.models import BenchmarkResult
    from llm_drift_bench.policy import DriftPolicy


def _measurement(value: float | None, spec: str = ".2f") -> str:
    """Format a measurement without turning missing data into zero."""
    return "n/a" if value is None else format(value, spec)


def _render_legacy_console_report(
    result: BenchmarkResult,
    console: Console | None = None,
) -> str:
    """Render a BenchmarkResult as a Rich tabular report to the console and return text."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    buffer = io.StringIO()
    capture_console = Console(file=buffer, width=100, force_terminal=False, color_system=None)
    active_console = console or Console()

    summary_table = Table(
        title=f"LLM Drift Benchmark Summary ({result.suite_name})",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column(f"Baseline ({result.baseline_model})", justify="right")
    summary_table.add_column(f"Candidate ({result.candidate_model})", justify="right")

    summary_table.add_row("Total Prompts", str(result.total_prompts), str(result.total_prompts))
    summary_table.add_row("Exact Match Rate", "100.0% (ref)", f"{result.exact_match_rate:.1%}")
    summary_table.add_row(
        "JSON / Format Adherence",
        f"{result.baseline_json_adherence_rate:.1%}",
        f"{result.candidate_json_adherence_rate:.1%}",
    )
    summary_table.add_row(
        "Mean Semantic Similarity",
        "1.0000 (ref)",
        f"{result.mean_semantic_similarity:.4f}",
    )
    summary_table.add_row(
        "Mean Token Diff Ratio",
        "0.0000 (ref)",
        f"{result.mean_token_diff_ratio:.4f}",
    )
    summary_table.add_row(
        "Mean Drift Score",
        "0.0000 (ref)",
        f"{result.mean_drift_score:.4f}",
    )
    summary_table.add_row(
        "Mean Latency (ms)",
        f"{_measurement(result.mean_baseline_latency_ms)}",
        f"{_measurement(result.mean_candidate_latency_ms)}",
    )
    summary_table.add_row(
        "Mean Throughput (tokens/s)",
        f"{_measurement(result.mean_baseline_tokens_per_sec)}",
        f"{_measurement(result.mean_candidate_tokens_per_sec)}",
    )

    detail_table = Table(
        title="Per-Prompt Drift Breakdown",
        show_header=True,
        header_style="bold magenta",
    )
    detail_table.add_column("Prompt ID")
    detail_table.add_column("Format OK", justify="center")
    detail_table.add_column("Exact", justify="center")
    detail_table.add_column("Sem Sim", justify="right")
    detail_table.add_column("Token Diff", justify="right")
    detail_table.add_column("Drift Score", justify="right")
    detail_table.add_column("Cand Tok/s", justify="right")

    for m in result.metrics:
        detail_table.add_row(
            m.prompt_id,
            "YES" if m.format_adherence else "NO",
            "YES" if m.exact_match else "NO",
            f"{m.semantic_similarity:.4f}",
            f"{m.token_diff_ratio:.4f}",
            f"{m.drift_score:.4f}",
            _measurement(m.candidate_tokens_per_sec, ".1f"),
        )

    header_panel = Panel(
        f"[bold]Baseline:[/bold] {result.baseline_model}   |   "
        f"[bold]Candidate:[/bold] {result.candidate_model}   |   "
        f"[bold]Mean Drift:[/bold] {result.mean_drift_score:.4f}",
        title="llm-drift-bench",
    )

    for target in (active_console, capture_console):
        target.print(header_panel)
        target.print(summary_table)
        target.print(detail_table)

    return buffer.getvalue()


def _export_legacy_markdown_report(
    result: BenchmarkResult,
    output_path: str | Path | None = None,
) -> str:
    """Format a BenchmarkResult as a GitHub-Flavored Markdown report and optionally save it."""
    lines = [
        f"# LLM Drift Benchmark Report: `{result.suite_name}`",
        "",
        f"- **Baseline Model**: `{result.baseline_model}`",
        f"- **Candidate Model**: `{result.candidate_model}`",
        f"- **Total Prompts**: `{result.total_prompts}`",
        "",
        "## Aggregate Summary",
        "",
        "| Metric | Baseline | Candidate |",
        "| :--- | ---: | ---: |",
        f"| **Exact Match Rate** | 100.0% (ref) | {result.exact_match_rate:.1%} |",
        (
            f"| **JSON / Format Adherence** | {result.baseline_json_adherence_rate:.1%} | "
            f"{result.candidate_json_adherence_rate:.1%} |"
        ),
        (
            f"| **Mean Semantic Similarity** | 1.0000 (ref) | "
            f"{result.mean_semantic_similarity:.4f} |"
        ),
        (
            f"| **Mean Token Diff Ratio** | 0.0000 (ref) | "
            f"{result.mean_token_diff_ratio:.4f} |"
        ),
        f"| **Mean Drift Score** | 0.0000 (ref) | {result.mean_drift_score:.4f} |",
        (
            f"| **Mean Latency (ms)** | {_measurement(result.mean_baseline_latency_ms)} | "
            f"{_measurement(result.mean_candidate_latency_ms)} |"
        ),
        (
            "| **Mean Throughput (tokens/sec)** | "
            f"{_measurement(result.mean_baseline_tokens_per_sec)} | "
            f"{_measurement(result.mean_candidate_tokens_per_sec)} |"
        ),
        "",
        "## Per-Prompt Breakdown",
        "",
        (
            "| Prompt ID | Category | Format OK | Exact Match | Semantic Sim | "
            "Token Diff | Drift Score | Latency Delta (ms) | Cand Tok/s |"
        ),
        "| :--- | :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for m in result.metrics:
        fmt_ok = "PASS" if m.format_adherence else "FAIL"
        exact = "YES" if m.exact_match else "NO"
        lines.append(
            f"| `{m.prompt_id}` | {m.category} | {fmt_ok} | {exact} | "
            f"{m.semantic_similarity:.4f} | {m.token_diff_ratio:.4f} | "
            f"{m.drift_score:.4f} | {_measurement(m.latency_delta_ms, '+.2f')} | "
            f"{_measurement(m.candidate_tokens_per_sec)} |"
        )

    markdown = "\n".join(lines) + "\n"
    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(markdown, encoding="utf-8")

    return markdown


def render_console_report(
    result: DriftBenchmarkReport | BenchmarkResult,
    console: Console | None = None,
) -> str:
    """Render existing scores, keeping legacy evaluator reports separate."""
    if not isinstance(result, DriftBenchmarkReport):
        return _render_legacy_console_report(result, console)
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    buffer = io.StringIO()
    targets = (console or Console(), Console(file=buffer, width=100, color_system=None))
    state = "PASSED" if result.passed else "FAILED"
    summary = Table("Metric", "Baseline", "Candidate / value", title="Summary", safe_box=True)
    for row in summary_rows(result):
        summary.add_row(*(Text(value) for value in row))
    details = Table(*DETAIL_HEADERS, title="Per-prompt results", safe_box=True)
    for detail in detail_rows(result):
        details.add_row(*(Text(value) for value in detail))
    heading = Text(
        f"{state} | Baseline: {result.baseline_model} | Candidate: {result.candidate_model}\n"
        f"Samples: {result.total_samples}; drifted: {result.drifted_samples}"
    )
    for target in targets:
        target.print(Panel(heading, title="llm-drift-bench", safe_box=True))
        target.print(summary)
        target.print(details)
        for note in report_notes(result):
            target.print(Text(note))
        if result.thresholds is not None:
            target.print(Text(f"Thresholds: {result.thresholds}"))
    return buffer.getvalue()


def _save_text(text: str, path: str | Path | None) -> str:
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return text


def _require_engine_report(
    result: DriftBenchmarkReport | BenchmarkResult, tool_checks: Sequence[ToolCheckRecord] | None,
) -> None:
    if tool_checks and not isinstance(result, DriftBenchmarkReport):
        raise ValueError(
            "Tool evidence can only be attached to engine reports. "
            "Rerun with 'run --output report.json'."
        )


def export_markdown_report(
    result: DriftBenchmarkReport | BenchmarkResult,
    output_path: str | Path | None = None,
    tool_checks: Sequence[ToolCheckRecord] | None = None,
) -> str:
    """Render canonical engine scores or an explicitly labeled legacy report.

    tool_checks adds a separate evidence section; it never changes the drift verdict.
    """
    _require_engine_report(result, tool_checks)
    if not isinstance(result, DriftBenchmarkReport):
        text = _export_legacy_markdown_report(result)
        text += "\nLegacy evaluator scores; not interchangeable with engine scores.\n"
    else:
        text = canonical_markdown(result, tuple(tool_checks or ()))
    return _save_text(text, output_path)


def export_html_report(
    result: DriftBenchmarkReport | BenchmarkResult,
    output_path: str | Path | None = None,
    tool_checks: Sequence[ToolCheckRecord] | None = None,
    *,
    diffs: Mapping[str, tuple[str, str]] | None = None,
    policy: DriftPolicy | None = None,
) -> str:
    """Export a self-contained, script-free canonical report with no network assets.

    tool_checks adds a separate evidence section; it never changes the drift verdict.
    diffs optionally embeds opt-in collapsible JSON/text diffs for changed prompts.
    """
    if not isinstance(result, DriftBenchmarkReport):
        raise ValueError("HTML requires an engine report. Rerun with 'run --output report.json'.")
    extra = render_html_diffs_section(result, diffs, policy=policy) if diffs else ""
    return _save_text(
        canonical_html(result, tuple(tool_checks or ()), extra_sections=extra),
        output_path,
    )


def export_pr_comment(
    result: DriftBenchmarkReport | BenchmarkResult,
    output_path: str | Path | None = None,
    *,
    previous: DriftBenchmarkReport | None = None,
    tool_checks: Sequence[ToolCheckRecord] | None = None,
) -> str:
    """Format a compact GitHub PR comment with status badge, deltas, and collapsible breakdown."""
    if not isinstance(result, DriftBenchmarkReport):
        raise ValueError("PR comment export requires a canonical engine report.")
    text = canonical_pr_comment(result, previous=previous, tool_checks=tuple(tool_checks or ()))
    return _save_text(text, output_path)


def export_comparison_html(
    reports: Sequence[DriftBenchmarkReport],
    output_path: str | Path | None = None,
    *,
    sort_by: SortKey = "drift",
) -> str:
    """Export a standalone multi-candidate HTML dashboard with an inline SVG Pareto chart."""
    return _save_text(comparison_html(reports, sort_by=sort_by), output_path)
