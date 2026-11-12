"""Console (Rich) tabular output and Markdown report exporter for llm-drift-bench."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_drift_bench.models import BenchmarkResult


def render_console_report(
    result: BenchmarkResult,
    console: Console | None = None,
) -> str:
    """Render a BenchmarkResult as a Rich tabular report to the console and return text."""
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
        f"{result.mean_baseline_latency_ms:.2f}",
        f"{result.mean_candidate_latency_ms:.2f}",
    )
    summary_table.add_row(
        "Mean Throughput (tokens/s)",
        f"{result.mean_baseline_tokens_per_sec:.2f}",
        f"{result.mean_candidate_tokens_per_sec:.2f}",
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
            f"{m.candidate_tokens_per_sec:.1f}",
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


def export_markdown_report(
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
            f"| **Mean Latency (ms)** | {result.mean_baseline_latency_ms:.2f} | "
            f"{result.mean_candidate_latency_ms:.2f} |"
        ),
        (
            f"| **Mean Throughput (tokens/sec)** | {result.mean_baseline_tokens_per_sec:.2f} | "
            f"{result.mean_candidate_tokens_per_sec:.2f} |"
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
            f"{m.drift_score:.4f} | {m.latency_delta_ms:+.2f} | "
            f"{m.candidate_tokens_per_sec:.2f} |"
        )

    markdown = "\n".join(lines) + "\n"
    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(markdown, encoding="utf-8")

    return markdown
