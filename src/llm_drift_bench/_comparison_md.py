"""GitHub-Flavored Markdown multi-candidate comparison leaderboard exporter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llm_drift_bench._report_views import markdown_cell
from llm_drift_bench.comparison import SortKey, compare_reports, compare_with_changes
from llm_drift_bench.models import DriftBenchmarkReport


def export_comparison_markdown(
    reports: Sequence[DriftBenchmarkReport],
    output_path: str | Path | None = None,
    *,
    against: str | None = None,
    sort_by: SortKey = "drift",
) -> str:
    """Export a GFM leaderboard comparing multiple candidates (plus optional transitions)."""
    if against is not None:
        rows, changes = compare_with_changes(reports, against, sort_by=sort_by)
    else:
        rows = compare_reports(reports, sort_by=sort_by)
        changes = []

    baseline = markdown_cell(reports[0].baseline_model)
    lines = [
        f"# Candidate Drift Comparison (Baseline: `{baseline}`)",
        "",
        f"Ranked by `{sort_by}` across **{len(rows)}** candidates.",
        "",
        (
            "| Rank | Candidate | Verdict | Drifted | Mean Drift | "
            "Max Drift | JSON | P95 (ms) | Worst Prompt |"
        ),
        "| ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for rank, row in enumerate(rows, start=1):
        verdict = "PASS" if row.passed else "FAIL"
        p95 = "n/a" if row.p95_ms is None else f"{row.p95_ms:.2f}"
        worst = "none" if row.max_drift == 0.0 else markdown_cell(row.worst_prompt_id or "n/a")
        lines.append(
            f"| {rank} | `{markdown_cell(row.candidate)}` | **{verdict}** | "
            f"{row.drifted}/{row.samples} | {row.mean_drift:.4f} | {row.max_drift:.4f} | "
            f"{row.conformity:.1%} | {p95} | `{worst}` |"
        )

    if against is not None:
        lines += ["", f"## Prompt transitions vs `{markdown_cell(against)}`", ""]
        if not changes:
            lines.append("No prompt verdicts flipped relative to the reference candidate.")
        else:
            lines += [
                "| Candidate | Prompt ID | Category | Transition | Ref Drift | Cand Drift |",
                "| :--- | :--- | :--- | :---: | ---: | ---: |",
            ]
            for ch in changes:
                lines.append(
                    f"| `{markdown_cell(ch.candidate)}` | `{markdown_cell(ch.prompt_id)}` | "
                    f"{markdown_cell(ch.category)} | **{ch.state}** | "
                    f"{ch.reference_drift:.4f} | {ch.candidate_drift:.4f} |"
                )

    markdown = "\n".join(lines) + "\n"
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
    return markdown
