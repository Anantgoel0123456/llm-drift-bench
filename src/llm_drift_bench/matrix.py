"""Multi-candidate Prompt x Candidate drift heatmap matrix (text and standalone HTML)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal

from llm_drift_bench.comparison import compare_reports
from llm_drift_bench.models import DriftBenchmarkReport

MatrixSort = Literal["max_drift", "prompt_id"]


@dataclass(frozen=True)
class MatrixRow:
    """One prompt's drift scores and drift flags across every candidate."""

    prompt_id: str
    category: str
    scores: dict[str, float]
    drifted: dict[str, bool]
    max_drift: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "max_drift": round(self.max_drift, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "drifted": dict(self.drifted),
        }


@dataclass(frozen=True)
class DriftMatrix:
    """Prompt x Candidate matrix across 2+ compatible DriftBenchmarkReports."""

    baseline_model: str
    candidates: tuple[str, ...]
    rows: tuple[MatrixRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_model": self.baseline_model,
            "candidates": list(self.candidates),
            "rows": [row.to_dict() for row in self.rows],
        }


def build_drift_matrix(
    reports: Sequence[DriftBenchmarkReport],
    *,
    sort_prompts_by: MatrixSort = "max_drift",
) -> DriftMatrix:
    """Build a validated Prompt x Candidate matrix ordered by candidate mean drift."""
    if sort_prompts_by not in ("max_drift", "prompt_id"):
        raise ValueError("sort_prompts_by must be 'max_drift' or 'prompt_id'.")
    ranked = compare_reports(reports, sort_by="drift")
    ordered_candidates = tuple(row.candidate for row in ranked)
    by_cand = {report.candidate_model: report for report in reports}

    indexed = {
        cand: {res.prompt_id: res for res in by_cand[cand].results}
        for cand in ordered_candidates
    }
    prompt_ids = sorted(indexed[ordered_candidates[0]])

    rows: list[MatrixRow] = []
    for pid in prompt_ids:
        category = indexed[ordered_candidates[0]][pid].category
        scores = {cand: indexed[cand][pid].composite_drift_score for cand in ordered_candidates}
        drifted = {cand: indexed[cand][pid].drifted for cand in ordered_candidates}
        rows.append(
            MatrixRow(
                prompt_id=pid,
                category=category,
                scores=scores,
                drifted=drifted,
                max_drift=max(scores.values()) if scores else 0.0,
            )
        )

    if sort_prompts_by == "max_drift":
        rows.sort(key=lambda r: (-r.max_drift, r.prompt_id))
    else:
        rows.sort(key=lambda r: r.prompt_id)

    return DriftMatrix(
        baseline_model=reports[0].baseline_model,
        candidates=ordered_candidates,
        rows=tuple(rows),
    )


def format_drift_matrix_text(matrix: DriftMatrix) -> str:
    """Render a DriftMatrix as an aligned terminal table (* marks drifted cells)."""
    cand_cols = [c if len(c) <= 14 else c[:11] + "..." for c in matrix.candidates]
    header = f"{'Prompt ID':<20} | " + " | ".join(f"{col:>14}" for col in cand_cols)
    sep = "-" * len(header)
    lines = [
        f"Drift Matrix (Baseline: {matrix.baseline_model}; * = drifted)",
        sep,
        header,
        sep,
    ]
    for row in matrix.rows:
        pid = row.prompt_id if len(row.prompt_id) <= 20 else row.prompt_id[:17] + "..."
        cells = []
        for cand in matrix.candidates:
            mark = "*" if row.drifted[cand] else " "
            cells.append(f"{row.scores[cand]:.4f}{mark}".rjust(14))
        lines.append(f"{pid:<20} | " + " | ".join(cells))
    lines.append(sep)
    return "\n".join(lines)


def _heat_color(score: float, drifted: bool) -> str:
    if not drifted and score <= 0.05:
        return "#0f2e24"
    if not drifted and score <= 0.20:
        return "#2b2a14"
    if score <= 0.50:
        return "#4a2511"
    return "#5c1724"


def export_drift_matrix_html(
    matrix: DriftMatrix,
    output_path: str | Path | None = None,
) -> str:
    """Export an offline script-free HTML heatmap comparing prompts across candidates."""
    headers = "".join(f"<th scope='col'>{escape(cand)}</th>" for cand in matrix.candidates)
    body_rows: list[str] = []
    for row in matrix.rows:
        cells = "".join(
            f"<td style='background:{_heat_color(row.scores[c], row.drifted[c])}'>"
            f"{row.scores[c]:.4f}{' <strong>*</strong>' if row.drifted[c] else ''}</td>"
            for c in matrix.candidates
        )
        body_rows.append(
            f"<tr><td><code>{escape(row.prompt_id)}</code></td>"
            f"<td class='muted'>{escape(row.category)}</td>{cells}</tr>"
        )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>LLM Drift Matrix</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#0b1220;color:#e8eef8;
font:15px/1.5 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:36px 24px}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:#111e32}}
th,td{{padding:10px 12px;border:1px solid #24324a;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
th{{color:#7dd3fc;background:#0f172a}}.muted{{color:#a8bad2}}
</style></head><body><main>
<p class="muted">llm-drift-bench / cross-candidate drift heatmap (* = drifted)</p>
<h1>Drift Matrix &mdash; Baseline: {escape(matrix.baseline_model)}</h1>
<div class="scroll"><table><thead><tr>
<th scope="col">Prompt ID</th><th scope="col">Category</th>{headers}
</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>
</main></body></html>
"""
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
    return html
