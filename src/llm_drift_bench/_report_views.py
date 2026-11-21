"""Presentation of canonical engine results, never a second evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from html import escape

from llm_drift_bench.models import DriftBenchmarkReport
from llm_drift_bench.tool_log import ToolCheckRecord

TOOL_HEADERS = (
    "Resource", "Required range", "Status", "Covered", "First edit", "Trace attested", "Reason",
)
TOOL_NOTE = (
    "Read-before-edit evidence from supplied normalized traces. It is reported "
    "separately, does not change the drift verdict above, and does not establish "
    "comprehension or correctness. UNKNOWN is not a pass."
)


def tool_check_rows(records: Sequence[ToolCheckRecord]) -> list[tuple[str, ...]]:
    return [
        (
            record.resource, f"[{record.required_start}, {record.required_end})",
            record.result.status,
            f"{record.result.covered_characters} of {record.result.required_characters}",
            "none" if record.result.first_edit_sequence is None
            else str(record.result.first_edit_sequence),
            "yes" if record.trace_complete else "no", record.result.reason,
        )
        for record in records
    ]


def measurement(value: float | None, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def summary_rows(report: DriftBenchmarkReport) -> list[tuple[str, str, str]]:
    return [
        ("JSON conformity", f"{report.baseline_json_conformity_rate:.1%}",
         f"{report.candidate_json_conformity_rate:.1%}"),
        ("Mean composite drift", "0.0000 (ref)", f"{report.mean_composite_drift:.4f}"),
        ("Max composite drift", "0.0000 (ref)", f"{report.max_composite_drift:.4f}"),
        ("Token Jaccard similarity", "1.0000 (ref)", f"{report.mean_token_jaccard:.4f}"),
        ("Levenshtein similarity", "1.0000 (ref)", f"{report.mean_levenshtein_similarity:.4f}"),
        ("JSON structural similarity", "1.0000 (ref)",
         measurement(report.mean_json_structural_similarity)),
        ("Bounded numeric drift", "0.0000 (ref)", measurement(report.mean_json_numeric_drift)),
        ("P50 latency (ms)", measurement(report.baseline_p50_latency_ms, ".2f"),
         measurement(report.candidate_p50_latency_ms, ".2f")),
        ("P95 latency (ms)", measurement(report.baseline_p95_latency_ms, ".2f"),
         measurement(report.candidate_p95_latency_ms, ".2f")),
    ]


def detail_rows(report: DriftBenchmarkReport) -> list[tuple[str, ...]]:
    return [
        (item.prompt_id, item.category, "YES" if item.candidate_validation.passed else "NO",
         measurement(item.json_structural_similarity), f"{item.composite_drift_score:.4f}",
         measurement(item.latency_delta_ms, "+.2f"), "DRIFT" if item.drifted else "OK")
        for item in report.results
    ]


DETAIL_HEADERS = (
    "Prompt ID", "Category", "Schema pass", "Structure", "Drift", "Latency delta", "State",
)


def report_notes(report: DriftBenchmarkReport) -> list[str]:
    notes = list(report.failure_reasons)
    comparable = sum(item.json_numeric_drift is not None for item in report.results)
    notes.append(
        f"Numeric drift: {comparable} of {report.total_samples} samples comparable. "
        "Each shared numeric field uses min(1, abs(a - b) / max(abs(a), abs(b), 1e-9)); "
        "sample means are averaged, excluding samples with no shared numeric fields. "
        "This is not percentage change relative to the baseline."
    )
    notes.append(
        "Schema pass requires valid JSON and conformity to the supplied schema; "
        "without a schema, only JSON parsing is checked."
    )
    for item in report.results:
        for issue in item.candidate_validation.issues:
            notes.append(f"{item.prompt_id}: {issue.path}: {issue.message} [{issue.rule}]")
    if (
        report.baseline_p95_latency_ms is None or report.candidate_p95_latency_ms is None
        or report.baseline_p95_latency_ms == 0
    ):
        notes.append("Latency regression check skipped: missing timings or zero baseline P95.")
    if report.thresholds is None:
        notes.append("Thresholds unavailable in this saved report; defaults are not substituted.")
    return notes


def markdown_cell(value: str) -> str:
    # Literal input must not become markup, extra table cells, or new rows.
    return "".join(
        f"&#{ord(char)};" if char in "&<>|`*_[\\]" else " " if char in "\r\n" else char
        for char in value
    )


def canonical_markdown(
    report: DriftBenchmarkReport, tool_checks: Sequence[ToolCheckRecord] = (),
) -> str:
    state = "PASSED" if report.passed else "FAILED"
    lines = [f"# LLM Drift Benchmark Report: {state}", "",
             f"Baseline: {markdown_cell(report.baseline_model)}", "",
             f"Candidate: {markdown_cell(report.candidate_model)}", "",
             f"Samples: {report.total_samples}; drifted: {report.drifted_samples}", "",
             "## Summary", "", "| Metric | Baseline | Candidate / value |",
             "| --- | ---: | ---: |"]
    lines.extend("| " + " | ".join(row) + " |" for row in summary_rows(report))
    lines += ["", "## Per-prompt results", "", "| " + " | ".join(DETAIL_HEADERS) + " |",
              "| " + " | ".join(["---"] * len(DETAIL_HEADERS)) + " |"]
    lines.extend("| " + " | ".join(markdown_cell(v) for v in row) + " |"
                 for row in detail_rows(report))
    if tool_checks:
        lines += ["", "## Tool evidence", "", TOOL_NOTE, "",
                  "| " + " | ".join(TOOL_HEADERS) + " |",
                  "| " + " | ".join(["---"] * len(TOOL_HEADERS)) + " |"]
        lines.extend("| " + " | ".join(markdown_cell(v) for v in row) + " |"
                     for row in tool_check_rows(tool_checks))
    lines += ["", "## Checks and notes", ""]
    lines.extend(f"- {markdown_cell(note)}" for note in report_notes(report))
    if report.thresholds is not None:
        lines += ["", "## Thresholds", ""]
        lines.extend(f"- {key}: {value}" for key, value in asdict(report.thresholds).items())
    return "\n".join(lines) + "\n"


def canonical_html(
    report: DriftBenchmarkReport,
    tool_checks: Sequence[ToolCheckRecord] = (),
    *,
    extra_sections: str = "",
) -> str:
    def table(headers: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
        head = "".join(f"<th scope='col'>{escape(v)}</th>" for v in headers)
        body = "".join("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in row) + "</tr>"
                       for row in rows)
        return (f"<div class='scroll'><table><thead><tr>{head}</tr></thead>"
                f"<tbody>{body}</tbody></table></div>")

    state = "PASSED" if report.passed else "FAILED"
    tool_section = ""
    if tool_checks:
        tool_section = (
            f"<h2>Tool evidence</h2><p class=\"muted\">{escape(TOOL_NOTE)}</p>"
            + table(TOOL_HEADERS, tool_check_rows(tool_checks))
        )
    notes = "".join(f"<li>{escape(note)}</li>" for note in report_notes(report))
    thresholds = ""
    if report.thresholds is not None:
        thresholds = "<h2>Thresholds</h2><ul>" + "".join(
            f"<li>{escape(key)}: {value}</li>" for key, value in asdict(report.thresholds).items()
        ) + "</ul>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>LLM Drift Benchmark Report: {state}</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#0b1220;color:#e8eef8;
font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:40px 24px}}
h1{{font-size:2.2rem;line-height:1.2}}h2{{margin-top:32px}}.status{{font-weight:700;
color:{'#6ee7b7' if report.passed else '#fda4af'}}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;background:#111e32}}th,td{{padding:12px;
border-bottom:1px solid #30405a;text-align:left}}th{{color:#7dd3fc}}td{{white-space:pre-wrap;
overflow-wrap:anywhere}}.muted{{color:#a8bad2}}footer{{margin-top:40px}}
.diff-card{{margin:14px 0;padding:12px 16px;background:#111e32;border:1px solid #30405a;
border-radius:8px}}.diff-card summary{{cursor:pointer}}
.diff-added td:last-child{{color:#6ee7b7}}.diff-removed td:last-child{{color:#fda4af}}
.diff-changed td:last-child,.diff-type_changed td:last-child{{color:#fcd34d}}
.side-by-side{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:10px}}
.side-by-side pre{{background:#0b1220;padding:10px;border-radius:6px;overflow-x:auto;
white-space:pre-wrap;margin:0}}
</style></head><body><main><p class="muted">llm-drift-bench / offline report</p>
<h1>Model comparison <span class="status">{state}</span></h1>
<p>Baseline: {escape(report.baseline_model)}<br>Candidate: {escape(report.candidate_model)}</p>
<p>{report.total_samples} samples / {report.drifted_samples} drifted</p>
<h2>Summary</h2>{table(('Metric', 'Baseline', 'Candidate / value'), summary_rows(report))}
<h2>Per-prompt results</h2>{table(DETAIL_HEADERS, detail_rows(report))}
{extra_sections}{tool_section}<h2>Checks and notes</h2><ul>{notes}</ul>{thresholds}
<footer class="muted">Saved evaluation, not live monitoring. Agreement is not correctness.
Missing measurements remain unavailable.</footer></main></body></html>
"""
