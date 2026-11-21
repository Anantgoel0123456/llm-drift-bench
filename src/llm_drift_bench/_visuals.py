"""PR comment exports, opt-in side-by-side JSON diffs, and inline SVG Pareto HTML dashboards."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

from llm_drift_bench._breakdowns import category_summary
from llm_drift_bench._report_views import (
    DETAIL_HEADERS,
    TOOL_HEADERS,
    TOOL_NOTE,
    detail_rows,
    markdown_cell,
    measurement,
    tool_check_rows,
)
from llm_drift_bench.comparison import _hotspots, compare_reports
from llm_drift_bench.metrics import _json_type_tag, _values_equal, flatten_json_paths
from llm_drift_bench.models import DriftBenchmarkReport
from llm_drift_bench.tool_log import ToolCheckRecord
from llm_drift_bench.validator import extract_json_payload


def build_json_field_diffs(
    baseline_text: str,
    candidate_text: str,
) -> list[tuple[str, str, str, str]]:
    """Return structured leaf-level diff rows: (path, baseline_val, candidate_val, status)."""
    ok_a, parsed_a, _ = extract_json_payload(baseline_text)
    ok_b, parsed_b, _ = extract_json_payload(candidate_text)
    if not (ok_a and ok_b):
        return []

    flat_a = flatten_json_paths(parsed_a)
    flat_b = flatten_json_paths(parsed_b)
    rows: list[tuple[str, str, str, str]] = []
    for path in sorted(set(flat_a) | set(flat_b)):
        in_a = path in flat_a
        in_b = path in flat_b
        if in_a and not in_b:
            rows.append((path, json.dumps(flat_a[path]), "(missing)", "removed"))
        elif in_b and not in_a:
            rows.append((path, "(missing)", json.dumps(flat_b[path]), "added"))
        else:
            val_a, val_b = flat_a[path], flat_b[path]
            if not _values_equal(val_a, val_b):
                kind = (
                    "type_changed"
                    if _json_type_tag(val_a) != _json_type_tag(val_b)
                    else "changed"
                )
                rows.append((path, json.dumps(val_a), json.dumps(val_b), kind))
    return rows


def render_html_diffs_section(
    report: DriftBenchmarkReport,
    raw_pairs: Mapping[str, tuple[str, str]],
) -> str:
    """Render opt-in pure-CSS collapsible side-by-side diffs for HTML reports."""
    blocks: list[str] = []
    for item in report.results:
        pair = raw_pairs.get(item.prompt_id)
        if pair is None:
            continue
        base_out, cand_out = pair
        if base_out == cand_out and not item.drifted:
            continue
        field_diffs = build_json_field_diffs(base_out, cand_out)
        badge = "DRIFT" if item.drifted else "CHANGED"
        if field_diffs:
            diff_rows = "".join(
                f"<tr class='diff-{escape(status)}'>"
                f"<td><code>{escape(path)}</code></td>"
                f"<td><code>{escape(bval)}</code></td>"
                f"<td><code>{escape(cval)}</code></td>"
                f"<td>{escape(status)}</td></tr>"
                for path, bval, cval, status in field_diffs
            )
            inner = (
                "<div class='scroll'><table><thead><tr>"
                "<th scope='col'>Field path</th><th scope='col'>Baseline</th>"
                "<th scope='col'>Candidate</th><th scope='col'>Change</th>"
                f"</tr></thead><tbody>{diff_rows}</tbody></table></div>"
            )
        else:
            inner = (
                "<div class='side-by-side'>"
                f"<div><h4>Baseline</h4><pre>{escape(base_out[:2000])}</pre></div>"
                f"<div><h4>Candidate</h4><pre>{escape(cand_out[:2000])}</pre></div>"
                "</div>"
            )
        open_attr = " open" if item.drifted else ""
        blocks.append(
            f"<details class='diff-card'{open_attr}>"
            f"<summary><strong>{escape(item.prompt_id)}</strong> "
            f"<span class='muted'>({escape(item.category)} | drift "
            f"{item.composite_drift_score:.4f} | {badge})</span></summary>"
            f"{inner}</details>"
        )
    if not blocks:
        return ""
    return "<h2>Opt-in response diffs</h2>" + "".join(blocks)


def _delta_str(current: float, previous: float | None, spec: str = "+.4f") -> str:
    if previous is None:
        return ""
    delta = current - previous
    return f" ({format(delta, spec)})"


def canonical_pr_comment(
    report: DriftBenchmarkReport,
    *,
    previous: DriftBenchmarkReport | None = None,
    tool_checks: Sequence[ToolCheckRecord] = (),
) -> str:
    """Format a compact GitHub PR comment with status badge, deltas, and collapsible details."""
    badge = "PASSED" if report.passed else "FAILED"
    marker = "<!-- llm-drift-bench-pr-comment -->"
    drift_delta = _delta_str(
        report.mean_composite_drift,
        previous.mean_composite_drift if previous is not None else None,
        "+.4f",
    )
    conf_delta = _delta_str(
        report.candidate_json_conformity_rate,
        previous.candidate_json_conformity_rate if previous is not None else None,
        "+.1%",
    )
    lines = [
        marker,
        f"### LLM Drift Benchmark: **{badge}**",
        "",
        (
            f"**Baseline:** `{markdown_cell(report.baseline_model)}` · "
            f"**Candidate:** `{markdown_cell(report.candidate_model)}` · "
            f"**Drifted:** `{report.drifted_samples}/{report.total_samples}`"
        ),
        "",
        "| Metric | Baseline | Candidate |",
        "| :--- | ---: | ---: |",
        (
            f"| JSON conformity | {report.baseline_json_conformity_rate:.1%} | "
            f"{report.candidate_json_conformity_rate:.1%}{conf_delta} |"
        ),
        f"| Mean composite drift | 0.0000 | {report.mean_composite_drift:.4f}{drift_delta} |",
        f"| Max composite drift | 0.0000 | {report.max_composite_drift:.4f} |",
        (
            f"| P95 latency (ms) | {measurement(report.baseline_p95_latency_ms, '.2f')} | "
            f"{measurement(report.candidate_p95_latency_ms, '.2f')} |"
        ),
    ]
    if report.failure_reasons:
        lines += ["", "**Gate failures:**"]
        lines.extend(f"- {markdown_cell(reason)}" for reason in report.failure_reasons)

    lines += [
        "",
        "<details>",
        f"<summary>Per-prompt breakdown ({report.drifted_samples} drifted / "
        f"{report.total_samples} total)</summary>",
        "",
        "| " + " | ".join(DETAIL_HEADERS) + " |",
        "| " + " | ".join(["---"] * len(DETAIL_HEADERS)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(markdown_cell(v) for v in row) + " |"
        for row in detail_rows(report)
    )
    lines += ["", "</details>"]

    if tool_checks:
        lines += [
            "",
            "<details>",
            f"<summary>Tool evidence ({len(tool_checks)} check(s))</summary>",
            "",
            TOOL_NOTE,
            "",
            "| " + " | ".join(TOOL_HEADERS) + " |",
            "| " + " | ".join(["---"] * len(TOOL_HEADERS)) + " |",
        ]
        lines.extend(
            "| " + " | ".join(markdown_cell(v) for v in row) + " |"
            for row in tool_check_rows(tool_checks)
        )
        lines += ["", "</details>"]

    return "\n".join(lines) + "\n"


def _pareto_svg(rows: Sequence[Any]) -> str:
    """Render an offline inline SVG scatter plot: Drift (X) vs Conformity or P95 Latency (Y)."""
    width, height = 720, 320
    pad_l, pad_r, pad_t, pad_b = 64, 32, 28, 48
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    has_latency = all(row.p95_ms is not None for row in rows)
    max_x = max([row.mean_drift for row in rows] + [0.1])
    max_x = min(1.0, max_x * 1.15)
    if has_latency:
        y_vals = [float(row.p95_ms) for row in rows]
        max_y = max(max(y_vals) * 1.15, 1.0)
        y_label = "P95 Latency (ms) — lower is faster"
    else:
        max_y = 100.0
        y_label = "JSON Conformity (%) — higher is better"

    points: list[str] = []
    for row in rows:
        cx = pad_l + (row.mean_drift / max_x) * plot_w
        if has_latency:
            y_val = float(row.p95_ms)
            cy = pad_t + plot_h - (y_val / max_y) * plot_h
        else:
            y_val = row.conformity * 100.0
            cy = pad_t + plot_h - (y_val / 100.0) * plot_h
        color = "#6ee7b7" if row.passed else "#fda4af"
        points.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='7' fill='{color}' "
            f"stroke='#0b1220' stroke-width='2'/>"
            f"<text x='{cx + 10:.1f}' y='{cy + 4:.1f}' fill='#e8eef8' font-size='12'>"
            f"{escape(row.candidate)}</text>"
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        "aria-label='Candidate Pareto scatter chart' "
        "style='width:100%;max-width:720px;height:auto;background:#111e32;border-radius:8px'>"
        f"<line x1='{pad_l}' y1='{pad_t + plot_h}' x2='{width - pad_r}' y2='{pad_t + plot_h}' "
        "stroke='#475569' stroke-width='1.5'/>"
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + plot_h}' "
        "stroke='#475569' stroke-width='1.5'/>"
        f"<text x='{pad_l + plot_w / 2:.1f}' y='{height - 12}' text-anchor='middle' "
        "fill='#a8bad2' font-size='12'>Mean Composite Drift (lower is closer to baseline)</text>"
        f"<text x='18' y='{pad_t + plot_h / 2:.1f}' text-anchor='middle' fill='#a8bad2' "
        f"font-size='12' transform='rotate(-90 18 {pad_t + plot_h / 2:.1f})'>"
        f"{escape(y_label)}</text>"
        + "".join(points)
        + "</svg>"
    )


def comparison_html(reports: Sequence[DriftBenchmarkReport]) -> str:
    """Export a standalone script-free HTML comparison dashboard with an inline SVG Pareto chart."""
    rows = compare_reports(reports)
    categories = category_summary(reports)
    hotspots = _hotspots(reports, 10)
    baseline_name = reports[0].baseline_model

    cand_rows_html = "".join(
        f"<tr><td>{escape(r.candidate)}</td>"
        f"<td class='{'pass' if r.passed else 'fail'}'>{'PASS' if r.passed else 'FAIL'}</td>"
        f"<td>{r.drifted}/{r.samples}</td>"
        f"<td>{r.mean_drift:.4f}</td><td>{r.max_drift:.4f}</td>"
        f"<td>{r.conformity:.1%}</td>"
        f"<td>{'n/a' if r.p95_ms is None else f'{r.p95_ms:.2f} ms'}</td>"
        f"<td>{escape(r.worst_prompt_id or 'n/a')}</td></tr>"
        for r in rows
    )
    cat_rows_html = "".join(
        f"<tr><td>{escape(c.category)}</td><td>{escape(c.candidate)}</td>"
        f"<td>{c.drifted}/{c.samples}</td><td>{c.mean_drift:.4f}</td></tr>"
        for c in categories
    )
    hot_rows_html = "".join(
        f"<tr><td>{escape(h.prompt_id)}</td><td>{escape(h.category)}</td>"
        f"<td>{h.drifted_candidates}/{len(reports)}</td><td>{h.max_drift:.4f}</td></tr>"
        for h in hotspots
    ) or "<tr><td colspan='4' class='muted'>No individual prompts drifted.</td></tr>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>LLM Drift Multi-Candidate Comparison</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#0b1220;color:#e8eef8;
font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:40px 24px}}
h1{{font-size:2rem}}h2{{margin-top:32px}}.pass{{color:#6ee7b7;font-weight:700}}
.fail{{color:#fda4af;font-weight:700}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;background:#111e32;margin-top:12px}}
th,td{{padding:10px 12px;border-bottom:1px solid #30405a;text-align:left}}
th{{color:#7dd3fc}}.muted{{color:#a8bad2}}footer{{margin-top:40px}}
</style></head><body><main>
<p class="muted">llm-drift-bench / multi-candidate comparison</p>
<h1>Candidate Ranking vs <span>{escape(baseline_name)}</span></h1>
<h2>Pareto View</h2>
{_pareto_svg(rows)}
<h2>Ranked Candidates</h2>
<div class="scroll"><table><thead><tr>
<th>Candidate</th><th>Verdict</th><th>Drifted</th><th>Mean Drift</th>
<th>Max Drift</th><th>JSON Conformity</th><th>P95 Latency</th><th>Worst Prompt</th>
</tr></thead><tbody>{cand_rows_html}</tbody></table></div>
<h2>Prompt Hotspots</h2>
<div class="scroll"><table><thead><tr>
<th>Prompt ID</th><th>Category</th><th>Drifted Candidates</th><th>Peak Drift</th>
</tr></thead><tbody>{hot_rows_html}</tbody></table></div>
<h2>Category Breakdown</h2>
<div class="scroll"><table><thead><tr>
<th>Category</th><th>Candidate</th><th>Drifted</th><th>Mean Drift</th>
</tr></thead><tbody>{cat_rows_html}</tbody></table></div>
<footer class="muted">Offline comparison dashboard. Zero scripts, zero external assets.</footer>
</main></body></html>
"""
