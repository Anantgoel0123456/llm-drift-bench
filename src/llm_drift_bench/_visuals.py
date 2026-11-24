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
from llm_drift_bench.comparison import SortKey, _hotspots, compare_reports
from llm_drift_bench.diffing import diff_sample_pair
from llm_drift_bench.models import DriftBenchmarkReport, ModelSample
from llm_drift_bench.policy import DriftPolicy
from llm_drift_bench.tool_log import ToolCheckRecord


def build_json_field_diffs(
    baseline_text: str,
    candidate_text: str,
    *,
    policy: DriftPolicy | None = None,
) -> list[tuple[str, str, str, str]]:
    """Return structured leaf-level diff rows: (path, baseline_val, candidate_val, status)."""
    diff = diff_sample_pair(
        ModelSample(prompt_id="_diff", output=baseline_text),
        ModelSample(prompt_id="_diff", output=candidate_text),
        policy=policy,
    )
    if diff.mode != "json":
        return []
    rows: list[tuple[str, str, str, str]] = []
    for change in diff.field_changes:
        if change.kind == "removed":
            rows.append((
                change.path,
                json.dumps(change.baseline, ensure_ascii=False),
                "(missing)",
                "removed",
            ))
        elif change.kind == "added":
            rows.append((
                change.path,
                "(missing)",
                json.dumps(change.candidate, ensure_ascii=False),
                "added",
            ))
        else:
            rows.append((
                change.path,
                json.dumps(change.baseline, ensure_ascii=False),
                json.dumps(change.candidate, ensure_ascii=False),
                change.kind,
            ))
    return rows


def render_html_diffs_section(
    report: DriftBenchmarkReport,
    raw_pairs: Mapping[str, tuple[str, str]],
    *,
    policy: DriftPolicy | None = None,
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
        field_diffs = build_json_field_diffs(base_out, cand_out, policy=policy)
        if not field_diffs and not item.drifted and policy is not None:
            # Skip prompts whose only JSON differences were explicitly ignored by policy.
            diff_check = diff_sample_pair(
                ModelSample(prompt_id=item.prompt_id, output=base_out),
                ModelSample(prompt_id=item.prompt_id, output=cand_out),
                policy=policy,
            )
            if diff_check.mode == "json" and not diff_check.changed:
                continue
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

    categories = {row.category for row in report.results}
    if len(categories) >= 2:
        cat_rows = sorted(
            category_summary([report]),
            key=lambda item: (-item.mean_drift, item.category),
        )
        lines += [
            "",
            "| Category | Drifted | Mean Drift |",
            "| :--- | ---: | ---: |",
        ]
        lines.extend(
            f"| `{markdown_cell(item.category)}` | {item.drifted}/{item.samples} | "
            f"{item.mean_drift:.4f} |"
            for item in cat_rows
        )

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


def _pareto_frontier(
    points: Sequence[tuple[float, float, bool]],
    *,
    higher_y_better: bool,
) -> list[tuple[float, float]]:
    """Return non-dominated points ordered by increasing X (lower X is always better)."""
    ordered = sorted(points, key=lambda pt: (pt[0], -pt[1] if higher_y_better else pt[1]))
    frontier: list[tuple[float, float]] = []
    best_y: float | None = None
    for x_val, y_val, _ in ordered:
        if best_y is None or (
            y_val > best_y if higher_y_better else y_val < best_y
        ):
            frontier.append((x_val, y_val))
            best_y = y_val
    return frontier


def _pareto_svg(rows: Sequence[Any]) -> str:
    """Render an offline inline SVG scatter plot with axis ticks and Pareto frontier."""
    width, height = 720, 340
    pad_l, pad_r, pad_t, pad_b = 68, 36, 30, 52
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    has_latency = all(row.p95_ms is not None for row in rows)
    max_x = min(1.0, max([row.mean_drift for row in rows] + [0.1]) * 1.15)
    if has_latency:
        y_vals = [float(row.p95_ms) for row in rows]
        max_y = max(max(y_vals) * 1.15, 1.0)
        y_label = "P95 Latency (ms) — lower is faster"
        higher_y_better = False
    else:
        max_y = 100.0
        y_label = "JSON Conformity (%) — higher is better"
        higher_y_better = True

    def to_svg(x_val: float, y_val: float) -> tuple[float, float]:
        cx = pad_l + (x_val / max_x) * plot_w
        cy = pad_t + plot_h - (y_val / max_y) * plot_h
        return cx, cy

    raw_pts = [
        (row.mean_drift, float(row.p95_ms) if has_latency else row.conformity * 100.0, row.passed)
        for row in rows
    ]
    frontier = _pareto_frontier(raw_pts, higher_y_better=higher_y_better)
    frontier_svg = ""
    if len(frontier) >= 2:
        pts_str = " ".join(f"{to_svg(x, y)[0]:.1f},{to_svg(x, y)[1]:.1f}" for x, y in frontier)
        frontier_svg = (
            f"<polyline fill='none' stroke='#38bdf8' stroke-width='1.8' "
            f"stroke-dasharray='6 4' opacity='0.8' points='{pts_str}'/>"
        )

    grid_svg: list[str] = []
    for frac in (0.0, 0.5, 1.0):
        gx = pad_l + frac * plot_w
        gy = pad_t + plot_h - frac * plot_h
        x_tick = f"{frac * max_x:.2f}"
        y_tick = f"{frac * max_y:.0f}" if has_latency else f"{frac * 100:.0f}%"
        grid_svg.append(
            f"<line x1='{gx:.1f}' y1='{pad_t}' x2='{gx:.1f}' y2='{pad_t + plot_h}' "
            "stroke='#1e293b' stroke-width='1'/>"
            f"<text x='{gx:.1f}' y='{pad_t + plot_h + 16}' text-anchor='middle' "
            f"fill='#94a3b8' font-size='11'>{x_tick}</text>"
            f"<line x1='{pad_l}' y1='{gy:.1f}' x2='{width - pad_r}' y2='{gy:.1f}' "
            "stroke='#1e293b' stroke-width='1'/>"
            f"<text x='{pad_l - 8}' y='{gy + 4:.1f}' text-anchor='end' "
            f"fill='#94a3b8' font-size='11'>{y_tick}</text>"
        )

    placed: list[tuple[float, float]] = []
    points_svg: list[str] = []
    for row, (xv, yv, _) in zip(rows, raw_pts, strict=True):
        cx, cy = to_svg(xv, yv)
        label_y = cy + 4.0
        for px, py in placed:
            if abs(cx - px) < 70 and abs(label_y - py) < 14:
                label_y += 15.0
        label_y = min(pad_t + plot_h - 4.0, max(pad_t + 12.0, label_y))
        placed.append((cx, label_y))
        anchor_end = cx > width * 0.65
        text_x = cx - 10.0 if anchor_end else cx + 10.0
        anchor_attr = "end" if anchor_end else "start"
        color = "#6ee7b7" if row.passed else "#fda4af"
        points_svg.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='7' fill='{color}' "
            f"stroke='#0b1220' stroke-width='2'/>"
            f"<text x='{text_x:.1f}' y='{label_y:.1f}' text-anchor='{anchor_attr}' "
            f"fill='#e8eef8' font-size='12'>{escape(row.candidate)}</text>"
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        "aria-label='Candidate Pareto scatter chart' "
        "style='width:100%;max-width:720px;height:auto;background:#111e32;border-radius:8px'>"
        + "".join(grid_svg)
        + frontier_svg
        + f"<line x1='{pad_l}' y1='{pad_t + plot_h}' x2='{width - pad_r}' y2='{pad_t + plot_h}' "
        "stroke='#475569' stroke-width='1.5'/>"
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + plot_h}' "
        "stroke='#475569' stroke-width='1.5'/>"
        f"<text x='{pad_l + plot_w / 2:.1f}' y='{height - 12}' text-anchor='middle' "
        "fill='#a8bad2' font-size='12'>Mean Composite Drift (lower is closer to baseline)</text>"
        f"<text x='18' y='{pad_t + plot_h / 2:.1f}' text-anchor='middle' fill='#a8bad2' "
        f"font-size='12' transform='rotate(-90 18 {pad_t + plot_h / 2:.1f})'>"
        f"{escape(y_label)}</text>"
        + "".join(points_svg)
        + "</svg>"
    )


def comparison_html(
    reports: Sequence[DriftBenchmarkReport],
    *,
    sort_by: SortKey = "drift",
) -> str:
    """Export a standalone script-free HTML comparison dashboard with an inline SVG Pareto chart."""
    rows = compare_reports(reports, sort_by=sort_by)
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
