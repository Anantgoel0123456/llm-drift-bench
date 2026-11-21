"""Append-only drift history ledger with terminal sparklines and offline SVG trend charts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from llm_drift_bench._file_io import parse_source_json, read_utf8
from llm_drift_bench.models import DriftBenchmarkReport

_SPARK_BARS = "▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class HistoryEntry:
    """One recorded benchmark run in an append-only JSONL history ledger."""

    recorded_at: str
    baseline_model: str
    candidate_model: str
    passed: bool
    total_samples: int
    drifted_samples: int
    mean_composite_drift: float
    max_composite_drift: float
    candidate_json_conformity_rate: float
    candidate_p95_latency_ms: float | None = None
    tag: str | None = None
    git_sha: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_report(
        cls,
        report: DriftBenchmarkReport,
        *,
        tag: str | None = None,
        git_sha: str | None = None,
        recorded_at: str | None = None,
    ) -> HistoryEntry:
        stamp = recorded_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return cls(
            recorded_at=stamp,
            baseline_model=report.baseline_model,
            candidate_model=report.candidate_model,
            passed=report.passed,
            total_samples=report.total_samples,
            drifted_samples=report.drifted_samples,
            mean_composite_drift=round(report.mean_composite_drift, 4),
            max_composite_drift=round(report.max_composite_drift, 4),
            candidate_json_conformity_rate=round(report.candidate_json_conformity_rate, 4),
            candidate_p95_latency_ms=(
                round(report.candidate_p95_latency_ms, 2)
                if report.candidate_p95_latency_ms is not None
                else None
            ),
            tag=tag,
            git_sha=git_sha,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        required = (
            "recorded_at", "baseline_model", "candidate_model", "passed",
            "total_samples", "drifted_samples", "mean_composite_drift",
            "max_composite_drift", "candidate_json_conformity_rate",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"History entry missing required field(s): {missing}.")
        p95 = data.get("candidate_p95_latency_ms")
        return cls(
            recorded_at=str(data["recorded_at"]),
            baseline_model=str(data["baseline_model"]),
            candidate_model=str(data["candidate_model"]),
            passed=bool(data["passed"]),
            total_samples=int(data["total_samples"]),
            drifted_samples=int(data["drifted_samples"]),
            mean_composite_drift=float(data["mean_composite_drift"]),
            max_composite_drift=float(data["max_composite_drift"]),
            candidate_json_conformity_rate=float(data["candidate_json_conformity_rate"]),
            candidate_p95_latency_ms=float(p95) if p95 is not None else None,
            tag=str(data["tag"]) if data.get("tag") is not None else None,
            git_sha=str(data["git_sha"]) if data.get("git_sha") is not None else None,
        )


def record_report(
    ledger_path: str | Path,
    report: DriftBenchmarkReport,
    *,
    tag: str | None = None,
    git_sha: str | None = None,
    recorded_at: str | None = None,
) -> HistoryEntry:
    """Append one HistoryEntry for a validated DriftBenchmarkReport to a JSONL ledger."""
    entry = HistoryEntry.from_report(report, tag=tag, git_sha=git_sha, recorded_at=recorded_at)
    target = Path(ledger_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry.to_dict()) + "\n")
    return entry


def load_history(ledger_path: str | Path) -> list[HistoryEntry]:
    """Load all recorded HistoryEntry items from a JSONL ledger."""
    source = Path(ledger_path)
    text = read_utf8(source)
    entries: list[HistoryEntry] = []
    for lineno, line in enumerate(text.split("\n"), start=1):
        if not line.strip(" \t\r"):
            continue
        parsed = parse_source_json(line, source, line=lineno)
        if not isinstance(parsed, dict):
            raise ValueError(f"{source} (line {lineno}): history record must be a JSON object.")
        entries.append(HistoryEntry.from_dict(parsed))
    return entries


def sparkline(
    values: Sequence[float],
    *,
    low: float | None = None,
    high: float | None = None,
) -> str:
    """Render a compact Unicode sparkline string for a numeric sequence."""
    if not values:
        return ""
    lo = min(values) if low is None else low
    hi = max(values) if high is None else high
    span = hi - lo
    if span <= 0:
        return _SPARK_BARS[len(_SPARK_BARS) // 2] * len(values)
    chars: list[str] = []
    last_idx = len(_SPARK_BARS) - 1
    for value in values:
        norm = max(0.0, min(1.0, (value - lo) / span))
        chars.append(_SPARK_BARS[round(norm * last_idx)])
    return "".join(chars)


def format_history_text(entries: Sequence[HistoryEntry]) -> str:
    """Format history entries with drift and conformity sparklines for terminal viewing."""
    if not entries:
        return "No history entries recorded."
    drift_spark = sparkline([e.mean_composite_drift for e in entries], low=0.0, high=1.0)
    conf_spark = sparkline([e.candidate_json_conformity_rate for e in entries], low=0.0, high=1.0)
    lines = [
        f"Drift History ({len(entries)} run(s))",
        f"  Mean Drift Trend      : {drift_spark}",
        f"  JSON Conformity Trend : {conf_spark}",
        "-" * 86,
        f" {'Recorded At':<20} | {'Candidate':<18} | {'Tag':<10} | "
        f"{'State':<5} | {'Drift':>6} | {'JSON':>6}",
        "-" * 86,
    ]
    for e in entries:
        state = "PASS" if e.passed else "FAIL"
        tag_str = (e.tag or e.git_sha or "-")[:10]
        cand = e.candidate_model if len(e.candidate_model) <= 18 else e.candidate_model[:15] + "..."
        lines.append(
            f" {e.recorded_at[:20]:<20} | {cand:<18} | {tag_str:<10} | "
            f"{state:<5} | {e.mean_composite_drift:>6.4f} | "
            f"{e.candidate_json_conformity_rate:>5.1%}"
        )
    lines.append("-" * 86)
    return "\n".join(lines)


def _trend_svg(title: str, values: Sequence[float], color: str, *, y_max: float = 1.0) -> str:
    width, height = 680, 200
    pad_l, pad_r, pad_t, pad_b = 52, 24, 28, 32
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(values)
    coords: list[tuple[float, float]] = []
    for idx, val in enumerate(values):
        x = pad_l + (plot_w / 2 if n == 1 else (idx / (n - 1)) * plot_w)
        y = pad_t + plot_h - (max(0.0, min(y_max, val)) / max(y_max, 1e-9)) * plot_h
        coords.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    dots = "".join(
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4.5' fill='{color}' stroke='#0b1220'/>"
        for x, y in coords
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}' "
        "style='width:100%;max-width:680px;height:auto;background:#111e32;border-radius:8px;"
        "margin-top:10px'>"
        f"<text x='{pad_l}' y='18' fill='#7dd3fc' font-size='13' font-weight='600'>"
        f"{escape(title)}</text>"
        f"<line x1='{pad_l}' y1='{pad_t + plot_h}' x2='{width - pad_r}' y2='{pad_t + plot_h}' "
        "stroke='#475569'/>"
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + plot_h}' stroke='#475569'/>"
        f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{polyline}'/>"
        f"{dots}</svg>"
    )


def export_history_html(
    entries: Sequence[HistoryEntry],
    output_path: str | Path | None = None,
) -> str:
    """Export an offline script-free HTML trend dashboard from a drift history ledger."""
    drift_chart = _trend_svg(
        "Mean Composite Drift Over Time (0.0 = identical)",
        [e.mean_composite_drift for e in entries],
        "#fbbf24",
    )
    conf_chart = _trend_svg(
        "Candidate JSON Conformity Rate Over Time",
        [e.candidate_json_conformity_rate for e in entries],
        "#6ee7b7",
    )
    def _fmt_p95(val: float | None) -> str:
        return "n/a" if val is None else f"{val:.2f} ms"

    rows_html = "".join(
        f"<tr><td>{escape(e.recorded_at)}</td><td>{escape(e.candidate_model)}</td>"
        f"<td>{escape(e.tag or '-')}</td><td>{escape(e.git_sha or '-')}</td>"
        f"<td class='{'pass' if e.passed else 'fail'}'>{'PASS' if e.passed else 'FAIL'}</td>"
        f"<td>{e.drifted_samples}/{e.total_samples}</td>"
        f"<td>{e.mean_composite_drift:.4f}</td>"
        f"<td>{e.candidate_json_conformity_rate:.1%}</td>"
        f"<td>{_fmt_p95(e.candidate_p95_latency_ms)}</td></tr>"
        for e in entries
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>LLM Drift History Trends</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#0b1220;color:#e8eef8;
font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:40px 24px}}
.pass{{color:#6ee7b7;font-weight:700}}.fail{{color:#fda4af;font-weight:700}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:#111e32;
margin-top:16px}}th,td{{padding:10px 12px;border-bottom:1px solid #30405a;text-align:left}}
th{{color:#7dd3fc}}.muted{{color:#a8bad2}}
</style></head><body><main>
<p class="muted">llm-drift-bench / offline drift ledger</p>
<h1>Drift History Trends ({len(entries)} runs)</h1>
{drift_chart}
{conf_chart}
<h2>Recorded Runs</h2>
<div class="scroll"><table><thead><tr>
<th>Timestamp</th><th>Candidate</th><th>Tag</th><th>Git SHA</th><th>Verdict</th>
<th>Drifted</th><th>Mean Drift</th><th>JSON Conformity</th><th>P95 Latency</th>
</tr></thead><tbody>{rows_html}</tbody></table></div>
</main></body></html>
"""
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
    return html
