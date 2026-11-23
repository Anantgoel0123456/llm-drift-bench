"""Append-only drift history ledger with terminal sparklines and offline SVG trend charts."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from llm_drift_bench._file_io import parse_source_json, read_utf8
from llm_drift_bench._numbers import finite_number
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
        if type(data["passed"]) is not bool:
            raise ValueError("History entry 'passed' must be a boolean.")
        for key in ("recorded_at", "baseline_model", "candidate_model"):
            if not isinstance(data[key], str) or not data[key].strip():
                raise ValueError(f"History entry {key!r} must be a non-empty string.")
        for key in ("total_samples", "drifted_samples"):
            val = data[key]
            if type(val) is not int or val < 0:
                raise ValueError(f"History entry {key!r} must be a non-negative integer.")
        if data["drifted_samples"] > data["total_samples"]:
            raise ValueError("History entry 'drifted_samples' cannot exceed 'total_samples'.")

        unit_vals: dict[str, float] = {}
        for key in (
            "mean_composite_drift",
            "max_composite_drift",
            "candidate_json_conformity_rate",
        ):
            raw_num = data[key]
            if isinstance(raw_num, bool) or not isinstance(raw_num, (int, float)):
                raise ValueError(f"History entry {key!r} must be a number in [0, 1].")
            num = finite_number(raw_num)
            if not 0.0 <= num <= 1.0:
                raise ValueError(f"History entry {key!r} must lie between 0 and 1.")
            unit_vals[key] = num

        p95_raw = data.get("candidate_p95_latency_ms")
        p95: float | None = None
        if p95_raw is not None:
            if isinstance(p95_raw, bool) or not isinstance(p95_raw, (int, float)):
                raise ValueError("History entry 'candidate_p95_latency_ms' must be numeric.")
            p95 = finite_number(p95_raw)
            if p95 < 0:
                raise ValueError("History entry 'candidate_p95_latency_ms' must be non-negative.")

        for opt in ("tag", "git_sha"):
            if data.get(opt) is not None and not isinstance(data[opt], str):
                raise ValueError(f"History entry {opt!r} must be a string or null.")

        return cls(
            recorded_at=data["recorded_at"],
            baseline_model=data["baseline_model"],
            candidate_model=data["candidate_model"],
            passed=data["passed"],
            total_samples=data["total_samples"],
            drifted_samples=data["drifted_samples"],
            mean_composite_drift=unit_vals["mean_composite_drift"],
            max_composite_drift=unit_vals["max_composite_drift"],
            candidate_json_conformity_rate=unit_vals["candidate_json_conformity_rate"],
            candidate_p95_latency_ms=p95,
            tag=data.get("tag"),
            git_sha=data.get("git_sha"),
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
    if not isinstance(report, DriftBenchmarkReport):
        raise TypeError("record_report requires a canonical DriftBenchmarkReport instance.")
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
    cleaned = [finite_number(v) for v in values]
    lo = min(cleaned) if low is None else finite_number(low)
    hi = max(cleaned) if high is None else finite_number(high)
    span = hi - lo
    if span <= 0:
        return _SPARK_BARS[len(_SPARK_BARS) // 2] * len(cleaned)
    chars: list[str] = []
    last_idx = len(_SPARK_BARS) - 1
    for value in cleaned:
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


def _trend_svg(
    title: str,
    values: Sequence[float],
    color: str,
    *,
    y_max: float = 1.0,
    labels: Sequence[str] | None = None,
) -> str:
    width, height = 680, 200
    pad_l, pad_r, pad_t, pad_b = 52, 24, 28, 32
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(values)
    if n == 0:
        return (
            f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}' "
            "style='width:100%;max-width:680px;height:auto;background:#111e32;border-radius:8px;"
            "margin-top:10px'>"
            f"<text x='{pad_l}' y='18' fill='#7dd3fc' font-size='13' font-weight='600'>"
            f"{escape(title)}</text>"
            f"<text x='{width / 2:.1f}' y='{height / 2:.1f}' text-anchor='middle' "
            "fill='#94a3b8' font-size='13'>No history entries recorded.</text></svg>"
        )
    grid_parts: list[str] = []
    for frac in (0.0, 0.5, 1.0):
        gy = pad_t + plot_h - frac * plot_h
        tick_val = frac * y_max
        grid_parts.append(
            f"<line x1='{pad_l}' y1='{gy:.1f}' x2='{width - pad_r}' y2='{gy:.1f}' "
            "stroke='#1e293b' stroke-dasharray='3,3'/>"
            f"<text x='{pad_l - 8}' y='{gy + 4:.1f}' text-anchor='end' fill='#94a3b8' "
            f"font-size='10'>{tick_val:.2f}</text>"
        )
    coords: list[tuple[float, float]] = []
    for idx, val in enumerate(values):
        x = pad_l + (plot_w / 2 if n == 1 else (idx / (n - 1)) * plot_w)
        y = pad_t + plot_h - (max(0.0, min(y_max, val)) / max(y_max, 1e-9)) * plot_h
        coords.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    dot_parts: list[str] = []
    for idx, (x, y) in enumerate(coords):
        tip = labels[idx] if labels and idx < len(labels) else f"Run {idx + 1}: {values[idx]:.4f}"
        dot_parts.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4.5' fill='{color}' stroke='#0b1220'>"
            f"<title>{escape(tip)}</title></circle>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}' "
        "style='width:100%;max-width:680px;height:auto;background:#111e32;border-radius:8px;"
        "margin-top:10px'>"
        f"<text x='{pad_l}' y='18' fill='#7dd3fc' font-size='13' font-weight='600'>"
        f"{escape(title)}</text>"
        f"{''.join(grid_parts)}"
        f"<line x1='{pad_l}' y1='{pad_t + plot_h}' x2='{width - pad_r}' y2='{pad_t + plot_h}' "
        "stroke='#475569'/>"
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t + plot_h}' stroke='#475569'/>"
        f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{polyline}'/>"
        f"{''.join(dot_parts)}</svg>"
    )


@dataclass(frozen=True)
class HistoryTrendCheck:
    """Result of comparing the latest recorded run against a moving window of prior runs."""

    regressed: bool
    latest_drift: float
    window_mean_drift: float
    drift_delta: float
    latest_conformity: float
    window_mean_conformity: float
    conformity_delta: float
    window_size: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_history_regression(
    entries: Sequence[HistoryEntry],
    *,
    window: int = 3,
    max_drift_increase: float = 0.05,
    max_conformity_drop: float = 0.02,
) -> HistoryTrendCheck:
    """Detect gradual drift or conformity degradation vs the preceding moving window."""
    if len(entries) < 2:
        raise ValueError("Trend regression check requires at least 2 recorded history entries.")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer.")
    for name, val in (
        ("max_drift_increase", max_drift_increase),
        ("max_conformity_drop", max_conformity_drop),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"{name} must be a finite non-negative number.")
        num = finite_number(val)
        if num < 0:
            raise ValueError(f"{name} must be a finite non-negative number.")

    prior = entries[max(0, len(entries) - 1 - window) : -1]
    latest = entries[-1]
    win_drift = sum(e.mean_composite_drift for e in prior) / len(prior)
    win_conf = sum(e.candidate_json_conformity_rate for e in prior) / len(prior)
    drift_delta = latest.mean_composite_drift - win_drift
    conf_delta = latest.candidate_json_conformity_rate - win_conf

    reasons: list[str] = []
    if drift_delta > max_drift_increase:
        reasons.append(
            f"Mean drift increased by +{drift_delta:.4f} vs prior {len(prior)}-run mean "
            f"({win_drift:.4f} -> {latest.mean_composite_drift:.4f}, "
            f"limit +{max_drift_increase:.4f})."
        )
    if -conf_delta > max_conformity_drop:
        reasons.append(
            f"JSON conformity dropped by {conf_delta:.1%} vs prior {len(prior)}-run mean "
            f"({win_conf:.1%} -> {latest.candidate_json_conformity_rate:.1%}, "
            f"limit -{max_conformity_drop:.1%})."
        )
    regressed = bool(reasons)
    reason = " ".join(reasons) if reasons else (
        f"Stable relative to prior {len(prior)} run(s) "
        f"(drift delta {drift_delta:+.4f}, conformity delta {conf_delta:+.1%})."
    )
    return HistoryTrendCheck(
        regressed=regressed,
        latest_drift=round(latest.mean_composite_drift, 4),
        window_mean_drift=round(win_drift, 4),
        drift_delta=round(drift_delta, 4),
        latest_conformity=round(latest.candidate_json_conformity_rate, 4),
        window_mean_conformity=round(win_conf, 4),
        conformity_delta=round(conf_delta, 4),
        window_size=len(prior),
        reason=reason,
    )


def export_history_html(
    entries: Sequence[HistoryEntry],
    output_path: str | Path | None = None,
) -> str:
    """Export an offline script-free HTML trend dashboard from a drift history ledger."""
    drift_labels = [
        f"{e.candidate_model} ({e.tag or e.recorded_at[:10]}): drift {e.mean_composite_drift:.4f}"
        for e in entries
    ]
    conf_labels = [
        f"{e.candidate_model} ({e.tag or e.recorded_at[:10]}): "
        f"conformity {e.candidate_json_conformity_rate:.1%}"
        for e in entries
    ]
    drift_chart = _trend_svg(
        "Mean Composite Drift Over Time (0.0 = identical)",
        [e.mean_composite_drift for e in entries],
        "#fbbf24",
        labels=drift_labels,
    )
    conf_chart = _trend_svg(
        "Candidate JSON Conformity Rate Over Time",
        [e.candidate_json_conformity_rate for e in entries],
        "#6ee7b7",
        labels=conf_labels,
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


def summarize_history(entries: Sequence[HistoryEntry]) -> dict[str, Any]:
    """Compute per-candidate pass rate, drift range, and latest-vs-first trend stats."""
    grouped: dict[str, list[HistoryEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.candidate_model].append(entry)

    candidates: dict[str, dict[str, Any]] = {}
    for cand in sorted(grouped):
        runs = grouped[cand]
        drifts = [r.mean_composite_drift for r in runs]
        passed_n = sum(1 for r in runs if r.passed)
        candidates[cand] = {
            "runs": len(runs),
            "passed_runs": passed_n,
            "pass_rate": round(passed_n / len(runs), 4),
            "min_mean_drift": round(min(drifts), 4),
            "max_mean_drift": round(max(drifts), 4),
            "latest_mean_drift": round(runs[-1].mean_composite_drift, 4),
            "drift_change_since_first": round(
                runs[-1].mean_composite_drift - runs[0].mean_composite_drift, 4
            ),
            "latest_conformity": round(runs[-1].candidate_json_conformity_rate, 4),
            "latest_passed": runs[-1].passed,
        }
    return {
        "total_entries": len(entries),
        "candidates": candidates,
    }


def prune_history(
    ledger_path: str | Path,
    *,
    keep_last: int,
    per_candidate: bool = True,
) -> tuple[int, int]:
    """Atomically compact a JSONL ledger, keeping the newest `keep_last` runs."""
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("keep_last must be a positive integer.")
    source = Path(ledger_path)
    entries = load_history(source)
    if per_candidate:
        seen_per_cand: dict[str, int] = defaultdict(int)
        kept_rev: list[HistoryEntry] = []
        for entry in reversed(entries):
            if seen_per_cand[entry.candidate_model] < keep_last:
                seen_per_cand[entry.candidate_model] += 1
                kept_rev.append(entry)
        kept = list(reversed(kept_rev))
    else:
        kept = entries[-keep_last:]

    removed = len(entries) - len(kept)
    payload = "".join(json.dumps(e.to_dict()) + "\n" for e in kept)
    source.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=source.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, source)
    finally:
        tmp_path.unlink(missing_ok=True)
    return len(kept), removed
