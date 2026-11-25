"""Unit and CLI tests for the append-only drift history ledger and trend exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main
from llm_drift_bench.history import (
    HistoryEntry,
    detect_history_regression,
    export_history_html,
    format_history_text,
    load_history,
    prune_history,
    record_report,
    sparkline,
    summarize_history,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_sparkline_scaling_and_constant_series() -> None:
    assert sparkline([]) == ""
    assert sparkline([0.0, 0.5, 1.0], low=0.0, high=1.0) == "▁▅█"
    const = sparkline([0.2, 0.2, 0.2])
    assert len(const) == 3 and len(set(const)) == 1


def test_cli_history_record_and_show_formats(tmp_path: Path, capsys) -> None:
    ledger = tmp_path / "ledger" / "history.jsonl"

    assert (
        main(
            [
                "history",
                "record",
                "--ledger",
                str(ledger),
                "-i",
                str(EXAMPLES / "expected-pass.json"),
                "--tag",
                "q8",
                "--git-sha",
                "abc1234",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "history",
                "record",
                "--ledger",
                str(ledger),
                "-i",
                str(EXAMPLES / "expected-drift.json"),
                "--tag",
                "q4",
            ]
        )
        == 0
    )
    capsys.readouterr()

    entries = load_history(ledger)
    assert len(entries) == 2
    assert entries[0].tag == "q8" and entries[0].git_sha == "abc1234"
    assert entries[1].tag == "q4" and entries[1].passed is False

    # Text sparklines
    assert main(["history", "show", "--ledger", str(ledger)]) == 0
    text_out = capsys.readouterr().out
    assert "Mean Drift Trend" in text_out
    assert "JSON Conformity Trend" in text_out
    assert "q8" in text_out and "q4" in text_out

    # JSON with --limit 1
    rc_lim = main(
        ["history", "show", "--ledger", str(ledger), "--limit", "1", "--format", "json"]
    )
    assert rc_lim == 0
    latest = json.loads(capsys.readouterr().out)
    assert len(latest) == 1 and latest[0]["tag"] == "q4"

    # Standalone HTML trend dashboard with SVG charts
    trend_html = tmp_path / "trends.html"
    assert (
        main(
            [
                "history",
                "show",
                "--ledger",
                str(ledger),
                "--format",
                "html",
                "-o",
                str(trend_html),
            ]
        )
        == 0
    )
    html_content = trend_html.read_text(encoding="utf-8")
    assert "<svg" in html_content
    assert "<polyline" in html_content
    assert "<script" not in html_content.lower()

    assert format_history_text([]) == "No history entries recorded."
    assert "<svg" in export_history_html(entries)
    assert "No history entries recorded." in export_history_html([])

    # Filter by --tag and --candidate
    capsys.readouterr()
    rc_tag = main(["history", "show", "--ledger", str(ledger), "--tag", "q8", "--format", "json"])
    assert rc_tag == 0
    filtered = json.loads(capsys.readouterr().out)
    assert len(filtered) == 1 and filtered[0]["tag"] == "q8"


def test_history_entry_strict_validation() -> None:
    valid = {
        "recorded_at": "2026-11-21T00:00:00Z",
        "baseline_model": "base",
        "candidate_model": "cand",
        "passed": True,
        "total_samples": 2,
        "drifted_samples": 0,
        "mean_composite_drift": 0.1,
        "max_composite_drift": 0.2,
        "candidate_json_conformity_rate": 1.0,
    }
    assert HistoryEntry.from_dict(valid).passed is True
    with pytest.raises(ValueError, match="boolean"):
        HistoryEntry.from_dict({**valid, "passed": "false"})
    with pytest.raises(ValueError, match="cannot exceed"):
        HistoryEntry.from_dict({**valid, "drifted_samples": 5})
    with pytest.raises(ValueError, match="between 0 and 1"):
        HistoryEntry.from_dict({**valid, "mean_composite_drift": 1.5})


def test_history_trend_regression_detection_and_svg_tooltips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def _entry(drift: float, conf: float, tag: str) -> HistoryEntry:
        return HistoryEntry(
            recorded_at="2026-11-23T00:00:00Z",
            baseline_model="base",
            candidate_model="cand",
            passed=drift < 0.25,
            total_samples=10,
            drifted_samples=0 if drift < 0.25 else 2,
            mean_composite_drift=drift,
            max_composite_drift=drift,
            candidate_json_conformity_rate=conf,
            tag=tag,
        )

    stable = detect_history_regression(
        [_entry(0.05, 1.0, "r1"), _entry(0.06, 1.0, "r2"), _entry(0.07, 1.0, "r3")],
        window=2,
        max_drift_increase=0.05,
    )
    assert stable.regressed is False
    assert stable.window_size == 2

    regressed = detect_history_regression(
        [_entry(0.05, 1.0, "r1"), _entry(0.05, 1.0, "r2"), _entry(0.18, 0.90, "r3")],
        window=2,
        max_drift_increase=0.05,
        max_conformity_drop=0.02,
    )
    assert regressed.regressed is True
    assert "Mean drift increased" in regressed.reason
    assert "JSON conformity dropped" in regressed.reason

    with pytest.raises(ValueError, match="at least 2"):
        detect_history_regression([_entry(0.05, 1.0, "r1")])

    html = export_history_html([_entry(0.05, 1.0, "v1"), _entry(0.12, 0.95, "v2")])
    assert "<title>cand (v1): drift 0.0500</title>" in html
    assert "0.50</text>" in html

    # End-to-end CLI 'history check' exits 0 when stable and 1 when regressed.
    ledger = tmp_path / "trend.jsonl"
    assert main([
        "history", "record", "--ledger", str(ledger),
        "-i", str(EXAMPLES / "expected-pass.json"), "--tag", "v1",
    ]) == 0
    assert main([
        "history", "record", "--ledger", str(ledger),
        "-i", str(EXAMPLES / "expected-pass.json"), "--tag", "v2",
    ]) == 0
    capsys.readouterr()

    assert main(["history", "check", "--ledger", str(ledger)]) == 0
    assert "STABLE" in capsys.readouterr().out

    assert main([
        "history", "record", "--ledger", str(ledger),
        "-i", str(EXAMPLES / "expected-drift.json"), "--tag", "v3",
    ]) == 0
    capsys.readouterr()

    assert main(["history", "check", "--ledger", str(ledger), "--format", "json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["regressed"] is True


def test_history_summary_prune_and_input_guards(tmp_path: Path) -> None:
    from llm_drift_bench.report_io import load_engine_report

    ledger = tmp_path / "compact.jsonl"
    pass_rep = load_engine_report(EXAMPLES / "expected-pass.json")
    drift_rep = load_engine_report(EXAMPLES / "expected-drift.json")

    for tag in ("r1", "r2", "r3"):
        record_report(ledger, pass_rep, tag=tag)
    for tag in ("d1", "d2"):
        record_report(ledger, drift_rep, tag=tag)

    entries = load_history(ledger)
    summary = summarize_history(entries)
    assert summary["total_entries"] == 5
    assert summary["candidates"]["demo-candidate-pass"]["pass_rate"] == 1.0
    assert summary["candidates"]["demo-candidate-drift"]["pass_rate"] == 0.0

    # Keep latest 1 per candidate -> 2 kept, 3 removed, preserving exact tags r3 & d2.
    kept, removed = prune_history(ledger, keep_last=1, per_candidate=True)
    assert (kept, removed) == (2, 3)
    after = load_history(ledger)
    assert [e.tag for e in after] == ["r3", "d2"]

    # Global prune keeps only latest 1 overall.
    kept2, removed2 = prune_history(ledger, keep_last=1, per_candidate=False)
    assert (kept2, removed2) == (1, 1)
    assert [e.tag for e in load_history(ledger)] == ["d2"]

    with pytest.raises(ValueError, match="positive integer"):
        prune_history(ledger, keep_last=True)
    with pytest.raises(TypeError, match="DriftBenchmarkReport"):
        record_report(ledger, {"passed": True})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        sparkline([0.1, float("nan")])


def test_history_persistence_windows_and_distribution_tracking(tmp_path: Path) -> None:
    from llm_drift_bench.report_io import load_engine_report

    def _entry(
        drift: float,
        *,
        tvd: float | None = None,
        w1: float | None = None,
    ) -> HistoryEntry:
        return HistoryEntry(
            recorded_at="2026-11-25T00:00:00Z",
            baseline_model="base",
            candidate_model="cand",
            passed=True,
            total_samples=20,
            drifted_samples=0,
            mean_composite_drift=drift,
            max_composite_drift=drift,
            candidate_json_conformity_rate=1.0,
            max_distribution_tvd=tvd,
            max_distribution_wasserstein=w1,
        )

    # One weird batch is weather: 1 spike with persist_windows=3 does NOT alert!
    weather_runs = [
        _entry(0.04, tvd=0.02, w1=0.02),
        _entry(0.05, tvd=0.03, w1=0.02),
        _entry(0.04, tvd=0.02, w1=0.03),
        _entry(0.05, tvd=0.03, w1=0.02),
        _entry(0.04, tvd=0.02, w1=0.02),
        _entry(0.20, tvd=0.35, w1=0.30),
    ]
    transient = detect_history_regression(
        weather_runs,
        window=3,
        max_drift_increase=0.05,
        persist_windows=3,
        max_tvd_increase=0.10,
        max_wasserstein_increase=0.10,
    )
    assert transient.regressed is False
    assert transient.persistent_runs == 1
    assert "Transient shift (1/3 consecutive runs)" in transient.reason

    # Sustained shift across 3 consecutive windows IS climate: alerts cleanly!
    climate_runs = [
        _entry(0.04, tvd=0.02, w1=0.02),
        _entry(0.05, tvd=0.03, w1=0.02),
        _entry(0.04, tvd=0.02, w1=0.03),
        _entry(0.18, tvd=0.25, w1=0.22),
        _entry(0.19, tvd=0.28, w1=0.25),
        _entry(0.20, tvd=0.30, w1=0.28),
    ]
    sustained = detect_history_regression(
        climate_runs,
        window=3,
        max_drift_increase=0.05,
        persist_windows=3,
        max_tvd_increase=0.10,
        max_wasserstein_increase=0.10,
    )
    assert sustained.regressed is True
    assert sustained.persistent_runs == 3
    assert "3 consecutive runs" in sustained.reason
    assert "Distribution max TVD increased" in sustained.reason
    assert "Distribution max norm W1 increased" in sustained.reason

    # Round-trip distribution fields through JSONL ledger
    ledger = tmp_path / "dist_ledger.jsonl"
    rep = load_engine_report(EXAMPLES / "expected-pass.json")
    record_report(ledger, rep, max_distribution_tvd=0.12, max_distribution_wasserstein=0.08)
    loaded = load_history(ledger)
    assert loaded[0].max_distribution_tvd == pytest.approx(0.12)
    assert loaded[0].max_distribution_wasserstein == pytest.approx(0.08)

    with pytest.raises(ValueError, match="persist_windows"):
        detect_history_regression(climate_runs, persist_windows=0)
    with pytest.raises(ValueError, match="at least 4"):
        detect_history_regression(climate_runs[:3], persist_windows=3)



