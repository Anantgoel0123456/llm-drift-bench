"""Unit and CLI tests for the append-only drift history ledger and trend exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main
from llm_drift_bench.history import (
    HistoryEntry,
    export_history_html,
    format_history_text,
    load_history,
    sparkline,
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
