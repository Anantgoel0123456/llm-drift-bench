"""Tests for PR comment exports, opt-in HTML JSON diffs, and SVG Pareto comparison dashboards."""

from __future__ import annotations

from pathlib import Path

from llm_drift_bench._visuals import build_json_field_diffs
from llm_drift_bench.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_build_json_field_diffs_detects_added_removed_changed_and_type_changed() -> None:
    base = '{"a": 1, "b": "old", "c": 10, "gone": true}'
    cand = '{"a": 1, "b": "new", "c": "10", "extra": 5}'
    rows = build_json_field_diffs(base, cand)
    by_path = {path: status for path, _, _, status in rows}
    assert by_path == {
        "$.b": "changed",
        "$.c": "type_changed",
        "$.extra": "added",
        "$.gone": "removed",
    }


def test_cli_pr_comment_format_with_previous_deltas(tmp_path: Path, capsys) -> None:
    out_md = tmp_path / "pr_comment.md"
    rc = main(
        [
            "report",
            "-i",
            str(EXAMPLES / "expected-drift.json"),
            "--format",
            "pr-comment",
            "--previous",
            str(EXAMPLES / "expected-pass.json"),
            "-o",
            str(out_md),
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "<!-- llm-drift-bench-pr-comment -->" in text
    assert "**FAILED**" in text
    assert "(+0.6694)" in text
    assert "(-50.0%)" in text
    assert "<details>" in text
    assert out_md.read_text(encoding="utf-8") == text


def test_cli_run_html_with_opt_in_diffs_and_pr_comment(capsys) -> None:
    rc = main(
        [
            "run",
            "-b",
            str(EXAMPLES / "baseline.jsonl"),
            "-c",
            str(EXAMPLES / "candidate-drift.jsonl"),
            "-s",
            str(EXAMPLES / "extraction-suite.json"),
            "--format",
            "html",
            "--include-diffs",
            "--no-fail-on-drift",
        ]
    )
    assert rc == 0
    html = capsys.readouterr().out
    assert "Opt-in response diffs" in html
    assert "$.total" in html
    assert "$.priority" in html
    assert "<script" not in html.lower()

    rc2 = main(
        [
            "run",
            "-b",
            str(EXAMPLES / "baseline.jsonl"),
            "-c",
            str(EXAMPLES / "candidate-pass.jsonl"),
            "-s",
            str(EXAMPLES / "extraction-suite.json"),
            "--format",
            "pr-comment",
        ]
    )
    assert rc2 == 0
    assert "**PASSED**" in capsys.readouterr().out


def test_cli_compare_reports_html_pareto_dashboard(tmp_path: Path) -> None:
    html_out = tmp_path / "dash" / "comparison.html"
    rc = main(
        [
            "compare-reports",
            str(EXAMPLES / "expected-pass.json"),
            str(EXAMPLES / "expected-drift.json"),
            "--html-output",
            str(html_out),
        ]
    )
    assert rc == 0
    content = html_out.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "Pareto View" in content
    assert "demo-candidate-pass" in content
    assert "demo-candidate-drift" in content
    assert "text-anchor='end'" in content
    assert "<script" not in content.lower()


def test_comparison_html_respects_sort_by_and_renders_pareto_frontier(tmp_path: Path) -> None:
    from dataclasses import replace

    from llm_drift_bench.report_io import load_engine_report
    from llm_drift_bench.reporter import export_comparison_html

    pass_accurate = replace(
        load_engine_report(EXAMPLES / "expected-pass.json"),
        candidate_model="pass-accurate",
        mean_composite_drift=0.02,
        candidate_p95_latency_ms=120.0,
    )
    pass_balanced = replace(
        load_engine_report(EXAMPLES / "expected-pass.json"),
        candidate_model="pass-balanced",
        mean_composite_drift=0.08,
        candidate_p95_latency_ms=60.0,
    )
    fail_fast = replace(
        load_engine_report(EXAMPLES / "expected-drift.json"),
        candidate_model="fail-fast",
        mean_composite_drift=0.05,
        candidate_p95_latency_ms=90.0,
    )
    html = export_comparison_html([fail_fast, pass_balanced, pass_accurate], sort_by="passed")
    assert html.index("<td>pass-balanced</td>") < html.index("<td>fail-fast</td>")
    assert "<polyline" in html
    assert "P95 Latency (ms)" in html
