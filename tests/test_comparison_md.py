"""Tests for the GFM multi-candidate comparison leaderboard exporter."""

from __future__ import annotations

from pathlib import Path

from llm_drift_bench._comparison_md import export_comparison_markdown
from llm_drift_bench.report_io import load_engine_report

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_export_comparison_markdown_leaderboard_and_transitions(tmp_path: Path) -> None:
    pass_rep = load_engine_report(EXAMPLES / "expected-pass.json")
    drift_rep = load_engine_report(EXAMPLES / "expected-drift.json")

    out = tmp_path / "reports" / "leaderboard.md"
    md = export_comparison_markdown(
        [drift_rep, pass_rep],
        output_path=out,
        against="demo-candidate-pass",
        sort_by="passed",
    )
    assert out.read_text(encoding="utf-8") == md
    assert "# Candidate Drift Comparison" in md
    assert "| 1 | `demo-candidate-pass` | **PASS**" in md
    assert "| 2 | `demo-candidate-drift` | **FAIL**" in md
    assert "## Prompt transitions vs `demo-candidate-pass`" in md
    assert "**regressed**" in md

    # Without 'against' (lines 24-25) and with identical candidates / zero flips (line 52)
    md_plain = export_comparison_markdown([drift_rep, pass_rep])
    assert "## Prompt transitions" not in md_plain

    from dataclasses import replace

    twin = replace(pass_rep, candidate_model="twin-pass")
    md_no_flips = export_comparison_markdown([pass_rep, twin], against="demo-candidate-pass")
    assert "No prompt verdicts flipped" in md_no_flips
