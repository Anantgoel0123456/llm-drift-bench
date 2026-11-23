"""Tests for the Prompt x Candidate drift heatmap matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_drift_bench.matrix import (
    build_drift_matrix,
    export_drift_matrix_html,
    format_drift_matrix_text,
)
from llm_drift_bench.report_io import load_engine_report

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_build_format_and_export_drift_matrix(tmp_path: Path) -> None:
    pass_rep = load_engine_report(EXAMPLES / "expected-pass.json")
    drift_rep = load_engine_report(EXAMPLES / "expected-drift.json")

    matrix = build_drift_matrix([drift_rep, pass_rep], sort_prompts_by="max_drift")
    assert matrix.candidates == ("demo-candidate-pass", "demo-candidate-drift")
    # ticket-01 has higher max drift (0.9597) than invoice-01 (0.3791)
    assert [r.prompt_id for r in matrix.rows] == ["ticket-01", "invoice-01"]
    assert matrix.to_dict()["rows"][0]["drifted"]["demo-candidate-drift"] is True

    text = format_drift_matrix_text(matrix)
    assert "Drift Matrix" in text
    assert "0.9597*" in text

    out_html = tmp_path / "matrix.html"
    html = export_drift_matrix_html(matrix, output_path=out_html)
    assert out_html.read_text(encoding="utf-8") == html
    assert "<table" in html and "<script" not in html.lower()
    assert "ticket-01" in html

    by_id = build_drift_matrix([drift_rep, pass_rep], sort_prompts_by="prompt_id")
    assert [r.prompt_id for r in by_id.rows] == ["invoice-01", "ticket-01"]

    with pytest.raises(ValueError, match="sort_prompts_by"):
        build_drift_matrix([drift_rep, pass_rep], sort_prompts_by="bad")  # type: ignore[arg-type]
