"""Tests for RFC-4180 CSV report export via Python API and CLI."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from llm_drift_bench.cli import main
from llm_drift_bench.csv_export import CSV_COLUMNS, export_csv_report
from llm_drift_bench.engine import DriftBenchmarkEngine
from llm_drift_bench.models import ModelSample
from llm_drift_bench.report_io import load_engine_report


def test_export_csv_report_columns_values_and_file_write(tmp_path: Path) -> None:
    examples = Path(__file__).resolve().parents[1] / "examples"
    report = load_engine_report(examples / "expected-drift.json")
    out_file = tmp_path / "nested" / "metrics.csv"

    csv_text = export_csv_report(report, output_path=out_file)
    assert out_file.read_text(encoding="utf-8") == csv_text

    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert tuple(rows[0].keys()) == CSV_COLUMNS
    assert len(rows) == 2

    inv = next(r for r in rows if r["prompt_id"] == "invoice-01")
    assert inv["drifted"] == "true"
    assert inv["json_numeric_drift"] == "0.4444"
    assert inv["candidate_passed"] == "true"
    assert inv["latency_delta_ms"] == ""

    ticket = next(r for r in rows if r["prompt_id"] == "ticket-01")
    assert ticket["candidate_passed"] == "false"
    assert "$.priority: required" in ticket["candidate_issues"]


def test_export_csv_report_with_latency_and_plain_text() -> None:
    base = [ModelSample(prompt_id="p1", output="hello world", latency_ms=100.0)]
    cand = [ModelSample(prompt_id="p1", output="hello earth", latency_ms=125.5)]
    report = DriftBenchmarkEngine().evaluate(base, cand)

    rows = list(csv.DictReader(io.StringIO(export_csv_report(report))))
    assert len(rows) == 1
    assert rows[0]["json_structural_similarity"] == ""
    assert rows[0]["latency_delta_ms"] == "25.50"


def test_cli_run_and_report_csv_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = Path(__file__).resolve().parents[1] / "examples"

    rc_run = main([
        "run",
        "-b",
        str(examples / "baseline.jsonl"),
        "-c",
        str(examples / "candidate-pass.jsonl"),
        "-s",
        str(examples / "extraction-suite.json"),
        "--format",
        "csv",
    ])
    assert rc_run == 0
    run_rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(run_rows) == 2
    assert all(r["drifted"] == "false" for r in run_rows)

    csv_out = tmp_path / "drift.csv"
    rc_rep = main([
        "report",
        "-i",
        str(examples / "expected-drift.json"),
        "--format",
        "csv",
        "-o",
        str(csv_out),
    ])
    assert rc_rep == 0
    assert csv_out.is_file()
    assert "invoice-01" in csv_out.read_text(encoding="utf-8")
