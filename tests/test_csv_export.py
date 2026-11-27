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


def test_csv_formula_injection_sanitization_preserves_negative_latency() -> None:
    base = [ModelSample(prompt_id="=SUM(A1:A2)", output="ok", model_name="+base", latency_ms=50.0)]
    cand = [ModelSample(prompt_id="=SUM(A1:A2)", output="ok", model_name="@cand", latency_ms=30.0)]
    report = DriftBenchmarkEngine().evaluate(base, cand)

    safe_rows = list(csv.DictReader(io.StringIO(export_csv_report(report))))
    assert safe_rows[0]["prompt_id"] == "'=SUM(A1:A2)"
    assert safe_rows[0]["baseline_model"] == "'+base"
    assert safe_rows[0]["candidate_model"] == "'@cand"
    # Numeric negative latency deltas remain numeric (never prefixed with apostrophe).
    assert safe_rows[0]["latency_delta_ms"] == "-20.00"

    raw_rows = list(csv.DictReader(io.StringIO(export_csv_report(report, sanitize_formulas=False))))
    assert raw_rows[0]["prompt_id"] == "=SUM(A1:A2)"


def test_export_conversation_csv_report_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_drift_bench.conversation import (
        ConversationRecord,
        ConversationTurn,
        analyze_conversation_drift,
        load_conversations,
    )
    from llm_drift_bench.csv_export import (
        CONVERSATION_CSV_COLUMNS,
        export_conversation_csv_report,
    )

    examples = Path(__file__).resolve().parents[1] / "examples"
    base = load_conversations(examples / "conversations-baseline.jsonl")
    drift = load_conversations(examples / "conversations-candidate-drift.jsonl")

    # Include a formula-like conversation ID and a zero-assistant-turn conversation
    extra_b = [
        *base,
        ConversationRecord("=CMD()", (ConversationTurn("user", "only user"),)),
    ]
    extra_c = [
        *drift,
        ConversationRecord("=CMD()", (ConversationTurn("user", "only user"),)),
    ]
    csv_file = tmp_path / "conv.csv"
    text = export_conversation_csv_report(
        analyze_conversation_drift(extra_b, extra_c),
        output_path=csv_file,
    )
    assert csv_file.read_text(encoding="utf-8") == text
    assert "ORD-1042" not in text
    rows = list(csv.DictReader(io.StringIO(text)))
    assert tuple(rows[0].keys()) == CONVERSATION_CSV_COLUMNS
    assert any(r["conversation_id"] == "'=CMD()" and r["turn_index"] == "" for r in rows)
    assert any(r["refusal_flip"] == "true" for r in rows)

    with pytest.raises(ValueError, match="ConversationDriftReport"):
        export_conversation_csv_report({})  # type: ignore[arg-type]

    assert (
        main(
            [
                "check-conversations",
                "-b",
                str(examples / "conversations-baseline.jsonl"),
                "-c",
                str(examples / "conversations-candidate-drift.jsonl"),
                "--format",
                "csv",
            ]
        )
        == 1
    )
    cli_rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(cli_rows) >= 4
    assert tuple(cli_rows[0].keys()) == CONVERSATION_CSV_COLUMNS

