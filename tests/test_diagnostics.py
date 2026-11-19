"""Regression checks for offline diagnosis of the shipped example reports."""

from pathlib import Path

from llm_drift_bench import diagnose_report
from llm_drift_bench.models import DriftBenchmarkReport
from llm_drift_bench.report_io import load_report

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_failing_example_explains_recorded_evidence() -> None:
    report = load_report(EXAMPLES / "expected-drift.json")
    assert isinstance(report, DriftBenchmarkReport)
    findings = diagnose_report(report)
    assert [(f.prompt_id, f.code) for f in findings if f.prompt_id] == [
        ("ticket-01", "schema"),
        ("invoice-01", "numeric_change"),
    ]
    assert "$.priority" in findings[0].evidence
    assert [f.evidence for f in findings if f.code == "suite"] == list(report.failure_reasons)


def test_passing_example_only_mentions_unmeasured_latency() -> None:
    report = load_report(EXAMPLES / "expected-pass.json")
    assert isinstance(report, DriftBenchmarkReport)
    assert [f.code for f in diagnose_report(report)] == ["missing_latency"]
