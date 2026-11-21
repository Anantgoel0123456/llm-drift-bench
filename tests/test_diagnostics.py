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


def test_cli_preserves_results_when_model_is_unavailable(tmp_path, capsys) -> None:
    import json

    from llm_drift_bench.cli import main

    status = main(["diagnose", "-i", str(EXAMPLES / "expected-drift.json"), "--format", "json",
                   "--limit", "1", "--ai-model", str(tmp_path / "missing"), "--device", "NPU"])
    result = json.loads(capsys.readouterr().out)
    assert status == 2 and result["verdict"] == "FAILED"
    assert len(result["findings"]) == 1 and result["omitted"] > 0
    assert result["ai_error"] and result["ai_advice"] is None


def test_diagnose_reports_non_drifted_conformity_failures_and_cooccurring_numeric_drift() -> None:
    from llm_drift_bench import DriftBenchmarkEngine, ModelSample, PromptCase

    # 1. Both baseline and candidate return identical invalid JSON against a schema
    suite = [PromptCase(prompt_id="p_bad", prompt="Return JSON", schema={"type": "object"})]
    rep_invalid = DriftBenchmarkEngine().evaluate(
        [ModelSample(prompt_id="p_bad", output="not json", latency_ms=10.0)],
        [ModelSample(prompt_id="p_bad", output="not json", latency_ms=10.0)],
        suite,
    )
    assert not rep_invalid.passed and rep_invalid.drifted_samples == 0
    codes = [(f.prompt_id, f.code) for f in diagnose_report(rep_invalid) if f.prompt_id]
    assert codes == [("p_bad", "invalid_json")]

    # 2. Candidate violates BOTH a schema enum rule AND numeric relative drift
    schema = {
        "type": "object",
        "required": ["total", "status"],
        "properties": {
            "total": {"type": "number"},
            "status": {"type": "string", "enum": ["paid"]},
        },
    }
    rep_both = DriftBenchmarkEngine().evaluate(
        [ModelSample(prompt_id="p_both", output='{"total": 100, "status": "paid"}', latency_ms=10)],
        [ModelSample(prompt_id="p_both", output='{"total": 9999, "status": "bad"}', latency_ms=10)],
        [PromptCase(prompt_id="p_both", prompt="Extract", schema=schema)],
    )
    both_codes = [f.code for f in diagnose_report(rep_both) if f.prompt_id == "p_both"]
    assert "schema" in both_codes and "numeric_change" in both_codes
