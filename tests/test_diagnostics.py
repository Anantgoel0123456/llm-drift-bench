"""Regression checks for offline diagnosis of the shipped example reports."""

from pathlib import Path

import pytest

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


def test_diagnose_report_gives_specific_advice_for_bounds_pattern_and_const() -> None:
    from llm_drift_bench.diagnostics import diagnose_report
    from llm_drift_bench.engine import DriftBenchmarkEngine
    from llm_drift_bench.models import ModelSample, PromptCase

    suite = [
        PromptCase(
            prompt_id="p1",
            prompt="Extract",
            schema={
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0},
                    "code": {"type": "string", "pattern": "^INV-[0-9]+$"},
                    "tags": {"type": "array", "minItems": 1},
                    "version": {"const": 2},
                },
            },
        )
    ]
    base = [ModelSample("p1", '{"score": 1, "code": "INV-1", "tags": ["a"], "version": 2}')]
    cand = [ModelSample("p1", '{"score": -5, "code": "BAD", "tags": [], "version": 1}')]
    report = DriftBenchmarkEngine().evaluate(base, cand, suite)
    findings = {f.evidence: f.suggestion for f in diagnose_report(report) if f.code == "schema"}
    assert "lower-bound" in findings["$.score: minimum"]
    assert "regex pattern" in findings["$.code: pattern"]
    assert "empty or truncated list" in findings["$.tags: minItems"]
    assert "constant value" in findings["$.version: const"]

    # Schema-clean JSON prompt that drifted purely on text content emits generic "drift" finding
    prose_rep = DriftBenchmarkEngine().evaluate(
        [ModelSample("p_prose", '{"msg": "alpha beta gamma"}', latency_ms=10.0)],
        [ModelSample("p_prose", '{"msg": "totally unrelated answer text"}', latency_ms=10.0)],
    )
    assert any(f.code == "drift" for f in diagnose_report(prose_rep))


def test_diagnose_conversation_report_root_causes() -> None:
    from llm_drift_bench.conversation import (
        ConversationRecord,
        ConversationTurn,
        analyze_conversation_drift,
    )
    from llm_drift_bench.diagnostics import diagnose_conversation_report

    def _c(cid: str, replies: list[str]) -> ConversationRecord:
        turns = []
        for idx, reply in enumerate(replies, start=1):
            turns.append(ConversationTurn("user", f"u{idx}"))
            turns.append(ConversationTurn("assistant", reply))
        return ConversationRecord(cid, tuple(turns))

    base = [
        _c("c_snowball", ['{"step": 1}', '{"step": 2}', '{"step": 3}']),
        _c("c_turn_only", ["hello world one two three"]),
        _c("c_ok", ["same answer"]),
    ]
    cand = [
        _c("c_snowball", ['{"step": 1}', "I'm sorry, I cannot assist with that."]),
        _c("c_turn_only", ["completely different wording here"]),
        _c("c_ok", ["same answer"]),
    ]
    rep = analyze_conversation_drift(base, cand, max_turn_count_delta=0)
    findings = diagnose_conversation_report(rep)
    by_cid: dict[str | None, set[str]] = {}
    for item in findings:
        by_cid.setdefault(item.prompt_id, set()).add(item.code)

    assert {"refusal_flip", "json_drop", "turn_count_mismatch", "snowball_escalation"} <= by_cid[
        "c_snowball"
    ]
    assert by_cid["c_turn_only"] == {"turn_drift"}
    assert "c_ok" not in by_cid
    assert "suite" in by_cid[None]


def test_format_findings_text_and_check_conversations_diagnose_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json
    from pathlib import Path

    from llm_drift_bench.cli import main
    from llm_drift_bench.diagnostics import Finding, format_findings_text

    assert "no actionable findings" in format_findings_text([])
    formatted = format_findings_text([
        Finding("refusal_flip", "conv-1", "Refused at turn 2.", "Check framing."),
        Finding("suite", None, "Gate failed.", "Review gates."),
    ])
    assert "[refusal_flip] (conv-1)" in formatted
    assert "[suite] (suite)" in formatted

    examples = Path(__file__).resolve().parents[1] / "examples"
    assert (
        main(
            [
                "check-conversations",
                "-b",
                str(examples / "conversations-baseline.jsonl"),
                "-c",
                str(examples / "conversations-candidate-drift.jsonl"),
                "--diagnose",
            ]
        )
        == 1
    )
    text_out = capsys.readouterr().out
    assert "Diagnostics (" in text_out and "[refusal_flip]" in text_out

    assert (
        main(
            [
                "check-conversations",
                "-b",
                str(examples / "conversations-baseline.jsonl"),
                "-c",
                str(examples / "conversations-candidate-drift.jsonl"),
                "--diagnose",
                "--format",
                "json",
            ]
        )
        == 1
    )
    doc = json.loads(capsys.readouterr().out)
    assert any(item["code"] == "refusal_flip" for item in doc["diagnostics"])


