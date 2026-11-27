"""Tests for OASIS SARIF 2.1.0 export via Python API and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main
from llm_drift_bench.report_io import load_engine_report
from llm_drift_bench.sarif import export_sarif_report

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_export_sarif_report_passing_and_drifted(tmp_path: Path) -> None:
    passed = load_engine_report(EXAMPLES / "expected-pass.json")
    pass_sarif = json.loads(export_sarif_report(passed))
    assert pass_sarif["version"] == "2.1.0"
    assert pass_sarif["runs"][0]["tool"]["driver"]["name"] == "llm-drift-bench"
    assert pass_sarif["runs"][0]["results"] == []
    assert pass_sarif["runs"][0]["properties"]["passed"] is True

    drifted = load_engine_report(EXAMPLES / "expected-drift.json")
    out_file = tmp_path / "out" / "results.sarif"
    raw = export_sarif_report(drifted, output_path=out_file, artifact_uri="extraction-suite.json")
    assert out_file.read_text(encoding="utf-8") == raw

    doc = json.loads(raw)
    results = doc["runs"][0]["results"]
    rule_ids = {item["ruleId"] for item in results}
    assert {"DRIFT001", "DRIFT002", "DRIFT004"}.issubset(rule_ids)
    assert doc["runs"][0]["invocations"] == [{"executionSuccessful": True}]
    hashes = [item["partialFingerprints"]["primaryLocationLineHash"] for item in results]
    assert len(hashes) == len(set(hashes))
    assert all(len(h) == 32 for h in hashes)
    # Deterministic across repeated exports.
    assert export_sarif_report(drifted, artifact_uri="extraction-suite.json") == raw
    # Never leaks raw response body text in SARIF annotations.
    assert "INV-2026" not in raw

    with pytest.raises(ValueError, match="artifact_uri"):
        export_sarif_report(drifted, artifact_uri="   ")
    with pytest.raises(ValueError, match="canonical engine report"):
        export_sarif_report({"passed": True})  # type: ignore[arg-type]

    # Pure string-field composite drift (valid JSON, no numeric drift) emits DRIFT003
    from llm_drift_bench.engine import DriftBenchmarkEngine
    from llm_drift_bench.models import ModelSample

    lex_rep = DriftBenchmarkEngine().evaluate(
        [ModelSample(prompt_id="p1", output='{"summary": "alpha beta gamma delta"}')],
        [ModelSample(prompt_id="p1", output='{"summary": "completely unrelated prose output"}')],
    )
    lex_doc = json.loads(export_sarif_report(lex_rep))
    assert any(r["ruleId"] == "DRIFT003" for r in lex_doc["runs"][0]["results"])


def test_cli_run_and_report_sarif_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sarif_file = tmp_path / "run.sarif"
    rc = main([
        "run",
        "-b", str(EXAMPLES / "baseline.jsonl"),
        "-c", str(EXAMPLES / "candidate-drift.jsonl"),
        "-s", str(EXAMPLES / "extraction-suite.json"),
        "--sarif-output", str(sarif_file),
    ])
    assert rc == 1
    capsys.readouterr()
    doc = json.loads(sarif_file.read_text(encoding="utf-8"))
    assert doc["version"] == "2.1.0"
    assert len(doc["runs"][0]["results"]) >= 3

    rc_rep = main([
        "report",
        "-i", str(EXAMPLES / "expected-pass.json"),
        "--format", "sarif",
    ])
    assert rc_rep == 0
    rep_doc = json.loads(capsys.readouterr().out)
    assert rep_doc["runs"][0]["results"] == []

    # Colliding --output and --sarif-output rejected cleanly.
    out_same = tmp_path / "same.json"
    assert main([
        "run",
        "-b", str(EXAMPLES / "baseline.jsonl"),
        "-c", str(EXAMPLES / "candidate-pass.jsonl"),
        "-o", str(out_same),
        "--sarif-output", str(out_same),
    ]) == 2


def test_export_conversation_sarif_report_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_drift_bench.conversation import (
        ConversationRecord,
        ConversationTurn,
        analyze_conversation_drift,
        load_conversations,
    )
    from llm_drift_bench.sarif import export_conversation_sarif_report

    base = load_conversations(EXAMPLES / "conversations-baseline.jsonl")
    drift = load_conversations(EXAMPLES / "conversations-candidate-drift.jsonl")

    pass_sarif = json.loads(
        export_conversation_sarif_report(analyze_conversation_drift(base, base))
    )
    assert pass_sarif["runs"][0]["results"] == []

    # Cover CONV001 (refusal flip), CONV002 (non-refusal drift), and CONV003 (suite gate)
    extra_b = [
        *base,
        ConversationRecord("c-lex", (ConversationTurn("assistant", "alpha beta gamma"),)),
    ]
    extra_c = [
        *drift,
        ConversationRecord("c-lex", (ConversationTurn("assistant", "totally different words"),)),
    ]
    sarif_file = tmp_path / "conv.sarif"
    raw = export_conversation_sarif_report(
        analyze_conversation_drift(extra_b, extra_c),
        output_path=sarif_file,
        artifact_uri="conversations-candidate-drift.jsonl",
    )
    assert sarif_file.read_text(encoding="utf-8") == raw
    doc = json.loads(raw)
    rule_ids = {r["ruleId"] for r in doc["runs"][0]["results"]}
    assert {"CONV001", "CONV002", "CONV003"} <= rule_ids

    with pytest.raises(ValueError, match="ConversationDriftReport"):
        export_conversation_sarif_report({})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="artifact_uri"):
        export_conversation_sarif_report(analyze_conversation_drift(base, base), artifact_uri=" ")

    assert (
        main(
            [
                "check-conversations",
                "-b",
                str(EXAMPLES / "conversations-baseline.jsonl"),
                "-c",
                str(EXAMPLES / "conversations-candidate-drift.jsonl"),
                "--format",
                "sarif",
            ]
        )
        == 1
    )
    cli_doc = json.loads(capsys.readouterr().out)
    assert cli_doc["version"] == "2.1.0"
    assert any(r["ruleId"] == "CONV001" for r in cli_doc["runs"][0]["results"])


def test_export_tool_trace_sarif_report_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_drift_bench.sarif import export_tool_trace_sarif_report
    from llm_drift_bench.tool_events import ToolEvent, compare_tool_traces

    base = [ToolEvent(0, "read", "src/a.py@1", start=0, end=10)]
    cand = [
        ToolEvent(0, "read", "src/a.py@1", start=0, end=10),
        ToolEvent(1, "edit", "src/secret.py@1", success=False),
    ]
    pass_doc = json.loads(export_tool_trace_sarif_report(compare_tool_traces(base, base)))
    assert pass_doc["runs"][0]["results"] == []

    fail_diff = compare_tool_traces(
        base, cand, max_failed_increase=0, fail_on_new_edit_resources=True
    )
    sarif_file = tmp_path / "trace.sarif"
    raw = export_tool_trace_sarif_report(
        fail_diff, output_path=sarif_file, artifact_uri="cand.jsonl"
    )
    assert sarif_file.read_text(encoding="utf-8") == raw
    doc = json.loads(raw)
    rule_ids = {r["ruleId"] for r in doc["runs"][0]["results"]}
    assert rule_ids == {"TOOL001", "TOOL002"}

    with pytest.raises(ValueError, match="ToolTraceDiff"):
        export_tool_trace_sarif_report({})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="artifact_uri"):
        export_tool_trace_sarif_report(fail_diff, artifact_uri=" ")

    b_file = tmp_path / "b.jsonl"
    c_file = tmp_path / "c.jsonl"
    b_file.write_text(
        '{"kind": "read", "resource": "src/a.py@1", "start": 0, "end": 10}\n', encoding="utf-8"
    )
    c_file.write_text(
        '{"kind": "read", "resource": "src/a.py@1", "start": 0, "end": 10}\n'
        '{"kind": "edit", "resource": "src/secret.py@1"}\n',
        encoding="utf-8",
    )
    assert (
        main(
            [
                "diff-tools",
                "-b",
                str(b_file),
                "-c",
                str(c_file),
                "--fail-on-new-edit-resources",
                "--format",
                "sarif",
            ]
        )
        == 1
    )
    cli_doc = json.loads(capsys.readouterr().out)
    assert cli_doc["version"] == "2.1.0"
    assert any(r["ruleId"] == "TOOL001" for r in cli_doc["runs"][0]["results"])

