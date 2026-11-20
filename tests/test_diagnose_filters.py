"""Filter diagnostics without modifying the saved benchmark verdict."""

import json
from pathlib import Path

from llm_drift_bench.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ARGS = ["diagnose", "-i", str(EXAMPLES / "expected-drift.json"), "--format", "json"]


def test_focus_on_one_prompt(capsys) -> None:
    assert main([*ARGS, "--prompt-id", "invoice-01"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "FAILED"
    assert [f["code"] for f in data["findings"]] == ["numeric_change"]


def test_unknown_prompt_is_input_error(capsys) -> None:
    assert main([*ARGS, "--prompt-id", "typo"]) == 2
    assert "typo" in capsys.readouterr().err


def test_repeatable_finding_types_intersect_prompt_filter(capsys) -> None:
    assert main([*ARGS, "--kind", "schema", "--kind", "numeric_change",
                 "--prompt-id", "ticket-01"]) == 0
    assert [f["code"] for f in json.loads(capsys.readouterr().out)["findings"]] == ["schema"]


def test_suite_scope_includes_gate_and_missing_measurements(capsys) -> None:
    assert main([*ARGS, "--scope", "suite"]) == 0
    findings = json.loads(capsys.readouterr().out)["findings"]
    assert {f["code"] for f in findings} == {"suite", "missing_latency"}
    assert all(f["prompt_id"] is None for f in findings)


def test_no_matches_is_an_explicit_successful_query(capsys) -> None:
    assert main([*ARGS, "--format", "text", "--prompt-id", "invoice-01", "--kind", "schema"]) == 0
    assert "No matching findings." in capsys.readouterr().out
