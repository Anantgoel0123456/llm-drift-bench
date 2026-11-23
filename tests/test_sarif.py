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
