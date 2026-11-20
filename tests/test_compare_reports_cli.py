"""End-to-end CLI behavior using published example captures."""

import json
from pathlib import Path

from llm_drift_bench.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ARGS = ["compare-reports", str(EXAMPLES / "expected-drift.json"),
        str(EXAMPLES / "expected-pass.json")]


def test_rank_saved_reports_in_text(capsys) -> None:
    assert main(ARGS) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "demo-reference" in lines[0]
    assert "PASS" in lines[1] and "P95 n/a" in lines[1]
    assert "FAIL" in lines[2]


def test_comparison_json_is_sorted_and_keeps_missing_latency(capsys) -> None:
    assert main([*ARGS, "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["baseline"] == "demo-reference"
    passing, failing = data["candidates"]
    assert passing["mean_drift"] < failing["mean_drift"]
    assert passing["p95_ms"] is None


def test_comparison_ci_exit_codes() -> None:
    assert main([*ARGS, "--fail-on-drift"]) == 1
    assert main([*ARGS, "--fail-on-drift", "--format", "json"]) == 1
    assert main(ARGS[:2]) == 2
