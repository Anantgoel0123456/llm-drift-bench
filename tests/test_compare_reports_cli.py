"""End-to-end CLI behavior using published example captures."""

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
