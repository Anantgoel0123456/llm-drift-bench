"""End-to-end checks for comparisons from raw captures."""

from pathlib import Path

from llm_drift_bench.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ARGS = ["compare-many", "-b", str(EXAMPLES / "baseline.jsonl"),
        "-c", "Q8", str(EXAMPLES / "candidate-pass.jsonl"),
        "-c", "Q4", str(EXAMPLES / "candidate-drift.jsonl")]


def test_named_candidates_rank_from_captures(capsys) -> None:
    assert main(ARGS) == 1
    text = capsys.readouterr().out
    assert text.index("'Q8' PASS") < text.index("'Q4' FAIL")


def test_many_checks_candidate_schema_with_one_suite(capsys) -> None:
    assert main([*ARGS, "-s", str(EXAMPLES / "extraction-suite.json")]) == 1
    assert "JSON 50.0%" in capsys.readouterr().out
