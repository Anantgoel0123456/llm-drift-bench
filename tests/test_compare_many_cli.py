"""End-to-end checks for comparisons from raw captures."""

import json
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


def test_many_uses_saved_report_json_contract(capsys) -> None:
    assert main([*ARGS, "--format", "json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert [row["candidate"] for row in data["candidates"]] == ["Q8", "Q4"]


def test_informational_mode_preserves_failed_verdict(capsys) -> None:
    assert main([*ARGS, "--no-fail-on-drift"]) == 0
    assert "'Q4' FAIL" in capsys.readouterr().out


def test_late_missing_capture_produces_no_partial_output(tmp_path, capsys) -> None:
    assert main([*ARGS, "-c", "missing-run", str(tmp_path / "missing.json")]) == 2
    output = capsys.readouterr()
    assert not output.out and "missing-run" in output.err and "missing.json" in output.err


def test_cli_checks_labels_before_opening_captures(capsys) -> None:
    assert main(["compare-many", "-b", "missing", "-c", "A", "a", "-c", "A", "b"]) == 2
    assert "unique" in capsys.readouterr().err


def test_against_uses_candidate_alias_for_direction(capsys) -> None:
    assert main([*ARGS, "--against", "Q8", "--format", "json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["against"] == "Q8"
    assert {row["state"] for row in data["changes"]} == {"regressed"}


def test_filter_direction(capsys) -> None:
    assert main([*ARGS, "--against", "Q8", "--change-type", "recovered"]) == 1
    assert "relative to 'Q8': 0" in capsys.readouterr().out
    assert main([*ARGS, "--change-type", "recovered"]) == 2


def test_category_filter_preserves_candidate_verdict(capsys) -> None:
    assert main([*ARGS, "-s", str(EXAMPLES / "extraction-suite.json"), "--against", "Q8",
                 "--change-category", "classification", "--format", "json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert [item["prompt_id"] for item in data["changes"]] == ["ticket-01"]
    assert len(data["candidates"]) == 2


def test_change_limit_records_how_many_were_hidden(capsys) -> None:
    assert main([*ARGS, "--against", "Q8", "--change-limit", "1", "--format", "json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert len(data["changes"]) == 1 and data["changes_omitted"] == 1


def test_regression_gate_ignores_display_filters() -> None:
    flags = ["--no-fail-on-drift", "--fail-on-regression"]
    assert main([*ARGS, *flags, "--against", "Q8", "--change-type", "recovered"]) == 1
    assert main([*ARGS, *flags, "--against", "Q4"]) == 0
    assert main([*ARGS, *flags]) == 2


def test_batch_matches_separately_saved_rankings(capsys) -> None:
    assert main([*ARGS, "-s", str(EXAMPLES / "extraction-suite.json"), "--format", "json"]) == 1
    fresh = json.loads(capsys.readouterr().out)["candidates"]
    assert main(["compare-reports", str(EXAMPLES / "expected-pass.json"),
                 str(EXAMPLES / "expected-drift.json"), "--format", "json"]) == 0
    saved = json.loads(capsys.readouterr().out)["candidates"]
    for current, recorded in zip(fresh, saved, strict=True):
        for field in ("passed", "samples", "drifted", "worst_prompt_id", "conformity", "p95_ms"):
            assert current[field] == recorded[field]
        assert round(current["mean_drift"], 4) == recorded["mean_drift"]
