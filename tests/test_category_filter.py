"""Tests for repeatable --category filtering on run and validate CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_run_category_filter_subsets_suite_and_captures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main([
        "run",
        "-b", str(EXAMPLES / "baseline.jsonl"),
        "-c", str(EXAMPLES / "candidate-pass.jsonl"),
        "-s", str(EXAMPLES / "extraction-suite.json"),
        "--category", "extraction",
        "--format", "json",
    ])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["total_samples"] == 1
    assert [row["prompt_id"] for row in report["results"]] == ["invoice-01"]

    # Unknown category returns exit 2 with available categories.
    rc_bad = main([
        "run",
        "-b", str(EXAMPLES / "baseline.jsonl"),
        "-c", str(EXAMPLES / "candidate-pass.jsonl"),
        "-s", str(EXAMPLES / "extraction-suite.json"),
        "--category", "nonexistent",
    ])
    assert rc_bad == 2
    assert "Available:" in capsys.readouterr().err

    # --category without --suite returns exit 2.
    rc_nosuite = main([
        "run",
        "-b", str(EXAMPLES / "baseline.jsonl"),
        "-c", str(EXAMPLES / "candidate-pass.jsonl"),
        "--category", "extraction",
    ])
    assert rc_nosuite == 2
    assert "requires --suite" in capsys.readouterr().err


def test_validate_category_filter_subsets_validation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # In candidate-drift.jsonl, invoice-01 (extraction) is schema-valid while ticket-01 fails!
    rc_pass = main([
        "validate",
        "-i", str(EXAMPLES / "candidate-drift.jsonl"),
        "-p", str(EXAMPLES / "extraction-suite.json"),
        "--category", "extraction",
        "--strict",
        "--format", "json",
    ])
    assert rc_pass == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["total_samples"] == 1
    assert summary["passed_samples"] == 1

    rc_fail = main([
        "validate",
        "-i", str(EXAMPLES / "candidate-drift.jsonl"),
        "-p", str(EXAMPLES / "extraction-suite.json"),
        "--category", "classification",
        "--strict",
    ])
    assert rc_fail == 1
    capsys.readouterr()

    rc_nosuite = main([
        "validate",
        "-i", str(EXAMPLES / "candidate-drift.jsonl"),
        "--category", "extraction",
    ])
    assert rc_nosuite == 2
    assert "requires --prompts" in capsys.readouterr().err
