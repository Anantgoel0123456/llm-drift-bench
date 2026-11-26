"""Integration tests keeping the offline quantization case study reproducible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main

CASE_DIR = Path(__file__).resolve().parents[1] / "examples" / "case-study"


def test_case_study_doctor_lint_and_q8_vs_q4_verdicts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = str(CASE_DIR / "prompts.json")
    cfg = str(CASE_DIR / "drift.toml")
    base = str(CASE_DIR / "baseline-fp16.jsonl")
    q8 = str(CASE_DIR / "candidate-q8.jsonl")
    q4 = str(CASE_DIR / "candidate-q4.jsonl")

    assert main(["doctor", "--config", cfg, "-s", suite, "-b", base, "-c", q8]) == 0
    _ = capsys.readouterr()

    assert main(["lint-suite", "-s", suite, "--strict"]) == 0
    _ = capsys.readouterr()

    # Q8 passes audit & run under drift.toml policy
    assert (
        main(
            [
                "audit",
                "-s",
                suite,
                "-b",
                base,
                "-c",
                q8,
                "--config",
                cfg,
                "--group-by",
                "metadata:route",
                "--fail-on-widespread",
                "--format",
                "json",
            ]
        )
        == 0
    )
    q8_audit = json.loads(capsys.readouterr().out)
    assert q8_audit["passed"] is True
    assert q8_audit["cost"]["complete"] is True

    # Q4 fails audit with schema + distribution shifts
    assert (
        main(
            [
                "audit",
                "-s",
                suite,
                "-b",
                base,
                "-c",
                q4,
                "--config",
                cfg,
                "--group-by",
                "metadata:route",
                "--shift-threshold",
                "0.15",
                "--format",
                "json",
            ]
        )
        == 1
    )
    q4_audit = json.loads(capsys.readouterr().out)
    assert q4_audit["passed"] is False
    assert q4_audit["segmented_distribution"]["segments"]["support"]["max_tvd"] == pytest.approx(
        1.0
    )

    # compare-many ranks Q8 ahead of Q4
    assert (
        main(
            [
                "compare-many",
                "-b",
                base,
                "-s",
                suite,
                "--config",
                cfg,
                "-c",
                "q8",
                q8,
                "-c",
                "q4",
                q4,
                "--against",
                "q8",
                "--format",
                "json",
                "--no-fail-on-drift",
            ]
        )
        == 0
    )
    cmp_out = json.loads(capsys.readouterr().out)
    assert cmp_out["candidates"][0]["candidate"] == "q8"
    assert cmp_out["candidates"][0]["passed"] is True
    assert cmp_out["candidates"][1]["passed"] is False
