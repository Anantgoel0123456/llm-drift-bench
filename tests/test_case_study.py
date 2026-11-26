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


def test_multiturn_conversation_examples_doctor_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    examples_dir = CASE_DIR.parent
    base_conv = str(examples_dir / "conversations-baseline.jsonl")
    drift_conv = str(examples_dir / "conversations-candidate-drift.jsonl")

    assert (
        main(
            [
                "doctor",
                "--baseline-conversations",
                base_conv,
                "--candidate-conversations",
                drift_conv,
                "--format",
                "json",
            ]
        )
        == 0
    )
    doc_payload = json.loads(capsys.readouterr().out)
    checks = {c["name"]: c for c in doc_payload["checks"]}
    assert checks["conversation_alignment"]["status"] == "OK"

    assert (
        main(
            [
                "check-conversations",
                "-b",
                base_conv,
                "-c",
                base_conv,
                "--format",
                "json",
            ]
        )
        == 0
    )
    pass_payload = json.loads(capsys.readouterr().out)
    assert pass_payload["passed"] is True
    assert pass_payload["mean_trajectory_drift"] == 0.0

    assert (
        main(
            [
                "check-conversations",
                "-b",
                base_conv,
                "-c",
                drift_conv,
                "--format",
                "json",
            ]
        )
        == 1
    )
    drift_payload = json.loads(capsys.readouterr().out)
    assert drift_payload["passed"] is False
    assert drift_payload["total_refusal_flip_turns"] == 1
    by_id = {c["conversation_id"]: c for c in drift_payload["conversations"]}
    assert by_id["conv-support-01"]["first_divergence_turn"] == 2
    assert by_id["conv-support-01"]["escalation_slope"] > 0.0
    assert by_id["conv-triage-02"]["first_divergence_turn"] is None

    html_out = tmp_path / "conversations.html"
    assert (
        main(
            [
                "check-conversations",
                "-b",
                base_conv,
                "-c",
                drift_conv,
                "--format",
                "html",
                "-o",
                str(html_out),
            ]
        )
        == 1
    )
    capsys.readouterr()
    html_text = html_out.read_text(encoding="utf-8")
    assert "<script" not in html_text.lower()
    assert "conv-support-01" in html_text

