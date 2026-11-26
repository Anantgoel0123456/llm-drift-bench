"""Targeted edge-case and format-branch tests for llm_drift_bench.cli."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench import DriftBenchmarkEngine, ModelSample, evaluate_suite
from llm_drift_bench.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_run_report_validate_diff_capture_and_ai_devices_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # run with --config + --long-text-mode + --format sarif
    rc = main(
        [
            "run",
            "-b",
            str(EXAMPLES / "baseline.jsonl"),
            "-c",
            str(EXAMPLES / "candidate-pass.jsonl"),
            "-s",
            str(EXAMPLES / "extraction-suite.json"),
            "--config",
            str(EXAMPLES / "drift.toml"),
            "--long-text-mode",
            "skip_levenshtein",
            "--format",
            "sarif",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["version"] == "2.1.0"

    # --github-summary without an existing GITHUB_STEP_SUMMARY file fails with exit 2
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert (
        main(
            [
                "run",
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-pass.jsonl"),
                "--github-summary",
            ]
        )
        == 2
    )
    assert "GITHUB_STEP_SUMMARY" in capsys.readouterr().err

    # report missing file and SARIF file output
    assert main(["report", "-i", str(tmp_path / "missing.json")]) == 2
    capsys.readouterr()

    sarif_out = tmp_path / "out.sarif"
    assert (
        main(
            [
                "report",
                "-i",
                str(EXAMPLES / "expected-drift.json"),
                "--format",
                "sarif",
                "-o",
                str(sarif_out),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert json.loads(sarif_out.read_text(encoding="utf-8"))["version"] == "2.1.0"

    # Legacy report rejected by junit / csv / sarif report formats
    samples = [ModelSample(prompt_id="p1", output="{}")]
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(evaluate_suite([], samples, samples).model_dump_json(), encoding="utf-8")
    for fmt in ("junit", "csv", "sarif"):
        assert main(["report", "-i", str(legacy_path), "--format", fmt]) == 2
        capsys.readouterr()

    # validate unknown category filter
    assert (
        main(
            [
                "validate",
                "-i",
                str(EXAMPLES / "baseline.jsonl"),
                "-p",
                str(EXAMPLES / "extraction-suite.json"),
                "--category",
                "nonexistent",
            ]
        )
        == 2
    )
    assert "Unknown category" in capsys.readouterr().err

    # diff --limit omission notes in both markdown and text modes
    for fmt in ("markdown", "text"):
        assert (
            main(
                [
                    "diff",
                    "-b",
                    str(EXAMPLES / "baseline.jsonl"),
                    "-c",
                    str(EXAMPLES / "candidate-drift.jsonl"),
                    "--limit",
                    "1",
                    "--format",
                    fmt,
                ]
            )
            == 0
        )
        assert "omitted" in capsys.readouterr().out

    # capture invalid --runs range
    assert (
        main(
            [
                "capture",
                "-s",
                str(EXAMPLES / "extraction-suite.json"),
                "--endpoint",
                "http://localhost:11434/v1",
                "--model",
                "m1",
                "--runs",
                "0",
                "-o",
                str(tmp_path / "cap.jsonl"),
            ]
        )
        == 2
    )
    capsys.readouterr()

    # ai-devices command
    monkeypatch.setattr("llm_drift_bench.local_advisor.available_devices", lambda: ["CPU"])
    assert main(["ai-devices"]) == 0
    assert "CPU" in capsys.readouterr().out


def test_compare_many_compare_reports_and_diagnose_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # compare-many with fewer than 2 candidates fails cleanly
    assert (
        main(
            [
                "compare-many",
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                "only",
                str(EXAMPLES / "candidate-pass.jsonl"),
            ]
        )
        == 2
    )
    capsys.readouterr()

    # compare-reports argument guards
    assert (
        main(
            [
                "compare-reports",
                str(EXAMPLES / "expected-pass.json"),
                str(EXAMPLES / "expected-drift.json"),
                "--change-category",
                "extraction",
            ]
        )
        == 2
    )
    capsys.readouterr()
    assert (
        main(
            [
                "compare-reports",
                str(EXAMPLES / "expected-pass.json"),
                str(EXAMPLES / "expected-drift.json"),
                "--against",
                "demo-candidate-pass",
                "--change-limit",
                "0",
            ]
        )
        == 2
    )
    capsys.readouterr()

    # compare-reports --format json combined with --html-output and --markdown-output
    html_out = tmp_path / "comp.html"
    md_out = tmp_path / "comp.md"
    rc = main(
        [
            "compare-reports",
            str(EXAMPLES / "expected-pass.json"),
            str(EXAMPLES / "expected-drift.json"),
            "--format",
            "json",
            "--html-output",
            str(html_out),
            "--markdown-output",
            str(md_out),
        ]
    )
    assert rc == 0
    capsys.readouterr()
    assert "<html" in html_out.read_text(encoding="utf-8").lower()
    assert "#" in md_out.read_text(encoding="utf-8")

    # compare-reports text mode with omitted changes
    assert (
        main(
            [
                "compare-reports",
                str(EXAMPLES / "expected-pass.json"),
                str(EXAMPLES / "expected-drift.json"),
                "--against",
                "demo-candidate-pass",
                "--change-limit",
                "1",
            ]
        )
        == 0
    )
    assert "additional changes omitted" in capsys.readouterr().out

    # compare-reports text mode with zero hotspots, zero rule issues, zero numeric drift,
    # and a positive latency slowdown
    base_timed = [
        ModelSample(prompt_id="p1", model_name="base", output='{"a": "x"}', latency_ms=10)
    ]
    c1_timed = [ModelSample(prompt_id="p1", model_name="c1", output='{"a": "x"}', latency_ms=10)]
    c2_timed = [ModelSample(prompt_id="p1", model_name="c2", output='{"a": "x"}', latency_ms=11)]
    engine = DriftBenchmarkEngine()
    p1 = tmp_path / "p1.json"
    p2 = tmp_path / "p2.json"
    p1.write_text(engine.evaluate(base_timed, c1_timed).to_json(), encoding="utf-8")
    p2.write_text(engine.evaluate(base_timed, c2_timed).to_json(), encoding="utf-8")
    assert (
        main(
            [
                "compare-reports",
                str(p1),
                str(p2),
                "--hotspots",
                "5",
                "--rule-summary",
                "--slowdowns",
                "5",
                "--numeric-changes",
                "5",
            ]
        )
        == 0
    )
    text_out = capsys.readouterr().out
    assert "No individual prompts drifted." in text_out
    assert "No recorded candidate validation issues." in text_out
    assert "Slowdown" in text_out
    assert "No nonzero numeric drift measurements." in text_out

    # diagnose argument validation and text output (findings, omitted, AI advice, AI error)
    assert main(["diagnose", "-i", str(EXAMPLES / "expected-drift.json"), "--limit", "0"]) == 2
    capsys.readouterr()
    assert main(["diagnose", "-i", str(EXAMPLES / "expected-drift.json"), "--device", "CPU"]) == 2
    capsys.readouterr()

    monkeypatch.setattr(
        "llm_drift_bench.local_advisor.generate_advice",
        lambda findings, model, device: "Inspect schema mismatch on ticket-01",
    )
    rc_diag = main(
        [
            "diagnose",
            "-i",
            str(EXAMPLES / "expected-drift.json"),
            "--limit",
            "1",
            "--ai-model",
            str(tmp_path),
        ]
    )
    assert rc_diag == 0
    diag_out = capsys.readouterr().out
    assert "Try:" in diag_out
    assert "omitted" in diag_out
    assert "Local AI suggestion: Inspect schema mismatch" in diag_out

    monkeypatch.setattr(
        "llm_drift_bench.local_advisor.generate_advice",
        lambda findings, model, device: (_ for _ in ()).throw(RuntimeError("openvino missing")),
    )
    assert (
        main(
            [
                "diagnose",
                "-i",
                str(EXAMPLES / "expected-drift.json"),
                "--ai-model",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "Local AI unavailable: openvino missing" in capsys.readouterr().err


def test_stats_and_history_cli_edges(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # stats text output with no flaky prompts and --compare-against significance output
    assert (
        main(
            [
                "stats",
                str(EXAMPLES / "expected-pass.json"),
                str(EXAMPLES / "expected-pass.json"),
                "--compare-against",
                str(EXAMPLES / "expected-drift.json"),
                "--resamples",
                "50",
            ]
        )
        == 0
    )
    stats_out = capsys.readouterr().out
    assert "Flaky prompts: none" in stats_out
    assert "Paired permutation test" in stats_out
    assert "McNemar conformity test" in stats_out

    # stats --compare-against baseline model mismatch -> exit 2
    other_base = DriftBenchmarkEngine().evaluate(
        [ModelSample(prompt_id="invoice-01", model_name="other-base", output="{}")],
        [ModelSample(prompt_id="invoice-01", model_name="c", output="{}")],
    )
    other_path = tmp_path / "other.json"
    other_path.write_text(other_base.to_json(), encoding="utf-8")
    assert (
        main(
            [
                "stats",
                str(EXAMPLES / "expected-pass.json"),
                "--compare-against",
                str(other_path),
            ]
        )
        == 2
    )
    capsys.readouterr()

    # history missing action -> exit 2
    assert main(["history"]) == 2
    capsys.readouterr()

    # history record + show/check/summary/prune filter and text paths
    ledger = tmp_path / "ledger.jsonl"
    for _ in range(2):
        assert (
            main(
                [
                    "history",
                    "record",
                    "--ledger",
                    str(ledger),
                    "-i",
                    str(EXAMPLES / "expected-pass.json"),
                    "--tag",
                    "v1",
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert (
        main(
            [
                "history",
                "show",
                "--ledger",
                str(ledger),
                "--candidate",
                "demo-candidate-pass",
                "--limit",
                "0",
            ]
        )
        == 2
    )
    capsys.readouterr()

    assert (
        main(
            [
                "history",
                "check",
                "--ledger",
                str(ledger),
                "--candidate",
                "demo-candidate-pass",
                "--tag",
                "v1",
            ]
        )
        == 0
    )
    assert "History Trend Check" in capsys.readouterr().out

    assert (
        main(
            [
                "history",
                "summary",
                "--ledger",
                str(ledger),
                "--candidate",
                "demo-candidate-pass",
                "--tag",
                "v1",
            ]
        )
        == 0
    )
    assert "Drift History Summary" in capsys.readouterr().out

    assert main(["history", "prune", "--ledger", str(ledger), "--keep-last", "1"]) == 0
    assert "Pruned" in capsys.readouterr().out
