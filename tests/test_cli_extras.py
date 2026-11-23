"""CLI tests for redact, matrix, lint-suite, subsample, and new flags on existing commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_drift_bench.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_cli_redact_secrets_and_pii(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = tmp_path / "raw.jsonl"
    out_all = tmp_path / "clean_all.jsonl"
    out_sec = tmp_path / "clean_sec.jsonl"
    sample = {
        "prompt_id": "p1",
        "model_name": "m1",
        "output": json.dumps(
            {
                "contact": "alice@example.com",
                "token": "sk-proj-abcdefghijklmnopqrstuvwxyz1234",
            }
        ),
    }
    raw.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    code = main(["redact", "-i", str(raw), "-o", str(out_all), "--format", "json"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["total_replacements"] == 2
    cleaned_all = json.loads(out_all.read_text(encoding="utf-8").strip())
    assert "alice@example.com" not in cleaned_all["output"]
    assert "[REDACTED:OPENAI_KEY]" in cleaned_all["output"]

    code2 = main(["redact", "-i", str(raw), "-o", str(out_sec), "--secrets-only"])
    assert code2 == 0
    assert "Redacted 1 occurrence(s)" in capsys.readouterr().out
    cleaned_sec = json.loads(out_sec.read_text(encoding="utf-8").strip())
    assert "alice@example.com" in cleaned_sec["output"]
    assert "[REDACTED:OPENAI_KEY]" in cleaned_sec["output"]

    # Refuses overwriting input file
    assert main(["redact", "-i", str(raw), "-o", str(raw)]) == 2


def test_cli_matrix_text_json_and_html(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pass_path = str(EXAMPLES / "expected-pass.json")
    drift_path = str(EXAMPLES / "expected-drift.json")

    assert main(["matrix", drift_path, pass_path]) == 0
    txt = capsys.readouterr().out
    assert "Drift Matrix" in txt and "ticket-01" in txt

    assert main(["matrix", drift_path, pass_path, "--sort-prompts", "prompt_id",
                 "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [row["prompt_id"] for row in data["rows"]] == ["invoice-01", "ticket-01"]

    html_out = tmp_path / "heatmap.html"
    assert main(["matrix", drift_path, pass_path, "--format", "html", "-o", str(html_out)]) == 0
    _ = capsys.readouterr()
    html = html_out.read_text(encoding="utf-8")
    assert "<table" in html and "<script" not in html.lower()


def test_cli_lint_suite_normal_and_strict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    warn_suite = tmp_path / "warn_suite.json"
    warn_suite.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "p1",
                    "category": "extraction",
                    "prompt": "Extract total",
                    "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                }
            ]
        ),
        encoding="utf-8",
    )

    assert main(["lint-suite", "-s", str(warn_suite)]) == 0
    assert "permissive_object" in capsys.readouterr().out

    assert main(["lint-suite", "-s", str(warn_suite), "--strict", "--format", "json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["warnings"] == 1 and payload["errors"] == 0

    err_suite = tmp_path / "err_suite.json"
    err_suite.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "p1",
                    "category": "extraction",
                    "prompt": "Extract",
                    "schema": {"type": "object", "anyOf": [{"type": "string"}]},
                }
            ]
        ),
        encoding="utf-8",
    )
    assert main(["lint-suite", "-s", str(err_suite)]) == 1


def test_cli_subsample_and_lockstep_slice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "suite.json"
    cap = tmp_path / "capture.jsonl"
    cases = [
        {"prompt_id": f"ext_{i}", "category": "extraction", "prompt": f"E {i}"}
        for i in range(4)
    ] + [
        {"prompt_id": f"cls_{i}", "category": "classification", "prompt": f"C {i}"}
        for i in range(4)
    ]
    suite.write_text(json.dumps(cases), encoding="utf-8")
    cap.write_text(
        "\n".join(
            json.dumps({"prompt_id": c["prompt_id"], "model_name": "m", "output": "ok"})
            for c in cases
        )
        + "\n",
        encoding="utf-8",
    )

    out_suite = tmp_path / "mini_suite.json"
    out_cap = tmp_path / "mini_capture.jsonl"
    code = main(
        [
            "subsample",
            "-s",
            str(suite),
            "--size",
            "4",
            "--seed",
            "7",
            "-o",
            str(out_suite),
            "--slice-capture",
            str(cap),
            str(out_cap),
        ]
    )
    assert code == 0
    assert "Sampled 4/8 prompt(s)" in capsys.readouterr().out
    sub_cases = json.loads(out_suite.read_text(encoding="utf-8"))
    assert len(sub_cases) == 4
    cats = [c["category"] for c in sub_cases]
    assert cats.count("extraction") == 2 and cats.count("classification") == 2
    sub_rows = [json.loads(line) for line in out_cap.read_text(encoding="utf-8").splitlines()]
    assert [r["prompt_id"] for r in sub_rows] == [c["prompt_id"] for c in sub_cases]

    # Colliding output paths fail cleanly
    assert (
        main(
            [
                "subsample",
                "-s",
                str(suite),
                "--size",
                "2",
                "-o",
                str(out_suite),
                "--slice-capture",
                str(cap),
                str(out_suite),
            ]
        )
        == 2
    )


def test_cli_compare_reports_markdown_output_and_collision_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pass_path = str(EXAMPLES / "expected-pass.json")
    drift_path = str(EXAMPLES / "expected-drift.json")
    md_out = tmp_path / "leaderboard.md"

    code = main(
        [
            "compare-reports",
            drift_path,
            pass_path,
            "--against",
            "demo-candidate-pass",
            "--markdown-output",
            str(md_out),
        ]
    )
    assert code == 0
    _ = capsys.readouterr()
    md = md_out.read_text(encoding="utf-8")
    assert "# Candidate Drift Comparison" in md
    assert "| Rank | Candidate |" in md
    assert "Prompt transitions vs `demo-candidate-pass`" in md

    # Same file for --html-output and --markdown-output must fail
    same = tmp_path / "same.out"
    assert (
        main(
            [
                "compare-reports",
                drift_path,
                pass_path,
                "--html-output",
                str(same),
                "--markdown-output",
                str(same),
            ]
        )
        == 2
    )


def test_cli_calibrate_numeric_tolerances(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_a = tmp_path / "a.jsonl"
    base_b = tmp_path / "b.jsonl"
    base_a.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "m", "output": '{"score": 0.90}'}) + "\n",
        encoding="utf-8",
    )
    base_b.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "m", "output": '{"score": 0.92}'}) + "\n",
        encoding="utf-8",
    )

    code = main(
        [
            "calibrate",
            "--baseline-a",
            str(base_a),
            "--baseline-b",
            str(base_b),
            "--numeric-tolerances",
        ]
    )
    assert code == 0
    toml_out = capsys.readouterr().out
    assert "tolerance_abs =" in toml_out
    assert "ignore = true" not in toml_out

    # --numeric-tolerances without baseline captures raises exit code 2
    assert main(["calibrate", str(EXAMPLES / "expected-pass.json"), "--numeric-tolerances"]) == 2


def test_cli_capture_provider_anthropic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "prompts.json"
    suite.write_text(
        json.dumps([{"prompt_id": "p1", "prompt": "Say hi"}]),
        encoding="utf-8",
    )
    out = tmp_path / "anthropic.jsonl"
    seen: list[dict[str, Any]] = []

    def fake_post(
        url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> dict[str, Any]:
        seen.append({"url": url, "body": body, "headers": headers})
        return {
            "content": [{"type": "text", "text": '{"ok": true}'}],
            "usage": {"output_tokens": 5},
        }

    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-test-key")
    monkeypatch.setattr("llm_drift_bench.capture._default_transport", fake_post)

    code = main(
        [
            "capture",
            "-s",
            str(suite),
            "--endpoint",
            "https://api.anthropic.com",
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-5",
            "--api-key-env",
            "TEST_ANTHROPIC_KEY",
            "-o",
            str(out),
        ]
    )
    assert code == 0
    assert "Captured 1 prompt(s)" in capsys.readouterr().out
    assert seen[0]["url"].endswith("/v1/messages")
    assert seen[0]["headers"]["x-api-key"] == "sk-ant-test-key"
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["output"] == '{"ok": true}'
    assert row["tokens_generated"] == 5
