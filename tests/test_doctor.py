"""Tests for the doctor pre-flight validation engine and CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main
from llm_drift_bench.doctor import run_doctor

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_doctor_validates_examples_config_suite_and_captures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main([
        "doctor",
        "--config", str(EXAMPLES / "drift.toml"),
        "-s", str(EXAMPLES / "extraction-suite.json"),
        "-b", str(EXAMPLES / "baseline.jsonl"),
        "-c", str(EXAMPLES / "candidate-pass.jsonl"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "llm-drift-bench doctor [OK]" in out
    assert "id_alignment" in out

    rc_json = main(["doctor", "--format", "json"])
    assert rc_json == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "python_ast" in payload["scorers"]


def test_doctor_catches_unsupported_schema_bad_policy_and_id_mismatch(tmp_path: Path) -> None:
    bad_suite = tmp_path / "bad_suite.json"
    bad_suite.write_text(
        json.dumps([
            {
                "prompt_id": "p1",
                "prompt": "x",
                "schema": {"type": "object", "anyOf": [{"type": "string"}]},
            }
        ]),
        encoding="utf-8",
    )
    bad_cfg = tmp_path / "bad.toml"
    bad_cfg.write_text("[unknown_section]\nx = 1\n", encoding="utf-8")

    rep = run_doctor(config_path=bad_cfg, suite_path=bad_suite)
    assert rep.ok is False
    by_name = {c.name: c for c in rep.checks}
    assert by_name["config"].status == "FAIL"
    assert by_name["suite_schemas"].status == "FAIL"
    assert "anyOf" in by_name["suite_schemas"].detail

    # Mismatched capture IDs fail alignment check and return exit code 1 from CLI.
    base = tmp_path / "base.jsonl"
    cand = tmp_path / "cand.jsonl"
    base.write_text('{"prompt_id": "p1", "model": "m1", "output": "ok"}\n', encoding="utf-8")
    cand.write_text('{"prompt_id": "p2", "model": "m2", "output": "ok"}\n', encoding="utf-8")
    assert main(["doctor", "-b", str(base), "-c", str(cand)]) == 1


def test_doctor_warns_on_suite_hygiene_and_capture_secrets(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps([
            {
                "prompt_id": "p1",
                "category": "extraction",
                "prompt": "Extract",
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            }
        ]),
        encoding="utf-8",
    )
    leaky = tmp_path / "leaky.jsonl"
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234"
    leaky.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "m1", "output": f"key={secret}"}) + "\n",
        encoding="utf-8",
    )
    rep = run_doctor(suite_path=suite, baseline_path=leaky)
    assert rep.ok is True  # warnings do not fail doctor
    by_name = {c.name: c for c in rep.checks}
    assert by_name["suite_hygiene"].status == "WARN"
    assert "permissive_object" in by_name["suite_hygiene"].detail
    assert by_name["secrets_scan"].status == "WARN"
    assert "OPENAI_KEY=1" in by_name["secrets_scan"].detail
    assert secret not in by_name["secrets_scan"].detail

