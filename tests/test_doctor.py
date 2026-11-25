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


def test_doctor_blank_outputs_consumer_decisions_and_error_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Complete timings, partial timings, blank outputs, and aligned consumer decisions
    base = tmp_path / "base.jsonl"
    cand = tmp_path / "cand.jsonl"
    b_dec = tmp_path / "b_dec.jsonl"
    c_dec = tmp_path / "c_dec.jsonl"
    base.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "b", "output": "ok", "latency_ms": 10.0})
        + "\n"
        + json.dumps({"prompt_id": "p2", "model_name": "b", "output": "ok", "latency_ms": 12.0})
        + "\n",
        encoding="utf-8",
    )
    cand.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "c", "output": "   ", "latency_ms": 15.0})
        + "\n"
        + json.dumps({"prompt_id": "p2", "model_name": "c", "output": "ok"})
        + "\n",
        encoding="utf-8",
    )
    b_dec.write_text(
        '{"prompt_id": "p1", "accepted": true}\n{"prompt_id": "p2", "accepted": true}\n',
        encoding="utf-8",
    )
    c_dec.write_text(
        '{"prompt_id": "p1", "accepted": false, "reason": "blank"}\n'
        '{"prompt_id": "p2", "accepted": true}\n',
        encoding="utf-8",
    )

    rc = main(
        [
            "doctor",
            "-b",
            str(base),
            "-c",
            str(cand),
            "--baseline-decisions",
            str(b_dec),
            "--candidate-decisions",
            str(c_dec),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    checks = {c["name"]: c for c in payload["checks"]}
    assert checks["baseline"]["status"] == "OK"
    assert "complete timings" in checks["baseline"]["detail"]
    assert checks["candidate"]["status"] == "WARN"
    assert "1 missing latency" in checks["candidate"]["detail"]
    assert "1 blank output(s)" in checks["candidate"]["detail"]
    assert checks["baseline_decisions"]["status"] == "OK"
    assert checks["candidate_decisions"]["status"] == "OK"
    assert "1 rejected" in checks["candidate_decisions"]["detail"]
    assert checks["id_alignment"]["status"] == "OK"

    # Empty suite, empty capture, mixed model names, missing files, and invalid decision log
    empty_suite = tmp_path / "empty_suite.json"
    empty_suite.write_text("[]", encoding="utf-8")
    empty_cap = tmp_path / "empty.jsonl"
    empty_cap.write_text("", encoding="utf-8")
    mixed_cap = tmp_path / "mixed.jsonl"
    mixed_cap.write_text(
        '{"prompt_id": "p1", "model_name": "m1", "output": "a"}\n'
        '{"prompt_id": "p2", "model_name": "m2", "output": "b"}\n',
        encoding="utf-8",
    )
    bad_dec = tmp_path / "bad_dec.jsonl"
    bad_dec.write_text('{"prompt_id": "p1", "accepted": "nope"}\n', encoding="utf-8")

    rep1 = run_doctor(
        suite_path=empty_suite,
        baseline_path=empty_cap,
        candidate_path=mixed_cap,
        baseline_decisions_path=bad_dec,
    )
    assert rep1.ok is False
    by_name1 = {c.name: c for c in rep1.checks}
    assert by_name1["suite"].status == "FAIL"
    assert by_name1["baseline"].status == "FAIL"
    assert by_name1["candidate"].status == "FAIL"
    assert "multiple model names" in by_name1["candidate"].detail
    assert by_name1["baseline_decisions"].status == "FAIL"

    rep2 = run_doctor(
        suite_path=tmp_path / "missing_suite.json",
        baseline_path=tmp_path / "missing_cap.jsonl",
    )
    assert rep2.ok is False
    by_name2 = {c.name: c for c in rep2.checks}
    assert by_name2["suite"].status == "FAIL"
    assert by_name2["baseline"].status == "FAIL"


