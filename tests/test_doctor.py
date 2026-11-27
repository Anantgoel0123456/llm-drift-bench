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
    assert payload["report_schema_version"] == 1
    assert payload["history_schema_version"] == 1
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


def test_doctor_validates_conversation_files_and_id_alignment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_conv = tmp_path / "base_conv.jsonl"
    cand_conv = tmp_path / "cand_conv.jsonl"
    mismatch_conv = tmp_path / "mismatch_conv.jsonl"
    user_only_conv = tmp_path / "user_only.jsonl"
    bad_conv = tmp_path / "bad_conv.jsonl"

    base_conv.write_text(
        json.dumps(
            {
                "conversation_id": "c1",
                "model_name": "m1",
                "turns": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cand_conv.write_text(
        json.dumps(
            {
                "conversation_id": "c1",
                "model_name": "m2",
                "turns": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello!"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mismatch_conv.write_text(
        json.dumps(
            {
                "conversation_id": "c999",
                "model_name": "m2",
                "turns": [{"role": "assistant", "content": "hi"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    user_only_conv.write_text(
        json.dumps(
            {
                "conversation_id": "c1",
                "model_name": "m1",
                "turns": [{"role": "user", "content": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bad_conv.write_text('{"conversation_id": "c1", "turns": []}\n', encoding="utf-8")

    rc = main(
        [
            "doctor",
            "--baseline-conversations",
            str(base_conv),
            "--candidate-conversations",
            str(cand_conv),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    checks = {c["name"]: c for c in payload["checks"]}
    assert checks["baseline_conversations"]["status"] == "OK"
    assert "1 assistant turn(s)" in checks["baseline_conversations"]["detail"]
    assert checks["candidate_conversations"]["status"] == "OK"
    assert checks["conversation_alignment"]["status"] == "OK"
    assert "report_schema=v1" in checks["runtime"]["detail"]

    # Mismatched conversation IDs fail alignment and exit 1
    assert (
        main(
            [
                "doctor",
                "--baseline-conversations",
                str(base_conv),
                "--candidate-conversations",
                str(mismatch_conv),
            ]
        )
        == 1
    )
    capsys.readouterr()

    # Zero assistant turns emits WARN (still ok=True); malformed file emits FAIL
    rep_warn = run_doctor(baseline_conversations_path=user_only_conv)
    assert rep_warn.ok is True
    assert {c.name: c.status for c in rep_warn.checks}["baseline_conversations"] == "WARN"

    rep_bad = run_doctor(baseline_conversations_path=bad_conv)
    assert rep_bad.ok is False
    assert {c.name: c.status for c in rep_bad.checks}["baseline_conversations"] == "FAIL"


def test_doctor_validates_tool_traces_adapters_and_secrets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    b_trace = tmp_path / "base_trace.jsonl"
    c_trace = tmp_path / "cand_trace.jsonl"
    empty_trace = tmp_path / "empty_trace.jsonl"
    secret_trace = tmp_path / "secret_trace.jsonl"
    bad_trace = tmp_path / "bad_trace.jsonl"
    oai_trace = tmp_path / "oai_trace.json"

    b_trace.write_text(
        '{"kind": "read", "resource": "src/a.py@1", "start": 0, "end": 10}\n'
        '{"kind": "edit", "resource": "src/a.py@1"}\n',
        encoding="utf-8",
    )
    c_trace.write_text(
        '{"kind": "read", "resource": "src/a.py@1", "start": 0, "end": 10}\n',
        encoding="utf-8",
    )
    empty_trace.write_text("", encoding="utf-8")
    secret_trace.write_text(
        '{"kind": "read", "resource": "sk-ant-api03-1234567890abcdefghij", "start": 0, "end": 5}\n',
        encoding="utf-8",
    )
    bad_trace.write_text('{"kind": "bogus"}\n', encoding="utf-8")
    oai_trace.write_text(
        json.dumps(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "app.py", "start": 0, "end": 20}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ]
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "doctor",
            "--baseline-trace",
            str(b_trace),
            "--candidate-trace",
            str(c_trace),
            "--format",
            "json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    checks = {c["name"]: c for c in payload["checks"]}
    assert checks["baseline_trace"]["status"] == "OK"
    assert "2 tool event(s)" in checks["baseline_trace"]["detail"]
    assert checks["candidate_trace"]["status"] == "OK"
    assert checks["trace_secrets"]["status"] == "OK"

    # Adapted OpenAI trace
    rep_oai = run_doctor(baseline_trace_path=oai_trace, trace_adapter="openai")
    assert rep_oai.ok is True
    assert {c.name: c.status for c in rep_oai.checks}["baseline_trace"] == "OK"

    # Adapted OpenAI trace with 0 tool calls emits WARN (still ok=True)
    no_tools_oai = tmp_path / "no_tools_oai.json"
    no_tools_oai.write_text(
        json.dumps([{"role": "assistant", "content": "no tool calls"}]),
        encoding="utf-8",
    )
    rep_warn = run_doctor(baseline_trace_path=no_tools_oai, trace_adapter="openai")
    assert rep_warn.ok is True
    assert {c.name: c.status for c in rep_warn.checks}["baseline_trace"] == "WARN"

    # Empty normalized trace fails; secret trace warns without echoing secret; malformed fails
    rep_empty = run_doctor(baseline_trace_path=empty_trace)
    assert rep_empty.ok is False
    assert {c.name: c.status for c in rep_empty.checks}["baseline_trace"] == "FAIL"

    rep_sec = run_doctor(baseline_trace_path=secret_trace)
    assert rep_sec.ok is True
    sec_check = {c.name: c for c in rep_sec.checks}["trace_secrets"]
    assert sec_check.status == "WARN" and "anthropic_key" in sec_check.detail
    assert "1234567890abcdefghij" not in sec_check.detail

    rep_bad = run_doctor(baseline_trace_path=bad_trace)
    assert rep_bad.ok is False
    assert {c.name: c.status for c in rep_bad.checks}["baseline_trace"] == "FAIL"



