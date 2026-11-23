"""Unit tests for OpenAI, Anthropic, and OTel transcript adapters and extra tool workflow checks."""

from __future__ import annotations

from llm_drift_bench.adapters import (
    from_anthropic_transcript,
    from_openai_tool_calls,
    from_otel_genai_spans,
)
from llm_drift_bench.tool_events import (
    ToolEvent,
    check_max_tool_retries,
    check_no_secret_in_tool_args,
    check_read_before_edit,
)


def test_openai_transcript_adapter_preserves_ranges_and_unknowns() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "read_file", "arguments": '{"path": "spec.md@v1"}'},
                },
                {
                    "id": "call_2",
                    "function": {"name": "edit_file", "arguments": '{"path": "app.py@v1"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "range": {"start": 0, "end": 500}},
        {"role": "tool", "tool_call_id": "call_2"},
    ]
    events = from_openai_tool_calls(messages)
    assert len(events) == 2
    verdict = check_read_before_edit(
        events, resource="spec.md@v1", required_start=0, required_end=500, trace_complete=True
    )
    assert verdict.status == "PASS"

    # Without explicit range metadata, read coverage stays None -> UNKNOWN (never false PASS!)
    no_range = [messages[0], {"role": "tool", "tool_call_id": "call_1"}]
    events_unknown = from_openai_tool_calls(no_range)
    verdict_u = check_read_before_edit(
        events_unknown,
        resource="spec.md@v1",
        required_start=0,
        required_end=500,
        trace_complete=True,
    )
    assert verdict_u.status == "UNKNOWN"


def test_anthropic_transcript_adapter_handles_errors_and_ranges() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "g@1"}},
                {"type": "tool_use", "id": "tu_2", "name": "Edit", "input": {"file_path": "a@1"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "is_error": True},
                {"type": "tool_result", "tool_use_id": "tu_2", "is_error": False},
            ],
        },
    ]
    events = from_anthropic_transcript(messages)
    assert len(events) == 2
    assert events[0].kind == "read" and events[0].success is False
    verdict = check_read_before_edit(
        events, resource="g@1", required_start=0, required_end=100, trace_complete=True
    )
    assert verdict.status == "FAIL"


def test_otel_genai_spans_adapter() -> None:
    spans = [
        {
            "attributes": {
                "gen_ai.tool.name": "read_file",
                "gen_ai.tool.call.arguments": '{"resource": "doc@1"}',
                "llm_drift_bench.read.start": 0,
                "llm_drift_bench.read.end": 200,
            },
            "status": {"code": "OK"},
        },
        {
            "attributes": {
                "gen_ai.tool.name": "write_file",
                "gen_ai.tool.call.arguments": '{"resource": "main.py@1"}',
            },
            "status": {"code": "ERROR"},
        },
    ]
    events = from_otel_genai_spans(spans)
    assert len(events) == 2
    assert events[0].start == 0 and events[0].end == 200
    assert events[1].success is False


def test_max_tool_retries_and_secret_scanning() -> None:
    failing_events = [
        ToolEvent(0, "read", "f", success=False),
        ToolEvent(1, "read", "f", success=False),
        ToolEvent(2, "edit", "f", success=False),
    ]
    assert check_max_tool_retries(failing_events, max_failed_calls=2).status == "FAIL"
    assert (
        check_max_tool_retries(failing_events, max_failed_calls=3, trace_complete=True).status
        == "PASS"
    )
    assert (
        check_max_tool_retries(failing_events, max_failed_calls=3, trace_complete=False).status
        == "UNKNOWN"
    )

    fake_key = "sk-proj-1234567890abcdefghijklmnop"
    secret_res = check_no_secret_in_tool_args([{"headers": f"Authorization: {fake_key}"}])
    assert secret_res.status == "FAIL"
    assert "openai_key" in secret_res.reason
    assert fake_key not in secret_res.reason

    clean_res = check_no_secret_in_tool_args([{"path": "README.md"}], trace_complete=True)
    assert clean_res.status == "PASS"


def test_openai_responses_api_and_cli_check_tools_adapter(tmp_path) -> None:
    import json

    from llm_drift_bench.cli import main

    transcript = tmp_path / "responses.json"
    transcript.write_text(
        json.dumps(
            [
                {
                    "type": "function_call",
                    "call_id": "fc_1",
                    "name": "read_file",
                    "arguments": '{"path": "guide@1"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "fc_1",
                    "range": {"start": 0, "end": 100},
                },
                {
                    "type": "function_call",
                    "call_id": "fc_2",
                    "name": "edit_file",
                    "arguments": '{"path": "app.py@1"}',
                },
            ]
        ),
        encoding="utf-8",
    )
    rc = main(
        [
            "check-tools",
            "-i",
            str(transcript),
            "--adapter",
            "openai",
            "--resource",
            "guide@1",
            "--end",
            "100",
            "--trace-complete",
        ]
    )
    assert rc == 0


def test_openai_responses_output_wrapper_and_otlp_resource_spans(tmp_path) -> None:
    import json

    import pytest

    from llm_drift_bench.adapters import load_adapted_tool_events

    resp_obj = tmp_path / "resp.json"
    resp_obj.write_text(
        json.dumps({
            "id": "resp_123",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "read_file",
                    "arguments": '{"path": "spec@1"}',
                },
                {"type": "function_call_output", "call_id": "c1", "status": "failed"},
            ],
        }),
        encoding="utf-8",
    )
    events = load_adapted_tool_events(resp_obj, adapter="openai")
    assert len(events) == 1 and events[0].success is False

    otlp_file = tmp_path / "otlp.json"
    otlp_file.write_text(
        json.dumps({
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "attributes": [
                                        {
                                            "key": "gen_ai.tool.name",
                                            "value": {"stringValue": "Read"},
                                        },
                                        {
                                            "key": "gen_ai.tool.call.arguments",
                                            "value": {"stringValue": '{"file_path": "doc@2"}'},
                                        },
                                        {
                                            "key": "llm_drift_bench.read.start",
                                            "value": {"intValue": "0"},
                                        },
                                        {
                                            "key": "llm_drift_bench.read.end",
                                            "value": {"intValue": "50"},
                                        },
                                    ],
                                    "status": {"code": "STATUS_CODE_OK"},
                                },
                                {
                                    "attributes": [
                                        {
                                            "key": "gen_ai.tool.name",
                                            "value": {"stringValue": "Edit"},
                                        },
                                        {
                                            "key": "gen_ai.tool.call.arguments",
                                            "value": {"stringValue": '{"file_path": "doc@2"}'},
                                        },
                                    ],
                                    "status": {"code": 2},
                                },
                            ]
                        }
                    ]
                }
            ]
        }),
        encoding="utf-8",
    )
    otel_events = load_adapted_tool_events(otlp_file, adapter="otel")
    assert len(otel_events) == 2
    assert otel_events[0].start == 0 and otel_events[0].end == 50
    assert otel_events[1].success is False

    bad_wrapper = tmp_path / "bad.json"
    bad_wrapper.write_text('{"unknown": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognized transcript wrapper"):
        load_adapted_tool_events(bad_wrapper, adapter="openai")


def test_expanded_secret_patterns_and_cli_max_failed_calls(tmp_path, capsys) -> None:
    import json

    from llm_drift_bench.cli import main

    ant_key = "sk-ant-api03-1234567890abcdefghij"
    google_key = "AIza" + "A" * 35
    slack_tok = "xoxb-1234567890-abcdef"
    res = check_no_secret_in_tool_args([ant_key, google_key, slack_tok])
    assert res.status == "FAIL"
    assert "anthropic_key" in res.reason
    assert "openai_key" not in res.reason
    assert "google_api_key" in res.reason
    assert "slack_token" in res.reason
    assert ant_key not in res.reason and google_key not in res.reason

    trace = tmp_path / "retries.jsonl"
    trace.write_text(
        '{"kind": "read", "resource": "g@1", "start": 0, "end": 10, "success": true}\n'
        '{"kind": "edit", "resource": "a@1", "success": false}\n'
        '{"kind": "edit", "resource": "a@1", "success": false}\n',
        encoding="utf-8",
    )
    # read_before_edit passes, but 2 failed calls exceeds --max-failed-calls 1 -> exit 1!
    rc = main([
        "check-tools",
        "-i", str(trace),
        "--resource", "g@1",
        "--end", "10",
        "--trace-complete",
        "--max-failed-calls", "1",
        "--format", "json",
    ])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "PASS"
    assert out["max_failed_calls_check"]["status"] == "FAIL"

    # Raising the budget to 2 succeeds with exit 0.
    rc_ok = main([
        "check-tools",
        "-i", str(trace),
        "--resource", "g@1",
        "--end", "10",
        "--trace-complete",
        "--max-failed-calls", "2",
    ])
    assert rc_ok == 0
    assert "max_tool_retries: PASS" in capsys.readouterr().out


