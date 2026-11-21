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
