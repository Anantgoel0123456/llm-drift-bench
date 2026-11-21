"""Offline transcript adapters normalizing OpenAI, Anthropic, and OTel GenAI logs to ToolEvent."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from llm_drift_bench.tool_events import ToolEvent

DEFAULT_READ_TOOLS = frozenset({"read_file", "Read", "view_file", "get_file", "cat"})
DEFAULT_EDIT_TOOLS = frozenset({
    "edit_file", "Edit", "write_file", "Write", "replace_file", "apply_patch", "MultiEdit",
})
DEFAULT_RESOURCE_KEYS = ("resource", "path", "file_path", "target_file", "filename")


def _extract_args_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_resource(args: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_range(meta: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    """Return explicit character offsets only when strictly valid integers; else (None, None)."""
    if not isinstance(meta, Mapping):
        return None, None
    start = meta.get("start")
    end = meta.get("end")
    if (
        type(start) is int
        and type(end) is int
        and start >= 0
        and end > start
    ):
        return start, end
    return None, None


def from_openai_tool_calls(
    messages: Sequence[Mapping[str, Any]],
    *,
    read_tools: frozenset[str] = DEFAULT_READ_TOOLS,
    edit_tools: frozenset[str] = DEFAULT_EDIT_TOOLS,
    resource_keys: Sequence[str] = DEFAULT_RESOURCE_KEYS,
) -> list[ToolEvent]:
    """Normalize sanitized OpenAI chat-completions tool_calls + tool result messages to ToolEvents.

    When exact character offsets are not explicitly supplied in tool result metadata,
    read ranges remain None so downstream checks preserve honest UNKNOWN verdicts.
    """
    tool_results: dict[str, Mapping[str, Any]] = {}
    for msg in messages:
        if msg.get("role") == "tool" and isinstance(msg.get("tool_call_id"), str):
            tool_results[msg["tool_call_id"]] = msg

    events: list[ToolEvent] = []
    seq = 0
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            fn = call.get("function")
            if not isinstance(fn, Mapping):
                continue
            name = fn.get("name")
            if not isinstance(name, str):
                continue
            if name in read_tools:
                kind = "read"
            elif name in edit_tools:
                kind = "edit"
            else:
                continue
            args = _extract_args_dict(fn.get("arguments"))
            resource = _extract_resource(args, resource_keys)
            if resource is None:
                continue
            call_id = call.get("id")
            result_msg = tool_results.get(call_id) if isinstance(call_id, str) else None
            success = True
            start: int | None = None
            end: int | None = None
            if result_msg is not None:
                if result_msg.get("is_error") is True or result_msg.get("status") == "error":
                    success = False
                if kind == "read":
                    start, end = _extract_range(result_msg.get("range"))
            events.append(
                ToolEvent(
                    sequence=seq,
                    kind=kind,  # type: ignore[arg-type]
                    resource=resource,
                    success=success,
                    start=start if kind == "read" else None,
                    end=end if kind == "read" else None,
                )
            )
            seq += 1
    return events


def from_anthropic_transcript(
    messages: Sequence[Mapping[str, Any]],
    *,
    read_tools: frozenset[str] = DEFAULT_READ_TOOLS,
    edit_tools: frozenset[str] = DEFAULT_EDIT_TOOLS,
    resource_keys: Sequence[str] = DEFAULT_RESOURCE_KEYS,
) -> list[ToolEvent]:
    """Normalize sanitized Anthropic/Claude tool_use and tool_result blocks to ToolEvents."""
    results_by_id: dict[str, Mapping[str, Any]] = {}
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, Mapping)
                and block.get("type") == "tool_result"
                and isinstance(block.get("tool_use_id"), str)
            ):
                results_by_id[block["tool_use_id"]] = block

    events: list[ToolEvent] = []
    seq = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if not isinstance(name, str):
                continue
            if name in read_tools:
                kind = "read"
            elif name in edit_tools:
                kind = "edit"
            else:
                continue
            args = _extract_args_dict(block.get("input"))
            resource = _extract_resource(args, resource_keys)
            if resource is None:
                continue
            use_id = block.get("id")
            res_block = results_by_id.get(use_id) if isinstance(use_id, str) else None
            success = not (res_block is not None and res_block.get("is_error") is True)
            start: int | None = None
            end: int | None = None
            if kind == "read" and res_block is not None:
                start, end = _extract_range(res_block.get("range"))
            events.append(
                ToolEvent(
                    sequence=seq,
                    kind=kind,  # type: ignore[arg-type]
                    resource=resource,
                    success=success,
                    start=start if kind == "read" else None,
                    end=end if kind == "read" else None,
                )
            )
            seq += 1
    return events


def from_otel_genai_spans(
    spans: Sequence[Mapping[str, Any]],
    *,
    read_tools: frozenset[str] = DEFAULT_READ_TOOLS,
    edit_tools: frozenset[str] = DEFAULT_EDIT_TOOLS,
    resource_keys: Sequence[str] = DEFAULT_RESOURCE_KEYS,
) -> list[ToolEvent]:
    """Normalize OpenTelemetry GenAI tool spans (gen_ai.tool.*) into ToolEvents."""
    events: list[ToolEvent] = []
    seq = 0
    for span in spans:
        attrs = span.get("attributes")
        if not isinstance(attrs, Mapping):
            continue
        name = attrs.get("gen_ai.tool.name")
        if not isinstance(name, str):
            continue
        if name in read_tools:
            kind = "read"
        elif name in edit_tools:
            kind = "edit"
        else:
            continue
        args = _extract_args_dict(attrs.get("gen_ai.tool.call.arguments"))
        resource = _extract_resource(args, resource_keys)
        if resource is None:
            continue
        status = span.get("status")
        success = True
        if isinstance(status, Mapping) and str(status.get("code", "")).upper() == "ERROR":
            success = False
        start, end = (
            _extract_range(
                {
                    "start": attrs.get("llm_drift_bench.read.start"),
                    "end": attrs.get("llm_drift_bench.read.end"),
                }
            )
            if kind == "read"
            else (None, None)
        )
        events.append(
            ToolEvent(
                sequence=seq,
                kind=kind,  # type: ignore[arg-type]
                resource=resource,
                success=success,
                start=start,
                end=end,
            )
        )
        seq += 1
    return events
