"""Offline transcript adapters normalizing OpenAI, Anthropic, and OTel GenAI logs to ToolEvent."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from llm_drift_bench._file_io import parse_source_json, read_utf8
from llm_drift_bench.tool_events import ToolEvent
from llm_drift_bench.tool_log import load_tool_events

AdapterName = Literal["normalized", "openai", "anthropic", "otel"]

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
        elif msg.get("type") == "function_call_output" and isinstance(msg.get("call_id"), str):
            tool_results[msg["call_id"]] = msg

    events: list[ToolEvent] = []
    seq = 0
    for msg in messages:
        # Support both Chat Completions tool_calls and Responses API function_call items.
        raw_calls: list[tuple[str | None, str | None, Any]] = []
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    continue
                fn = call.get("function")
                if not isinstance(fn, Mapping):
                    continue
                cid = call.get("id") if isinstance(call.get("id"), str) else None
                name = fn.get("name") if isinstance(fn.get("name"), str) else None
                raw_calls.append((cid, name, fn.get("arguments")))
        elif msg.get("type") == "function_call":
            cid = msg.get("call_id") if isinstance(msg.get("call_id"), str) else None
            name = msg.get("name") if isinstance(msg.get("name"), str) else None
            raw_calls.append((cid, name, msg.get("arguments")))

        for call_id, name, raw_args in raw_calls:
            if name is None:
                continue
            if name in read_tools:
                kind = "read"
            elif name in edit_tools:
                kind = "edit"
            else:
                continue
            args = _extract_args_dict(raw_args)
            resource = _extract_resource(args, resource_keys)
            if resource is None:
                continue
            result_msg = tool_results.get(call_id) if call_id is not None else None
            success = msg.get("status") not in ("error", "failed", "incomplete")
            start: int | None = None
            end: int | None = None
            if result_msg is not None:
                if (
                    result_msg.get("is_error") is True
                    or result_msg.get("status") in ("error", "failed", "incomplete")
                ):
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


def _normalize_otel_attributes(raw_attrs: Any) -> dict[str, Any]:
    """Normalize flat span attribute maps or OTLP [{key, value: {...}}] arrays."""
    if isinstance(raw_attrs, Mapping):
        return dict(raw_attrs)
    if not isinstance(raw_attrs, list):
        return {}
    normalized: dict[str, Any] = {}
    for entry in raw_attrs:
        if not isinstance(entry, Mapping):
            continue
        key = entry.get("key")
        val = entry.get("value")
        if not isinstance(key, str) or not isinstance(val, Mapping):
            continue
        if "stringValue" in val:
            normalized[key] = val["stringValue"]
        elif "intValue" in val:
            try:
                normalized[key] = int(val["intValue"])
            except (TypeError, ValueError):
                normalized[key] = val["intValue"]
        elif "boolValue" in val:
            normalized[key] = val["boolValue"]
        elif "doubleValue" in val:
            normalized[key] = val["doubleValue"]
    return normalized


def _extract_otlp_resource_spans(parsed: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Flatten standard OTLP JSON resourceSpans -> scopeSpans -> spans."""
    spans: list[Mapping[str, Any]] = []
    resource_spans = parsed.get("resourceSpans")
    if not isinstance(resource_spans, list):
        return spans
    for res_span in resource_spans:
        if not isinstance(res_span, Mapping):
            continue
        scope_spans = res_span.get("scopeSpans") or res_span.get("instrumentationLibrarySpans")
        if not isinstance(scope_spans, list):
            continue
        for scope in scope_spans:
            if not isinstance(scope, Mapping):
                continue
            nested = scope.get("spans")
            if isinstance(nested, list):
                spans.extend(item for item in nested if isinstance(item, Mapping))
    return spans


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
        attrs = _normalize_otel_attributes(span.get("attributes"))
        if not attrs:
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
        if isinstance(status, Mapping):
            raw_code = status.get("code", "")
            if raw_code == 2 or str(raw_code).upper() in ("ERROR", "STATUS_CODE_ERROR"):
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


def load_adapted_tool_events(
    path: str | Path,
    adapter: AdapterName = "normalized",
) -> list[ToolEvent]:
    """Load a trace file using either the strict normalized format or a transcript adapter."""
    if adapter == "normalized":
        return load_tool_events(path)
    source = Path(path)
    raw = read_utf8(source)
    stripped = raw.strip(" \t\r\n")
    if not stripped:
        raise ValueError(f"{source}: transcript file is empty.")
    if source.suffix.lower() == ".jsonl":
        items: Any = [
            parse_source_json(line, source, line=idx)
            for idx, line in enumerate(raw.split("\n"), start=1)
            if line.strip(" \t\r")
        ]
    else:
        parsed = parse_source_json(raw, source)
        if isinstance(parsed, dict):
            if "resourceSpans" in parsed:
                items = _extract_otlp_resource_spans(parsed)
            else:
                for key in ("messages", "output", "spans", "events"):
                    if key in parsed:
                        items = parsed[key]
                        break
                else:
                    raise ValueError(
                        f"{source}: unrecognized transcript wrapper object; expected "
                        "'messages', 'output', 'spans', 'events', or 'resourceSpans'."
                    )
        else:
            items = parsed
    if not isinstance(items, list):
        raise ValueError(f"{source}: transcript must contain a JSON array or JSONL records.")
    if any(not isinstance(item, Mapping) for item in items):
        raise ValueError(f"{source}: every transcript record must be a JSON object.")
    records: list[Mapping[str, Any]] = list(items)
    if adapter == "openai":
        return from_openai_tool_calls(records)
    if adapter == "anthropic":
        return from_anthropic_transcript(records)
    if adapter == "otel":
        return from_otel_genai_spans(records)
    raise ValueError(f"Unsupported adapter: {adapter!r}.")
