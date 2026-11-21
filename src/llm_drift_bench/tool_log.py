"""Strict import of normalized tool traces and persistence of check verdicts.

This module reads files only. It never executes tools, infers events from raw agent
logs, or resolves path aliases: adapters must normalize a run into the documented
event records first, and must attest that the trace is complete.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from llm_drift_bench._file_io import parse_json, parse_source_json, read_utf8
from llm_drift_bench.tool_events import ToolCheckResult, ToolEvent, check_read_before_edit

EVENT_FIELDS = frozenset({"sequence", "kind", "resource", "success", "start", "end"})
CHECK_FORMAT = "llm-drift-bench/tool-check"
CHECK_VERSION = 1
_CHECK_KEYS = frozenset({
    "format", "version", "check", "resource", "required_start", "required_end",
    "trace_complete", "status", "reason", "covered_characters", "required_characters",
    "first_edit_sequence",
})
_STATUSES = ("PASS", "FAIL", "UNKNOWN")


def _event_from_record(record: Any, position: str, sequence: int | None) -> ToolEvent:
    if not isinstance(record, dict):
        raise ValueError(f"{position}: each event must be a JSON object.")
    unknown = sorted(set(record) - EVENT_FIELDS)
    if unknown:
        raise ValueError(
            f"{position}: unknown event field(s) {unknown}; allowed {sorted(EVENT_FIELDS)}."
        )
    missing = [name for name in ("kind", "resource") if name not in record]
    if missing:
        raise ValueError(f"{position}: missing required field(s) {missing}.")
    if "sequence" in record:
        sequence = record["sequence"]
    if sequence is None:
        raise ValueError(f"{position}: sequence is required.")
    try:
        return ToolEvent(
            sequence=sequence, kind=record["kind"], resource=record["resource"],
            success=record.get("success", True), start=record.get("start"),
            end=record.get("end"),
        )
    except ValueError as exc:
        raise ValueError(f"{position}: {exc}") from exc


def parse_tool_events(text: str) -> list[ToolEvent]:
    """Parse normalized events from a JSON array or JSON Lines text.

    Each record has kind ("read" or "edit"), resource, and optionally sequence,
    success (default true), and start/end for read ranges. Sequence numbers must
    be supplied on every record or on none; if none, file order defines sequence.
    Unknown fields, duplicate keys, nonfinite numbers and non-integer offsets are
    rejected instead of ignored.
    """
    stripped = text.strip(" \t\r\n")
    if not stripped:
        raise ValueError("Tool trace contains no events.")
    records: list[tuple[str, Any]]
    if stripped.startswith("["):
        loaded = parse_json(stripped)
        records = [(f"event {index + 1}", item) for index, item in enumerate(loaded)]
    else:
        records = []
        for number, line in enumerate(text.split("\n"), start=1):
            if line.strip(" \t\r"):
                try:
                    records.append((f"line {number}", parse_json(line)))
                except ValueError as exc:
                    raise ValueError(f"line {number}: {exc}") from exc
    if not records:
        raise ValueError("Tool trace contains no events.")
    with_sequence = [isinstance(item, dict) and "sequence" in item for _, item in records]
    if any(with_sequence) and not all(with_sequence):
        raise ValueError("Provide sequence on every event or on none (file order is used).")
    return [
        _event_from_record(item, position, index)
        for index, (position, item) in enumerate(records)
    ]


def load_tool_events(path: str | Path) -> list[ToolEvent]:
    """Load a normalized trace file (.json array or .jsonl) with bounded strict parsing."""
    text = read_utf8(path)
    try:
        return parse_tool_events(text)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


@dataclass(frozen=True)
class ToolCheckRecord:
    """A verdict together with the exact question it answered, for saving and display."""

    resource: str
    required_start: int
    required_end: int
    trace_complete: bool
    result: ToolCheckResult

    def __post_init__(self) -> None:
        result = self.result
        if not isinstance(self.resource, str) or not self.resource.strip():
            raise ValueError("Tool check resource must be a non-empty string.")
        for name in ("required_start", "required_end"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Tool check {name} must be a non-negative integer.")
        if self.required_end <= self.required_start:
            raise ValueError("Tool check required range must be non-empty.")
        if type(self.trace_complete) is not bool:
            raise ValueError("Tool check trace_complete must be boolean.")
        if result.status not in _STATUSES:
            raise ValueError("Tool check status must be PASS, FAIL or UNKNOWN.")
        if not isinstance(result.reason, str) or not result.reason.strip():
            raise ValueError("Tool check reason must be a non-empty string.")
        counts = (result.covered_characters, result.required_characters)
        if any(type(count) is not int for count in counts):
            raise ValueError("Tool check character counts must be integers.")
        if result.required_characters != self.required_end - self.required_start:
            raise ValueError("Tool check required_characters contradicts its range.")
        if not 0 <= result.covered_characters <= result.required_characters:
            raise ValueError("Tool check covered_characters must lie within the required range.")
        edit = result.first_edit_sequence
        if edit is not None and (type(edit) is not int or edit < 0):
            raise ValueError("Tool check first_edit_sequence must be a non-negative integer.")
        full = result.covered_characters == result.required_characters
        if not self.trace_complete and result.status != "UNKNOWN":
            raise ValueError("An unattested trace can only produce UNKNOWN.")
        if result.status == "PASS" and not (full and edit is not None):
            raise ValueError("PASS requires full coverage before a recorded edit.")
        if result.status == "FAIL" and (full or edit is None):
            raise ValueError("FAIL requires partial coverage and a recorded edit.")
        if result.status == "UNKNOWN" and self.trace_complete and full and edit is not None:
            raise ValueError("A complete trace with full coverage before an edit is PASS.")

    def to_dict(self) -> dict[str, Any]:
        result = self.result
        return {
            "format": CHECK_FORMAT, "version": CHECK_VERSION, "check": "read_before_edit",
            "resource": self.resource, "required_start": self.required_start,
            "required_end": self.required_end, "trace_complete": self.trace_complete,
            "status": result.status, "reason": result.reason,
            "covered_characters": result.covered_characters,
            "required_characters": result.required_characters,
            "first_edit_sequence": result.first_edit_sequence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolCheckRecord:
        keys = set(data)
        if keys != _CHECK_KEYS:
            raise ValueError(
                "Unrecognized tool check file: "
                f"missing {sorted(_CHECK_KEYS - keys)}, unexpected {sorted(keys - _CHECK_KEYS)}."
            )
        if data["format"] != CHECK_FORMAT or data["version"] != CHECK_VERSION:
            raise ValueError(f"Tool check file must be {CHECK_FORMAT} version {CHECK_VERSION}.")
        if data["check"] != "read_before_edit":
            raise ValueError("Only read_before_edit tool checks are supported.")
        result = ToolCheckResult(
            status=cast("Literal['PASS', 'FAIL', 'UNKNOWN']", data["status"]),
            reason=data["reason"],
            covered_characters=data["covered_characters"],
            required_characters=data["required_characters"],
            first_edit_sequence=data["first_edit_sequence"],
        )
        return cls(
            resource=data["resource"], required_start=data["required_start"],
            required_end=data["required_end"], trace_complete=data["trace_complete"],
            result=result,
        )


def check_trace(
    events: Sequence[ToolEvent], *, resource: str, required_start: int, required_end: int,
    trace_complete: bool = False,
) -> ToolCheckRecord:
    """Run the read-before-edit check and keep the question beside the verdict."""
    result = check_read_before_edit(
        events, resource=resource, required_start=required_start,
        required_end=required_end, trace_complete=trace_complete,
    )
    return ToolCheckRecord(resource, required_start, required_end, trace_complete, result)


def check_trace_file(
    path: str | Path, *, resource: str, required_start: int, required_end: int,
    trace_complete: bool = False,
) -> ToolCheckRecord:
    """Load a normalized trace file and check it."""
    return check_trace(
        load_tool_events(path), resource=resource, required_start=required_start,
        required_end=required_end, trace_complete=trace_complete,
    )


def load_tool_check(path: str | Path) -> ToolCheckRecord:
    """Load a saved check file, rejecting any verdict its own evidence contradicts."""
    raw = parse_source_json(read_utf8(path), path)
    if not isinstance(raw, dict):
        raise ValueError(f"A saved tool check must be a JSON object: {path}")
    return ToolCheckRecord.from_dict(raw)
