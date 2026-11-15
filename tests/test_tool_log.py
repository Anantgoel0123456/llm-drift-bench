"""Normalized trace import and saved check records."""

import json
import random
from pathlib import Path
from typing import Any

import pytest

from llm_drift_bench import (
    ToolCheckRecord,
    ToolCheckResult,
    ToolEvent,
    check_trace,
    check_trace_file,
    load_tool_check,
    load_tool_events,
    parse_tool_events,
)

RES = "guide.md@sha256:1f2e"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tool-trace" / "trace.jsonl"


def test_jsonl_uses_file_order_when_sequence_is_omitted() -> None:
    events = parse_tool_events(
        '{"kind":"read","resource":"a","start":0,"end":5}\n\n{"kind":"edit","resource":"b"}\n'
    )
    assert events == [
        ToolEvent(0, "read", "a", start=0, end=5),
        ToolEvent(1, "edit", "b"),
    ]


def test_json_array_and_explicit_sequences() -> None:
    events = parse_tool_events(json.dumps([
        {"sequence": 7, "kind": "edit", "resource": "x", "success": False},
        {"sequence": 3, "kind": "read", "resource": "x", "start": 0, "end": 2},
    ]))
    assert [(e.sequence, e.kind, e.success) for e in events] == [
        (7, "edit", False), (3, "read", True)]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "no events"),
        ("\n  \n", "no events"),
        ("[]", "no events"),
        ('{"kind":"read","resource":"a","start":0,"end":5,"ends":6}', "unknown event field"),
        ('{"kind":"read"}', "missing required"),
        ('{"kind":"open","resource":"a"}', "kind must be"),
        ('{"kind":"read","resource":"a","start":0}', "range endpoints"),
        ('{"kind":"read","resource":"a","start":0.0,"end":5.0}', "integer"),
        ('{"kind":"read","resource":"a","start":"0","end":5}', "integer"),
        ('{"kind":"read","resource":"a","start":0,"end":5,"success":"yes"}', "boolean"),
        ('{"kind":"read","resource":"a","start":5,"end":5}', "non-empty"),
        ('{"kind":"edit","resource":"a","start":0,"end":5}', "Only read events"),
        ('{"kind":"read","resource":"  "}', "non-empty"),
        ('{"kind":"read","resource":"a","kind":"edit"}', "Duplicate JSON object key"),
        ('{"kind":"read","resource":"a","start":NaN,"end":5}', "NaN"),
        ('{"sequence":-1,"kind":"edit","resource":"a"}', "sequence"),
        ('{"sequence":true,"kind":"edit","resource":"a"}', "sequence"),
        ('["not an object"]', "JSON object"),
        ('{"kind":"edit","resource":"a"}\nnot json', "line 2"),
        ('{"sequence":0,"kind":"edit","resource":"a"}\n{"kind":"edit","resource":"b"}',
         "every event or on none"),
    ],
)
def test_malformed_traces_are_rejected_with_a_location(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_tool_events(text)


def test_error_messages_name_the_offending_line() -> None:
    text = '{"kind":"edit","resource":"a"}\n{"kind":"read","resource":"a","start":9}\n'
    with pytest.raises(ValueError, match="line 2"):
        parse_tool_events(text)


def test_example_trace_loads_and_checks() -> None:
    assert len(load_tool_events(EXAMPLE)) == 4
    record = check_trace_file(
        EXAMPLE, resource=RES, required_start=0, required_end=4000, trace_complete=True)
    assert record.result.status == "FAIL"
    assert record.result.covered_characters == 1000
    assert record.result.first_edit_sequence == 2
    done = check_trace_file(
        EXAMPLE, resource=RES, required_start=0, required_end=1000, trace_complete=True)
    assert done.result.status == "PASS"
    unattested = check_trace_file(EXAMPLE, resource=RES, required_start=0, required_end=1000)
    assert unattested.result.status == "UNKNOWN"


def test_oversized_and_non_utf8_files_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b'{"kind":"edit","resource":"\xff"}')
    with pytest.raises(ValueError, match="UTF-8"):
        load_tool_events(bad)


def _saved(tmp_path: Path, **overrides: Any) -> Path:
    data = check_trace_file(
        EXAMPLE, resource=RES, required_start=0, required_end=4000, trace_complete=True,
    ).to_dict()
    data.update(overrides)
    path = tmp_path / "check.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_saved_check_round_trips(tmp_path: Path) -> None:
    path = _saved(tmp_path)
    loaded = load_tool_check(path)
    assert loaded.result.status == "FAIL"
    assert json.loads(path.read_text(encoding="utf-8")) == loaded.to_dict()


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "PASS"},
        {"status": "MAYBE"},
        {"status": "UNKNOWN", "covered_characters": 4000},
        {"trace_complete": False},
        {"covered_characters": 5000},
        {"covered_characters": -1},
        {"required_characters": 10},
        {"required_end": 0},
        {"required_start": 4000},
        {"first_edit_sequence": None},
        {"first_edit_sequence": -1},
        {"first_edit_sequence": 1.5},
        {"covered_characters": 1000.0},
        {"covered_characters": True},
        {"reason": ""},
        {"resource": " "},
        {"version": 2},
        {"format": "other"},
        {"check": "something_else"},
        {"extra": 1},
    ],
)
def test_contradictory_or_unrecognized_checks_are_rejected(
    tmp_path: Path, overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        load_tool_check(_saved(tmp_path, **overrides))


def test_missing_keys_and_non_objects_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_tool_check(path)
    path.write_text('{"format":"llm-drift-bench/tool-check"}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_tool_check(path)


def test_pass_cannot_be_forged_without_attestation_or_edit() -> None:
    forged = ToolCheckResult("PASS", "ok", 10, 10, None)
    with pytest.raises(ValueError, match="PASS requires"):
        ToolCheckRecord("r", 0, 10, True, forged)
    with pytest.raises(ValueError, match="unattested"):
        ToolCheckRecord("r", 0, 10, False, ToolCheckResult("PASS", "ok", 10, 10, 3))


@pytest.mark.parametrize("seed", range(200))
def test_every_checker_verdict_is_a_valid_saved_record(seed: int) -> None:
    """The record's consistency rules must never reject what the checker itself returns."""
    rng = random.Random(seed)
    events: list[ToolEvent] = []
    for sequence in rng.sample(range(30), rng.randint(0, 12)):
        resource = rng.choice([RES, "other"])
        if rng.random() < 0.4:
            events.append(ToolEvent(sequence, "edit", resource, rng.random() < 0.8))
        elif rng.random() < 0.15:
            events.append(ToolEvent(sequence, "read", resource, rng.random() < 0.9))
        else:
            start = rng.randint(0, 90)
            events.append(ToolEvent(
                sequence, "read", resource, rng.random() < 0.9, start, start + rng.randint(1, 60)))
    start = rng.randint(0, 40)
    record = check_trace(
        events, resource=RES, required_start=start, required_end=start + rng.randint(1, 100),
        trace_complete=rng.random() < 0.8,
    )
    assert ToolCheckRecord.from_dict(record.to_dict()) == record
