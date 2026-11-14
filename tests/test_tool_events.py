"""Behavioral checks on read completions and edit invocations."""

import pytest

from llm_drift_bench.tool_events import ToolEvent, check_read_before_edit


def test_overlapping_reads_count_once_and_input_order_does_not_matter() -> None:
    events = [
        ToolEvent(4, "edit", "code@v1"),
        ToolEvent(3, "read", "guide@v1", start=5, end=12),
        ToolEvent(1, "read", "guide@v1", start=0, end=7),
        ToolEvent(2, "read", "guide@v1", start=0, end=7),
    ]
    result = check_read_before_edit(
        events, resource="guide@v1", required_start=0, required_end=10, trace_complete=True,
    )
    assert result.status == "PASS"
    assert result.covered_characters == result.required_characters == 10
    assert result.first_edit_sequence == 4


@pytest.mark.parametrize("read", [
    ToolEvent(2, "read", "guide@v1", start=0, end=10),  # after edit
    ToolEvent(0, "read", "guide@v2", start=0, end=10),  # different revision
    ToolEvent(0, "read", "guide@v1", success=False, start=0, end=10),
    ToolEvent(0, "read", "guide@v1", start=0, end=5),
    ToolEvent(0, "read", "guide@v1", start=10, end=20),
])
def test_missing_coverage_fails_even_if_edit_itself_failed(read: ToolEvent) -> None:
    result = check_read_before_edit(
        [read, ToolEvent(1, "edit", "code@v1", success=False)], resource="guide@v1",
        required_start=0, required_end=10, trace_complete=True,
    )
    assert result.status == "FAIL"


def test_unknown_trace_or_range_never_becomes_a_false_pass() -> None:
    read = ToolEvent(0, "read", "guide@v1", start=0, end=10)
    edit = ToolEvent(1, "edit", "code@v1")
    for events, complete in [([read, edit], False), ([read], True),
                             ([ToolEvent(0, "read", "guide@v1"), edit], True)]:
        result = check_read_before_edit(
            events, resource="guide@v1", required_start=0, required_end=10,
            trace_complete=complete,
        )
        assert result.status == "UNKNOWN"


def test_disjoint_ranges_leave_a_gap() -> None:
    result = check_read_before_edit(
        [ToolEvent(0, "read", "g", start=0, end=4),
         ToolEvent(1, "read", "g", start=6, end=10), ToolEvent(2, "edit", "code")],
        resource="g", required_start=2, required_end=8, trace_complete=True,
    )
    assert result.status == "FAIL"
    assert result.covered_characters == 4


def test_duplicate_sequence_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        check_read_before_edit(
            [ToolEvent(0, "read", "g"), ToolEvent(0, "edit", "code")],
            resource="g", required_start=0, required_end=10,
        )


@pytest.mark.parametrize(("start", "end"), [(0, 0), (-1, 3), (4, 2), (True, 3)])
def test_invalid_required_ranges_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError):
        check_read_before_edit([], resource="g", required_start=start, required_end=end)


def test_invalid_events_rejected() -> None:
    with pytest.raises(ValueError):
        ToolEvent(-1, "read", "g")
    with pytest.raises(ValueError):
        ToolEvent(0, "read", " ")
    with pytest.raises(ValueError):
        ToolEvent(0, "read", "g", start=0)
    with pytest.raises(ValueError):
        ToolEvent(0, "edit", "g", start=0, end=1)
