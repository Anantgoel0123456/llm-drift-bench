"""Offline checks on normalized tool events, without executing tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from llm_drift_bench.scores import EvidenceScore


@dataclass(frozen=True)
class ToolEvent:
    """A read completion or edit invocation in a single ordered run.

    resource identifies an exact resource revision, using the caller's naming
    convention. Read ranges describe content actually returned, in zero-based
    Unicode character offsets with an exclusive end. They are not requested ranges.
    Edit events represent invocation, even if the attempted edit subsequently fails.
    """

    sequence: int
    kind: Literal["read", "edit"]
    resource: str
    success: bool = True
    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer.")
        if self.kind not in ("read", "edit"):
            raise ValueError("kind must be read or edit.")
        if not isinstance(self.resource, str) or not self.resource.strip():
            raise ValueError("resource must be a non-empty revision identifier.")
        if type(self.success) is not bool:
            raise ValueError("success must be boolean.")
        if (self.start is None) != (self.end is None):
            raise ValueError("Both range endpoints must be provided together.")
        if self.start is not None and self.end is not None:
            if (
                type(self.start) is not int or type(self.end) is not int
                or self.start < 0 or self.end <= self.start
            ):
                raise ValueError("Read ranges must be non-empty non-negative integer intervals.")
            if self.kind != "read":
                raise ValueError("Only read events have content ranges.")


@dataclass(frozen=True)
class ToolCheckResult:
    """Evidence verdict, separate from output drift and model correctness."""

    status: Literal["PASS", "FAIL", "UNKNOWN"]
    reason: str
    covered_characters: int
    required_characters: int
    first_edit_sequence: int | None

    def to_score(self) -> EvidenceScore:
        """Expose the verdict as a structured local evidence score."""
        return EvidenceScore(
            name="read_before_edit",
            value=1.0 if self.status == "PASS" else 0.0,
            reason=self.reason,
            category_name="workflow_compliance",
            scoring_failed=self.status == "UNKNOWN",
            metadata={
                "status": self.status,
                "covered_characters": self.covered_characters,
                "required_characters": self.required_characters,
                "first_edit_sequence": self.first_edit_sequence,
            },
        )


def check_read_before_edit(
    events: Sequence[ToolEvent], *, resource: str, required_start: int,
    required_end: int, trace_complete: bool = False,
) -> ToolCheckResult:
    """Check returned content coverage before the first edit attempt in a run.

    The caller must attest that the normalized trace is complete. This function
    cannot verify collector integrity, infer shell commands, resolve path aliases,
    or prove the agent understood the returned content. No filesystem is accessed.
    """
    ToolEvent(0, "read", resource, start=required_start, end=required_end)
    if type(trace_complete) is not bool:
        raise ValueError("trace_complete must be boolean.")
    ordered = sorted(events, key=lambda event: event.sequence)
    if len({event.sequence for event in ordered}) != len(ordered):
        raise ValueError("Event sequences must be unique within one run.")
    first_edit = next((event.sequence for event in ordered if event.kind == "edit"), None)
    intervals: list[tuple[int, int]] = []
    unknown_range = False
    for event in ordered:
        if first_edit is not None and event.sequence >= first_edit:
            break
        if event.kind != "read" or event.resource != resource or not event.success:
            continue
        if event.start is None or event.end is None:
            unknown_range = True
            continue
        start, end = max(required_start, event.start), min(required_end, event.end)
        if start < end:
            intervals.append((start, end))
    covered = 0
    cursor = required_start
    for start, end in sorted(intervals):
        covered += max(0, end - max(start, cursor))
        cursor = max(cursor, end)
    required = required_end - required_start
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    if not trace_complete:
        status, reason = "UNKNOWN", "Trace completeness is not confirmed."
    elif first_edit is None:
        status, reason = "UNKNOWN", "No edit invocation was recorded."
    elif covered == required:
        status, reason = "PASS", "Required content was returned before the first edit invocation."
    elif unknown_range:
        status, reason = "UNKNOWN", "A successful read has unknown returned-content coverage."
    else:
        status, reason = "FAIL", "Required content was not fully returned before the first edit."
    return ToolCheckResult(status, reason, covered, required, first_edit)
