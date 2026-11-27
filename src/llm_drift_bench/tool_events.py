"""Offline checks on normalized tool events, without executing tools."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from llm_drift_bench.scores import EvidenceScore

_SECRET_PATTERNS = (
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}\b")),
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{24,}\b")),
    (
        "jwt_token",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    (
        "database_url",
        re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^/\s:@]*:[^/\s@]+@"),
    ),
    ("private_key_pem", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)


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


def check_max_tool_retries(
    events: Sequence[ToolEvent],
    *,
    max_failed_calls: int = 3,
    trace_complete: bool = False,
) -> ToolCheckResult:
    """Verify an agent did not exceed a bounded count of failed tool invocations."""
    if isinstance(max_failed_calls, bool) or max_failed_calls < 0:
        raise ValueError("max_failed_calls must be a non-negative integer.")
    if type(trace_complete) is not bool:
        raise ValueError("trace_complete must be boolean.")
    failed = sum(1 for event in events if not event.success)
    if failed > max_failed_calls:
        return ToolCheckResult(
            "FAIL",
            f"Recorded {failed} failed tool call(s), exceeding limit {max_failed_calls}.",
            0,
            0,
            None,
        )
    if not trace_complete:
        return ToolCheckResult("UNKNOWN", "Trace completeness is not confirmed.", 0, 0, None)
    return ToolCheckResult(
        "PASS",
        f"Recorded {failed} failed tool call(s) within limit {max_failed_calls}.",
        0,
        0,
        None,
    )


def check_no_secret_in_tool_args(
    tool_args: Sequence[Any],
    *,
    trace_complete: bool = False,
) -> ToolCheckResult:
    """Scan serialized tool call arguments offline for common leaked credential patterns.

    Never echoes matched secret text in the result reason.
    """
    if type(trace_complete) is not bool:
        raise ValueError("trace_complete must be boolean.")
    matched_kinds: list[str] = []
    for item in tool_args:
        text = item if isinstance(item, str) else str(item)
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(text) is not None and label not in matched_kinds:
                matched_kinds.append(label)
    if matched_kinds:
        kinds = ", ".join(sorted(matched_kinds))
        return ToolCheckResult(
            "FAIL",
            f"Detected potential credential pattern(s) in tool arguments: {kinds}.",
            0,
            0,
            None,
        )
    if not trace_complete:
        return ToolCheckResult("UNKNOWN", "Trace completeness is not confirmed.", 0, 0, None)
    return ToolCheckResult(
        "PASS",
        "No common credential patterns detected in supplied tool arguments.",
        0,
        0,
        None,
    )


def check_max_tool_calls(
    events: Sequence[ToolEvent],
    *,
    max_calls: int,
    trace_complete: bool = False,
) -> ToolCheckResult:
    """Verify total tool invocations stay within an explicit step budget."""
    if type(max_calls) is not int or max_calls < 0:
        raise ValueError("max_calls must be a non-negative integer.")
    if type(trace_complete) is not bool:
        raise ValueError("trace_complete must be boolean.")
    total = len(events)
    if total > max_calls:
        return ToolCheckResult(
            "FAIL",
            f"Recorded {total} tool call(s), exceeding call budget {max_calls}.",
            0,
            0,
            None,
        )
    if not trace_complete:
        return ToolCheckResult("UNKNOWN", "Trace completeness is not confirmed.", 0, 0, None)
    return ToolCheckResult(
        "PASS",
        f"Recorded {total} tool call(s) within call budget {max_calls}.",
        0,
        0,
        None,
    )


def check_allowed_resources(
    events: Sequence[ToolEvent],
    *,
    allowed_prefixes: Sequence[str],
    trace_complete: bool = False,
) -> ToolCheckResult:
    """Verify every recorded tool event targeted a resource within allowed prefixes."""
    cleaned_prefixes = tuple(p for p in allowed_prefixes if isinstance(p, str) and p.strip())
    if not cleaned_prefixes or len(cleaned_prefixes) != len(allowed_prefixes):
        raise ValueError("allowed_prefixes must contain one or more non-empty strings.")
    if type(trace_complete) is not bool:
        raise ValueError("trace_complete must be boolean.")

    disallowed: list[str] = []
    for event in events:
        if (
            not any(event.resource.startswith(prefix) for prefix in cleaned_prefixes)
            and event.resource not in disallowed
        ):
            disallowed.append(event.resource)
    if disallowed:
        preview = ", ".join(disallowed[:3])
        extra = f" (+{len(disallowed) - 3} more)" if len(disallowed) > 3 else ""
        return ToolCheckResult(
            "FAIL",
            f"Tool touched disallowed resource(s) outside allowed prefixes: {preview}{extra}.",
            0,
            0,
            None,
        )
    if not trace_complete:
        return ToolCheckResult("UNKNOWN", "Trace completeness is not confirmed.", 0, 0, None)
    return ToolCheckResult(
        "PASS",
        f"All {len(events)} tool event(s) stayed within allowed resource prefixes.",
        0,
        0,
        None,
    )


@dataclass(frozen=True)
class ToolTraceDiff:
    """Comparison of agent tool-usage behavior between baseline and candidate traces."""

    baseline_calls: int
    candidate_calls: int
    call_delta: int
    baseline_reads: int
    candidate_reads: int
    baseline_edits: int
    candidate_edits: int
    baseline_failed: int
    candidate_failed: int
    failed_delta: int
    added_resources: tuple[str, ...]
    removed_resources: tuple[str, ...]
    added_edit_resources: tuple[str, ...]
    action_jaccard: float
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "baseline_calls": self.baseline_calls,
            "candidate_calls": self.candidate_calls,
            "call_delta": self.call_delta,
            "baseline_reads": self.baseline_reads,
            "candidate_reads": self.candidate_reads,
            "baseline_edits": self.baseline_edits,
            "candidate_edits": self.candidate_edits,
            "baseline_failed": self.baseline_failed,
            "candidate_failed": self.candidate_failed,
            "failed_delta": self.failed_delta,
            "added_resources": list(self.added_resources),
            "removed_resources": list(self.removed_resources),
            "added_edit_resources": list(self.added_edit_resources),
            "action_jaccard": round(self.action_jaccard, 6),
        }


def compare_tool_traces(
    baseline_events: Sequence[ToolEvent],
    candidate_events: Sequence[ToolEvent],
    *,
    max_call_increase: int | None = None,
    max_failed_increase: int | None = None,
    fail_on_new_edit_resources: bool = False,
) -> ToolTraceDiff:
    """Compare two normalized agent tool traces for call-count, retry, and scope drift."""
    if max_call_increase is not None and (
        type(max_call_increase) is not int or max_call_increase < 0
    ):
        raise ValueError("max_call_increase must be a non-negative integer.")
    if max_failed_increase is not None and (
        type(max_failed_increase) is not int or max_failed_increase < 0
    ):
        raise ValueError("max_failed_increase must be a non-negative integer.")
    if type(fail_on_new_edit_resources) is not bool:
        raise ValueError("fail_on_new_edit_resources must be boolean.")

    b_calls = len(baseline_events)
    c_calls = len(candidate_events)
    b_reads = sum(1 for e in baseline_events if e.kind == "read")
    c_reads = sum(1 for e in candidate_events if e.kind == "read")
    b_edits = sum(1 for e in baseline_events if e.kind == "edit")
    c_edits = sum(1 for e in candidate_events if e.kind == "edit")
    b_failed = sum(1 for e in baseline_events if not e.success)
    c_failed = sum(1 for e in candidate_events if not e.success)

    call_delta = c_calls - b_calls
    failed_delta = c_failed - b_failed

    b_resources = {e.resource for e in baseline_events}
    c_resources = {e.resource for e in candidate_events}
    added_resources = tuple(sorted(c_resources - b_resources))
    removed_resources = tuple(sorted(b_resources - c_resources))

    b_edit_res = {e.resource for e in baseline_events if e.kind == "edit"}
    c_edit_res = {e.resource for e in candidate_events if e.kind == "edit"}
    added_edit_resources = tuple(sorted(c_edit_res - b_edit_res))

    b_actions = {(e.kind, e.resource) for e in baseline_events}
    c_actions = {(e.kind, e.resource) for e in candidate_events}
    union = b_actions | c_actions
    action_jaccard = (len(b_actions & c_actions) / len(union)) if union else 1.0

    failures: list[str] = []
    if max_call_increase is not None and call_delta > max_call_increase:
        failures.append(
            f"Tool call increase ({call_delta:+d}) exceeded limit (+{max_call_increase})."
        )
    if max_failed_increase is not None and failed_delta > max_failed_increase:
        failures.append(
            f"Failed tool call increase ({failed_delta:+d}) exceeded limit "
            f"(+{max_failed_increase})."
        )
    if fail_on_new_edit_resources and added_edit_resources:
        preview = ", ".join(added_edit_resources[:3])
        extra = (
            f" (+{len(added_edit_resources) - 3} more)"
            if len(added_edit_resources) > 3
            else ""
        )
        failures.append(
            f"Candidate edited new resource(s) not edited in baseline: {preview}{extra}."
        )

    return ToolTraceDiff(
        baseline_calls=b_calls,
        candidate_calls=c_calls,
        call_delta=call_delta,
        baseline_reads=b_reads,
        candidate_reads=c_reads,
        baseline_edits=b_edits,
        candidate_edits=c_edits,
        baseline_failed=b_failed,
        candidate_failed=c_failed,
        failed_delta=failed_delta,
        added_resources=added_resources,
        removed_resources=removed_resources,
        added_edit_resources=added_edit_resources,
        action_jaccard=action_jaccard,
        passed=len(failures) == 0,
        failure_reasons=tuple(failures),
    )


def format_tool_trace_diff_text(diff: ToolTraceDiff) -> str:
    """Render a ToolTraceDiff as a concise terminal report."""
    verdict = "PASSED" if diff.passed else "FAILED"
    lines = [
        f"Agent Tool-Trace Drift Comparison [{verdict}]",
        f"  Total calls     : {diff.baseline_calls} -> {diff.candidate_calls} "
        f"({diff.call_delta:+d})",
        f"  Reads / Edits   : {diff.baseline_reads}/{diff.baseline_edits} -> "
        f"{diff.candidate_reads}/{diff.candidate_edits}",
        f"  Failed calls    : {diff.baseline_failed} -> {diff.candidate_failed} "
        f"({diff.failed_delta:+d})",
        f"  Action Jaccard  : {diff.action_jaccard:.4f}",
    ]
    if diff.added_resources:
        lines.append(f"  Added resources : {', '.join(diff.added_resources)}")
    if diff.removed_resources:
        lines.append(f"  Removed         : {', '.join(diff.removed_resources)}")
    if diff.added_edit_resources:
        lines.append(f"  New edit targets: {', '.join(diff.added_edit_resources)}")
    if diff.failure_reasons:
        lines.append("Failure reasons:")
        for reason in diff.failure_reasons:
            lines.append(f"  - {reason}")
    return "\n".join(lines) + "\n"
