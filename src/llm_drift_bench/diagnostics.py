"""Explain the recorded verdict of an offline benchmark report."""

from __future__ import annotations

from dataclasses import dataclass

from llm_drift_bench.conversation import ConversationDriftReport
from llm_drift_bench.models import DriftBenchmarkReport


@dataclass(frozen=True)
class Finding:
    """One observed condition and a suggested next check."""

    code: str
    prompt_id: str | None
    evidence: str
    suggestion: str


_RULE_ADVICE = {
    "invalid_schema": "Correct the schema definition and rerun both captures against it.",
    "unsupported_keyword": "Replace unsupported JSON Schema keywords with the supported subset.",
    "additionalProperties": "Review unexpected keys against the allowed object properties.",
    "enum": "Compare the generated label with the allowed labels and their casing.",
    "const": "Check whether the output drifted from the required constant value.",
    "type": "Check serialization of the field against its declared JSON type.",
    "required": "Restore the required field in the extraction prompt or output mapping.",
    "minimum": "Verify numeric scaling, sign, and lower-bound enforcement in the prompt.",
    "maximum": "Verify numeric scaling and upper-bound enforcement in the prompt.",
    "minLength": "Check whether the model truncated or omitted required text content.",
    "maxLength": "Tighten verbosity constraints or max_tokens to keep strings within maxLength.",
    "pattern": "Compare the formatted string against the required regex pattern.",
    "pattern_limit": "Trim oversized field output before regex matching (16,384-char cap).",
    "minItems": "Check whether the model returned an empty or truncated list.",
    "maxItems": "Constrain list cardinality in the prompt to stay within maxItems.",
}


def diagnose_report(report: DriftBenchmarkReport) -> list[Finding]:
    """Return actionable observations without recomputing the benchmark verdict."""
    findings: list[Finding] = []
    conformity_failed = not report.passed and report.candidate_json_conformity_rate < 1.0
    ordered = sorted(report.results, key=lambda row: (-row.composite_drift_score, row.prompt_id))
    for result in ordered:
        validation = result.candidate_validation
        schema_failed = validation.is_valid_json and bool(validation.issues)
        json_failed_in_failing_suite = conformity_failed and not validation.is_valid_json
        if not result.drifted and not schema_failed and not json_failed_in_failing_suite:
            continue

        if not validation.is_valid_json and (
            result.baseline_validation.is_valid_json or json_failed_in_failing_suite
        ):
            evidence = (
                "Baseline parsed as JSON and candidate did not."
                if result.baseline_validation.is_valid_json
                else "Candidate output did not parse as JSON."
            )
            findings.append(
                Finding(
                    "invalid_json",
                    result.prompt_id,
                    evidence,
                    "Check the capture for truncation, malformed JSON or duplicate keys.",
                )
            )
            continue

        specific_emitted = False
        if validation.issues:
            for issue in validation.issues:
                suggestion = _RULE_ADVICE.get(
                    issue.rule, "Compare the field with the saved schema."
                )
                findings.append(
                    Finding("schema", result.prompt_id, f"{issue.path}: {issue.rule}", suggestion)
                )
            specific_emitted = True

        if (
            result.drifted
            and report.thresholds is not None
            and result.json_numeric_drift is not None
            and result.json_numeric_drift > report.thresholds.max_numeric_relative_drift
        ):
            findings.append(
                Finding(
                    "numeric_change",
                    result.prompt_id,
                    f"Shared numeric fields have relative drift {result.json_numeric_drift:.4f}.",
                    "Compare units, decimal placement and rounding against the source record.",
                )
            )
            specific_emitted = True

        if result.drifted and not specific_emitted:
            findings.append(
                Finding(
                    "drift",
                    result.prompt_id,
                    f"Composite drift {result.composite_drift_score:.4f}",
                    "Compare the saved baseline and candidate responses for this prompt.",
                )
            )
    if not report.passed:
        for reason in report.failure_reasons:
            findings.append(
                Finding("suite", None, reason, "Review this gate against the captures.")
            )
    if report.baseline_p95_latency_ms is None or report.candidate_p95_latency_ms is None:
        findings.append(
            Finding(
                "missing_latency",
                None,
                "P95 latency comparison unavailable.",
                "Capture latency for every baseline and candidate response.",
            )
        )
    return findings


def diagnose_conversation_report(report: ConversationDriftReport) -> list[Finding]:
    """Explain a ConversationDriftReport with actionable per-conversation root causes."""
    findings: list[Finding] = []
    ordered = sorted(
        report.conversations,
        key=lambda item: (-item.max_turn_drift, item.conversation_id),
    )
    for conv in ordered:
        if not conv.drifted:
            continue
        emitted = False
        if conv.refusal_flip_turns:
            turns_str = ", ".join(f"turn {idx}" for idx in conv.refusal_flip_turns)
            findings.append(
                Finding(
                    "refusal_flip",
                    conv.conversation_id,
                    f"Candidate refused previously answered prompt(s) at {turns_str}.",
                    "Inspect system prompt safety framing and multi-turn context wording.",
                )
            )
            emitted = True

        json_drop_turns = [
            t.turn_index
            for t in conv.turn_results
            if t.json_valid_baseline and not t.json_valid_candidate
        ]
        if json_drop_turns:
            turns_str = ", ".join(f"turn {idx}" for idx in json_drop_turns)
            findings.append(
                Finding(
                    "json_drop",
                    conv.conversation_id,
                    f"Candidate lost valid JSON formatting at {turns_str}.",
                    "Reinforce JSON output reminders or enable json_mode on later turns.",
                )
            )
            emitted = True

        if conv.turn_count_delta != 0:
            findings.append(
                Finding(
                    "turn_count_mismatch",
                    conv.conversation_id,
                    f"Assistant turn count changed ({conv.baseline_assistant_turns} -> "
                    f"{conv.candidate_assistant_turns}).",
                    "Check stop sequences, early termination, or unexpected follow-up loops.",
                )
            )
            emitted = True

        if conv.escalation_slope is not None and conv.escalation_slope > 0.15:
            findings.append(
                Finding(
                    "snowball_escalation",
                    conv.conversation_id,
                    f"Drift escalated across turns (slope {conv.escalation_slope:+.4f}, "
                    f"first divergence at turn {conv.first_divergence_turn}).",
                    "Inspect the earliest diverging turn with 'check-conversations --show-diffs'.",
                )
            )
            emitted = True

        if not emitted:
            findings.append(
                Finding(
                    "turn_drift",
                    conv.conversation_id,
                    f"Max turn drift {conv.max_turn_drift:.4f} "
                    f"(first divergence at turn {conv.first_divergence_turn}).",
                    "Run 'check-conversations --show-diffs' to inspect turn-by-turn changes.",
                )
            )

    if not report.passed:
        for reason in report.failure_reasons:
            findings.append(
                Finding("suite", None, reason, "Review conversation gates and transcripts.")
            )
    return findings
