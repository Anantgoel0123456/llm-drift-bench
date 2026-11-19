"""Explain the recorded verdict of an offline benchmark report."""

from __future__ import annotations

from dataclasses import dataclass

from llm_drift_bench.models import DriftBenchmarkReport


@dataclass(frozen=True)
class Finding:
    """One observed condition and a suggested next check."""

    code: str
    prompt_id: str | None
    evidence: str
    suggestion: str


_RULE_ADVICE = {
    "additionalProperties": "Review unexpected keys against the allowed object properties.",
    "enum": "Compare the generated label with the allowed labels and their casing.",
    "type": "Check serialization of the field against its declared JSON type.",
    "required": "Restore the required field in the extraction prompt or output mapping.",
}


def diagnose_report(report: DriftBenchmarkReport) -> list[Finding]:
    """Return actionable observations without recomputing the benchmark verdict."""
    findings: list[Finding] = []
    for result in report.results:
        if not result.drifted:
            continue
        validation = result.candidate_validation
        if result.baseline_validation.is_valid_json and not validation.is_valid_json:
            findings.append(Finding(
                "invalid_json", result.prompt_id, "Baseline parsed as JSON and candidate did not.",
                "Inspect the captured response for truncation, malformed JSON or duplicate keys.",
            ))
            continue
        if validation.issues:
            for issue in validation.issues:
                suggestion = _RULE_ADVICE.get(issue.rule, "Compare the field with the saved schema.")
                findings.append(Finding("schema", result.prompt_id,
                                        f"{issue.path}: {issue.rule}", suggestion))
            continue
        findings.append(Finding(
            "drift", result.prompt_id, f"Composite drift {result.composite_drift_score:.4f}",
            "Compare the saved baseline and candidate responses for this prompt.",
        ))
    return findings
