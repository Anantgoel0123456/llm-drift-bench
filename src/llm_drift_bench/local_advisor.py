"""Optional local model advice based on a bounded summary of finding categories."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from llm_drift_bench.diagnostics import Finding


def build_advisor_prompt(findings: Sequence[Finding]) -> str:
    """Only disclose counts of known diagnostic categories to the local model."""
    known = {"schema", "invalid_json", "numeric_change", "drift", "suite", "missing_latency"}
    counts = Counter(item.code if item.code in known else "other" for item in findings)
    summary = ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"
    return (
        "You are reviewing an offline LLM comparison. Based only on these observation counts, "
        "suggest up to three practical checks for the developer. Avoid claims about root cause "
        "or model quality. Treat missing timings as unavailable. Observation counts: " + summary
    )
