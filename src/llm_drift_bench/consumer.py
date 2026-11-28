"""Downstream consumer validation and rejection-rate creep analysis.

Tracks how often the actual downstream consumer rejects model outputs (business rules,
API parsers, domain invariants) beyond mere JSON syntax or schema compliance.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from llm_drift_bench._file_io import parse_source_json, read_utf8
from llm_drift_bench._matching import index_records, require_same_ids
from llm_drift_bench._report_views import markdown_cell
from llm_drift_bench.models import ModelSample

ConsumerValidator = Callable[[ModelSample], bool | tuple[bool, str | None]]


def _validate_unit_limit(name: str, value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between 0.0 and 1.0.")


@dataclass(frozen=True)
class ConsumerDecision:
    """One downstream accept/reject verdict for a prompt output."""

    prompt_id: str
    accepted: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError("ConsumerDecision.prompt_id must be a non-empty string.")
        if type(self.accepted) is not bool:
            raise ValueError("ConsumerDecision.accepted must be a boolean.")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("ConsumerDecision.reason must be a string or None.")

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "prompt_id": self.prompt_id,
            "accepted": self.accepted,
        }
        if self.reason is not None:
            row["reason"] = self.reason
        return row


@dataclass(frozen=True)
class ConsumerComparison:
    """Comparison of downstream consumer rejection rates between baseline and candidate."""

    total_samples: int
    baseline_rejected: int
    candidate_rejected: int
    baseline_rejection_rate: float
    candidate_rejection_rate: float
    rejection_rate_delta: float
    newly_rejected_ids: tuple[str, ...]
    newly_accepted_ids: tuple[str, ...]
    candidate_reasons: dict[str, str]
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "total_samples": self.total_samples,
            "baseline_rejected": self.baseline_rejected,
            "candidate_rejected": self.candidate_rejected,
            "baseline_rejection_rate": round(self.baseline_rejection_rate, 6),
            "candidate_rejection_rate": round(self.candidate_rejection_rate, 6),
            "rejection_rate_delta": round(self.rejection_rate_delta, 6),
            "newly_rejected_ids": list(self.newly_rejected_ids),
            "newly_accepted_ids": list(self.newly_accepted_ids),
            "candidate_reasons": dict(self.candidate_reasons),
        }


def _parse_decision_row(row: Any, where: str) -> ConsumerDecision:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: consumer decision record must be a JSON object.")
    pid = row.get("prompt_id")
    accepted = row.get("accepted")
    reason = row.get("reason")
    if not isinstance(pid, str) or not pid.strip():
        raise ValueError(f"{where}: missing non-empty string 'prompt_id'.")
    if type(accepted) is not bool:
        raise ValueError(f"{where}: 'accepted' must be a boolean.")
    if reason is not None and not isinstance(reason, str):
        raise ValueError(f"{where}: 'reason' must be a string or null.")
    return ConsumerDecision(prompt_id=pid, accepted=accepted, reason=reason)


def load_consumer_decisions(path: str | Path) -> list[ConsumerDecision]:
    """Load downstream accept/reject decisions from a JSON or JSONL file."""
    source = Path(path)
    raw = read_utf8(source)
    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"{source}: consumer decisions file is empty.")

    decisions: list[ConsumerDecision] = []
    if source.suffix.lower() == ".json":
        parsed = parse_source_json(stripped, source)
        if isinstance(parsed, dict) and isinstance(parsed.get("decisions"), list):
            items = parsed["decisions"]
        elif isinstance(parsed, list):
            items = parsed
        else:
            raise ValueError(
                f"{source}: expected a JSON array or {{'decisions': [...]}} object."
            )
        for idx, item in enumerate(items, start=1):
            decisions.append(_parse_decision_row(item, f"{source} (item {idx})"))
    else:
        for lineno, line in enumerate(raw.split("\n"), start=1):
            if not line.strip(" \t\r"):
                continue
            parsed_line = parse_source_json(line, source, line=lineno)
            decisions.append(_parse_decision_row(parsed_line, f"{source} (line {lineno})"))

    if not decisions:
        raise ValueError(f"{source}: consumer decisions file contains no records.")
    index_records(decisions, f"consumer decisions ({source})")
    return decisions


def compare_consumer_decisions(
    baseline_decisions: Sequence[ConsumerDecision],
    candidate_decisions: Sequence[ConsumerDecision],
    *,
    max_rejection_rate: float | None = None,
    max_rejection_increase: float | None = None,
) -> ConsumerComparison:
    """Compare baseline vs candidate downstream rejection rates and newly rejected prompts."""
    if not baseline_decisions or not candidate_decisions:
        raise ValueError("Both baseline and candidate decision collections must be non-empty.")
    _validate_unit_limit("max_rejection_rate", max_rejection_rate)
    _validate_unit_limit("max_rejection_increase", max_rejection_increase)

    base_map = index_records(baseline_decisions, "baseline consumer decisions")
    cand_map = index_records(candidate_decisions, "candidate consumer decisions")
    require_same_ids(set(base_map), set(cand_map), "Candidate consumer decisions")

    ordered = sorted(base_map)
    total = len(ordered)
    base_rej = sum(1 for pid in ordered if not base_map[pid].accepted)
    cand_rej = sum(1 for pid in ordered if not cand_map[pid].accepted)
    base_rate = base_rej / total
    cand_rate = cand_rej / total
    delta = cand_rate - base_rate

    newly_rejected = tuple(
        pid for pid in ordered if base_map[pid].accepted and not cand_map[pid].accepted
    )
    newly_accepted = tuple(
        pid for pid in ordered if not base_map[pid].accepted and cand_map[pid].accepted
    )
    cand_reasons = {
        pid: cand_map[pid].reason or "rejected"
        for pid in ordered
        if not cand_map[pid].accepted
    }

    failures: list[str] = []
    if max_rejection_rate is not None and cand_rate > max_rejection_rate:
        failures.append(
            f"Candidate rejection rate ({cand_rate:.1%}) exceeded limit "
            f"({max_rejection_rate:.1%})."
        )
    if max_rejection_increase is not None and delta > max_rejection_increase:
        failures.append(
            f"Rejection rate increased by {delta:+.1%} ({base_rate:.1%} -> {cand_rate:.1%}, "
            f"limit +{max_rejection_increase:.1%})."
        )

    return ConsumerComparison(
        total_samples=total,
        baseline_rejected=base_rej,
        candidate_rejected=cand_rej,
        baseline_rejection_rate=base_rate,
        candidate_rejection_rate=cand_rate,
        rejection_rate_delta=delta,
        newly_rejected_ids=newly_rejected,
        newly_accepted_ids=newly_accepted,
        candidate_reasons=cand_reasons,
        passed=len(failures) == 0,
        failure_reasons=tuple(failures),
    )


def _run_validator(sample: ModelSample, validator: ConsumerValidator) -> ConsumerDecision:
    raw = validator(sample)
    if type(raw) is bool:
        return ConsumerDecision(prompt_id=sample.prompt_id, accepted=raw)
    if (
        isinstance(raw, tuple)
        and len(raw) == 2
        and type(raw[0]) is bool
        and (raw[1] is None or isinstance(raw[1], str))
    ):
        return ConsumerDecision(prompt_id=sample.prompt_id, accepted=raw[0], reason=raw[1])
    raise TypeError(
        "Consumer validator must return bool or (bool, str | None), "
        f"got {type(raw).__name__} for prompt {sample.prompt_id!r}."
    )


def evaluate_consumer_rejection(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
    validator: ConsumerValidator,
    *,
    max_rejection_rate: float | None = None,
    max_rejection_increase: float | None = None,
) -> ConsumerComparison:
    """Run a user-supplied downstream consumer validator on baseline and candidate captures."""
    if not callable(validator):
        raise TypeError("validator must be callable.")
    if not baseline_samples or not candidate_samples:
        raise ValueError("Both baseline and candidate sample collections must be non-empty.")

    base_map = index_records(baseline_samples, "baseline samples")
    cand_map = index_records(candidate_samples, "candidate samples")
    require_same_ids(set(base_map), set(cand_map), "Candidate samples")

    ordered = sorted(base_map)
    base_decisions = [_run_validator(base_map[pid], validator) for pid in ordered]
    cand_decisions = [_run_validator(cand_map[pid], validator) for pid in ordered]
    return compare_consumer_decisions(
        base_decisions,
        cand_decisions,
        max_rejection_rate=max_rejection_rate,
        max_rejection_increase=max_rejection_increase,
    )


def load_validator_callable(spec: str) -> ConsumerValidator:
    """Import a downstream validator callable from 'module.path:callable_name'."""
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"Validator spec must use 'module.path:callable_name' syntax, got {spec!r}."
        )
    mod_name, attr_name = spec.split(":", 1)
    mod_name = mod_name.strip()
    attr_name = attr_name.strip()
    if not mod_name or not attr_name:
        raise ValueError(
            f"Validator spec requires non-empty module and callable name, got {spec!r}."
        )
    module = importlib.import_module(mod_name)
    fn = getattr(module, attr_name, None)
    if not callable(fn):
        raise ValueError(f"Resolved object {spec!r} is not callable.")
    return fn  # type: ignore[no-any-return]


def format_consumer_text(comparison: ConsumerComparison) -> str:
    """Format a ConsumerComparison as a clear human-readable summary."""
    status = "PASSED" if comparison.passed else "FAILED"
    lines = [
        f"Downstream Consumer Rejection Check [{status}] ({comparison.total_samples} prompt(s))",
        f"  Baseline rejected  : {comparison.baseline_rejected}/{comparison.total_samples} "
        f"({comparison.baseline_rejection_rate:.1%})",
        f"  Candidate rejected : {comparison.candidate_rejected}/{comparison.total_samples} "
        f"({comparison.candidate_rejection_rate:.1%}, "
        f"delta {comparison.rejection_rate_delta:+.1%})",
        f"  Newly rejected     : {len(comparison.newly_rejected_ids)} | "
        f"Newly accepted: {len(comparison.newly_accepted_ids)}",
    ]
    if comparison.newly_rejected_ids:
        lines.append("")
        lines.append("Newly rejected prompts:")
        for pid in comparison.newly_rejected_ids:
            reason = comparison.candidate_reasons.get(pid, "rejected")
            lines.append(f"  - {pid}: {reason}")
    if comparison.failure_reasons:
        lines.append("")
        lines.append("Threshold failures:")
        for msg in comparison.failure_reasons:
            lines.append(f"  - {msg}")
    return "\n".join(lines) + "\n"


def format_consumer_markdown(comparison: ConsumerComparison) -> str:
    """Format a ConsumerComparison as GitHub-flavored Markdown."""
    status = "PASSED" if comparison.passed else "FAILED"
    lines = [
        f"# Downstream Consumer Rejection Check: {status}",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Rejected prompts | {comparison.baseline_rejected}/{comparison.total_samples} "
        f"| {comparison.candidate_rejected}/{comparison.total_samples} "
        f"| {comparison.candidate_rejected - comparison.baseline_rejected:+d} |",
        f"| Rejection rate | {comparison.baseline_rejection_rate:.1%} "
        f"| {comparison.candidate_rejection_rate:.1%} "
        f"| {comparison.rejection_rate_delta:+.1%} |",
    ]
    if comparison.newly_rejected_ids:
        lines += [
            "",
            "## Newly rejected prompts",
            "",
            "| Prompt ID | Candidate rejection reason |",
            "| --- | --- |",
        ]
        for pid in comparison.newly_rejected_ids:
            reason = comparison.candidate_reasons.get(pid, "rejected")
            lines.append(f"| {markdown_cell(pid)} | {markdown_cell(reason)} |")
    if comparison.failure_reasons:
        lines += ["", "## Threshold failures", ""]
        lines.extend(f"- {markdown_cell(msg)}" for msg in comparison.failure_reasons)
    return "\n".join(lines) + "\n"


def export_consumer_html(
    comparison: ConsumerComparison,
    output_path: str | Path | None = None,
) -> str:
    """Export a standalone script-free offline HTML report for a ConsumerComparison."""
    status = "PASSED" if comparison.passed else "FAILED"
    badge_cls = "pass" if comparison.passed else "fail"
    rej_rows = ""
    if comparison.newly_rejected_ids:
        body = "".join(
            f"<tr><td>{escape(pid)}</td>"
            f"<td>{escape(comparison.candidate_reasons.get(pid, 'rejected'))}</td></tr>"
            for pid in comparison.newly_rejected_ids
        )
        rej_rows = (
            "<h2>Newly Rejected Prompts</h2>"
            "<table><thead><tr><th>Prompt ID</th><th>Candidate Reason</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )
    fail_html = ""
    if comparison.failure_reasons:
        items = "".join(f"<li>{escape(msg)}</li>" for msg in comparison.failure_reasons)
        fail_html = (
            f"<section class='failures'><h2>Threshold Failures</h2><ul>{items}</ul></section>"
        )
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>LLM Drift Bench — Downstream Consumer Rejection</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,sans-serif;background:#0b1320;color:#e6edf7;"
        "margin:0;padding:24px;line-height:1.5}"
        "main{max-width:900px;margin:0 auto}h1,h2{color:#f8fafc}.muted{color:#94a3b8}"
        ".badge{display:inline-block;padding:4px 10px;border-radius:6px;font-weight:700}"
        ".pass{background:#064e3b;color:#6ee7b7}.fail{background:#4c0519;color:#fda4af}"
        "table{width:100%;border-collapse:collapse;background:#111e32;margin-top:12px}"
        "th,td{padding:10px 12px;border-bottom:1px solid #26354d;text-align:left}"
        "th{color:#7dd3fc}.failures{margin-top:16px;padding:12px;background:#3b0a17;"
        "border:1px solid #881337;border-radius:8px}"
        "</style></head><body><main>"
        "<p class='muted'>llm-drift-bench / downstream consumer validation</p>"
        "<h1>Downstream Consumer Rejection Check "
        f"<span class='badge {badge_cls}'>{status}</span></h1>"
        "<table><thead><tr><th>Metric</th><th>Baseline</th>"
        "<th>Candidate</th><th>Delta</th></tr></thead><tbody>"
        f"<tr><td>Rejected prompts</td>"
        f"<td>{comparison.baseline_rejected}/{comparison.total_samples}</td>"
        f"<td>{comparison.candidate_rejected}/{comparison.total_samples}</td>"
        f"<td>{comparison.candidate_rejected - comparison.baseline_rejected:+d}</td></tr>"
        f"<tr><td>Rejection rate</td>"
        f"<td>{comparison.baseline_rejection_rate:.1%}</td>"
        f"<td>{comparison.candidate_rejection_rate:.1%}</td>"
        f"<td>{comparison.rejection_rate_delta:+.1%}</td></tr>"
        "</tbody></table>"
        f"{rej_rows}{fail_html}</main></body></html>\n"
    )
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
    return html


def export_consumer_pr_comment(comparison: ConsumerComparison) -> str:
    """Render a compact GitHub PR comment summarizing downstream consumer rejection drift."""
    verdict = "PASS" if comparison.passed else "FAIL"
    lines = [
        "<!-- llm-drift-bench:consumer -->",
        f"### Downstream Consumer Validation: **{verdict}**",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| :--- | ---: | ---: | ---: |",
        f"| Rejected prompts | {comparison.baseline_rejected}/{comparison.total_samples} | "
        f"{comparison.candidate_rejected}/{comparison.total_samples} | "
        f"{comparison.candidate_rejected - comparison.baseline_rejected:+d} |",
        f"| Rejection rate | {comparison.baseline_rejection_rate:.1%} | "
        f"{comparison.candidate_rejection_rate:.1%} | "
        f"{comparison.rejection_rate_delta:+.1%} |",
    ]
    if comparison.newly_rejected_ids or comparison.failure_reasons:
        lines += ["", "<details><summary>Details &amp; newly rejected prompts</summary>", ""]
        for reason in comparison.failure_reasons:
            lines.append(f"- {markdown_cell(reason)}")
        for pid in comparison.newly_rejected_ids[:5]:
            reason_note = comparison.candidate_reasons.get(pid, "")
            suffix = f": {markdown_cell(reason_note)}" if reason_note else ""
            lines.append(f"- Newly rejected `{markdown_cell(pid)}`{suffix}")
        if len(comparison.newly_rejected_ids) > 5:
            extra = len(comparison.newly_rejected_ids) - 5
            lines.append(f"- _...and {extra} more newly rejected prompt(s)._")
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"

