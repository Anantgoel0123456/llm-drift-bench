"""Multi-turn conversation trajectory drift analyzer.

Compares stateful multi-turn assistant trajectories turn-by-turn between baseline and
candidate captures, detecting snowballing divergence (escalation slope), earliest
divergence turn, turn-count mismatches, JSON adherence drops, and refusal flips.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Literal

from llm_drift_bench._file_io import parse_source_json, read_utf8
from llm_drift_bench._matching import require_same_ids
from llm_drift_bench._report_views import markdown_cell
from llm_drift_bench.diffing import PromptDiff, diff_sample_pair, format_diffs_text
from llm_drift_bench.metrics import (
    _MAX_EDIT_DISTANCE_CHARS,
    compute_composite_drift_score,
    jaccard_similarity,
    json_exact_field_match,
    json_numeric_drift,
    json_structural_similarity,
    ngram_overlap_similarity,
    normalized_levenshtein_similarity,
)
from llm_drift_bench.models import ModelSample
from llm_drift_bench.quality_checks import is_likely_refusal
from llm_drift_bench.validator import extract_json_payload

TurnRole = Literal["system", "user", "assistant", "tool"]
_VALID_ROLES: tuple[TurnRole, ...] = ("system", "user", "assistant", "tool")


def _validate_unit_limit(name: str, val: float | None) -> None:
    if val is not None and (
        isinstance(val, bool)
        or not isinstance(val, (int, float))
        or not math.isfinite(val)
        or not 0.0 <= val <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between 0.0 and 1.0.")


@dataclass(frozen=True)
class ConversationTurn:
    """Single message turn inside a captured multi-turn conversation."""

    role: TurnRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"ConversationTurn.role must be one of {_VALID_ROLES}, got {self.role!r}."
            )
        if not isinstance(self.content, str):
            raise ValueError("ConversationTurn.content must be a string.")
        if not isinstance(self.metadata, dict):
            raise ValueError("ConversationTurn.metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.metadata:
            row["metadata"] = dict(self.metadata)
        return row


@dataclass(frozen=True)
class ConversationRecord:
    """One multi-turn conversation transcript identified by a unique conversation_id."""

    conversation_id: str
    turns: tuple[ConversationTurn, ...]
    model_name: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str) or not self.conversation_id.strip():
            raise ValueError("ConversationRecord.conversation_id must be a non-empty string.")
        if not self.turns:
            raise ValueError("ConversationRecord.turns must be non-empty.")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("ConversationRecord.model_name must be a non-empty string.")

    def assistant_turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(turn for turn in self.turns if turn.role == "assistant")

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "conversation_id": self.conversation_id,
            "model_name": self.model_name,
            "turns": [t.to_dict() for t in self.turns],
        }
        if self.metadata:
            row["metadata"] = dict(self.metadata)
        return row


def _parse_conversation_row(row: Any, where: str) -> ConversationRecord:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: conversation record must be a JSON object.")
    cid = row.get("conversation_id")
    if not isinstance(cid, str) or not cid.strip():
        raise ValueError(f"{where}: missing non-empty string 'conversation_id'.")
    raw_turns = row.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError(f"{where}: 'turns' must be a non-empty list.")
    parsed_turns: list[ConversationTurn] = []
    for idx, item in enumerate(raw_turns, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{where}: turn #{idx} must be a JSON object.")
        role = item.get("role")
        content = item.get("content")
        meta = item.get("metadata", {})
        if role not in _VALID_ROLES:
            raise ValueError(f"{where}: turn #{idx} invalid role {role!r}.")
        if not isinstance(content, str):
            raise ValueError(f"{where}: turn #{idx} 'content' must be a string.")
        if not isinstance(meta, dict):
            raise ValueError(f"{where}: turn #{idx} 'metadata' must be a JSON object.")
        parsed_turns.append(ConversationTurn(role=role, content=content, metadata=dict(meta)))
    model = row.get("model_name", "unknown")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{where}: 'model_name' must be a non-empty string.")
    meta_top = row.get("metadata", {})
    if not isinstance(meta_top, dict):
        raise ValueError(f"{where}: 'metadata' must be a JSON object.")
    return ConversationRecord(
        conversation_id=cid,
        turns=tuple(parsed_turns),
        model_name=model,
        metadata=dict(meta_top),
    )


def _index_conversations(
    records: Sequence[ConversationRecord], label: str
) -> dict[str, ConversationRecord]:
    indexed: dict[str, ConversationRecord] = {}
    for rec in records:
        if rec.conversation_id in indexed:
            raise ValueError(
                f"{label} contains duplicate conversation_id {rec.conversation_id!r}."
            )
        indexed[rec.conversation_id] = rec
    return indexed


def load_conversations(path: str | Path) -> list[ConversationRecord]:
    """Load multi-turn conversation records from a JSON or JSONL file."""
    source = Path(path)
    raw = read_utf8(source)
    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"{source}: conversation file is empty.")

    records: list[ConversationRecord] = []
    if source.suffix.lower() == ".json":
        parsed = parse_source_json(stripped, source)
        if isinstance(parsed, dict) and isinstance(parsed.get("conversations"), list):
            items = parsed["conversations"]
        elif isinstance(parsed, list):
            items = parsed
        else:
            raise ValueError(
                f"{source}: expected a JSON array or {{'conversations': [...]}} object."
            )
        for idx, item in enumerate(items, start=1):
            records.append(_parse_conversation_row(item, f"{source} (item {idx})"))
    else:
        for lineno, line in enumerate(raw.split("\n"), start=1):
            if not line.strip(" \t\r"):
                continue
            parsed_line = parse_source_json(line, source, line=lineno)
            records.append(_parse_conversation_row(parsed_line, f"{source} (line {lineno})"))

    if not records:
        raise ValueError(f"{source}: conversation file contains no records.")
    _index_conversations(records, f"conversations ({source})")
    return records


@dataclass(frozen=True)
class TurnDriftResult:
    """Drift metrics for one aligned assistant turn inside a conversation."""

    turn_index: int
    composite_drift: float
    token_jaccard: float
    levenshtein_similarity: float
    refusal_flip: bool
    json_valid_baseline: bool
    json_valid_candidate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "composite_drift": round(self.composite_drift, 6),
            "token_jaccard": round(self.token_jaccard, 6),
            "levenshtein_similarity": round(self.levenshtein_similarity, 6),
            "refusal_flip": self.refusal_flip,
            "json_valid_baseline": self.json_valid_baseline,
            "json_valid_candidate": self.json_valid_candidate,
        }


@dataclass(frozen=True)
class ConversationDriftResult:
    """Trajectory-level drift summary for one matched conversation."""

    conversation_id: str
    baseline_assistant_turns: int
    candidate_assistant_turns: int
    turn_count_delta: int
    compared_turns: int
    mean_turn_drift: float
    max_turn_drift: float
    first_divergence_turn: int | None
    final_turn_drift: float | None
    escalation_slope: float | None
    refusal_flip_turns: tuple[int, ...]
    turn_results: tuple[TurnDriftResult, ...]
    drifted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "drifted": self.drifted,
            "baseline_assistant_turns": self.baseline_assistant_turns,
            "candidate_assistant_turns": self.candidate_assistant_turns,
            "turn_count_delta": self.turn_count_delta,
            "compared_turns": self.compared_turns,
            "mean_turn_drift": round(self.mean_turn_drift, 6),
            "max_turn_drift": round(self.max_turn_drift, 6),
            "first_divergence_turn": self.first_divergence_turn,
            "final_turn_drift": (
                round(self.final_turn_drift, 6) if self.final_turn_drift is not None else None
            ),
            "escalation_slope": (
                round(self.escalation_slope, 6) if self.escalation_slope is not None else None
            ),
            "refusal_flip_turns": list(self.refusal_flip_turns),
            "turn_results": [t.to_dict() for t in self.turn_results],
        }


@dataclass(frozen=True)
class ConversationDriftReport:
    """Aggregate multi-turn conversation trajectory drift report."""

    total_conversations: int
    drifted_conversations: int
    mean_trajectory_drift: float
    max_trajectory_drift: float
    mean_final_turn_drift: float | None
    turn_count_mismatches: int
    total_refusal_flip_turns: int
    conversations: tuple[ConversationDriftResult, ...]
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "total_conversations": self.total_conversations,
            "drifted_conversations": self.drifted_conversations,
            "mean_trajectory_drift": round(self.mean_trajectory_drift, 6),
            "max_trajectory_drift": round(self.max_trajectory_drift, 6),
            "mean_final_turn_drift": (
                round(self.mean_final_turn_drift, 6)
                if self.mean_final_turn_drift is not None
                else None
            ),
            "turn_count_mismatches": self.turn_count_mismatches,
            "total_refusal_flip_turns": self.total_refusal_flip_turns,
            "conversations": [c.to_dict() for c in self.conversations],
        }


def _score_turn_pair(
    turn_index: int,
    base_text: str,
    cand_text: str,
    *,
    long_text_mode: Literal["error", "skip_levenshtein"],
) -> TurnDriftResult:
    tok_jac = jaccard_similarity(base_text, cand_text)
    ngram_sim = ngram_overlap_similarity(base_text, cand_text)
    if (
        long_text_mode == "skip_levenshtein"
        and (
            len(base_text) > _MAX_EDIT_DISTANCE_CHARS
            or len(cand_text) > _MAX_EDIT_DISTANCE_CHARS
        )
    ):
        lev_sim = 0.5 * (tok_jac + ngram_sim)
    else:
        lev_sim = normalized_levenshtein_similarity(base_text, cand_text)

    ok_a, payload_a, _ = extract_json_payload(base_text)
    ok_b, payload_b, _ = extract_json_payload(cand_text)
    json_struct: float | None = None
    json_match: float | None = None
    json_num: float | None = None
    if ok_a and ok_b:
        json_struct = json_structural_similarity(payload_a, payload_b)
        json_match = json_exact_field_match(payload_a, payload_b)
        json_num = json_numeric_drift(payload_a, payload_b)

    composite = compute_composite_drift_score(
        token_jaccard=tok_jac,
        ngram_sim=ngram_sim,
        levenshtein_sim=lev_sim,
        json_structural_sim=json_struct,
        json_field_match=json_match,
        json_num_drift=json_num,
        baseline_schema_passed=ok_a,
        candidate_schema_passed=ok_b if ok_a else True,
    )
    refusal_flip = (not is_likely_refusal(base_text)) and is_likely_refusal(cand_text)
    return TurnDriftResult(
        turn_index=turn_index,
        composite_drift=composite,
        token_jaccard=tok_jac,
        levenshtein_similarity=lev_sim,
        refusal_flip=refusal_flip,
        json_valid_baseline=ok_a,
        json_valid_candidate=ok_b,
    )


def analyze_conversation_drift(
    baseline_conversations: Sequence[ConversationRecord],
    candidate_conversations: Sequence[ConversationRecord],
    *,
    max_turn_drift: float = 0.25,
    max_mean_drift: float = 0.25,
    max_final_turn_drift: float | None = None,
    max_turn_count_delta: int | None = None,
    fail_on_refusal_flip: bool = True,
    long_text_mode: Literal["error", "skip_levenshtein"] = "error",
) -> ConversationDriftReport:
    """Compare multi-turn assistant trajectories turn-by-turn across matched conversations."""
    if not baseline_conversations or not candidate_conversations:
        raise ValueError("Both baseline and candidate conversation collections must be non-empty.")
    _validate_unit_limit("max_turn_drift", max_turn_drift)
    _validate_unit_limit("max_mean_drift", max_mean_drift)
    _validate_unit_limit("max_final_turn_drift", max_final_turn_drift)
    if max_turn_count_delta is not None and (
        type(max_turn_count_delta) is not int or max_turn_count_delta < 0
    ):
        raise ValueError("max_turn_count_delta must be a non-negative integer.")
    if long_text_mode not in ("error", "skip_levenshtein"):
        raise ValueError("long_text_mode must be 'error' or 'skip_levenshtein'.")

    base_map = _index_conversations(baseline_conversations, "baseline conversations")
    cand_map = _index_conversations(candidate_conversations, "candidate conversations")
    require_same_ids(set(base_map), set(cand_map), "Candidate conversations")

    conv_results: list[ConversationDriftResult] = []
    for cid in sorted(base_map):
        b_asst = base_map[cid].assistant_turns()
        c_asst = cand_map[cid].assistant_turns()
        compared = min(len(b_asst), len(c_asst))
        turn_results = tuple(
            _score_turn_pair(
                idx,
                b_asst[idx - 1].content,
                c_asst[idx - 1].content,
                long_text_mode=long_text_mode,
            )
            for idx in range(1, compared + 1)
        )
        drifts = [t.composite_drift for t in turn_results]
        mean_d = (sum(drifts) / len(drifts)) if drifts else 0.0
        max_d = max(drifts, default=0.0)
        first_div = next(
            (t.turn_index for t in turn_results if t.composite_drift > max_turn_drift), None
        )
        final_d = drifts[-1] if drifts else None
        slope = (drifts[-1] - drifts[0]) if len(drifts) >= 2 else None
        flips = tuple(t.turn_index for t in turn_results if t.refusal_flip)
        delta_turns = len(c_asst) - len(b_asst)

        conv_drifted = (
            max_d > max_turn_drift
            or (
                max_final_turn_drift is not None
                and final_d is not None
                and final_d > max_final_turn_drift
            )
            or (max_turn_count_delta is not None and abs(delta_turns) > max_turn_count_delta)
            or (fail_on_refusal_flip and bool(flips))
        )
        conv_results.append(
            ConversationDriftResult(
                conversation_id=cid,
                baseline_assistant_turns=len(b_asst),
                candidate_assistant_turns=len(c_asst),
                turn_count_delta=delta_turns,
                compared_turns=compared,
                mean_turn_drift=mean_d,
                max_turn_drift=max_d,
                first_divergence_turn=first_div,
                final_turn_drift=final_d,
                escalation_slope=slope,
                refusal_flip_turns=flips,
                turn_results=turn_results,
                drifted=conv_drifted,
            )
        )

    total = len(conv_results)
    drifted_n = sum(1 for c in conv_results if c.drifted)
    mean_traj = sum(c.mean_turn_drift for c in conv_results) / total
    max_traj = max(c.max_turn_drift for c in conv_results)
    final_vals = [c.final_turn_drift for c in conv_results if c.final_turn_drift is not None]
    mean_final = (sum(final_vals) / len(final_vals)) if final_vals else None
    mismatches = sum(1 for c in conv_results if c.turn_count_delta != 0)
    total_flips = sum(len(c.refusal_flip_turns) for c in conv_results)

    failures: list[str] = []
    if drifted_n > 0:
        failures.append(f"{drifted_n} of {total} conversation(s) exceeded trajectory drift gates.")
    if mean_traj > max_mean_drift:
        failures.append(
            f"Mean trajectory drift ({mean_traj:.4f}) exceeded limit ({max_mean_drift:.4f})."
        )
    if (
        max_final_turn_drift is not None
        and mean_final is not None
        and mean_final > max_final_turn_drift
    ):
        failures.append(
            f"Mean final-turn drift ({mean_final:.4f}) exceeded limit "
            f"({max_final_turn_drift:.4f})."
        )
    if fail_on_refusal_flip and total_flips > 0:
        failures.append(f"Detected {total_flips} assistant refusal flip turn(s).")

    return ConversationDriftReport(
        total_conversations=total,
        drifted_conversations=drifted_n,
        mean_trajectory_drift=mean_traj,
        max_trajectory_drift=max_traj,
        mean_final_turn_drift=mean_final,
        turn_count_mismatches=mismatches,
        total_refusal_flip_turns=total_flips,
        conversations=tuple(conv_results),
        passed=len(failures) == 0,
        failure_reasons=tuple(failures),
    )


def format_conversation_text(report: ConversationDriftReport) -> str:
    """Render a ConversationDriftReport as a readable terminal summary."""
    status = "PASSED" if report.passed else "FAILED"
    final_s = (
        "n/a" if report.mean_final_turn_drift is None else f"{report.mean_final_turn_drift:.4f}"
    )
    lines = [
        f"Multi-Turn Conversation Trajectory Drift [{status}] "
        f"({report.total_conversations} conversation(s), {report.drifted_conversations} drifted)",
        f"  Mean / Max trajectory drift : {report.mean_trajectory_drift:.4f} / "
        f"{report.max_trajectory_drift:.4f}",
        f"  Mean final-turn drift       : {final_s}",
        f"  Turn-count mismatches       : {report.turn_count_mismatches} | "
        f"Refusal flip turns: {report.total_refusal_flip_turns}",
    ]
    for conv in report.conversations:
        tag = "DRIFT" if conv.drifted else "OK"
        div = "none" if conv.first_divergence_turn is None else f"turn {conv.first_divergence_turn}"
        slope = "n/a" if conv.escalation_slope is None else f"{conv.escalation_slope:+.4f}"
        lines.append(
            f"  [{tag:<5}] {conv.conversation_id} | turns {conv.baseline_assistant_turns}->"
            f"{conv.candidate_assistant_turns} | max={conv.max_turn_drift:.4f} | "
            f"first_div={div} | slope={slope}"
        )
    if report.failure_reasons:
        lines.append("")
        lines.append("Failure reasons:")
        for msg in report.failure_reasons:
            lines.append(f"  - {msg}")
    return "\n".join(lines) + "\n"


def format_conversation_markdown(report: ConversationDriftReport) -> str:
    """Render a ConversationDriftReport as GitHub-flavored Markdown."""
    status = "PASSED" if report.passed else "FAILED"
    final_s = (
        "n/a" if report.mean_final_turn_drift is None else f"{report.mean_final_turn_drift:.4f}"
    )
    lines = [
        f"# Multi-Turn Conversation Trajectory Drift: {status}",
        "",
        f"- **Conversations**: {report.total_conversations} "
        f"({report.drifted_conversations} drifted)",
        f"- **Mean / Max trajectory drift**: `{report.mean_trajectory_drift:.4f}` / "
        f"`{report.max_trajectory_drift:.4f}`",
        f"- **Mean final-turn drift**: `{final_s}`",
        f"- **Turn-count mismatches / Refusal flips**: "
        f"`{report.turn_count_mismatches}` / `{report.total_refusal_flip_turns}`",
        "",
        "| Conversation | Turns (B -> C) | Mean Drift | Max Drift | First Div | Slope | Status |",
        "| --- | :---: | ---: | ---: | :---: | ---: | :---: |",
    ]
    for conv in report.conversations:
        div = "-" if conv.first_divergence_turn is None else f"T{conv.first_divergence_turn}"
        slope = "n/a" if conv.escalation_slope is None else f"{conv.escalation_slope:+.4f}"
        verdict = "DRIFT" if conv.drifted else "OK"
        lines.append(
            f"| `{markdown_cell(conv.conversation_id)}` "
            f"| {conv.baseline_assistant_turns} -> {conv.candidate_assistant_turns} "
            f"| {conv.mean_turn_drift:.4f} | {conv.max_turn_drift:.4f} "
            f"| {div} | {slope} | **{verdict}** |"
        )
    if report.failure_reasons:
        lines += ["", "## Failure reasons", ""]
        lines.extend(f"- {markdown_cell(msg)}" for msg in report.failure_reasons)
    return "\n".join(lines) + "\n"


def _render_trajectory_svg(conv: ConversationDriftResult) -> str:
    """Render an inline script-free SVG sparkline of turn-by-turn composite drift."""
    turns = conv.turn_results
    if not turns:
        return "<span class='muted'>n/a</span>"
    width, height = 120.0, 28.0
    pad_x, pad_top, pad_bot = 6.0, 4.0, 4.0
    usable_w = width - 2 * pad_x
    usable_h = height - pad_top - pad_bot
    coords: list[tuple[float, float, TurnDriftResult]] = []
    n = len(turns)
    for idx, t in enumerate(turns):
        x = (pad_x + usable_w / 2.0) if n == 1 else (pad_x + (idx / (n - 1)) * usable_w)
        y = pad_top + usable_h * (1.0 - max(0.0, min(1.0, t.composite_drift)))
        coords.append((x, y, t))
    stroke = "#fda4af" if conv.drifted else "#6ee7b7"
    tip = escape(
        ", ".join(f"T{t.turn_index}={t.composite_drift:.2f}" for t in turns)
    )
    poly = (
        f"<polyline fill='none' stroke='{stroke}' stroke-width='1.8' "
        f"points='{' '.join(f'{x:.1f},{y:.1f}' for x, y, _ in coords)}'/>"
        if n >= 2
        else ""
    )
    dots = "".join(
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.4' "
        f"fill='{'#fb7185' if t.refusal_flip else stroke}'/>"
        for x, y, t in coords
    )
    return (
        f"<svg viewBox='0 0 {int(width)} {int(height)}' width='120' height='28' "
        f"role='img' aria-label='Drift trajectory for {escape(conv.conversation_id)}'>"
        f"<title>{tip}</title>"
        f"<line x1='{pad_x:.0f}' y1='{height - pad_bot:.0f}' "
        f"x2='{width - pad_x:.0f}' y2='{height - pad_bot:.0f}' "
        "stroke='#334155' stroke-width='1'/>"
        f"{poly}{dots}</svg>"
    )


def export_conversation_html(
    report: ConversationDriftReport,
    output_path: str | Path | None = None,
) -> str:
    """Export a standalone script-free offline HTML report for a ConversationDriftReport."""
    status = "PASSED" if report.passed else "FAILED"
    badge_cls = "pass" if report.passed else "fail"
    final_s = (
        "n/a" if report.mean_final_turn_drift is None else f"{report.mean_final_turn_drift:.4f}"
    )
    rows_html = "".join(
        f"<tr><td>{escape(c.conversation_id)}</td>"
        f"<td>{c.baseline_assistant_turns} &rarr; {c.candidate_assistant_turns}</td>"
        f"<td>{_render_trajectory_svg(c)}</td>"
        f"<td>{c.mean_turn_drift:.4f}</td><td>{c.max_turn_drift:.4f}</td>"
        f"<td>{'-' if c.first_divergence_turn is None else f'Turn {c.first_divergence_turn}'}</td>"
        f"<td>{'n/a' if c.escalation_slope is None else f'{c.escalation_slope:+.4f}'}</td>"
        f"<td class='{'fail' if c.drifted else 'pass'}'>{'DRIFT' if c.drifted else 'OK'}</td></tr>"
        for c in report.conversations
    )
    fail_html = ""
    if report.failure_reasons:
        items = "".join(f"<li>{escape(msg)}</li>" for msg in report.failure_reasons)
        fail_html = (
            f"<section class='failures'><h2>Failure Reasons</h2><ul>{items}</ul></section>"
        )
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta http-equiv='Content-Security-Policy' "
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>LLM Drift Bench — Multi-Turn Conversation Drift</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,sans-serif;background:#0b1320;color:#e6edf7;"
        "margin:0;padding:24px;line-height:1.5}"
        "main{max-width:1060px;margin:0 auto}h1,h2{color:#f8fafc}.muted{color:#94a3b8}"
        ".badge{display:inline-block;padding:4px 10px;border-radius:6px;font-weight:700}"
        ".pass{color:#6ee7b7;font-weight:700}.fail{color:#fda4af;font-weight:700}"
        ".badge.pass{background:#064e3b}.badge.fail{background:#4c0519}"
        "table{width:100%;border-collapse:collapse;background:#111e32;margin-top:14px}"
        "th,td{padding:10px 12px;border-bottom:1px solid #26354d;text-align:left}"
        "th{color:#7dd3fc}.failures{margin-top:16px;padding:12px;background:#3b0a17;"
        "border:1px solid #881337;border-radius:8px}"
        "</style></head><body><main>"
        "<p class='muted'>llm-drift-bench / multi-turn conversation trajectory audit</p>"
        "<h1>Multi-Turn Conversation Trajectory Drift "
        f"<span class='badge {badge_cls}'>{status}</span></h1>"
        f"<p>Conversations: <strong>{report.total_conversations}</strong> "
        f"(drifted: <strong>{report.drifted_conversations}</strong>) | "
        f"Mean / Max drift: <strong>{report.mean_trajectory_drift:.4f} / "
        f"{report.max_trajectory_drift:.4f}</strong> | "
        f"Mean final-turn drift: <strong>{final_s}</strong></p>"
        "<table><thead><tr><th>Conversation</th><th>Assistant Turns</th>"
        "<th>Trajectory</th><th>Mean Drift</th><th>Max Drift</th><th>First Divergence</th>"
        "<th>Escalation Slope</th><th>Status</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
        f"{fail_html}</main></body></html>\n"
    )
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
    return html


@dataclass(frozen=True)
class TurnDiff:
    """Semantic JSON or unified text diff for one aligned assistant turn pair."""

    turn_index: int
    diff: PromptDiff

    def to_dict(self) -> dict[str, Any]:
        return {"turn_index": self.turn_index, **self.diff.to_dict()}


@dataclass(frozen=True)
class ConversationDiff:
    """Per-turn semantic diffs for one matched multi-turn conversation."""

    conversation_id: str
    changed: bool
    turn_diffs: tuple[TurnDiff, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "changed": self.changed,
            "turn_diffs": [td.to_dict() for td in self.turn_diffs],
        }


def diff_conversations(
    baseline_conversations: Sequence[ConversationRecord],
    candidate_conversations: Sequence[ConversationRecord],
    *,
    only_changed: bool = False,
) -> list[ConversationDiff]:
    """Compute per-turn semantic JSON/text diffs across matched multi-turn conversations."""
    if not baseline_conversations or not candidate_conversations:
        raise ValueError("Both baseline and candidate conversation collections must be non-empty.")
    base_map = _index_conversations(baseline_conversations, "baseline conversations")
    cand_map = _index_conversations(candidate_conversations, "candidate conversations")
    require_same_ids(set(base_map), set(cand_map), "Candidate conversations")

    results: list[ConversationDiff] = []
    for cid in sorted(base_map):
        b_asst = base_map[cid].assistant_turns()
        c_asst = cand_map[cid].assistant_turns()
        compared = min(len(b_asst), len(c_asst))
        turn_diffs: list[TurnDiff] = []
        for idx in range(1, compared + 1):
            label = f"{cid}#turn-{idx}"
            pdiff = diff_sample_pair(
                ModelSample(prompt_id=label, output=b_asst[idx - 1].content),
                ModelSample(prompt_id=label, output=c_asst[idx - 1].content),
            )
            if not only_changed or pdiff.changed:
                turn_diffs.append(TurnDiff(turn_index=idx, diff=pdiff))
        conv_changed = any(td.diff.changed for td in turn_diffs) or len(b_asst) != len(c_asst)
        if only_changed and not conv_changed:
            continue
        results.append(
            ConversationDiff(
                conversation_id=cid,
                changed=conv_changed,
                turn_diffs=tuple(turn_diffs),
            )
        )
    return results


def format_conversation_diffs_text(diffs: Sequence[ConversationDiff]) -> str:
    """Render per-turn conversation semantic diffs as concise terminal text."""
    if not diffs:
        return "No changed conversation turns.\n"
    blocks: list[str] = []
    for conv in diffs:
        label = "CHANGED" if conv.changed else "IDENTICAL"
        header = f"Conversation {conv.conversation_id} ({label})"
        if conv.turn_diffs:
            body = format_diffs_text([td.diff for td in conv.turn_diffs])
            blocks.append(f"{header}\n{body}")
        else:
            blocks.append(f"{header}\n  (no compared assistant turn diffs)")
    return "\n\n".join(blocks) + "\n"


def export_conversation_pr_comment(report: ConversationDriftReport) -> str:
    """Render a compact GitHub PR comment summarizing multi-turn conversation drift."""
    verdict = "PASS" if report.passed else "FAIL"
    lines = [
        "<!-- llm-drift-bench:conversations -->",
        f"### Multi-Turn Conversation Drift: **{verdict}**",
        "",
        "| Conversations | Drifted | Mean Drift | Max Drift | Refusal Flips | Turn Mismatches |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {report.total_conversations} | {report.drifted_conversations} | "
        f"{report.mean_trajectory_drift:.4f} | {report.max_trajectory_drift:.4f} | "
        f"{report.total_refusal_flip_turns} | {report.turn_count_mismatches} |",
    ]
    drifted = sorted(
        (c for c in report.conversations if c.drifted),
        key=lambda item: (-item.max_turn_drift, item.conversation_id),
    )
    if drifted:
        lines += [
            "",
            "<details><summary>Drifted conversations "
            f"({len(drifted)})</summary>",
            "",
            "| Conversation | Max Drift | First Div Turn | Refusal Flips |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for conv in drifted[:5]:
            first_div = (
                str(conv.first_divergence_turn)
                if conv.first_divergence_turn is not None
                else "-"
            )
            lines.append(
                f"| `{markdown_cell(conv.conversation_id)}` | {conv.max_turn_drift:.4f} | "
                f"{first_div} | {len(conv.refusal_flip_turns)} |"
            )
        if len(drifted) > 5:
            lines.append(f"\n_...and {len(drifted) - 5} more drifted conversation(s)._")
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"
