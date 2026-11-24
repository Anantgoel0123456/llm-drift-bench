"""Semantic JSON field-level and unified text diffing for captured LLM outputs."""

from __future__ import annotations

import difflib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from llm_drift_bench._matching import require_same_ids
from llm_drift_bench._numbers import finite_number
from llm_drift_bench._report_views import markdown_cell
from llm_drift_bench.metrics import _json_type_tag, _values_equal, flatten_json_paths
from llm_drift_bench.models import ModelSample
from llm_drift_bench.policy import (
    DriftPolicy,
    _normalize_unordered_arrays,
    _numeric_within_tolerance,
    match_field_rule,
    path_is_ignored,
)
from llm_drift_bench.validator import extract_json_payload

ChangeKind = Literal["added", "removed", "changed", "type_changed"]


@dataclass(frozen=True)
class FieldChange:
    """Single JSON leaf-path difference between baseline and candidate outputs."""

    path: str
    kind: ChangeKind
    baseline: Any = None
    candidate: Any = None
    numeric_rel_drift: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "baseline": self.baseline,
            "candidate": self.candidate,
        }
        if self.numeric_rel_drift is not None:
            payload["numeric_rel_drift"] = self.numeric_rel_drift
        return payload


@dataclass(frozen=True)
class PromptDiff:
    """Structured JSON or unified text diff for one prompt pair."""

    prompt_id: str
    mode: Literal["json", "text"]
    changed: bool
    field_changes: tuple[FieldChange, ...] = ()
    unified_diff: tuple[str, ...] = ()
    ignored_paths: tuple[str, ...] = ()

    def summary_counts(self) -> dict[str, int]:
        counts = {"added": 0, "removed": 0, "changed": 0, "type_changed": 0}
        for change in self.field_changes:
            counts[change.kind] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "mode": self.mode,
            "changed": self.changed,
            "summary": self.summary_counts(),
            "field_changes": [change.to_dict() for change in self.field_changes],
            "unified_diff": list(self.unified_diff),
            "ignored_paths": list(self.ignored_paths),
        }


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def diff_sample_pair(
    baseline: ModelSample,
    candidate: ModelSample,
    *,
    policy: DriftPolicy | None = None,
) -> PromptDiff:
    """Compute a policy-aware JSON field diff or unified text diff for one prompt pair."""
    if baseline.prompt_id != candidate.prompt_id:
        raise ValueError(
            f"Prompt ID mismatch: baseline {baseline.prompt_id!r} vs "
            f"candidate {candidate.prompt_id!r}."
        )

    ok_a, payload_a, _ = extract_json_payload(baseline.output)
    ok_b, payload_b, _ = extract_json_payload(candidate.output)
    if ok_a and ok_b:
        field_rules = policy.field_rules if policy is not None else {}
        unordered = frozenset(p for p, r in field_rules.items() if r.unordered_array)
        if unordered:
            payload_a = _normalize_unordered_arrays(payload_a, unordered)
            payload_b = _normalize_unordered_arrays(payload_b, unordered)

        flat_a_raw = flatten_json_paths(payload_a)
        flat_b_raw = flatten_json_paths(payload_b)
        # Comparing an empty container ({} or []) with a non-empty container of the same kind
        # should report only the added/removed child paths, not a spurious root '$' removal.
        same_container_empty_mismatch = (
            (isinstance(payload_a, dict) and isinstance(payload_b, dict))
            or (isinstance(payload_a, list) and isinstance(payload_b, list))
        ) and (bool(payload_a) != bool(payload_b))
        if same_container_empty_mismatch:
            if not payload_a:
                flat_a_raw = {}
            else:
                flat_b_raw = {}
        ignored_rules = tuple(p for p, r in field_rules.items() if r.ignore)

        ignored_seen: set[str] = set()
        flat_a: dict[str, Any] = {}
        for path, val in flat_a_raw.items():
            if path_is_ignored(path, ignored_rules):
                ignored_seen.add(path)
            else:
                flat_a[path] = val
        flat_b: dict[str, Any] = {}
        for path, val in flat_b_raw.items():
            if path_is_ignored(path, ignored_rules):
                ignored_seen.add(path)
            else:
                flat_b[path] = val

        changes: list[FieldChange] = []
        for path in sorted(set(flat_a) | set(flat_b)):
            if path not in flat_a:
                changes.append(FieldChange(path=path, kind="added", candidate=flat_b[path]))
                continue
            if path not in flat_b:
                changes.append(FieldChange(path=path, kind="removed", baseline=flat_a[path]))
                continue

            val_a, val_b = flat_a[path], flat_b[path]
            if _is_numeric(val_a) and _is_numeric(val_b):
                num_a, num_b = finite_number(val_a), finite_number(val_b)
                rule = match_field_rule(path, field_rules)
                within_tol, rel = _numeric_within_tolerance(num_a, num_b, rule)
                if not within_tol and not _values_equal(val_a, val_b):
                    changes.append(
                        FieldChange(
                            path=path,
                            kind="changed",
                            baseline=val_a,
                            candidate=val_b,
                            numeric_rel_drift=round(rel, 4),
                        )
                    )
                continue

            if _json_type_tag(val_a) != _json_type_tag(val_b):
                changes.append(
                    FieldChange(
                        path=path,
                        kind="type_changed",
                        baseline=val_a,
                        candidate=val_b,
                    )
                )
            elif not _values_equal(val_a, val_b):
                changes.append(
                    FieldChange(
                        path=path,
                        kind="changed",
                        baseline=val_a,
                        candidate=val_b,
                    )
                )

        return PromptDiff(
            prompt_id=baseline.prompt_id,
            mode="json",
            changed=bool(changes),
            field_changes=tuple(changes),
            ignored_paths=tuple(sorted(ignored_seen)),
        )

    changed = baseline.output != candidate.output
    diff_lines = (
        tuple(
            difflib.unified_diff(
                baseline.output.splitlines(),
                candidate.output.splitlines(),
                fromfile="baseline",
                tofile="candidate",
                lineterm="",
            )
        )
        if changed
        else ()
    )
    return PromptDiff(
        prompt_id=baseline.prompt_id,
        mode="text",
        changed=changed,
        unified_diff=diff_lines,
    )


def diff_captures(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
    *,
    policy: DriftPolicy | None = None,
    prompt_ids: Sequence[str] | None = None,
    only_changed: bool = False,
) -> list[PromptDiff]:
    """Compute sorted diffs across two matched capture collections."""
    if not baseline_samples or not candidate_samples:
        raise ValueError("Baseline and candidate captures must both be non-empty.")
    base_map = {s.prompt_id: s for s in baseline_samples}
    cand_map = {s.prompt_id: s for s in candidate_samples}
    if len(base_map) != len(baseline_samples) or len(cand_map) != len(candidate_samples):
        raise ValueError("Captures contain duplicate prompt_id entries.")
    require_same_ids(set(base_map), set(cand_map), "Candidate capture")

    target_ids = sorted(base_map)
    if prompt_ids:
        requested = set(prompt_ids)
        unknown = sorted(requested - set(base_map))
        if unknown:
            raise ValueError(f"Unknown prompt_id(s): {unknown!r}.")
        target_ids = [pid for pid in target_ids if pid in requested]

    diffs = [diff_sample_pair(base_map[pid], cand_map[pid], policy=policy) for pid in target_ids]
    if only_changed:
        diffs = [item for item in diffs if item.changed]
    return diffs


def _fmt_val(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def format_diffs_text(diffs: Sequence[PromptDiff]) -> str:
    """Render PromptDiff objects as a crisp, readable terminal report."""
    total = len(diffs)
    changed_count = sum(1 for d in diffs if d.changed)
    lines = [f"Diff summary: {changed_count} of {total} prompt(s) changed"]
    if not diffs:
        lines.append("No matching prompts to display.")
        return "\n".join(lines)

    for item in diffs:
        lines.append("-" * 72)
        if not item.changed:
            note = (
                f" ({len(item.ignored_paths)} ignored via policy)"
                if item.ignored_paths
                else ""
            )
            lines.append(f"[UNCHANGED] {item.prompt_id} ({item.mode}){note}")
            continue
        if item.mode == "json":
            lines.append(
                f"[CHANGED] {item.prompt_id} (json, {len(item.field_changes)} field change(s))"
            )
            for fc in item.field_changes:
                if fc.kind == "added":
                    lines.append(f"  + {fc.path}: {_fmt_val(fc.candidate)}")
                elif fc.kind == "removed":
                    lines.append(f"  - {fc.path}: {_fmt_val(fc.baseline)}")
                elif fc.kind == "type_changed":
                    lines.append(
                        f"  ! {fc.path}: {_fmt_val(fc.baseline)} -> {_fmt_val(fc.candidate)} "
                        f"(type changed)"
                    )
                else:
                    suffix = (
                        f" (rel drift {fc.numeric_rel_drift:.4f})"
                        if fc.numeric_rel_drift is not None
                        else ""
                    )
                    lines.append(
                        f"  ~ {fc.path}: {_fmt_val(fc.baseline)} -> "
                        f"{_fmt_val(fc.candidate)}{suffix}"
                    )
            if item.ignored_paths:
                lines.append(
                    f"  (ignored {len(item.ignored_paths)} path(s) via policy: "
                    f"{', '.join(item.ignored_paths[:5])})"
                )
        else:
            lines.append(f"[CHANGED] {item.prompt_id} (text)")
            for raw_line in item.unified_diff:
                lines.append(f"  {raw_line}")
    return "\n".join(lines)


def format_diffs_markdown(
    diffs: Sequence[PromptDiff],
    *,
    max_changes_per_prompt: int = 25,
) -> str:
    """Render PromptDiff items as GitHub-Flavored Markdown tables and diff blocks."""
    if type(max_changes_per_prompt) is not int or max_changes_per_prompt < 1:
        raise ValueError("max_changes_per_prompt must be a positive integer.")
    total = len(diffs)
    changed_count = sum(1 for d in diffs if d.changed)
    lines = [
        "## Semantic Response Diff",
        "",
        f"**Changed prompts:** `{changed_count}/{total}`",
    ]
    if not diffs:
        lines += ["", "No matching prompts to display."]
        return "\n".join(lines) + "\n"

    for item in diffs:
        lines += ["", f"### `{markdown_cell(item.prompt_id)}` ({item.mode})"]
        if not item.changed:
            note = (
                f" ({len(item.ignored_paths)} path(s) ignored via policy)"
                if item.ignored_paths
                else ""
            )
            lines.append(f"Unchanged{note}.")
            continue
        if item.mode == "json":
            lines += [
                "",
                "| Field Path | Change | Baseline | Candidate | Rel Drift |",
                "| :--- | :---: | :--- | :--- | ---: |",
            ]
            shown = item.field_changes[:max_changes_per_prompt]
            for fc in shown:
                base_s = "(missing)" if fc.kind == "added" else _fmt_val(fc.baseline)
                cand_s = "(missing)" if fc.kind == "removed" else _fmt_val(fc.candidate)
                rel_s = "-" if fc.numeric_rel_drift is None else f"{fc.numeric_rel_drift:.4f}"
                lines.append(
                    f"| `{markdown_cell(fc.path)}` | **{fc.kind}** | "
                    f"`{markdown_cell(base_s)}` | `{markdown_cell(cand_s)}` | {rel_s} |"
                )
            omitted = len(item.field_changes) - len(shown)
            if omitted > 0:
                lines += ["", f"_{omitted} additional field change(s) omitted._"]
            if item.ignored_paths:
                lines += ["", f"_Ignored {len(item.ignored_paths)} path(s) via policy._"]
        else:
            lines += ["", "```diff", *item.unified_diff[:max_changes_per_prompt], "```"]
    return "\n".join(lines) + "\n"
