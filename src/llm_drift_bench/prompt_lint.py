"""Quality and JSON Schema hygiene linter for benchmark prompt suites."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from llm_drift_bench.models import PromptCase
from llm_drift_bench.validator import _schema_support_issues

LintSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class LintIssue:
    """Single prompt suite quality or schema hygiene finding."""

    severity: LintSeverity
    code: str
    prompt_id: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _walk_schema_warnings(schema: dict[str, Any], prompt_id: str) -> list[LintIssue]:
    warnings: list[LintIssue] = []
    stack: list[tuple[dict[str, Any], str]] = [(schema, "$")]
    while stack:
        node, path = stack.pop()
        raw_type = node.get("type")
        is_object = raw_type == "object" or (isinstance(raw_type, list) and "object" in raw_type)
        if is_object:
            props = node.get("properties")
            required = node.get("required")
            if isinstance(props, dict) and props:
                if "additionalProperties" not in node:
                    warnings.append(
                        LintIssue(
                            "warning",
                            "permissive_object",
                            prompt_id,
                            f"{path}: object schema omits 'additionalProperties'; "
                            "unexpected keys will pass silently.",
                        )
                    )
                elif node.get("additionalProperties") is False and isinstance(required, list):
                    missing = [
                        key for key in required if isinstance(key, str) and key not in props
                    ]
                    if missing:
                        warnings.append(
                            LintIssue(
                                "error",
                                "unsatisfiable_required",
                                prompt_id,
                                f"{path}: required keys {missing!r} are absent from 'properties' "
                                "while additionalProperties is false.",
                            )
                        )
                for key in sorted(props, reverse=True):
                    if isinstance(props[key], dict):
                        stack.append((props[key], f"{path}.{key}"))
            elif not required:
                warnings.append(
                    LintIssue(
                        "warning",
                        "unconstrained_object",
                        prompt_id,
                        f"{path}: object schema defines neither 'properties' nor 'required'.",
                    )
                )
        items = node.get("items")
        if isinstance(items, dict):
            stack.append((items, f"{path}[]"))
    return warnings


def lint_prompt_suite(cases: Sequence[PromptCase]) -> list[LintIssue]:
    """Inspect a PromptCase suite for schema errors, duplicate prompts, and hygiene gaps."""
    if not cases:
        return [LintIssue("error", "empty_suite", None, "Prompt suite contains zero cases.")]

    issues: list[LintIssue] = []
    seen_ids: set[str] = set()
    reported_dup_ids: set[str] = set()
    seen_texts: dict[str, list[str]] = defaultdict(list)

    for case in sorted(cases, key=lambda c: c.prompt_id):
        if case.prompt_id in seen_ids:
            if case.prompt_id not in reported_dup_ids:
                reported_dup_ids.add(case.prompt_id)
                issues.append(
                    LintIssue(
                        "error",
                        "duplicate_prompt_id",
                        case.prompt_id,
                        f"Duplicate prompt_id {case.prompt_id!r} appears multiple times.",
                    )
                )
            continue
        seen_ids.add(case.prompt_id)
        normalized = " ".join(case.prompt.split()).lower()
        if not normalized:
            issues.append(
                LintIssue("warning", "empty_prompt", case.prompt_id, "Prompt text is empty.")
            )
        else:
            seen_texts[normalized].append(case.prompt_id)

        if case.schema is not None:
            support_issues = _schema_support_issues(case.schema, "$")
            if support_issues:
                for sup in support_issues:
                    issues.append(
                        LintIssue(
                            "error",
                            f"schema_{sup.rule}",
                            case.prompt_id,
                            f"{sup.path}: {sup.message}",
                        )
                    )
            else:
                issues.extend(_walk_schema_warnings(case.schema, case.prompt_id))

    for ids in seen_texts.values():
        if len(ids) > 1:
            issues.append(
                LintIssue(
                    "warning",
                    "duplicate_prompt_text",
                    ids[0],
                    f"Identical prompt text shared by prompt IDs: {ids!r}.",
                )
            )

    categories = {case.category for case in cases}
    if len(cases) >= 5 and categories == {"general"}:
        issues.append(
            LintIssue(
                "warning",
                "uncategorized_suite",
                None,
                "All prompts use default category 'general'; add categories for breakdown views.",
            )
        )

    return issues


def format_lint_issues_text(issues: Sequence[LintIssue]) -> str:
    """Format prompt suite lint findings for terminal output."""
    errors = sum(1 for item in issues if item.severity == "error")
    warnings = sum(1 for item in issues if item.severity == "warning")
    if not issues:
        return "Prompt suite lint: OK (0 errors, 0 warnings)."
    lines = [f"Prompt suite lint: {errors} error(s), {warnings} warning(s)"]
    for item in issues:
        target = item.prompt_id or "suite"
        lines.append(f"  [{item.severity.upper():<7}] {target} ({item.code}): {item.message}")
    return "\n".join(lines)
