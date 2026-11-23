"""Tests for the prompt suite quality and schema hygiene linter."""

from __future__ import annotations

from llm_drift_bench.models import PromptCase
from llm_drift_bench.prompt_lint import format_lint_issues_text, lint_prompt_suite


def test_lint_prompt_suite_detects_errors_and_hygiene_warnings() -> None:
    assert lint_prompt_suite([])[0].code == "empty_suite"

    cases = [
        PromptCase(prompt_id="p1", prompt="   ", category="general"),
        PromptCase(
            prompt_id="p2",
            prompt="Extract invoice",
            category="general",
            schema={"type": "object", "properties": {"a": {"type": "string"}}},
        ),
        PromptCase(
            prompt_id="p3",
            prompt="Extract   invoice",
            category="general",
            schema={"type": "object", "anyOf": [{"type": "string"}]},
        ),
        PromptCase(
            prompt_id="p4",
            prompt="Classify",
            category="general",
            schema={"type": "object"},
        ),
        PromptCase(
            prompt_id="p5",
            prompt="Strict",
            category="general",
            schema={
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
                "additionalProperties": False,
            },
        ),
    ]
    issues = lint_prompt_suite(cases)
    codes = {i.code for i in issues}
    assert "empty_prompt" in codes
    assert "permissive_object" in codes
    assert "schema_unsupported_keyword" in codes
    assert "unconstrained_object" in codes
    assert "duplicate_prompt_text" in codes
    assert "uncategorized_suite" in codes

    rendered = format_lint_issues_text(issues)
    assert "1 error(s)" in rendered

    clean = [
        PromptCase(
            prompt_id="c1",
            prompt="Extract",
            category="extraction",
            schema={
                "type": "object",
                "required": ["x"],
                "properties": {"x": {"type": "integer"}},
                "additionalProperties": False,
            },
        )
    ]
    assert lint_prompt_suite(clean) == []
    assert "0 errors, 0 warnings" in format_lint_issues_text([])


def test_lint_prompt_suite_catches_duplicate_ids_and_unsatisfiable_required() -> None:
    cases = [
        PromptCase(
            prompt_id="dup",
            prompt="Extract",
            category="extraction",
            schema={
                "type": "object",
                "required": ["missing_key"],
                "properties": {"present": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        PromptCase(prompt_id="dup", prompt="Second", category="extraction"),
    ]
    issues = lint_prompt_suite(cases)
    by_code = {issue.code: issue for issue in issues}
    assert by_code["unsatisfiable_required"].severity == "error"
    assert by_code["duplicate_prompt_id"].severity == "error"
    assert by_code["duplicate_prompt_id"].to_dict()["prompt_id"] == "dup"
