"""Unsafe patterns must be rejected before matching, even for absent fields."""

from unittest.mock import patch

import pytest

from llm_drift_bench.validator import validate_against_schema


@pytest.mark.parametrize("pattern", [
    "^(a+)+$", "(a|aa)+$", "^a*a*$", "a+$", "^a{1,100}$", "^a?$",
    r"^(a)\1$", "(?=a)a", "^a+?$", "^a++$", "a" * 257, r"^\x61+$",
])
def test_unsafe_patterns_never_reach_search(pattern: str) -> None:
    with patch("llm_drift_bench.validator.re.search", side_effect=AssertionError("matched")):
        for schema in ({"pattern": pattern}, {"properties": {"absent": {"pattern": pattern}}}):
            issues = validate_against_schema("a" * 10000 + "!", schema)
            assert issues and issues[0].rule == "invalid_schema"


@pytest.mark.parametrize(("pattern", "value"), [
    (r"^INV-\d+$", "INV-123"), ("^[A-Z]+$", "ABC"),
    ("^.*end$", "the end"), ("word", "a word here"),
    (r"^a\+b$", "a+b"), ("^[()|]+$", "()|"), ("", "anything"),
    (r"^[^]+]+$", "abc"), (r"^[]()+]+$", "]()+"), (r"^[^]]+$", "hello"),
])
def test_documented_subset_matches(pattern: str, value: str) -> None:
    assert not validate_against_schema(value, {"pattern": pattern})


def test_long_inputs_fail_explicitly_without_matching() -> None:
    with patch("llm_drift_bench.validator.re.search", side_effect=AssertionError("matched")):
        issues = validate_against_schema("a" * 16385, {"pattern": "^a+$"})
    assert issues[0].rule == "pattern_limit"
    assert not validate_against_schema("a" * 16384, {"pattern": "^a+$"})
