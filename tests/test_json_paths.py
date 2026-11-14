"""Distinct JSON fields must remain distinct in every metric."""

from llm_drift_bench.metrics import (
    flatten_json_paths,
    json_exact_field_match,
    json_numeric_drift,
    json_structural_similarity,
)


def test_literal_dots_and_array_syntax_do_not_collide() -> None:
    payload = {"a.b": 1, "a": {"b": 2}, "x[0]": 3, "x": [4], "": 5}
    flat = flatten_json_paths(payload)
    assert len(flat) == 5
    assert flat['$["a.b"]'] == 1
    assert flat["$.a.b"] == 2
    assert flat['$["x[0]"]'] == 3
    assert flat["$.x[0]"] == 4
    assert flat['$[""]'] == 5


def test_escaped_and_unicode_keys_remain_unique() -> None:
    payload = {'"': 1, '\\': 2, '\n': 3, "é": 4, "['a']": 5, "$": 6}
    assert len(flatten_json_paths(payload)) == len(payload)
    assert json_exact_field_match(payload, payload) == 1.0


def test_different_structures_cannot_score_as_identical() -> None:
    left, right = {"a.b": 1}, {"a": {"b": 1}}
    assert json_structural_similarity(left, right) == 0.0
    assert json_exact_field_match(left, right) == 0.0
    assert json_numeric_drift(left, right) is None


def test_changed_literal_field_cannot_be_overwritten_by_nested_field() -> None:
    left = {"a.b": 10, "a": {"b": 1}}
    right = {"a.b": 20, "a": {"b": 1}}
    assert json_exact_field_match(left, right) == 0.5
    assert json_numeric_drift(left, right) == 0.25
