"""Malformed schemas must fail even when the affected property is absent."""

import pytest

from llm_drift_bench.validator import validate_against_schema


@pytest.mark.parametrize("schema", [
    {"type": "typo"}, {"type": None}, {"type": []}, {"type": ["string", "string"]},
    {"type": [{}]}, {"required": "name"}, {"required": [[]]},
    {"required": ["x", "x"]}, {"enum": None}, {"enum": []},
    {"enum": [1, 1.0]}, {"minLength": True}, {"maxLength": -1},
    {"minItems": "2"}, {"maxItems": 1.5}, {"minimum": True},
    {"maximum": float("nan")}, {"pattern": "["}, {"pattern": 3},
])
def test_invalid_keyword_shapes_fail_at_all_schema_locations(schema: dict) -> None:
    for candidate in (schema, {"properties": {"absent": schema}}, {"items": schema}):
        issues = validate_against_schema({}, candidate)
        assert issues
        assert issues[0].rule == "invalid_schema"


@pytest.mark.parametrize(("instance", "expected"), [
    (True, 1), (False, 0), ({"x": True}, {"x": 1}), ([True], [1]),
])
def test_boolean_number_distinction_is_recursive(instance: object, expected: object) -> None:
    for keyword, value in (("const", expected), ("enum", [expected])):
        assert validate_against_schema(instance, {keyword: value})[0].rule == keyword


def test_valid_nullable_and_numeric_equality() -> None:
    assert not validate_against_schema(None, {"type": ["string", "null"]})
    assert not validate_against_schema(1, {"enum": [True, 1.0]})
    assert not validate_against_schema({"x": [1]}, {"const": {"x": [1.0]}})
    assert not validate_against_schema({}, {"required": []})


@pytest.mark.parametrize(("keyword", "passing", "failing"), [
    ("minLength", "abc", "ab"), ("maxLength", "abc", "abcd"),
    ("minItems", [1, 2, 3], [1, 2]), ("maxItems", [1, 2, 3], [1, 2, 3, 4]),
])
def test_integral_float_schema_limits_are_enforced(keyword, passing, failing) -> None:
    schema = {keyword: 3.0}
    assert not validate_against_schema(passing, schema)
    assert validate_against_schema(failing, schema)[0].rule == keyword
    assert schema[keyword] == 3.0


@pytest.mark.parametrize("keyword", ["minLength", "maxLength", "minItems", "maxItems"])
@pytest.mark.parametrize("value", [True, False, -1.0, 1.5, float("nan"), float("inf"), "3"])
def test_invalid_schema_limits_fail_even_for_absent_properties(keyword, value) -> None:
    issues = validate_against_schema({}, {"properties": {"absent": {keyword: value}}})
    assert issues[0].rule == "invalid_schema"
