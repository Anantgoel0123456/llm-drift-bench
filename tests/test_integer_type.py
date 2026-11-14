"""JSON Schema integers are finite numbers without a fractional part."""

import pytest

from llm_drift_bench import validate_output
from llm_drift_bench.validator import _check_json_type

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


@pytest.mark.parametrize("literal", ["1", "1.0", "-3.0", "1e2", "0.0", "-0"])
def test_integral_numbers_satisfy_integer(literal: str) -> None:
    assert validate_output(f'{{"n": {literal}}}', SCHEMA).passed


@pytest.mark.parametrize("literal", ["1.5", "0.1", "1e-2", "true", "false", '"1"', "null", "[1]"])
def test_non_integral_or_non_numbers_fail_integer(literal: str) -> None:
    result = validate_output(f'{{"n": {literal}}}', SCHEMA)
    assert not result.passed
    assert result.issues[0].rule == "type"


def test_type_checker_edges() -> None:
    assert _check_json_type(1.0, "integer")
    assert _check_json_type(10**30, "integer")
    assert not _check_json_type(True, "integer")
    assert not _check_json_type(float("inf"), "integer")
    assert not _check_json_type(float("nan"), "integer")
    assert not _check_json_type(2.5, "integer")
    assert _check_json_type(2.5, "number") and not _check_json_type(True, "number")


def test_integer_in_type_union_and_bounds() -> None:
    schema = {"type": ["integer", "null"], "minimum": 1, "maximum": 3}
    assert validate_output("2.0", schema).passed
    assert validate_output("null", schema).passed
    assert not validate_output("3.5", schema).passed
    assert not validate_output("4.0", schema).passed
