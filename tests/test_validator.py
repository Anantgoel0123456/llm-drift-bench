"""Unit tests for JSON extraction and schema validation in llm_drift_bench.validator."""

from __future__ import annotations

import unittest

from llm_drift_bench.validator import (
    _check_json_type,
    extract_json_payload,
    validate_against_schema,
    validate_output,
)


class TestValidator(unittest.TestCase):
    """Test suite for extract_json_payload, validate_against_schema, and validate_output."""

    def test_extract_json_payload_variants(self) -> None:
        ok, data, err = extract_json_payload('{"a": 1}')
        self.assertTrue(ok)
        self.assertEqual(data, {"a": 1})
        self.assertIsNone(err)

        fenced = 'Here is the output:\n```json\n{"status": "ok", "count": 3}\n```\nThanks!'
        ok_f, data_f, _ = extract_json_payload(fenced)
        self.assertTrue(ok_f)
        self.assertEqual(data_f, {"status": "ok", "count": 3})

        prose_array = "Result list: [10, 20, 30] end."
        ok_a, data_a, _ = extract_json_payload(prose_array)
        self.assertTrue(ok_a)
        self.assertEqual(data_a, [10, 20, 30])

        bad_fence_fallback = 'Prefix {"valid": true} ```json\n{bad\n```'
        ok_b, data_b, _ = extract_json_payload(bad_fence_fallback)
        self.assertTrue(ok_b)
        self.assertEqual(data_b, {"valid": True})

    def test_extract_json_payload_failures(self) -> None:
        ok_empty, _, err_empty = extract_json_payload("   ")
        self.assertFalse(ok_empty)
        self.assertIn("Empty", err_empty or "")

        ok_bad, _, err_bad = extract_json_payload("No JSON here {unclosed")
        self.assertFalse(ok_bad)
        self.assertIn("Invalid JSON", err_bad or "")

    def test_check_json_type_primitives(self) -> None:
        self.assertTrue(_check_json_type(None, "null"))
        self.assertTrue(_check_json_type(True, "boolean"))
        self.assertFalse(_check_json_type(True, "integer"))
        self.assertFalse(_check_json_type(False, "number"))
        self.assertTrue(_check_json_type(42, "integer"))
        self.assertTrue(_check_json_type(3.14, "number"))
        self.assertTrue(_check_json_type("hello", "string"))
        self.assertTrue(_check_json_type([1, 2], "array"))
        self.assertTrue(_check_json_type({"k": "v"}, "object"))
        self.assertTrue(_check_json_type("anything", "custom_unknown"))

    def test_validate_against_schema_comprehensive_rules(self) -> None:
        schema = {
            "type": "object",
            "required": ["code", "score", "mode", "tags", "meta"],
            "properties": {
                "code": {"type": "string", "minLength": 3, "maxLength": 6, "pattern": r"^[A-Z]+$"},
                "score": {"type": "number", "minimum": 0.0, "maximum": 10.0},
                "mode": {"type": "string", "enum": ["fast", "accurate"], "const": "fast"},
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "items": {"type": "string"},
                },
                "meta": {
                    "type": "object",
                    "properties": {"fixed": {"type": "integer"}},
                    "additionalProperties": {"type": "string"},
                },
            },
            "additionalProperties": False,
        }

        valid_instance = {
            "code": "ABC",
            "score": 9.5,
            "mode": "fast",
            "tags": ["nlp"],
            "meta": {"fixed": 1, "extra_str": "ok"},
        }
        self.assertEqual(validate_against_schema(valid_instance, schema), [])

        invalid_instance = {
            "code": "a1",
            "score": 25.0,
            "mode": "slow",
            "tags": [],
            "meta": {"fixed": True, "extra_bad": 999},
            "unexpected": 1,
        }
        issues = validate_against_schema(invalid_instance, schema)
        rules_triggered = {issue.rule for issue in issues}
        self.assertIn("minLength", rules_triggered)
        self.assertIn("pattern", rules_triggered)
        self.assertIn("maximum", rules_triggered)
        self.assertIn("enum", rules_triggered)
        self.assertIn("const", rules_triggered)
        self.assertIn("minItems", rules_triggered)
        self.assertIn("type", rules_triggered)
        self.assertIn("additionalProperties", rules_triggered)

    def test_validate_against_schema_upper_bounds_and_missing_required(self) -> None:
        schema = {
            "type": "object",
            "required": ["name", "val", "arr"],
            "properties": {
                "name": {"type": "string", "maxLength": 3},
                "val": {"type": "number", "minimum": 5},
                "arr": {"type": "array", "maxItems": 1, "items": {"type": "integer"}},
            },
        }
        issues = validate_against_schema(
            {"name": "toolong", "val": 2, "arr": [1, "bad"]},
            schema,
        )
        rules = {iss.rule for iss in issues}
        self.assertEqual(rules, {"maxLength", "minimum", "maxItems", "type"})

        missing_issues = validate_against_schema({}, schema)
        self.assertEqual(len(missing_issues), 3)
        self.assertTrue(all(i.rule == "required" for i in missing_issues))

    def test_validate_output_entrypoint(self) -> None:
        res_syntax_err = validate_output("{broken json")
        self.assertFalse(res_syntax_err.passed)
        self.assertEqual(res_syntax_err.issues[0].rule, "json_syntax")

        res_no_schema = validate_output('{"ok": true}', schema=None)
        self.assertTrue(res_no_schema.passed)

        res_with_schema = validate_output(
            '```json\n{"count": 5}\n```',
            schema={
                "type": "object",
                "required": ["count"],
                "properties": {"count": {"type": "integer"}},
            },
        )
        self.assertTrue(res_with_schema.passed)
