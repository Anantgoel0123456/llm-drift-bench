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

    def test_unsupported_schema_constraints_fail_explicitly(self) -> None:
        schemas = [
            {"type": "object", "anyOf": [{"required": ["b"]}]},
            {"oneOf": [{"type": "string"}]},
            {"allOf": [{"type": "string"}]},
            {"not": {"type": "object"}},
            {"$ref": "#/$defs/s", "$defs": {"s": {"type": "string"}}},
            {"if": {"type": "object"}, "then": {"required": ["b"]}, "else": {}},
            {"patternProperties": {"^x": {"type": "string"}}},
            {"uniqueItems": True},
            {"multipleOf": 2},
            {"exclusiveMinimum": 0},
            {"exclusiveMaximum": 10},
            {"format": "email"},
        ]
        for schema in schemas:
            with self.subTest(schema=schema):
                issues = validate_against_schema({"a": 1}, schema)
                self.assertTrue(issues)
                self.assertTrue(
                    all(issue.rule == "unsupported_keyword" for issue in issues)
                )

    def test_checks_nested_rules_without_matching_instance_fields(self) -> None:
        schemas = [
            {"properties": {"absent": {"format": "email"}}},
            {"items": {"oneOf": [{"type": "string"}]}},
            {"additionalProperties": {"multipleOf": 2}},
        ]
        for schema in schemas:
            with self.subTest(schema=schema):
                issues = validate_against_schema({}, schema)
                self.assertTrue(issues)
                self.assertEqual(issues[0].rule, "unsupported_keyword")

    def test_schema_annotations_and_property_names_are_not_constraints(self) -> None:
        schema = {
            "title": "Example",
            "default": {"anyOf": 1},
            "examples": [{"$ref": "ordinary data"}],
            "properties": {"anyOf": {"type": "integer"}},
        }
        self.assertEqual(validate_against_schema({"anyOf": 1}, schema), [])

    def test_unsupported_nested_schema_shapes_fail_explicitly(self) -> None:
        schemas = [
            {"properties": {"absent": False}},
            {"items": False},
            {"items": [{"type": "string"}]},
        ]
        for schema in schemas:
            with self.subTest(schema=schema):
                issues = validate_against_schema({}, schema)
                self.assertTrue(issues)
                self.assertEqual(issues[0].rule, "unsupported_schema")

        issues = validate_against_schema({}, {"properties": []})
        self.assertEqual(issues[0].rule, "invalid_schema")

    def test_additional_properties_support_is_preserved(self) -> None:
        self.assertEqual(
            validate_against_schema(
                {"x": "value"}, {"additionalProperties": {"type": "string"}}
            ),
            [],
        )
        self.assertEqual(
            validate_against_schema({}, {"additionalProperties": False}),
            [],
        )
        issues = validate_against_schema(
            {"x": 1}, {"additionalProperties": False}
        )
        self.assertEqual(issues[0].rule, "additionalProperties")

    def test_schema_depth_limit(self) -> None:
        schema = {}
        for _ in range(66):
            schema = {"items": schema}
        issues = validate_against_schema([], schema)
        self.assertEqual(issues[0].rule, "schema_limit")

    def test_validate_output_propagates_unsupported_schema_issue(self) -> None:
        result = validate_output(
            '{"a": 1}',
            {"type": "object", "anyOf": [{"required": ["b"]}]},
        )
        self.assertTrue(result.is_valid_json)
        self.assertFalse(result.is_schema_conformant)
        self.assertFalse(result.passed)
        self.assertEqual(result.issues[0].rule, "unsupported_keyword")

    def test_extract_json_prose_arrays_multi_fence_and_extra_braces(self) -> None:
        ok, data, _ = extract_json_payload('Here is your list: [{"a": 1}]')
        self.assertTrue(ok)
        self.assertEqual(data, [{"a": 1}])

        multi_fence = '```python\nd = {1: 2}\n```\n```json\n{"ok": true}\n```'
        ok_m, data_m, _ = extract_json_payload(multi_fence)
        self.assertTrue(ok_m)
        self.assertEqual(data_m, {"ok": True})

        ok_pre, data_pre, _ = extract_json_payload('Format {k: v}. Answer: {"k": 1}')
        self.assertTrue(ok_pre)
        self.assertEqual(data_pre, {"k": 1})

        ok_post, data_post, _ = extract_json_payload('Answer: {"a": 1} (see {docs})')
        self.assertTrue(ok_post)
        self.assertEqual(data_post, {"a": 1})

    def test_nonfinite_numbers_rejected_and_dotted_keys_quoted(self) -> None:
        for bad_num in (float("nan"), float("inf"), float("-inf")):
            issues = validate_against_schema(
                bad_num, {"type": "number", "minimum": 0, "maximum": 100}
            )
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0].rule, "type")

        key_issues = validate_against_schema(
            {"a.b": 123},
            {"type": "object", "properties": {"a.b": {"type": "string"}}},
        )
        self.assertEqual(key_issues[0].path, '$["a.b"]')
