"""Zero-bloat JSON payload extraction and JSON Schema conformity validation."""

from __future__ import annotations

import json
import re
from typing import Any

from llm_drift_bench.models import SchemaValidationResult, ValidationIssue

_CODE_FENCE_PATTERN = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def extract_json_payload(raw_text: str) -> tuple[bool, Any, str | None]:
    """Extract and parse a JSON payload from raw LLM completion text.

    Supports pure JSON strings, markdown code fences (```json ... ```), and
    fallback extraction of the outermost JSON object/array when wrapped in prose.

    Args:
        raw_text: Raw text completion produced by an LLM.

    Returns:
        A tuple of (is_valid_json, parsed_value, error_message).
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return False, None, "Empty or non-string output."

    stripped = raw_text.strip()

    # 1. Try direct JSON parsing first.
    try:
        return True, json.loads(stripped), None
    except json.JSONDecodeError as direct_err:
        last_error = str(direct_err)

    # 2. Try extracting from markdown ```json ... ``` fences.
    fence_match = _CODE_FENCE_PATTERN.search(stripped)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return True, json.loads(candidate), None
        except json.JSONDecodeError as fence_err:
            last_error = str(fence_err)

    # 3. Try locating the first '{' or '[' and matching to last '}' or ']'.
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start_idx = stripped.find(open_char)
        end_idx = stripped.rfind(close_char)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            sub = stripped[start_idx : end_idx + 1]
            try:
                return True, json.loads(sub), None
            except json.JSONDecodeError:
                continue

    return False, None, f"Invalid JSON syntax: {last_error}"


def _check_json_type(instance: Any, expected_type: str) -> bool:
    """Return True if instance matches the JSON Schema primitive or container type."""
    if expected_type == "null":
        return instance is None
    if expected_type == "boolean":
        return isinstance(instance, bool)
    if expected_type == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected_type == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected_type == "string":
        return isinstance(instance, str)
    if expected_type == "array":
        return isinstance(instance, list)
    if expected_type == "object":
        return isinstance(instance, dict)
    return True


_SUPPORTED_SCHEMA_KEYWORDS = frozenset({
    "type", "enum", "const", "minLength", "maxLength", "pattern",
    "minimum", "maximum", "minItems", "maxItems", "items",
    "required", "properties", "additionalProperties",
})
_SCHEMA_ANNOTATIONS = frozenset({
    "title", "description", "default", "examples", "$comment",
    "readOnly", "writeOnly", "deprecated",
})


def _schema_support_issues(schema: dict[str, Any], path: str) -> list[ValidationIssue]:
    """Check every schema node, including rules for absent instance fields."""
    issues: list[ValidationIssue] = []
    pending: list[tuple[Any, str, int]] = [(schema, path, 0)]
    visited = 0
    while pending:
        node, location, depth = pending.pop()
        visited += 1
        if depth > 64 or visited > 10_000:
            issues.append(ValidationIssue(
                location, "Schema exceeds the depth or node limit.", "schema_limit"
            ))
            break
        if not isinstance(node, dict):
            issues.append(ValidationIssue(
                location, "Only object-valued schemas are supported.", "unsupported_schema"
            ))
            continue
        for keyword in node:
            if keyword not in _SUPPORTED_SCHEMA_KEYWORDS | _SCHEMA_ANNOTATIONS:
                issues.append(ValidationIssue(
                    location, f"Unsupported JSON Schema keyword: {keyword!r}.",
                    "unsupported_keyword",
                ))
        if "properties" in node:
            properties = node["properties"]
            if not isinstance(properties, dict):
                issues.append(ValidationIssue(
                    location, "properties must be an object.", "invalid_schema"
                ))
            else:
                for key, child in reversed(list(properties.items())):
                    child_path = f"{location}.properties[{json.dumps(key)}]"
                    pending.append((child, child_path, depth + 1))
        if "items" in node:
            pending.append((node["items"], f"{location}.items", depth + 1))
        if "additionalProperties" in node:
            additional = node["additionalProperties"]
            if not isinstance(additional, bool):
                pending.append((additional, f"{location}.additionalProperties", depth + 1))
    return issues


def validate_against_schema(
    instance: Any,
    schema: dict[str, Any],
    path: str = "$",
) -> list[ValidationIssue]:
    """Reject unsupported schemas before validating the instance against the subset."""
    support_issues = _schema_support_issues(schema, path)
    if support_issues:
        return support_issues
    return _validate_instance(instance, schema, path)


def _validate_instance(
    instance: Any,
    schema: dict[str, Any],
    path: str = "$",
) -> list[ValidationIssue]:
    """Recursively validate a parsed JSON value against a JSON Schema subset.

    Args:
        instance: Parsed JSON value.
        schema: JSON Schema dictionary.
        path: JSONPath-like location string for error reporting.

    Returns:
        A list of ValidationIssue objects (empty if conformant).
    """
    issues: list[ValidationIssue] = []

    # 1. Check type constraint
    expected_type = schema.get("type")
    if expected_type is not None:
        allowed_types = [expected_type] if isinstance(expected_type, str) else list(expected_type)
        if not any(_check_json_type(instance, t) for t in allowed_types):
            actual_type = type(instance).__name__
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Expected type {expected_type!r}, got {actual_type}.",
                    rule="type",
                )
            )
            return issues

    # 2. Check enum and const constraints
    if "enum" in schema and instance not in schema["enum"]:
        issues.append(
            ValidationIssue(
                path=path,
                message=f"Value {instance!r} is not in allowed enum {schema['enum']!r}.",
                rule="enum",
            )
        )

    if "const" in schema and instance != schema["const"]:
        issues.append(
            ValidationIssue(
                path=path,
                message=f"Value {instance!r} does not match const {schema['const']!r}.",
                rule="const",
            )
        )

    # 3. String constraints
    if isinstance(instance, str):
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(instance) < min_len:
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"String length {len(instance)} is less than minLength {min_len}.",
                    rule="minLength",
                )
            )
        max_len = schema.get("maxLength")
        if isinstance(max_len, int) and len(instance) > max_len:
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"String length {len(instance)} exceeds maxLength {max_len}.",
                    rule="maxLength",
                )
            )
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, instance) is None:
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"String {instance!r} does not match pattern {pattern!r}.",
                    rule="pattern",
                )
            )

    # 4. Numeric constraints
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and instance < minimum:
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Value {instance} is less than minimum {minimum}.",
                    rule="minimum",
                )
            )
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and instance > maximum:
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Value {instance} exceeds maximum {maximum}.",
                    rule="maximum",
                )
            )

    # 5. Array constraints
    if isinstance(instance, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(instance) < min_items:
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Array length {len(instance)} is less than minItems {min_items}.",
                    rule="minItems",
                )
            )
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(instance) > max_items:
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Array length {len(instance)} exceeds maxItems {max_items}.",
                    rule="maxItems",
                )
            )
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for idx, elem in enumerate(instance):
                issues.extend(_validate_instance(elem, items_schema, f"{path}[{idx}]"))

    # 6. Object constraints
    if isinstance(instance, dict):
        required_keys = schema.get("required", [])
        for req_key in required_keys:
            if req_key not in instance:
                issues.append(
                    ValidationIssue(
                        path=f"{path}.{req_key}",
                        message=f"Missing required property {req_key!r}.",
                        rule="required",
                    )
                )

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for prop_key, prop_schema in properties.items():
                if prop_key in instance and isinstance(prop_schema, dict):
                    issues.extend(
                        _validate_instance(
                            instance[prop_key],
                            prop_schema,
                            f"{path}.{prop_key}",
                        )
                    )

        additional = schema.get("additionalProperties", True)
        if additional is False and isinstance(properties, dict):
            allowed_keys = set(properties.keys())
            for actual_key in instance:
                if actual_key not in allowed_keys:
                    issues.append(
                        ValidationIssue(
                            path=f"{path}.{actual_key}",
                            message=f"Additional property {actual_key!r} is not allowed.",
                            rule="additionalProperties",
                        )
                    )
        elif isinstance(additional, dict) and isinstance(properties, dict):
            allowed_keys = set(properties.keys())
            for actual_key, actual_val in instance.items():
                if actual_key not in allowed_keys:
                    issues.extend(
                        _validate_instance(
                            actual_val,
                            additional,
                            f"{path}.{actual_key}",
                        )
                    )

    return issues


def validate_output(
    raw_text: str,
    schema: dict[str, Any] | None = None,
) -> SchemaValidationResult:
    """Parse raw LLM output as JSON and optionally validate against a JSON Schema.

    Args:
        raw_text: Raw LLM response string.
        schema: Optional JSON Schema dictionary. If None, only JSON syntax is checked.

    Returns:
        SchemaValidationResult containing parsed payload and any validation issues.
    """
    is_valid_json, parsed, error_msg = extract_json_payload(raw_text)
    if not is_valid_json:
        return SchemaValidationResult(
            is_valid_json=False,
            is_schema_conformant=False,
            parsed_json=None,
            issues=(
                ValidationIssue(
                    path="$",
                    message=error_msg or "Invalid JSON syntax.",
                    rule="json_syntax",
                ),
            ),
        )

    if schema is None:
        return SchemaValidationResult(
            is_valid_json=True,
            is_schema_conformant=True,
            parsed_json=parsed,
            issues=(),
        )

    schema_issues = validate_against_schema(parsed, schema, path="$")
    return SchemaValidationResult(
        is_valid_json=True,
        is_schema_conformant=len(schema_issues) == 0,
        parsed_json=parsed,
        issues=tuple(schema_issues),
    )


