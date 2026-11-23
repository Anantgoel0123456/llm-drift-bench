"""Automatic JSON Schema inference from captured baseline JSON outputs."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from llm_drift_bench.models import ModelSample
from llm_drift_bench.validator import extract_json_payload


def _value_json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float values are not valid JSON.")
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise ValueError(f"Unsupported JSON value type: {type(value).__name__}.")


def _normalize_types(type_val: Any) -> set[str]:
    if isinstance(type_val, str):
        return {type_val}
    if isinstance(type_val, list):
        return {str(item) for item in type_val}
    return set()


def _collapse_types(types: set[str]) -> str | list[str]:
    cleaned = set(types)
    if "integer" in cleaned and "number" in cleaned:
        cleaned.discard("integer")
    ordered = sorted(cleaned)
    return ordered[0] if len(ordered) == 1 else ordered


def _apply_strict_additional(node: dict[str, Any]) -> dict[str, Any]:
    if "object" in _normalize_types(node.get("type")):
        node["additionalProperties"] = False
        props = node.get("properties")
        if isinstance(props, dict):
            for child in props.values():
                if isinstance(child, dict):
                    _apply_strict_additional(child)
    items = node.get("items")
    if isinstance(items, dict):
        _apply_strict_additional(items)
    return node


def merge_schemas(
    schemas: Sequence[dict[str, Any]],
    *,
    strict_additional: bool = False,
) -> dict[str, Any]:
    """Merge multiple inferred schemas into one supported-subset JSON Schema."""
    if not schemas:
        return {}
    if len(schemas) == 1:
        cloned = copy.deepcopy(schemas[0])
        return _apply_strict_additional(cloned) if strict_additional else cloned

    all_types: set[str] = set()
    for schema in schemas:
        all_types.update(_normalize_types(schema.get("type")))

    merged: dict[str, Any] = {}
    if all_types:
        merged["type"] = _collapse_types(all_types)

    obj_schemas = [s for s in schemas if "object" in _normalize_types(s.get("type"))]
    if obj_schemas:
        req_sets = [set(s.get("required", [])) for s in obj_schemas]
        common_req = sorted(set.intersection(*req_sets)) if req_sets else []
        if common_req:
            merged["required"] = common_req
        all_keys: set[str] = set()
        for s in obj_schemas:
            props = s.get("properties")
            if isinstance(props, dict):
                all_keys.update(props)
        if all_keys:
            merged_props: dict[str, Any] = {}
            for key in sorted(all_keys):
                child_schemas = [
                    s["properties"][key]
                    for s in obj_schemas
                    if isinstance(s.get("properties"), dict) and key in s["properties"]
                ]
                merged_props[key] = merge_schemas(
                    child_schemas, strict_additional=strict_additional
                )
            merged["properties"] = merged_props
        if strict_additional:
            merged["additionalProperties"] = False

    arr_schemas = [s for s in schemas if "array" in _normalize_types(s.get("type"))]
    if arr_schemas:
        item_schemas = [
            s["items"] for s in arr_schemas if isinstance(s.get("items"), dict) and s["items"]
        ]
        if item_schemas:
            merged["items"] = merge_schemas(item_schemas, strict_additional=strict_additional)

    return merged


def infer_schema_from_value(
    value: Any,
    *,
    strict_additional: bool = False,
) -> dict[str, Any]:
    """Infer a supported-subset JSON Schema directly from one parsed JSON value."""
    jtype = _value_json_type(value)
    schema: dict[str, Any] = {"type": jtype}
    if jtype == "object":
        keys = sorted(value.keys())
        if keys:
            schema["required"] = keys
            schema["properties"] = {
                k: infer_schema_from_value(value[k], strict_additional=strict_additional)
                for k in keys
            }
        if strict_additional:
            schema["additionalProperties"] = False
    elif jtype == "array" and value:
        elem_schemas = [
            infer_schema_from_value(elem, strict_additional=strict_additional)
            for elem in value
        ]
        schema["items"] = merge_schemas(elem_schemas, strict_additional=strict_additional)
    return schema


def infer_prompt_suite(
    samples: Sequence[ModelSample],
    *,
    strict_additional: bool = False,
    shared_schema: bool = False,
) -> list[dict[str, Any]]:
    """Build a ready-to-use prompts.json suite with inferred schemas from baseline captures."""
    if not samples:
        raise ValueError("Cannot infer schemas from an empty sample list.")

    grouped: dict[str, list[tuple[bool, Any]]] = defaultdict(list)
    all_json_schemas: list[dict[str, Any]] = []
    for sample in samples:
        ok, payload, _ = extract_json_payload(sample.output)
        grouped[sample.prompt_id].append((ok, payload))
        if ok:
            all_json_schemas.append(
                infer_schema_from_value(payload, strict_additional=strict_additional)
            )

    unified = (
        merge_schemas(all_json_schemas, strict_additional=strict_additional)
        if shared_schema and all_json_schemas
        else None
    )

    suite: list[dict[str, Any]] = []
    for prompt_id in sorted(grouped):
        records = grouped[prompt_id]
        all_json = all(ok for ok, _ in records)
        entry: dict[str, Any] = {
            "prompt_id": prompt_id,
            "prompt": f"Inferred prompt case for {prompt_id}",
            "category": "inferred",
        }
        if all_json:
            if unified is not None:
                entry["schema"] = unified
            else:
                per_prompt = [
                    infer_schema_from_value(payload, strict_additional=strict_additional)
                    for _, payload in records
                ]
                entry["schema"] = merge_schemas(
                    per_prompt, strict_additional=strict_additional
                )
        suite.append(entry)
    return suite
