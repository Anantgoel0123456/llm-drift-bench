"""Tests for automatic JSON Schema inference and the infer-schema CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main
from llm_drift_bench.models import ModelSample
from llm_drift_bench.schema_infer import (
    infer_prompt_suite,
    infer_schema_from_value,
    merge_schemas,
)
from llm_drift_bench.validator import validate_output


def test_infer_schema_primitives_and_nested_structures() -> None:
    payload = {
        "id": 42,
        "score": 0.95,
        "active": True,
        "note": None,
        "tags": ["a", "b"],
        "meta": {"source": "api"},
        "empty_list": [],
        "empty_obj": {},
    }
    schema = infer_schema_from_value(payload, strict_additional=True)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(payload.keys())
    assert schema["properties"]["id"] == {"type": "integer"}
    assert schema["properties"]["score"] == {"type": "number"}
    assert schema["properties"]["active"] == {"type": "boolean"}
    assert schema["properties"]["note"] == {"type": "null"}
    assert schema["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["meta"]["additionalProperties"] is False
    assert schema["properties"]["empty_list"] == {"type": "array"}
    assert schema["properties"]["empty_obj"] == {"type": "object", "additionalProperties": False}

    # Inferred schema must validate the source payload with zero issues.
    result = validate_output(json.dumps(payload), schema)
    assert result.passed


def test_infer_schema_rejects_non_finite_and_unsupported_values() -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        infer_schema_from_value(float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        infer_schema_from_value(float("inf"))
    with pytest.raises(ValueError, match="Unsupported JSON value"):
        infer_schema_from_value(object())


def test_merge_schemas_required_intersection_and_type_widening() -> None:
    assert merge_schemas([]) == {}
    s1 = infer_schema_from_value({"id": 1, "total": 10, "extra": "yes"})
    s2 = infer_schema_from_value({"id": 2, "total": 12.5, "other": True})
    merged = merge_schemas([s1, s2], strict_additional=True)

    assert merged["type"] == "object"
    assert merged["required"] == ["id", "total"]
    assert merged["properties"]["id"] == {"type": "integer"}
    assert merged["properties"]["total"] == {"type": "number"}
    assert merged["properties"]["extra"] == {"type": "string"}
    assert merged["properties"]["other"] == {"type": "boolean"}
    assert merged["additionalProperties"] is False

    assert validate_output(json.dumps({"id": 1, "total": 10, "extra": "yes"}), merged).passed
    assert validate_output(json.dumps({"id": 2, "total": 12.5, "other": True}), merged).passed
    assert not validate_output(json.dumps({"id": 2, "total": 12.5, "bad": 1}), merged).passed


def test_merge_schemas_preserves_nullable_object_and_array_constraints() -> None:
    nullable_obj = merge_schemas([
        infer_schema_from_value({"city": "NYC"}),
        infer_schema_from_value(None),
    ])
    assert nullable_obj["type"] == ["null", "object"]
    assert nullable_obj["required"] == ["city"]
    assert nullable_obj["properties"]["city"] == {"type": "string"}
    assert validate_output("null", nullable_obj).passed
    assert validate_output('{"city": "Boston"}', nullable_obj).passed
    assert not validate_output('{"city": 123}', nullable_obj).passed

    nullable_arr = merge_schemas([
        infer_schema_from_value([]),
        infer_schema_from_value(["x"]),
        infer_schema_from_value(None),
    ])
    assert nullable_arr["type"] == ["array", "null"]
    assert nullable_arr["items"] == {"type": "string"}
    assert validate_output("null", nullable_arr).passed
    assert validate_output('["ok"]', nullable_arr).passed
    assert not validate_output("[123]", nullable_arr).passed


def test_infer_prompt_suite_merges_repeated_prompt_ids_and_handles_plain_text() -> None:
    with pytest.raises(ValueError, match="empty"):
        infer_prompt_suite([])

    samples = [
        ModelSample(prompt_id="p1", model_name="m", output='{"a": 1, "b": "hi"}'),
        ModelSample(prompt_id="p1", model_name="m", output='{"a": 2.5}'),
        ModelSample(prompt_id="p2", model_name="m", output="Just plain prose response"),
    ]
    suite = infer_prompt_suite(samples)
    assert [item["prompt_id"] for item in suite] == ["p1", "p2"]
    assert suite[0]["schema"]["required"] == ["a"]
    assert suite[0]["schema"]["properties"]["a"] == {"type": "number"}
    assert "b" in suite[0]["schema"]["properties"]
    assert "schema" not in suite[1]


def test_infer_prompt_suite_shared_schema() -> None:
    samples = [
        ModelSample(prompt_id="p1", model_name="m", output='{"k": "v1", "n": 1}'),
        ModelSample(prompt_id="p2", model_name="m", output='{"k": "v2", "flag": true}'),
    ]
    suite = infer_prompt_suite(samples, shared_schema=True)
    assert suite[0]["schema"] == suite[1]["schema"]
    assert suite[0]["schema"]["required"] == ["k"]


def test_cli_infer_schema_round_trips_with_validate_strict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    baseline_path = examples_dir / "baseline.jsonl"
    out_suite = tmp_path / "inferred_prompts.json"

    exit_code = main([
        "infer-schema",
        "-i",
        str(baseline_path),
        "-o",
        str(out_suite),
        "--strict-additional",
    ])
    assert exit_code == 0
    assert out_suite.is_file()
    out = capsys.readouterr().out
    assert "Inferred" in out

    # Immediately validate the baseline against the inferred strict schema.
    val_code = main([
        "validate",
        "-i",
        str(baseline_path),
        "-p",
        str(out_suite),
        "--strict",
    ])
    assert val_code == 0


def test_cli_infer_schema_stdout_and_input_protection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_file = tmp_path / "base.jsonl"
    sample_file.write_text(
        json.dumps({"prompt_id": "p1", "model": "m", "output": '{"x": 1}'}) + "\n",
        encoding="utf-8",
    )

    assert main(["infer-schema", "-i", str(sample_file)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed) == 1
    assert parsed[0]["schema"]["properties"]["x"] == {"type": "integer"}

    # Cannot overwrite input capture file.
    assert main(["infer-schema", "-i", str(sample_file), "-o", str(sample_file)]) == 2
    assert "input file" in capsys.readouterr().err


def test_merge_schemas_single_schema_deepcopy_and_strict_additional() -> None:
    original = infer_schema_from_value({"meta": {"id": 1}}, strict_additional=False)
    strict = merge_schemas([original], strict_additional=True)
    assert strict["additionalProperties"] is False
    assert strict["properties"]["meta"]["additionalProperties"] is False
    assert "additionalProperties" not in original
    strict["properties"]["meta"]["properties"]["id"]["type"] = "string"
    assert original["properties"]["meta"]["properties"]["id"]["type"] == "integer"

