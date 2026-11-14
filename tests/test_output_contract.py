"""Captured outputs must be strings, and strict parsing applies in every extraction mode."""

import pytest

from llm_drift_bench import ModelSample, validate_output


@pytest.mark.parametrize(
    "value", [{"ok": True}, [1, 2], None, 7, 1.5, True],
    ids=["object", "array", "null", "int", "float", "bool"],
)
def test_non_string_outputs_are_rejected_not_stringified(value: object) -> None:
    for key in ("output", "response"):
        with pytest.raises(ValueError, match="must be a string"):
            ModelSample.from_dict({"prompt_id": "p", key: value})
    with pytest.raises(ValueError, match="must be a string"):
        ModelSample(prompt_id="p", output=value)  # type: ignore[arg-type]


def test_string_outputs_are_preserved_exactly() -> None:
    for text in ("", '{"ok": true}', "None", "  spaced  "):
        assert ModelSample.from_dict({"prompt_id": "p", "output": text}).output == text


def test_serialized_structures_validate_as_json() -> None:
    import json

    sample = ModelSample.from_dict({"prompt_id": "p", "output": json.dumps({"ok": True})})
    assert validate_output(sample.output).is_valid_json


@pytest.mark.parametrize(
    "text",
    [
        '{"total":100,"total":1}',
        '```json\n{"total":100,"total":1}\n```',
        'Here is the answer: {"total":100,"total":1} done',
        '{"outer":{"a":1,"a":2}}',
        '[{"a":1},{"b":1,"b":2}]',
        'Answer:\n```\n[{"a":1,"a":2}]\n```',
    ],
    ids=["direct", "fenced", "prose", "nested", "array", "plain-fence"],
)
def test_duplicate_keys_are_rejected_in_every_extraction_mode(text: str) -> None:
    result = validate_output(text)
    assert not result.is_valid_json
    assert result.parsed_json is None
    assert "Duplicate JSON object key" in result.issues[0].message


@pytest.mark.parametrize(
    "text",
    [
        '{"a":1,"b":2}',
        '```json\n{"a":1,"b":{"a":2}}\n```',
        'Result: {"a":1} ok',
        '[{"a":1},{"a":2}]',
    ],
)
def test_same_key_in_separate_objects_is_still_valid(text: str) -> None:
    assert validate_output(text).is_valid_json
