"""Regression checks for validation input and schema precedence."""

import json
from pathlib import Path

from llm_drift_bench.cli import main


def test_empty_validation_input_is_an_error(tmp_path: Path) -> None:
    samples = tmp_path / "empty.json"
    samples.write_text("[]", encoding="utf-8")
    assert main(["validate", "-i", str(samples), "--strict"]) == 2


def test_empty_prompt_schema_overrides_global_schema(tmp_path: Path) -> None:
    samples = tmp_path / "samples.json"
    suite = tmp_path / "suite.json"
    schema = tmp_path / "schema.json"
    samples.write_text(json.dumps([{"prompt_id": "p", "output": "{}"}]), encoding="utf-8")
    schema.write_text('{"required": ["missing"]}', encoding="utf-8")
    args = ["validate", "-i", str(samples), "-s", str(schema), "--strict"]
    assert main(args) == 1
    suite.write_text(json.dumps([{"prompt_id": "p", "schema": {}}]), encoding="utf-8")
    assert main([*args, "-p", str(suite)]) == 0
    suite.write_text(json.dumps([{"prompt_id": "p"}]), encoding="utf-8")
    assert main([*args, "-p", str(suite)]) == 1
