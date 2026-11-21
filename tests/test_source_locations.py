"""Actionable input errors preserve locations without printing capture content."""

from pathlib import Path

import pytest

from llm_drift_bench._file_io import read_utf8


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf"])
def test_invalid_byte_offset_includes_bom(tmp_path: Path, prefix: bytes) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(prefix + b"abc\xff")
    with pytest.raises(ValueError, match=rf"offset {len(prefix) + 3}"):
        read_utf8(path)


def test_duplicate_keys_do_not_echo_capture_content() -> None:
    from llm_drift_bench._file_io import parse_json

    with pytest.raises(ValueError, match="Duplicate JSON object key") as error:
        parse_json('{"private value":1,"private value":2}')
    assert "private value" not in str(error.value)


def test_malformed_json_identifies_document_and_column(tmp_path: Path) -> None:
    from llm_drift_bench.engine import load_model_samples

    path = tmp_path / "bad capture.json"
    path.write_text('[\n  !]', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2, column 3") as error:
        load_model_samples(path)
    assert str(path) in str(error.value)


def test_jsonl_columns_include_leading_whitespace(tmp_path: Path) -> None:
    from llm_drift_bench.engine import load_prompt_cases

    path = tmp_path / "suite.jsonl"
    path.write_bytes(b'\r\n    !\r\n')
    with pytest.raises(ValueError, match="line 2, column 5"):
        load_prompt_cases(path)


@pytest.mark.parametrize("value", ["private value", [], {}])
def test_bad_latency_reports_field_without_value(value: object) -> None:
    from llm_drift_bench import ModelSample

    with pytest.raises(ValueError, match="latency_ms") as error:
        ModelSample.from_dict({"id": "p", "output": "", "latency_ms": value})
    assert "private value" not in str(error.value)


def test_bad_token_count_reports_field_without_value() -> None:
    from llm_drift_bench import ModelSample

    with pytest.raises(ValueError, match="tokens_generated") as error:
        ModelSample.from_dict({"id": "p", "output": "", "tokens_generated": "private value"})
    assert "private value" not in str(error.value)


@pytest.mark.parametrize(("ext", "data", "where"), [
    ("json", '{"samples":[{"id":"p"}]}', "record 1"),
    ("jsonl", '\n\n{"id":"p"}', "line 3"),
])
def test_invalid_sample_keeps_source_location(tmp_path: Path, ext: str, data: str, where: str):
    from llm_drift_bench.engine import load_model_samples

    path = tmp_path / f"capture.{ext}"
    path.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError, match=where) as error:
        load_model_samples(path)
    assert str(path) in str(error.value)


def test_invalid_suite_schema_locates_prompt(tmp_path: Path) -> None:
    from llm_drift_bench.engine import load_prompt_cases

    path = tmp_path / "suite.jsonl"
    path.write_text('\n{"id":"p","schema":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2.*schema"):
        load_prompt_cases(path)


def test_legacy_suite_hides_pydantic_input_values(tmp_path: Path) -> None:
    from llm_drift_bench.evaluator import load_test_prompts

    path = tmp_path / "suite.jsonl"
    path.write_text('\n{"prompt":"secret user content"}', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2") as error:
        load_test_prompts(path)
    assert "secret user content" not in str(error.value)


@pytest.mark.parametrize("data", ['[{},42]', '{"samples":[{},42]}'])
def test_nonobject_rows_identify_array_position(tmp_path: Path, data: str) -> None:
    from llm_drift_bench.engine import load_model_samples

    path = tmp_path / "samples.json"
    path.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError, match="record 2.*JSON objects"):
        load_model_samples(path)


def test_invalid_container_names_field_and_expected_shape(tmp_path: Path) -> None:
    from llm_drift_bench.engine import load_model_samples

    path = tmp_path / "samples.json"
    path.write_text('{"outputs":"private value"}', encoding="utf-8")
    with pytest.raises(ValueError, match="outputs.*JSON array") as error:
        load_model_samples(path)
    assert "private value" not in str(error.value)


def test_schema_errors_identify_the_schema_file(tmp_path: Path, capsys) -> None:
    from llm_drift_bench.cli import main

    capture = tmp_path / "sample.jsonl"
    capture.write_text('{"id":"p","output":"valid"}', encoding="utf-8")
    schema = tmp_path / "rules.json"
    schema.write_text("{", encoding="utf-8")
    assert main(["validate", "-i", str(capture), "-s", str(schema)]) == 2
    assert str(schema) in capsys.readouterr().err


def test_malformed_saved_report_identifies_source(tmp_path: Path) -> None:
    from llm_drift_bench.report_io import load_report

    path = tmp_path / "previous.json"
    path.write_text('{"bad":}', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1, column 8") as error:
        load_report(path)
    assert str(path) in str(error.value)


def test_tool_trace_errors_include_file_and_row(tmp_path: Path) -> None:
    from llm_drift_bench.tool_log import load_tool_events

    path = tmp_path / "trace.jsonl"
    path.write_text('\n{"kind":"read"}', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2.*resource") as error:
        load_tool_events(path)
    assert str(path) in str(error.value)


def test_malformed_tool_check_names_file(tmp_path: Path) -> None:
    from llm_drift_bench.tool_log import load_tool_check

    path = tmp_path / "tool-check.json"
    path.write_text('!', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1, column 1") as error:
        load_tool_check(path)
    assert str(path) in str(error.value)
