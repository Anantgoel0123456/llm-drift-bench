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
