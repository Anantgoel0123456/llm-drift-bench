"""Input files have one encoding, size and numeric policy."""

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_drift_bench._file_io import parse_json, read_utf8
from llm_drift_bench.engine import load_model_samples, load_prompt_cases
from llm_drift_bench.evaluator import load_test_prompts
from llm_drift_bench.report_io import load_report


def test_bom_and_crlf_for_both_suite_loaders(tmp_path: Path) -> None:
    path = tmp_path / "suite with spaces.jsonl"
    path.write_bytes(b'\xef\xbb\xbf{"prompt_id":"p","prompt":"hello"}\r\n')
    assert load_prompt_cases(path)[0].prompt_id == "p"
    assert load_test_prompts(path)[0].id == "p"


def test_bom_model_samples(tmp_path: Path) -> None:
    path = tmp_path / "samples.json"
    path.write_text('[{"prompt_id":"p","output":"{}"}]', encoding="utf-8-sig")
    assert load_model_samples(path)[0].output == "{}"


def test_encoding_and_size_fail_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b'\xff')
    with pytest.raises(ValueError, match="UTF-8"):
        read_utf8(path)
    path.write_bytes(b'123456')
    with (
        patch("llm_drift_bench._file_io.MAX_INPUT_BYTES", 5),
        pytest.raises(ValueError, match="limit"),
    ):
        read_utf8(path)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '[1e400]'])
def test_ambiguous_and_nonfinite_json_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        parse_json(text)


def test_malformed_records_have_actionable_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{}\n{broken', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_model_samples(path)
    path = tmp_path / "bad.json"
    path.write_text('{"samples":[1]}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON objects"):
        load_model_samples(path)


@pytest.mark.parametrize("change", [{"passed": True}, {"mean_composite_drift": 3}])
def test_report_status_and_ranges_checked(tmp_path: Path, change: dict) -> None:
    source = Path(__file__).resolve().parents[1] / "examples" / "expected-drift.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw.update(change)
    path = tmp_path / "bad-report.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_report(path)


def test_module_entry_point_help(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(sys, "argv", ["llm_drift_bench", "--help"]),
        pytest.raises(SystemExit) as error,
    ):
        runpy.run_module("llm_drift_bench", run_name="__main__")
    assert error.value.code == 0
    assert "init-sample" in capsys.readouterr().out
