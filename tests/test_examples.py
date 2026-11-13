"""Keep the documented offline demo reproducible."""

import contextlib
import io
import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main


@pytest.mark.parametrize(("name", "exit_code"), [("pass", 0), ("drift", 1)])
def test_documented_demo(name: str, exit_code: int) -> None:
    examples = Path(__file__).resolve().parents[1] / "examples"
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        actual_exit = main([
            "run", "-b", str(examples / "baseline.jsonl"),
            "-c", str(examples / f"candidate-{name}.jsonl"),
            "-p", str(examples / "extraction-suite.json"), "--format", "json",
        ])
    assert actual_exit == exit_code
    expected = json.loads((examples / f"expected-{name}.json").read_text(encoding="utf-8"))
    assert json.loads(stdout.getvalue()) == expected
