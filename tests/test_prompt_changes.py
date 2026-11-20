"""Direction and evidence for comparisons between candidate verdicts."""

from pathlib import Path

import pytest

from llm_drift_bench.comparison import prompt_changes
from llm_drift_bench.report_io import load_engine_report

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def reports():
    return [load_engine_report(EXAMPLES / f"expected-{name}.json") for name in ("drift", "pass")]


def test_direction_and_order() -> None:
    rows = reports()
    regressions = prompt_changes(rows, "demo-candidate-pass")
    assert [(r.prompt_id, r.state) for r in regressions] == [
        ("invoice-01", "regressed"), ("ticket-01", "regressed")
    ]
    assert {r.state for r in prompt_changes(rows, "demo-candidate-drift")} == {"recovered"}


def test_unknown_reference_is_an_input_error() -> None:
    with pytest.raises(ValueError, match="Unknown reference"):
        prompt_changes(reports(), "unknown")
