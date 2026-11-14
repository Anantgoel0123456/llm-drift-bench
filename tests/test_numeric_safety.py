"""Invalid numeric observations must never turn into plausible drift scores."""

import pytest

from llm_drift_bench.engine import DriftBenchmarkEngine
from llm_drift_bench.metrics import json_numeric_drift
from llm_drift_bench.models import ModelSample, PromptCase
from llm_drift_bench.validator import extract_json_payload


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400", "9" * 400])
@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Answer: {} End."])
def test_invalid_numbers_fail_in_every_extraction_mode(number: str, wrapper: str) -> None:
    payload = '{"number":' + number + ',"nested":{}}'
    valid, parsed, error = extract_json_payload(wrapper.format(payload))
    assert not valid
    assert parsed is None
    assert error


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10**400])
def test_numeric_metric_rejects_invalid_shared_and_unmatched_values(value: float) -> None:
    for candidate in ({"x": 1}, {"other": 1}):
        with pytest.raises(ValueError, match="finite"):
            json_numeric_drift({"x": value}, candidate)
        with pytest.raises(ValueError, match="finite"):
            json_numeric_drift(candidate, {"x": value})


def test_finite_extremes_and_existing_scores() -> None:
    assert json_numeric_drift({"x": 1e308}, {"x": -1e308}) == 1.0
    assert json_numeric_drift({"x": 1e308}, {"x": 1e308}) == 0.0
    assert json_numeric_drift({"x": 100}, {"x": 180}) == pytest.approx(80 / 180)
    assert extract_json_payload('{"x":1e308}')[0]
    assert extract_json_payload('{"text":"NaN Infinity"}')[0]


def test_nonfinite_candidate_fails_engine_gate_without_numeric_score() -> None:
    report = DriftBenchmarkEngine().evaluate(
        [ModelSample(prompt_id="p", output='{"x":1}')],
        [ModelSample(prompt_id="p", output='{"x":NaN}')],
        [PromptCase(prompt_id="p", prompt="", schema={"type": "object"})],
    )
    assert not report.passed
    assert report.drifted_samples == 1
    assert report.mean_json_numeric_drift is None
    assert not report.results[0].candidate_validation.passed
