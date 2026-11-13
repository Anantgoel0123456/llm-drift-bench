"""Regression assertions must preserve canonical decisions and useful failures."""

from dataclasses import replace

import pytest

from llm_drift_bench import DriftBenchmarkEngine, ModelSample, assert_no_drift


def test_passing_report_needs_no_timing_measurements() -> None:
    samples = [ModelSample(prompt_id="p", output='{"total":100}')]
    report = DriftBenchmarkEngine().evaluate(samples, samples)
    assert report.baseline_p95_latency_ms is None
    assert assert_no_drift(report) is None
    with pytest.raises(ValueError, match="no evaluated samples"):
        assert_no_drift(replace(report, total_samples=0))


def test_failure_identifies_prompt_without_response_body() -> None:
    baseline = [ModelSample(prompt_id="invoice", output='{"total":100}')]
    candidate = [ModelSample(prompt_id="invoice", output="private-response-body")]
    report = DriftBenchmarkEngine().evaluate(baseline, candidate)
    with pytest.raises(AssertionError) as error:
        assert_no_drift(report)
    message = str(error.value)
    assert "Prompt 'invoice'" in message
    assert "candidate schema passed=False" in message
    assert "private-response-body" not in message
    assert report.failure_reasons[0] in message


def test_suite_latency_failure_is_not_lost_when_no_samples_drift() -> None:
    baseline = [ModelSample(prompt_id="p", output="same", latency_ms=10)]
    candidate = [ModelSample(prompt_id="p", output="same", latency_ms=100)]
    report = DriftBenchmarkEngine().evaluate(baseline, candidate)
    assert report.drifted_samples == 0
    assert not report.passed
    with pytest.raises(AssertionError) as error:
        assert_no_drift(report)
    assert report.failure_reasons[0] in str(error.value)


def test_failure_details_are_bounded_and_deterministic() -> None:
    baseline = [ModelSample(prompt_id=str(i), output='{"total":100}') for i in range(7)]
    candidate = [ModelSample(prompt_id=str(i), output="invalid") for i in reversed(range(7))]
    report = DriftBenchmarkEngine().evaluate(baseline, candidate)
    with pytest.raises(AssertionError) as error:
        assert_no_drift(report)
    message = str(error.value)
    assert message.count("- Prompt") == 5
    assert message.index("Prompt '0'") < message.index("Prompt '4'")
    assert "2 additional drifted samples" in message
