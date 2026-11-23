"""Tests for explicit token and cost drift estimation."""

from __future__ import annotations

import pytest

from llm_drift_bench.cost import ModelPricing, estimate_cost_drift
from llm_drift_bench.models import ModelSample


def test_estimate_cost_drift_complete_tokens_and_pricing() -> None:
    base = [
        ModelSample(prompt_id="p1", output="a", tokens_generated=500),
        ModelSample(prompt_id="p2", output="b", tokens_generated=500),
    ]
    cand = [
        ModelSample(prompt_id="p1", output="a", tokens_generated=600),
        ModelSample(prompt_id="p2", output="b", tokens_generated=600),
    ]
    res = estimate_cost_drift(
        base,
        cand,
        baseline_pricing=ModelPricing(output_per_million_usd=10.0),
        candidate_pricing=ModelPricing(output_per_million_usd=5.0),
    )
    assert res.complete is True
    assert res.baseline_tokens == 1000
    assert res.candidate_tokens == 1200
    assert res.token_delta == 200
    assert res.token_delta_pct == pytest.approx(20.0)
    assert res.baseline_output_cost_usd == pytest.approx(0.01)
    assert res.candidate_output_cost_usd == pytest.approx(0.006)
    assert res.cost_delta_usd == pytest.approx(-0.004)
    assert res.cost_delta_pct == pytest.approx(-40.0)


def test_estimate_cost_drift_never_fabricates_missing_tokens() -> None:
    base = [ModelSample(prompt_id="p1", output="a", tokens_generated=100)]
    cand = [ModelSample(prompt_id="p1", output="a", tokens_generated=None)]
    res = estimate_cost_drift(base, cand)
    assert res.complete is False
    assert res.baseline_tokens is None and res.candidate_tokens is None

    with pytest.raises(ValueError, match="non-negative"):
        ModelPricing(output_per_million_usd=-1.0)
