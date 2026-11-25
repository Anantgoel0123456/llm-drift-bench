"""Tests for explicit token and cost drift estimation."""

from __future__ import annotations

import pytest

from llm_drift_bench.cost import (
    ModelPricing,
    check_cost_budget,
    estimate_cost_drift,
    rank_candidate_costs,
)
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


def test_check_cost_budget_and_rank_candidate_costs() -> None:
    base = [ModelSample(prompt_id="p1", output="a", tokens_generated=1000)]
    cheap = [ModelSample(prompt_id="p1", output="a", tokens_generated=1100)]
    verbose = [ModelSample(prompt_id="p1", output="a", tokens_generated=2000)]
    missing = [ModelSample(prompt_id="p1", output="a", tokens_generated=None)]

    comp_ok = estimate_cost_drift(
        base,
        cheap,
        baseline_pricing=ModelPricing(output_per_million_usd=10.0),
        candidate_pricing=ModelPricing(output_per_million_usd=10.0),
    )
    ok, _ = check_cost_budget(comp_ok, max_token_increase_pct=15.0, max_cost_increase_pct=15.0)
    assert ok is True

    comp_bad = estimate_cost_drift(
        base,
        verbose,
        baseline_pricing=ModelPricing(output_per_million_usd=10.0),
        candidate_pricing=ModelPricing(output_per_million_usd=10.0),
    )
    bad, reason = check_cost_budget(comp_bad, max_token_increase_pct=20.0)
    assert bad is False and "+100.00%" in reason

    inc_ok, inc_reason = check_cost_budget(
        estimate_cost_drift(base, missing), max_token_increase_pct=10.0
    )
    assert inc_ok is False and "incomplete" in inc_reason

    ranked = rank_candidate_costs(
        base,
        {"verbose": verbose, "missing": missing, "cheap": cheap},
        baseline_pricing=ModelPricing(output_per_million_usd=10.0),
        candidate_pricing={
            "cheap": ModelPricing(output_per_million_usd=5.0),
            "verbose": ModelPricing(output_per_million_usd=10.0),
        },
    )
    assert [r["candidate"] for r in ranked] == ["cheap", "verbose", "missing"]


def test_cost_validation_zero_baseline_and_unpriced_ranking() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        ModelPricing(output_per_million_usd=True)  # type: ignore[arg-type]

    sample = [ModelSample(prompt_id="p1", output="a", tokens_generated=10)]
    with pytest.raises(ValueError, match="non-empty"):
        estimate_cost_drift([], sample)
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        estimate_cost_drift(sample * 2, sample)

    # Zero baseline tokens yields unavailable percentage delta on token budget check
    zero_base = [ModelSample(prompt_id="p1", output="", tokens_generated=0)]
    zero_comp = estimate_cost_drift(zero_base, sample)
    ok_none, msg_none = check_cost_budget(zero_comp)
    assert ok_none is True and "No token or cost" in msg_none

    ok_z, msg_z = check_cost_budget(zero_comp, max_token_increase_pct=10.0)
    assert ok_z is False and "baseline tokens = 0" in msg_z

    # Missing pricing yields unavailable cost percentage delta on cost budget check
    unpriced = estimate_cost_drift(
        sample, [ModelSample(prompt_id="p1", output="a", tokens_generated=20)]
    )
    ok_u, msg_u = check_cost_budget(unpriced, max_cost_increase_pct=5.0)
    assert ok_u is False and "Cost percentage delta unavailable" in msg_u

    # Exceeding explicit USD cost budget
    priced_over = estimate_cost_drift(
        sample,
        [ModelSample(prompt_id="p1", output="a", tokens_generated=20)],
        baseline_pricing=ModelPricing(output_per_million_usd=10.0),
        candidate_pricing=ModelPricing(output_per_million_usd=10.0),
    )
    ok_over, msg_over = check_cost_budget(priced_over, max_cost_increase_pct=25.0)
    assert ok_over is False and "Output cost increased" in msg_over

    # Input validation on check_cost_budget and rank_candidate_costs
    with pytest.raises(TypeError, match="CostComparison"):
        check_cost_budget("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite non-negative"):
        check_cost_budget(unpriced, max_token_increase_pct=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite non-negative"):
        check_cost_budget(unpriced, max_cost_increase_pct=-1.0)
    with pytest.raises(ValueError, match="at least one candidate"):
        rank_candidate_costs(sample, {})
    with pytest.raises(ValueError, match="non-empty strings"):
        rank_candidate_costs(sample, {"  ": sample})

    # Ranking without pricing orders complete candidates by token count
    small = [ModelSample(prompt_id="p1", output="a", tokens_generated=5)]
    large = [ModelSample(prompt_id="p1", output="a", tokens_generated=50)]
    ranked_unpriced = rank_candidate_costs(sample, {"large": large, "small": small})
    assert [r["candidate"] for r in ranked_unpriced] == ["small", "large"]


