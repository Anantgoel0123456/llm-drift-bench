"""Explicit token and output-cost drift estimation (never fabricates token counts)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from llm_drift_bench._matching import require_same_ids
from llm_drift_bench._numbers import finite_number
from llm_drift_bench.models import ModelSample


@dataclass(frozen=True)
class ModelPricing:
    """Explicit USD-per-1M-tokens pricing supplied by the caller."""

    output_per_million_usd: float
    input_per_million_usd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("output_per_million_usd", "input_per_million_usd"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number.")
            num = finite_number(val)
            if num < 0:
                raise ValueError(f"{name} must be a finite non-negative number.")


@dataclass(frozen=True)
class CostComparison:
    """Token volume and explicit output-cost comparison between baseline and candidate."""

    total_samples: int
    complete: bool
    baseline_tokens: int | None
    candidate_tokens: int | None
    token_delta: int | None
    token_delta_pct: float | None
    baseline_output_cost_usd: float | None
    candidate_output_cost_usd: float | None
    cost_delta_usd: float | None
    cost_delta_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_cost_drift(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
    *,
    baseline_pricing: ModelPricing | None = None,
    candidate_pricing: ModelPricing | None = None,
) -> CostComparison:
    """Compare explicit generated token counts and optional USD output costs.

    If any sample lacks an explicit `tokens_generated` measurement, token and cost
    totals remain None (complete=False) rather than estimating tokens from text.
    """
    if not baseline_samples or not candidate_samples:
        raise ValueError("Baseline and candidate captures must both be non-empty.")
    base_map = {s.prompt_id: s for s in baseline_samples}
    cand_map = {s.prompt_id: s for s in candidate_samples}
    if len(base_map) != len(baseline_samples) or len(cand_map) != len(candidate_samples):
        raise ValueError("Captures contain duplicate prompt_id entries.")
    require_same_ids(set(base_map), set(cand_map), "Candidate capture")

    base_complete = all(s.tokens_generated is not None for s in baseline_samples)
    cand_complete = all(s.tokens_generated is not None for s in candidate_samples)
    if not (base_complete and cand_complete):
        return CostComparison(
            total_samples=len(base_map),
            complete=False,
            baseline_tokens=None,
            candidate_tokens=None,
            token_delta=None,
            token_delta_pct=None,
            baseline_output_cost_usd=None,
            candidate_output_cost_usd=None,
            cost_delta_usd=None,
            cost_delta_pct=None,
        )

    b_toks = sum(s.tokens_generated for s in baseline_samples if s.tokens_generated is not None)
    c_toks = sum(s.tokens_generated for s in candidate_samples if s.tokens_generated is not None)
    tok_delta = c_toks - b_toks
    tok_pct = round((tok_delta / b_toks) * 100.0, 2) if b_toks > 0 else None

    b_cost = (
        round((b_toks / 1_000_000.0) * baseline_pricing.output_per_million_usd, 6)
        if baseline_pricing is not None
        else None
    )
    c_cost = (
        round((c_toks / 1_000_000.0) * candidate_pricing.output_per_million_usd, 6)
        if candidate_pricing is not None
        else None
    )
    cost_delta = round(c_cost - b_cost, 6) if (b_cost is not None and c_cost is not None) else None
    cost_pct = (
        round(((c_cost - b_cost) / b_cost) * 100.0, 2)
        if (b_cost is not None and c_cost is not None and b_cost > 0)
        else None
    )
    return CostComparison(
        total_samples=len(base_map),
        complete=True,
        baseline_tokens=b_toks,
        candidate_tokens=c_toks,
        token_delta=tok_delta,
        token_delta_pct=tok_pct,
        baseline_output_cost_usd=b_cost,
        candidate_output_cost_usd=c_cost,
        cost_delta_usd=cost_delta,
        cost_delta_pct=cost_pct,
    )
