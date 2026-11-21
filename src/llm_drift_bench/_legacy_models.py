"""Legacy Pydantic data models preserved for backward compatibility."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TestPrompt(BaseModel):
    """Pydantic model representing a benchmark prompt item in a test suite."""

    __test__ = False
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = Field(..., alias="prompt_id", min_length=1)
    prompt: str = Field(default="")
    expected_format: str = Field(default="json")
    expected_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    baseline_output: str | None = Field(default=None)
    candidate_output: str | None = Field(default=None)
    baseline_latency_ms: float | None = Field(default=None)
    candidate_latency_ms: float | None = Field(default=None)
    category: str = Field(default="general")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def prompt_id(self) -> str:
        """Return canonical prompt identifier."""
        return self.id


class DriftMetrics(BaseModel):
    """Pydantic model capturing pairwise evaluation and drift metrics for a single prompt."""

    model_config = ConfigDict(extra="ignore")

    prompt_id: str
    category: str = "general"
    exact_match: bool
    baseline_valid_json: bool
    candidate_valid_json: bool
    format_adherence: bool
    semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    token_diff_ratio: float = Field(..., ge=0.0, le=1.0)
    drift_score: float = Field(..., ge=0.0, le=1.0)
    baseline_latency_ms: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    candidate_latency_ms: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    latency_delta_ms: float | None = Field(default=None, allow_inf_nan=False)
    baseline_tokens_per_sec: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    candidate_tokens_per_sec: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    tokens_per_sec_delta: float | None = Field(default=None, allow_inf_nan=False)


class BenchmarkResult(BaseModel):
    """Pydantic model representing the aggregate benchmark result across a prompt suite."""

    model_config = ConfigDict(extra="ignore")

    suite_name: str = "default"
    baseline_model: str
    candidate_model: str
    total_prompts: int = Field(..., ge=0)
    exact_match_rate: float = Field(..., ge=0.0, le=1.0)
    baseline_json_adherence_rate: float = Field(..., ge=0.0, le=1.0)
    candidate_json_adherence_rate: float = Field(..., ge=0.0, le=1.0)
    mean_semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    mean_token_diff_ratio: float = Field(..., ge=0.0, le=1.0)
    mean_drift_score: float = Field(..., ge=0.0, le=1.0)
    mean_baseline_latency_ms: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    mean_candidate_latency_ms: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    mean_baseline_tokens_per_sec: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    mean_candidate_tokens_per_sec: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    metrics: list[DriftMetrics] = Field(default_factory=list)
