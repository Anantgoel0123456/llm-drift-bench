"""Domain data models and schemas for llm-drift-bench."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class PromptCase:
    """Represents a single benchmark prompt case and optional expected JSON schema."""

    prompt_id: str
    prompt: str
    schema: dict[str, Any] | None = None
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptCase:
        """Construct a PromptCase from a dictionary payload."""
        prompt_id = str(data.get("prompt_id") or data.get("id") or "")
        if not prompt_id:
            raise ValueError("PromptCase requires a non-empty 'prompt_id' or 'id'.")
        prompt = str(data.get("prompt", ""))
        schema = data.get("schema")
        if schema is not None and not isinstance(schema, dict):
            raise ValueError(f"PromptCase '{prompt_id}' schema must be a JSON object/dict.")
        category = str(data.get("category", "general"))
        metadata_raw = data.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        return cls(
            prompt_id=prompt_id,
            prompt=prompt,
            schema=schema,
            category=category,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ModelSample:
    """Represents a model's output sample for a specific prompt_id."""

    prompt_id: str
    output: str
    model_name: str = "unknown"
    latency_ms: float | None = None
    tokens_generated: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject invalid measurements instead of treating them as real observations."""
        if self.latency_ms is not None and (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be a finite non-negative number or None.")
        if self.tokens_generated is not None and (
            isinstance(self.tokens_generated, bool)
            or not isinstance(self.tokens_generated, int)
            or self.tokens_generated < 0
        ):
            raise ValueError("tokens_generated must be a non-negative integer or None.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_model_name: str = "unknown") -> ModelSample:
        """Construct a ModelSample from a dictionary payload."""
        prompt_id = str(data.get("prompt_id") or data.get("id") or "")
        if not prompt_id:
            raise ValueError("ModelSample requires a non-empty 'prompt_id' or 'id'.")
        if "output" not in data and "response" not in data:
            raise ValueError(f"ModelSample '{prompt_id}' requires an 'output' or 'response' field.")
        raw_output = data.get("output", data.get("response", ""))
        output = raw_output if isinstance(raw_output, str) else str(raw_output)
        model_name = str(data.get("model_name") or data.get("model") or default_model_name)
        latency_raw = data.get("latency_ms")
        if isinstance(latency_raw, bool):
            raise ValueError("latency_ms must not be a boolean.")
        latency_ms = float(latency_raw) if latency_raw is not None else None
        tokens_raw = data.get("tokens_generated")
        if tokens_raw is not None and (
            isinstance(tokens_raw, bool) or not isinstance(tokens_raw, (int, str))
        ):
            raise ValueError("tokens_generated must be an integer or integer string.")
        tokens_generated = int(tokens_raw) if tokens_raw is not None else None
        metadata_raw = data.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        return cls(
            prompt_id=prompt_id,
            output=output,
            model_name=model_name,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ValidationIssue:
    """Represents a single JSON syntax or schema validation violation."""

    path: str
    message: str
    rule: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the validation issue to a dictionary."""
        return {
            "path": self.path,
            "message": self.message,
            "rule": self.rule,
        }


@dataclass(frozen=True)
class SchemaValidationResult:
    """Result of parsing and validating an LLM output against a JSON schema."""

    is_valid_json: bool
    is_schema_conformant: bool
    parsed_json: Any = None
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def passed(self) -> bool:
        """Return True if output is both valid JSON and conforms to the schema."""
        return self.is_valid_json and self.is_schema_conformant

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation result to a JSON-compatible dictionary."""
        return {
            "is_valid_json": self.is_valid_json,
            "is_schema_conformant": self.is_schema_conformant,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class DriftThresholds:
    """Configurable thresholds for deciding if drift exceeds acceptable limits."""

    max_composite_drift: float = 0.25
    min_json_conformity_rate: float = 0.95
    max_numeric_relative_drift: float = 0.15
    max_latency_regression_pct: float = 50.0


@dataclass(frozen=True)
class SampleDriftResult:
    """Pairwise comparison and drift metrics for a single prompt_id."""

    prompt_id: str
    category: str
    baseline_validation: SchemaValidationResult
    candidate_validation: SchemaValidationResult
    token_jaccard_similarity: float
    ngram_similarity: float
    levenshtein_similarity: float
    json_structural_similarity: float | None
    json_exact_field_match: float | None
    json_numeric_drift: float | None
    composite_drift_score: float
    latency_delta_ms: float | None = None
    drifted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize pairwise drift result to a dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "baseline_validation": self.baseline_validation.to_dict(),
            "candidate_validation": self.candidate_validation.to_dict(),
            "token_jaccard_similarity": round(self.token_jaccard_similarity, 4),
            "ngram_similarity": round(self.ngram_similarity, 4),
            "levenshtein_similarity": round(self.levenshtein_similarity, 4),
            "json_structural_similarity": (
                round(self.json_structural_similarity, 4)
                if self.json_structural_similarity is not None
                else None
            ),
            "json_exact_field_match": (
                round(self.json_exact_field_match, 4)
                if self.json_exact_field_match is not None
                else None
            ),
            "json_numeric_drift": (
                round(self.json_numeric_drift, 4)
                if self.json_numeric_drift is not None
                else None
            ),
            "composite_drift_score": round(self.composite_drift_score, 4),
            "latency_delta_ms": (
                round(self.latency_delta_ms, 2)
                if self.latency_delta_ms is not None
                else None
            ),
            "drifted": self.drifted,
        }


@dataclass(frozen=True)
class DriftBenchmarkReport:
    """Aggregate report summarizing a full baseline-vs-candidate benchmark run."""

    baseline_model: str
    candidate_model: str
    total_samples: int
    drifted_samples: int
    baseline_json_conformity_rate: float
    candidate_json_conformity_rate: float
    mean_composite_drift: float
    max_composite_drift: float
    mean_token_jaccard: float
    mean_levenshtein_similarity: float
    mean_json_structural_similarity: float | None
    mean_json_numeric_drift: float | None
    baseline_p50_latency_ms: float | None
    candidate_p50_latency_ms: float | None
    baseline_p95_latency_ms: float | None
    candidate_p95_latency_ms: float | None
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    results: tuple[SampleDriftResult, ...] = ()

    @property
    def drift_rate(self) -> float:
        """Fraction of total samples that exceeded the drift threshold."""
        if self.total_samples == 0:
            return 0.0
        return self.drifted_samples / self.total_samples

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full benchmark report to a dictionary."""
        return {
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "total_samples": self.total_samples,
            "drifted_samples": self.drifted_samples,
            "drift_rate": round(self.drift_rate, 4),
            "baseline_json_conformity_rate": round(self.baseline_json_conformity_rate, 4),
            "candidate_json_conformity_rate": round(self.candidate_json_conformity_rate, 4),
            "mean_composite_drift": round(self.mean_composite_drift, 4),
            "max_composite_drift": round(self.max_composite_drift, 4),
            "mean_token_jaccard": round(self.mean_token_jaccard, 4),
            "mean_levenshtein_similarity": round(self.mean_levenshtein_similarity, 4),
            "mean_json_structural_similarity": (
                round(self.mean_json_structural_similarity, 4)
                if self.mean_json_structural_similarity is not None
                else None
            ),
            "mean_json_numeric_drift": (
                round(self.mean_json_numeric_drift, 4)
                if self.mean_json_numeric_drift is not None
                else None
            ),
            "baseline_p50_latency_ms": (
                round(self.baseline_p50_latency_ms, 2)
                if self.baseline_p50_latency_ms is not None
                else None
            ),
            "candidate_p50_latency_ms": (
                round(self.candidate_p50_latency_ms, 2)
                if self.candidate_p50_latency_ms is not None
                else None
            ),
            "baseline_p95_latency_ms": (
                round(self.baseline_p95_latency_ms, 2)
                if self.baseline_p95_latency_ms is not None
                else None
            ),
            "candidate_p95_latency_ms": (
                round(self.candidate_p95_latency_ms, 2)
                if self.candidate_p95_latency_ms is not None
                else None
            ),
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full benchmark report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


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




