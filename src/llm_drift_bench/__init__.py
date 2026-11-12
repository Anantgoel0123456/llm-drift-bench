"""llm-drift-bench: Benchmark, validate JSON conformity, and detect LLM output drift."""

from __future__ import annotations

from llm_drift_bench.engine import (
    DriftBenchmarkEngine,
    evaluate_sample_pair,
    format_report_table,
    load_model_samples,
    load_prompt_cases,
)
from llm_drift_bench.metrics import (
    compute_composite_drift_score,
    compute_percentile,
    flatten_json_paths,
    jaccard_similarity,
    json_exact_field_match,
    json_numeric_drift,
    json_structural_similarity,
    ngram_overlap_similarity,
    normalized_levenshtein_similarity,
    tokenize_text,
)
from llm_drift_bench.models import (
    DriftBenchmarkReport,
    DriftThresholds,
    ModelSample,
    PromptCase,
    SampleDriftResult,
    SchemaValidationResult,
    ValidationIssue,
)
from llm_drift_bench.validator import (
    extract_json_payload,
    validate_against_schema,
    validate_output,
)

__version__ = "0.1.0"

__all__ = [
    "DriftBenchmarkEngine",
    "DriftBenchmarkReport",
    "DriftThresholds",
    "ModelSample",
    "PromptCase",
    "SampleDriftResult",
    "SchemaValidationResult",
    "ValidationIssue",
    "__version__",
    "compute_composite_drift_score",
    "compute_percentile",
    "evaluate_sample_pair",
    "extract_json_payload",
    "flatten_json_paths",
    "format_report_table",
    "jaccard_similarity",
    "json_exact_field_match",
    "json_numeric_drift",
    "json_structural_similarity",
    "load_model_samples",
    "load_prompt_cases",
    "ngram_overlap_similarity",
    "normalized_levenshtein_similarity",
    "tokenize_text",
    "validate_against_schema",
    "validate_output",
]
