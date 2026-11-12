"""llm-drift-bench: Benchmark, validate JSON conformity, and detect LLM output drift."""

from __future__ import annotations

from llm_drift_bench.engine import (
    DriftBenchmarkEngine,
    evaluate_sample_pair,
    format_report_table,
    load_model_samples,
    load_prompt_cases,
)
from llm_drift_bench.evaluator import (
    check_json_adherence,
    compute_semantic_similarity,
    compute_token_diff_ratio,
    compute_tokens_per_sec,
    evaluate_response_pair,
    evaluate_suite,
    load_benchmark_result,
    load_test_prompts,
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
    BenchmarkResult,
    DriftBenchmarkReport,
    DriftMetrics,
    DriftThresholds,
    ModelSample,
    PromptCase,
    SampleDriftResult,
    SchemaValidationResult,
    TestPrompt,
    ValidationIssue,
)
from llm_drift_bench.reporter import (
    export_markdown_report,
    render_console_report,
)
from llm_drift_bench.validator import (
    extract_json_payload,
    validate_against_schema,
    validate_output,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkResult",
    "DriftBenchmarkEngine",
    "DriftBenchmarkReport",
    "DriftMetrics",
    "DriftThresholds",
    "ModelSample",
    "PromptCase",
    "SampleDriftResult",
    "SchemaValidationResult",
    "TestPrompt",
    "ValidationIssue",
    "__version__",
    "check_json_adherence",
    "compute_composite_drift_score",
    "compute_percentile",
    "compute_semantic_similarity",
    "compute_token_diff_ratio",
    "compute_tokens_per_sec",
    "evaluate_response_pair",
    "evaluate_sample_pair",
    "evaluate_suite",
    "export_markdown_report",
    "extract_json_payload",
    "flatten_json_paths",
    "format_report_table",
    "jaccard_similarity",
    "json_exact_field_match",
    "json_numeric_drift",
    "json_structural_similarity",
    "load_benchmark_result",
    "load_model_samples",
    "load_prompt_cases",
    "load_test_prompts",
    "ngram_overlap_similarity",
    "normalized_levenshtein_similarity",
    "render_console_report",
    "tokenize_text",
    "validate_against_schema",
    "validate_output",
]
