"""llm-drift-bench: Benchmark, validate JSON conformity, and detect LLM output drift."""

from __future__ import annotations

from llm_drift_bench.adapters import (
    from_anthropic_transcript,
    from_openai_tool_calls,
    from_otel_genai_spans,
)
from llm_drift_bench.batch import evaluate_many
from llm_drift_bench.capture import (
    CaptureConfig,
    capture_suite,
    serialize_samples_jsonl,
)
from llm_drift_bench.comparison import (
    CandidateSummary,
    PromptChange,
    compare_reports,
    prompt_changes,
)
from llm_drift_bench.diagnostics import Finding, diagnose_report
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
from llm_drift_bench.history import (
    HistoryEntry,
    export_history_html,
    format_history_text,
    load_history,
    record_report,
    sparkline,
)
from llm_drift_bench.junit import export_junit_report
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
from llm_drift_bench.policy import (
    DriftPolicy,
    FieldRule,
    MetricWeights,
    load_policy,
)
from llm_drift_bench.report_io import load_report
from llm_drift_bench.reporter import (
    export_comparison_html,
    export_html_report,
    export_markdown_report,
    export_pr_comment,
    render_console_report,
)
from llm_drift_bench.scores import EvidenceScore
from llm_drift_bench.stats import (
    BootstrapCI,
    MultiRunSummary,
    PairedSignificance,
    PromptStability,
    bootstrap_mean_ci,
    paired_permutation_test,
    summarize_multi_run,
)
from llm_drift_bench.testing import assert_no_drift
from llm_drift_bench.tool_events import (
    ToolCheckResult,
    ToolEvent,
    check_max_tool_retries,
    check_no_secret_in_tool_args,
    check_read_before_edit,
)
from llm_drift_bench.tool_log import (
    ToolCheckRecord,
    check_trace,
    check_trace_file,
    load_tool_check,
    load_tool_events,
    parse_tool_events,
)
from llm_drift_bench.validator import (
    extract_json_payload,
    validate_against_schema,
    validate_output,
)

__version__ = "0.3.0.dev0"

__all__ = [
    "BenchmarkResult",
    "BootstrapCI",
    "CandidateSummary",
    "CaptureConfig",
    "DriftBenchmarkEngine",
    "DriftBenchmarkReport",
    "DriftMetrics",
    "DriftPolicy",
    "DriftThresholds",
    "EvidenceScore",
    "FieldRule",
    "Finding",
    "HistoryEntry",
    "MetricWeights",
    "ModelSample",
    "MultiRunSummary",
    "PairedSignificance",
    "PromptCase",
    "PromptChange",
    "PromptStability",
    "SampleDriftResult",
    "SchemaValidationResult",
    "TestPrompt",
    "ToolCheckRecord",
    "ToolCheckResult",
    "ToolEvent",
    "ValidationIssue",
    "__version__",
    "assert_no_drift",
    "bootstrap_mean_ci",
    "capture_suite",
    "check_json_adherence",
    "check_max_tool_retries",
    "check_no_secret_in_tool_args",
    "check_read_before_edit",
    "check_trace",
    "check_trace_file",
    "compare_reports",
    "compute_composite_drift_score",
    "compute_percentile",
    "compute_semantic_similarity",
    "compute_token_diff_ratio",
    "compute_tokens_per_sec",
    "diagnose_report",
    "evaluate_many",
    "evaluate_response_pair",
    "evaluate_sample_pair",
    "evaluate_suite",
    "export_comparison_html",
    "export_history_html",
    "export_html_report",
    "export_junit_report",
    "export_markdown_report",
    "export_pr_comment",
    "extract_json_payload",
    "flatten_json_paths",
    "format_history_text",
    "format_report_table",
    "from_anthropic_transcript",
    "from_openai_tool_calls",
    "from_otel_genai_spans",
    "jaccard_similarity",
    "json_exact_field_match",
    "json_numeric_drift",
    "json_structural_similarity",
    "load_benchmark_result",
    "load_history",
    "load_model_samples",
    "load_policy",
    "load_prompt_cases",
    "load_report",
    "load_test_prompts",
    "load_tool_check",
    "load_tool_events",
    "ngram_overlap_similarity",
    "normalized_levenshtein_similarity",
    "paired_permutation_test",
    "parse_tool_events",
    "prompt_changes",
    "record_report",
    "render_console_report",
    "serialize_samples_jsonl",
    "sparkline",
    "summarize_multi_run",
    "tokenize_text",
    "validate_against_schema",
    "validate_output",
]
