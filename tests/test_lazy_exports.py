"""Tests for PEP 562 lazy package exports and lightweight startup guarantees."""

from __future__ import annotations

import subprocess
import sys

import pytest

import llm_drift_bench
import llm_drift_bench.models as models_mod


def test_all_exports_resolve_cleanly() -> None:
    exported = set(llm_drift_bench.__all__)
    assert exported == {"__version__", *llm_drift_bench._EXPORTS}
    assert "__version__" in exported
    assert "DriftBenchmarkEngine" in exported
    assert "infer_prompt_suite" in exported
    assert "calibrate_from_reports" in exported
    assert "mcnemar_exact_test" in exported
    assert "register_scorer" in exported
    assert "diff_captures" in exported
    assert "export_csv_report" in exported
    assert "markdown_structure_similarity" in exported
    assert "detect_history_regression" in exported
    assert "export_sarif_report" in exported
    assert "run_doctor" in exported
    assert "json_key_jaccard_similarity" in exported
    assert "export_comparison_markdown" in exported
    assert "detect_numeric_tolerances" in exported
    assert "assert_no_regression" in exported
    assert "redact_samples" in exported
    assert "sample_prompt_suite" in exported
    assert "estimate_cost_drift" in exported
    assert "build_drift_matrix" in exported
    assert "lint_prompt_suite" in exported
    assert "evaluate_release_gates" in exported
    assert "from_gemini_transcript" in exported
    assert "numeric_prose_similarity" in exported
    assert "holm_bonferroni_adjust" in exported
    assert "wilson_score_interval" in exported
    assert "summarize_history" in exported
    assert "prune_history" in exported
    assert "filter_prompt_suite" in exported
    assert "create_snapshot_manifest" in exported
    assert "verify_snapshot_manifest" in exported
    assert "check_max_tool_calls" in exported
    assert "check_allowed_resources" in exported
    assert "merge_policies" in exported
    assert "render_policy_toml" in exported
    assert "format_diffs_markdown" in exported
    assert "check_cost_budget" in exported
    assert "rank_candidate_costs" in exported
    assert "import_openai_batch_jsonl" in exported
    assert "import_anthropic_batch_jsonl" in exported
    assert "import_csv_captures" in exported
    assert "export_drift_badge_svg" in exported
    assert "analyze_field_distributions" in exported
    assert "export_distribution_html" in exported
    assert "run_guided_wizard" in exported
    assert "PRESET_ANSWERS" in exported
    assert "analyze_logprobs_drift" in exported
    assert "run_quality_checks" in exported
    assert "detect_repetition_loop" in exported
    assert "is_likely_refusal" in exported
    assert "check_field_formats" in exported
    assert "SUPPORTED_FORMATS" in exported
    assert "get_pack" in exported
    assert "scaffold_pack" in exported
    assert "run_full_audit" in exported
    assert "export_audit_html" in exported
    assert "analyze_segmented_distributions" in exported
    assert "SegmentedDistributionReport" in exported
    assert "evaluate_consumer_rejection" in exported
    assert "export_consumer_html" in exported
    assert "compare_consumer_decisions" in exported
    assert "load_consumer_decisions" in exported
    assert "analyze_rolling_distribution" in exported
    assert "export_rolling_html" in exported
    assert "RollingWindowReport" in exported
    assert "HISTORY_SCHEMA_VERSION" in exported
    assert "REPORT_SCHEMA_VERSION" in exported
    assert "analyze_capture_embeddings" in exported
    assert "compare_embedding_vectors" in exported
    assert "cosine_similarity" in exported
    assert "EmbeddingConfig" in exported
    assert "export_embedding_drift_html" in exported
    assert "analyze_conversation_drift" in exported
    assert "ConversationDriftReport" in exported
    assert "load_conversations" in exported

    for name in llm_drift_bench.__all__:
        value = getattr(llm_drift_bench, name)
        assert value is not None

    assert exported.issubset(set(dir(llm_drift_bench)))


def test_unknown_attributes_raise_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = llm_drift_bench.definitely_nonexistent_symbol  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = models_mod.definitely_nonexistent_model  # type: ignore[attr-defined]


def test_models_lazy_legacy_exports() -> None:
    for name in ("BenchmarkResult", "DriftMetrics", "TestPrompt"):
        assert name in dir(models_mod)
        cls = getattr(models_mod, name)
        assert cls.__name__ == name


def test_importing_package_and_cli_stays_lightweight() -> None:
    code = (
        "import sys, llm_drift_bench, llm_drift_bench.cli; "
        "assert 'pydantic' not in sys.modules, 'pydantic eagerly imported'; "
        "assert 'rich' not in sys.modules, 'rich eagerly imported'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
