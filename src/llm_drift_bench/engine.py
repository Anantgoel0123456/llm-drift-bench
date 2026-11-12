"""Core benchmark and drift evaluation engine for llm-drift-bench."""

from __future__ import annotations

from llm_drift_bench.metrics import (
    compute_composite_drift_score,
    jaccard_similarity,
    json_exact_field_match,
    json_numeric_drift,
    json_structural_similarity,
    ngram_overlap_similarity,
    normalized_levenshtein_similarity,
)
from llm_drift_bench.models import (
    DriftThresholds,
    ModelSample,
    PromptCase,
    SampleDriftResult,
)
from llm_drift_bench.validator import validate_output


def evaluate_sample_pair(
    baseline: ModelSample,
    candidate: ModelSample,
    prompt_case: PromptCase | None = None,
    thresholds: DriftThresholds | None = None,
) -> SampleDriftResult:
    """Compare a baseline ModelSample and candidate ModelSample for a single prompt_id.

    Args:
        baseline: Baseline model completion sample.
        candidate: Candidate model completion sample.
        prompt_case: Optional PromptCase with expected JSON Schema and category.
        thresholds: Optional DriftThresholds configuration.

    Returns:
        SampleDriftResult detailing validation, similarity, and composite drift metrics.
    """
    active_thresholds = thresholds or DriftThresholds()
    schema = prompt_case.schema if prompt_case is not None else None
    category = prompt_case.category if prompt_case is not None else "general"

    base_val = validate_output(baseline.output, schema=schema)
    cand_val = validate_output(candidate.output, schema=schema)

    tok_jaccard = jaccard_similarity(baseline.output, candidate.output)
    ngram_sim = ngram_overlap_similarity(baseline.output, candidate.output, n=2)
    lev_sim = normalized_levenshtein_similarity(baseline.output, candidate.output)

    struct_sim: float | None = None
    field_match: float | None = None
    num_drift: float | None = None

    if base_val.is_valid_json and cand_val.is_valid_json:
        struct_sim = json_structural_similarity(base_val.parsed_json, cand_val.parsed_json)
        field_match = json_exact_field_match(base_val.parsed_json, cand_val.parsed_json)
        num_drift = json_numeric_drift(base_val.parsed_json, cand_val.parsed_json)

    composite_drift = compute_composite_drift_score(
        token_jaccard=tok_jaccard,
        ngram_sim=ngram_sim,
        levenshtein_sim=lev_sim,
        json_structural_sim=struct_sim,
        json_field_match=field_match,
        json_num_drift=num_drift,
        baseline_schema_passed=base_val.passed,
        candidate_schema_passed=cand_val.passed,
    )

    latency_delta: float | None = None
    if baseline.latency_ms is not None and candidate.latency_ms is not None:
        latency_delta = candidate.latency_ms - baseline.latency_ms

    drifted = (
        composite_drift > active_thresholds.max_composite_drift
        or (base_val.passed and not cand_val.passed)
        or (
            num_drift is not None
            and num_drift > active_thresholds.max_numeric_relative_drift
        )
    )

    return SampleDriftResult(
        prompt_id=baseline.prompt_id,
        category=category,
        baseline_validation=base_val,
        candidate_validation=cand_val,
        token_jaccard_similarity=tok_jaccard,
        ngram_similarity=ngram_sim,
        levenshtein_similarity=lev_sim,
        json_structural_similarity=struct_sim,
        json_exact_field_match=field_match,
        json_numeric_drift=num_drift,
        composite_drift_score=composite_drift,
        latency_delta_ms=latency_delta,
        drifted=drifted,
    )
