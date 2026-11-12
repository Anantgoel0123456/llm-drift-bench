"""Core benchmark and drift evaluation engine for llm-drift-bench."""

from __future__ import annotations

from collections.abc import Sequence

from llm_drift_bench.metrics import (
    compute_composite_drift_score,
    compute_percentile,
    jaccard_similarity,
    json_exact_field_match,
    json_numeric_drift,
    json_structural_similarity,
    ngram_overlap_similarity,
    normalized_levenshtein_similarity,
)
from llm_drift_bench.models import (
    DriftBenchmarkReport,
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


class DriftBenchmarkEngine:
    """Orchestrates benchmark comparisons between baseline and candidate LLM outputs."""

    def __init__(self, thresholds: DriftThresholds | None = None) -> None:
        self.thresholds = thresholds or DriftThresholds()

    def evaluate(
        self,
        baseline_samples: Sequence[ModelSample],
        candidate_samples: Sequence[ModelSample],
        prompt_cases: Sequence[PromptCase] | None = None,
    ) -> DriftBenchmarkReport:
        """Evaluate drift across matched prompt_ids between baseline and candidate runs."""
        if not baseline_samples:
            raise ValueError("baseline_samples cannot be empty.")
        if not candidate_samples:
            raise ValueError("candidate_samples cannot be empty.")

        cand_map = {sample.prompt_id: sample for sample in candidate_samples}
        prompt_map = (
            {case.prompt_id: case for case in prompt_cases} if prompt_cases else {}
        )

        pair_results: list[SampleDriftResult] = []
        base_latencies: list[float] = []
        cand_latencies: list[float] = []

        for base_sample in baseline_samples:
            cand_sample = cand_map.get(base_sample.prompt_id)
            if cand_sample is None:
                raise ValueError(
                    f"Missing candidate sample for prompt_id {base_sample.prompt_id!r}."
                )
            prompt_case = prompt_map.get(base_sample.prompt_id)
            res = evaluate_sample_pair(
                baseline=base_sample,
                candidate=cand_sample,
                prompt_case=prompt_case,
                thresholds=self.thresholds,
            )
            pair_results.append(res)
            if base_sample.latency_ms is not None:
                base_latencies.append(base_sample.latency_ms)
            if cand_sample.latency_ms is not None:
                cand_latencies.append(cand_sample.latency_ms)

        total = len(pair_results)
        drifted_count = sum(1 for r in pair_results if r.drifted)
        base_conformant = sum(1 for r in pair_results if r.baseline_validation.passed) / total
        cand_conformant = sum(1 for r in pair_results if r.candidate_validation.passed) / total

        mean_drift = sum(r.composite_drift_score for r in pair_results) / total
        max_drift = max(r.composite_drift_score for r in pair_results)
        mean_jaccard = sum(r.token_jaccard_similarity for r in pair_results) / total
        mean_lev = sum(r.levenshtein_similarity for r in pair_results) / total

        struct_vals = [
            r.json_structural_similarity
            for r in pair_results
            if r.json_structural_similarity is not None
        ]
        mean_struct = (sum(struct_vals) / len(struct_vals)) if struct_vals else None

        num_vals = [
            r.json_numeric_drift for r in pair_results if r.json_numeric_drift is not None
        ]
        mean_num = (sum(num_vals) / len(num_vals)) if num_vals else None

        base_p50 = compute_percentile(base_latencies, 50.0)
        cand_p50 = compute_percentile(cand_latencies, 50.0)
        base_p95 = compute_percentile(base_latencies, 95.0)
        cand_p95 = compute_percentile(cand_latencies, 95.0)

        failure_reasons: list[str] = []
        if mean_drift > self.thresholds.max_composite_drift:
            failure_reasons.append(
                f"Mean composite drift ({mean_drift:.4f}) exceeded threshold "
                f"({self.thresholds.max_composite_drift:.4f})."
            )
        if cand_conformant < self.thresholds.min_json_conformity_rate:
            failure_reasons.append(
                f"Candidate JSON conformity rate ({cand_conformant:.2%}) fell below "
                f"minimum ({self.thresholds.min_json_conformity_rate:.2%})."
            )
        if mean_num is not None and mean_num > self.thresholds.max_numeric_relative_drift:
            failure_reasons.append(
                f"Mean numeric relative drift ({mean_num:.4f}) exceeded threshold "
                f"({self.thresholds.max_numeric_relative_drift:.4f})."
            )
        if (
            base_p95 is not None
            and cand_p95 is not None
            and base_p95 > 0
            and ((cand_p95 - base_p95) / base_p95) * 100.0
            > self.thresholds.max_latency_regression_pct
        ):
            reg_pct = ((cand_p95 - base_p95) / base_p95) * 100.0
            failure_reasons.append(
                f"Candidate P95 latency regression (+{reg_pct:.1f}%) exceeded threshold "
                f"(+{self.thresholds.max_latency_regression_pct:.1f}%)."
            )

        baseline_model = baseline_samples[0].model_name
        candidate_model = candidate_samples[0].model_name

        return DriftBenchmarkReport(
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            total_samples=total,
            drifted_samples=drifted_count,
            baseline_json_conformity_rate=base_conformant,
            candidate_json_conformity_rate=cand_conformant,
            mean_composite_drift=mean_drift,
            max_composite_drift=max_drift,
            mean_token_jaccard=mean_jaccard,
            mean_levenshtein_similarity=mean_lev,
            mean_json_structural_similarity=mean_struct,
            mean_json_numeric_drift=mean_num,
            baseline_p50_latency_ms=base_p50,
            candidate_p50_latency_ms=cand_p50,
            baseline_p95_latency_ms=base_p95,
            candidate_p95_latency_ms=cand_p95,
            passed=len(failure_reasons) == 0,
            failure_reasons=tuple(failure_reasons),
            results=tuple(pair_results),
        )

