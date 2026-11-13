"""Response evaluation logic for JSON format adherence, token diff, similarity, and throughput."""

from __future__ import annotations

from collections.abc import Sequence
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
from typing import Any

from llm_drift_bench._matching import index_records, require_same_ids
from llm_drift_bench.engine import load_model_samples
from llm_drift_bench.metrics import (
    jaccard_similarity,
    json_exact_field_match,
    json_structural_similarity,
    ngram_overlap_similarity,
    normalized_levenshtein_similarity,
    tokenize_text,
)
from llm_drift_bench.models import BenchmarkResult, DriftMetrics, ModelSample, TestPrompt
from llm_drift_bench.validator import extract_json_payload, validate_output


def check_json_adherence(
    text: str,
    expected_schema: dict[str, Any] | None = None,
) -> tuple[bool, bool, Any]:
    """Check whether text is valid JSON and adheres to an optional JSON Schema.

    Returns:
        Tuple of (is_valid_json, adheres_to_schema, parsed_payload).
    """
    res = validate_output(text, schema=expected_schema)
    return res.is_valid_json, res.passed, res.parsed_json


def compute_token_diff_ratio(baseline_text: str, candidate_text: str) -> float:
    """Compute token-level diff ratio in [0.0, 1.0] (0.0 = identical, 1.0 = distinct)."""
    if baseline_text.strip() == candidate_text.strip():
        return 0.0
    tokens_a = tokenize_text(baseline_text)
    tokens_b = tokenize_text(candidate_text)
    if not tokens_a and not tokens_b:
        return 0.0
    if not tokens_a or not tokens_b:
        return 1.0
    matcher = SequenceMatcher(None, tokens_a, tokens_b)
    similarity = matcher.ratio()
    return round(max(0.0, min(1.0, 1.0 - similarity)), 4)


def compute_semantic_similarity(baseline_text: str, candidate_text: str) -> float:
    """Compute a composite semantic/structural similarity score in [0.0, 1.0].

    Blends token Jaccard, bigram overlap, normalized Levenshtein similarity, and
    (when both outputs are valid JSON) JSON structural and leaf-field similarity.
    """
    if baseline_text.strip() == candidate_text.strip():
        return 1.0

    jaccard = jaccard_similarity(baseline_text, candidate_text)
    bigram = ngram_overlap_similarity(baseline_text, candidate_text, n=2)
    levenshtein = normalized_levenshtein_similarity(baseline_text, candidate_text)
    text_sim = 0.40 * jaccard + 0.30 * bigram + 0.30 * levenshtein

    ok_a, json_a, _ = extract_json_payload(baseline_text)
    ok_b, json_b, _ = extract_json_payload(candidate_text)
    if ok_a and ok_b:
        struct_sim = json_structural_similarity(json_a, json_b)
        field_sim = json_exact_field_match(json_a, json_b)
        combined = 0.40 * text_sim + 0.35 * struct_sim + 0.25 * field_sim
    else:
        combined = text_sim

    return round(max(0.0, min(1.0, combined)), 4)


def compute_tokens_per_sec(
    output_text: str,
    latency_ms: float | None,
    explicit_tokens: int | None = None,
) -> float | None:
    """Return measured tokens/sec, or None without a count and positive latency.

    output_text is retained for API compatibility, but never used to guess
    model tokens. Invalid measurements raise ValueError.
    """
    ModelSample(
        prompt_id="measurement", output=output_text,
        latency_ms=latency_ms, tokens_generated=explicit_tokens,
    )
    if latency_ms is None or latency_ms == 0 or explicit_tokens is None:
        return None
    throughput = (explicit_tokens / latency_ms) * 1000.0
    if not math.isfinite(throughput):
        raise ValueError("Computed throughput must be finite.")
    return throughput


def _optional_delta(candidate: float | None, baseline: float | None) -> float | None:
    """Return a difference only when both observations exist."""
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _complete_mean(values: Sequence[float | None]) -> float | None:
    """Avoid presenting a partial sample mean as the complete suite mean."""
    measured = [value for value in values if value is not None]
    if not measured or len(measured) != len(values):
        return None
    return sum(value / len(measured) for value in measured)


def evaluate_response_pair(
    prompt: TestPrompt,
    baseline_output: str,
    candidate_output: str,
    baseline_latency_ms: float | None = None,
    candidate_latency_ms: float | None = None,
    baseline_tokens: int | None = None,
    candidate_tokens: int | None = None,
) -> DriftMetrics:
    """Evaluate a baseline and candidate response pair for a single TestPrompt."""
    exact_match = baseline_output.strip() == candidate_output.strip()

    base_valid_json, base_schema_ok, _ = check_json_adherence(
        baseline_output, expected_schema=prompt.expected_schema
    )
    cand_valid_json, cand_schema_ok, _ = check_json_adherence(
        candidate_output, expected_schema=prompt.expected_schema
    )

    if prompt.expected_format.lower() == "json" or prompt.expected_schema is not None:
        format_adherence = cand_valid_json and cand_schema_ok
    else:
        format_adherence = bool(candidate_output.strip())

    semantic_sim = compute_semantic_similarity(baseline_output, candidate_output)
    token_diff = compute_token_diff_ratio(baseline_output, candidate_output)

    drift_score = round(0.55 * (1.0 - semantic_sim) + 0.45 * token_diff, 4)
    if (base_valid_json and base_schema_ok) and not format_adherence:
        drift_score = round(max(drift_score, 0.75), 4)

    base_tps = compute_tokens_per_sec(baseline_output, baseline_latency_ms, baseline_tokens)
    cand_tps = compute_tokens_per_sec(candidate_output, candidate_latency_ms, candidate_tokens)

    return DriftMetrics(
        prompt_id=prompt.id,
        category=prompt.category,
        exact_match=exact_match,
        baseline_valid_json=base_valid_json,
        candidate_valid_json=cand_valid_json,
        format_adherence=format_adherence,
        semantic_similarity=semantic_sim,
        token_diff_ratio=token_diff,
        drift_score=min(1.0, max(0.0, drift_score)),
        baseline_latency_ms=baseline_latency_ms,
        candidate_latency_ms=candidate_latency_ms,
        latency_delta_ms=_optional_delta(candidate_latency_ms, baseline_latency_ms),
        baseline_tokens_per_sec=base_tps,
        candidate_tokens_per_sec=cand_tps,
        tokens_per_sec_delta=_optional_delta(cand_tps, base_tps),
    )


def load_test_prompts(path: str | Path) -> list[TestPrompt]:
    """Load TestPrompt items from a JSON or JSONL suite file."""
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if file_path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            items = parsed.get("prompts") or parsed.get("cases") or parsed.get("suite") or []
        elif isinstance(parsed, list):
            items = parsed
        else:
            raise ValueError(f"Unsupported prompt suite format in {file_path}.")

    prompts = [TestPrompt.model_validate(item) for item in items]
    index_records(prompts, str(file_path))
    return prompts


def _resolve_target_samples(
    target: str | Path | Sequence[ModelSample],
) -> tuple[str, dict[str, ModelSample]]:
    """Resolve captured samples or an existing output file, never a model name."""
    if not isinstance(target, (str, Path)):
        sample_list = list(target)
        model_label = sample_list[0].model_name if sample_list else "unknown"
        return model_label, index_records(sample_list, "target samples")

    target_path = Path(target)
    if not target_path.is_file():
        raise ValueError(
            f"Benchmark target {str(target_path)!r} must be an existing outputs file. "
            "Model identifiers are not supported; supply captured JSON/JSONL outputs."
        )
    loaded = load_model_samples(target_path)
    model_label = loaded[0].model_name if loaded else target_path.stem
    return model_label, index_records(loaded, str(target_path))


def evaluate_suite(
    prompts: Sequence[TestPrompt],
    baseline: str | Path | Sequence[ModelSample],
    candidate: str | Path | Sequence[ModelSample],
    suite_name: str = "default",
) -> BenchmarkResult:
    """Evaluate TestPrompts against captured baseline and candidate outputs."""
    base_model, base_map = _resolve_target_samples(baseline)
    cand_model, cand_map = _resolve_target_samples(candidate)

    if not base_map or not cand_map:
        raise ValueError("Baseline and candidate samples cannot be empty.")
    require_same_ids(set(base_map), set(cand_map), "Candidate")
    if prompts:
        prompt_map = index_records(prompts, "prompt suite")
        require_same_ids(set(base_map), set(prompt_map), "Prompt suite")
        active_prompts = [prompt_map[pid] for pid in sorted(base_map)]
    else:
        active_prompts = [TestPrompt(prompt_id=pid, prompt="") for pid in sorted(base_map)]

    metrics_list: list[DriftMetrics] = []
    for prompt in active_prompts:
        b_sample = base_map[prompt.id]
        c_sample = cand_map[prompt.id]
        item_metrics = evaluate_response_pair(
            prompt=prompt,
            baseline_output=b_sample.output,
            candidate_output=c_sample.output,
            baseline_latency_ms=b_sample.latency_ms,
            candidate_latency_ms=c_sample.latency_ms,
            baseline_tokens=b_sample.tokens_generated,
            candidate_tokens=c_sample.tokens_generated,
        )
        metrics_list.append(item_metrics)

    total = len(metrics_list)
    return BenchmarkResult(
        suite_name=suite_name,
        baseline_model=base_model,
        candidate_model=cand_model,
        total_prompts=total,
        exact_match_rate=round(sum(1 for m in metrics_list if m.exact_match) / total, 4),
        baseline_json_adherence_rate=round(
            sum(1 for m in metrics_list if m.baseline_valid_json) / total, 4
        ),
        candidate_json_adherence_rate=round(
            sum(1 for m in metrics_list if m.format_adherence) / total, 4
        ),
        mean_semantic_similarity=round(
            sum(m.semantic_similarity for m in metrics_list) / total, 4
        ),
        mean_token_diff_ratio=round(
            sum(m.token_diff_ratio for m in metrics_list) / total, 4
        ),
        mean_drift_score=round(sum(m.drift_score for m in metrics_list) / total, 4),
        mean_baseline_latency_ms=_complete_mean([m.baseline_latency_ms for m in metrics_list]),
        mean_candidate_latency_ms=_complete_mean([m.candidate_latency_ms for m in metrics_list]),
        mean_baseline_tokens_per_sec=_complete_mean(
            [m.baseline_tokens_per_sec for m in metrics_list]
        ),
        mean_candidate_tokens_per_sec=_complete_mean(
            [m.candidate_tokens_per_sec for m in metrics_list]
        ),
        metrics=metrics_list,
    )


def load_benchmark_result(path: str | Path) -> BenchmarkResult:
    """Load a BenchmarkResult from a JSON file (supporting BenchmarkResult and legacy schemas)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "total_prompts" in raw and "metrics" in raw:
        return BenchmarkResult.model_validate(raw)

    # Preserve missing measurements; percentiles cannot reconstruct means or samples.
    results_raw = raw.get("results", [])
    converted_metrics: list[DriftMetrics] = []
    for item in results_raw:
        base_v = item.get("baseline_validation", {})
        cand_v = item.get("candidate_validation", {})
        drift_val = float(item.get("composite_drift_score", 0.0))
        sim_val = round(max(0.0, 1.0 - drift_val), 4)
        converted_metrics.append(
            DriftMetrics(
                prompt_id=str(item.get("prompt_id", "unknown")),
                category=str(item.get("category", "general")),
                exact_match=drift_val == 0.0,
                baseline_valid_json=bool(base_v.get("is_valid_json", True)),
                candidate_valid_json=bool(cand_v.get("is_valid_json", True)),
                format_adherence=bool(cand_v.get("passed", True)),
                semantic_similarity=sim_val,
                token_diff_ratio=round(1.0 - float(item.get("levenshtein_similarity", sim_val)), 4),
                drift_score=drift_val,
                latency_delta_ms=item.get("latency_delta_ms"),
            )
        )
    total = max(1, len(converted_metrics))
    return BenchmarkResult(
        suite_name=str(raw.get("suite_name", "default")),
        baseline_model=str(raw.get("baseline_model", "baseline")),
        candidate_model=str(raw.get("candidate_model", "candidate")),
        total_prompts=len(converted_metrics),
        exact_match_rate=round(sum(1 for m in converted_metrics if m.exact_match) / total, 4),
        baseline_json_adherence_rate=float(raw.get("baseline_json_conformity_rate", 1.0)),
        candidate_json_adherence_rate=float(raw.get("candidate_json_conformity_rate", 1.0)),
        mean_semantic_similarity=round(
            sum(m.semantic_similarity for m in converted_metrics) / total, 4
        ),
        mean_token_diff_ratio=round(
            sum(m.token_diff_ratio for m in converted_metrics) / total, 4
        ),
        mean_drift_score=float(raw.get("mean_composite_drift", 0.0)),
        metrics=converted_metrics,
    )

