"""Response evaluation logic for JSON format adherence, token diff, similarity, and throughput."""

from __future__ import annotations

from collections.abc import Sequence
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
from typing import Any

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
    latency_ms: float,
    explicit_tokens: int | None = None,
) -> float:
    """Compute token generation throughput in tokens/sec from output text and latency (ms)."""
    if latency_ms <= 0.0:
        return 0.0
    token_count = (
        explicit_tokens if explicit_tokens is not None else max(1, len(tokenize_text(output_text)))
    )
    seconds = latency_ms / 1000.0
    return round(token_count / seconds, 2)


def evaluate_response_pair(
    prompt: TestPrompt,
    baseline_output: str,
    candidate_output: str,
    baseline_latency_ms: float = 100.0,
    candidate_latency_ms: float = 100.0,
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
        baseline_latency_ms=round(baseline_latency_ms, 2),
        candidate_latency_ms=round(candidate_latency_ms, 2),
        latency_delta_ms=round(candidate_latency_ms - baseline_latency_ms, 2),
        baseline_tokens_per_sec=base_tps,
        candidate_tokens_per_sec=cand_tps,
        tokens_per_sec_delta=round(cand_tps - base_tps, 2),
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

    return [TestPrompt.model_validate(item) for item in items]


def _synthesize_sample_for_model(
    prompt: TestPrompt,
    model_name: str,
    is_candidate: bool,
) -> ModelSample:
    """Synthesize a deterministic response when a model identifier is supplied instead of a file."""
    if not is_candidate and prompt.baseline_output is not None:
        return ModelSample(
            prompt_id=prompt.id,
            output=prompt.baseline_output,
            model_name=model_name,
            latency_ms=prompt.baseline_latency_ms or 120.0,
        )
    if is_candidate and prompt.candidate_output is not None:
        return ModelSample(
            prompt_id=prompt.id,
            output=prompt.candidate_output,
            model_name=model_name,
            latency_ms=prompt.candidate_latency_ms or 95.0,
        )

    digest = hashlib.sha256(f"{prompt.id}:{model_name}".encode()).hexdigest()
    latency = 90.0 + (int(digest[:4], 16) % 60)
    payload: dict[str, Any] = {"prompt_id": prompt.id, "status": "ok"}
    if prompt.expected_schema and isinstance(prompt.expected_schema.get("properties"), dict):
        props = prompt.expected_schema["properties"]
        payload = {}
        for key, prop_spec in props.items():
            ptype = prop_spec.get("type", "string") if isinstance(prop_spec, dict) else "string"
            if isinstance(prop_spec, dict) and "enum" in prop_spec and prop_spec["enum"]:
                payload[key] = prop_spec["enum"][0]
            elif isinstance(prop_spec, dict) and "const" in prop_spec:
                payload[key] = prop_spec["const"]
            elif ptype == "number":
                base_val = 100.0
                jitter = (int(digest[4:6], 16) % 5) * 0.5
                payload[key] = base_val if not is_candidate else round(base_val + jitter, 2)
            elif ptype == "integer":
                payload[key] = 1
            elif ptype == "boolean":
                payload[key] = True
            elif ptype == "array":
                payload[key] = ["sample_item"]
            else:
                payload[key] = f"INV-{digest[:4].upper()}"

    return ModelSample(
        prompt_id=prompt.id,
        output=json.dumps(payload),
        model_name=model_name,
        latency_ms=round(latency, 2),
    )


def _resolve_target_samples(
    target: str | Path | Sequence[ModelSample],
    prompts: Sequence[TestPrompt],
    is_candidate: bool,
) -> tuple[str, dict[str, ModelSample]]:
    """Resolve a target argument (ModelSample list, file path, or model name) into a sample map."""
    if not isinstance(target, (str, Path)):
        sample_list = list(target)
        model_label = sample_list[0].model_name if sample_list else "unknown"
        return model_label, {s.prompt_id: s for s in sample_list}

    target_path = Path(target)
    if target_path.exists() and target_path.is_file():
        loaded = load_model_samples(target_path)
        model_label = loaded[0].model_name if loaded else target_path.stem
        return model_label, {s.prompt_id: s for s in loaded}

    model_label = str(target)
    synth_map = {
        p.id: _synthesize_sample_for_model(p, model_label, is_candidate=is_candidate)
        for p in prompts
    }
    return model_label, synth_map


def evaluate_suite(
    prompts: Sequence[TestPrompt],
    baseline: str | Path | Sequence[ModelSample],
    candidate: str | Path | Sequence[ModelSample],
    suite_name: str = "default",
) -> BenchmarkResult:
    """Evaluate a suite of TestPrompts across baseline and candidate runs or models."""
    base_model, base_map = _resolve_target_samples(baseline, prompts, is_candidate=False)
    cand_model, cand_map = _resolve_target_samples(candidate, prompts, is_candidate=True)

    active_prompts = list(prompts)
    if not active_prompts:
        shared_ids = [pid for pid in base_map if pid in cand_map]
        active_prompts = [TestPrompt(prompt_id=pid, prompt="") for pid in shared_ids]

    if not active_prompts:
        raise ValueError("No prompts available to evaluate.")

    metrics_list: list[DriftMetrics] = []
    for prompt in active_prompts:
        b_sample = base_map.get(prompt.id)
        c_sample = cand_map.get(prompt.id)
        if b_sample is None or c_sample is None:
            raise ValueError(f"Missing baseline or candidate output for prompt {prompt.id!r}.")
        item_metrics = evaluate_response_pair(
            prompt=prompt,
            baseline_output=b_sample.output,
            candidate_output=c_sample.output,
            baseline_latency_ms=b_sample.latency_ms if b_sample.latency_ms is not None else 100.0,
            candidate_latency_ms=c_sample.latency_ms if c_sample.latency_ms is not None else 100.0,
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
        mean_baseline_latency_ms=round(
            sum(m.baseline_latency_ms for m in metrics_list) / total, 2
        ),
        mean_candidate_latency_ms=round(
            sum(m.candidate_latency_ms for m in metrics_list) / total, 2
        ),
        mean_baseline_tokens_per_sec=round(
            sum(m.baseline_tokens_per_sec for m in metrics_list) / total, 2
        ),
        mean_candidate_tokens_per_sec=round(
            sum(m.candidate_tokens_per_sec for m in metrics_list) / total, 2
        ),
        metrics=metrics_list,
    )


def load_benchmark_result(path: str | Path) -> BenchmarkResult:
    """Load a BenchmarkResult from a JSON file (supporting BenchmarkResult and legacy schemas)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "total_prompts" in raw and "metrics" in raw:
        return BenchmarkResult.model_validate(raw)

    # Convert legacy DriftBenchmarkReport dict into BenchmarkResult seamlessly
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
                baseline_latency_ms=float(raw.get("baseline_p50_latency_ms") or 100.0),
                candidate_latency_ms=float(raw.get("candidate_p50_latency_ms") or 100.0),
                latency_delta_ms=float(item.get("latency_delta_ms") or 0.0),
                baseline_tokens_per_sec=100.0,
                candidate_tokens_per_sec=100.0,
                tokens_per_sec_delta=0.0,
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
        mean_baseline_latency_ms=float(raw.get("baseline_p50_latency_ms") or 0.0),
        mean_candidate_latency_ms=float(raw.get("candidate_p50_latency_ms") or 0.0),
        mean_baseline_tokens_per_sec=100.0,
        mean_candidate_tokens_per_sec=100.0,
        metrics=converted_metrics,
    )

