"""Response evaluation logic for JSON format adherence, token diff, similarity, and throughput."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from llm_drift_bench.metrics import (
    jaccard_similarity,
    json_exact_field_match,
    json_structural_similarity,
    ngram_overlap_similarity,
    normalized_levenshtein_similarity,
    tokenize_text,
)
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
    """Compute the token-level diff ratio in [0.0, 1.0] (0.0 = identical, 1.0 = completely different)."""
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
