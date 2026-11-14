"""Lightweight, zero-bloat similarity and drift metrics for LLM outputs."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from llm_drift_bench._numbers import finite_number

_TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_MAX_EDIT_DISTANCE_CHARS = 16_384


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase word and punctuation tokens."""
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_REGEX.findall(text)]


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute token-set Jaccard similarity in [0.0, 1.0] between two strings."""
    if text_a == text_b:
        return 1.0
    tokens_a = set(tokenize_text(text_a))
    tokens_b = set(tokenize_text(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def ngram_overlap_similarity(text_a: str, text_b: str, n: int = 2) -> float:
    """Compute token n-gram Dice overlap similarity in [0.0, 1.0]."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if text_a == text_b:
        return 1.0
    tokens_a = tokenize_text(text_a)
    tokens_b = tokenize_text(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    if len(tokens_a) < n or len(tokens_b) < n:
        return jaccard_similarity(text_a, text_b)

    ngrams_a = Counter(tuple(tokens_a[i : i + n]) for i in range(len(tokens_a) - n + 1))
    ngrams_b = Counter(tuple(tokens_b[i : i + n]) for i in range(len(tokens_b) - n + 1))
    overlap = sum((ngrams_a & ngrams_b).values())
    total = sum(ngrams_a.values()) + sum(ngrams_b.values())
    return (2.0 * overlap) / total if total > 0 else 1.0


def normalized_levenshtein_similarity(text_a: str, text_b: str) -> float:
    """Return exact character similarity: 1 - distance / max input length.

    Characters are Unicode code points, including whitespace and punctuation;
    no case folding or Unicode normalization is performed. Both empty inputs
    score 1.0; exactly one empty input scores 0.0.

    Bit-vector dynamic programming uses Python integers to process many cells
    per operation. Each input is limited to 16,384 characters to bound work and
    character-mask memory. Larger inputs raise ValueError, never a truncated
    score or a score computed in different units.
    """
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        raise TypeError("Levenshtein inputs must be strings.")
    max_len = max(len(text_a), len(text_b))
    if max_len > _MAX_EDIT_DISTANCE_CHARS:
        raise ValueError(
            f"Levenshtein inputs must not exceed {_MAX_EDIT_DISTANCE_CHARS:,} characters."
        )
    if text_a == text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    # The shorter input is the pattern, keeping each bit vector small.
    if len(text_a) > len(text_b):
        text_a, text_b = text_b, text_a
    masks: dict[str, int] = {}
    for index, char in enumerate(text_a):
        masks[char] = masks.get(char, 0) | (1 << index)

    width = len(text_a)
    mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    positive = mask
    negative = 0
    distance = width
    for char in text_b:
        equal = masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        if positive_horizontal & high_bit:
            distance += 1
        if negative_horizontal & high_bit:
            distance -= 1
        positive_horizontal = ((positive_horizontal << 1) | 1) & mask
        negative_horizontal = (negative_horizontal << 1) & mask
        positive = (negative_horizontal | ~(vertical | positive_horizontal)) & mask
        negative = positive_horizontal & vertical

    return 1.0 - distance / max_len


def _json_type_tag(value: Any) -> str:
    """Return a canonical JSON type label for a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def flatten_json_paths(data: Any, prefix: str = "$") -> dict[str, Any]:
    """Flatten nested JSON objects and arrays into a mapping of JSONPath -> leaf value."""
    flattened: dict[str, Any] = {}
    if isinstance(data, dict):
        if not data:
            flattened[prefix] = {}
        else:
            for key in sorted(data.keys()):
                # Preserve readable simple keys, quote all other keys to avoid
                # collisions between literal dots/brackets and nested structure.
                segment = (
                    f".{key}" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                    else f"[{json.dumps(key, ensure_ascii=True)}]"
                )
                flattened.update(flatten_json_paths(data[key], f"{prefix}{segment}"))
    elif isinstance(data, list):
        if not data:
            flattened[prefix] = []
        else:
            for idx, item in enumerate(data):
                flattened.update(flatten_json_paths(item, f"{prefix}[{idx}]"))
    else:
        flattened[prefix] = data
    return flattened


def json_structural_similarity(json_a: Any, json_b: Any) -> float:
    """Compute structural similarity in [0.0, 1.0] based on JSON key-paths and value types."""
    flat_a = flatten_json_paths(json_a)
    flat_b = flatten_json_paths(json_b)
    if not flat_a and not flat_b:
        return 1.0
    if not flat_a or not flat_b:
        return 0.0

    typed_paths_a = {(path, _json_type_tag(val)) for path, val in flat_a.items()}
    typed_paths_b = {(path, _json_type_tag(val)) for path, val in flat_b.items()}
    intersection = len(typed_paths_a & typed_paths_b)
    union = len(typed_paths_a | typed_paths_b)
    return intersection / union if union > 0 else 1.0


def _values_equal(val_a: Any, val_b: Any) -> bool:
    """Compare two JSON leaf values with strict boolean-vs-number distinction."""
    if isinstance(val_a, bool) or isinstance(val_b, bool):
        return isinstance(val_a, bool) and isinstance(val_b, bool) and val_a == val_b
    return bool(val_a == val_b)


def json_exact_field_match(json_a: Any, json_b: Any) -> float:
    """Compute the fraction of union JSON leaf paths with identical values in [0.0, 1.0]."""
    flat_a = flatten_json_paths(json_a)
    flat_b = flatten_json_paths(json_b)
    all_paths = set(flat_a.keys()) | set(flat_b.keys())
    if not all_paths:
        return 1.0
    matched = sum(
        1
        for path in all_paths
        if path in flat_a and path in flat_b and _values_equal(flat_a[path], flat_b[path])
    )
    return matched / len(all_paths)


def json_numeric_drift(json_a: Any, json_b: Any) -> float | None:
    """Compute mean bounded relative drift in [0.0, 1.0] across shared numeric JSON fields.

    Returns None if the two JSON payloads share no numeric leaf fields.
    """
    flat_a = flatten_json_paths(json_a)
    flat_b = flatten_json_paths(json_b)
    # Invalid values must not disappear just because their path is unmatched.
    for values in (flat_a, flat_b):
        for value in values.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                finite_number(value)
    shared_paths = sorted(set(flat_a.keys()) & set(flat_b.keys()))

    drifts: list[float] = []
    for path in shared_paths:
        val_a = flat_a[path]
        val_b = flat_b[path]
        if (
            isinstance(val_a, (int, float))
            and not isinstance(val_a, bool)
            and isinstance(val_b, (int, float))
            and not isinstance(val_b, bool)
        ):
            number_a, number_b = finite_number(val_a), finite_number(val_b)
            denom = max(abs(number_a), abs(number_b), 1e-9)
            # Opposite large values can overflow on subtraction before division.
            if (number_a < 0) != (number_b < 0):
                rel_diff = abs(number_a / denom - number_b / denom)
            else:
                rel_diff = abs(number_a - number_b) / denom
            drifts.append(min(1.0, rel_diff))

    if not drifts:
        return None
    return sum(drifts) / len(drifts)


def compute_composite_drift_score(
    *,
    token_jaccard: float,
    ngram_sim: float,
    levenshtein_sim: float,
    json_structural_sim: float | None = None,
    json_field_match: float | None = None,
    json_num_drift: float | None = None,
    baseline_schema_passed: bool = True,
    candidate_schema_passed: bool = True,
) -> float:
    """Compute a unified drift score in [0.0, 1.0] (0.0 = identical, 1.0 = completely drifted)."""
    text_drift = (
        0.35 * (1.0 - token_jaccard)
        + 0.30 * (1.0 - ngram_sim)
        + 0.35 * (1.0 - levenshtein_sim)
    )

    if json_structural_sim is not None and json_field_match is not None:
        struct_drift = 1.0 - json_structural_sim
        field_drift = 1.0 - json_field_match
        if json_num_drift is not None:
            json_drift = 0.40 * struct_drift + 0.35 * field_drift + 0.25 * json_num_drift
        else:
            json_drift = 0.50 * struct_drift + 0.50 * field_drift
        combined = 0.35 * text_drift + 0.65 * json_drift
    else:
        combined = text_drift

    # Apply a conformity regression penalty if baseline passed schema but candidate failed
    if baseline_schema_passed and not candidate_schema_passed:
        combined = max(combined, 0.75) + 0.25 * (1.0 - max(combined, 0.75))

    return max(0.0, min(1.0, combined))


def compute_percentile(values: list[float], percentile: float) -> float | None:
    """Compute linear-interpolated percentile (0-100) for a non-empty list of floats."""
    if not values:
        return None
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    floor_idx = int(k)
    ceil_idx = min(floor_idx + 1, len(sorted_vals) - 1)
    fraction = k - floor_idx
    return sorted_vals[floor_idx] + fraction * (sorted_vals[ceil_idx] - sorted_vals[floor_idx])





