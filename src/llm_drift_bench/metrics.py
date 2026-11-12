"""Lightweight, zero-bloat similarity and drift metrics for LLM outputs."""

from __future__ import annotations

from collections import Counter
import re

_TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)


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

