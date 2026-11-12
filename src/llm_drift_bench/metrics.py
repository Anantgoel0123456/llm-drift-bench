"""Lightweight, zero-bloat similarity and drift metrics for LLM outputs."""

from __future__ import annotations

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
