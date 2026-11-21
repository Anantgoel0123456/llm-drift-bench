"""Unit tests for pluggable scorers: char n-gram cosine, TF-IDF cosine, and Python AST."""

from __future__ import annotations

import pytest

from llm_drift_bench.models import PromptCase
from llm_drift_bench.scorers import (
    ScorerResult,
    char_ngram_cosine_similarity,
    get_scorer,
    list_scorers,
    python_ast_similarity,
    register_scorer,
    tfidf_cosine_similarity,
)


def test_char_ngram_and_tfidf_cosine_similarity() -> None:
    assert char_ngram_cosine_similarity("hello world", "hello world") == 1.0
    assert char_ngram_cosine_similarity("hello", "") == 0.0
    typo_sim = char_ngram_cosine_similarity("quantization", "quantisation")
    assert typo_sim == pytest.approx(0.7)

    assert tfidf_cosine_similarity("same text", "same text") == 1.0
    corpus = ["the invoice total is 100", "the invoice total is 200", "the ticket is urgent"]
    sim_close = tfidf_cosine_similarity(
        "the invoice total is 100",
        "the invoice total is 105",
        corpus=corpus,
    )
    sim_far = tfidf_cosine_similarity(
        "the invoice total is 100",
        "unrelated cooking recipe",
        corpus=corpus,
    )
    assert sim_close > sim_far


def test_python_ast_similarity_ignores_renames_and_comments() -> None:
    code_a = """```python
def reverse_string(text):
    # Reverse input text
    return text[::-1]
```"""
    code_b = """def rev(s):
    return s[::-1]"""
    assert python_ast_similarity(code_a, code_b) == 1.0

    different = (
        "def rev(s):\n    out = ''\n    for ch in s:\n        out = ch + out\n    return out"
    )
    diff_sim = python_ast_similarity(code_a, different)
    assert diff_sim is not None and 0.0 < diff_sim < 1.0

    assert python_ast_similarity("def broken(:", "x = 1") is None


def test_scorer_registry_builtins_and_custom_plugin() -> None:
    assert {"char_ngram_cosine", "python_ast", "tfidf_cosine"}.issubset(set(list_scorers()))
    res = get_scorer("python_ast").score("x = 1", "y = 2")
    assert res.available is True and res.similarity == 1.0

    unavail = get_scorer("python_ast").score("not python {", "x = 1")
    assert unavail.available is False and unavail.similarity is None

    class CustomLenScorer:
        name = "custom_length_ratio"

        def score(
            self, baseline: str, candidate: str, *, prompt: PromptCase | None = None
        ) -> ScorerResult:
            del prompt
            if not baseline and not candidate:
                return ScorerResult(self.name, 1.0)
            ratio = min(len(baseline), len(candidate)) / max(len(baseline), len(candidate), 1)
            return ScorerResult(self.name, round(ratio, 4))

    register_scorer(CustomLenScorer())
    assert "custom_length_ratio" in list_scorers()
    assert get_scorer("custom_length_ratio").score("abcd", "ab").similarity == 0.5

    with pytest.raises(ValueError, match="Unknown scorer"):
        get_scorer("nonexistent_scorer")
