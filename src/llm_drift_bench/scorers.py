"""Pluggable scorer protocol and zero-dependency cosine & Python AST structural scorers."""

from __future__ import annotations

import ast
import importlib.metadata
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from llm_drift_bench.metrics import tokenize_text
from llm_drift_bench.models import PromptCase

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


@dataclass(frozen=True)
class ScorerResult:
    """Normalized similarity result in [0.0, 1.0] returned by any pluggable Scorer."""

    name: str
    similarity: float | None
    available: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.similarity is not None and (
            isinstance(self.similarity, bool)
            or not math.isfinite(self.similarity)
            or not 0.0 <= self.similarity <= 1.0
        ):
            raise ValueError("ScorerResult.similarity must be between 0 and 1, or None.")


class Scorer(Protocol):
    """Protocol implemented by built-in and third-party entry-point scorers."""

    @property
    def name(self) -> str: ...

    def score(
        self,
        baseline: str,
        candidate: str,
        *,
        prompt: PromptCase | None = None,
    ) -> ScorerResult: ...


def _cosine_counter(
    vec_a: Counter[str] | dict[str, float],
    vec_b: Counter[str] | dict[str, float],
) -> float:
    if not vec_a and not vec_b:
        return 1.0
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in vec_a if k in vec_b)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def char_ngram_cosine_similarity(text_a: str, text_b: str, *, n: int = 3) -> float:
    """Compute character n-gram cosine similarity in [0.0, 1.0] (robust to phrasing/typos)."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")
    norm_a = " ".join(text_a.lower().split())
    norm_b = " ".join(text_b.lower().split())
    if norm_a == norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0
    if len(norm_a) < n or len(norm_b) < n:
        return _cosine_counter(Counter(norm_a), Counter(norm_b))
    grams_a = Counter(norm_a[i : i + n] for i in range(len(norm_a) - n + 1))
    grams_b = Counter(norm_b[i : i + n] for i in range(len(norm_b) - n + 1))
    return _cosine_counter(grams_a, grams_b)


def tfidf_cosine_similarity(
    text_a: str,
    text_b: str,
    *,
    corpus: Sequence[str] | None = None,
) -> float:
    """Compute TF-IDF weighted token cosine similarity in [0.0, 1.0]."""
    if text_a == text_b:
        return 1.0
    toks_a = tokenize_text(text_a)
    toks_b = tokenize_text(text_b)
    if not toks_a and not toks_b:
        return 1.0
    if not toks_a or not toks_b:
        return 0.0

    docs = [toks_a, toks_b]
    if corpus:
        docs.extend(tokenize_text(doc) for doc in corpus)

    doc_count = len(docs)
    df: Counter[str] = Counter()
    for doc in docs:
        for token in set(doc):
            df[token] += 1

    def to_tfidf(tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        total = len(tokens)
        return {
            tok: (cnt / total) * (math.log((doc_count + 1) / (df[tok] + 1)) + 1.0)
            for tok, cnt in tf.items()
        }

    return _cosine_counter(to_tfidf(toks_a), to_tfidf(toks_b))


def _extract_python_code(text: str) -> str:
    stripped = text.strip()
    match = _CODE_BLOCK_RE.search(stripped)
    return match.group(1).strip() if match else stripped


def _ast_structural_bag(tree: ast.AST) -> Counter[str]:
    """Extract normalized parent->child AST edge multiset, ignoring names and formatting."""
    bag: Counter[str] = Counter()

    def visit(node: ast.AST, parent_tag: str) -> None:
        tag = type(node).__name__
        if isinstance(node, ast.Constant):
            tag = f"Const:{type(node.value).__name__}"
        elif isinstance(node, ast.FunctionDef):
            tag = f"FunctionDef(args={len(node.args.args)})"
        bag[f"{parent_tag}->{tag}"] += 1
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Load, ast.Store, ast.Del)):
                continue
            visit(child, tag)

    visit(tree, "ROOT")
    return bag


def python_ast_similarity(code_a: str, code_b: str) -> float | None:
    """Compute structural Python AST Dice similarity in [0.0, 1.0], or None on SyntaxError.

    Ignores comments, whitespace, formatting, and variable renames so equivalent
    refactored Python implementations score near 1.0.
    """
    src_a = _extract_python_code(code_a)
    src_b = _extract_python_code(code_b)
    if not src_a or not src_b:
        return None
    try:
        tree_a = ast.parse(src_a)
        tree_b = ast.parse(src_b)
    except SyntaxError:
        return None
    bag_a = _ast_structural_bag(tree_a)
    bag_b = _ast_structural_bag(tree_b)
    total = sum(bag_a.values()) + sum(bag_b.values())
    if total == 0:
        return 1.0
    overlap = sum((bag_a & bag_b).values())
    return (2.0 * overlap) / total


class _CharNgramScorer:
    name = "char_ngram_cosine"

    def score(
        self, baseline: str, candidate: str, *, prompt: PromptCase | None = None
    ) -> ScorerResult:
        del prompt
        return ScorerResult(self.name, round(char_ngram_cosine_similarity(baseline, candidate), 4))


class _TfidfScorer:
    name = "tfidf_cosine"

    def score(
        self, baseline: str, candidate: str, *, prompt: PromptCase | None = None
    ) -> ScorerResult:
        del prompt
        return ScorerResult(self.name, round(tfidf_cosine_similarity(baseline, candidate), 4))


class _PythonAstScorer:
    name = "python_ast"

    def score(
        self, baseline: str, candidate: str, *, prompt: PromptCase | None = None
    ) -> ScorerResult:
        del prompt
        sim = python_ast_similarity(baseline, candidate)
        if sim is None:
            return ScorerResult(self.name, None, available=False)
        return ScorerResult(self.name, round(sim, 4), available=True)


_SCORER_REGISTRY: dict[str, Scorer] = {
    "char_ngram_cosine": _CharNgramScorer(),
    "tfidf_cosine": _TfidfScorer(),
    "python_ast": _PythonAstScorer(),
}


def register_scorer(scorer: Scorer) -> None:
    """Register a custom Scorer instance by its unique name."""
    if not isinstance(scorer.name, str) or not scorer.name.strip():
        raise ValueError("Scorer name must be a non-empty string.")
    _SCORER_REGISTRY[scorer.name] = scorer


def list_scorers() -> list[str]:
    """Return sorted names of built-in, registered, and entry-point scorers."""
    names = set(_SCORER_REGISTRY)
    for ep in importlib.metadata.entry_points(group="llm_drift_bench.scorers"):
        names.add(ep.name)
    return sorted(names)


def get_scorer(name: str) -> Scorer:
    """Retrieve a registered or entry-point Scorer by name."""
    if name in _SCORER_REGISTRY:
        return _SCORER_REGISTRY[name]
    for ep in importlib.metadata.entry_points(group="llm_drift_bench.scorers"):
        if ep.name == name:
            loaded = ep.load()
            scorer: Scorer = loaded() if callable(loaded) else loaded
            _SCORER_REGISTRY[name] = scorer
            return scorer
    raise ValueError(f"Unknown scorer {name!r}. Available: {list_scorers()}.")
