"""Unit tests for pluggable scorers: char n-gram cosine, TF-IDF cosine, and Python AST."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main
from llm_drift_bench.models import ModelSample, PromptCase
from llm_drift_bench.scorers import (
    _SCORER_REGISTRY,
    ScorerResult,
    char_ngram_cosine_similarity,
    get_scorer,
    list_scorers,
    markdown_structure_similarity,
    python_ast_similarity,
    register_scorer,
    score_captures,
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
    try:
        assert "custom_length_ratio" in list_scorers()
        assert get_scorer("custom_length_ratio").score("abcd", "ab").similarity == 0.5
    finally:
        _SCORER_REGISTRY.pop("custom_length_ratio", None)

    with pytest.raises(ValueError, match="Unknown scorer"):
        get_scorer("nonexistent_scorer")
    with pytest.raises(ValueError, match="between 0 and 1"):
        ScorerResult("bad", 1.5)
    with pytest.raises(ValueError, match="positive integer"):
        char_ngram_cosine_similarity("a", "b", n=0)


def test_markdown_structure_similarity() -> None:
    md_a = (
        "# Summary\n\n"
        "- Item one\n"
        "- Item two\n\n"
        "> Note\n\n"
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
        "```python\nx = 1\n```\n"
    )
    md_b = (
        "# Overview\n\n"
        "- Alpha\n"
        "- Beta\n\n"
        "> Callout\n\n"
        "| K | V |\n| :--- | ---: |\n| 3 | 4 |\n\n"
        "```python\ny = 2\n```\n"
    )
    assert markdown_structure_similarity(md_a, md_b) == 1.0
    assert markdown_structure_similarity("plain prose", "more prose") == 1.0
    assert markdown_structure_similarity(md_a, "plain prose only") == 0.0
    assert get_scorer("markdown_structure").score(md_a, md_b).similarity == 1.0


def test_json_key_jaccard_scorer() -> None:
    from llm_drift_bench.scorers import json_key_jaccard_similarity

    assert json_key_jaccard_similarity('{"a": 1, "b": "x"}', '{"a": 99, "b": "y"}') == 1.0
    assert json_key_jaccard_similarity('{"a": 1, "b": 2}', '{"a": 1, "c": 3}') == pytest.approx(
        1 / 3
    )
    assert json_key_jaccard_similarity("plain text", '{"a": 1}') is None
    res_ok = get_scorer("json_key_jaccard").score('{"a": 1}', '{"a": 2}')
    assert res_ok.available is True and res_ok.similarity == 1.0
    res_na = get_scorer("json_key_jaccard").score("plain", "text")
    assert res_na.available is False and res_na.similarity is None


def test_score_captures_and_cli_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = Path(__file__).resolve().parents[1] / "examples"
    base = examples / "baseline.jsonl"
    drift = examples / "candidate-drift.jsonl"

    assert main(["score", "--list-scorers"]) == 0
    listed = capsys.readouterr().out
    assert "markdown_structure" in listed
    assert "python_ast" in listed

    assert main(["score", "--list-scorers", "--format", "json"]) == 0
    assert "tfidf_cosine" in json.loads(capsys.readouterr().out)["scorers"]

    assert main([
        "score",
        "-b",
        str(base),
        "-c",
        str(drift),
        "--scorer",
        "char_ngram_cosine",
        "--scorer",
        "python_ast",
    ]) == 0
    out_text = capsys.readouterr().out
    assert "Pluggable Scorer Summary (2 prompt(s))" in out_text
    assert "char_ngram_cosine" in out_text

    out_json = tmp_path / "scores.json"
    assert main([
        "score",
        "-b",
        str(base),
        "-c",
        str(drift),
        "--scorer",
        "tfidf_cosine",
        "--format",
        "json",
        "-o",
        str(out_json),
    ]) == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["total_prompts"] == 2
    assert payload["summary"]["tfidf_cosine"]["available_prompts"] == 2

    # Missing required capture args fails cleanly with exit 2.
    assert main(["score"]) == 2
    capsys.readouterr()

    with pytest.raises(ValueError, match="non-empty"):
        score_captures([], [ModelSample(prompt_id="p1", output="x")])


def test_numeric_prose_nested_fences_null_bytes_and_dup_suite() -> None:
    from llm_drift_bench.scorers import numeric_prose_similarity

    # Normalizes comma grouping, trailing decimals, currency symbols, and percentages;
    # ignores alphanumeric IDs like INV-2026 or v2.
    assert numeric_prose_similarity(
        "Invoice INV-2026 (v2) grew 14.2% to $1,250.00.",
        "Total reached 1250 after a 14.20% gain.",
    ) == 1.0
    assert numeric_prose_similarity("Revenue was $100.", "Revenue was $200.") == 0.0
    assert numeric_prose_similarity("No digits here.", "Still no digits.") is None

    sc = get_scorer("numeric_prose")
    assert sc.score("Price $10", "Price 10.0").similarity == 1.0
    assert sc.score("alpha", "beta").available is False

    # Nested 4-backtick outer fence containing a 3-backtick inner fence counts as 1 code block.
    nested_md = "````markdown\n# Not a heading\n```python\nx = 1\n```\n````\n# Real Heading\n"
    tilde_md = "~~~markdown\n# Also ignored\n~~~\n# Real Heading\n"
    assert markdown_structure_similarity(nested_md, tilde_md) == 1.0

    # Null bytes in Python source return None cleanly instead of raising ValueError.
    assert python_ast_similarity("x = 1\x00", "x = 1") is None

    # Duplicate prompt_id in prompt_cases is rejected explicitly.
    samples = [ModelSample(prompt_id="p1", output="ok")]
    dups = [PromptCase(prompt_id="p1", prompt="a"), PromptCase(prompt_id="p1", prompt="b")]
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        score_captures(samples, samples, prompt_cases=dups)


def test_scorers_edge_cases_entry_points_and_error_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata

    from llm_drift_bench.scorers import (
        _cosine_counter,
        json_key_jaccard_similarity,
        numeric_prose_similarity,
    )

    # Cosine helper zero/empty vector guards
    assert _cosine_counter({}, {}) == 1.0
    assert _cosine_counter({"a": 1.0}, {}) == 0.0
    assert _cosine_counter({"a": 0.0}, {"a": 1.0}) == 0.0

    # TF-IDF when one or both inputs contain only whitespace (empty token lists)
    assert tfidf_cosine_similarity(" ", "\t\n") == 1.0
    assert tfidf_cosine_similarity("hello world", "   ") == 0.0

    # Short inputs (< n) fall back to single-character cosine cleanly
    assert char_ngram_cosine_similarity("ab", "ac", n=3) == pytest.approx(0.5)

    # Empty Python source, Markdown ordered lists, empty JSON objects, and numeric zeros
    assert python_ast_similarity("   ", "x = 1") is None
    assert markdown_structure_similarity("1. First\n2. Second\n", "1) One\n2) Two\n") == 1.0
    assert json_key_jaccard_similarity("{}", "{}") == 1.0
    assert numeric_prose_similarity("Balance: 0.00", "Balance: -0") == 1.0

    # Duplicate capture IDs rejected; matched prompt_cases pass through cleanly
    s1 = ModelSample(prompt_id="p1", output='{"a": 1}')
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        score_captures([s1, s1], [s1])

    out = score_captures(
        [s1],
        [s1],
        scorer_names=["json_key_jaccard"],
        prompt_cases=[PromptCase(prompt_id="p1", prompt="Extract")],
    )
    assert out["summary"]["json_key_jaccard"]["mean"] == 1.0

    # Invalid empty scorer name rejected
    class _BlankNameScorer:
        name = "   "

        def score(
            self, baseline: str, candidate: str, *, prompt: PromptCase | None = None
        ) -> ScorerResult:
            del baseline, candidate, prompt
            return ScorerResult("blank", 1.0)

    with pytest.raises(ValueError, match="non-empty string"):
        register_scorer(_BlankNameScorer())

    # Third-party entry-point discovery in list_scorers() and get_scorer()
    class _PluginScorer:
        name = "ep_plugin_scorer"

        def score(
            self, baseline: str, candidate: str, *, prompt: PromptCase | None = None
        ) -> ScorerResult:
            del prompt
            return ScorerResult(self.name, 1.0 if baseline == candidate else 0.0)

    class _FakeEntryPoint:
        name = "ep_plugin_scorer"

        def load(self) -> type[_PluginScorer]:
            return _PluginScorer

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group="": [_FakeEntryPoint()] if group == "llm_drift_bench.scorers" else [],
    )
    try:
        assert "ep_plugin_scorer" in list_scorers()
        loaded = get_scorer("ep_plugin_scorer")
        assert loaded.score("hi", "hi").similarity == 1.0
    finally:
        _SCORER_REGISTRY.pop("ep_plugin_scorer", None)



