"""Unit tests for text, structural, and numeric drift metrics in llm_drift_bench.metrics."""

from __future__ import annotations

import unittest

from llm_drift_bench.metrics import (
    _json_type_tag,
    compute_composite_drift_score,
    compute_percentile,
    flatten_json_paths,
    jaccard_similarity,
    json_exact_field_match,
    json_numeric_drift,
    json_structural_similarity,
    ngram_overlap_similarity,
    normalized_levenshtein_similarity,
    tokenize_text,
)


class TestMetrics(unittest.TestCase):
    """Test suite for similarity, structural, and drift metric functions."""

    def test_tokenize_and_jaccard_similarity(self) -> None:
        self.assertEqual(tokenize_text(""), [])
        self.assertEqual(tokenize_text("Hello, World!"), ["hello", ",", "world", "!"])
        self.assertEqual(jaccard_similarity("same", "same"), 1.0)
        self.assertEqual(jaccard_similarity("   ", "   "), 1.0)
        self.assertEqual(jaccard_similarity("hello", ""), 0.0)
        self.assertAlmostEqual(jaccard_similarity("a b c", "a b d"), 0.5)

    def test_ngram_overlap_similarity(self) -> None:
        with self.assertRaises(ValueError):
            ngram_overlap_similarity("a", "b", n=0)
        self.assertEqual(ngram_overlap_similarity("identical text", "identical text"), 1.0)
        self.assertEqual(ngram_overlap_similarity("   ", "   "), 1.0)
        self.assertEqual(ngram_overlap_similarity("word", ""), 0.0)
        self.assertEqual(ngram_overlap_similarity("one", "one", n=3), 1.0)
        self.assertEqual(ngram_overlap_similarity("one", "two", n=3), 0.0)
        sim = ngram_overlap_similarity("the quick brown fox", "the quick red fox", n=2)
        self.assertGreater(sim, 0.3)
        self.assertLess(sim, 1.0)

    def test_normalized_levenshtein_similarity(self) -> None:
        self.assertEqual(normalized_levenshtein_similarity("abc", "abc"), 1.0)
        self.assertEqual(normalized_levenshtein_similarity("", ""), 1.0)
        self.assertEqual(normalized_levenshtein_similarity("abc", ""), 0.0)
        self.assertAlmostEqual(normalized_levenshtein_similarity("kitten", "sitting"), 1.0 - 3 / 7)

        # Long string (>1024 chars) exercises token-level Levenshtein branch
        long_a = " ".join(["token"] * 300)
        long_b = " ".join(["token"] * 290 + ["changed"] * 10)
        sim_long = normalized_levenshtein_similarity(long_a, long_b)
        self.assertGreater(sim_long, 0.9)

        # Long whitespace-only string (>1024 chars) exercises max_len == 0 after tokenization
        self.assertEqual(normalized_levenshtein_similarity(" " * 1050, "\t" * 1050), 1.0)

    def test_flatten_json_paths_and_type_tags(self) -> None:
        self.assertEqual(_json_type_tag(None), "null")
        self.assertEqual(_json_type_tag(True), "boolean")
        self.assertEqual(_json_type_tag(1.5), "number")
        self.assertEqual(_json_type_tag("s"), "string")
        self.assertEqual(_json_type_tag([]), "array")
        self.assertEqual(_json_type_tag({}), "object")
        self.assertEqual(_json_type_tag(object()), "unknown")

        nested = {"a": {"b": [10, {"c": "val"}]}, "empty_obj": {}, "empty_arr": []}
        flat = flatten_json_paths(nested)
        self.assertEqual(flat["$.a.b[0]"], 10)
        self.assertEqual(flat["$.a.b[1].c"], "val")
        self.assertEqual(flat["$.empty_obj"], {})
        self.assertEqual(flat["$.empty_arr"], [])

    def test_json_structural_and_field_similarity(self) -> None:
        obj_a = {"id": "1", "count": 5, "active": True}
        obj_b = {"id": "2", "count": 5, "active": False}
        self.assertEqual(json_structural_similarity(obj_a, obj_b), 1.0)
        self.assertAlmostEqual(json_exact_field_match(obj_a, obj_b), 1 / 3)

        # Boolean vs int distinction in _values_equal
        self.assertEqual(json_exact_field_match({"flag": True}, {"flag": 1}), 0.0)
        self.assertLess(json_structural_similarity({"a": 1}, {"a": "1", "b": 2}), 1.0)

    def test_json_numeric_drift(self) -> None:
        self.assertIsNone(json_numeric_drift({"a": "text"}, {"a": "other"}))
        self.assertIsNone(json_numeric_drift({"a": True}, {"a": 1}))
        self.assertEqual(json_numeric_drift({"x": 100.0}, {"x": 100.0}), 0.0)
        drift = json_numeric_drift({"x": 100.0, "y": 50.0}, {"x": 90.0, "y": 50.0})
        self.assertIsNotNone(drift)
        self.assertAlmostEqual(drift or 0.0, 0.05)

    def test_compute_composite_drift_and_percentile(self) -> None:
        zero_drift = compute_composite_drift_score(
            token_jaccard=1.0,
            ngram_sim=1.0,
            levenshtein_sim=1.0,
            json_structural_sim=1.0,
            json_field_match=1.0,
            json_num_drift=0.0,
        )
        self.assertEqual(zero_drift, 0.0)

        no_num_drift = compute_composite_drift_score(
            token_jaccard=0.8,
            ngram_sim=0.8,
            levenshtein_sim=0.8,
            json_structural_sim=1.0,
            json_field_match=0.5,
            json_num_drift=None,
        )
        self.assertGreater(no_num_drift, 0.0)

        penalized = compute_composite_drift_score(
            token_jaccard=0.9,
            ngram_sim=0.9,
            levenshtein_sim=0.9,
            baseline_schema_passed=True,
            candidate_schema_passed=False,
        )
        self.assertGreaterEqual(penalized, 0.75)

        self.assertIsNone(compute_percentile([], 50.0))
        self.assertEqual(compute_percentile([10.0, 20.0, 30.0], 0.0), 10.0)
        self.assertEqual(compute_percentile([10.0, 20.0, 30.0], 100.0), 30.0)
        self.assertEqual(compute_percentile([10.0, 20.0, 30.0], 50.0), 20.0)
