"""Unit tests for domain data models and serialization in llm_drift_bench.models."""

from __future__ import annotations

import json
import unittest

from llm_drift_bench.models import (
    BenchmarkResult,
    DriftBenchmarkReport,
    DriftMetrics,
    DriftThresholds,
    ModelSample,
    PromptCase,
    SampleDriftResult,
    SchemaValidationResult,
    TestPrompt,
    ValidationIssue,
)


class TestModels(unittest.TestCase):
    """Test suite for PromptCase, ModelSample, ValidationIssue, and Report models."""

    def test_prompt_case_from_dict_valid(self) -> None:
        data = {
            "id": "case_01",
            "prompt": "Return JSON",
            "schema": {"type": "object"},
            "category": "json_gen",
            "metadata": {"source": "synth"},
        }
        case = PromptCase.from_dict(data)
        self.assertEqual(case.prompt_id, "case_01")
        self.assertEqual(case.prompt, "Return JSON")
        self.assertEqual(case.schema, {"type": "object"})
        self.assertEqual(case.category, "json_gen")
        self.assertEqual(case.metadata, {"source": "synth"})

    def test_prompt_case_from_dict_errors(self) -> None:
        with self.assertRaises(ValueError):
            PromptCase.from_dict({"prompt": "Missing id"})
        with self.assertRaises(ValueError):
            PromptCase.from_dict({"prompt_id": "bad_schema", "schema": ["not_a_dict"]})

    def test_model_sample_from_dict_valid_and_aliases(self) -> None:
        sample = ModelSample.from_dict(
            {
                "id": "s1",
                "response": "12345",
                "model": "qwen-7b",
                "latency_ms": "88.5",
                "tokens_generated": "42",
                "metadata": {"quant": "int4"},
            }
        )
        self.assertEqual(sample.prompt_id, "s1")
        self.assertEqual(sample.output, "12345")
        self.assertEqual(sample.model_name, "qwen-7b")
        self.assertEqual(sample.latency_ms, 88.5)
        self.assertEqual(sample.tokens_generated, 42)
        self.assertEqual(sample.metadata, {"quant": "int4"})

    def test_model_sample_from_dict_errors(self) -> None:
        with self.assertRaises(ValueError):
            ModelSample.from_dict({"output": "no id"})
        with self.assertRaises(ValueError):
            ModelSample.from_dict({"prompt_id": "s1"})

    def test_validation_issue_and_schema_result_serialization(self) -> None:
        issue = ValidationIssue(path="$.total", message="Too small", rule="minimum")
        res = SchemaValidationResult(
            is_valid_json=True,
            is_schema_conformant=False,
            parsed_json={"total": -1},
            issues=(issue,),
        )
        self.assertFalse(res.passed)
        serialized = res.to_dict()
        self.assertFalse(serialized["passed"])
        self.assertEqual(serialized["issues"][0]["rule"], "minimum")

    def test_sample_drift_result_and_report_serialization(self) -> None:
        ok_val = SchemaValidationResult(is_valid_json=True, is_schema_conformant=True)
        sample_res = SampleDriftResult(
            prompt_id="p1",
            category="extraction",
            baseline_validation=ok_val,
            candidate_validation=ok_val,
            token_jaccard_similarity=0.95,
            ngram_similarity=0.92,
            levenshtein_similarity=0.96,
            json_structural_similarity=1.0,
            json_exact_field_match=0.8,
            json_numeric_drift=0.05,
            composite_drift_score=0.08,
            latency_delta_ms=-15.2,
            drifted=False,
        )
        report = DriftBenchmarkReport(
            baseline_model="base-v1",
            candidate_model="cand-v2",
            total_samples=1,
            drifted_samples=0,
            baseline_json_conformity_rate=1.0,
            candidate_json_conformity_rate=1.0,
            mean_composite_drift=0.08,
            max_composite_drift=0.08,
            mean_token_jaccard=0.95,
            mean_levenshtein_similarity=0.96,
            mean_json_structural_similarity=1.0,
            mean_json_numeric_drift=0.05,
            baseline_p50_latency_ms=100.0,
            candidate_p50_latency_ms=84.8,
            baseline_p95_latency_ms=100.0,
            candidate_p95_latency_ms=84.8,
            passed=True,
            failure_reasons=(),
            results=(sample_res,),
        )
        self.assertEqual(report.drift_rate, 0.0)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["baseline_model"], "base-v1")
        self.assertEqual(len(parsed["results"]), 1)

        empty_report = DriftBenchmarkReport(
            baseline_model="b",
            candidate_model="c",
            total_samples=0,
            drifted_samples=0,
            baseline_json_conformity_rate=1.0,
            candidate_json_conformity_rate=1.0,
            mean_composite_drift=0.0,
            max_composite_drift=0.0,
            mean_token_jaccard=1.0,
            mean_levenshtein_similarity=1.0,
            mean_json_structural_similarity=None,
            mean_json_numeric_drift=None,
            baseline_p50_latency_ms=None,
            candidate_p50_latency_ms=None,
            baseline_p95_latency_ms=None,
            candidate_p95_latency_ms=None,
            passed=True,
        )
        self.assertEqual(empty_report.drift_rate, 0.0)
        thresholds = DriftThresholds()
        self.assertEqual(thresholds.max_composite_drift, 0.25)

    def test_pydantic_models_validation_and_serialization(self) -> None:
        prompt = TestPrompt.model_validate(
            {
                "prompt_id": "tp_01",
                "prompt": "Generate user profile JSON",
                "schema": {"type": "object"},
                "category": "structured",
            }
        )
        self.assertEqual(prompt.id, "tp_01")
        self.assertEqual(prompt.prompt_id, "tp_01")
        self.assertEqual(prompt.expected_schema, {"type": "object"})

        metrics = DriftMetrics(
            prompt_id="tp_01",
            category="structured",
            exact_match=True,
            baseline_valid_json=True,
            candidate_valid_json=True,
            format_adherence=True,
            semantic_similarity=1.0,
            token_diff_ratio=0.0,
            drift_score=0.0,
            baseline_latency_ms=100.0,
            candidate_latency_ms=80.0,
            latency_delta_ms=-20.0,
            baseline_tokens_per_sec=120.0,
            candidate_tokens_per_sec=150.0,
            tokens_per_sec_delta=30.0,
        )
        bench = BenchmarkResult(
            suite_name="test_suite",
            baseline_model="gpt-4o-base",
            candidate_model="gpt-4o-cand",
            total_prompts=1,
            exact_match_rate=1.0,
            baseline_json_adherence_rate=1.0,
            candidate_json_adherence_rate=1.0,
            mean_semantic_similarity=1.0,
            mean_token_diff_ratio=0.0,
            mean_drift_score=0.0,
            mean_baseline_latency_ms=100.0,
            mean_candidate_latency_ms=80.0,
            mean_baseline_tokens_per_sec=120.0,
            mean_candidate_tokens_per_sec=150.0,
            metrics=[metrics],
        )
        dumped = json.loads(bench.model_dump_json())
        self.assertEqual(dumped["suite_name"], "test_suite")
        self.assertEqual(dumped["metrics"][0]["prompt_id"], "tp_01")


    def test_model_sample_rejects_invalid_measurements(self) -> None:
        for latency in (-1, float("nan"), float("inf"), True):
            with self.subTest(latency=latency):
                with self.assertRaises(ValueError):
                    ModelSample(prompt_id="p", output="{}", latency_ms=latency)
                with self.assertRaises(ValueError):
                    ModelSample.from_dict({"prompt_id": "p", "output": "{}", "latency_ms": latency})
        for tokens in (-1, True, 1.5):
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):
                    ModelSample(prompt_id="p", output="{}", tokens_generated=tokens)
                with self.assertRaises(ValueError):
                    ModelSample.from_dict({
                        "prompt_id": "p", "output": "{}", "tokens_generated": tokens,
                    })
