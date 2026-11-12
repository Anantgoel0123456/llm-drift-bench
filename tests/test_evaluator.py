"""Unit tests for evaluation logic, drift score math, and throughput in evaluator.py."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from llm_drift_bench.evaluator import (
    check_json_adherence,
    compute_semantic_similarity,
    compute_token_diff_ratio,
    compute_tokens_per_sec,
    evaluate_response_pair,
    evaluate_suite,
    load_benchmark_result,
    load_test_prompts,
)
from llm_drift_bench.models import ModelSample, TestPrompt


class TestEvaluator(unittest.TestCase):
    """Test suite for evaluator helpers, evaluate_response_pair, and evaluate_suite."""

    def test_check_json_adherence_and_token_diff_math(self) -> None:
        valid_json, adheres, payload = check_json_adherence(
            '{"status": "ok"}',
            expected_schema={"type": "object", "required": ["status"]},
        )
        self.assertTrue(valid_json)
        self.assertTrue(adheres)
        self.assertEqual(payload, {"status": "ok"})

        self.assertEqual(compute_token_diff_ratio("same text", "same text"), 0.0)
        self.assertEqual(compute_token_diff_ratio("   ", "   "), 0.0)
        self.assertEqual(compute_token_diff_ratio("hello", ""), 1.0)
        diff = compute_token_diff_ratio("alpha beta gamma", "alpha beta delta")
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 1.0)

    def test_semantic_similarity_and_tokens_per_sec(self) -> None:
        self.assertEqual(compute_semantic_similarity("exact", "exact"), 1.0)
        json_sim = compute_semantic_similarity('{"a": 1, "b": 2}', '{"a": 1, "b": 3}')
        self.assertGreater(json_sim, 0.5)
        self.assertLess(json_sim, 1.0)

        text_sim = compute_semantic_similarity("fast inference engine", "slow inference engine")
        self.assertGreater(text_sim, 0.4)

        self.assertIsNone(compute_tokens_per_sec("hello world", 0.0))
        self.assertEqual(compute_tokens_per_sec("hello world", 500.0, explicit_tokens=50), 100.0)
        self.assertIsNone(compute_tokens_per_sec("one two three four", 250.0))

    def test_evaluate_response_pair_json_and_text_formats(self) -> None:
        json_prompt = TestPrompt(
            id="p_json",
            prompt="Return JSON",
            expected_format="json",
            expected_schema={"type": "object", "required": ["key"]},
        )
        m_ok = evaluate_response_pair(
            json_prompt,
            baseline_output='{"key": "val"}',
            candidate_output='{"key": "val"}',
            baseline_latency_ms=100.0,
            candidate_latency_ms=80.0,
        )
        self.assertTrue(m_ok.exact_match)
        self.assertTrue(m_ok.format_adherence)
        self.assertEqual(m_ok.drift_score, 0.0)
        self.assertEqual(m_ok.latency_delta_ms, -20.0)

        # Candidate breaks JSON schema adherence -> drift_score penalized >= 0.75
        m_broken = evaluate_response_pair(
            json_prompt,
            baseline_output='{"key": "val"}',
            candidate_output="not json at all",
        )
        self.assertFalse(m_broken.format_adherence)
        self.assertGreaterEqual(m_broken.drift_score, 0.75)

        # Plain text format prompt
        txt_prompt = TestPrompt(id="p_txt", prompt="Say hi", expected_format="text")
        m_txt = evaluate_response_pair(txt_prompt, "Hello there", "Hello there!")
        self.assertTrue(m_txt.format_adherence)
        self.assertFalse(m_txt.exact_match)

    def test_evaluate_suite_with_captured_samples(self) -> None:
        prompts = [
            TestPrompt(
                id="p1",
                prompt="Extract",
                expected_schema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"type": "string", "enum": ["paid"]},
                        "const_f": {"type": "string", "const": "USD"},
                        "amount": {"type": "number"},
                        "count": {"type": "integer"},
                        "flag": {"type": "boolean"},
                        "tags": {"type": "array"},
                    },
                },
            ),
            TestPrompt(
                id="p2",
                prompt="Inline outputs",
                baseline_output='{"ok": true}',
                candidate_output='{"ok": true}',
            ),
        ]

        # Model identifiers must not produce fabricated outputs, even with inline data.
        with self.assertRaisesRegex(ValueError, "existing outputs file"):
            evaluate_suite(prompts, "llama-3-fp16", "llama-3-int4")

        # 2. Sequence of ModelSamples with empty prompts list (auto-discovers shared IDs)
        samples_a = [ModelSample(prompt_id="p1", model_name="mA", output='{"x": 1}')]
        samples_b = [ModelSample(prompt_id="p1", model_name="mB", output='{"x": 1}')]
        res_seq = evaluate_suite([], samples_a, samples_b)
        self.assertEqual(res_seq.exact_match_rate, 1.0)

        with self.assertRaises(ValueError):
            evaluate_suite([], [], [])
        with self.assertRaises(ValueError):
            evaluate_suite(prompts, samples_a, samples_b)

    def test_evaluate_suite_files_and_missing_targets(self) -> None:
        prompts = [TestPrompt(prompt_id="p1", prompt="Return JSON")]
        samples = [ModelSample(prompt_id="p1", output='{"x": 1}')]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            outputs = root / "captured outputs.json"
            outputs.write_text(json.dumps([
                {"prompt_id": "p1", "output": '{"x": 1}', "model_name": "captured"}
            ]), encoding="utf-8")
            result = evaluate_suite(prompts, str(outputs), outputs)
            self.assertEqual(result.baseline_model, "captured")
            self.assertEqual(result.total_prompts, 1)
            self.assertEqual(result.exact_match_rate, 1.0)
            for target in (root / "missing.json", str(root / "missing.jsonl"), root):
                with self.subTest(target=target):
                    with self.assertRaisesRegex(ValueError, "existing outputs file"):
                        evaluate_suite(prompts, target, samples)
                    with self.assertRaisesRegex(ValueError, "existing outputs file"):
                        evaluate_suite(prompts, samples, target)

    def test_load_test_prompts_and_benchmark_result_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            # 1. Empty suite file
            empty_p = root / "empty.json"
            empty_p.write_text("", encoding="utf-8")
            self.assertEqual(load_test_prompts(empty_p), [])

            # 2. JSONL suite file
            jsonl_p = root / "suite.jsonl"
            jsonl_p.write_text('{"id": "p1", "prompt": "Test"}\n', encoding="utf-8")
            self.assertEqual(len(load_test_prompts(jsonl_p)), 1)

            # 3. Dict-wrapped suite file
            dict_p = root / "suite_dict.json"
            dict_p.write_text(
                json.dumps({"prompts": [{"id": "p1", "prompt": "Test"}]}),
                encoding="utf-8",
            )
            self.assertEqual(len(load_test_prompts(dict_p)), 1)

            # 4. Invalid suite file
            bad_p = root / "bad.json"
            bad_p.write_text("123", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_test_prompts(bad_p)

            # 5. Load BenchmarkResult directly and from legacy DriftBenchmarkReport schema
            legacy_file = root / "legacy_report.json"
            legacy_file.write_text(
                json.dumps(
                    {
                        "baseline_model": "base-m",
                        "candidate_model": "cand-m",
                        "baseline_json_conformity_rate": 1.0,
                        "candidate_json_conformity_rate": 1.0,
                        "mean_composite_drift": 0.05,
                        "baseline_p50_latency_ms": 110.0,
                        "candidate_p50_latency_ms": 90.0,
                        "results": [
                            {
                                "prompt_id": "p1",
                                "category": "extraction",
                                "baseline_validation": {"is_valid_json": True, "passed": True},
                                "candidate_validation": {"is_valid_json": True, "passed": True},
                                "levenshtein_similarity": 0.95,
                                "composite_drift_score": 0.05,
                                "latency_delta_ms": -20.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            converted = load_benchmark_result(legacy_file)
            self.assertEqual(converted.baseline_model, "base-m")
            self.assertEqual(converted.total_prompts, 1)

    def test_missing_measurements_remain_none(self) -> None:
        prompt = TestPrompt(prompt_id="p", prompt="Return JSON")
        metric = evaluate_response_pair(prompt, "{}", "{}")
        for field in (
            "baseline_latency_ms", "candidate_latency_ms", "latency_delta_ms",
            "baseline_tokens_per_sec", "candidate_tokens_per_sec", "tokens_per_sec_delta",
        ):
            self.assertIsNone(getattr(metric, field))
            self.assertIsNone(json.loads(metric.model_dump_json())[field])
        self.assertIsNone(compute_tokens_per_sec("{}", None, 10))
        self.assertIsNone(compute_tokens_per_sec("{}", 0, 10))
        self.assertEqual(compute_tokens_per_sec("", 500, 0), 0.0)

    def test_partial_suite_does_not_invent_means(self) -> None:
        baseline = [
            ModelSample(prompt_id="p1", output="{}", latency_ms=100, tokens_generated=10),
            ModelSample(prompt_id="p2", output="{}"),
        ]
        candidate = [
            ModelSample(prompt_id="p1", output="{}", latency_ms=200, tokens_generated=10),
            ModelSample(prompt_id="p2", output="{}", latency_ms=400, tokens_generated=20),
        ]
        result = evaluate_suite([], baseline, candidate)
        self.assertIsNone(result.mean_baseline_latency_ms)
        self.assertIsNone(result.mean_baseline_tokens_per_sec)
        self.assertEqual(result.mean_candidate_latency_ms, 300)
        self.assertEqual(result.mean_candidate_tokens_per_sec, 50)
        self.assertEqual(result.metrics[0].latency_delta_ms, 100)
        self.assertIsNone(result.metrics[1].latency_delta_ms)
        self.assertIsNone(result.metrics[1].tokens_per_sec_delta)

    def test_invalid_measurements_are_rejected(self) -> None:
        for latency in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(latency=latency), self.assertRaises(ValueError):
                compute_tokens_per_sec("{}", latency, 10)
        for tokens in (-1, True, 1.5):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                compute_tokens_per_sec("{}", 100, tokens)

    def test_legacy_report_does_not_invent_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "report.json"
            report_path.write_text(json.dumps({
                "baseline_p50_latency_ms": 120,
                "candidate_p50_latency_ms": 90,
                "results": [{"prompt_id": "p", "latency_delta_ms": 0}],
            }), encoding="utf-8")
            result = load_benchmark_result(report_path)
            self.assertIsNone(result.mean_baseline_latency_ms)
            self.assertIsNone(result.mean_candidate_latency_ms)
            self.assertIsNone(result.mean_baseline_tokens_per_sec)
            self.assertIsNone(result.mean_candidate_tokens_per_sec)
            self.assertIsNone(result.metrics[0].baseline_latency_ms)
            self.assertIsNone(result.metrics[0].candidate_latency_ms)
            self.assertIsNone(result.metrics[0].candidate_tokens_per_sec)
            self.assertEqual(result.metrics[0].latency_delta_ms, 0)
