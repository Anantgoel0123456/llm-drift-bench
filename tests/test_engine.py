"""Unit tests for the benchmark evaluation engine and dataset loaders in llm_drift_bench.engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_drift_bench.engine import (
    DriftBenchmarkEngine,
    evaluate_sample_pair,
    format_report_table,
    load_model_samples,
    load_prompt_cases,
)
from llm_drift_bench.models import DriftThresholds, ModelSample, PromptCase


def test_evaluate_passing_suite_with_fixtures(
    sample_prompt_cases: list[PromptCase],
    baseline_samples: list[ModelSample],
    candidate_minor_drift_samples: list[ModelSample],
) -> None:
    """Verify DriftBenchmarkEngine passes a low-drift candidate run using pytest fixtures."""
    engine = DriftBenchmarkEngine()
    report = engine.evaluate(
        baseline_samples,
        candidate_minor_drift_samples,
        sample_prompt_cases,
    )
    assert report.passed
    assert report.drifted_samples == 0
    assert report.candidate_json_conformity_rate == 1.0
    table = format_report_table(report)
    assert "PASSED" in table
    assert "model-fp16" in table
    assert "model-q4_k_m" in table


def test_evaluate_failing_suite_with_fixtures(
    sample_prompt_cases: list[PromptCase],
    baseline_samples: list[ModelSample],
    candidate_major_drift_samples: list[ModelSample],
) -> None:
    """Verify DriftBenchmarkEngine flags threshold violations on a high-drift candidate run."""
    engine = DriftBenchmarkEngine(
        thresholds=DriftThresholds(
            max_composite_drift=0.15,
            min_json_conformity_rate=0.95,
            max_numeric_relative_drift=0.10,
            max_latency_regression_pct=25.0,
        )
    )
    report = engine.evaluate(
        baseline_samples,
        candidate_major_drift_samples,
        sample_prompt_cases,
    )
    assert not report.passed
    assert report.drifted_samples == 2
    assert len(report.failure_reasons) >= 3
    table = format_report_table(report)
    assert "FAILED" in table
    assert "Threshold Violations:" in table


class TestEngine(unittest.TestCase):
    """Test suite for DriftBenchmarkEngine input validation, evaluate_sample_pair, and loaders."""

    def test_engine_input_validation_errors(self) -> None:
        engine = DriftBenchmarkEngine()
        sample = ModelSample(prompt_id="p1", output="hello")
        with self.assertRaises(ValueError):
            engine.evaluate([], [sample])
        with self.assertRaises(ValueError):
            engine.evaluate([sample], [])
        with self.assertRaises(ValueError):
            engine.evaluate([sample], [ModelSample(prompt_id="other", output="world")])

    def test_evaluate_sample_pair_without_prompt_case(self) -> None:
        base = ModelSample(prompt_id="p1", output="The sky is blue")
        cand = ModelSample(prompt_id="p1", output="The sky is clear and blue")
        res = evaluate_sample_pair(base, cand)
        self.assertEqual(res.category, "general")
        self.assertIsNone(res.latency_delta_ms)

        engine = DriftBenchmarkEngine()
        plain_report = engine.evaluate([base], [cand])
        self.assertEqual(plain_report.candidate_json_conformity_rate, 1.0)

    def test_dataset_loaders_json_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            # 1. JSONL model samples
            jsonl_path = root / "samples.jsonl"
            jsonl_path.write_text(
                '{"prompt_id": "p1", "output": "{\\"ok\\": true}"}\n\n'
                '{"prompt_id": "p2", "output": "plain text"}\n',
                encoding="utf-8",
            )
            loaded_jsonl = load_model_samples(jsonl_path)
            self.assertEqual(len(loaded_jsonl), 2)
            self.assertEqual(loaded_jsonl[0].model_name, "samples")

            # 2. JSON wrapped object with model_name
            wrapped_path = root / "wrapped.json"
            wrapped_path.write_text(
                json.dumps(
                    {
                        "model_name": "custom-m1",
                        "samples": [{"prompt_id": "p1", "output": "hi"}],
                    }
                ),
                encoding="utf-8",
            )
            loaded_wrapped = load_model_samples(wrapped_path)
            self.assertEqual(loaded_wrapped[0].model_name, "custom-m1")

            # 3. JSON list of PromptCases
            prompts_path = root / "prompts.json"
            prompts_path.write_text(
                json.dumps([{"prompt_id": "p1", "prompt": "Say hi"}]),
                encoding="utf-8",
            )
            loaded_prompts = load_prompt_cases(prompts_path)
            self.assertIsInstance(loaded_prompts[0], PromptCase)

            # 4. Empty file
            empty_path = root / "empty.json"
            empty_path.write_text("   ", encoding="utf-8")
            self.assertEqual(load_model_samples(empty_path), [])

            # 5. Invalid formats
            bad_jsonl = root / "bad.jsonl"
            bad_jsonl.write_text("[1, 2, 3]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model_samples(bad_jsonl)

            bad_list = root / "bad_list.json"
            bad_list.write_text("[123]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model_samples(bad_list)

            bad_dict = root / "bad_dict.json"
            bad_dict.write_text('{"unknown_key": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model_samples(bad_dict)

    def test_incomplete_latency_skips_regression_check(self) -> None:
        baseline = [
            ModelSample(prompt_id="p1", output="{}", latency_ms=100),
            ModelSample(prompt_id="p2", output="{}"),
        ]
        candidate = [
            ModelSample(prompt_id="p1", output="{}", latency_ms=1000),
            ModelSample(prompt_id="p2", output="{}", latency_ms=1000),
        ]
        report = DriftBenchmarkEngine().evaluate(baseline, candidate)
        self.assertIsNone(report.baseline_p95_latency_ms)
        self.assertEqual(report.candidate_p95_latency_ms, 1000)
        self.assertTrue(report.passed)
        self.assertIn("incomplete measurements", format_report_table(report))
        self.assertIn("n/a", format_report_table(report))

    def test_zero_latency_is_not_missing(self) -> None:
        baseline = [ModelSample(prompt_id="p", output="{}", latency_ms=0)]
        candidate = [ModelSample(prompt_id="p", output="{}", latency_ms=10)]
        report = DriftBenchmarkEngine().evaluate(baseline, candidate)
        self.assertEqual(report.baseline_p95_latency_ms, 0)
        self.assertIn("zero baseline P95", format_report_table(report))
