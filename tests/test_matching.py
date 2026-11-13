"""Regressions for records that previously disappeared during matching."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from llm_drift_bench.cli import main
from llm_drift_bench.engine import (
    DriftBenchmarkEngine,
    evaluate_sample_pair,
    load_model_samples,
    load_prompt_cases,
)
from llm_drift_bench.evaluator import evaluate_suite, load_test_prompts
from llm_drift_bench.models import ModelSample, PromptCase, TestPrompt


class TestMatching(unittest.TestCase):
    def test_both_evaluators_reject_duplicate_and_unmatched_samples(self) -> None:
        p = ModelSample(prompt_id="p", output="{}")
        q = ModelSample(prompt_id="q", output="{}")
        cases = [([p, p], [p]), ([p], [p, p]), ([p, q], [p]), ([p], [p, q])]
        for baseline, candidate in cases:
            with self.subTest(baseline=baseline, candidate=candidate):
                with self.assertRaises(ValueError):
                    DriftBenchmarkEngine().evaluate(baseline, candidate)
                with self.assertRaises(ValueError):
                    evaluate_suite([], baseline, candidate)

    def test_suites_must_match_all_output_ids(self) -> None:
        samples = [ModelSample(prompt_id="p", output="{}")]
        for ids in (("q",), ("p", "q"), ("p", "p")):
            with self.subTest(ids=ids):
                cases = [PromptCase(prompt_id=pid, prompt="") for pid in ids]
                prompts = [TestPrompt(prompt_id=pid, prompt="") for pid in ids]
                with self.assertRaises(ValueError):
                    DriftBenchmarkEngine().evaluate(samples, samples, cases)
                with self.assertRaises(ValueError):
                    evaluate_suite(prompts, samples, samples)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            DriftBenchmarkEngine().evaluate(samples, samples, [])

    def test_pair_ids_and_blank_ids_are_rejected(self) -> None:
        p = ModelSample(prompt_id="p", output="{}")
        q = ModelSample(prompt_id="q", output="{}")
        with self.assertRaisesRegex(ValueError, "must match"):
            evaluate_sample_pair(p, q)
        with self.assertRaisesRegex(ValueError, "must match"):
            evaluate_sample_pair(p, p, PromptCase(prompt_id="q", prompt=""))
        for prompt_id in ("", " \t"):
            bad = ModelSample(prompt_id=prompt_id, output="{}")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                DriftBenchmarkEngine().evaluate([bad], [bad])
            with self.assertRaisesRegex(ValueError, "non-empty"):
                evaluate_suite([], [bad], [bad])

    def test_report_order_is_independent_of_file_order(self) -> None:
        p = ModelSample(prompt_id="p", output="{}")
        q = ModelSample(prompt_id="q", output="{}")
        engine = DriftBenchmarkEngine()
        report_a = engine.evaluate([q, p], [p, q])
        report_b = engine.evaluate([p, q], [q, p])
        self.assertEqual(report_a.to_dict(), report_b.to_dict())
        self.assertEqual([result.prompt_id for result in report_a.results], ["p", "q"])
        prompts = [TestPrompt(prompt_id="q"), TestPrompt(prompt_id="p")]
        result = evaluate_suite(prompts, [q, p], [p, q])
        self.assertEqual([metric.prompt_id for metric in result.metrics], ["p", "q"])

    def test_loaders_reject_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for suffix in ("json", "jsonl"):
                path = Path(tmp_dir) / f"duplicates.{suffix}"
                records = [{"prompt_id": "p", "output": "{}"}] * 2
                content = (
                    json.dumps(records) if suffix == "json"
                    else "\n".join(json.dumps(record) for record in records)
                )
                path.write_text(content, encoding="utf-8")
                for loader in (load_model_samples, load_prompt_cases, load_test_prompts):
                    with (
                        self.subTest(suffix=suffix, loader=loader.__name__),
                        self.assertRaisesRegex(ValueError, "Duplicate prompt_id"),
                    ):
                        loader(path)

    def test_cli_rejects_incomplete_and_empty_suites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            samples = root / "samples.json"
            samples.write_text(json.dumps([
                {"prompt_id": "p", "output": "{}"},
                {"prompt_id": "q", "output": "{}"},
            ]), encoding="utf-8")
            suite = root / "suite.json"
            result = root / "should-not-exist.json"
            for records in ([], [{"prompt_id": "p"}]):
                suite.write_text(json.dumps(records), encoding="utf-8")
                commands = [
                    ["run", "-b", str(samples), "-c", str(samples),
                     "-p", str(suite), "-o", str(result)],
                    ["validate", "-i", str(samples), "-p", str(suite)],
                ]
                for command in commands:
                    with self.subTest(command=command, suite=records):
                        with (
                            contextlib.redirect_stdout(io.StringIO()),
                            contextlib.redirect_stderr(io.StringIO()),
                        ):
                            code = main(command)
                        self.assertEqual(code, 2)
                        self.assertFalse(result.exists())
