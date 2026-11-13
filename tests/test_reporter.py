"""Unit tests for Rich console output and Markdown report export in llm_drift_bench.reporter."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

from llm_drift_bench.models import BenchmarkResult, DriftMetrics
from llm_drift_bench.reporter import export_markdown_report, render_console_report


class TestReporter(unittest.TestCase):
    """Test suite for render_console_report and export_markdown_report."""

    def setUp(self) -> None:
        self.sample_result = BenchmarkResult(
            suite_name="invoice_suite",
            baseline_model="llama-3-8b-fp16",
            candidate_model="llama-3-8b-q4_k_m",
            total_prompts=2,
            exact_match_rate=0.5,
            baseline_json_adherence_rate=1.0,
            candidate_json_adherence_rate=0.5,
            mean_semantic_similarity=0.82,
            mean_token_diff_ratio=0.18,
            mean_drift_score=0.18,
            mean_baseline_latency_ms=120.0,
            mean_candidate_latency_ms=85.0,
            mean_baseline_tokens_per_sec=95.0,
            mean_candidate_tokens_per_sec=130.0,
            metrics=[
                DriftMetrics(
                    prompt_id="p_ok",
                    category="extraction",
                    exact_match=True,
                    baseline_valid_json=True,
                    candidate_valid_json=True,
                    format_adherence=True,
                    semantic_similarity=1.0,
                    token_diff_ratio=0.0,
                    drift_score=0.0,
                    baseline_latency_ms=120.0,
                    candidate_latency_ms=80.0,
                    latency_delta_ms=-40.0,
                    baseline_tokens_per_sec=100.0,
                    candidate_tokens_per_sec=140.0,
                    tokens_per_sec_delta=40.0,
                ),
                DriftMetrics(
                    prompt_id="p_drift",
                    category="classification",
                    exact_match=False,
                    baseline_valid_json=True,
                    candidate_valid_json=False,
                    format_adherence=False,
                    semantic_similarity=0.64,
                    token_diff_ratio=0.36,
                    drift_score=0.36,
                    baseline_latency_ms=120.0,
                    candidate_latency_ms=90.0,
                    latency_delta_ms=-30.0,
                    baseline_tokens_per_sec=90.0,
                    candidate_tokens_per_sec=120.0,
                    tokens_per_sec_delta=30.0,
                ),
            ],
        )

    def test_render_console_report_with_rich(self) -> None:
        quiet_buf = io.StringIO()
        custom_console = Console(file=quiet_buf, width=100, force_terminal=False)
        captured = render_console_report(self.sample_result, console=custom_console)
        self.assertIn("llama-3-8b-fp16", captured)
        self.assertIn("llama-3-8b-q4_k_m", captured)
        self.assertIn("p_ok", captured)
        self.assertIn("p_drift", captured)

    def test_export_markdown_report_and_file_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "reports" / "drift.md"
            md_text = export_markdown_report(self.sample_result, output_path=md_path)
            self.assertTrue(md_path.exists())
            self.assertEqual(md_path.read_text(encoding="utf-8"), md_text)
            self.assertIn("# LLM Drift Benchmark Report: `invoice_suite`", md_text)
            self.assertIn("| `p_ok` | extraction | PASS | YES |", md_text)
            self.assertIn("| `p_drift` | classification | FAIL | NO |", md_text)

    def test_missing_measurements_render_as_na(self) -> None:
        result = self.sample_result.model_copy(deep=True)
        result.mean_baseline_latency_ms = None
        result.mean_candidate_latency_ms = None
        result.mean_baseline_tokens_per_sec = None
        result.mean_candidate_tokens_per_sec = None
        for metric in result.metrics:
            metric.latency_delta_ms = None
            metric.candidate_tokens_per_sec = None
        markdown = export_markdown_report(result)
        self.assertIn("| **Mean Latency (ms)** | n/a | n/a |", markdown)
        self.assertIn("| **Mean Throughput (tokens/sec)** | n/a | n/a |", markdown)
        self.assertIn("| n/a | n/a |", markdown)
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        self.assertIn("n/a", render_console_report(result, console=console))
