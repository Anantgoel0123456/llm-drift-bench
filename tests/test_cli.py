"""Unit and integration tests for the llm-drift-bench CLI in llm_drift_bench.cli."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from llm_drift_bench.cli import main


class TestCLI(unittest.TestCase):
    """Test suite for CLI subcommands (`init-sample`, `run`, `validate`) and exit codes."""

    def test_no_args_prints_help(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main([])
        self.assertEqual(code, 0)
        self.assertIn("llm-drift-bench", buf.getvalue())

    def test_init_sample_and_run_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_buf = io.StringIO()
            with contextlib.redirect_stdout(out_buf):
                init_code = main(["init-sample", "--dir", str(root)])
            self.assertEqual(init_code, 0)

            prompts_file = root / "prompts.json"
            base_file = root / "baseline_fp16.jsonl"
            cand_file = root / "candidate_int4.jsonl"
            report_file = root / "out" / "report.json"

            self.assertTrue(prompts_file.exists())
            self.assertTrue(base_file.exists())
            self.assertTrue(cand_file.exists())

            # 1. Run with table format and output file
            table_buf = io.StringIO()
            with contextlib.redirect_stdout(table_buf):
                run_code = main(
                    [
                        "run",
                        "-b",
                        str(base_file),
                        "-c",
                        str(cand_file),
                        "-p",
                        str(prompts_file),
                        "-o",
                        str(report_file),
                        "--fail-on-drift",
                    ]
                )
            self.assertEqual(run_code, 0)
            self.assertTrue(report_file.exists())
            self.assertIn("LLM DRIFT BENCHMARK REPORT", table_buf.getvalue())

            # 2. Run with JSON format and ultra-strict threshold to trigger --fail-on-drift
            json_buf = io.StringIO()
            with contextlib.redirect_stdout(json_buf):
                fail_code = main(
                    [
                        "run",
                        "-b",
                        str(base_file),
                        "-c",
                        str(cand_file),
                        "-p",
                        str(prompts_file),
                        "--max-drift",
                        "0.0001",
                        "--format",
                        "json",
                        "--fail-on-drift",
                    ]
                )
            self.assertEqual(fail_code, 1)
            parsed_report = json.loads(json_buf.getvalue())
            self.assertEqual(parsed_report["total_samples"], 2)

    def test_validate_subcommand_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            main(["init-sample", "--dir", str(root)])
            prompts_file = root / "prompts.json"
            cand_file = root / "candidate_int4.jsonl"

            # 1. Validate with prompts file (table format)
            val_buf = io.StringIO()
            with contextlib.redirect_stdout(val_buf):
                code = main(
                    [
                        "validate",
                        "-i",
                        str(cand_file),
                        "-p",
                        str(prompts_file),
                        "--strict",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("JSON CONFORMITY VALIDATION", val_buf.getvalue())

            # 2. Validate with standalone global schema (json format) and failing sample
            bad_samples = root / "bad_samples.jsonl"
            bad_samples.write_text(
                '{"prompt_id": "s1", "output": "{\\"req\\": 123}"}\n',
                encoding="utf-8",
            )
            schema_file = root / "schema.json"
            schema_file.write_text(
                json.dumps(
                    {
                        "type": "object",
                        "required": ["req"],
                        "properties": {"req": {"type": "string"}},
                    }
                ),
                encoding="utf-8",
            )
            json_buf = io.StringIO()
            with contextlib.redirect_stdout(json_buf):
                strict_fail_code = main(
                    [
                        "validate",
                        "-i",
                        str(bad_samples),
                        "-s",
                        str(schema_file),
                        "--format",
                        "json",
                        "--strict",
                    ]
                )
            self.assertEqual(strict_fail_code, 1)
            val_summary = json.loads(json_buf.getvalue())
            self.assertEqual(val_summary["passed_samples"], 0)

            # 3. Invalid non-dict schema file triggers exit code 2
            bad_schema = root / "bad_schema.json"
            bad_schema.write_text("[1, 2]", encoding="utf-8")
            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                err_code = main(
                    ["validate", "-i", str(bad_samples), "-s", str(bad_schema)]
                )
            self.assertEqual(err_code, 2)
            self.assertIn("Error:", err_buf.getvalue())
