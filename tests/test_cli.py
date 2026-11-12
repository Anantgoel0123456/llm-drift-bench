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

    def test_run_suite_with_captured_files_and_report_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            main(["init-sample", "--dir", str(root)])
            prompts_file = root / "prompts.json"
            results_json = root / "results.json"
            report_md = root / "report.md"

            # Compare captured files, then render the saved report.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(
                    [
                        "run",
                        "--suite",
                        str(prompts_file),
                        "--baseline",
                        str(root / "baseline_fp16.jsonl"),
                        "--candidate",
                        str(root / "candidate_int4.jsonl"),
                        "-o",
                        str(results_json),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(results_json.exists())

            # 2. llm-drift-bench report --input <results.json> --format markdown
            md_buf = io.StringIO()
            with contextlib.redirect_stdout(md_buf):
                rep_code = main(
                    [
                        "report",
                        "--input",
                        str(results_json),
                        "--format",
                        "markdown",
                        "-o",
                        str(report_md),
                    ]
                )
            self.assertEqual(rep_code, 0)
            self.assertTrue(report_md.exists())
            self.assertIn("# LLM Drift Benchmark Report:", md_buf.getvalue())

            # 3. report --format json and --format console
            json_rep_out = root / "rep_out.json"
            console_rep_out = root / "rep_out.txt"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "report",
                            "--input",
                            str(results_json),
                            "--format",
                            "json",
                            "-o",
                            str(json_rep_out),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "report",
                            "--input",
                            str(results_json),
                            "--format",
                            "console",
                            "-o",
                            str(console_rep_out),
                        ]
                    ),
                    0,
                )
            self.assertTrue(json_rep_out.exists())
            self.assertTrue(console_rep_out.exists())

            # Model identifiers are rejected as input errors.
            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                err_code = main(["run", "--baseline", "model_a", "--candidate", "model_b"])
            self.assertEqual(err_code, 2)

    def test_missing_targets_never_create_benchmark_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["init-sample", "--dir", str(root)])
            baseline = str(root / "baseline_fp16.jsonl")
            candidate = str(root / "candidate_int4.jsonl")
            missing = str(root / "missing outputs.json")
            targets = [
                (missing, candidate),
                (baseline, missing),
                (missing, missing),
                (str(root), candidate),
                ("model-a", "model-b"),
            ]
            for base, cand in targets:
                for output_format in ("table", "json", "rich", "markdown"):
                    with self.subTest(base=base, cand=cand, format=output_format):
                        result_path = root / "must-not-exist.json"
                        stdout, stderr = io.StringIO(), io.StringIO()
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            code = main([
                                "run", "-b", base, "-c", cand,
                                "--suite", str(root / "prompts.json"),
                                "--format", output_format, "--output", str(result_path),
                            ])
                        self.assertEqual(code, 2)
                        self.assertEqual(stdout.getvalue(), "")
                        self.assertIn("existing outputs file", stderr.getvalue())
                        self.assertFalse(result_path.exists())

