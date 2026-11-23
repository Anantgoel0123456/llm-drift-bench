"""Command-line interface for llm-drift-bench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, cast

from llm_drift_bench import __version__

if TYPE_CHECKING:
    from llm_drift_bench.comparison import SortKey
    from llm_drift_bench.models import DriftBenchmarkReport, DriftThresholds, ModelSample
    from llm_drift_bench.policy import DriftPolicy
    from llm_drift_bench.tool_log import ToolCheckRecord

# UNKNOWN is neither success nor an error: scripts must decide what it means.
EXIT_TOOL_UNKNOWN = 3


def _add_tool_check_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tool-check",
        action="append",
        default=[],
        metavar="FILE",
        help=(
            "Saved 'check-tools' result to show in a separate Tool evidence section "
            "(repeatable; markdown or html output only). Never changes the drift verdict."
        ),
    )


def _load_tool_checks(paths: Sequence[str], output_format: str) -> list[ToolCheckRecord]:
    if not paths:
        return []
    if output_format not in ("markdown", "html", "pr-comment"):
        raise ValueError("--tool-check requires markdown or html (or pr-comment) output.")
    from llm_drift_bench.tool_log import load_tool_check

    return [load_tool_check(path) for path in paths]


def _add_threshold_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help="Optional drift.toml or drift.json policy file (thresholds, weights, field rules).",
    )
    parser.add_argument(
        "--long-text-mode",
        choices=("error", "skip_levenshtein"),
        default=None,
        help="Behavior for outputs exceeding 16,384 chars: error (default) or skip_levenshtein.",
    )
    parser.add_argument(
        "--max-drift",
        type=float,
        default=None,
        help="Maximum allowed composite drift score (default: 0.25).",
    )
    parser.add_argument(
        "--min-conformity",
        type=float,
        default=None,
        help="Minimum required candidate JSON schema conformity rate (default: 0.95).",
    )
    parser.add_argument(
        "--max-numeric-drift",
        type=float,
        default=None,
        help="Maximum allowed relative drift on numeric JSON fields (default: 0.15).",
    )
    parser.add_argument(
        "--max-latency-regression",
        type=float,
        default=None,
        help="Maximum allowed P95 latency regression percentage (default: 50.0).",
    )


def _add_run_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'run' subcommand and its CLI flags."""
    run_parser = subparsers.add_parser(
        "run",
        help="Compare captured baseline and candidate LLM outputs.",
    )
    run_parser.add_argument(
        "-s",
        "--suite",
        "-p",
        "--prompts",
        dest="suite",
        default=None,
        help="Path to prompt suite JSON/JSONL file (--suite or --prompts).",
    )
    run_parser.add_argument(
        "-b",
        "--baseline",
        required=True,
        help="Path to captured baseline outputs (.json/.jsonl).",
    )
    run_parser.add_argument(
        "-c",
        "--candidate",
        required=True,
        help="Path to captured candidate outputs (.json/.jsonl).",
    )
    _add_threshold_options(run_parser)
    run_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional file path to write the full JSON benchmark result.",
    )
    run_parser.add_argument(
        "--format",
        choices=("table", "rich", "json", "markdown", "html", "junit", "pr-comment", "csv"),
        default="table",
        help="Stdout format: table (default), rich, json, markdown, html, junit, pr-comment, csv.",
    )
    run_parser.add_argument(
        "--include-diffs",
        action="store_true",
        help="Include opt-in collapsible response diffs in HTML output.",
    )
    run_parser.add_argument(
        "--junit-output", metavar="FILE",
        help="Write JUnit XML alongside the console result and optional JSON report.",
    )
    run_parser.add_argument(
        "--github-summary", action="store_true",
        help="Append the benchmark verdict and metrics to GITHUB_STEP_SUMMARY.",
    )
    _add_tool_check_option(run_parser)
    run_parser.add_argument(
        "--fail-on-drift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Return exit code 1 on drift or threshold failure (default). "
            "Use --no-fail-on-drift for an informational run."
        ),
    )


def _protect_inputs(output: str | None, *inputs: str | None) -> None:
    """Refuse to overwrite a capture, saved report, or attached tool check."""
    if output is None:
        return
    target = Path(output)
    if target.exists():
        for source in inputs:
            if source and Path(source).exists() and target.samefile(source):
                raise ValueError(f"Choose a different --output path. {source!r} is an input file.")


def _resolve_policy_and_thresholds(
    args: argparse.Namespace,
) -> tuple[DriftThresholds, DriftPolicy | None]:
    from llm_drift_bench.models import DriftThresholds
    from llm_drift_bench.policy import DriftPolicy, load_policy

    policy: DriftPolicy | None = load_policy(args.config) if args.config else None
    base_thresholds = policy.thresholds if policy is not None else DriftThresholds()
    thresholds = DriftThresholds(
        max_composite_drift=(
            args.max_drift if args.max_drift is not None else base_thresholds.max_composite_drift
        ),
        min_json_conformity_rate=(
            args.min_conformity
            if args.min_conformity is not None
            else base_thresholds.min_json_conformity_rate
        ),
        max_numeric_relative_drift=(
            args.max_numeric_drift
            if args.max_numeric_drift is not None
            else base_thresholds.max_numeric_relative_drift
        ),
        max_latency_regression_pct=(
            args.max_latency_regression
            if args.max_latency_regression is not None
            else base_thresholds.max_latency_regression_pct
        ),
    )
    if args.long_text_mode is not None:
        if policy is None:
            policy = DriftPolicy(thresholds=thresholds, long_text_mode=args.long_text_mode)
        else:
            policy = DriftPolicy(
                thresholds=thresholds,
                weights=policy.weights,
                field_rules=policy.field_rules,
                long_text_mode=args.long_text_mode,
            )
    elif policy is not None:
        policy = DriftPolicy(
            thresholds=thresholds,
            weights=policy.weights,
            field_rules=policy.field_rules,
            long_text_mode=policy.long_text_mode,
        )
    return thresholds, policy


def _handle_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    for label, target in (("baseline", args.baseline), ("candidate", args.candidate)):
        if not Path(target).is_file():
            raise ValueError(
                f"--{label} must name an existing outputs file: {target!r}. "
                "Model identifiers are not supported; supply captured JSON/JSONL outputs."
            )

    _protect_inputs(
        args.output, args.baseline, args.candidate, args.suite, args.config, *args.tool_check
    )
    if args.junit_output:
        _protect_inputs(args.junit_output, args.baseline, args.candidate, args.suite,
                        args.config, args.output, *args.tool_check)
        if args.output and Path(args.output).resolve() == Path(args.junit_output).resolve():
            raise ValueError("--output and --junit-output require different files.")
    summary_path = None
    if args.github_summary:
        destination = os.environ.get("GITHUB_STEP_SUMMARY")
        if not destination or not Path(destination).is_file():
            raise ValueError("--github-summary requires an existing GITHUB_STEP_SUMMARY file.")
        _protect_inputs(destination, args.baseline, args.candidate, args.suite, args.config,
                        args.output, args.junit_output, *args.tool_check)
        summary_path = Path(destination)
    thresholds, policy = _resolve_policy_and_thresholds(args)
    from llm_drift_bench.engine import (
        DriftBenchmarkEngine,
        format_report_table,
        load_model_samples,
        load_prompt_cases,
    )

    baseline_samples = load_model_samples(args.baseline)
    candidate_samples = load_model_samples(args.candidate)
    prompt_cases = load_prompt_cases(args.suite) if args.suite else None
    tool_checks = _load_tool_checks(args.tool_check, args.format)

    engine = DriftBenchmarkEngine(thresholds=thresholds, policy=policy)
    report = engine.evaluate(
        baseline_samples=baseline_samples,
        candidate_samples=candidate_samples,
        prompt_cases=prompt_cases,
    )

    if args.output:
        _atomic_write_text(Path(args.output), report.to_json(indent=2))

    if args.junit_output:
        from llm_drift_bench.junit import export_junit_report

        _atomic_write_text(Path(args.junit_output), export_junit_report(report))

    if args.format == "junit":
        from llm_drift_bench.junit import export_junit_report

        print(export_junit_report(report), end="")
    elif args.format == "json":
        print(report.to_json(indent=2))
    elif args.format == "markdown":
        from llm_drift_bench.reporter import export_markdown_report

        print(export_markdown_report(report, tool_checks=tool_checks), end="")
    elif args.format == "pr-comment":
        from llm_drift_bench.reporter import export_pr_comment

        print(export_pr_comment(report, tool_checks=tool_checks), end="")
    elif args.format == "html":
        from llm_drift_bench.reporter import export_html_report

        diffs_map: dict[str, tuple[str, str]] | None = None
        if args.include_diffs:
            base_by_id = {s.prompt_id: s.output for s in baseline_samples}
            cand_by_id = {s.prompt_id: s.output for s in candidate_samples}
            diffs_map = {pid: (base_by_id[pid], cand_by_id[pid]) for pid in base_by_id}
        print(export_html_report(report, tool_checks=tool_checks, diffs=diffs_map), end="")
    elif args.format == "csv":
        from llm_drift_bench.csv_export import export_csv_report

        print(export_csv_report(report), end="")
    elif args.format == "rich":
        from llm_drift_bench.reporter import render_console_report

        render_console_report(report)
    else:
        print(format_report_table(report))

    if summary_path is not None:
        verdict = "PASSED" if report.passed else "FAILED"
        with summary_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"\n## LLM drift comparison: {verdict}\n\n"
                f"- Drifted prompts: {report.drifted_samples} of {report.total_samples}\n"
                f"- Mean composite drift: {report.mean_composite_drift:.4f}\n"
                f"- Candidate JSON conformity: {report.candidate_json_conformity_rate:.1%}\n"
            )

    if args.fail_on_drift and not report.passed:
        return 1
    return 0


def _add_report_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the 'report' subcommand for formatting saved benchmark results."""
    rep_parser = subparsers.add_parser(
        "report",
        help="Render a saved benchmark results JSON file as Markdown or Rich console tables.",
    )
    rep_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to saved benchmark results JSON file.",
    )
    rep_parser.add_argument(
        "-f",
        "--format",
        choices=("markdown", "console", "table", "json", "html", "junit", "pr-comment", "csv"),
        default="markdown",
        help="Format: markdown (default), console/table, json, html, junit, pr-comment, or csv.",
    )
    rep_parser.add_argument(
        "--previous",
        metavar="FILE",
        default=None,
        help="Optional previous saved engine report to show metric deltas in pr-comment output.",
    )
    rep_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output file path to write the rendered report.",
    )
    _add_tool_check_option(rep_parser)


def _handle_report(args: argparse.Namespace) -> int:
    """Execute the 'report' subcommand."""
    tool_checks = _load_tool_checks(args.tool_check, args.format)
    if not Path(args.input).is_file():
        raise ValueError(
            f"Saved report file not found: {args.input!r}. "
            "Create it with 'run --output', or try -i examples/expected-pass.json "
            "from the repository root."
        )
    _protect_inputs(args.output, args.input, args.previous, *args.tool_check)
    from llm_drift_bench.models import DriftBenchmarkReport
    from llm_drift_bench.report_io import load_engine_report, load_report

    result = load_report(args.input)
    previous_report = load_engine_report(args.previous) if args.previous else None
    if args.format == "junit":
        if not isinstance(result, DriftBenchmarkReport):
            raise ValueError("JUnit export requires a canonical engine report.")
        from llm_drift_bench.junit import export_junit_report

        xml = export_junit_report(result)
        if args.output:
            _atomic_write_text(Path(args.output), xml)
        print(xml, end="")
    elif args.format == "markdown":
        from llm_drift_bench.reporter import export_markdown_report

        md = export_markdown_report(result, output_path=args.output, tool_checks=tool_checks)
        print(md, end="")
    elif args.format == "pr-comment":
        from llm_drift_bench.reporter import export_pr_comment

        comment = export_pr_comment(
            result,
            output_path=args.output,
            previous=previous_report,
            tool_checks=tool_checks,
        )
        print(comment, end="")
    elif args.format == "json":
        payload = (result.to_json(indent=2) if isinstance(result, DriftBenchmarkReport)
                   else result.model_dump_json(indent=2))
        if args.output:
            _atomic_write_text(Path(args.output), payload)
        print(payload)
    elif args.format == "csv":
        if not isinstance(result, DriftBenchmarkReport):
            raise ValueError("CSV export requires a canonical engine report.")
        from llm_drift_bench.csv_export import export_csv_report

        csv_text = export_csv_report(result)
        if args.output:
            _atomic_write_text(Path(args.output), csv_text)
        print(csv_text, end="")
    elif args.format == "html":
        from llm_drift_bench.reporter import export_html_report

        print(
            export_html_report(result, output_path=args.output, tool_checks=tool_checks), end=""
        )
    else:
        from llm_drift_bench.reporter import render_console_report

        rendered = render_console_report(result)
        if args.output:
            _atomic_write_text(Path(args.output), rendered)
    return 0


def _add_validate_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the 'validate' subcommand and its CLI flags."""
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate LLM outputs for JSON syntax and JSON Schema conformity.",
    )
    val_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to model outputs file (.json or .jsonl).",
    )
    val_parser.add_argument(
        "-s",
        "--schema",
        default=None,
        help="Optional path to a global JSON Schema file (.json).",
    )
    val_parser.add_argument(
        "-p",
        "--prompts",
        default=None,
        help="Optional path to prompt cases file containing per-prompt schemas.",
    )
    val_parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Stdout display format: 'table' (default) or 'json'.",
    )
    val_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 if any sample fails JSON or schema validation.",
    )


def _handle_validate(args: argparse.Namespace) -> int:
    """Execute the 'validate' subcommand."""
    from llm_drift_bench._file_io import parse_source_json, read_utf8
    from llm_drift_bench._matching import require_same_ids
    from llm_drift_bench.engine import load_model_samples, load_prompt_cases
    from llm_drift_bench.validator import validate_output

    samples = load_model_samples(args.input)
    if not samples:
        raise ValueError("Validation requires at least one output sample.")
    global_schema: dict[str, Any] | None = None
    if args.schema:
        loaded = parse_source_json(read_utf8(args.schema), args.schema)
        if not isinstance(loaded, dict):
            raise ValueError(f"Global schema file must contain a JSON object: {args.schema}")
        global_schema = loaded

    prompt_map = (
        {case.prompt_id: case.schema for case in load_prompt_cases(args.prompts)}
        if args.prompts
        else {}
    )
    if args.prompts:
        if not prompt_map:
            raise ValueError("An explicitly supplied prompt suite cannot be empty.")
        require_same_ids(
            {sample.prompt_id for sample in samples}, set(prompt_map), "Prompt suite"
        )

    records: list[dict[str, Any]] = []
    passed_count = 0
    for sample in samples:
        active_schema = prompt_map.get(sample.prompt_id)
        if active_schema is None:
            active_schema = global_schema
        res = validate_output(sample.output, schema=active_schema)
        if res.passed:
            passed_count += 1
        records.append(
            {
                "prompt_id": sample.prompt_id,
                "model_name": sample.model_name,
                **res.to_dict(),
            }
        )

    total = len(samples)
    rate = (passed_count / total) if total > 0 else 1.0
    summary = {
        "total_samples": total,
        "passed_samples": passed_count,
        "conformity_rate": round(rate, 4),
        "results": records,
    }

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print("=" * 72)
        print(
            f" JSON CONFORMITY VALIDATION  [{passed_count}/{total} passed ({rate:.1%})]"
        )
        print("=" * 72)
        for item in records:
            state = "PASS" if item["passed"] else "FAIL"
            issue_str = (
                "; ".join(f"{iss['path']}: {iss['message']}" for iss in item["issues"])
                if item["issues"]
                else "Conformant"
            )
            print(f" [{state}] {item['prompt_id']:<18} | {issue_str}")
        print("=" * 72)

    if args.strict and passed_count < total:
        return 1
    return 0


def _add_check_tools_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the 'check-tools' subcommand for offline read-before-edit checks."""
    parser = subparsers.add_parser(
        "check-tools",
        help="Check that required content was read before the first edit in a normalized trace.",
        description=(
            "Offline read-before-edit check over a normalized trace (.json array or .jsonl). "
            "Exit codes: 0 PASS, 1 FAIL, 3 UNKNOWN, 2 error. UNKNOWN is never treated as PASS."
        ),
    )
    parser.add_argument("-i", "--input", required=True, help="Trace or transcript file.")
    parser.add_argument(
        "--adapter",
        choices=("normalized", "openai", "anthropic", "otel"),
        default="normalized",
        help="Input trace adapter: normalized (default), openai, anthropic, or otel.",
    )
    parser.add_argument(
        "--resource", required=True,
        help="Exact resource revision identifier to require (must match the trace).",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="First required character offset, zero-based (default: 0).",
    )
    parser.add_argument(
        "--end", type=int, required=True,
        help="Exclusive end of the required character range (e.g. the revision's length).",
    )
    parser.add_argument(
        "--trace-complete", action="store_true",
        help="Attest that the trace includes every read and edit of the run.",
    )
    parser.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Stdout format: text (default) or json (the saved check file).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write the check result JSON here, for 'run/report --tool-check'.",
    )


def _handle_check_tools(args: argparse.Namespace) -> int:
    """Execute the 'check-tools' subcommand."""
    _protect_inputs(args.output, args.input)
    if args.adapter == "normalized":
        from llm_drift_bench.tool_log import check_trace_file

        record = check_trace_file(
            args.input, resource=args.resource, required_start=args.start,
            required_end=args.end, trace_complete=args.trace_complete,
        )
    else:
        from llm_drift_bench.adapters import load_adapted_tool_events
        from llm_drift_bench.tool_log import check_trace

        events = load_adapted_tool_events(args.input, adapter=args.adapter)
        record = check_trace(
            events, resource=args.resource, required_start=args.start,
            required_end=args.end, trace_complete=args.trace_complete,
        )
    payload = json.dumps(record.to_dict(), indent=2)
    if args.output:
        _atomic_write_text(Path(args.output), payload + "\n")
    result = record.result
    if args.format == "json":
        print(payload)
    else:
        print(f"read_before_edit: {result.status}")
        print(f"Resource: {record.resource}")
        print(
            f"Required: [{record.required_start}, {record.required_end}) "
            f"({result.required_characters} characters); returned before first edit: "
            f"{result.covered_characters}"
        )
        edit = "none recorded" if result.first_edit_sequence is None else result.first_edit_sequence
        print(f"First edit sequence: {edit}")
        print(f"Trace attested complete: {'yes' if record.trace_complete else 'no'}")
        print(f"Reason: {result.reason}")
    if result.status == "PASS":
        return 0
    return 1 if result.status == "FAIL" else EXIT_TOOL_UNKNOWN


def _add_init_sample_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the 'init-sample' subcommand to scaffold starter benchmark files."""
    init_parser = subparsers.add_parser(
        "init-sample",
        help="Generate sample prompts, baseline, and candidate datasets in a directory.",
    )
    init_parser.add_argument(
        "-d",
        "--dir",
        default=".",
        help="Target directory for generated sample files (default: current directory).",
    )


def _handle_init_sample(args: argparse.Namespace) -> int:
    """Execute the 'init-sample' subcommand."""
    target_dir = Path(args.dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    prompts_data = [
        {
            "prompt_id": "extract_invoice_01",
            "category": "extraction",
            "prompt": "Extract invoice fields as JSON.",
            "schema": {
                "type": "object",
                "required": ["invoice_id", "total_usd", "status"],
                "properties": {
                    "invoice_id": {"type": "string", "minLength": 3},
                    "total_usd": {"type": "number", "minimum": 0},
                    "status": {"type": "string", "enum": ["paid", "pending", "overdue"]},
                },
                "additionalProperties": False,
            },
        },
        {
            "prompt_id": "classify_ticket_02",
            "category": "classification",
            "prompt": "Classify customer support ticket priority and tags.",
            "schema": {
                "type": "object",
                "required": ["priority", "confidence", "tags"],
                "properties": {
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "tags": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                },
            },
        },
    ]

    baseline_rows = [
        {
            "prompt_id": "extract_invoice_01",
            "model_name": "llama-3-8b-fp16",
            "latency_ms": 142.5,
            "output": '{"invoice_id": "INV-2026-09", "total_usd": 1250.00, "status": "paid"}',
        },
        {
            "prompt_id": "classify_ticket_02",
            "model_name": "llama-3-8b-fp16",
            "latency_ms": 118.0,
            "output": '{"priority": "high", "confidence": 0.94, "tags": ["billing", "urgent"]}',
        },
    ]

    candidate_rows = [
        {
            "prompt_id": "extract_invoice_01",
            "model_name": "llama-3-8b-q4_k_m",
            "latency_ms": 89.2,
            "output": '{"invoice_id": "INV-2026-09", "total_usd": 1250.00, "status": "paid"}',
        },
        {
            "prompt_id": "classify_ticket_02",
            "model_name": "llama-3-8b-q4_k_m",
            "latency_ms": 76.4,
            "output": '{"priority": "high", "confidence": 0.91, "tags": ["billing", "urgent"]}',
        },
    ]

    prompts_path = target_dir / "prompts.json"
    baseline_path = target_dir / "baseline_fp16.jsonl"
    candidate_path = target_dir / "candidate_int4.jsonl"

    outputs = {
        prompts_path: json.dumps(prompts_data, indent=2),
        baseline_path: "\n".join(json.dumps(row) for row in baseline_rows) + "\n",
        candidate_path: "\n".join(json.dumps(row) for row in candidate_rows) + "\n",
    }
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise ValueError(f"Sample file already exists: {path}. Choose a new --dir.")
    for path, content in outputs.items():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)

    print(f"Created sample benchmark suite in {target_dir.resolve()}:")
    print(f"  - {prompts_path.name}")
    print(f"  - {baseline_path.name}")
    print(f"  - {candidate_path.name}")
    return 0


def _add_comparison_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json-output", metavar="FILE",
                        help="Save comparison JSON alongside console output.")
    parser.add_argument("--html-output", metavar="FILE",
                        help="Save standalone HTML comparison dashboard with SVG Pareto chart.")
    parser.add_argument("--by-category", action="store_true",
                        help="Summarize prompt counts and mean drift per candidate and category.")
    parser.add_argument("--rule-summary", action="store_true",
                        help="Count candidate validation issues grouped by rule.")
    parser.add_argument("--slowdowns", type=int, metavar="N",
                        help="Show the N largest measured latency increases in milliseconds.")
    parser.add_argument("--numeric-changes", type=int, metavar="N",
                        help="Show the N largest nonzero numeric drift scores.")
    parser.add_argument("--inspect-prompt", action="append", metavar="ID",
                        help="Show one prompt across candidates (repeatable).")
    parser.add_argument("--hotspots", type=int, metavar="N",
                        help="Show the N prompts that drifted across the most candidates.")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit 1 for any new prompt failure relative to --against.")
    parser.add_argument("--change-limit", type=int, help="Maximum changed prompts to display.")
    parser.add_argument("--change-category", action="append", metavar="CATEGORY",
                        help="Filter changed prompts by category (repeatable, requires --against).")
    parser.add_argument("--change-type", choices=("regressed", "recovered"),
                        help="Show one direction of change (requires --against).")
    parser.add_argument("--against", metavar="LABEL", help="Show changes relative to a candidate.")
    parser.add_argument(
        "--sort-by",
        choices=("drift", "passed", "conformity", "latency"),
        default="drift",
        help="Candidate ranking order: drift (default), passed, conformity, or latency.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "rich"),
        default="text",
        help="Stdout display format: text (default), json, or rich.",
    )


def _add_many_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("compare-many", help="Evaluate multiple candidate captures.")
    parser.add_argument("-b", "--baseline", required=True, help="Common captured baseline.")
    parser.add_argument("-c", "--candidate", nargs=2, metavar=("LABEL", "FILE"), required=True,
                        action="append", help="Candidate label and capture path. Repeat per model.")
    parser.add_argument("-s", "--suite", help="Shared prompt suite and schemas.")
    _add_comparison_output_options(parser)
    parser.add_argument("--fail-on-drift", action=argparse.BooleanOptionalAction, default=True,
                        help="Exit 1 if any candidate fails (default).")
    _add_threshold_options(parser)


def _candidate_captures(
    pairs: Sequence[tuple[str, str]],
) -> Iterator[tuple[str, list[ModelSample]]]:
    from llm_drift_bench.engine import load_model_samples

    for label, path in pairs:
        try:
            samples = load_model_samples(path)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Cannot load candidate {label!r} from {path!r}: {exc}") from exc
        yield label, samples


def _handle_many(args: argparse.Namespace) -> int:
    _protect_inputs(
        args.json_output, args.baseline, args.suite, args.config, *(p for _, p in args.candidate)
    )
    _protect_inputs(
        args.html_output, args.baseline, args.suite, args.config, args.json_output,
        *(p for _, p in args.candidate),
    )
    labels = [label for label, _ in args.candidate]
    if len(labels) < 2:
        raise ValueError("Provide at least two labeled candidate captures.")
    if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise ValueError("Candidate labels must be non-empty and unique.")
    from llm_drift_bench.batch import evaluate_many
    from llm_drift_bench.engine import load_model_samples, load_prompt_cases

    thresholds, policy = _resolve_policy_and_thresholds(args)
    baseline = load_model_samples(args.baseline)
    candidates = _candidate_captures(args.candidate)
    suite = load_prompt_cases(args.suite) if args.suite else None
    reports = evaluate_many(baseline, candidates, suite, thresholds, policy=policy)
    return _render_comparison(reports, args)


def _handle_compare_reports(args: argparse.Namespace) -> int:
    _protect_inputs(args.json_output, *args.reports)
    _protect_inputs(args.html_output, args.json_output, *args.reports)
    from llm_drift_bench.report_io import load_engine_report

    reports = [load_engine_report(path) for path in args.reports]
    return _render_comparison(reports, args)


def _atomic_write_text(target: Path, content: str) -> None:
    """Create parent directories and replace target only after a complete write."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _write_comparison_json(target: Path, serialized: str) -> None:
    """Replace the target only after the complete document has been written."""
    _atomic_write_text(target, serialized + "\n")


def _render_comparison_rich(
    baseline_model: str,
    rows: Sequence[Any],
) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    table = Table(
        title=f"Candidate Comparison (Baseline: {baseline_model})",
        safe_box=True,
    )
    table.add_column("Candidate", no_wrap=True)
    table.add_column("Verdict", justify="center")
    table.add_column("Drifted", justify="right")
    table.add_column("Mean Drift", justify="right")
    table.add_column("Max Drift", justify="right")
    table.add_column("JSON", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("Worst Prompt", no_wrap=True)
    for row in rows:
        verdict = "PASS" if row.passed else "FAIL"
        latency = "n/a" if row.p95_ms is None else f"{row.p95_ms:.2f} ms"
        worst = "none" if row.max_drift == 0.0 else (row.worst_prompt_id or "n/a")
        table.add_row(
            Text(row.candidate),
            Text(verdict, style="green" if row.passed else "red"),
            Text(f"{row.drifted}/{row.samples}"),
            Text(f"{row.mean_drift:.4f}"),
            Text(f"{row.max_drift:.4f}"),
            Text(f"{row.conformity:.1%}"),
            Text(latency),
            Text(worst),
        )
    Console(width=120).print(table)


def _render_comparison(reports: Sequence[DriftBenchmarkReport], args: argparse.Namespace) -> int:
    from llm_drift_bench._breakdowns import (
        category_summary,
        latency_increases,
        numeric_changes,
        rule_counts,
    )
    from llm_drift_bench.comparison import (
        _hotspots,
        _inspect_prompts,
        compare_reports,
        compare_with_changes,
    )

    sort_by = cast("SortKey", getattr(args, "sort_by", "drift"))
    if args.against is not None:
        rows, changes = compare_with_changes(reports, args.against, sort_by=sort_by)
    else:
        rows = compare_reports(reports, sort_by=sort_by)
        changes = []
    status = int(args.fail_on_drift and any(not row.passed for row in rows))
    if args.fail_on_regression:
        if args.against is None:
            raise ValueError("--fail-on-regression requires --against.")
        if any(change.state == "regressed" for change in changes):
            status = 1
    if args.change_category:
        if args.against is None:
            raise ValueError("--change-category requires --against.")
        selected_categories = set(args.change_category)
        changes = [change for change in changes if change.category in selected_categories]
    if args.change_type:
        if args.against is None:
            raise ValueError("--change-type requires --against.")
        changes = [change for change in changes if change.state == args.change_type]
    categories = category_summary(reports) if args.by_category else []
    validation_rules = rule_counts(reports) if args.rule_summary else []
    slowdowns = latency_increases(reports, args.slowdowns) if args.slowdowns is not None else []
    numeric_rows = (numeric_changes(reports, args.numeric_changes)
                    if args.numeric_changes is not None else [])
    inspections = _inspect_prompts(reports, args.inspect_prompt) if args.inspect_prompt else []
    hotspots = _hotspots(reports, args.hotspots) if args.hotspots is not None else []
    matched = len(changes)
    if args.change_limit is not None:
        if args.against is None or args.change_limit < 1:
            raise ValueError("--change-limit requires --against and a positive integer.")
        changes = changes[:args.change_limit]
    if args.format == "json" or args.json_output is not None:
        payload: dict[str, Any] = {"baseline": reports[0].baseline_model,
                                   "candidates": [asdict(row) for row in rows]}
        if args.against is not None:
            payload.update(against=args.against, changes=[asdict(change) for change in changes],
                           changes_matched=matched, changes_omitted=matched - len(changes))
        if args.by_category:
            payload["categories"] = [asdict(item) for item in categories]
        if args.rule_summary:
            payload["validation_rules"] = [asdict(item) for item in validation_rules]
        if args.slowdowns is not None:
            payload["slowdowns"] = [asdict(item) for item in slowdowns]
        if args.numeric_changes is not None:
            payload["numeric_changes"] = [asdict(item) for item in numeric_rows]
        if args.inspect_prompt:
            payload["prompt_details"] = [asdict(detail) for detail in inspections]
        if args.hotspots is not None:
            payload["hotspots"] = [asdict(hotspot) for hotspot in hotspots]
        serialized = json.dumps(payload)
        if args.json_output is not None:
            _write_comparison_json(Path(args.json_output), serialized)
        if getattr(args, "html_output", None) is not None:
            from llm_drift_bench.reporter import export_comparison_html

            _atomic_write_text(
                Path(args.html_output), export_comparison_html(reports, sort_by=sort_by)
            )
        if args.format == "json":
            print(serialized)
            return status
    elif getattr(args, "html_output", None) is not None:
        from llm_drift_bench.reporter import export_comparison_html

        _atomic_write_text(
            Path(args.html_output), export_comparison_html(reports, sort_by=sort_by)
        )
    if args.format == "rich":
        _render_comparison_rich(reports[0].baseline_model, rows)
    else:
        print(f"Baseline {ascii(reports[0].baseline_model)}")
        for row in rows:
            verdict = "PASS" if row.passed else "FAIL"
            latency = "n/a" if row.p95_ms is None else f"{row.p95_ms:.2f} ms"
            print(f"{ascii(row.candidate)} {verdict} | drifted {row.drifted}/{row.samples} | "
                  f"mean {row.mean_drift:.4f} | max {row.max_drift:.4f} | "
                  f"JSON {row.conformity:.1%} | P95 {latency}"
                  f" | worst prompt {ascii(row.worst_prompt_id)}")
    if args.against is not None:
        print(f"Prompt changes relative to {ascii(args.against)}: {matched}")
        for change in changes:
            print(f"{ascii(change.candidate)} {ascii(change.prompt_id)} {change.state} "
                  f"({change.reference_drift:.4f} -> {change.candidate_drift:.4f})")
        if matched > len(changes):
            print(f"{matched - len(changes)} additional changes omitted.")
    if args.hotspots is not None:
        print("Prompt hotspots:")
        if not hotspots:
            print("No individual prompts drifted.")
        for hotspot in hotspots:
            print(f"  {ascii(hotspot.prompt_id)}: drifted in {hotspot.drifted_candidates}/"
                  f"{len(reports)} candidates, max drift {hotspot.max_drift:.4f}")
    for group in categories:
        print(f"Category {ascii(group.category)} | {ascii(group.candidate)} | "
              f"drifted {group.drifted}/{group.samples} | mean {group.mean_drift:.4f}")
    if args.rule_summary and not validation_rules:
        print("No recorded candidate validation issues.")
    for rule_count in validation_rules:
        print(f"Rule {ascii(rule_count.rule)} | {ascii(rule_count.candidate)} | "
              f"{rule_count.affected_prompts} prompt(s), {rule_count.occurrences} issue(s)")
    if args.slowdowns is not None and not slowdowns:
        print("No positive paired latency measurements.")
    for slowdown in slowdowns:
        print(f"Slowdown {ascii(slowdown.candidate)} {ascii(slowdown.prompt_id)}: "
              f"+{slowdown.delta_ms:.2f} ms")
    if args.numeric_changes is not None and not numeric_rows:
        print("No nonzero numeric drift measurements.")
    for numeric_row in numeric_rows:
        print(f"Numeric change {ascii(numeric_row.candidate)} {ascii(numeric_row.prompt_id)}: "
              f"{numeric_row.numeric_drift:.4f}")
    for detail in inspections:
        state = "DRIFT" if detail.drifted else "OK"
        numeric = "n/a" if detail.numeric_drift is None else f"{detail.numeric_drift:.4f}"
        latency = "n/a" if detail.latency_delta_ms is None else f"{detail.latency_delta_ms:+.2f} ms"
        print(f"{ascii(detail.prompt_id)} | {ascii(detail.candidate)} | {state} | "
              f"drift {detail.composite_drift:.4f} | numeric {numeric} | latency delta {latency}")
        for path, rule in detail.issues:
            print(f"  {ascii(path)}: {ascii(rule)}")
    return status


def _handle_diagnose(args: argparse.Namespace) -> int:
    from llm_drift_bench.diagnostics import diagnose_report
    from llm_drift_bench.report_io import load_engine_report

    report = load_engine_report(args.input)
    findings = diagnose_report(report)
    if args.scope == "suite":
        findings = [item for item in findings if item.prompt_id is None]
    elif args.scope == "prompt":
        findings = [item for item in findings if item.prompt_id is not None]
    if args.prompt_id:
        requested = set(args.prompt_id)
        unknown = requested.difference(row.prompt_id for row in report.results)
        if unknown:
            raise ValueError(f"Unknown prompt IDs: {sorted(unknown)!r}.")
        findings = [item for item in findings if item.prompt_id in requested]
    if args.kind:
        kinds = set(args.kind)
        findings = [item for item in findings if item.code in kinds]
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    displayed = findings[:args.limit]
    omitted = len(findings) - len(displayed)
    verdict = "PASSED" if report.passed else "FAILED"
    if args.device and args.ai_model is None:
        raise ValueError("--device requires --ai-model.")
    advice = None
    error = None
    if args.ai_model is not None:
        from llm_drift_bench.local_advisor import generate_advice

        try:
            advice = generate_advice(findings, args.ai_model, args.device or "CPU")
        except (ValueError, RuntimeError, OSError) as exc:
            error = str(exc)
    if args.format == "json":
        print(json.dumps({"verdict": verdict, "matched": len(findings),
                          "findings": [asdict(item) for item in displayed],
                          "omitted": omitted, "ai_advice": advice, "ai_error": error}))
        return 2 if error is not None else 0
    print(f"Benchmark {verdict}")
    if not displayed:
        print("No matching findings.")
    for item in displayed:
        print(f"[{item.code}] {item.prompt_id or 'suite'}: {item.evidence}")
        print(f"  Try: {item.suggestion}")
    if omitted:
        print(f"{omitted} additional findings omitted.")
    if advice:
        print(f"Local AI suggestion: {advice}")
    if error is not None:
        print(f"Local AI unavailable: {error}", file=sys.stderr)
        return 2
    return 0


def _add_stats_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "stats",
        help="Compute bootstrap confidence intervals, multi-run variance, and significance tests.",
    )
    parser.add_argument(
        "reports",
        nargs="+",
        metavar="REPORT",
        help="One or more saved engine reports (repeated runs of the same candidate).",
    )
    parser.add_argument(
        "--compare-against",
        metavar="REPORT",
        default=None,
        help="Optional saved engine report for a paired permutation significance test.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level between 0 and 1 (default: 0.95).",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples / permutations (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic random seed for resampling (default: 0).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Stdout display format: text (default) or json.",
    )


def _handle_stats(args: argparse.Namespace) -> int:
    from llm_drift_bench._matching import require_same_ids
    from llm_drift_bench.report_io import load_engine_report
    from llm_drift_bench.stats import (
        mcnemar_exact_test,
        paired_permutation_test,
        summarize_multi_run,
    )

    reports = [load_engine_report(path) for path in args.reports]
    summary = summarize_multi_run(
        reports,
        confidence=args.confidence,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    sig_payload: dict[str, Any] | None = None
    mcnemar_payload: dict[str, Any] | None = None
    if args.compare_against is not None:
        ref = load_engine_report(args.compare_against)
        if ref.baseline_model != summary.baseline_model:
            raise ValueError(
                "Comparison report baseline model must match the evaluated runs."
            )
        ref_map = {row.prompt_id: row for row in ref.results}
        cand_map = {item.prompt_id: item for item in summary.prompt_stability}
        require_same_ids(set(ref_map), set(cand_map), "Comparison report")
        ordered_ids = sorted(ref_map)
        sig = paired_permutation_test(
            [ref_map[pid].composite_drift_score for pid in ordered_ids],
            [cand_map[pid].mean_drift for pid in ordered_ids],
            n_permutations=args.resamples,
            seed=args.seed,
        )
        sig_payload = sig.to_dict()
        mcnemar = mcnemar_exact_test(
            [ref_map[pid].candidate_validation.passed for pid in ordered_ids],
            [cand_map[pid].conformity_rate >= 0.5 for pid in ordered_ids],
        )
        mcnemar_payload = mcnemar.to_dict()

    if args.format == "json":
        payload = summary.to_dict()
        if sig_payload is not None:
            payload["significance"] = sig_payload
        if mcnemar_payload is not None:
            payload["mcnemar_conformity"] = mcnemar_payload
        print(json.dumps(payload, indent=2))
        return 0

    ci = summary.mean_drift_ci
    conf_ci = summary.conformity_ci
    pct = int(round(ci.confidence_level * 100))
    print(f"Multi-run stability ({summary.runs} run(s), {summary.prompts} prompt(s))")
    print(f"Baseline: {summary.baseline_model} | Candidate: {summary.candidate_model}")
    print(f"Mean drift ({pct}% CI): {ci.mean:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]")
    print(f"Conformity ({pct}% CI): {conf_ci.mean:.1%} [{conf_ci.lower:.1%}, {conf_ci.upper:.1%}]")
    if summary.flaky_prompts:
        print(f"Flaky prompts ({len(summary.flaky_prompts)}): {', '.join(summary.flaky_prompts)}")
    else:
        print("Flaky prompts: none")
    if sig_payload is not None:
        verdict = "SIGNIFICANT" if sig_payload["significant"] else "not significant"
        print(
            f"Paired permutation test vs {args.compare_against}: "
            f"diff={sig_payload['observed_diff']:+.4f}, p={sig_payload['p_value']:.4f} ({verdict})"
        )
    if mcnemar_payload is not None:
        m_verdict = "SIGNIFICANT" if mcnemar_payload["significant"] else "not significant"
        print(
            f"McNemar conformity test: regressed={mcnemar_payload['b_regressed']}, "
            f"recovered={mcnemar_payload['c_recovered']}, "
            f"p={mcnemar_payload['p_value']:.4f} ({m_verdict})"
        )
    return 0


def _add_history_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "history",
        help="Record benchmark reports to a JSONL ledger and inspect drift trends.",
    )
    hist_sub = parser.add_subparsers(dest="history_action", metavar="<action>")

    rec = hist_sub.add_parser("record", help="Append a saved engine report to the drift ledger.")
    rec.add_argument("--ledger", required=True, metavar="FILE", help="Path to JSONL ledger file.")
    rec.add_argument("-i", "--input", required=True, help="Saved engine report JSON file.")
    rec.add_argument("--tag", default=None, help="Optional release or quantization tag.")
    rec.add_argument("--git-sha", default=None, help="Optional git commit SHA.")

    show = hist_sub.add_parser("show", help="Display sparklines or export trend HTML from ledger.")
    show.add_argument("--ledger", required=True, metavar="FILE", help="Path to JSONL ledger file.")
    show.add_argument("--candidate", default=None, help="Filter entries by candidate model name.")
    show.add_argument("--tag", default=None, help="Filter entries by tag.")
    show.add_argument("--limit", type=int, default=None, help="Show only the latest N runs.")
    show.add_argument(
        "--format",
        choices=("text", "json", "html"),
        default="text",
        help="Output format: text (default), json, or html.",
    )
    show.add_argument("-o", "--output", default=None, help="Optional file path to save output.")


def _handle_history(args: argparse.Namespace) -> int:
    action = getattr(args, "history_action", None)
    if action is None:
        raise ValueError("Specify a history action: 'record' or 'show'.")
    if action == "record":
        _protect_inputs(args.ledger, args.input)
        from llm_drift_bench.history import record_report
        from llm_drift_bench.report_io import load_engine_report

        report = load_engine_report(args.input)
        entry = record_report(args.ledger, report, tag=args.tag, git_sha=args.git_sha)
        print(
            f"Recorded {entry.candidate_model} ({'PASS' if entry.passed else 'FAIL'}, "
            f"mean drift {entry.mean_composite_drift:.4f}) to {args.ledger}"
        )
        return 0
    if action == "show":
        _protect_inputs(args.output, args.ledger)
        from llm_drift_bench.history import export_history_html, format_history_text, load_history

        entries = load_history(args.ledger)
        if args.candidate is not None:
            entries = [e for e in entries if e.candidate_model == args.candidate]
        if args.tag is not None:
            entries = [e for e in entries if e.tag == args.tag]
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive.")
            entries = entries[-args.limit:]
        if args.format == "json":
            rendered = json.dumps([e.to_dict() for e in entries], indent=2) + "\n"
        elif args.format == "html":
            rendered = export_history_html(entries)
        else:
            rendered = format_history_text(entries) + "\n"
        if args.output:
            _atomic_write_text(Path(args.output), rendered)
        print(rendered, end="")
        return 0
    raise ValueError(f"Unsupported history action: {action!r}.")


def _add_capture_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "capture",
        help="Capture model outputs from an OpenAI-compatible endpoint (Ollama, vLLM, OpenAI).",
    )
    parser.add_argument(
        "-s",
        "--suite",
        "-p",
        "--prompts",
        dest="suite",
        required=True,
        help="Path to prompt suite JSON/JSONL file.",
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="OpenAI-compatible base URL (e.g. http://localhost:11434/v1).",
    )
    parser.add_argument("--model", required=True, help="Model identifier to query.")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Optional environment variable name holding the Bearer API key.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional max_tokens limit.")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Bounded transient retries between 0 and 5 (default: 2).",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Request response_format={'type': 'json_object'} from the endpoint.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent requests between 1 and 32 (default: 1).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated captures (1..20; >1 writes <stem>-runN.jsonl).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSONL file path for captured ModelSamples.",
    )


def _handle_capture(args: argparse.Namespace) -> int:
    if args.runs < 1 or args.runs > 20:
        raise ValueError("--runs must be between 1 and 20.")
    _protect_inputs(args.output, args.suite)
    from llm_drift_bench.capture import CaptureConfig, capture_suite, serialize_samples_jsonl
    from llm_drift_bench.engine import load_prompt_cases

    prompts = load_prompt_cases(args.suite)
    config = CaptureConfig(
        endpoint=args.endpoint,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout_sec=args.timeout,
        max_retries=args.retries,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens,
        system_prompt=args.system,
        json_mode=args.json_mode,
        concurrency=args.concurrency,
    )
    out_base = Path(args.output)
    if args.runs == 1:
        samples = capture_suite(prompts, config)
        _atomic_write_text(out_base, serialize_samples_jsonl(samples))
        print(f"Captured {len(samples)} prompt(s) for model {config.model!r} -> {args.output}")
        return 0

    suffix = out_base.suffix or ".jsonl"
    for run_idx in range(1, args.runs + 1):
        run_path = out_base.with_name(f"{out_base.stem}-run{run_idx}{suffix}")
        _protect_inputs(str(run_path), args.suite)
        samples = capture_suite(prompts, config)
        _atomic_write_text(run_path, serialize_samples_jsonl(samples))
        print(f"Run {run_idx}/{args.runs}: captured {len(samples)} prompt(s) -> {run_path}")
    return 0


def _add_infer_schema_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "infer-schema",
        help="Infer a JSON Schema prompt suite directly from captured baseline JSON outputs.",
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more captured baseline output files (.json/.jsonl).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output path for the generated prompts.json suite (stdout if omitted).",
    )
    parser.add_argument(
        "--strict-additional",
        action="store_true",
        help="Emit additionalProperties: false on all inferred object schemas.",
    )
    parser.add_argument(
        "--shared-schema",
        action="store_true",
        help="Merge all JSON samples into one shared schema applied across JSON prompts.",
    )


def _handle_infer_schema(args: argparse.Namespace) -> int:
    from llm_drift_bench.engine import load_model_samples
    from llm_drift_bench.schema_infer import infer_prompt_suite

    _protect_inputs(args.output, *args.input)
    samples: list[ModelSample] = []
    for path in args.input:
        samples.extend(load_model_samples(path))
    suite = infer_prompt_suite(
        samples,
        strict_additional=args.strict_additional,
        shared_schema=args.shared_schema,
    )
    serialized = json.dumps(suite, indent=2) + "\n"
    if args.output:
        _atomic_write_text(Path(args.output), serialized)
        with_schema = sum(1 for item in suite if "schema" in item)
        print(
            f"Inferred {len(suite)} prompt case(s) "
            f"({with_schema} with JSON schema) -> {args.output}"
        )
    else:
        print(serialized, end="")
    return 0


def _add_calibrate_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "calibrate",
        help="Auto-calibrate drift.toml thresholds and detect volatile fields from baseline runs.",
    )
    parser.add_argument(
        "reports",
        nargs="*",
        metavar="REPORT",
        help="Optional saved engine report(s) from repeated/noise baseline runs.",
    )
    parser.add_argument(
        "--baseline-a",
        default=None,
        metavar="FILE",
        help="First baseline capture (.json/.jsonl) for self-drift & volatile field detection.",
    )
    parser.add_argument(
        "--baseline-b",
        default=None,
        metavar="FILE",
        help="Second repeat capture (.json/.jsonl) of the same baseline model.",
    )
    parser.add_argument(
        "-s",
        "--suite",
        "-p",
        "--prompts",
        dest="suite",
        default=None,
        help="Optional prompt suite file used when calibrating directly from baseline captures.",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=1.25,
        help="Safety multiplier >= 1.0 applied above observed baseline noise (default: 1.25).",
    )
    parser.add_argument(
        "--min-drift-floor",
        type=float,
        default=0.05,
        help="Minimum floor for recommended max_composite_drift in [0, 1] (default: 0.05).",
    )
    parser.add_argument(
        "--format",
        choices=("toml", "json"),
        default="toml",
        help="Output format: toml (default, ready-to-use drift.toml) or json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional file path to write the calibrated drift.toml or JSON summary.",
    )


def _handle_calibrate(args: argparse.Namespace) -> int:
    from llm_drift_bench.calibrate import (
        calibrate_from_reports,
        detect_volatile_fields,
        render_calibrated_toml,
    )
    from llm_drift_bench.engine import (
        DriftBenchmarkEngine,
        load_model_samples,
        load_prompt_cases,
    )
    from llm_drift_bench.report_io import load_engine_report

    if bool(args.baseline_a) != bool(args.baseline_b):
        raise ValueError("--baseline-a and --baseline-b must be provided together.")
    if not args.reports and not args.baseline_a:
        raise ValueError(
            "Provide at least one saved REPORT or both --baseline-a and --baseline-b."
        )
    _protect_inputs(args.output, *args.reports, args.baseline_a, args.baseline_b, args.suite)

    reports = [load_engine_report(path) for path in args.reports]
    volatile_fields: list[str] = []
    if args.baseline_a and args.baseline_b:
        samples_a = load_model_samples(args.baseline_a)
        samples_b = load_model_samples(args.baseline_b)
        volatile_fields = detect_volatile_fields(samples_a, samples_b)
        if not reports:
            prompt_cases = load_prompt_cases(args.suite) if args.suite else None
            reports = [
                DriftBenchmarkEngine().evaluate(
                    baseline_samples=samples_a,
                    candidate_samples=samples_b,
                    prompt_cases=prompt_cases,
                )
            ]

    result = calibrate_from_reports(
        reports,
        headroom=args.headroom,
        min_floor_drift=args.min_drift_floor,
        volatile_fields=volatile_fields,
    )
    if args.format == "json":
        rendered = json.dumps(result.to_dict(), indent=2) + "\n"
    else:
        rendered = render_calibrated_toml(result)

    if args.output:
        _atomic_write_text(Path(args.output), rendered)
        print(
            f"Calibrated policy from {result.runs_analyzed} run(s) "
            f"({len(result.volatile_fields)} volatile field(s)) -> {args.output}"
        )
    else:
        print(rendered, end="")
    return 0


def _add_diff_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "diff",
        help="Show semantic JSON field-level or unified text diffs between two captures.",
    )
    parser.add_argument(
        "-b", "--baseline", required=True, help="Baseline capture (.json/.jsonl)."
    )
    parser.add_argument(
        "-c", "--candidate", required=True, help="Candidate capture (.json/.jsonl)."
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help="Optional drift.toml/drift.json policy for ignore, unordered_array, and tolerances.",
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        metavar="ID",
        help="Inspect only the specified prompt_id (repeatable).",
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Hide identical prompts and show only prompts with differences.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show at most N prompt diffs.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional file path to save the rendered diff output.",
    )


def _handle_diff(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer.")
    _protect_inputs(args.output, args.baseline, args.candidate, args.config)

    from llm_drift_bench.diffing import diff_captures, format_diffs_text
    from llm_drift_bench.engine import load_model_samples
    from llm_drift_bench.policy import load_policy

    policy = load_policy(args.config) if args.config else None
    baseline = load_model_samples(args.baseline)
    candidate = load_model_samples(args.candidate)
    diffs = diff_captures(
        baseline,
        candidate,
        policy=policy,
        prompt_ids=args.prompt_id,
        only_changed=args.only_changed,
    )
    total_matched = len(diffs)
    displayed = diffs[: args.limit] if args.limit is not None else diffs
    omitted = total_matched - len(displayed)

    if args.format == "json":
        payload = {
            "total_matched": total_matched,
            "changed_prompts": sum(1 for item in diffs if item.changed),
            "omitted": omitted,
            "diffs": [item.to_dict() for item in displayed],
        }
        rendered = json.dumps(payload, indent=2) + "\n"
    else:
        rendered = format_diffs_text(displayed) + "\n"
        if omitted:
            rendered += f"{omitted} additional prompt diff(s) omitted.\n"

    if args.output:
        _atomic_write_text(Path(args.output), rendered)
    print(rendered, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level CLI ArgumentParser."""
    parser = argparse.ArgumentParser(
        prog="llm-drift-bench",
        description=(
            "CLI to compare captured LLM outputs, validate JSON "
            "conformity, and detect output drift across model updates or quantizations."
        ),
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _add_run_subparser(subparsers)
    _add_diff_subparser(subparsers)
    _add_validate_subparser(subparsers)
    _add_report_subparser(subparsers)
    diagnosis = subparsers.add_parser("diagnose", help="Explain a saved engine report.")
    diagnosis.add_argument("-i", "--input", required=True, help="Saved engine report JSON.")
    diagnosis.add_argument("--ai-model", type=Path, help="Directory of a local OpenVINO model.")
    diagnosis.add_argument("--device", choices=("CPU", "GPU", "NPU"), help="Defaults to CPU.")
    diagnosis.add_argument("--scope", choices=("all", "suite", "prompt"), default="all",
                           help="Show aggregate or per-prompt observations.")
    diagnosis.add_argument("--kind", action="append", help="Include a finding type (repeatable).",
                           choices=("schema", "invalid_json", "numeric_change", "drift",
                                    "suite", "missing_latency"))
    diagnosis.add_argument("--prompt-id", action="append", help="Focus on a prompt (repeatable).")
    diagnosis.add_argument("--limit", type=int, help="Maximum findings to display.")
    diagnosis.add_argument("--format", choices=("text", "json"), default="text")
    _add_many_subparser(subparsers)
    comparison = subparsers.add_parser("compare-reports", help="Rank saved candidate reports.")
    comparison.add_argument("--fail-on-drift", action="store_true", help="Exit 1 for any failure.")
    _add_comparison_output_options(comparison)
    comparison.add_argument("reports", nargs="+", metavar="REPORT", help="Saved engine report.")
    _add_check_tools_subparser(subparsers)
    _add_stats_subparser(subparsers)
    _add_history_subparser(subparsers)
    _add_capture_subparser(subparsers)
    _add_infer_schema_subparser(subparsers)
    _add_calibrate_subparser(subparsers)
    _add_init_sample_subparser(subparsers)
    subparsers.add_parser("ai-devices", help="List devices visible to OpenVINO.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main entrypoint for the llm-drift-bench CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "ai-devices":
            from llm_drift_bench.local_advisor import available_devices

            print("\n".join(available_devices()) or "No OpenVINO devices detected.")
            return 0
        if args.command == "compare-many":
            return _handle_many(args)
        if args.command == "compare-reports":
            return _handle_compare_reports(args)
        if args.command == "diagnose":
            return _handle_diagnose(args)
        if args.command == "run":
            return _handle_run(args)
        if args.command == "diff":
            return _handle_diff(args)
        if args.command == "report":
            return _handle_report(args)
        if args.command == "validate":
            return _handle_validate(args)
        if args.command == "check-tools":
            return _handle_check_tools(args)
        if args.command == "stats":
            return _handle_stats(args)
        if args.command == "history":
            return _handle_history(args)
        if args.command == "capture":
            return _handle_capture(args)
        if args.command == "infer-schema":
            return _handle_infer_schema(args)
        if args.command == "calibrate":
            return _handle_calibrate(args)
        if args.command == "init-sample":
            return _handle_init_sample(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
