"""Command-line interface for llm-drift-bench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_drift_bench import __version__
from llm_drift_bench._file_io import parse_json, read_utf8
from llm_drift_bench._matching import require_same_ids
from llm_drift_bench.batch import evaluate_many
from llm_drift_bench.comparison import compare_reports, prompt_changes
from llm_drift_bench.diagnostics import diagnose_report
from llm_drift_bench.engine import (
    DriftBenchmarkEngine,
    format_report_table,
    load_model_samples,
    load_prompt_cases,
)
from llm_drift_bench.junit import export_junit_report
from llm_drift_bench.local_advisor import available_devices, generate_advice
from llm_drift_bench.models import DriftBenchmarkReport, DriftThresholds, ModelSample
from llm_drift_bench.report_io import load_engine_report, load_report
from llm_drift_bench.reporter import (
    export_html_report,
    export_markdown_report,
    render_console_report,
)
from llm_drift_bench.tool_log import (
    ToolCheckRecord,
    check_trace_file,
    load_tool_check,
)
from llm_drift_bench.validator import validate_output

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
    if paths and output_format not in ("markdown", "html"):
        raise ValueError("--tool-check requires markdown or html output.")
    return [load_tool_check(path) for path in paths]


def _add_threshold_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-drift",
        type=float,
        default=0.25,
        help="Maximum allowed composite drift score (default: 0.25).",
    )
    parser.add_argument(
        "--min-conformity",
        type=float,
        default=0.95,
        help="Minimum required candidate JSON schema conformity rate (default: 0.95).",
    )
    parser.add_argument(
        "--max-numeric-drift",
        type=float,
        default=0.15,
        help="Maximum allowed relative drift on numeric JSON fields (default: 0.15).",
    )
    parser.add_argument(
        "--max-latency-regression",
        type=float,
        default=50.0,
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
        choices=("table", "rich", "json", "markdown", "html", "junit"),
        default="table",
        help="Stdout format: table (default), rich, json, markdown, html, or junit.",
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


def _thresholds(args: argparse.Namespace) -> DriftThresholds:
    return DriftThresholds(
        max_composite_drift=args.max_drift,
        min_json_conformity_rate=args.min_conformity,
        max_numeric_relative_drift=args.max_numeric_drift,
        max_latency_regression_pct=args.max_latency_regression,
    )


def _handle_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    for label, target in (("baseline", args.baseline), ("candidate", args.candidate)):
        if not Path(target).is_file():
            raise ValueError(
                f"--{label} must name an existing outputs file: {target!r}. "
                "Model identifiers are not supported; supply captured JSON/JSONL outputs."
            )

    _protect_inputs(args.output, args.baseline, args.candidate, args.suite, *args.tool_check)
    if args.junit_output:
        _protect_inputs(args.junit_output, args.baseline, args.candidate, args.suite,
                        args.output, *args.tool_check)
        if args.output and Path(args.output).resolve() == Path(args.junit_output).resolve():
            raise ValueError("--output and --junit-output require different files.")
    summary_path = None
    if args.github_summary:
        destination = os.environ.get("GITHUB_STEP_SUMMARY")
        if not destination or not Path(destination).is_file():
            raise ValueError("--github-summary requires an existing GITHUB_STEP_SUMMARY file.")
        _protect_inputs(destination, args.baseline, args.candidate, args.suite,
                        args.output, args.junit_output, *args.tool_check)
        summary_path = Path(destination)
    baseline_samples = load_model_samples(args.baseline)
    candidate_samples = load_model_samples(args.candidate)
    prompt_cases = load_prompt_cases(args.suite) if args.suite else None
    tool_checks = _load_tool_checks(args.tool_check, args.format)

    engine = DriftBenchmarkEngine(thresholds=_thresholds(args))
    report = engine.evaluate(
        baseline_samples=baseline_samples,
        candidate_samples=candidate_samples,
        prompt_cases=prompt_cases,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_json(indent=2), encoding="utf-8")

    if args.junit_output:
        junit_path = Path(args.junit_output)
        junit_path.parent.mkdir(parents=True, exist_ok=True)
        junit_path.write_text(export_junit_report(report), encoding="utf-8")

    if args.format == "junit":
        print(export_junit_report(report), end="")
    elif args.format == "json":
        print(report.to_json(indent=2))
    elif args.format == "markdown":
        print(export_markdown_report(report, tool_checks=tool_checks), end="")
    elif args.format == "html":
        print(export_html_report(report, tool_checks=tool_checks), end="")
    elif args.format == "rich":
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
        choices=("markdown", "console", "table", "json", "html", "junit"),
        default="markdown",
        help="Output format: markdown (default), console/table, json, html, or junit.",
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
    _protect_inputs(args.output, args.input, *args.tool_check)
    result = load_report(args.input)
    if args.format == "junit":
        if not isinstance(result, DriftBenchmarkReport):
            raise ValueError("JUnit export requires a canonical engine report.")
        xml = export_junit_report(result)
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(xml, encoding="utf-8")
        print(xml, end="")
    elif args.format == "markdown":
        md = export_markdown_report(result, output_path=args.output, tool_checks=tool_checks)
        print(md, end="")
    elif args.format == "json":
        payload = (result.to_json(indent=2) if isinstance(result, DriftBenchmarkReport)
                   else result.model_dump_json(indent=2))
        if args.output:
            out_file = Path(args.output)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(payload, encoding="utf-8")
        print(payload)
    elif args.format == "html":
        print(
            export_html_report(result, output_path=args.output, tool_checks=tool_checks), end=""
        )
    else:
        rendered = render_console_report(result)
        if args.output:
            out_file = Path(args.output)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(rendered, encoding="utf-8")
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
    samples = load_model_samples(args.input)
    if not samples:
        raise ValueError("Validation requires at least one output sample.")
    global_schema: dict[str, Any] | None = None
    if args.schema:
        loaded = parse_json(read_utf8(args.schema))
        if not isinstance(loaded, dict):
            raise ValueError("Global schema file must contain a JSON object.")
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
    parser.add_argument("-i", "--input", required=True, help="Normalized trace file.")
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
    record = check_trace_file(
        args.input, resource=args.resource, required_start=args.start,
        required_end=args.end, trace_complete=args.trace_complete,
    )
    payload = json.dumps(record.to_dict(), indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
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
    parser.add_argument("--change-limit", type=int, help="Maximum changed prompts to display.")
    parser.add_argument("--change-category", action="append", metavar="CATEGORY",
                        help="Filter changed prompts by category (repeatable, requires --against).")
    parser.add_argument("--change-type", choices=("regressed", "recovered"),
                        help="Show one direction of change (requires --against).")
    parser.add_argument("--against", metavar="LABEL", help="Show prompt flips relative to a candidate.")
    parser.add_argument("--format", choices=("text", "json"), default="text")


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


def _candidate_captures(pairs: Sequence[tuple[str, str]]) -> Iterator[tuple[str, list[ModelSample]]]:
    for label, path in pairs:
        try:
            samples = load_model_samples(path)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Cannot load candidate {label!r} from {path!r}: {exc}") from exc
        yield label, samples


def _handle_many(args: argparse.Namespace) -> int:
    labels = [label for label, _ in args.candidate]
    if len(labels) < 2:
        raise ValueError("Provide at least two labeled candidate captures.")
    if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise ValueError("Candidate labels must be non-empty and unique.")
    baseline = load_model_samples(args.baseline)
    candidates = _candidate_captures(args.candidate)
    suite = load_prompt_cases(args.suite) if args.suite else None
    reports = evaluate_many(baseline, candidates, suite, _thresholds(args))
    return _render_comparison(reports, args)


def _handle_compare_reports(args: argparse.Namespace) -> int:
    reports = [load_engine_report(path) for path in args.reports]
    return _render_comparison(reports, args)


def _render_comparison(reports: Sequence[DriftBenchmarkReport], args: argparse.Namespace) -> int:
    rows = compare_reports(reports)
    status = int(args.fail_on_drift and any(not row.passed for row in rows))
    changes = prompt_changes(reports, args.against) if args.against is not None else []
    if args.change_category:
        if args.against is None:
            raise ValueError("--change-category requires --against.")
        categories = set(args.change_category)
        changes = [change for change in changes if change.category in categories]
    if args.change_type:
        if args.against is None:
            raise ValueError("--change-type requires --against.")
        changes = [change for change in changes if change.state == args.change_type]
    matched = len(changes)
    if args.change_limit is not None:
        if args.against is None or args.change_limit < 1:
            raise ValueError("--change-limit requires --against and a positive integer.")
        changes = changes[:args.change_limit]
    if args.format == "json":
        payload: dict[str, Any] = {"baseline": reports[0].baseline_model,
                                   "candidates": [asdict(row) for row in rows]}
        if args.against is not None:
            payload.update(against=args.against, changes=[asdict(change) for change in changes],
                           changes_matched=matched, changes_omitted=matched - len(changes))
        print(json.dumps(payload))
        return status
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
    return status


def _handle_diagnose(args: argparse.Namespace) -> int:
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
    subparsers.add_parser("ai-devices", help="List devices visible to OpenVINO.")
    comparison = subparsers.add_parser("compare-reports", help="Rank saved candidate reports.")
    comparison.add_argument("--fail-on-drift", action="store_true", help="Exit 1 for any failure.")
    _add_comparison_output_options(comparison)
    comparison.add_argument("reports", nargs="+", metavar="REPORT", help="Saved engine report.")
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
    _add_run_subparser(subparsers)
    _add_report_subparser(subparsers)
    _add_validate_subparser(subparsers)
    _add_check_tools_subparser(subparsers)
    _add_init_sample_subparser(subparsers)
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
        if args.command == "report":
            return _handle_report(args)
        if args.command == "validate":
            return _handle_validate(args)
        if args.command == "check-tools":
            return _handle_check_tools(args)
        if args.command == "init-sample":
            return _handle_init_sample(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
