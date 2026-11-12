"""Command-line interface for llm-drift-bench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

from llm_drift_bench import __version__
from llm_drift_bench.engine import (
    DriftBenchmarkEngine,
    format_report_table,
    load_model_samples,
    load_prompt_cases,
)
from llm_drift_bench.models import DriftThresholds
from llm_drift_bench.validator import validate_output


def _add_run_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the 'run' subcommand and its CLI flags."""
    run_parser = subparsers.add_parser(
        "run",
        help="Run a drift benchmark comparing baseline and candidate LLM outputs.",
    )
    run_parser.add_argument(
        "-b",
        "--baseline",
        required=True,
        help="Path to baseline model outputs (.json or .jsonl).",
    )
    run_parser.add_argument(
        "-c",
        "--candidate",
        required=True,
        help="Path to candidate model outputs (.json or .jsonl).",
    )
    run_parser.add_argument(
        "-p",
        "--prompts",
        default=None,
        help="Optional path to prompt cases with expected JSON schemas (.json or .jsonl).",
    )
    run_parser.add_argument(
        "--max-drift",
        type=float,
        default=0.25,
        help="Maximum allowed composite drift score (default: 0.25).",
    )
    run_parser.add_argument(
        "--min-conformity",
        type=float,
        default=0.95,
        help="Minimum required candidate JSON schema conformity rate (default: 0.95).",
    )
    run_parser.add_argument(
        "--max-numeric-drift",
        type=float,
        default=0.15,
        help="Maximum allowed relative drift on numeric JSON fields (default: 0.15).",
    )
    run_parser.add_argument(
        "--max-latency-regression",
        type=float,
        default=50.0,
        help="Maximum allowed P95 latency regression percentage (default: 50.0).",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional file path to write the full JSON benchmark report.",
    )
    run_parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Stdout display format: 'table' (default) or 'json'.",
    )
    run_parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Return exit code 1 if benchmark thresholds fail or any sample drifts.",
    )


def _handle_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    baseline_samples = load_model_samples(args.baseline)
    candidate_samples = load_model_samples(args.candidate)
    prompt_cases = load_prompt_cases(args.prompts) if args.prompts else None

    thresholds = DriftThresholds(
        max_composite_drift=args.max_drift,
        min_json_conformity_rate=args.min_conformity,
        max_numeric_relative_drift=args.max_numeric_drift,
        max_latency_regression_pct=args.max_latency_regression,
    )
    engine = DriftBenchmarkEngine(thresholds=thresholds)
    report = engine.evaluate(
        baseline_samples=baseline_samples,
        candidate_samples=candidate_samples,
        prompt_cases=prompt_cases,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_json(indent=2), encoding="utf-8")

    if args.format == "json":
        print(report.to_json(indent=2))
    else:
        print(format_report_table(report))

    if args.fail_on_drift and (not report.passed or report.drifted_samples > 0):
        return 1
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
    global_schema: dict[str, Any] | None = None
    if args.schema:
        loaded = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Global schema file must contain a JSON object.")
        global_schema = loaded

    prompt_map = (
        {case.prompt_id: case.schema for case in load_prompt_cases(args.prompts)}
        if args.prompts
        else {}
    )

    records: list[dict[str, Any]] = []
    passed_count = 0
    for sample in samples:
        active_schema = prompt_map.get(sample.prompt_id) or global_schema
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

    prompts_path.write_text(json.dumps(prompts_data, indent=2), encoding="utf-8")
    baseline_path.write_text(
        "\n".join(json.dumps(r) for r in baseline_rows) + "\n", encoding="utf-8"
    )
    candidate_path.write_text(
        "\n".join(json.dumps(r) for r in candidate_rows) + "\n", encoding="utf-8"
    )

    print(f"Created sample benchmark suite in {target_dir.resolve()}:")
    print(f"  - {prompts_path.name}")
    print(f"  - {baseline_path.name}")
    print(f"  - {candidate_path.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level CLI ArgumentParser."""
    parser = argparse.ArgumentParser(
        prog="llm-drift-bench",
        description=(
            "Lightweight, zero-bloat CLI to benchmark LLM outputs, validate JSON "
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
    _add_validate_subparser(subparsers)
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
        if args.command == "run":
            return _handle_run(args)
        if args.command == "validate":
            return _handle_validate(args)
        if args.command == "init-sample":
            return _handle_init_sample(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
