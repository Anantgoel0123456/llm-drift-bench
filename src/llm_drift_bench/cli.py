"""Command-line interface for llm-drift-bench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from llm_drift_bench import __version__
from llm_drift_bench.engine import (
    DriftBenchmarkEngine,
    format_report_table,
    load_model_samples,
    load_prompt_cases,
)
from llm_drift_bench.models import DriftThresholds


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
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
