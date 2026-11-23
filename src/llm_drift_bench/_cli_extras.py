"""Additional CLI subcommands (redact, matrix, lint-suite, subsample) with lazy imports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from llm_drift_bench.matrix import MatrixSort
    from llm_drift_bench.models import PromptCase

ProtectFn = Callable[..., None]
AtomicWriteFn = Callable[[Path, str], None]


def register_extra_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register redact, matrix, lint-suite, and subsample subcommands."""
    redact_p = subparsers.add_parser(
        "redact",
        help="Deterministically redact API keys, private keys, and PII from a capture file.",
    )
    redact_p.add_argument("-i", "--input", required=True, help="Input capture (.json/.jsonl).")
    redact_p.add_argument("-o", "--output", required=True, help="Output redacted JSONL file.")
    redact_p.add_argument(
        "--secrets-only",
        action="store_true",
        help="Redact credentials/keys only and skip PII (emails, SSNs, IPv4, credit cards).",
    )
    redact_p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Summary output format: text (default) or json.",
    )

    matrix_p = subparsers.add_parser(
        "matrix",
        help="Render a Prompt x Candidate drift heatmap matrix from 2+ saved engine reports.",
    )
    matrix_p.add_argument(
        "reports",
        nargs="+",
        metavar="REPORT",
        help="Two or more saved engine report JSON files sharing the same baseline and prompts.",
    )
    matrix_p.add_argument(
        "--sort-prompts",
        choices=("max_drift", "prompt_id"),
        default="max_drift",
        help="Row order: max_drift (default) or prompt_id.",
    )
    matrix_p.add_argument(
        "--format",
        choices=("text", "json", "html"),
        default="text",
        help="Output format: text (default), json, or html.",
    )
    matrix_p.add_argument(
        "-o", "--output", default=None, help="Optional output file path to save the matrix."
    )

    lint_p = subparsers.add_parser(
        "lint-suite",
        help="Audit a prompt suite for unsupported schemas, duplicate prompts, and hygiene gaps.",
    )
    lint_p.add_argument(
        "-s",
        "--suite",
        "-p",
        "--prompts",
        dest="suite",
        required=True,
        help="Path to prompt suite JSON/JSONL file.",
    )
    lint_p.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 on warnings as well as errors.",
    )
    lint_p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json.",
    )

    sub_p = subparsers.add_parser(
        "subsample",
        help="Deterministically subsample a prompt suite (and optionally slice matching captures).",
    )
    sub_p.add_argument(
        "-s",
        "--suite",
        "-p",
        "--prompts",
        dest="suite",
        required=True,
        help="Path to prompt suite JSON/JSONL file.",
    )
    sub_p.add_argument(
        "--size",
        type=int,
        required=True,
        help="Number of prompts to draw into the subsample.",
    )
    sub_p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic random seed (default: 0).",
    )
    sub_p.add_argument(
        "--no-stratify",
        action="store_true",
        help="Use uniform random sampling instead of category stratification.",
    )
    sub_p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output path for the sampled prompt suite JSON (stdout if omitted).",
    )
    sub_p.add_argument(
        "--slice-capture",
        nargs=2,
        metavar=("IN", "OUT"),
        action="append",
        default=[],
        help="Slice a capture file in lockstep with the sampled suite (repeatable).",
    )


def _serialize_prompt_case(case: PromptCase) -> dict[str, Any]:
    record: dict[str, Any] = {
        "prompt_id": case.prompt_id,
        "category": case.category,
        "prompt": case.prompt,
    }
    if case.schema is not None:
        record["schema"] = case.schema
    if case.metadata:
        record["metadata"] = dict(case.metadata)
    return record


def _handle_redact(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(args.output, args.input)
    from llm_drift_bench.capture import serialize_samples_jsonl
    from llm_drift_bench.engine import load_model_samples
    from llm_drift_bench.redaction import redact_samples

    samples = load_model_samples(args.input)
    cleaned, summary = redact_samples(samples, include_pii=not args.secrets_only)
    atomic_write_text(Path(args.output), serialize_samples_jsonl(cleaned))
    if args.format == "json":
        payload = summary.to_dict()
        payload["output"] = str(args.output)
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Redacted {summary.total_replacements} occurrence(s) across "
            f"{summary.redacted_samples}/{summary.total_samples} sample(s) -> {args.output}"
        )
        if summary.counts_by_kind:
            details = ", ".join(f"{k}={v}" for k, v in summary.counts_by_kind.items())
            print(f"  Breakdown: {details}")
    return 0


def _handle_matrix(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(args.output, *args.reports)
    from llm_drift_bench.matrix import (
        build_drift_matrix,
        export_drift_matrix_html,
        format_drift_matrix_text,
    )
    from llm_drift_bench.report_io import load_engine_report

    reports = [load_engine_report(path) for path in args.reports]
    sort_by = cast("MatrixSort", args.sort_prompts)
    matrix = build_drift_matrix(reports, sort_prompts_by=sort_by)
    if args.format == "json":
        rendered = json.dumps(matrix.to_dict(), indent=2) + "\n"
    elif args.format == "html":
        rendered = export_drift_matrix_html(matrix)
    else:
        rendered = format_drift_matrix_text(matrix) + "\n"
    if args.output:
        atomic_write_text(Path(args.output), rendered)
    print(rendered, end="")
    return 0


def _handle_lint_suite(args: argparse.Namespace) -> int:
    from llm_drift_bench.engine import load_prompt_cases
    from llm_drift_bench.prompt_lint import format_lint_issues_text, lint_prompt_suite

    cases = load_prompt_cases(args.suite)
    issues = lint_prompt_suite(cases)
    errors = sum(1 for item in issues if item.severity == "error")
    warnings = sum(1 for item in issues if item.severity == "warning")
    if args.format == "json":
        payload = {
            "total_cases": len(cases),
            "errors": errors,
            "warnings": warnings,
            "issues": [item.to_dict() for item in issues],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_lint_issues_text(issues))
    if errors > 0 or (args.strict and warnings > 0):
        return 1
    return 0


def _handle_subsample(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    slice_pairs: list[tuple[str, str]] = [
        (src, dst) for src, dst in getattr(args, "slice_capture", [])
    ]
    input_paths = [args.suite, *(src for src, _ in slice_pairs)]
    protect_inputs(args.output, *input_paths)
    seen_outputs: set[Path] = set()
    if args.output:
        seen_outputs.add(Path(args.output).resolve())
    for _, dst in slice_pairs:
        protect_inputs(dst, *input_paths)
        resolved = Path(dst).resolve()
        if resolved in seen_outputs:
            raise ValueError(f"Duplicate output destination in subsample: {dst!r}.")
        seen_outputs.add(resolved)

    from llm_drift_bench.capture import serialize_samples_jsonl
    from llm_drift_bench.engine import load_model_samples, load_prompt_cases
    from llm_drift_bench.sampling import sample_prompt_suite, slice_samples_for_suite

    cases = load_prompt_cases(args.suite)
    sampled = sample_prompt_suite(
        cases,
        size=args.size,
        seed=args.seed,
        stratify_by_category=not args.no_stratify,
    )
    serialized = json.dumps([_serialize_prompt_case(c) for c in sampled], indent=2) + "\n"
    if args.output:
        atomic_write_text(Path(args.output), serialized)
    for src, dst in slice_pairs:
        samples = load_model_samples(src)
        sliced = slice_samples_for_suite(samples, sampled)
        atomic_write_text(Path(dst), serialize_samples_jsonl(sliced))
    if args.output:
        extra = f" and sliced {len(slice_pairs)} capture(s)" if slice_pairs else ""
        print(f"Sampled {len(sampled)}/{len(cases)} prompt(s) -> {args.output}{extra}")
    else:
        print(serialized, end="")
    return 0


def dispatch_extra_command(
    args: argparse.Namespace,
    *,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int | None:
    """Dispatch extra subcommands; return None when command is not owned by this module."""
    if args.command == "redact":
        return _handle_redact(args, protect_inputs, atomic_write_text)
    if args.command == "matrix":
        return _handle_matrix(args, protect_inputs, atomic_write_text)
    if args.command == "lint-suite":
        return _handle_lint_suite(args)
    if args.command == "subsample":
        return _handle_subsample(args, protect_inputs, atomic_write_text)
    return None
