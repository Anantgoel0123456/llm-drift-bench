"""Additional CLI subcommands (redact, matrix, lint-suite, subsample, gate, cost, snapshot)."""

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

    gate_p = subparsers.add_parser(
        "gate",
        help="Evaluate a composite GO/NO-GO gate combining drift, history, stats, and tools.",
    )
    gate_p.add_argument("-i", "--input", required=True, help="Saved engine report JSON file.")
    gate_p.add_argument(
        "--ledger",
        default=None,
        metavar="FILE",
        help="Optional history JSONL ledger for moving-window regression gating.",
    )
    gate_p.add_argument("--candidate", default=None, help="Optional ledger candidate filter.")
    gate_p.add_argument("--window", type=int, default=3, help="History window size (default: 3).")
    gate_p.add_argument(
        "--max-drift-increase",
        type=float,
        default=0.05,
        help="Max allowed mean drift increase vs history window (default: 0.05).",
    )
    gate_p.add_argument(
        "--max-conformity-drop",
        type=float,
        default=0.02,
        help="Max allowed conformity drop vs history window (default: 0.02).",
    )
    gate_p.add_argument(
        "--compare-against",
        default=None,
        metavar="REPORT",
        help="Optional reference report for paired permutation and McNemar significance tests.",
    )
    gate_p.add_argument(
        "--resamples",
        type=int,
        default=1000,
        help="Permutation resamples for --compare-against (default: 1000).",
    )
    gate_p.add_argument("--seed", type=int, default=0, help="Deterministic seed (default: 0).")
    gate_p.add_argument(
        "--tool-check",
        action="append",
        default=[],
        metavar="FILE",
        help="Saved 'check-tools' JSON file to include in the release verdict (repeatable).",
    )
    gate_p.add_argument(
        "--fail-on-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block release on moving-window history regression (default: enabled).",
    )
    gate_p.add_argument(
        "--fail-on-significant",
        action="store_true",
        help="Promote statistically significant drift/conformity degradation to blocking.",
    )
    gate_p.add_argument(
        "--require-tool-pass",
        action="store_true",
        help="Require every attached --tool-check result to be PASS (blocks on FAIL/UNKNOWN).",
    )
    gate_p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json.",
    )

    cost_p = subparsers.add_parser(
        "cost",
        help="Compare explicit generated token volume and optional USD output cost drift.",
    )
    cost_p.add_argument("-b", "--baseline", required=True, help="Baseline capture (.json/.jsonl).")
    cost_p.add_argument(
        "-c", "--candidate", required=True, help="Candidate capture (.json/.jsonl)."
    )
    cost_p.add_argument(
        "--baseline-price",
        type=float,
        default=None,
        metavar="USD_PER_1M",
        help="Optional baseline output price in USD per 1M tokens.",
    )
    cost_p.add_argument(
        "--candidate-price",
        type=float,
        default=None,
        metavar="USD_PER_1M",
        help="Optional candidate output price in USD per 1M tokens.",
    )
    cost_p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json.",
    )

    snap_p = subparsers.add_parser(
        "snapshot",
        help="Create or verify offline SHA-256 integrity manifests for golden baseline fixtures.",
    )
    snap_sub = snap_p.add_subparsers(dest="snapshot_action", metavar="<action>")

    snap_create = snap_sub.add_parser(
        "create", help="Create a SHA-256 manifest pinning baseline capture, suite, and policy."
    )
    snap_create.add_argument("-b", "--baseline", required=True, help="Baseline capture file.")
    snap_create.add_argument(
        "-s", "--suite", "-p", "--prompts", dest="suite", default=None, help="Prompt suite file."
    )
    snap_create.add_argument("--config", default=None, help="Optional drift.toml/drift.json file.")
    snap_create.add_argument(
        "--base-dir", default=None, help="Optional root directory for relative manifest paths."
    )
    snap_create.add_argument(
        "-o", "--output", required=True, help="Output manifest JSON file path."
    )
    snap_create.add_argument(
        "--format", choices=("text", "json"), default="text", help="Stdout format: text or json."
    )

    snap_verify = snap_sub.add_parser(
        "verify", help="Verify every file hash and baseline ID digest in a snapshot manifest."
    )
    snap_verify.add_argument("-i", "--input", required=True, help="Snapshot manifest JSON file.")
    snap_verify.add_argument(
        "--base-dir", default=None, help="Optional root directory overriding manifest parent dir."
    )
    snap_verify.add_argument(
        "--format", choices=("text", "json"), default="text", help="Stdout format: text or json."
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


def _handle_gate(args: argparse.Namespace) -> int:
    from llm_drift_bench._matching import require_same_ids
    from llm_drift_bench.gates import evaluate_release_gates
    from llm_drift_bench.report_io import load_engine_report

    report = load_engine_report(args.input)
    history_check = None
    if args.ledger is not None:
        from llm_drift_bench.history import detect_history_regression, load_history

        entries = load_history(args.ledger)
        if args.candidate is not None:
            entries = [e for e in entries if e.candidate_model == args.candidate]
        history_check = detect_history_regression(
            entries,
            window=args.window,
            max_drift_increase=args.max_drift_increase,
            max_conformity_drop=args.max_conformity_drop,
        )

    sig = None
    mcnemar = None
    if args.compare_against is not None:
        from llm_drift_bench.stats import mcnemar_exact_test, paired_permutation_test

        ref = load_engine_report(args.compare_against)
        if ref.baseline_model != report.baseline_model:
            raise ValueError("Reference report baseline model must match the evaluated report.")
        ref_map = {r.prompt_id: r for r in ref.results}
        cand_map = {r.prompt_id: r for r in report.results}
        require_same_ids(set(ref_map), set(cand_map), "Reference report")
        ordered = sorted(ref_map)
        sig = paired_permutation_test(
            [ref_map[pid].composite_drift_score for pid in ordered],
            [cand_map[pid].composite_drift_score for pid in ordered],
            n_permutations=args.resamples,
            seed=args.seed,
        )
        mcnemar = mcnemar_exact_test(
            [ref_map[pid].candidate_validation.passed for pid in ordered],
            [cand_map[pid].candidate_validation.passed for pid in ordered],
        )

    tool_checks = []
    if args.tool_check:
        from llm_drift_bench.tool_log import load_tool_check

        tool_checks = [load_tool_check(path) for path in args.tool_check]

    verdict = evaluate_release_gates(
        report,
        history_check=history_check,
        significance=sig,
        mcnemar=mcnemar,
        tool_checks=tool_checks,
        fail_on_history_regression=args.fail_on_history,
        fail_on_significant_degradation=args.fail_on_significant,
        require_tool_checks_pass=args.require_tool_pass,
    )
    if args.format == "json":
        print(json.dumps(verdict.to_dict(), indent=2))
    else:
        state = "GO (APPROVED)" if verdict.approved else "NO-GO (BLOCKED)"
        print(f"Release Gate Verdict: {state}")
        for check in verdict.checks:
            tag = "PASS" if check.passed else ("BLOCK" if check.blocking else "WARN")
            print(f"  [{tag:<5}] {check.name:<24} | {check.detail}")
    return 0 if verdict.approved else 1


def _handle_cost(args: argparse.Namespace) -> int:
    from llm_drift_bench.cost import ModelPricing, estimate_cost_drift
    from llm_drift_bench.engine import load_model_samples

    base = load_model_samples(args.baseline)
    cand = load_model_samples(args.candidate)
    b_price = (
        ModelPricing(output_per_million_usd=args.baseline_price)
        if args.baseline_price is not None
        else None
    )
    c_price = (
        ModelPricing(output_per_million_usd=args.candidate_price)
        if args.candidate_price is not None
        else None
    )
    res = estimate_cost_drift(base, cand, baseline_pricing=b_price, candidate_pricing=c_price)
    if args.format == "json":
        print(json.dumps(res.to_dict(), indent=2))
        return 0
    if not res.complete:
        print(
            f"Token & Cost Drift ({res.total_samples} sample(s)): "
            "n/a (incomplete explicit tokens_generated measurements)."
        )
        return 0
    pct_s = "n/a" if res.token_delta_pct is None else f"{res.token_delta_pct:+.2f}%"
    print(f"Token & Cost Drift ({res.total_samples} sample(s))")
    print(
        f"  Output tokens : baseline={res.baseline_tokens}, "
        f"candidate={res.candidate_tokens} (delta={res.token_delta:+d}, {pct_s})"
    )
    if res.baseline_output_cost_usd is not None or res.candidate_output_cost_usd is not None:
        b_s = (
            "n/a"
            if res.baseline_output_cost_usd is None
            else f"${res.baseline_output_cost_usd:.6f}"
        )
        c_s = (
            "n/a"
            if res.candidate_output_cost_usd is None
            else f"${res.candidate_output_cost_usd:.6f}"
        )
        d_s = "n/a" if res.cost_delta_usd is None else f"${res.cost_delta_usd:+.6f}"
        print(f"  Output cost   : baseline={b_s}, candidate={c_s} (delta={d_s})")
    return 0


def _handle_snapshot(args: argparse.Namespace, protect_inputs: ProtectFn) -> int:
    action = getattr(args, "snapshot_action", None)
    if action is None:
        raise ValueError("Specify a snapshot action: 'create' or 'verify'.")
    if action == "create":
        protect_inputs(args.output, args.baseline, args.suite, args.config)
        from llm_drift_bench.snapshot import create_snapshot_manifest

        manifest = create_snapshot_manifest(
            args.baseline,
            suite_path=args.suite,
            config_path=args.config,
            base_dir=args.base_dir,
            output_path=args.output,
        )
        if args.format == "json":
            print(manifest.to_json(indent=2), end="")
        else:
            print(
                f"Created snapshot manifest ({len(manifest.artifacts)} artifact(s), "
                f"{manifest.sample_count} sample(s)) -> {args.output}"
            )
        return 0
    if action == "verify":
        from llm_drift_bench.snapshot import (
            format_snapshot_verification_text,
            verify_snapshot_manifest,
        )

        verification = verify_snapshot_manifest(args.input, base_dir=args.base_dir)
        if args.format == "json":
            print(json.dumps(verification.to_dict(), indent=2))
        else:
            print(format_snapshot_verification_text(verification))
        return 0 if verification.valid else 1
    raise ValueError(f"Unsupported snapshot action: {action!r}.")


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
    if args.command == "gate":
        return _handle_gate(args)
    if args.command == "cost":
        return _handle_cost(args)
    if args.command == "snapshot":
        return _handle_snapshot(args, protect_inputs)
    return None
