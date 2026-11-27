"""Additional CLI subcommands (redact, matrix, lint-suite, distribution, wizard, logprobs, ...)."""

from __future__ import annotations

import argparse
import json
import math
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
        "--conversations",
        action="store_true",
        help="Treat input as multi-turn ConversationRecord JSON/JSONL instead of ModelSamples.",
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
        "--persist-windows",
        type=int,
        default=1,
        help="Consecutive shifted runs required before history regression blocks (default: 1).",
    )
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
        "--baseline-decisions",
        default=None,
        metavar="FILE",
        help="Optional baseline downstream consumer decisions file (.json/.jsonl).",
    )
    gate_p.add_argument(
        "--candidate-decisions",
        default=None,
        metavar="FILE",
        help="Optional candidate downstream consumer decisions file (.json/.jsonl).",
    )
    gate_p.add_argument(
        "--max-rejection-rate",
        type=float,
        default=None,
        help="Optional max allowed candidate consumer rejection rate in [0, 1].",
    )
    gate_p.add_argument(
        "--max-rejection-increase",
        type=float,
        default=None,
        help="Optional max allowed consumer rejection rate increase vs baseline in [0, 1].",
    )
    gate_p.add_argument(
        "--fail-on-consumer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block release when attached consumer rejection check fails (default: enabled).",
    )
    gate_p.add_argument(
        "--baseline-capture",
        default=None,
        metavar="FILE",
        help="Optional baseline capture (.json/.jsonl) for segmented distribution blast-radius.",
    )
    gate_p.add_argument(
        "--candidate-capture",
        default=None,
        metavar="FILE",
        help="Optional candidate capture (.json/.jsonl) for segmented distribution blast-radius.",
    )
    gate_p.add_argument(
        "-s",
        "--suite",
        default=None,
        help="Optional prompt suite JSON/JSONL used with --group-by on gate.",
    )
    gate_p.add_argument(
        "--config",
        default=None,
        help="Optional drift.toml/drift.json policy for ignored fields in segmented distribution.",
    )
    gate_p.add_argument(
        "--group-by",
        default=None,
        metavar="SPEC",
        help="Segment key ('category' or 'metadata:<key>') for distribution blast-radius check.",
    )
    gate_p.add_argument(
        "--shift-threshold",
        type=float,
        default=0.10,
        help="Per-segment TVD/W1 threshold for localized vs widespread scope (default: 0.10).",
    )
    gate_p.add_argument(
        "--fail-on-widespread",
        action="store_true",
        help="Block release when segmented distribution shift_scope is 'widespread'.",
    )
    gate_p.add_argument(
        "--min-embedding-similarity",
        type=float,
        default=None,
        help=(
            "Optional min mean embedding cosine similarity in [-1, 1] over "
            "--baseline-capture/--candidate-capture metadata embeddings."
        ),
    )
    gate_p.add_argument(
        "--fail-on-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block release when attached embedding similarity check fails (default: enabled).",
    )
    gate_p.add_argument(
        "--baseline-conversations",
        default=None,
        metavar="FILE",
        help="Optional baseline multi-turn conversations file (.json/.jsonl).",
    )
    gate_p.add_argument(
        "--candidate-conversations",
        default=None,
        metavar="FILE",
        help="Optional candidate multi-turn conversations file (.json/.jsonl).",
    )
    gate_p.add_argument(
        "--max-turn-drift",
        type=float,
        default=0.20,
        help="Per-turn drift threshold for attached conversation checks (default: 0.20).",
    )
    gate_p.add_argument(
        "--max-mean-trajectory-drift",
        type=float,
        default=0.25,
        help="Max mean trajectory drift for attached conversation checks (default: 0.25).",
    )
    gate_p.add_argument(
        "--fail-on-conversations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block release when attached conversation trajectory check fails (default: enabled).",
    )
    gate_p.add_argument(
        "--baseline-trace",
        default=None,
        metavar="FILE",
        help="Optional baseline agent tool trace file for tool-usage drift gating.",
    )
    gate_p.add_argument(
        "--candidate-trace",
        default=None,
        metavar="FILE",
        help="Optional candidate agent tool trace file for tool-usage drift gating.",
    )
    gate_p.add_argument(
        "--trace-adapter",
        choices=("normalized", "openai", "anthropic", "gemini", "otel"),
        default="normalized",
        help="Adapter for --baseline-trace/--candidate-trace (default: normalized).",
    )
    gate_p.add_argument(
        "--max-tool-call-increase",
        type=int,
        default=None,
        help="Optional max allowed tool call increase vs baseline trace.",
    )
    gate_p.add_argument(
        "--max-tool-failed-increase",
        type=int,
        default=None,
        help="Optional max allowed failed tool call increase vs baseline trace.",
    )
    gate_p.add_argument(
        "--fail-on-new-edit-resources",
        action="store_true",
        help="Fail tool-trace check when candidate edits resources not edited in baseline.",
    )
    gate_p.add_argument(
        "--fail-on-tool-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block release when attached tool-trace drift check fails (default: enabled).",
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
        "--max-token-increase-pct",
        type=float,
        default=None,
        metavar="PCT",
        help="Optional output token increase budget in percent (exits 1 if exceeded).",
    )
    cost_p.add_argument(
        "--max-cost-increase-pct",
        type=float,
        default=None,
        metavar="PCT",
        help="Optional USD output cost increase budget in percent (exits 1 if exceeded).",
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
        "--conversations",
        default=None,
        help="Optional baseline multi-turn conversations file (.json/.jsonl) to pin.",
    )
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

    imp_p = subparsers.add_parser(
        "import-captures",
        help="Convert OpenAI Batch, Anthropic Batch, or CSV exports into ModelSample JSONL.",
    )
    imp_p.add_argument(
        "--from",
        dest="source_format",
        choices=("openai-batch", "anthropic-batch", "csv"),
        required=True,
        help="Source format: openai-batch, anthropic-batch, or csv.",
    )
    imp_p.add_argument("-i", "--input", required=True, help="Source batch JSONL or CSV file.")
    imp_p.add_argument("-o", "--output", required=True, help="Output ModelSample JSONL file.")
    imp_p.add_argument(
        "--model",
        default=None,
        help="Optional model_name override (for CSV, overrides default when column is blank).",
    )
    imp_p.add_argument(
        "--format", choices=("text", "json"), default="text", help="Summary format: text or json."
    )

    badge_p = subparsers.add_parser(
        "badge",
        help="Generate an offline shields-style static SVG status badge from a saved report.",
    )
    badge_p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Saved engine, audit, or conversation report JSON file.",
    )
    badge_p.add_argument(
        "-o", "--output", default=None, help="Optional output .svg path (stdout if omitted)."
    )
    badge_p.add_argument(
        "--label",
        default=None,
        help="Optional left-side badge label (defaults by report type).",
    )

    dist_p = subparsers.add_parser(
        "distribution",
        help="Measure cross-prompt JSON field distribution drift (TVD, JSD, Wasserstein).",
    )
    dist_p.add_argument("-b", "--baseline", required=True, help="Baseline capture (.json/.jsonl).")
    dist_p.add_argument(
        "-c", "--candidate", required=True, help="Candidate capture (.json/.jsonl)."
    )
    dist_p.add_argument(
        "--config", default=None, help="Optional drift.toml/drift.json policy for ignored fields."
    )
    dist_p.add_argument(
        "-s",
        "--suite",
        "-p",
        "--prompts",
        dest="suite",
        default=None,
        help="Optional prompt suite JSON/JSONL (used with --group-by category/metadata:<key>).",
    )
    dist_p.add_argument(
        "--group-by",
        default=None,
        metavar="SPEC",
        help="Segment distribution before aggregating: 'category' or 'metadata:<key>'.",
    )
    dist_p.add_argument(
        "--shift-threshold",
        type=float,
        default=0.10,
        help="Per-segment TVD/W1 threshold used to classify localized vs widespread scope (0.10).",
    )
    dist_p.add_argument(
        "--fail-on-widespread",
        action="store_true",
        help="Exit 1 when segmented shift_scope is 'widespread' (requires --group-by).",
    )
    dist_p.add_argument(
        "--no-collapse-arrays",
        action="store_true",
        help="Keep concrete array indices ([0], [1]) separate instead of collapsing into [*].",
    )
    dist_p.add_argument(
        "--max-tvd",
        type=float,
        default=None,
        help="Optional max categorical Total Variation Distance in [0, 1] (exits 1 if exceeded).",
    )
    dist_p.add_argument(
        "--max-jsd",
        type=float,
        default=None,
        help="Optional max categorical Jensen-Shannon Divergence in [0, 1] (exits 1 if exceeded).",
    )
    dist_p.add_argument(
        "--max-wasserstein",
        type=float,
        default=None,
        help="Optional max normalized 1D Wasserstein shift in [0, 1] (exits 1 if exceeded).",
    )
    dist_p.add_argument(
        "--format",
        choices=("text", "json", "markdown", "html"),
        default="text",
        help="Output format: text (default), json, markdown, or html.",
    )
    dist_p.add_argument(
        "-o", "--output", default=None, help="Optional output file path to write the report."
    )

    wiz_p = subparsers.add_parser(
        "wizard",
        help="Guided plain-English setup wizard or preset generator for drift.toml.",
    )
    wiz_p.add_argument(
        "--preset",
        choices=("extraction", "classification", "coding", "rag", "strict"),
        default=None,
        help="Generate drift.toml directly from a use-case preset without interactive prompts.",
    )
    wiz_p.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive questions (defaults to --preset extraction when omitted).",
    )
    wiz_p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output path (e.g. drift.toml); prints TOML to stdout if omitted.",
    )

    lp_p = subparsers.add_parser(
        "logprobs",
        help="Compare optional top_logprobs token entropy and top-k JSD across captures.",
    )
    lp_p.add_argument("-b", "--baseline", required=True, help="Baseline capture (.json/.jsonl).")
    lp_p.add_argument(
        "-c", "--candidate", required=True, help="Candidate capture (.json/.jsonl)."
    )
    lp_p.add_argument(
        "--max-entropy-delta",
        type=float,
        default=None,
        help="Optional max allowed absolute mean entropy shift in bits (exits 1 if exceeded).",
    )
    lp_p.add_argument(
        "--max-jsd",
        type=float,
        default=None,
        help="Optional max allowed prompt top-k JSD in [0, 1] (exits 1 if exceeded).",
    )
    lp_p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json.",
    )

    qual_p = subparsers.add_parser(
        "check-quality",
        help="Detect repetition loops, refusal flips, and length anomalies across captures.",
    )
    qual_p.add_argument("-b", "--baseline", required=True, help="Baseline capture (.json/.jsonl).")
    qual_p.add_argument(
        "-c", "--candidate", required=True, help="Candidate capture (.json/.jsonl)."
    )
    qual_p.add_argument(
        "--max-length-ratio",
        type=float,
        default=3.0,
        help="Flag candidate outputs longer than this multiple of baseline (default: 3.0).",
    )
    qual_p.add_argument(
        "--min-length-ratio",
        type=float,
        default=0.25,
        help="Flag candidate outputs shorter than this fraction of baseline (default: 0.25).",
    )
    qual_p.add_argument(
        "--min-baseline-chars",
        type=int,
        default=40,
        help="Minimum baseline character count before length-ratio checks apply (default: 40).",
    )
    qual_p.add_argument(
        "--fail-on-issues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit 1 when any quality regression is detected (default: enabled).",
    )
    qual_p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json.",
    )

    fmt_p = subparsers.add_parser(
        "check-formats",
        help="Validate semantic JSON field formats (date, date-time, email, uri, uuid, ipv4, ...).",
    )
    fmt_p.add_argument("-i", "--input", required=True, help="Capture file (.json/.jsonl).")
    fmt_p.add_argument(
        "--rule",
        action="append",
        required=True,
        metavar="JSONPATH=FORMAT",
        help="Format rule such as '$.email=email' or '$.items[*].date=date' (repeatable).",
    )
    fmt_p.add_argument(
        "--allow-null",
        action="store_true",
        help="Skip format validation when a matched field value is JSON null.",
    )
    fmt_p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format: text (default) or json.",
    )

    audit_p = subparsers.add_parser(
        "audit",
        help="Run a unified offline audit (drift, distributions, quality, logprobs, and cost).",
    )
    audit_p.add_argument("-b", "--baseline", required=True, help="Baseline capture (.json/.jsonl).")
    audit_p.add_argument(
        "-c", "--candidate", required=True, help="Candidate capture (.json/.jsonl)."
    )
    audit_p.add_argument(
        "-s",
        "--suite",
        "-p",
        "--prompts",
        dest="suite",
        default=None,
        help="Optional prompt suite JSON/JSONL file.",
    )
    audit_p.add_argument("--config", default=None, help="Optional drift.toml/drift.json policy.")
    audit_p.add_argument("--max-tvd", type=float, default=None, help="Optional max TVD in [0, 1].")
    audit_p.add_argument("--max-jsd", type=float, default=None, help="Optional max JSD in [0, 1].")
    audit_p.add_argument(
        "--max-wasserstein",
        type=float,
        default=None,
        help="Optional max normalized 1D Wasserstein in [0, 1].",
    )
    audit_p.add_argument(
        "--group-by",
        default=None,
        metavar="SPEC",
        help="Optional segmented distribution key ('category' or 'metadata:<key>').",
    )
    audit_p.add_argument(
        "--shift-threshold",
        type=float,
        default=0.10,
        help="Per-segment TVD/W1 threshold for localized vs widespread scope (default: 0.10).",
    )
    audit_p.add_argument(
        "--fail-on-widespread",
        action="store_true",
        help="Fail audit when segmented shift_scope is 'widespread' (requires --group-by).",
    )
    audit_p.add_argument(
        "--max-entropy-delta",
        type=float,
        default=None,
        help="Optional max absolute mean logprobs entropy delta in bits.",
    )
    audit_p.add_argument(
        "--max-top-k-jsd",
        type=float,
        default=None,
        help="Optional max prompt top-k logprobs JSD in [0, 1].",
    )
    audit_p.add_argument(
        "--fail-on-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail audit when quality regressions are detected (default: enabled).",
    )
    audit_p.add_argument(
        "--baseline-decisions",
        default=None,
        metavar="FILE",
        help="Optional baseline downstream consumer decision log (.json/.jsonl).",
    )
    audit_p.add_argument(
        "--candidate-decisions",
        default=None,
        metavar="FILE",
        help="Optional candidate downstream consumer decision log (.json/.jsonl).",
    )
    audit_p.add_argument(
        "--validator-module",
        default=None,
        metavar="MOD:FUNC",
        help="Optional Python consumer validator callable ('pkg.mod:fn') run on captures.",
    )
    audit_p.add_argument(
        "--max-rejection-rate",
        type=float,
        default=None,
        help="Optional max allowed candidate consumer rejection rate in [0, 1].",
    )
    audit_p.add_argument(
        "--max-rejection-increase",
        type=float,
        default=None,
        help="Optional max allowed consumer rejection rate increase vs baseline in [0, 1].",
    )
    audit_p.add_argument(
        "--min-embedding-similarity",
        type=float,
        default=None,
        help="Optional min mean embedding cosine similarity in [-1, 1] on capture embeddings.",
    )
    audit_p.add_argument(
        "--baseline-conversations",
        default=None,
        metavar="FILE",
        help="Optional baseline multi-turn conversations file (.json/.jsonl).",
    )
    audit_p.add_argument(
        "--candidate-conversations",
        default=None,
        metavar="FILE",
        help="Optional candidate multi-turn conversations file (.json/.jsonl).",
    )
    audit_p.add_argument(
        "--max-turn-drift",
        type=float,
        default=0.20,
        help="Per-turn drift threshold for attached conversation checks (default: 0.20).",
    )
    audit_p.add_argument(
        "--max-mean-trajectory-drift",
        type=float,
        default=0.25,
        help="Max mean trajectory drift for attached conversation checks (default: 0.25).",
    )
    audit_p.add_argument(
        "--baseline-trace",
        default=None,
        metavar="FILE",
        help="Optional baseline agent tool-trace file for tool-usage drift audit.",
    )
    audit_p.add_argument(
        "--candidate-trace",
        default=None,
        metavar="FILE",
        help="Optional candidate agent tool-trace file for tool-usage drift audit.",
    )
    audit_p.add_argument(
        "--trace-adapter",
        choices=("normalized", "openai", "anthropic", "gemini", "otel"),
        default="normalized",
        help="Format adapter for attached tool traces (default: normalized).",
    )
    audit_p.add_argument(
        "--max-tool-call-increase",
        type=int,
        default=None,
        help="Optional max allowed increase in total tool invocations vs baseline trace.",
    )
    audit_p.add_argument(
        "--max-tool-failed-increase",
        type=int,
        default=None,
        help="Optional max allowed increase in failed tool invocations vs baseline trace.",
    )
    audit_p.add_argument(
        "--fail-on-new-edit-resources",
        action="store_true",
        help="Fail tool-trace audit when candidate edits resources not edited in baseline.",
    )
    audit_p.add_argument(
        "--format",
        choices=("text", "json", "markdown", "html"),
        default="text",
        help="Output format: text (default), json, markdown, or html.",
    )
    audit_p.add_argument(
        "-o", "--output", default=None, help="Optional output path to save the rendered audit."
    )

    cons_p = subparsers.add_parser(
        "check-consumer",
        help="Compare downstream consumer rejection rates across baseline and candidate.",
    )
    cons_p.add_argument(
        "--baseline-decisions",
        default=None,
        metavar="FILE",
        help="Baseline downstream accept/reject decision log (.json/.jsonl).",
    )
    cons_p.add_argument(
        "--candidate-decisions",
        default=None,
        metavar="FILE",
        help="Candidate downstream accept/reject decision log (.json/.jsonl).",
    )
    cons_p.add_argument("-b", "--baseline", default=None, help="Baseline capture (.json/.jsonl).")
    cons_p.add_argument(
        "-c", "--candidate", default=None, help="Candidate capture (.json/.jsonl)."
    )
    cons_p.add_argument(
        "--validator-module",
        default=None,
        metavar="MOD:FUNC",
        help="Python validator callable ('pkg.mod:fn') returning bool or (bool, reason).",
    )
    cons_p.add_argument(
        "--max-rejection-rate",
        type=float,
        default=None,
        help="Optional max allowed candidate rejection rate in [0, 1] (exits 1 if exceeded).",
    )
    cons_p.add_argument(
        "--max-rejection-increase",
        type=float,
        default=None,
        help="Optional max allowed rejection rate increase vs baseline in [0, 1].",
    )
    cons_p.add_argument(
        "--format",
        choices=("text", "json", "markdown", "html"),
        default="text",
        help="Output format: text (default), json, markdown, or html.",
    )
    cons_p.add_argument(
        "-o", "--output", default=None, help="Optional output file path to write the report."
    )

    roll_p = subparsers.add_parser(
        "rolling-distribution",
        help="Compare a candidate capture window against a rolling pool of prior baseline windows.",
    )
    roll_p.add_argument(
        "-w",
        "--window",
        action="append",
        required=True,
        metavar="FILE",
        help="Prior baseline capture file (.json/.jsonl); repeat for multi-window rolling pool.",
    )
    roll_p.add_argument(
        "-c", "--candidate", required=True, help="Candidate capture window (.json/.jsonl)."
    )
    roll_p.add_argument(
        "-s",
        "--suite",
        "-p",
        "--prompts",
        dest="suite",
        default=None,
        help="Optional prompt suite JSON/JSONL for category/metadata segment lookups.",
    )
    roll_p.add_argument(
        "--config", default=None, help="Optional drift.toml/drift.json policy for ignored fields."
    )
    roll_p.add_argument(
        "--group-by",
        default=None,
        metavar="SPEC",
        help="Optional segment key: 'category' or 'metadata:<key>'.",
    )
    roll_p.add_argument(
        "--shift-threshold",
        type=float,
        default=0.10,
        help="Per-segment TVD/W1 threshold for localized vs widespread scope (default: 0.10).",
    )
    roll_p.add_argument(
        "--fail-on-widespread",
        action="store_true",
        help="Exit 1 when segmented rolling shift_scope is 'widespread' (requires --group-by).",
    )
    roll_p.add_argument(
        "--no-collapse-arrays",
        action="store_true",
        help="Keep concrete array indices ([0], [1]) separate instead of collapsing into [*].",
    )
    roll_p.add_argument(
        "--max-tvd", type=float, default=None, help="Optional max TVD in [0, 1] (exits 1)."
    )
    roll_p.add_argument(
        "--max-jsd", type=float, default=None, help="Optional max JSD in [0, 1] (exits 1)."
    )
    roll_p.add_argument(
        "--max-wasserstein",
        type=float,
        default=None,
        help="Optional max normalized 1D Wasserstein in [0, 1] (exits 1).",
    )
    roll_p.add_argument(
        "--max-jitter-multiplier",
        type=float,
        default=None,
        help="Optional max multiple of natural baseline window-to-window jitter (exits 1).",
    )
    roll_p.add_argument(
        "--format",
        choices=("text", "json", "markdown", "html"),
        default="text",
        help="Output format: text (default), json, markdown, or html.",
    )
    roll_p.add_argument(
        "-o", "--output", default=None, help="Optional output file path to write the report."
    )

    emb_p = subparsers.add_parser(
        "check-embeddings",
        help="Compare semantic embedding cosine similarity offline or via an opt-in endpoint.",
    )
    emb_p.add_argument("-b", "--baseline", required=True, help="Baseline capture (.json/.jsonl).")
    emb_p.add_argument(
        "-c", "--candidate", required=True, help="Candidate capture (.json/.jsonl)."
    )
    emb_p.add_argument(
        "--endpoint",
        default=None,
        help="Optional OpenAI-compatible embeddings base URL (omit for offline metadata mode).",
    )
    emb_p.add_argument(
        "--model",
        default=None,
        help="Embedding model name (required when --endpoint is set).",
    )
    emb_p.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding optional Bearer token (default: OPENAI_API_KEY).",
    )
    emb_p.add_argument(
        "--timeout", type=float, default=30.0, help="HTTP timeout in seconds (default: 30)."
    )
    emb_p.add_argument(
        "--batch-size", type=int, default=32, help="Max texts per embeddings batch (default: 32)."
    )
    emb_p.add_argument(
        "--min-mean-similarity",
        type=float,
        default=None,
        help="Optional minimum mean cosine similarity in [-1, 1] (exits 1 if below).",
    )
    emb_p.add_argument(
        "--min-prompt-similarity",
        type=float,
        default=None,
        help="Optional minimum per-prompt cosine similarity in [-1, 1] (exits 1 if below).",
    )
    emb_p.add_argument(
        "--format",
        choices=("text", "json", "markdown", "html"),
        default="text",
        help="Output format: text (default), json, markdown, or html.",
    )
    emb_p.add_argument(
        "-o", "--output", default=None, help="Optional output file path to write the report."
    )

    conv_p = subparsers.add_parser(
        "check-conversations",
        help="Compare multi-turn assistant conversation trajectories turn-by-turn.",
    )
    conv_p.add_argument(
        "-b", "--baseline", required=True, help="Baseline conversations file (.json/.jsonl)."
    )
    conv_p.add_argument(
        "-c", "--candidate", required=True, help="Candidate conversations file (.json/.jsonl)."
    )
    conv_p.add_argument(
        "--max-turn-drift",
        type=float,
        default=0.25,
        help="Max allowed composite drift per assistant turn in [0, 1] (default: 0.25).",
    )
    conv_p.add_argument(
        "--max-mean-drift",
        type=float,
        default=0.25,
        help="Max allowed suite mean trajectory drift in [0, 1] (default: 0.25).",
    )
    conv_p.add_argument(
        "--max-final-turn-drift",
        type=float,
        default=None,
        help="Optional max allowed drift on the final assistant turn in [0, 1].",
    )
    conv_p.add_argument(
        "--max-turn-count-delta",
        type=int,
        default=None,
        help="Optional max allowed absolute difference in assistant turn counts.",
    )
    conv_p.add_argument(
        "--fail-on-refusal-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when any baseline answer flips into a refusal in candidate (default: enabled).",
    )
    conv_p.add_argument(
        "--long-text-mode",
        choices=("error", "skip_levenshtein"),
        default="error",
        help="How to handle turns > 16,384 chars: error (default) or skip_levenshtein.",
    )
    conv_p.add_argument(
        "--show-diffs",
        action="store_true",
        help="Include per-turn semantic JSON/text diffs for changed turns (text/json formats).",
    )
    conv_p.add_argument(
        "--diagnose",
        action="store_true",
        help="Include actionable root-cause diagnostics (text/json formats).",
    )
    conv_p.add_argument(
        "--format",
        choices=("text", "json", "markdown", "html", "sarif", "junit", "csv"),
        default="text",
        help="Output format: text (default), json, markdown, html, sarif, junit, or csv.",
    )
    conv_p.add_argument(
        "-o", "--output", default=None, help="Optional output file path to write the report."
    )

    imp_conv_p = subparsers.add_parser(
        "import-conversations",
        help="Convert OpenAI chat, Anthropic messages, or ShareGPT JSONL into conversations JSONL.",
    )
    imp_conv_p.add_argument(
        "--from",
        dest="source_format",
        choices=("openai-chat", "anthropic-messages", "sharegpt"),
        required=True,
        help="Source transcript format: openai-chat, anthropic-messages, or sharegpt.",
    )
    imp_conv_p.add_argument("-i", "--input", required=True, help="Input transcript JSONL file.")
    imp_conv_p.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output canonical ConversationRecord JSONL file.",
    )
    imp_conv_p.add_argument(
        "--model",
        default=None,
        help="Optional model_name override for all imported conversations.",
    )

    diff_tools_p = subparsers.add_parser(
        "diff-tools",
        help="Compare baseline vs candidate agent tool traces for call, retry, and edit drift.",
    )
    diff_tools_p.add_argument("-b", "--baseline", required=True, help="Baseline trace file.")
    diff_tools_p.add_argument("-c", "--candidate", required=True, help="Candidate trace file.")
    diff_tools_p.add_argument(
        "--adapter",
        choices=("normalized", "openai", "anthropic", "gemini", "otel"),
        default="normalized",
        help="Trace format adapter: normalized (default), openai, anthropic, gemini, or otel.",
    )
    diff_tools_p.add_argument(
        "--max-call-increase",
        type=int,
        default=None,
        help="Optional max allowed increase in total tool invocations.",
    )
    diff_tools_p.add_argument(
        "--max-failed-increase",
        type=int,
        default=None,
        help="Optional max allowed increase in failed tool invocations.",
    )
    diff_tools_p.add_argument(
        "--fail-on-new-edit-resources",
        action="store_true",
        help="Exit 1 if candidate edits any resource not edited in baseline.",
    )
    diff_tools_p.add_argument(
        "--format",
        choices=("text", "json", "markdown", "html", "sarif", "junit"),
        default="text",
        help="Output format: text (default), json, markdown, html, sarif, or junit.",
    )
    diff_tools_p.add_argument(
        "-o", "--output", default=None, help="Optional output file path to write the report."
    )

    cap_conv_p = subparsers.add_parser(
        "capture-conversations",
        help="Replay multi-turn conversation templates against an endpoint and save JSONL.",
    )
    cap_conv_p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Template conversations file (.json/.jsonl) containing scripted user turns.",
    )
    cap_conv_p.add_argument("--endpoint", required=True, help="Endpoint base URL.")
    cap_conv_p.add_argument("--model", required=True, help="Model identifier to query.")
    cap_conv_p.add_argument(
        "--provider",
        choices=("openai", "anthropic", "gemini"),
        default="openai",
        help="Wire protocol: openai (default), anthropic, or gemini.",
    )
    cap_conv_p.add_argument("--api-key-env", default=None, help="Env var holding API key.")
    cap_conv_p.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)."
    )
    cap_conv_p.add_argument("--seed", type=int, default=None, help="Optional deterministic seed.")
    cap_conv_p.add_argument("--max-tokens", type=int, default=None, help="Optional max_tokens.")
    cap_conv_p.add_argument("--system", default=None, help="Optional default system prompt.")
    cap_conv_p.add_argument(
        "--timeout", type=float, default=60.0, help="Per-turn timeout in seconds (default: 60.0)."
    )
    cap_conv_p.add_argument(
        "--retries", type=int, default=2, help="Bounded transient retries 0..5 (default: 2)."
    )
    cap_conv_p.add_argument(
        "--json-mode", action="store_true", help="Request structured JSON output mode."
    )
    cap_conv_p.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="KEY:VALUE",
        help="Optional custom HTTP header (repeatable).",
    )
    cap_conv_p.add_argument(
        "-o", "--output", required=True, help="Output canonical ConversationRecord JSONL path."
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
    if args.conversations:
        from llm_drift_bench.batch_import import serialize_conversations_jsonl
        from llm_drift_bench.conversation import load_conversations
        from llm_drift_bench.redaction import redact_conversations

        convs = load_conversations(args.input)
        cleaned_convs, summary = redact_conversations(convs, include_pii=not args.secrets_only)
        atomic_write_text(Path(args.output), serialize_conversations_jsonl(cleaned_convs))
        noun = "conversation(s)"
    else:
        from llm_drift_bench.capture import serialize_samples_jsonl
        from llm_drift_bench.engine import load_model_samples
        from llm_drift_bench.redaction import redact_samples

        samples = load_model_samples(args.input)
        cleaned, summary = redact_samples(samples, include_pii=not args.secrets_only)
        atomic_write_text(Path(args.output), serialize_samples_jsonl(cleaned))
        noun = "sample(s)"
    if args.format == "json":
        payload = summary.to_dict()
        payload["output"] = str(args.output)
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Redacted {summary.total_replacements} occurrence(s) across "
            f"{summary.redacted_samples}/{summary.total_samples} {noun} -> {args.output}"
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
            persist_windows=args.persist_windows,
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

    consumer_comp = None
    if args.baseline_decisions is not None or args.candidate_decisions is not None:
        if not (args.baseline_decisions and args.candidate_decisions):
            raise ValueError(
                "Both --baseline-decisions and --candidate-decisions are required."
            )
        from llm_drift_bench.consumer import (
            compare_consumer_decisions,
            load_consumer_decisions,
        )

        consumer_comp = compare_consumer_decisions(
            load_consumer_decisions(args.baseline_decisions),
            load_consumer_decisions(args.candidate_decisions),
            max_rejection_rate=args.max_rejection_rate,
            max_rejection_increase=args.max_rejection_increase,
        )

    using_emb = args.min_embedding_similarity is not None
    segmented_dist = None
    using_dist = (
        args.group_by is not None
        or args.fail_on_widespread
        or (
            (args.baseline_capture is not None or args.candidate_capture is not None)
            and not using_emb
        )
    )
    loaded_base_samples = None
    loaded_cand_samples = None
    if using_dist or using_emb:
        if using_dist and not (args.baseline_capture and args.candidate_capture and args.group_by):
            raise ValueError(
                "Segmented gate check requires --baseline-capture, --candidate-capture, "
                "and --group-by."
            )
        if using_emb and not (args.baseline_capture and args.candidate_capture):
            raise ValueError(
                "Embedding gate check requires --baseline-capture and --candidate-capture."
            )
        from llm_drift_bench.engine import load_model_samples

        loaded_base_samples = load_model_samples(args.baseline_capture)
        loaded_cand_samples = load_model_samples(args.candidate_capture)

    if using_dist:
        from llm_drift_bench.distribution import analyze_segmented_distributions
        from llm_drift_bench.engine import load_prompt_cases
        from llm_drift_bench.policy import load_policy

        assert loaded_base_samples is not None and loaded_cand_samples is not None
        segmented_dist = analyze_segmented_distributions(
            loaded_base_samples,
            loaded_cand_samples,
            group_by=args.group_by,
            prompt_cases=load_prompt_cases(args.suite) if args.suite else None,
            policy=load_policy(args.config) if args.config else None,
            shift_threshold=args.shift_threshold,
        )

    embedding_rep = None
    if using_emb:
        from llm_drift_bench.embeddings import analyze_capture_embeddings

        assert loaded_base_samples is not None and loaded_cand_samples is not None
        embedding_rep = analyze_capture_embeddings(
            loaded_base_samples,
            loaded_cand_samples,
            min_mean_similarity=args.min_embedding_similarity,
        )

    conversation_rep = None
    if args.baseline_conversations is not None or args.candidate_conversations is not None:
        if not (args.baseline_conversations and args.candidate_conversations):
            raise ValueError(
                "Both --baseline-conversations and --candidate-conversations are required."
            )
        from llm_drift_bench.conversation import (
            analyze_conversation_drift,
            load_conversations,
        )

        conversation_rep = analyze_conversation_drift(
            load_conversations(args.baseline_conversations),
            load_conversations(args.candidate_conversations),
            max_turn_drift=args.max_turn_drift,
            max_mean_drift=args.max_mean_trajectory_drift,
        )

    trace_diff = None
    if args.baseline_trace is not None or args.candidate_trace is not None:
        if not (args.baseline_trace and args.candidate_trace):
            raise ValueError("Both --baseline-trace and --candidate-trace are required.")
        from llm_drift_bench.adapters import load_adapted_tool_events
        from llm_drift_bench.tool_events import compare_tool_traces
        from llm_drift_bench.tool_log import load_tool_events

        if args.trace_adapter == "normalized":
            b_ev = load_tool_events(args.baseline_trace)
            c_ev = load_tool_events(args.candidate_trace)
        else:
            b_ev = load_adapted_tool_events(args.baseline_trace, adapter=args.trace_adapter)
            c_ev = load_adapted_tool_events(args.candidate_trace, adapter=args.trace_adapter)
        trace_diff = compare_tool_traces(
            b_ev,
            c_ev,
            max_call_increase=args.max_tool_call_increase,
            max_failed_increase=args.max_tool_failed_increase,
            fail_on_new_edit_resources=args.fail_on_new_edit_resources,
        )

    verdict = evaluate_release_gates(
        report,
        history_check=history_check,
        significance=sig,
        mcnemar=mcnemar,
        tool_checks=tool_checks,
        segmented_distribution=segmented_dist,
        consumer_comparison=consumer_comp,
        embedding_report=embedding_rep,
        conversation_report=conversation_rep,
        tool_trace_diff=trace_diff,
        fail_on_history_regression=args.fail_on_history,
        fail_on_significant_degradation=args.fail_on_significant,
        require_tool_checks_pass=args.require_tool_pass,
        fail_on_widespread_shift=args.fail_on_widespread,
        fail_on_consumer_rejection=args.fail_on_consumer,
        fail_on_embedding_drift=args.fail_on_embeddings,
        fail_on_conversation_drift=args.fail_on_conversations,
        fail_on_tool_trace_drift=args.fail_on_tool_trace,
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
    from llm_drift_bench.cost import ModelPricing, check_cost_budget, estimate_cost_drift
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
    budget_ok = True
    budget_detail: str | None = None
    if args.max_token_increase_pct is not None or args.max_cost_increase_pct is not None:
        budget_ok, budget_detail = check_cost_budget(
            res,
            max_token_increase_pct=args.max_token_increase_pct,
            max_cost_increase_pct=args.max_cost_increase_pct,
        )
    if args.format == "json":
        payload = res.to_dict()
        if budget_detail is not None:
            payload["budget_passed"] = budget_ok
            payload["budget_detail"] = budget_detail
        print(json.dumps(payload, indent=2))
        return 0 if budget_ok else 1
    if not res.complete:
        print(
            f"Token & Cost Drift ({res.total_samples} sample(s)): "
            "n/a (incomplete explicit tokens_generated measurements)."
        )
        if budget_detail is not None:
            print(f"  Budget check  : {'PASS' if budget_ok else 'FAIL'} ({budget_detail})")
        return 0 if budget_ok else 1
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
    if budget_detail is not None:
        print(f"  Budget check  : {'PASS' if budget_ok else 'FAIL'} ({budget_detail})")
    return 0 if budget_ok else 1


def _handle_snapshot(args: argparse.Namespace, protect_inputs: ProtectFn) -> int:
    action = getattr(args, "snapshot_action", None)
    if action is None:
        raise ValueError("Specify a snapshot action: 'create' or 'verify'.")
    if action == "create":
        protect_inputs(args.output, args.baseline, args.suite, args.config, args.conversations)
        from llm_drift_bench.snapshot import create_snapshot_manifest

        manifest = create_snapshot_manifest(
            args.baseline,
            suite_path=args.suite,
            config_path=args.config,
            conversations_path=args.conversations,
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


def _handle_import_captures(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(args.output, args.input)
    from llm_drift_bench.batch_import import (
        import_anthropic_batch_jsonl,
        import_csv_captures,
        import_openai_batch_jsonl,
    )
    from llm_drift_bench.capture import serialize_samples_jsonl

    if args.source_format == "openai-batch":
        samples = import_openai_batch_jsonl(args.input, model_name=args.model)
    elif args.source_format == "anthropic-batch":
        samples = import_anthropic_batch_jsonl(args.input, model_name=args.model)
    else:
        samples = import_csv_captures(
            args.input,
            default_model=args.model or "unknown",
        )
    atomic_write_text(Path(args.output), serialize_samples_jsonl(samples))
    if args.format == "json":
        print(
            json.dumps(
                {
                    "source_format": args.source_format,
                    "imported_samples": len(samples),
                    "model_names": sorted({s.model_name for s in samples}),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
    else:
        print(
            f"Imported {len(samples)} sample(s) from {args.source_format} -> {args.output}"
        )
    return 0


def _handle_badge(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(args.output, args.input)
    from llm_drift_bench.badge import export_badge_from_json

    svg = export_badge_from_json(args.input, label=args.label)
    if args.output:
        atomic_write_text(Path(args.output), svg)
        print(f"Wrote drift status badge -> {args.output}")
    else:
        print(svg, end="")
    return 0


def _validate_unit_threshold(flag: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError(f"{flag} must be a finite number between 0.0 and 1.0.")


def _handle_distribution(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    _validate_unit_threshold("--max-tvd", args.max_tvd)
    _validate_unit_threshold("--max-jsd", args.max_jsd)
    _validate_unit_threshold("--max-wasserstein", args.max_wasserstein)
    if args.fail_on_widespread and not args.group_by:
        raise ValueError("--fail-on-widespread requires --group-by.")
    protect_inputs(args.output, args.baseline, args.candidate, args.config, args.suite)

    from llm_drift_bench.distribution import (
        analyze_field_distributions,
        analyze_segmented_distributions,
        export_distribution_html,
        export_segmented_distribution_html,
        format_distribution_markdown,
        format_distribution_text,
        format_segmented_distribution_markdown,
        format_segmented_distribution_text,
    )
    from llm_drift_bench.engine import load_model_samples, load_prompt_cases
    from llm_drift_bench.policy import load_policy

    base = load_model_samples(args.baseline)
    cand = load_model_samples(args.candidate)
    policy = load_policy(args.config) if args.config else None
    prompts = load_prompt_cases(args.suite) if args.suite else None

    failures: list[str] = []
    has_limits = bool(args.fail_on_widespread) or any(
        val is not None for val in (args.max_tvd, args.max_jsd, args.max_wasserstein)
    )
    if args.group_by:
        seg_report = analyze_segmented_distributions(
            base,
            cand,
            group_by=args.group_by,
            prompt_cases=prompts,
            policy=policy,
            collapse_array_indices=not args.no_collapse_arrays,
            shift_threshold=args.shift_threshold,
        )
        for seg_name, seg in seg_report.segments.items():
            if args.max_tvd is not None and seg.max_tvd > args.max_tvd:
                failures.append(
                    f"[{seg_name}] max_tvd {seg.max_tvd:.4f} exceeded limit {args.max_tvd:.4f}"
                )
            if args.max_jsd is not None and seg.max_jsd > args.max_jsd:
                failures.append(
                    f"[{seg_name}] max_jsd {seg.max_jsd:.4f} exceeded limit {args.max_jsd:.4f}"
                )
            if (
                args.max_wasserstein is not None
                and seg.max_normalized_wasserstein > args.max_wasserstein
            ):
                failures.append(
                    f"[{seg_name}] max_normalized_wasserstein "
                    f"{seg.max_normalized_wasserstein:.4f} exceeded limit "
                    f"{args.max_wasserstein:.4f}"
                )
        if args.fail_on_widespread and seg_report.shift_scope == "widespread":
            shifted = ", ".join(seg_report.shifted_segments)
            failures.append(
                f"widespread shift across {len(seg_report.shifted_segments)}/"
                f"{len(seg_report.segments)} segment(s): {shifted}"
            )
        passed = len(failures) == 0
        if args.format == "json":
            payload = seg_report.to_dict()
            if has_limits:
                payload["passed"] = passed
                payload["threshold_failures"] = failures
            rendered = json.dumps(payload, indent=2) + "\n"
        elif args.format == "markdown":
            rendered = format_segmented_distribution_markdown(seg_report)
        elif args.format == "html":
            rendered = export_segmented_distribution_html(seg_report)
        else:
            rendered = format_segmented_distribution_text(seg_report)
            if failures:
                rendered += "Threshold failures:\n" + "".join(f"  - {msg}\n" for msg in failures)
    else:
        report = analyze_field_distributions(
            base,
            cand,
            policy=policy,
            collapse_array_indices=not args.no_collapse_arrays,
        )
        if args.max_tvd is not None and report.max_tvd > args.max_tvd:
            failures.append(f"max_tvd {report.max_tvd:.4f} exceeded limit {args.max_tvd:.4f}")
        if args.max_jsd is not None and report.max_jsd > args.max_jsd:
            failures.append(f"max_jsd {report.max_jsd:.4f} exceeded limit {args.max_jsd:.4f}")
        if (
            args.max_wasserstein is not None
            and report.max_normalized_wasserstein > args.max_wasserstein
        ):
            failures.append(
                f"max_normalized_wasserstein {report.max_normalized_wasserstein:.4f} "
                f"exceeded limit {args.max_wasserstein:.4f}"
            )
        passed = len(failures) == 0
        if args.format == "json":
            payload = report.to_dict()
            if has_limits:
                payload["passed"] = passed
                payload["threshold_failures"] = failures
            rendered = json.dumps(payload, indent=2) + "\n"
        elif args.format == "markdown":
            rendered = format_distribution_markdown(report)
        elif args.format == "html":
            rendered = export_distribution_html(report)
        else:
            rendered = format_distribution_text(report)
            if failures:
                rendered += "Threshold failures:\n" + "".join(f"  - {msg}\n" for msg in failures)

    if args.output:
        atomic_write_text(Path(args.output), rendered)
        if args.format == "html":
            print(f"Wrote field distribution report -> {args.output}")
        else:
            print(rendered, end="")
    else:
        print(rendered, end="")
    return 0 if passed else 1


def _handle_wizard(
    args: argparse.Namespace,
    atomic_write_text: AtomicWriteFn,
) -> int:
    from llm_drift_bench.wizard import (
        PRESET_ANSWERS,
        format_wizard_next_steps,
        render_wizard_config,
        run_guided_wizard,
    )

    if args.preset is not None or args.non_interactive:
        answers = PRESET_ANSWERS[args.preset or "extraction"]
    else:
        answers, _ = run_guided_wizard()

    toml_text = render_wizard_config(answers)
    if args.output:
        atomic_write_text(Path(args.output), toml_text)
        print(f"Wrote {answers.use_case} drift policy -> {args.output}")
        print(format_wizard_next_steps(answers, config_path=args.output), end="")
    else:
        print(toml_text, end="")
    return 0


def _handle_logprobs(args: argparse.Namespace) -> int:
    if args.max_entropy_delta is not None and (
        not math.isfinite(args.max_entropy_delta) or args.max_entropy_delta < 0.0
    ):
        raise ValueError("--max-entropy-delta must be a finite non-negative number.")
    _validate_unit_threshold("--max-jsd", args.max_jsd)

    from llm_drift_bench.engine import load_model_samples
    from llm_drift_bench.logprobs import analyze_logprobs_drift, format_logprobs_text

    base = load_model_samples(args.baseline)
    cand = load_model_samples(args.candidate)
    summary = analyze_logprobs_drift(base, cand)

    has_limits = args.max_entropy_delta is not None or args.max_jsd is not None
    failures: list[str] = []
    if has_limits and not summary.complete:
        failures.append(
            f"incomplete logprobs coverage ({summary.comparable_samples}/{summary.total_samples})"
        )
    if (
        args.max_entropy_delta is not None
        and summary.mean_entropy_delta_bits is not None
        and abs(summary.mean_entropy_delta_bits) > args.max_entropy_delta
    ):
        failures.append(
            f"abs(mean_entropy_delta_bits) {abs(summary.mean_entropy_delta_bits):.4f} "
            f"exceeded limit {args.max_entropy_delta:.4f}"
        )
    if (
        args.max_jsd is not None
        and summary.max_top_k_jsd is not None
        and summary.max_top_k_jsd > args.max_jsd
    ):
        failures.append(
            f"max_top_k_jsd {summary.max_top_k_jsd:.4f} exceeded limit {args.max_jsd:.4f}"
        )
    passed = len(failures) == 0

    if args.format == "json":
        payload = summary.to_dict()
        if has_limits:
            payload["passed"] = passed
            payload["threshold_failures"] = failures
        print(json.dumps(payload, indent=2))
    else:
        text = format_logprobs_text(summary)
        if failures:
            text += "Threshold failures:\n" + "".join(f"  - {msg}\n" for msg in failures)
        print(text, end="")
    return 0 if passed else 1


def _handle_check_quality(args: argparse.Namespace) -> int:
    from llm_drift_bench.engine import load_model_samples
    from llm_drift_bench.quality_checks import format_quality_report_text, run_quality_checks

    base = load_model_samples(args.baseline)
    cand = load_model_samples(args.candidate)
    report = run_quality_checks(
        base,
        cand,
        max_length_ratio=args.max_length_ratio,
        min_length_ratio=args.min_length_ratio,
        min_baseline_chars=args.min_baseline_chars,
    )
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_quality_report_text(report), end="")
    if not report.passed and args.fail_on_issues:
        return 1
    return 0


def _handle_check_formats(args: argparse.Namespace) -> int:
    from llm_drift_bench.engine import load_model_samples
    from llm_drift_bench.formats import (
        check_field_formats,
        format_check_report_text,
        parse_format_rule_spec,
    )

    rules = dict(parse_format_rule_spec(raw) for raw in args.rule)
    samples = load_model_samples(args.input)
    report = check_field_formats(samples, rules, allow_null=args.allow_null)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_check_report_text(report), end="")
    return 0 if report.passed else 1


def _handle_audit(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    if args.fail_on_widespread and not args.group_by:
        raise ValueError("--fail-on-widespread requires --group-by.")
    using_decisions = (
        args.baseline_decisions is not None or args.candidate_decisions is not None
    )
    if using_decisions and args.validator_module is not None:
        raise ValueError(
            "Provide either --baseline-decisions/--candidate-decisions OR "
            "--validator-module on audit, not both."
        )
    if using_decisions and not (args.baseline_decisions and args.candidate_decisions):
        raise ValueError("Both --baseline-decisions and --candidate-decisions are required.")
    using_convs = (
        args.baseline_conversations is not None or args.candidate_conversations is not None
    )
    if using_convs and not (args.baseline_conversations and args.candidate_conversations):
        raise ValueError(
            "Both --baseline-conversations and --candidate-conversations are required."
        )
    using_traces = args.baseline_trace is not None or args.candidate_trace is not None
    if using_traces and not (args.baseline_trace and args.candidate_trace):
        raise ValueError("Both --baseline-trace and --candidate-trace are required.")
    protect_inputs(
        args.output,
        args.baseline,
        args.candidate,
        args.suite,
        args.config,
        args.baseline_decisions,
        args.candidate_decisions,
        args.baseline_conversations,
        args.candidate_conversations,
        args.baseline_trace,
        args.candidate_trace,
    )
    from llm_drift_bench.audit import (
        export_audit_html,
        format_audit_markdown,
        format_audit_text,
        run_full_audit,
    )
    from llm_drift_bench.engine import load_model_samples, load_prompt_cases
    from llm_drift_bench.policy import load_policy

    base = load_model_samples(args.baseline)
    cand = load_model_samples(args.candidate)
    prompts = load_prompt_cases(args.suite) if args.suite else None
    policy = load_policy(args.config) if args.config else None

    consumer_comp = None
    if using_decisions:
        from llm_drift_bench.consumer import (
            compare_consumer_decisions,
            load_consumer_decisions,
        )

        consumer_comp = compare_consumer_decisions(
            load_consumer_decisions(args.baseline_decisions),
            load_consumer_decisions(args.candidate_decisions),
            max_rejection_rate=args.max_rejection_rate,
            max_rejection_increase=args.max_rejection_increase,
        )
    elif args.validator_module is not None:
        from llm_drift_bench.consumer import (
            evaluate_consumer_rejection,
            load_validator_callable,
        )

        consumer_comp = evaluate_consumer_rejection(
            base,
            cand,
            load_validator_callable(args.validator_module),
            max_rejection_rate=args.max_rejection_rate,
            max_rejection_increase=args.max_rejection_increase,
        )

    embedding_rep = None
    if args.min_embedding_similarity is not None:
        from llm_drift_bench.embeddings import analyze_capture_embeddings

        embedding_rep = analyze_capture_embeddings(
            base,
            cand,
            min_mean_similarity=args.min_embedding_similarity,
        )

    conversation_rep = None
    if using_convs:
        from llm_drift_bench.conversation import (
            analyze_conversation_drift,
            load_conversations,
        )

        conversation_rep = analyze_conversation_drift(
            load_conversations(args.baseline_conversations),
            load_conversations(args.candidate_conversations),
            max_turn_drift=args.max_turn_drift,
            max_mean_drift=args.max_mean_trajectory_drift,
        )

    tool_trace_diff = None
    if using_traces:
        from llm_drift_bench.adapters import load_adapted_tool_events
        from llm_drift_bench.tool_events import compare_tool_traces
        from llm_drift_bench.tool_log import load_tool_events

        if args.trace_adapter == "normalized":
            base_ev = load_tool_events(args.baseline_trace)
            cand_ev = load_tool_events(args.candidate_trace)
        else:
            base_ev = load_adapted_tool_events(args.baseline_trace, adapter=args.trace_adapter)
            cand_ev = load_adapted_tool_events(args.candidate_trace, adapter=args.trace_adapter)
        tool_trace_diff = compare_tool_traces(
            base_ev,
            cand_ev,
            max_call_increase=args.max_tool_call_increase,
            max_failed_increase=args.max_tool_failed_increase,
            fail_on_new_edit_resources=args.fail_on_new_edit_resources,
        )

    report = run_full_audit(
        base,
        cand,
        prompt_cases=prompts,
        policy=policy,
        max_tvd=args.max_tvd,
        max_jsd=args.max_jsd,
        max_wasserstein=args.max_wasserstein,
        max_entropy_delta=args.max_entropy_delta,
        max_top_k_jsd=args.max_top_k_jsd,
        fail_on_quality_issues=args.fail_on_quality,
        group_by=args.group_by,
        shift_threshold=args.shift_threshold,
        fail_on_widespread_shift=args.fail_on_widespread,
        consumer_comparison=consumer_comp,
        embedding_report=embedding_rep,
        conversation_report=conversation_rep,
        tool_trace_diff=tool_trace_diff,
    )
    if args.format == "json":
        rendered = json.dumps(report.to_dict(), indent=2) + "\n"
    elif args.format == "markdown":
        rendered = format_audit_markdown(report)
    elif args.format == "html":
        rendered = export_audit_html(report)
    else:
        rendered = format_audit_text(report)

    if args.output:
        atomic_write_text(Path(args.output), rendered)
        if args.format == "html":
            print(f"Wrote unified offline audit dashboard -> {args.output}")
        else:
            print(rendered, end="")
    else:
        print(rendered, end="")
    return 0 if report.passed else 1


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
    if args.command == "import-captures":
        return _handle_import_captures(args, protect_inputs, atomic_write_text)
    if args.command == "badge":
        return _handle_badge(args, protect_inputs, atomic_write_text)
    if args.command == "distribution":
        return _handle_distribution(args, protect_inputs, atomic_write_text)
    if args.command == "wizard":
        return _handle_wizard(args, atomic_write_text)
    if args.command == "logprobs":
        return _handle_logprobs(args)
    if args.command == "check-quality":
        return _handle_check_quality(args)
    if args.command == "check-formats":
        return _handle_check_formats(args)
    if args.command == "audit":
        return _handle_audit(args, protect_inputs, atomic_write_text)
    if args.command == "check-consumer":
        return _handle_check_consumer(args, protect_inputs, atomic_write_text)
    if args.command == "rolling-distribution":
        return _handle_rolling_distribution(args, protect_inputs, atomic_write_text)
    if args.command == "check-embeddings":
        return _handle_check_embeddings(args, protect_inputs, atomic_write_text)
    if args.command == "check-conversations":
        return _handle_check_conversations(args, protect_inputs, atomic_write_text)
    if args.command == "import-conversations":
        return _handle_import_conversations(args, protect_inputs, atomic_write_text)
    if args.command == "diff-tools":
        return _handle_diff_tools(args, protect_inputs, atomic_write_text)
    if args.command == "capture-conversations":
        return _handle_capture_conversations(args, protect_inputs, atomic_write_text)
    return None


def _handle_check_consumer(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(
        args.output,
        args.baseline_decisions,
        args.candidate_decisions,
        args.baseline,
        args.candidate,
    )
    from llm_drift_bench.consumer import (
        compare_consumer_decisions,
        evaluate_consumer_rejection,
        export_consumer_html,
        format_consumer_markdown,
        format_consumer_text,
        load_consumer_decisions,
        load_validator_callable,
    )

    using_decisions = (
        args.baseline_decisions is not None or args.candidate_decisions is not None
    )
    using_validator = (
        args.baseline is not None
        or args.candidate is not None
        or args.validator_module is not None
    )
    if using_decisions and using_validator:
        raise ValueError(
            "Provide either --baseline-decisions/--candidate-decisions OR "
            "-b/-c with --validator-module, not both."
        )
    if using_decisions:
        if not (args.baseline_decisions and args.candidate_decisions):
            raise ValueError(
                "Both --baseline-decisions and --candidate-decisions are required."
            )
        comp = compare_consumer_decisions(
            load_consumer_decisions(args.baseline_decisions),
            load_consumer_decisions(args.candidate_decisions),
            max_rejection_rate=args.max_rejection_rate,
            max_rejection_increase=args.max_rejection_increase,
        )
    elif using_validator:
        if not (args.baseline and args.candidate and args.validator_module):
            raise ValueError(
                "Provide -b/--baseline, -c/--candidate, and --validator-module MOD:FUNC."
            )
        from llm_drift_bench.engine import load_model_samples

        fn = load_validator_callable(args.validator_module)
        comp = evaluate_consumer_rejection(
            load_model_samples(args.baseline),
            load_model_samples(args.candidate),
            fn,
            max_rejection_rate=args.max_rejection_rate,
            max_rejection_increase=args.max_rejection_increase,
        )
    else:
        raise ValueError(
            "Specify --baseline-decisions/--candidate-decisions or "
            "-b/-c with --validator-module."
        )

    if args.format == "json":
        rendered = json.dumps(comp.to_dict(), indent=2) + "\n"
    elif args.format == "markdown":
        rendered = format_consumer_markdown(comp)
    elif args.format == "html":
        rendered = export_consumer_html(comp)
    else:
        rendered = format_consumer_text(comp)
    if args.output:
        atomic_write_text(Path(args.output), rendered)
        if args.format == "html":
            print(f"Wrote downstream consumer rejection report -> {args.output}")
        else:
            print(rendered, end="")
    else:
        print(rendered, end="")
    return 0 if comp.passed else 1


def _handle_rolling_distribution(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(args.output, *args.window, args.candidate, args.config, args.suite)
    from llm_drift_bench.engine import load_model_samples, load_prompt_cases
    from llm_drift_bench.policy import load_policy
    from llm_drift_bench.rolling import (
        analyze_rolling_distribution,
        export_rolling_html,
        format_rolling_markdown,
        format_rolling_text,
    )

    windows = [load_model_samples(path) for path in args.window]
    cand = load_model_samples(args.candidate)
    policy = load_policy(args.config) if args.config else None
    prompts = load_prompt_cases(args.suite) if args.suite else None

    raw_stems = [Path(path).stem or f"window_{i}" for i, path in enumerate(args.window, start=1)]
    seen: dict[str, int] = {}
    labels: list[str] = []
    for stem in raw_stems:
        count = seen.get(stem, 0) + 1
        seen[stem] = count
        labels.append(stem if count == 1 else f"{stem}_{count}")

    report = analyze_rolling_distribution(
        windows,
        cand,
        window_labels=labels,
        group_by=args.group_by,
        prompt_cases=prompts,
        policy=policy,
        collapse_array_indices=not args.no_collapse_arrays,
        shift_threshold=args.shift_threshold,
        max_tvd=args.max_tvd,
        max_jsd=args.max_jsd,
        max_wasserstein=args.max_wasserstein,
        fail_on_widespread_shift=args.fail_on_widespread,
        max_jitter_multiplier=args.max_jitter_multiplier,
    )
    if args.format == "json":
        rendered = json.dumps(report.to_dict(), indent=2) + "\n"
    elif args.format == "markdown":
        rendered = format_rolling_markdown(report)
    elif args.format == "html":
        rendered = export_rolling_html(report)
    else:
        rendered = format_rolling_text(report)
    if args.output:
        atomic_write_text(Path(args.output), rendered)
        if args.format == "html":
            print(f"Wrote rolling baseline distribution report -> {args.output}")
        else:
            print(rendered, end="")
    else:
        print(rendered, end="")
    return 0 if report.passed else 1


def _handle_check_embeddings(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(args.output, args.baseline, args.candidate)
    if bool(args.endpoint) != bool(args.model):
        raise ValueError("Provide both --endpoint and --model together for live embeddings.")
    from llm_drift_bench.embeddings import (
        EmbeddingConfig,
        analyze_capture_embeddings,
        export_embedding_drift_html,
        format_embedding_drift_markdown,
        format_embedding_drift_text,
    )
    from llm_drift_bench.engine import load_model_samples

    base = load_model_samples(args.baseline)
    cand = load_model_samples(args.candidate)
    cfg = (
        EmbeddingConfig(
            endpoint=args.endpoint,
            model=args.model,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout,
            batch_size=args.batch_size,
        )
        if args.endpoint
        else None
    )
    report = analyze_capture_embeddings(
        base,
        cand,
        config=cfg,
        min_mean_similarity=args.min_mean_similarity,
        min_prompt_similarity=args.min_prompt_similarity,
    )
    if args.format == "json":
        rendered = json.dumps(report.to_dict(), indent=2) + "\n"
    elif args.format == "markdown":
        rendered = format_embedding_drift_markdown(report)
    elif args.format == "html":
        rendered = export_embedding_drift_html(report)
    else:
        rendered = format_embedding_drift_text(report)
    if args.output:
        atomic_write_text(Path(args.output), rendered)
        if args.format == "html":
            print(f"Wrote embedding similarity drift report -> {args.output}")
        else:
            print(rendered, end="")
    else:
        print(rendered, end="")
    return 0 if report.passed else 1


def _handle_check_conversations(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(args.output, args.baseline, args.candidate)
    from llm_drift_bench.conversation import (
        analyze_conversation_drift,
        diff_conversations,
        export_conversation_html,
        format_conversation_diffs_text,
        format_conversation_markdown,
        format_conversation_text,
        load_conversations,
    )

    base = load_conversations(args.baseline)
    cand = load_conversations(args.candidate)
    report = analyze_conversation_drift(
        base,
        cand,
        max_turn_drift=args.max_turn_drift,
        max_mean_drift=args.max_mean_drift,
        max_final_turn_drift=args.max_final_turn_drift,
        max_turn_count_delta=args.max_turn_count_delta,
        fail_on_refusal_flip=args.fail_on_refusal_flip,
        long_text_mode=args.long_text_mode,
    )
    turn_diffs = (
        diff_conversations(base, cand, only_changed=True)
        if getattr(args, "show_diffs", False)
        else None
    )
    findings = None
    if getattr(args, "diagnose", False):
        from llm_drift_bench.diagnostics import diagnose_conversation_report

        findings = diagnose_conversation_report(report)
    if args.format == "json":
        payload = report.to_dict()
        if turn_diffs is not None:
            payload["turn_diffs"] = [d.to_dict() for d in turn_diffs]
        if findings is not None:
            payload["diagnostics"] = [
                {
                    "code": f.code,
                    "conversation_id": f.prompt_id,
                    "evidence": f.evidence,
                    "suggestion": f.suggestion,
                }
                for f in findings
            ]
        rendered = json.dumps(payload, indent=2) + "\n"
    elif args.format == "markdown":
        rendered = format_conversation_markdown(report)
    elif args.format == "html":
        rendered = export_conversation_html(report)
    elif args.format == "sarif":
        from llm_drift_bench.sarif import export_conversation_sarif_report

        rendered = export_conversation_sarif_report(
            report, artifact_uri=Path(args.candidate).name
        )
    elif args.format == "junit":
        from llm_drift_bench.junit import export_conversation_junit_report

        rendered = export_conversation_junit_report(report)
    elif args.format == "csv":
        from llm_drift_bench.csv_export import export_conversation_csv_report

        rendered = export_conversation_csv_report(report)
    else:
        rendered = format_conversation_text(report)
        if turn_diffs is not None:
            rendered += "\n" + format_conversation_diffs_text(turn_diffs)
        if findings is not None:
            from llm_drift_bench.diagnostics import format_findings_text

            rendered += "\n" + format_findings_text(findings)
    if args.output:
        atomic_write_text(Path(args.output), rendered)
        if args.format == "html":
            print(f"Wrote multi-turn conversation drift report -> {args.output}")
        else:
            print(rendered, end="")
    else:
        print(rendered, end="")
    return 0 if report.passed else 1


def _handle_import_conversations(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(args.output, args.input)
    from llm_drift_bench.batch_import import (
        import_conversations_jsonl,
        serialize_conversations_jsonl,
    )

    records = import_conversations_jsonl(
        args.input,
        source_format=args.source_format,
        model_name=args.model,
    )
    atomic_write_text(Path(args.output), serialize_conversations_jsonl(records))
    print(f"Imported {len(records)} conversation(s) ({args.source_format}) -> {args.output}")
    return 0


def _handle_diff_tools(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(args.output, args.baseline, args.candidate)
    from llm_drift_bench.adapters import load_adapted_tool_events
    from llm_drift_bench.tool_events import (
        compare_tool_traces,
        export_tool_trace_diff_html,
        format_tool_trace_diff_markdown,
        format_tool_trace_diff_text,
    )
    from llm_drift_bench.tool_log import load_tool_events

    if args.adapter == "normalized":
        base_events = load_tool_events(args.baseline)
        cand_events = load_tool_events(args.candidate)
    else:
        base_events = load_adapted_tool_events(args.baseline, adapter=args.adapter)
        cand_events = load_adapted_tool_events(args.candidate, adapter=args.adapter)

    diff = compare_tool_traces(
        base_events,
        cand_events,
        max_call_increase=args.max_call_increase,
        max_failed_increase=args.max_failed_increase,
        fail_on_new_edit_resources=args.fail_on_new_edit_resources,
    )
    if args.format == "json":
        rendered = json.dumps(diff.to_dict(), indent=2) + "\n"
    elif args.format == "markdown":
        rendered = format_tool_trace_diff_markdown(diff)
    elif args.format == "html":
        rendered = export_tool_trace_diff_html(diff)
    elif args.format == "sarif":
        from llm_drift_bench.sarif import export_tool_trace_sarif_report

        rendered = export_tool_trace_sarif_report(
            diff, artifact_uri=Path(args.candidate).name
        )
    elif args.format == "junit":
        from llm_drift_bench.junit import export_tool_trace_junit_report

        rendered = export_tool_trace_junit_report(diff)
    else:
        rendered = format_tool_trace_diff_text(diff)
    if args.output:
        atomic_write_text(Path(args.output), rendered)
        if args.format == "html":
            print(f"Wrote agent tool-trace drift report -> {args.output}")
        else:
            print(rendered, end="")
    else:
        print(rendered, end="")
    return 0 if diff.passed else 1


def _handle_capture_conversations(
    args: argparse.Namespace,
    protect_inputs: ProtectFn,
    atomic_write_text: AtomicWriteFn,
) -> int:
    protect_inputs(args.output, args.input)
    from llm_drift_bench.batch_import import serialize_conversations_jsonl
    from llm_drift_bench.capture import (
        CaptureConfig,
        capture_conversations,
        parse_header_spec,
    )
    from llm_drift_bench.conversation import load_conversations

    templates = load_conversations(args.input)
    cfg = CaptureConfig(
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
        extra_headers=tuple(parse_header_spec(raw) for raw in args.header),
        provider=args.provider,
    )
    captured = capture_conversations(templates, cfg)
    atomic_write_text(Path(args.output), serialize_conversations_jsonl(captured))
    total_turns = sum(len(rec.assistant_turns()) for rec in captured)
    print(
        f"Captured {len(captured)} conversation(s) ({total_turns} assistant turn(s)) "
        f"for model {cfg.model!r} -> {args.output}"
    )
    return 0
