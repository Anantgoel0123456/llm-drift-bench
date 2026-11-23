"""RFC-4180 CSV exporter for canonical DriftBenchmarkReport per-prompt metrics."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from llm_drift_bench.models import DriftBenchmarkReport

CSV_COLUMNS = (
    "baseline_model",
    "candidate_model",
    "prompt_id",
    "category",
    "drifted",
    "composite_drift_score",
    "token_jaccard_similarity",
    "ngram_similarity",
    "levenshtein_similarity",
    "json_structural_similarity",
    "json_exact_field_match",
    "json_numeric_drift",
    "baseline_passed",
    "candidate_passed",
    "candidate_issues",
    "latency_delta_ms",
)


_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _fmt_opt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _safe_csv_text(value: str, *, sanitize: bool) -> str:
    if sanitize and value and value[0] in _FORMULA_PREFIXES:
        return "'" + value
    return value


def export_csv_report(
    report: DriftBenchmarkReport,
    output_path: str | Path | None = None,
    *,
    sanitize_formulas: bool = True,
) -> str:
    """Export per-prompt benchmark metrics as an RFC-4180 CSV string (and optional file).

    When sanitize_formulas=True (default), text identifier columns starting with
    spreadsheet formula triggers (=, +, -, @, tab, CR) are prefixed with a leading
    apostrophe so Excel/Sheets cannot execute them as formulas.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in report.results:
        issues_str = "; ".join(
            f"{issue.path}: {issue.rule}" for issue in row.candidate_validation.issues
        )
        writer.writerow([
            _safe_csv_text(report.baseline_model, sanitize=sanitize_formulas),
            _safe_csv_text(report.candidate_model, sanitize=sanitize_formulas),
            _safe_csv_text(row.prompt_id, sanitize=sanitize_formulas),
            _safe_csv_text(row.category, sanitize=sanitize_formulas),
            "true" if row.drifted else "false",
            f"{row.composite_drift_score:.4f}",
            f"{row.token_jaccard_similarity:.4f}",
            f"{row.ngram_similarity:.4f}",
            f"{row.levenshtein_similarity:.4f}",
            _fmt_opt(row.json_structural_similarity, 4),
            _fmt_opt(row.json_exact_field_match, 4),
            _fmt_opt(row.json_numeric_drift, 4),
            "true" if row.baseline_validation.passed else "false",
            "true" if row.candidate_validation.passed else "false",
            _safe_csv_text(issues_str, sanitize=sanitize_formulas),
            _fmt_opt(row.latency_delta_ms, 2),
        ])

    serialized = buffer.getvalue()
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized, encoding="utf-8")
    return serialized
