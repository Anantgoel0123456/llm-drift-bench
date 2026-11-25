"""Cross-prompt JSON field distribution drift analysis (TVD, JSD, quantiles, 1D Wasserstein)."""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal

from llm_drift_bench._matching import index_records, require_same_ids
from llm_drift_bench._numbers import finite_number
from llm_drift_bench._report_views import markdown_cell
from llm_drift_bench.metrics import flatten_json_paths
from llm_drift_bench.models import ModelSample, PromptCase
from llm_drift_bench.policy import DriftPolicy, path_is_ignored
from llm_drift_bench.validator import extract_json_payload

_INDEX_OR_QUOTED_RE = re.compile(r'\["(?:\\.|[^"\\])*"\]|\[\d+\]')


def _collapse_array_indices(path: str) -> str:
    return _INDEX_OR_QUOTED_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith('["') else "[*]",
        path,
    )


def _canonical_category(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _wasserstein_1d(sorted_a: Sequence[float], sorted_b: Sequence[float]) -> float:
    """Exact 1D empirical Wasserstein-1 distance: integral |F_A(x) - F_B(x)| dx."""
    n_a = len(sorted_a)
    n_b = len(sorted_b)
    points = sorted(set(sorted_a) | set(sorted_b))
    if len(points) < 2:
        return 0.0
    total = 0.0
    for idx in range(len(points) - 1):
        x0 = points[idx]
        x1 = points[idx + 1]
        cdf_a = bisect_right(sorted_a, x0) / n_a
        cdf_b = bisect_right(sorted_b, x0) / n_b
        total += abs(cdf_a - cdf_b) * (x1 - x0)
    return total


def _tvd_and_jsd(
    shares_a: dict[str, float],
    shares_b: dict[str, float],
) -> tuple[float, float]:
    """Compute Total Variation Distance and base-2 Jensen-Shannon Divergence in [0, 1]."""
    labels = sorted(set(shares_a) | set(shares_b))
    tvd = 0.5 * sum(abs(shares_a.get(lbl, 0.0) - shares_b.get(lbl, 0.0)) for lbl in labels)
    kl_a = 0.0
    kl_b = 0.0
    for lbl in labels:
        p = shares_a.get(lbl, 0.0)
        q = shares_b.get(lbl, 0.0)
        m = 0.5 * (p + q)
        if p > 0.0:
            kl_a += p * math.log2(p / m)
        if q > 0.0:
            kl_b += q * math.log2(q / m)
    jsd = 0.5 * (kl_a + kl_b)
    return max(0.0, min(1.0, tvd)), max(0.0, min(1.0, jsd))


@dataclass(frozen=True)
class CategoricalFieldDrift:
    """Cross-prompt label-share shift for a categorical JSON field."""

    path: str
    baseline_count: int
    candidate_count: int
    tvd: float
    jsd: float
    baseline_shares: dict[str, float]
    candidate_shares: dict[str, float]
    top_shifts: tuple[tuple[str, float, float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "baseline_count": self.baseline_count,
            "candidate_count": self.candidate_count,
            "tvd": round(self.tvd, 6),
            "jsd": round(self.jsd, 6),
            "baseline_shares": {k: round(v, 6) for k, v in self.baseline_shares.items()},
            "candidate_shares": {k: round(v, 6) for k, v in self.candidate_shares.items()},
            "top_shifts": [
                {
                    "label": label,
                    "baseline_share": round(base_s, 6),
                    "candidate_share": round(cand_s, 6),
                    "delta": round(delta, 6),
                }
                for label, base_s, cand_s, delta in self.top_shifts
            ],
        }


@dataclass(frozen=True)
class NumericFieldDrift:
    """Cross-prompt quantile and 1D Wasserstein shift for a numeric JSON field."""

    path: str
    baseline_count: int
    candidate_count: int
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    baseline_p10: float
    baseline_p50: float
    baseline_p90: float
    candidate_p10: float
    candidate_p50: float
    candidate_p90: float
    median_delta: float
    wasserstein_1d: float
    normalized_wasserstein: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "baseline_count": self.baseline_count,
            "candidate_count": self.candidate_count,
            "baseline_mean": round(self.baseline_mean, 6),
            "candidate_mean": round(self.candidate_mean, 6),
            "mean_delta": round(self.mean_delta, 6),
            "baseline_p10": round(self.baseline_p10, 6),
            "baseline_p50": round(self.baseline_p50, 6),
            "baseline_p90": round(self.baseline_p90, 6),
            "candidate_p10": round(self.candidate_p10, 6),
            "candidate_p50": round(self.candidate_p50, 6),
            "candidate_p90": round(self.candidate_p90, 6),
            "median_delta": round(self.median_delta, 6),
            "wasserstein_1d": round(self.wasserstein_1d, 6),
            "normalized_wasserstein": round(self.normalized_wasserstein, 6),
        }


@dataclass(frozen=True)
class DistributionReport:
    """Aggregate cross-prompt JSON field distribution drift summary."""

    total_prompts: int
    comparable_json_prompts: int
    categorical_fields: tuple[CategoricalFieldDrift, ...]
    numeric_fields: tuple[NumericFieldDrift, ...]
    max_tvd: float
    max_jsd: float
    max_normalized_wasserstein: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_prompts": self.total_prompts,
            "comparable_json_prompts": self.comparable_json_prompts,
            "max_tvd": round(self.max_tvd, 6),
            "max_jsd": round(self.max_jsd, 6),
            "max_normalized_wasserstein": round(self.max_normalized_wasserstein, 6),
            "categorical_fields": [item.to_dict() for item in self.categorical_fields],
            "numeric_fields": [item.to_dict() for item in self.numeric_fields],
        }


def analyze_field_distributions(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
    *,
    policy: DriftPolicy | None = None,
    collapse_array_indices: bool = True,
) -> DistributionReport:
    """Detect cross-prompt categorical and numeric distribution shifts across JSON fields.

    Catches soft slides where individual outputs still satisfy JSON schema constraints,
    such as numeric scores drifting toward the middle of their range or categorical label
    shares flattening across the suite.
    """
    if not baseline_samples or not candidate_samples:
        raise ValueError("Both baseline and candidate sample collections must be non-empty.")

    base_map = index_records(baseline_samples, "baseline samples")
    cand_map = index_records(candidate_samples, "candidate samples")
    require_same_ids(set(base_map), set(cand_map), "Candidate")

    ignored = (
        tuple(path for path, rule in policy.field_rules.items() if rule.ignore)
        if policy is not None
        else ()
    )

    base_nums: dict[str, list[float]] = defaultdict(list)
    cand_nums: dict[str, list[float]] = defaultdict(list)
    base_cats: dict[str, Counter[str]] = defaultdict(Counter)
    cand_cats: dict[str, Counter[str]] = defaultdict(Counter)
    comparable = 0

    for prompt_id in sorted(base_map):
        ok_a, payload_a, _ = extract_json_payload(base_map[prompt_id].output)
        ok_b, payload_b, _ = extract_json_payload(cand_map[prompt_id].output)
        if not (ok_a and ok_b):
            continue
        comparable += 1
        for flat, nums_out, cats_out in (
            (flatten_json_paths(payload_a), base_nums, base_cats),
            (flatten_json_paths(payload_b), cand_nums, cand_cats),
        ):
            for raw_path, val in flat.items():
                if ignored and path_is_ignored(raw_path, ignored):
                    continue
                key = _collapse_array_indices(raw_path) if collapse_array_indices else raw_path
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    nums_out[key].append(finite_number(val))
                else:
                    cats_out[key][_canonical_category(val)] += 1

    cat_results: list[CategoricalFieldDrift] = []
    for path in sorted(set(base_cats) & set(cand_cats)):
        b_counts = base_cats[path]
        c_counts = cand_cats[path]
        b_total = sum(b_counts.values())
        c_total = sum(c_counts.values())
        if b_total == 0 or c_total == 0:
            continue
        b_shares = {lbl: b_counts[lbl] / b_total for lbl in sorted(b_counts)}
        c_shares = {lbl: c_counts[lbl] / c_total for lbl in sorted(c_counts)}
        tvd, jsd = _tvd_and_jsd(b_shares, c_shares)
        all_labels = sorted(set(b_shares) | set(c_shares))
        shifts = sorted(
            (
                (
                    lbl,
                    b_shares.get(lbl, 0.0),
                    c_shares.get(lbl, 0.0),
                    c_shares.get(lbl, 0.0) - b_shares.get(lbl, 0.0),
                )
                for lbl in all_labels
            ),
            key=lambda item: (-abs(item[3]), item[0]),
        )
        cat_results.append(
            CategoricalFieldDrift(
                path=path,
                baseline_count=b_total,
                candidate_count=c_total,
                tvd=tvd,
                jsd=jsd,
                baseline_shares=b_shares,
                candidate_shares=c_shares,
                top_shifts=tuple(shifts[:5]),
            )
        )
    cat_results.sort(key=lambda item: (-item.tvd, -item.jsd, item.path))

    num_results: list[NumericFieldDrift] = []
    for path in sorted(set(base_nums) & set(cand_nums)):
        b_vals = sorted(base_nums[path])
        c_vals = sorted(cand_nums[path])
        if not b_vals or not c_vals:
            continue
        b_mean = sum(b_vals) / len(b_vals)
        c_mean = sum(c_vals) / len(c_vals)
        b_p10, b_p50, b_p90 = _quantile(b_vals, 0.1), _quantile(b_vals, 0.5), _quantile(b_vals, 0.9)
        c_p10, c_p50, c_p90 = _quantile(c_vals, 0.1), _quantile(c_vals, 0.5), _quantile(c_vals, 0.9)
        w1 = _wasserstein_1d(b_vals, c_vals)
        span = max(b_vals[-1], c_vals[-1]) - min(b_vals[0], c_vals[0])
        norm_w1 = 0.0 if span == 0.0 else min(1.0, w1 / span)
        num_results.append(
            NumericFieldDrift(
                path=path,
                baseline_count=len(b_vals),
                candidate_count=len(c_vals),
                baseline_mean=b_mean,
                candidate_mean=c_mean,
                mean_delta=c_mean - b_mean,
                baseline_p10=b_p10,
                baseline_p50=b_p50,
                baseline_p90=b_p90,
                candidate_p10=c_p10,
                candidate_p50=c_p50,
                candidate_p90=c_p90,
                median_delta=c_p50 - b_p50,
                wasserstein_1d=w1,
                normalized_wasserstein=norm_w1,
            )
        )
    num_results.sort(
        key=lambda item: (-item.normalized_wasserstein, -abs(item.mean_delta), item.path)
    )

    return DistributionReport(
        total_prompts=len(base_map),
        comparable_json_prompts=comparable,
        categorical_fields=tuple(cat_results),
        numeric_fields=tuple(num_results),
        max_tvd=max((item.tvd for item in cat_results), default=0.0),
        max_jsd=max((item.jsd for item in cat_results), default=0.0),
        max_normalized_wasserstein=max(
            (item.normalized_wasserstein for item in num_results), default=0.0
        ),
    )


def format_distribution_text(report: DistributionReport) -> str:
    """Format a DistributionReport as a concise human-readable text summary."""
    lines = [
        f"Cross-Prompt Field Distribution Drift "
        f"({report.comparable_json_prompts}/{report.total_prompts} comparable JSON prompts)",
        f"Max categorical TVD: {report.max_tvd:.4f} | "
        f"Max categorical JSD: {report.max_jsd:.4f} | "
        f"Max normalized 1D Wasserstein: {report.max_normalized_wasserstein:.4f}",
    ]
    if report.categorical_fields:
        lines.append("")
        lines.append("Categorical fields (sorted by TVD):")
        for cat in report.categorical_fields:
            top = ", ".join(
                f"{lbl}: {b:.1%}->{c:.1%} ({d:+.1%})"
                for lbl, b, c, d in cat.top_shifts[:3]
            )
            lines.append(
                f"  {cat.path} (n={cat.baseline_count}/{cat.candidate_count}): "
                f"TVD={cat.tvd:.4f}, JSD={cat.jsd:.4f} | {top}"
            )
    if report.numeric_fields:
        lines.append("")
        lines.append("Numeric fields (sorted by normalized Wasserstein):")
        for num in report.numeric_fields:
            lines.append(
                f"  {num.path} (n={num.baseline_count}/{num.candidate_count}): "
                f"norm_W1={num.normalized_wasserstein:.4f} (W1={num.wasserstein_1d:.4g}), "
                f"mean {num.baseline_mean:.4g}->{num.candidate_mean:.4g} ({num.mean_delta:+.4g}), "
                f"p10/p50/p90 {num.baseline_p10:.4g}/{num.baseline_p50:.4g}/{num.baseline_p90:.4g}"
                f" -> {num.candidate_p10:.4g}/{num.candidate_p50:.4g}/{num.candidate_p90:.4g}"
            )
    if not report.categorical_fields and not report.numeric_fields:
        lines.append("")
        lines.append("No shared JSON leaf fields found across comparable prompts.")
    return "\n".join(lines) + "\n"


def format_distribution_markdown(report: DistributionReport) -> str:
    """Format a DistributionReport as a safe GitHub-flavored Markdown document."""
    lines = [
        "# Cross-Prompt JSON Field Distribution Drift",
        "",
        f"Comparable JSON prompts: {report.comparable_json_prompts} / {report.total_prompts}",
        "",
        f"- Max categorical TVD: `{report.max_tvd:.4f}`",
        f"- Max categorical JSD: `{report.max_jsd:.4f}`",
        f"- Max normalized 1D Wasserstein: `{report.max_normalized_wasserstein:.4f}`",
    ]
    if report.categorical_fields:
        lines += [
            "",
            "## Categorical fields",
            "",
            "| Path | Count (base/cand) | TVD | JSD | Top label shifts |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for cat in report.categorical_fields:
            shifts_text = "; ".join(
                f"{lbl}: {b:.1%} -> {c:.1%} ({d:+.1%})"
                for lbl, b, c, d in cat.top_shifts[:3]
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        markdown_cell(cat.path),
                        f"{cat.baseline_count}/{cat.candidate_count}",
                        f"{cat.tvd:.4f}",
                        f"{cat.jsd:.4f}",
                        markdown_cell(shifts_text),
                    )
                )
                + " |"
            )
    if report.numeric_fields:
        lines += [
            "",
            "## Numeric fields",
            "",
            "| Path | Count (base/cand) | Norm W1 | W1 | Mean (base -> cand) "
            "| Median (base -> cand) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for num in report.numeric_fields:
            lines.append(
                "| "
                + " | ".join(
                    (
                        markdown_cell(num.path),
                        f"{num.baseline_count}/{num.candidate_count}",
                        f"{num.normalized_wasserstein:.4f}",
                        f"{num.wasserstein_1d:.4g}",
                        f"{num.baseline_mean:.4g} -> {num.candidate_mean:.4g}",
                        f"{num.baseline_p50:.4g} -> {num.candidate_p50:.4g}",
                    )
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def export_distribution_html(
    report: DistributionReport,
    output_path: str | Path | None = None,
) -> str:
    """Export a self-contained, script-free offline HTML field distribution report."""
    def _format_shifts(cat: CategoricalFieldDrift) -> str:
        return "; ".join(
            f"{lbl}: {b:.1%}->{c:.1%} ({d:+.1%})" for lbl, b, c, d in cat.top_shifts[:3]
        )

    cat_rows = "".join(
        "<tr>"
        f"<td>{escape(cat.path)}</td>"
        f"<td>{cat.baseline_count}/{cat.candidate_count}</td>"
        f"<td>{cat.tvd:.4f}</td>"
        f"<td>{cat.jsd:.4f}</td>"
        f"<td>{escape(_format_shifts(cat))}</td>"
        "</tr>"
        for cat in report.categorical_fields
    )
    num_rows = "".join(
        "<tr>"
        f"<td>{escape(num.path)}</td>"
        f"<td>{num.baseline_count}/{num.candidate_count}</td>"
        f"<td>{num.normalized_wasserstein:.4f}</td>"
        f"<td>{num.wasserstein_1d:.4g}</td>"
        f"<td>{num.baseline_mean:.4g} &rarr; {num.candidate_mean:.4g} ({num.mean_delta:+.4g})</td>"
        f"<td>{num.baseline_p10:.4g} / {num.baseline_p50:.4g} / {num.baseline_p90:.4g} &rarr; "
        f"{num.candidate_p10:.4g} / {num.candidate_p50:.4g} / {num.candidate_p90:.4g}</td>"
        "</tr>"
        for num in report.numeric_fields
    )
    cat_section = (
        "<h2>Categorical fields</h2><div class='scroll'><table><thead><tr>"
        "<th>Path</th><th>Count</th><th>TVD</th><th>JSD</th><th>Top shifts</th>"
        f"</tr></thead><tbody>{cat_rows}</tbody></table></div>"
        if cat_rows
        else ""
    )
    num_section = (
        "<h2>Numeric fields</h2><div class='scroll'><table><thead><tr>"
        "<th>Path</th><th>Count</th><th>Norm W1</th><th>W1</th>"
        "<th>Mean</th><th>p10 / p50 / p90</th>"
        f"</tr></thead><tbody>{num_rows}</tbody></table></div>"
        if num_rows
        else ""
    )
    empty_note = (
        "<p class='muted'>No shared JSON leaf fields found across comparable prompts.</p>"
        if not cat_rows and not num_rows
        else ""
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>LLM Drift Bench - Cross-Prompt Field Distribution</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#0b1220;color:#e8eef8;
font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:40px 24px}}
h1{{font-size:2rem;line-height:1.2}}h2{{margin-top:32px}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;background:#111e32}}th,td{{padding:12px;
border-bottom:1px solid #30405a;text-align:left}}th{{color:#7dd3fc}}
td{{white-space:pre-wrap;overflow-wrap:anywhere}}.muted{{color:#a8bad2}}
</style></head><body><main>
<p class="muted">llm-drift-bench / cross-prompt distribution report</p>
<h1>JSON Field Distribution Drift</h1>
<p>Comparable JSON prompts: {report.comparable_json_prompts} / {report.total_prompts}<br>
Max TVD: {report.max_tvd:.4f} &middot; Max JSD: {report.max_jsd:.4f} &middot;
Max Normalized 1D Wasserstein: {report.max_normalized_wasserstein:.4f}</p>
{cat_section}{num_section}{empty_note}
</main></body></html>
"""
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    return html


ShiftScope = Literal["none", "single_segment", "localized", "partial", "widespread"]

_SCOPE_HINTS: dict[ShiftScope, str] = {
    "none": "no segment crossed shift threshold",
    "single_segment": "only one segment present (cannot compare across routes)",
    "localized": "isolated to one segment (likely route/prompt/customer change)",
    "partial": "multiple segments shifted (subset of traffic)",
    "widespread": "most or all segments shifted together (likely model/provider shift)",
}


def _classify_shift_scope(
    segments: dict[str, DistributionReport],
    shift_threshold: float,
) -> tuple[ShiftScope, tuple[str, ...]]:
    shifted_pairs = [
        (name, max(rep.max_tvd, rep.max_normalized_wasserstein))
        for name, rep in segments.items()
        if max(rep.max_tvd, rep.max_normalized_wasserstein) >= shift_threshold
    ]
    shifted_pairs.sort(key=lambda item: (-item[1], item[0]))
    shifted_names = tuple(name for name, _ in shifted_pairs)
    total = len(segments)
    n_shifted = len(shifted_names)
    if n_shifted == 0:
        return "none", shifted_names
    if total == 1:
        return "single_segment", shifted_names
    if n_shifted == 1:
        return "localized", shifted_names
    if (n_shifted / total) >= 0.75:
        return "widespread", shifted_names
    return "partial", shifted_names


@dataclass(frozen=True)
class SegmentedDistributionReport:
    """Distribution drift segmented by route, category, or metadata key before global totals."""

    group_by: str
    overall: DistributionReport
    segments: dict[str, DistributionReport]
    worst_segment_name: str | None
    worst_segment_tvd: float
    worst_segment_jsd: float
    worst_segment_wasserstein: float
    shift_threshold: float = 0.10
    shifted_segments: tuple[str, ...] = ()
    shift_scope: ShiftScope = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_by": self.group_by,
            "shift_scope": self.shift_scope,
            "shift_threshold": round(self.shift_threshold, 6),
            "shifted_segments": list(self.shifted_segments),
            "worst_segment_name": self.worst_segment_name,
            "worst_segment_tvd": round(self.worst_segment_tvd, 6),
            "worst_segment_jsd": round(self.worst_segment_jsd, 6),
            "worst_segment_wasserstein": round(self.worst_segment_wasserstein, 6),
            "segments": {name: rep.to_dict() for name, rep in self.segments.items()},
            "overall": self.overall.to_dict(),
        }


def _parse_group_by(group_by: str) -> tuple[str, str | None]:
    if not isinstance(group_by, str) or not group_by.strip():
        raise ValueError("group_by must be 'category' or 'metadata:<key>'.")
    cleaned = group_by.strip()
    if cleaned == "category":
        return "category", None
    if cleaned.startswith("metadata:"):
        key = cleaned[len("metadata:") :].strip()
        if not key:
            raise ValueError("group_by='metadata:<key>' requires a non-empty metadata key.")
        return "metadata", key
    raise ValueError(
        f"Unsupported group_by {group_by!r}. Use 'category' or 'metadata:<key>'."
    )


def analyze_segmented_distributions(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
    *,
    group_by: str = "category",
    prompt_cases: Sequence[PromptCase] | None = None,
    policy: DriftPolicy | None = None,
    collapse_array_indices: bool = True,
    shift_threshold: float = 0.10,
) -> SegmentedDistributionReport:
    """Segment prompts by category or metadata key before computing overall distribution drift.

    Isolates shifts confined to a single route, prompt category, or customer segment
    before global aggregation can dilute them, and classifies blast radius as
    none / single_segment / localized / partial / widespread.
    """
    mode, meta_key = _parse_group_by(group_by)
    if (
        isinstance(shift_threshold, bool)
        or not isinstance(shift_threshold, (int, float))
        or not math.isfinite(shift_threshold)
        or not 0.0 < shift_threshold <= 1.0
    ):
        raise ValueError("shift_threshold must be a finite number in (0.0, 1.0].")
    if not baseline_samples or not candidate_samples:
        raise ValueError("Both baseline and candidate sample collections must be non-empty.")

    base_map = index_records(baseline_samples, "baseline samples")
    cand_map = index_records(candidate_samples, "candidate samples")
    require_same_ids(set(base_map), set(cand_map), "Candidate")

    case_map: dict[str, PromptCase] = {}
    if prompt_cases is not None:
        if not prompt_cases:
            raise ValueError("Prompt suite cannot be empty.")
        case_map = index_records(prompt_cases, "prompt suite")
        require_same_ids(set(base_map), set(case_map), "Prompt suite")

    buckets_base: dict[str, list[ModelSample]] = defaultdict(list)
    buckets_cand: dict[str, list[ModelSample]] = defaultdict(list)

    for pid in sorted(base_map):
        b_sample = base_map[pid]
        c_sample = cand_map[pid]
        raw_seg: Any = None
        if mode == "category":
            if pid in case_map and case_map[pid].category:
                raw_seg = case_map[pid].category
            else:
                raw_seg = b_sample.metadata.get("category")
        else:
            assert meta_key is not None
            raw_seg = b_sample.metadata.get(meta_key)
            if raw_seg is None and pid in case_map:
                raw_seg = case_map[pid].metadata.get(meta_key)

        seg_label = (
            str(raw_seg).strip()
            if raw_seg is not None and str(raw_seg).strip()
            else "(unassigned)"
        )
        buckets_base[seg_label].append(b_sample)
        buckets_cand[seg_label].append(c_sample)

    segments: dict[str, DistributionReport] = {}
    for seg_name in sorted(buckets_base):
        segments[seg_name] = analyze_field_distributions(
            buckets_base[seg_name],
            buckets_cand[seg_name],
            policy=policy,
            collapse_array_indices=collapse_array_indices,
        )

    overall = analyze_field_distributions(
        baseline_samples,
        candidate_samples,
        policy=policy,
        collapse_array_indices=collapse_array_indices,
    )

    worst_name: str | None = None
    worst_score = -1.0
    for name, rep in segments.items():
        score = max(rep.max_tvd, rep.max_normalized_wasserstein)
        if score > worst_score:
            worst_score = score
            worst_name = name

    scope, shifted_names = _classify_shift_scope(segments, float(shift_threshold))

    return SegmentedDistributionReport(
        group_by=group_by.strip(),
        overall=overall,
        segments=segments,
        worst_segment_name=worst_name,
        worst_segment_tvd=max((r.max_tvd for r in segments.values()), default=0.0),
        worst_segment_jsd=max((r.max_jsd for r in segments.values()), default=0.0),
        worst_segment_wasserstein=max(
            (r.max_normalized_wasserstein for r in segments.values()), default=0.0
        ),
        shift_threshold=float(shift_threshold),
        shifted_segments=shifted_names,
        shift_scope=scope,
    )


def format_segmented_distribution_text(report: SegmentedDistributionReport) -> str:
    """Format a SegmentedDistributionReport showing per-segment shifts before overall totals."""
    scope_hint = _SCOPE_HINTS[report.shift_scope]
    shifted_list = (
        f" [{', '.join(report.shifted_segments)}]" if report.shifted_segments else ""
    )
    lines = [
        f"Segmented Field Distribution Drift (group_by={report.group_by}, "
        f"{len(report.segments)} segment(s))",
        f"Blast radius : {report.shift_scope} "
        f"({len(report.shifted_segments)}/{len(report.segments)} segments "
        f">= {report.shift_threshold:.2f}{shifted_list} - {scope_hint})",
        f"Worst segment: {report.worst_segment_name or 'n/a'} | "
        f"max TVD={report.worst_segment_tvd:.4f}, "
        f"max JSD={report.worst_segment_jsd:.4f}, "
        f"max norm W1={report.worst_segment_wasserstein:.4f}",
        "",
        "Per-segment breakdown (sorted by worst shift):",
    ]
    ordered = sorted(
        report.segments.items(),
        key=lambda item: (
            -max(item[1].max_tvd, item[1].max_normalized_wasserstein),
            item[0],
        ),
    )
    for name, seg in ordered:
        lines.append(
            f"  [{name}] ({seg.comparable_json_prompts}/{seg.total_prompts} JSON): "
            f"max TVD={seg.max_tvd:.4f}, max JSD={seg.max_jsd:.4f}, "
            f"max norm W1={seg.max_normalized_wasserstein:.4f}"
        )
    lines.append("")
    lines.append("Overall totals:")
    lines.append(format_distribution_text(report.overall).rstrip("\n"))
    return "\n".join(lines) + "\n"


def format_segmented_distribution_markdown(report: SegmentedDistributionReport) -> str:
    """Format a SegmentedDistributionReport as GitHub-flavored Markdown."""
    scope_hint = _SCOPE_HINTS[report.shift_scope]
    lines = [
        f"# Segmented Field Distribution Drift (`{markdown_cell(report.group_by)}`)",
        "",
        f"Blast radius: **{markdown_cell(report.shift_scope)}** "
        f"({len(report.shifted_segments)}/{len(report.segments)} segments "
        f"&gt;= `{report.shift_threshold:.2f}`: {markdown_cell(scope_hint)})",
        "",
        "| Segment | JSON Prompts | Max TVD | Max JSD | Max Norm W1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(
        report.segments.items(),
        key=lambda item: (
            -max(item[1].max_tvd, item[1].max_normalized_wasserstein),
            item[0],
        ),
    )
    for name, seg in ordered:
        lines.append(
            "| "
            + " | ".join(
                (
                    markdown_cell(name),
                    f"{seg.comparable_json_prompts}/{seg.total_prompts}",
                    f"{seg.max_tvd:.4f}",
                    f"{seg.max_jsd:.4f}",
                    f"{seg.max_normalized_wasserstein:.4f}",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Overall totals",
        "",
        format_distribution_markdown(report.overall).rstrip("\n"),
    ]
    return "\n".join(lines) + "\n"


def export_segmented_distribution_html(
    report: SegmentedDistributionReport,
    output_path: str | Path | None = None,
) -> str:
    """Export a standalone script-free offline HTML segmented distribution report."""
    ordered = sorted(
        report.segments.items(),
        key=lambda item: (
            -max(item[1].max_tvd, item[1].max_normalized_wasserstein),
            item[0],
        ),
    )
    seg_rows = "".join(
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{seg.comparable_json_prompts}/{seg.total_prompts}</td>"
        f"<td>{seg.max_tvd:.4f}</td>"
        f"<td>{seg.max_jsd:.4f}</td>"
        f"<td>{seg.max_normalized_wasserstein:.4f}</td>"
        "</tr>"
        for name, seg in ordered
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>LLM Drift Bench - Segmented Field Distribution</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#0b1220;color:#e8eef8;
font:16px/1.6 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:40px 24px}}
h1{{font-size:2rem;line-height:1.2}}h2{{margin-top:32px}}.scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;background:#111e32}}th,td{{padding:12px;
border-bottom:1px solid #30405a;text-align:left}}th{{color:#7dd3fc}}
td{{white-space:pre-wrap;overflow-wrap:anywhere}}.muted{{color:#a8bad2}}
</style></head><body><main>
<p class="muted">llm-drift-bench / segmented distribution report</p>
<h1>Segmented JSON Field Distribution ({escape(report.group_by)})</h1>
<p>Segments: {len(report.segments)} &middot;
 Blast radius: <strong>{escape(report.shift_scope)}</strong>
 ({len(report.shifted_segments)}/{len(report.segments)} &ge; {report.shift_threshold:.2f};
 {escape(_SCOPE_HINTS[report.shift_scope])})<br>
 Worst segment: {escape(report.worst_segment_name or 'n/a')} &middot;
 Worst TVD: {report.worst_segment_tvd:.4f} &middot;
 Worst Norm W1: {report.worst_segment_wasserstein:.4f}</p>
<h2>Per-Segment Breakdown</h2>
<div class="scroll"><table><thead><tr>
<th>Segment</th><th>JSON Prompts</th><th>Max TVD</th><th>Max JSD</th><th>Max Norm W1</th>
</tr></thead><tbody>{seg_rows}</tbody></table></div>
<h2>Overall Totals</h2>
<p>Comparable JSON prompts: {report.overall.comparable_json_prompts} /
 {report.overall.total_prompts}<br>
Overall Max TVD: {report.overall.max_tvd:.4f} &middot;
Overall Max JSD: {report.overall.max_jsd:.4f} &middot;
Overall Max Norm W1: {report.overall.max_normalized_wasserstein:.4f}</p>
</main></body></html>
"""
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    return html

