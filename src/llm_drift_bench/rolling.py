"""Rolling-baseline cross-window JSON field distribution drift analysis.

Compares a candidate capture window against a rolling pool of prior traffic windows
(which need not share identical prompt IDs or sample counts), measures the rolling
baseline's own window-to-window natural jitter, and supports optional route/category
segmentation with localized-vs-widespread blast-radius classification.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llm_drift_bench._numbers import finite_number
from llm_drift_bench._report_views import markdown_cell
from llm_drift_bench.distribution import (
    CategoricalFieldDrift,
    DistributionReport,
    NumericFieldDrift,
    SegmentedDistributionReport,
    _canonical_category,
    _classify_shift_scope,
    _collapse_array_indices,
    _parse_group_by,
    _quantile,
    _tvd_and_jsd,
    _wasserstein_1d,
)
from llm_drift_bench.metrics import flatten_json_paths
from llm_drift_bench.models import ModelSample, PromptCase
from llm_drift_bench.policy import DriftPolicy, path_is_ignored
from llm_drift_bench.validator import extract_json_payload


def _validate_unit_limit(name: str, val: float | None) -> None:
    if val is not None and (
        isinstance(val, bool)
        or not isinstance(val, (int, float))
        or not math.isfinite(val)
        or not 0.0 <= val <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between 0.0 and 1.0.")


def _extract_fields(
    samples: Sequence[ModelSample],
    *,
    ignored: tuple[str, ...],
    collapse_array_indices: bool,
) -> tuple[int, dict[str, list[float]], dict[str, Counter[str]]]:
    nums: dict[str, list[float]] = defaultdict(list)
    cats: dict[str, Counter[str]] = defaultdict(Counter)
    json_count = 0
    for sample in samples:
        ok, payload, _ = extract_json_payload(sample.output)
        if not ok:
            continue
        json_count += 1
        for raw_path, val in flatten_json_paths(payload).items():
            if ignored and path_is_ignored(raw_path, ignored):
                continue
            key = _collapse_array_indices(raw_path) if collapse_array_indices else raw_path
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                nums[key].append(finite_number(val))
            else:
                cats[key][_canonical_category(val)] += 1
    return json_count, nums, cats


def _analyze_unpaired_distribution(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
    *,
    policy: DriftPolicy | None = None,
    collapse_array_indices: bool = True,
) -> DistributionReport:
    """Compare empirical field distributions between two unpaired capture windows."""
    ignored = (
        tuple(path for path, rule in policy.field_rules.items() if rule.ignore)
        if policy is not None
        else ()
    )
    b_json, base_nums, base_cats = _extract_fields(
        baseline_samples, ignored=ignored, collapse_array_indices=collapse_array_indices
    )
    c_json, cand_nums, cand_cats = _extract_fields(
        candidate_samples, ignored=ignored, collapse_array_indices=collapse_array_indices
    )

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
        total_prompts=len(candidate_samples),
        comparable_json_prompts=min(b_json, c_json),
        categorical_fields=tuple(cat_results),
        numeric_fields=tuple(num_results),
        max_tvd=max((item.tvd for item in cat_results), default=0.0),
        max_jsd=max((item.jsd for item in cat_results), default=0.0),
        max_normalized_wasserstein=max(
            (item.normalized_wasserstein for item in num_results), default=0.0
        ),
    )


def _sample_segment(
    sample: ModelSample,
    *,
    mode: str,
    meta_key: str | None,
    case_map: dict[str, PromptCase],
) -> str:
    raw: Any = None
    if mode == "category":
        case = case_map.get(sample.prompt_id)
        if case is not None and case.category:
            raw = case.category
        else:
            raw = sample.metadata.get("category")
    else:
        assert meta_key is not None
        raw = sample.metadata.get(meta_key)
        if raw is None and sample.prompt_id in case_map:
            raw = case_map[sample.prompt_id].metadata.get(meta_key)
    return str(raw).strip() if raw is not None and str(raw).strip() else "(unassigned)"


@dataclass(frozen=True)
class RollingWindowReport:
    """Distribution shift of a candidate window vs a rolling pool of prior baseline windows."""

    window_count: int
    window_labels: tuple[str, ...]
    baseline_sample_count: int
    candidate_sample_count: int
    overall: DistributionReport
    segmented: SegmentedDistributionReport | None
    per_window_max_tvd: dict[str, float]
    per_window_max_wasserstein: dict[str, float]
    baseline_jitter_tvd: float | None
    baseline_jitter_wasserstein: float | None
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "window_count": self.window_count,
            "window_labels": list(self.window_labels),
            "baseline_sample_count": self.baseline_sample_count,
            "candidate_sample_count": self.candidate_sample_count,
            "baseline_jitter_tvd": (
                round(self.baseline_jitter_tvd, 6)
                if self.baseline_jitter_tvd is not None
                else None
            ),
            "baseline_jitter_wasserstein": (
                round(self.baseline_jitter_wasserstein, 6)
                if self.baseline_jitter_wasserstein is not None
                else None
            ),
            "per_window_max_tvd": {
                k: round(v, 6) for k, v in self.per_window_max_tvd.items()
            },
            "per_window_max_wasserstein": {
                k: round(v, 6) for k, v in self.per_window_max_wasserstein.items()
            },
            "overall": self.overall.to_dict(),
        }
        if self.segmented is not None:
            payload["segmented"] = self.segmented.to_dict()
        return payload


def analyze_rolling_distribution(
    baseline_windows: Sequence[Sequence[ModelSample]],
    candidate_samples: Sequence[ModelSample],
    *,
    window_labels: Sequence[str] | None = None,
    group_by: str | None = None,
    prompt_cases: Sequence[PromptCase] | None = None,
    policy: DriftPolicy | None = None,
    collapse_array_indices: bool = True,
    shift_threshold: float = 0.10,
    max_tvd: float | None = None,
    max_jsd: float | None = None,
    max_wasserstein: float | None = None,
    fail_on_widespread_shift: bool = False,
    max_jitter_multiplier: float | None = None,
) -> RollingWindowReport:
    """Compare a candidate window against a rolling baseline of prior traffic windows."""
    if not baseline_windows:
        raise ValueError("baseline_windows must contain at least one window.")
    if not candidate_samples:
        raise ValueError("candidate_samples must be non-empty.")
    for idx, win in enumerate(baseline_windows, start=1):
        if not win:
            raise ValueError(f"Baseline window #{idx} is empty.")

    if window_labels is not None:
        if len(window_labels) != len(baseline_windows):
            raise ValueError(
                f"window_labels length ({len(window_labels)}) must match "
                f"baseline_windows length ({len(baseline_windows)})."
            )
        labels = tuple(lbl.strip() for lbl in window_labels)
        if any(not lbl for lbl in labels) or len(set(labels)) != len(labels):
            raise ValueError("window_labels must be unique non-empty strings.")
    else:
        labels = tuple(f"window_{i}" for i in range(1, len(baseline_windows) + 1))

    _validate_unit_limit("max_tvd", max_tvd)
    _validate_unit_limit("max_jsd", max_jsd)
    _validate_unit_limit("max_wasserstein", max_wasserstein)
    if (
        isinstance(shift_threshold, bool)
        or not isinstance(shift_threshold, (int, float))
        or not math.isfinite(shift_threshold)
        or not 0.0 < shift_threshold <= 1.0
    ):
        raise ValueError("shift_threshold must be a finite number in (0.0, 1.0].")
    if max_jitter_multiplier is not None and (
        isinstance(max_jitter_multiplier, bool)
        or not isinstance(max_jitter_multiplier, (int, float))
        or not math.isfinite(max_jitter_multiplier)
        or max_jitter_multiplier <= 0.0
    ):
        raise ValueError("max_jitter_multiplier must be a finite positive number.")
    if fail_on_widespread_shift and group_by is None:
        raise ValueError("fail_on_widespread_shift requires group_by.")

    pooled_baseline: list[ModelSample] = [s for win in baseline_windows for s in win]
    overall = _analyze_unpaired_distribution(
        pooled_baseline,
        candidate_samples,
        policy=policy,
        collapse_array_indices=collapse_array_indices,
    )

    per_win_tvd: dict[str, float] = {}
    per_win_w1: dict[str, float] = {}
    for lbl, win in zip(labels, baseline_windows, strict=True):
        win_rep = _analyze_unpaired_distribution(
            win,
            candidate_samples,
            policy=policy,
            collapse_array_indices=collapse_array_indices,
        )
        per_win_tvd[lbl] = win_rep.max_tvd
        per_win_w1[lbl] = win_rep.max_normalized_wasserstein

    jitter_tvd: float | None = None
    jitter_w1: float | None = None
    if len(baseline_windows) >= 2:
        pair_tvds: list[float] = []
        pair_w1s: list[float] = []
        for prev_win, next_win in zip(baseline_windows, baseline_windows[1:], strict=False):
            pair_rep = _analyze_unpaired_distribution(
                prev_win,
                next_win,
                policy=policy,
                collapse_array_indices=collapse_array_indices,
            )
            pair_tvds.append(pair_rep.max_tvd)
            pair_w1s.append(pair_rep.max_normalized_wasserstein)
        jitter_tvd = max(pair_tvds)
        jitter_w1 = max(pair_w1s)

    segmented: SegmentedDistributionReport | None = None
    if group_by is not None:
        mode, meta_key = _parse_group_by(group_by)
        case_map = {c.prompt_id: c for c in prompt_cases} if prompt_cases else {}
        base_buckets: dict[str, list[ModelSample]] = defaultdict(list)
        cand_buckets: dict[str, list[ModelSample]] = defaultdict(list)
        for sample in pooled_baseline:
            seg = _sample_segment(sample, mode=mode, meta_key=meta_key, case_map=case_map)
            base_buckets[seg].append(sample)
        for sample in candidate_samples:
            seg = _sample_segment(sample, mode=mode, meta_key=meta_key, case_map=case_map)
            cand_buckets[seg].append(sample)

        shared_segments = sorted(set(base_buckets) & set(cand_buckets))
        seg_reports: dict[str, DistributionReport] = {
            seg: _analyze_unpaired_distribution(
                base_buckets[seg],
                cand_buckets[seg],
                policy=policy,
                collapse_array_indices=collapse_array_indices,
            )
            for seg in shared_segments
        }
        worst_name: str | None = None
        if seg_reports:
            worst_name = max(
                seg_reports,
                key=lambda k: (
                    max(seg_reports[k].max_tvd, seg_reports[k].max_normalized_wasserstein),
                    seg_reports[k].max_jsd,
                    k,
                ),
            )
        scope, shifted_names = _classify_shift_scope(seg_reports, float(shift_threshold))
        segmented = SegmentedDistributionReport(
            group_by=group_by.strip(),
            overall=overall,
            segments=seg_reports,
            worst_segment_name=worst_name,
            worst_segment_tvd=max((r.max_tvd for r in seg_reports.values()), default=0.0),
            worst_segment_jsd=max((r.max_jsd for r in seg_reports.values()), default=0.0),
            worst_segment_wasserstein=max(
                (r.max_normalized_wasserstein for r in seg_reports.values()), default=0.0
            ),
            shift_threshold=float(shift_threshold),
            shifted_segments=shifted_names,
            shift_scope=scope,
        )

    failures: list[str] = []
    if segmented is not None:
        for seg_name, seg_rep in segmented.segments.items():
            if max_tvd is not None and seg_rep.max_tvd > max_tvd:
                failures.append(
                    f"[{seg_name}] max_tvd {seg_rep.max_tvd:.4f} exceeded limit {max_tvd:.4f}"
                )
            if max_jsd is not None and seg_rep.max_jsd > max_jsd:
                failures.append(
                    f"[{seg_name}] max_jsd {seg_rep.max_jsd:.4f} exceeded limit {max_jsd:.4f}"
                )
            if (
                max_wasserstein is not None
                and seg_rep.max_normalized_wasserstein > max_wasserstein
            ):
                failures.append(
                    f"[{seg_name}] max_normalized_wasserstein "
                    f"{seg_rep.max_normalized_wasserstein:.4f} exceeded limit "
                    f"{max_wasserstein:.4f}"
                )
        if fail_on_widespread_shift and segmented.shift_scope == "widespread":
            shifted_str = ", ".join(segmented.shifted_segments)
            failures.append(
                f"Widespread rolling shift across {len(segmented.shifted_segments)}/"
                f"{len(segmented.segments)} segments ({shifted_str})."
            )
    else:
        if max_tvd is not None and overall.max_tvd > max_tvd:
            failures.append(f"max_tvd {overall.max_tvd:.4f} exceeded limit {max_tvd:.4f}")
        if max_jsd is not None and overall.max_jsd > max_jsd:
            failures.append(f"max_jsd {overall.max_jsd:.4f} exceeded limit {max_jsd:.4f}")
        if max_wasserstein is not None and overall.max_normalized_wasserstein > max_wasserstein:
            failures.append(
                f"max_normalized_wasserstein {overall.max_normalized_wasserstein:.4f} "
                f"exceeded limit {max_wasserstein:.4f}"
            )

    if max_jitter_multiplier is not None:
        if jitter_tvd is not None and jitter_tvd > 0.0:
            limit_tvd = jitter_tvd * max_jitter_multiplier
            if overall.max_tvd > limit_tvd:
                failures.append(
                    f"max_tvd {overall.max_tvd:.4f} exceeded {max_jitter_multiplier:.2f}x "
                    f"rolling baseline jitter ({jitter_tvd:.4f}, limit {limit_tvd:.4f})"
                )
        if jitter_w1 is not None and jitter_w1 > 0.0:
            limit_w1 = jitter_w1 * max_jitter_multiplier
            if overall.max_normalized_wasserstein > limit_w1:
                failures.append(
                    f"max_normalized_wasserstein {overall.max_normalized_wasserstein:.4f} "
                    f"exceeded {max_jitter_multiplier:.2f}x rolling baseline jitter "
                    f"({jitter_w1:.4f}, limit {limit_w1:.4f})"
                )

    return RollingWindowReport(
        window_count=len(baseline_windows),
        window_labels=labels,
        baseline_sample_count=len(pooled_baseline),
        candidate_sample_count=len(candidate_samples),
        overall=overall,
        segmented=segmented,
        per_window_max_tvd=per_win_tvd,
        per_window_max_wasserstein=per_win_w1,
        baseline_jitter_tvd=jitter_tvd,
        baseline_jitter_wasserstein=jitter_w1,
        passed=len(failures) == 0,
        failure_reasons=tuple(failures),
    )


def format_rolling_text(report: RollingWindowReport) -> str:
    """Format a RollingWindowReport as a concise terminal summary."""
    status = "PASSED" if report.passed else "FAILED"
    jit_tvd = "n/a" if report.baseline_jitter_tvd is None else f"{report.baseline_jitter_tvd:.4f}"
    jit_w1 = (
        "n/a"
        if report.baseline_jitter_wasserstein is None
        else f"{report.baseline_jitter_wasserstein:.4f}"
    )
    lines = [
        f"Rolling Baseline Distribution Drift [{status}] "
        f"({report.window_count} baseline window(s), "
        f"{report.baseline_sample_count} baseline vs {report.candidate_sample_count} candidate)",
        f"  Pooled shift    : max_tvd={report.overall.max_tvd:.4f}, "
        f"max_jsd={report.overall.max_jsd:.4f}, "
        f"max_norm_w1={report.overall.max_normalized_wasserstein:.4f}",
        f"  Baseline jitter : tvd={jit_tvd}, norm_w1={jit_w1}",
    ]
    if report.segmented is not None:
        lines.append(
            f"  Segments ({report.segmented.group_by}) : scope={report.segmented.shift_scope}, "
            f"worst={report.segmented.worst_segment_name or 'n/a'} "
            f"(tvd={report.segmented.worst_segment_tvd:.4f}, "
            f"w1={report.segmented.worst_segment_wasserstein:.4f})"
        )
    if report.failure_reasons:
        lines.append("")
        lines.append("Threshold failures:")
        for msg in report.failure_reasons:
            lines.append(f"  - {msg}")
    return "\n".join(lines) + "\n"


def format_rolling_markdown(report: RollingWindowReport) -> str:
    """Format a RollingWindowReport as GitHub-flavored Markdown."""
    status = "PASSED" if report.passed else "FAILED"
    jit_tvd = "n/a" if report.baseline_jitter_tvd is None else f"{report.baseline_jitter_tvd:.4f}"
    jit_w1 = (
        "n/a"
        if report.baseline_jitter_wasserstein is None
        else f"{report.baseline_jitter_wasserstein:.4f}"
    )
    lines = [
        f"# Rolling Baseline Distribution Drift: {status}",
        "",
        f"- **Baseline windows**: {report.window_count} ({report.baseline_sample_count} samples)",
        f"- **Candidate window**: {report.candidate_sample_count} samples",
        f"- **Pooled shift**: max TVD `{report.overall.max_tvd:.4f}`, "
        f"max JSD `{report.overall.max_jsd:.4f}`, "
        f"max norm W1 `{report.overall.max_normalized_wasserstein:.4f}`",
        f"- **Natural baseline jitter**: TVD `{jit_tvd}`, norm W1 `{jit_w1}`",
        "",
        "| Baseline Window | Candidate vs Window Max TVD | Candidate vs Window Max Norm W1 |",
        "| --- | ---: | ---: |",
    ]
    for lbl in report.window_labels:
        lines.append(
            f"| {markdown_cell(lbl)} | {report.per_window_max_tvd[lbl]:.4f} "
            f"| {report.per_window_max_wasserstein[lbl]:.4f} |"
        )
    if report.segmented is not None and report.segmented.segments:
        lines += [
            "",
            f"## Segments (`{markdown_cell(report.segmented.group_by)}`, "
            f"scope: `{report.segmented.shift_scope}`)",
            "",
            "| Segment | Max TVD | Max JSD | Max Norm W1 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for seg_name, seg in report.segmented.segments.items():
            lines.append(
                f"| {markdown_cell(seg_name)} | {seg.max_tvd:.4f} "
                f"| {seg.max_jsd:.4f} | {seg.max_normalized_wasserstein:.4f} |"
            )
    if report.failure_reasons:
        lines += ["", "## Threshold failures", ""]
        lines.extend(f"- {markdown_cell(msg)}" for msg in report.failure_reasons)
    return "\n".join(lines) + "\n"
