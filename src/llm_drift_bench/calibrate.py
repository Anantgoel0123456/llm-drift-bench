"""Automatic threshold calibration and volatile JSON field detection."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llm_drift_bench._matching import require_same_ids
from llm_drift_bench.metrics import _values_equal, flatten_json_paths
from llm_drift_bench.models import DriftBenchmarkReport, DriftThresholds, ModelSample
from llm_drift_bench.validator import extract_json_payload


@dataclass(frozen=True)
class CalibrationResult:
    """Calibrated drift.toml recommendations derived from baseline noise runs."""

    recommended_thresholds: DriftThresholds
    volatile_fields: tuple[str, ...]
    observed_max_drift: float
    observed_min_conformity: float
    observed_max_numeric_drift: float | None
    observed_max_latency_regression_pct: float | None
    headroom: float
    runs_analyzed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibration summary to a JSON-compatible dictionary."""
        return {
            "runs_analyzed": self.runs_analyzed,
            "headroom": self.headroom,
            "observed": {
                "max_composite_drift": self.observed_max_drift,
                "min_json_conformity_rate": self.observed_min_conformity,
                "max_numeric_relative_drift": self.observed_max_numeric_drift,
                "max_latency_regression_pct": self.observed_max_latency_regression_pct,
            },
            "recommended_thresholds": {
                "max_composite_drift": self.recommended_thresholds.max_composite_drift,
                "min_json_conformity_rate": self.recommended_thresholds.min_json_conformity_rate,
                "max_numeric_relative_drift": (
                    self.recommended_thresholds.max_numeric_relative_drift
                ),
                "max_latency_regression_pct": (
                    self.recommended_thresholds.max_latency_regression_pct
                ),
            },
            "volatile_fields": list(self.volatile_fields),
        }


def detect_volatile_fields(
    baseline_samples: Sequence[ModelSample],
    repeat_samples: Sequence[ModelSample],
) -> list[str]:
    """Return sorted JSONPath leaf keys that changed across repeated runs of the same model."""
    if not baseline_samples or not repeat_samples:
        raise ValueError("Both baseline and repeat sample collections must be non-empty.")

    base_map = {sample.prompt_id: sample for sample in baseline_samples}
    repeat_map = {sample.prompt_id: sample for sample in repeat_samples}
    if len(base_map) != len(baseline_samples) or len(repeat_map) != len(repeat_samples):
        raise ValueError("Duplicate prompt_id found while detecting volatile fields.")
    require_same_ids(set(base_map), set(repeat_map), "Repeat capture")

    volatile: set[str] = set()
    for prompt_id in sorted(base_map):
        ok_a, payload_a, _ = extract_json_payload(base_map[prompt_id].output)
        ok_b, payload_b, _ = extract_json_payload(repeat_map[prompt_id].output)
        if not (ok_a and ok_b):
            continue
        flat_a = flatten_json_paths(payload_a)
        flat_b = flatten_json_paths(payload_b)
        for path in set(flat_a) | set(flat_b):
            if (
                path not in flat_a
                or path not in flat_b
                or not _values_equal(flat_a[path], flat_b[path])
            ):
                volatile.add(path)
    return sorted(volatile)


def calibrate_from_reports(
    reports: Sequence[DriftBenchmarkReport],
    *,
    headroom: float = 1.25,
    min_floor_drift: float = 0.05,
    volatile_fields: Sequence[str] = (),
) -> CalibrationResult:
    """Recommend DriftThresholds with headroom from one or more baseline noise reports."""
    if not reports:
        raise ValueError("At least one benchmark report is required for calibration.")
    if (
        isinstance(headroom, bool)
        or not isinstance(headroom, (int, float))
        or not math.isfinite(headroom)
        or headroom < 1.0
    ):
        raise ValueError("headroom must be a finite number >= 1.0.")
    if (
        isinstance(min_floor_drift, bool)
        or not isinstance(min_floor_drift, (int, float))
        or not math.isfinite(min_floor_drift)
        or not (0.0 <= min_floor_drift <= 1.0)
    ):
        raise ValueError("min_floor_drift must be a finite number in [0.0, 1.0].")

    prompt_drifts = [row.composite_drift_score for r in reports for row in r.results]
    obs_max_drift = round(max(prompt_drifts) if prompt_drifts else 0.0, 4)
    obs_min_conf = round(min(r.candidate_json_conformity_rate for r in reports), 4)

    num_drifts = [
        row.json_numeric_drift
        for r in reports
        for row in r.results
        if row.json_numeric_drift is not None
    ]
    obs_max_num = round(max(num_drifts), 4) if num_drifts else None

    lat_regs: list[float] = []
    for report in reports:
        base_p95 = report.baseline_p95_latency_ms
        cand_p95 = report.candidate_p95_latency_ms
        if base_p95 is not None and cand_p95 is not None and base_p95 > 0:
            lat_regs.append(((cand_p95 - base_p95) / base_p95) * 100.0)
    obs_max_lat = round(max(lat_regs), 2) if lat_regs else None

    rec_drift = round(min(1.0, max(float(min_floor_drift), obs_max_drift * headroom)), 4)
    failure_rate = max(0.0, 1.0 - obs_min_conf)
    rec_conf = round(max(0.0, 1.0 - failure_rate * headroom), 4)
    rec_num = (
        round(min(1.0, max(0.01, obs_max_num * headroom)), 4)
        if obs_max_num is not None
        else 0.15
    )
    rec_lat = (
        round(max(10.0, max(0.0, obs_max_lat) * headroom), 2)
        if obs_max_lat is not None
        else 50.0
    )

    thresholds = DriftThresholds(
        max_composite_drift=rec_drift,
        min_json_conformity_rate=rec_conf,
        max_numeric_relative_drift=rec_num,
        max_latency_regression_pct=rec_lat,
    )
    cleaned_volatile = tuple(sorted({p for p in volatile_fields if p.startswith("$")}))
    return CalibrationResult(
        recommended_thresholds=thresholds,
        volatile_fields=cleaned_volatile,
        observed_max_drift=obs_max_drift,
        observed_min_conformity=obs_min_conf,
        observed_max_numeric_drift=obs_max_num,
        observed_max_latency_regression_pct=obs_max_lat,
        headroom=float(headroom),
        runs_analyzed=len(reports),
    )


def render_calibrated_toml(result: CalibrationResult) -> str:
    """Render a ready-to-use drift.toml policy file from a CalibrationResult."""
    t = result.recommended_thresholds
    lines = [
        "# Auto-calibrated llm-drift-bench policy",
        f"# Runs analyzed: {result.runs_analyzed} | Headroom multiplier: {result.headroom:.2f}x",
        f"# Observed max prompt drift: {result.observed_max_drift:.4f}",
        f"# Observed min JSON conformity: {result.observed_min_conformity:.4f}",
        'long_text_mode = "error"',
        "",
        "[thresholds]",
        f"max_composite_drift = {t.max_composite_drift}",
        f"min_json_conformity_rate = {t.min_json_conformity_rate}",
        f"max_numeric_relative_drift = {t.max_numeric_relative_drift}",
        f"max_latency_regression_pct = {t.max_latency_regression_pct}",
    ]
    if result.volatile_fields:
        lines.append("")
        lines.append("# Volatile fields that changed across identical-model runs")
        for path in result.volatile_fields:
            lines.append(f"[fields.{json.dumps(path, ensure_ascii=True)}]")
            lines.append("ignore = true")
            lines.append("")
    else:
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
