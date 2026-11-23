"""Tests for threshold auto-calibration, volatile field detection, and calibrate CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.calibrate import (
    calibrate_from_reports,
    detect_volatile_fields,
    render_calibrated_toml,
)
from llm_drift_bench.cli import main
from llm_drift_bench.engine import DriftBenchmarkEngine
from llm_drift_bench.models import ModelSample
from llm_drift_bench.policy import _parse_toml_fallback, load_policy


def _make_samples(
    payloads: dict[str, str],
    *,
    model: str = "llama-3-8b",
    latency: float | None = 100.0,
) -> list[ModelSample]:
    return [
        ModelSample(prompt_id=pid, output=text, model_name=model, latency_ms=latency)
        for pid, text in payloads.items()
    ]


def test_detect_volatile_fields_identifies_changing_and_missing_paths() -> None:
    run1 = _make_samples({
        "p1": json.dumps({"answer": 42, "timestamp": "2026-01-01T00:00:00Z", "meta": {"id": "a"}}),
        "p2": "Plain text output without JSON",
    })
    run2 = _make_samples({
        "p1": json.dumps({
            "answer": 42,
            "timestamp": "2026-01-01T00:00:05Z",
            "meta": {"id": "b"},
            "trace": "xyz",
        }),
        "p2": "Different plain text",
    })

    volatile = detect_volatile_fields(run1, run2)
    assert volatile == ["$.meta.id", "$.timestamp", "$.trace"]


def test_detect_volatile_fields_rejects_invalid_inputs() -> None:
    sample = ModelSample(prompt_id="p1", output='{"a": 1}', model_name="m")
    with pytest.raises(ValueError, match="non-empty"):
        detect_volatile_fields([], [sample])
    with pytest.raises(ValueError, match="Duplicate"):
        detect_volatile_fields([sample, sample], [sample])
    other = ModelSample(prompt_id="p2", output='{"a": 1}', model_name="m")
    with pytest.raises(ValueError, match="Repeat capture"):
        detect_volatile_fields([sample], [other])


def test_calibrate_from_reports_computes_headroom_and_bounds() -> None:
    base = _make_samples({"p1": '{"total": 100, "tag": "ok"}'}, latency=100.0)
    cand = _make_samples({"p1": '{"total": 108, "tag": "ok"}'}, latency=120.0)
    report = DriftBenchmarkEngine().evaluate(base, cand)

    cal = calibrate_from_reports([report], headroom=1.5, min_floor_drift=0.05)
    assert cal.runs_analyzed == 1
    assert cal.observed_max_drift == round(report.results[0].composite_drift_score, 4)
    assert cal.recommended_thresholds.max_composite_drift >= 0.05
    assert cal.recommended_thresholds.min_json_conformity_rate == 1.0
    assert cal.recommended_thresholds.max_numeric_relative_drift == pytest.approx(
        round(0.0741 * 1.5, 4), abs=0.002
    )
    # 20% latency increase * 1.5 headroom = 30.0%
    assert cal.recommended_thresholds.max_latency_regression_pct == pytest.approx(30.0)


def test_calibrate_from_reports_defaults_and_validation() -> None:
    base = _make_samples({"p1": "exact same plain text"}, latency=None)
    report = DriftBenchmarkEngine().evaluate(base, base)
    cal = calibrate_from_reports([report], headroom=1.25, min_floor_drift=0.08)
    assert cal.observed_max_drift == 0.0
    assert cal.observed_max_numeric_drift is None
    assert cal.observed_max_latency_regression_pct is None
    assert cal.recommended_thresholds.max_composite_drift == 0.08
    assert cal.recommended_thresholds.max_numeric_relative_drift == 0.15
    assert cal.recommended_thresholds.max_latency_regression_pct == 50.0

    # Negative latency regression clamps cleanly to the 10% floor.
    fast_base = _make_samples({"p1": "hi"}, latency=100.0)
    fast_cand = _make_samples({"p1": "hi"}, latency=80.0)
    fast_report = DriftBenchmarkEngine().evaluate(fast_base, fast_cand)
    fast_cal = calibrate_from_reports([fast_report])
    assert fast_cal.recommended_thresholds.max_latency_regression_pct == 10.0

    with pytest.raises(ValueError, match="At least one"):
        calibrate_from_reports([])
    with pytest.raises(ValueError, match="headroom"):
        calibrate_from_reports([report], headroom=0.9)
    with pytest.raises(ValueError, match="headroom"):
        calibrate_from_reports([report], headroom=True)
    with pytest.raises(ValueError, match="min_floor_drift"):
        calibrate_from_reports([report], min_floor_drift=1.5)


def test_rendered_calibrated_toml_round_trips_and_suppresses_volatile_noise(
    tmp_path: Path,
) -> None:
    run1 = _make_samples({"p1": '{"val": 10, "timestamp": "2026-01-01T00:00:00Z"}'})
    run2 = _make_samples({"p1": '{"val": 10, "timestamp": "2026-01-01T00:00:09Z"}'})
    volatile = detect_volatile_fields(run1, run2)
    report = DriftBenchmarkEngine().evaluate(run1, run2)
    cal = calibrate_from_reports([report], volatile_fields=volatile)

    toml_text = render_calibrated_toml(cal)
    assert '[fields."$.timestamp"]' in toml_text
    assert "ignore = true" in toml_text

    # Verify fallback TOML parser (Python 3.10 path) parses it cleanly too.
    parsed_fallback = _parse_toml_fallback(toml_text)
    assert parsed_fallback["fields"]["$.timestamp"]["ignore"] is True

    policy_path = tmp_path / "drift.toml"
    policy_path.write_text(toml_text, encoding="utf-8")
    loaded = load_policy(policy_path)
    assert loaded.thresholds == cal.recommended_thresholds
    assert loaded.field_rules["$.timestamp"].ignore is True

    # Re-evaluating with the calibrated policy reduces JSON drift on volatile timestamp to zero.
    re_report = DriftBenchmarkEngine(policy=loaded).evaluate(run1, run2)
    assert re_report.results[0].json_exact_field_match == 1.0
    assert re_report.passed


def test_cli_calibrate_from_captures_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_a = tmp_path / "run_a.jsonl"
    run_b = tmp_path / "run_b.jsonl"
    out_toml = tmp_path / "drift.toml"

    run_a.write_text(
        json.dumps({
            "prompt_id": "p1",
            "model": "m",
            "output": '{"status": "ok", "req_id": "111"}',
            "latency_ms": 100.0,
        }) + "\n",
        encoding="utf-8",
    )
    run_b.write_text(
        json.dumps({
            "prompt_id": "p1",
            "model": "m",
            "output": '{"status": "ok", "req_id": "222"}',
            "latency_ms": 115.0,
        }) + "\n",
        encoding="utf-8",
    )

    code = main([
        "calibrate",
        "--baseline-a",
        str(run_a),
        "--baseline-b",
        str(run_b),
        "-o",
        str(out_toml),
    ])
    assert code == 0
    assert "1 volatile field(s)" in capsys.readouterr().out
    loaded = load_policy(out_toml)
    assert loaded.field_rules["$.req_id"].ignore is True

    # Also test JSON stdout from a saved engine report.
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    rep_code = main([
        "calibrate",
        str(examples_dir / "expected-pass.json"),
        "--format",
        "json",
    ])
    assert rep_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs_analyzed"] == 1
    assert "recommended_thresholds" in payload


def test_cli_calibrate_error_handling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["calibrate"]) == 2
    assert "Provide at least one" in capsys.readouterr().err

    dummy = tmp_path / "a.jsonl"
    dummy.write_text("", encoding="utf-8")
    assert main(["calibrate", "--baseline-a", str(dummy)]) == 2
    assert "together" in capsys.readouterr().err

    assert main(["calibrate", "--baseline-a", str(dummy), "--baseline-b", str(dummy),
                 "-o", str(dummy)]) == 2
    assert "input file" in capsys.readouterr().err


def test_detect_volatile_fields_collapses_repeated_array_indices_and_init_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run1 = _make_samples({
        "p1": json.dumps({
            "items": [
                {"val": 1, "ts": "t1"},
                {"val": 2, "ts": "t2"},
            ]
        })
    })
    run2 = _make_samples({
        "p1": json.dumps({
            "items": [
                {"val": 1, "ts": "x1"},
                {"val": 2, "ts": "x2"},
            ]
        })
    })
    collapsed = detect_volatile_fields(run1, run2)
    assert collapsed == ["$.items[*].ts"]
    raw = detect_volatile_fields(run1, run2, collapse_array_wildcards=False)
    assert raw == ["$.items[0].ts", "$.items[1].ts"]

    # Render calibrated TOML and verify wildcard rule suppresses drift even for index 2!
    cal = calibrate_from_reports([DriftBenchmarkEngine().evaluate(run1, run2)],
                                 volatile_fields=collapsed)
    toml_path = tmp_path / "wildcard_drift.toml"
    toml_path.write_text(render_calibrated_toml(cal), encoding="utf-8")
    loaded = load_policy(toml_path)
    assert "$.items[*].ts" in loaded.field_rules

    three_a = _make_samples({
        "p1": json.dumps({"items": [{"val": i, "ts": f"a{i}"} for i in range(3)]})
    })
    three_b = _make_samples({
        "p1": json.dumps({"items": [{"val": i, "ts": f"b{i}"} for i in range(3)]})
    })
    rep = DriftBenchmarkEngine(policy=loaded).evaluate(three_a, three_b)
    assert rep.results[0].json_exact_field_match == 1.0

    # init-sample --with-policy creates a valid drift.toml alongside the starter suite
    sample_dir = tmp_path / "starter"
    assert main(["init-sample", "--dir", str(sample_dir), "--with-policy"]) == 0
    capsys.readouterr()
    assert (sample_dir / "drift.toml").is_file()
    starter_policy = load_policy(sample_dir / "drift.toml")
    assert starter_policy.field_rules["$.tags"].unordered_array is True


def test_detect_numeric_tolerances_and_invalid_volatile_path_rejection(tmp_path: Path) -> None:
    from llm_drift_bench.calibrate import detect_numeric_tolerances

    run1 = _make_samples({"p1": '{"confidence": 0.92, "timestamp": "t1"}'})
    run2 = _make_samples({"p1": '{"confidence": 0.95, "timestamp": "t2"}'})
    tols = detect_numeric_tolerances(run1, run2, headroom=1.25)
    assert tols == {"$.confidence": pytest.approx(0.0375, abs=1e-6)}

    report = DriftBenchmarkEngine().evaluate(run1, run2)
    cal = calibrate_from_reports(
        [report],
        volatile_fields=["$.timestamp", "$.confidence"],
        numeric_tolerances=tols,
    )
    assert cal.volatile_fields == ("$.timestamp",)
    assert cal.numeric_tolerances["$.confidence"] == pytest.approx(0.0375, abs=1e-6)
    toml_path = tmp_path / "tol.toml"
    toml_path.write_text(render_calibrated_toml(cal), encoding="utf-8")
    loaded = load_policy(toml_path)
    assert loaded.field_rules["$.confidence"].tolerance_abs == pytest.approx(0.0375, abs=1e-6)
    assert loaded.field_rules["$.timestamp"].ignore is True

    with pytest.raises(ValueError, match="must start with"):
        calibrate_from_reports([report], volatile_fields=["timestamp"])


