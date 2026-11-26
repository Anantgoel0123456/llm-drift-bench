"""Ensure presentation and persistence cannot silently change the CI decision."""

import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from llm_drift_bench import (
    DriftBenchmarkEngine,
    DriftBenchmarkReport,
    DriftThresholds,
    ModelSample,
    evaluate_suite,
    export_html_report,
    export_markdown_report,
    load_model_samples,
    load_prompt_cases,
    load_report,
    render_console_report,
)
from llm_drift_bench.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_demo_failure_explains_schema_and_numeric_population() -> None:
    report = load_report(EXAMPLES / "expected-drift.json")
    assert isinstance(report, DriftBenchmarkReport)
    assert not report.passed
    assert report.results[1].candidate_validation.is_valid_json
    assert not report.results[1].candidate_validation.is_schema_conformant
    output = io.StringIO()
    render_console_report(report, Console(file=output, width=240))
    for text in (export_html_report(report), export_markdown_report(report), output.getvalue()):
        assert "Schema pass" in text
        assert "Bounded numeric drift" in text
        assert "1 of 2 samples comparable" in text
        assert "ticket-01: $.priority: Missing required property" in text
        assert "not percentage change" in text
        assert "FAILED" in text


def test_no_numeric_fields_are_unavailable_in_report() -> None:
    samples = [ModelSample(prompt_id="p", output='{"label":"same"}')]
    report = DriftBenchmarkEngine().evaluate(samples, samples)
    assert report.passed
    assert report.mean_json_numeric_drift is None
    assert "0 of 1 samples comparable" in export_html_report(report)
    assert "only JSON parsing is checked" in export_markdown_report(report)


@pytest.mark.parametrize(("name", "code"), [("pass", 0), ("drift", 1)])
def test_every_format_uses_one_evaluation_and_preserves_saved_result(
    name: str, code: int, tmp_path: Path,
) -> None:
    baseline = load_model_samples(EXAMPLES / "baseline.jsonl")
    candidate = load_model_samples(EXAMPLES / f"candidate-{name}.jsonl")
    cases = load_prompt_cases(EXAMPLES / "extraction-suite.json")
    expected = DriftBenchmarkEngine().evaluate(baseline, candidate, cases)
    for fmt in ("table", "json", "rich", "markdown", "html"):
        saved = tmp_path / f"{fmt}.json"
        stdout = io.StringIO()
        with (
            patch.object(DriftBenchmarkEngine, "evaluate", return_value=expected) as evaluate,
            patch("llm_drift_bench.evaluator.evaluate_suite",
                  side_effect=AssertionError("Unexpected second evaluator")),
            contextlib.redirect_stdout(stdout),
        ):
            actual_code = main([
                "run", "-b", str(EXAMPLES / "baseline.jsonl"),
                "-c", str(EXAMPLES / f"candidate-{name}.jsonl"),
                "-p", str(EXAMPLES / "extraction-suite.json"),
                "--format", fmt, "--output", str(saved),
            ])
        assert actual_code == code
        evaluate.assert_called_once()
        assert json.loads(saved.read_text(encoding="utf-8")) == expected.to_dict()
        if fmt == "json":
            assert json.loads(stdout.getvalue()) == expected.to_dict()
        else:
            assert ("PASSED" if expected.passed else "FAILED") in stdout.getvalue()
            assert f"{expected.mean_composite_drift:.4f}" in stdout.getvalue()


def test_saved_report_roundtrip_and_exports(tmp_path: Path) -> None:
    samples = [ModelSample(prompt_id="p", output="{}")]
    report = DriftBenchmarkEngine(DriftThresholds(max_composite_drift=0.01)).evaluate(
        samples, samples,
    )
    path = tmp_path / "result.json"
    path.write_text(report.to_json(), encoding="utf-8-sig")
    loaded = load_report(path)
    assert isinstance(loaded, DriftBenchmarkReport)
    assert loaded.to_dict() == report.to_dict()
    for fmt in ("json", "markdown", "html", "console"):
        output = tmp_path / "nested" / f"result.{fmt}"
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["report", "-i", str(path), "--format", fmt, "-o", str(output)]) == 0
        text = output.read_text(encoding="utf-8")
        if fmt == "json":
            assert json.loads(text) == report.to_dict()
        else:
            assert "PASSED" in text
            assert "0.01" in text
            assert "n/a" in text
    raw = report.to_dict()
    raw.pop("thresholds")
    path.write_text(json.dumps(raw), encoding="utf-8")
    older = load_report(path)
    assert isinstance(older, DriftBenchmarkReport)
    assert older.thresholds is None
    assert "Thresholds unavailable" in export_markdown_report(older)


def test_latency_failure_and_missing_data_remain_visible() -> None:
    baseline = [ModelSample(prompt_id="p", output="{}", latency_ms=10)]
    candidate = [ModelSample(prompt_id="p", output="{}", latency_ms=100)]
    report = DriftBenchmarkEngine().evaluate(baseline, candidate)
    assert not report.passed and report.drifted_samples == 0
    for text in (export_html_report(report), export_markdown_report(report)):
        assert "FAILED" in text
        assert report.failure_reasons[0] in text
    no_times = DriftBenchmarkEngine().evaluate(
        [ModelSample(prompt_id="p", output="{}")], [ModelSample(prompt_id="p", output="{}")],
    )
    assert "Latency regression check skipped" in export_html_report(no_times)


def test_report_text_cannot_become_html_markdown_or_rich_markup() -> None:
    label = '<script>alert(1)</script>|[red]name[/red]'
    samples = [ModelSample(prompt_id=label, model_name=label, output="{}")]
    report = DriftBenchmarkEngine().evaluate(samples, samples)
    html = export_html_report(report)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "default-src 'none'" in html
    markdown = export_markdown_report(report)
    assert label not in markdown
    assert "&#124;" in markdown
    output = io.StringIO()
    render_console_report(report, Console(file=output, width=200))
    assert "[red]name[/red]" in output.getvalue()


@pytest.mark.parametrize("raw", [[], {}, {"results": []}, {"total_samples": 2, "results": []}])
def test_invalid_reports_are_rejected(raw: object, tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_report(path)


def test_inconsistent_counts_ids_and_nonfinite_values_are_rejected(tmp_path: Path) -> None:
    samples = [ModelSample(prompt_id="p", output="{}")]
    report = DriftBenchmarkEngine().evaluate(samples, samples)
    path = tmp_path / "bad.json"
    for broken in (
        replace(report, total_samples=2),
        replace(report, drifted_samples=1),
        replace(report, total_samples=2, results=report.results * 2),
        replace(report, mean_composite_drift=float("nan")),
    ):
        path.write_text(broken.to_json(), encoding="utf-8")
        with pytest.raises(ValueError):
            load_report(path)


def test_legacy_evaluator_reports_are_not_reinterpreted(tmp_path: Path) -> None:
    samples = [ModelSample(prompt_id="p", output="{}")]
    legacy = evaluate_suite([], samples, samples)
    path = tmp_path / "legacy.json"
    path.write_text(legacy.model_dump_json(), encoding="utf-8")
    loaded = load_report(path)
    assert type(loaded) is type(legacy)
    assert "Legacy evaluator scores" in export_markdown_report(loaded)
    with pytest.raises(ValueError, match="HTML requires an engine report"):
        export_html_report(loaded)


def test_canonical_report_schema_version_backward_compat_and_rejection(tmp_path: Path) -> None:
    from llm_drift_bench import REPORT_SCHEMA_VERSION

    samples = [ModelSample(prompt_id="p", output="{}")]
    report = DriftBenchmarkEngine().evaluate(samples, samples)
    payload = report.to_dict()
    assert payload["schema_version"] == REPORT_SCHEMA_VERSION

    # Legacy pre-versioned report omitting schema_version loads cleanly with default 1
    legacy_dict = dict(payload)
    legacy_dict.pop("schema_version")
    legacy_path = tmp_path / "pre_version.json"
    legacy_path.write_text(json.dumps(legacy_dict), encoding="utf-8")
    loaded = load_report(legacy_path)
    assert isinstance(loaded, DriftBenchmarkReport)
    assert loaded.schema_version == 1

    # Unsupported or non-integer schema_version values fail cleanly
    bad_path = tmp_path / "bad_ver.json"
    for bad_ver in (0, 99, True, "1"):
        bad_path.write_text(json.dumps({**payload, "schema_version": bad_ver}), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            load_report(bad_path)

