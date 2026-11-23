"""Verify optional comparison breakdowns against saved engine reports."""

import json
from dataclasses import replace
from pathlib import Path

from llm_drift_bench._breakdowns import category_summary
from llm_drift_bench.cli import main
from llm_drift_bench.report_io import load_engine_report

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FILES = [str(EXAMPLES / f"expected-{state}.json") for state in ("pass", "drift")]


def test_category_means_use_the_individual_prompts() -> None:
    report = load_engine_report(FILES[1])
    combined = replace(report, results=tuple(replace(r, category="mixed") for r in report.results))
    row, = category_summary([combined])
    assert row.samples == row.drifted == 2
    assert row.mean_drift == sum(r.composite_drift_score for r in report.results) / 2


def test_rule_counts_distinguish_repeated_issues_from_affected_prompts() -> None:
    from llm_drift_bench._breakdowns import rule_counts

    report = load_engine_report(FILES[1])
    ticket = next(row for row in report.results if row.prompt_id == "ticket-01")
    validation = ticket.candidate_validation
    repeated = replace(validation, issues=validation.issues * 2)
    doubled = replace(ticket, candidate_validation=repeated)
    result, = rule_counts([replace(report, results=(doubled,))])
    assert (result.rule, result.affected_prompts, result.occurrences) == ("required", 1, 2)


def test_slowdowns_require_positive_paired_measurements() -> None:
    from llm_drift_bench._breakdowns import latency_increases

    report = load_engine_report(FILES[0])
    row = report.results[0]
    timed = tuple(replace(row, prompt_id=str(i), latency_delta_ms=d)
                  for i, d in enumerate((None, 0, -5, 8, 16)))
    results = latency_increases([replace(report, results=timed)], 1)
    assert [(item.prompt_id, item.delta_ms) for item in results] == [("4", 16)]


def test_numeric_ranking_includes_measured_changes_only() -> None:
    from llm_drift_bench._breakdowns import numeric_changes

    reports = [load_engine_report(path) for path in FILES]
    changed, = numeric_changes(reports, 4)
    assert (changed.prompt_id, changed.numeric_drift) == ("invoice-01", 0.4444)


def test_saved_comparison_export_matches_json_stdout(tmp_path, capsys) -> None:
    output = tmp_path / "comparison result.json"
    args = ["compare-reports", *FILES, "--by-category", "--rule-summary",
            "--slowdowns", "3", "--numeric-changes", "2", "--json-output", str(output)]
    assert main(args) == 0
    assert "invoice-01" in capsys.readouterr().out
    assert main([*args, "--fail-on-drift", "--format", "json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data == json.loads(output.read_text(encoding="utf-8"))
    assert len(data["categories"]) == 4
    assert data["validation_rules"][0]["rule"] == "required"
    assert data["slowdowns"] == []
    assert data["numeric_changes"][0]["prompt_id"] == "invoice-01"


def test_export_protects_source_reports(tmp_path) -> None:
    source = tmp_path / "saved.json"
    original = Path(FILES[0]).read_bytes()
    source.write_bytes(original)
    assert main(["compare-reports", str(source), FILES[1], "--json-output", str(source)]) == 2
    assert source.read_bytes() == original


def test_raw_comparison_breakdowns_and_input_protection(tmp_path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    original = (EXAMPLES / "baseline.jsonl").read_bytes()
    baseline.write_bytes(original)
    args = ["compare-many", "-b", str(baseline), "-s", str(EXAMPLES / "extraction-suite.json"),
            "-c", "A", str(EXAMPLES / "candidate-pass.jsonl"),
            "-c", "B", str(EXAMPLES / "candidate-drift.jsonl")]
    assert main([*args, "--json-output", str(baseline)]) == 2
    assert baseline.read_bytes() == original
    output = tmp_path / "comparison.json"
    assert main([*args, "--by-category", "--rule-summary", "--numeric-changes", "1",
                 "--slowdowns", "1", "--no-fail-on-drift", "--json-output", str(output)]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["validation_rules"][0]["candidate"] == "B"
    assert data["numeric_changes"][0]["prompt_id"] == "invoice-01"
    assert len(data["categories"]) == 4


def test_invalid_ranking_limits_preserve_existing_output(tmp_path) -> None:
    target = tmp_path / "previous.json"
    target.write_bytes(b"previous")
    for flag in ("--slowdowns", "--numeric-changes"):
        assert main(["compare-reports", *FILES, flag, "0", "--json-output", str(target)]) == 2
    assert target.read_bytes() == b"previous"


def test_failed_export_preserves_previous_result(tmp_path, monkeypatch) -> None:
    def fail_replace(source, target):
        raise OSError("destination busy")

    target = tmp_path / "output.json"
    target.write_bytes(b"old")
    monkeypatch.setattr("llm_drift_bench.cli.os.replace", fail_replace)
    assert main(["compare-reports", *FILES, "--json-output", str(target)]) == 2
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


def test_comparison_json_output_creates_nested_parent_directories(tmp_path) -> None:
    nested = tmp_path / "artifacts" / "sub" / "comparison.json"
    assert main(["compare-reports", *FILES, "--json-output", str(nested)]) == 0
    assert nested.is_file()
    data = json.loads(nested.read_text(encoding="utf-8"))
    assert len(data["candidates"]) == 2


def test_boolean_ranking_limits_are_rejected() -> None:
    import pytest

    from llm_drift_bench._breakdowns import latency_increases, numeric_changes
    from llm_drift_bench.comparison import _hotspots

    reports = [load_engine_report(path) for path in FILES]
    with pytest.raises(ValueError, match="positive"):
        latency_increases(reports, True)
    with pytest.raises(ValueError, match="positive"):
        numeric_changes(reports, True)
    with pytest.raises(ValueError, match="positive"):
        _hotspots(reports, True)

