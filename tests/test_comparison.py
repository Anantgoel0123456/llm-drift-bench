"""Comparability and ordering of saved benchmark reports."""

from dataclasses import replace
from pathlib import Path

import pytest

from llm_drift_bench.comparison import compare_reports
from llm_drift_bench.models import DriftBenchmarkReport
from llm_drift_bench.report_io import load_report

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def fixture(name: str) -> DriftBenchmarkReport:
    report = load_report(EXAMPLES / f"expected-{name}.json")
    assert isinstance(report, DriftBenchmarkReport)
    return report


def test_rank_demo_candidates() -> None:
    rows = compare_reports([fixture("drift"), fixture("pass")])
    assert [(r.passed, r.drifted) for r in rows] == [(True, 0), (False, 2)]


def test_require_two_reports_with_common_baseline() -> None:
    report = fixture("pass")
    with pytest.raises(ValueError, match="two"):
        compare_reports([report])
    with pytest.raises(ValueError, match="baseline"):
        compare_reports([report, replace(report, baseline_model="other")])


def test_rejects_different_prompt_sets() -> None:
    report = fixture("drift")
    with pytest.raises(ValueError, match="prompt IDs"):
        compare_reports([fixture("pass"), replace(report, results=report.results[:1])])


def test_rejects_changed_prompt_categories() -> None:
    report = fixture("drift")
    rows = tuple(replace(row, category="other") for row in report.results)
    with pytest.raises(ValueError, match="categories"):
        compare_reports([fixture("pass"), replace(report, results=rows)])


def test_rejects_unknown_and_different_gates() -> None:
    report = fixture("drift")
    assert report.thresholds is not None
    for gate in (None, replace(report.thresholds, max_composite_drift=0.9)):
        with pytest.raises(ValueError, match="thresholds"):
            compare_reports([fixture("pass"), replace(report, thresholds=gate)])


def test_detects_changed_baseline_evidence() -> None:
    report = fixture("drift")
    rows = tuple(replace(row, baseline_validation=row.candidate_validation)
                 for row in report.results)
    with pytest.raises(ValueError, match="Baseline validation"):
        compare_reports([fixture("pass"), replace(report, results=rows)])


def test_distinguishes_tied_candidates_and_rejects_duplicate_labels() -> None:
    report = fixture("pass")
    tied = replace(report, candidate_model="a")
    assert compare_reports([report, tied])[0].candidate == "a"
    with pytest.raises(ValueError, match="Duplicate candidate"):
        compare_reports([report, report])


def test_comparison_preserves_candidate_conformity() -> None:
    rows = compare_reports([fixture("pass"), fixture("drift")])
    assert [row.conformity for row in rows] == [1.0, 0.5]


def test_p95_retains_missing_and_measured_values() -> None:
    report = replace(fixture("drift"), candidate_p95_latency_ms=12.5)
    rows = compare_reports([fixture("pass"), report])
    assert [row.p95_ms for row in rows] == [None, 12.5]


def test_comparison_exposes_worst_case_drift() -> None:
    report = fixture("drift")
    row = compare_reports([fixture("pass"), report])[1]
    assert row.max_drift == report.max_composite_drift > row.mean_drift
