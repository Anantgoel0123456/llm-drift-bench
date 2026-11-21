"""Verify optional comparison breakdowns against saved engine reports."""

from dataclasses import replace
from pathlib import Path

from llm_drift_bench._breakdowns import category_summary
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
    doubled = replace(ticket, candidate_validation=replace(validation, issues=validation.issues * 2))
    result, = rule_counts([replace(report, results=(doubled,))])
    assert (result.rule, result.affected_prompts, result.occurrences) == ("required", 1, 2)
