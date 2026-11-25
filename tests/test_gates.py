"""Tests for the composite GO/NO-GO release readiness gate evaluator."""

from __future__ import annotations

from pathlib import Path

from llm_drift_bench.gates import evaluate_release_gates
from llm_drift_bench.history import HistoryTrendCheck
from llm_drift_bench.report_io import load_engine_report
from llm_drift_bench.stats import McNemarResult, PairedSignificance
from llm_drift_bench.tool_events import ToolCheckResult
from llm_drift_bench.tool_log import ToolCheckRecord

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_evaluate_release_gates_combines_blocking_and_advisory_signals() -> None:
    pass_rep = load_engine_report(EXAMPLES / "expected-pass.json")
    drift_rep = load_engine_report(EXAMPLES / "expected-drift.json")

    ok = evaluate_release_gates(pass_rep)
    assert ok.approved is True
    assert ok.blocking_failures == ()

    trend_bad = HistoryTrendCheck(
        regressed=True,
        latest_drift=0.20,
        window_mean_drift=0.05,
        drift_delta=0.15,
        latest_conformity=1.0,
        window_mean_conformity=1.0,
        conformity_delta=0.0,
        window_size=3,
        reason="Mean drift increased by +0.1500.",
    )
    sig_bad = PairedSignificance(
        observed_diff=0.12,
        p_value=0.005,
        alpha=0.05,
        significant=True,
        permutations=1000,
        sample_size=20,
        effect_size_d=1.4,
    )
    mc_bad = McNemarResult(
        b_regressed=8,
        c_recovered=0,
        p_value=0.0078,
        alpha=0.05,
        significant=True,
        sample_size=20,
    )
    tool_unknown = ToolCheckRecord(
        resource="app.py@1",
        required_start=0,
        required_end=100,
        trace_complete=False,
        result=ToolCheckResult("UNKNOWN", "Trace completeness is not confirmed.", 0, 100, None),
    )

    # With default flags: history regression blocks, stats + tool checks emit warnings.
    verdict = evaluate_release_gates(
        pass_rep,
        history_check=trend_bad,
        significance=sig_bad,
        mcnemar=mc_bad,
        tool_checks=[tool_unknown],
    )
    assert verdict.approved is False
    assert len(verdict.blocking_failures) == 1
    assert "history_trend" in verdict.blocking_failures[0]
    assert len(verdict.warnings) == 3

    # Strict mode promotes all checks to blocking.
    strict = evaluate_release_gates(
        drift_rep,
        history_check=trend_bad,
        significance=sig_bad,
        mcnemar=mc_bad,
        tool_checks=[tool_unknown],
        fail_on_significant_degradation=True,
        require_tool_checks_pass=True,
    )
    assert strict.approved is False
    assert len(strict.blocking_failures) == 5
    assert strict.to_dict()["approved"] is False


def test_evaluate_release_gates_ignores_significant_improvements_and_passing_tools() -> None:
    pass_rep = load_engine_report(EXAMPLES / "expected-pass.json")
    sig_improved = PairedSignificance(
        observed_diff=-0.15,
        p_value=0.001,
        alpha=0.05,
        significant=True,
        permutations=1000,
        sample_size=20,
        effect_size_d=-1.2,
    )
    mc_improved = McNemarResult(
        b_regressed=0,
        c_recovered=6,
        p_value=0.0312,
        alpha=0.05,
        significant=True,
        sample_size=20,
    )
    tool_ok = ToolCheckRecord(
        resource="app.py@1",
        required_start=0,
        required_end=100,
        trace_complete=True,
        result=ToolCheckResult("PASS", "Coverage verified.", 100, 100, 1),
    )
    verdict = evaluate_release_gates(
        pass_rep,
        significance=sig_improved,
        mcnemar=mc_improved,
        tool_checks=[tool_ok],
        fail_on_significant_degradation=True,
        require_tool_checks_pass=True,
    )
    assert verdict.approved is True
    assert verdict.blocking_failures == ()
    assert verdict.warnings == ()


def test_evaluate_release_gates_with_widespread_distribution_and_consumer_rejection() -> None:
    from llm_drift_bench.consumer import ConsumerDecision, compare_consumer_decisions
    from llm_drift_bench.distribution import analyze_segmented_distributions
    from llm_drift_bench.models import ModelSample

    pass_rep = load_engine_report(EXAMPLES / "expected-pass.json")
    base = [
        ModelSample(prompt_id="a1", output='{"label": "high"}', metadata={"route": "r1"}),
        ModelSample(prompt_id="b1", output='{"label": "high"}', metadata={"route": "r2"}),
    ]
    cand = [
        ModelSample(prompt_id="a1", output='{"label": "low"}', metadata={"route": "r1"}),
        ModelSample(prompt_id="b1", output='{"label": "low"}', metadata={"route": "r2"}),
    ]
    seg = analyze_segmented_distributions(base, cand, group_by="metadata:route")
    cons = compare_consumer_decisions(
        [ConsumerDecision("a1", True), ConsumerDecision("b1", True)],
        [ConsumerDecision("a1", False, "bad"), ConsumerDecision("b1", True)],
        max_rejection_increase=0.1,
    )

    verdict = evaluate_release_gates(
        pass_rep,
        segmented_distribution=seg,
        consumer_comparison=cons,
        fail_on_widespread_shift=True,
        fail_on_consumer_rejection=True,
    )
    assert verdict.approved is False
    names = {c.name for c in verdict.checks}
    assert {"distribution_blast_radius", "consumer_rejection"} <= names
    assert len(verdict.blocking_failures) == 2

