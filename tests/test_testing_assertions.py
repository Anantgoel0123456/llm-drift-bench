"""Tests for framework-independent regression assertion helpers."""

from __future__ import annotations

import pytest

from llm_drift_bench.conversation import (
    ConversationRecord,
    ConversationTurn,
    analyze_conversation_drift,
)
from llm_drift_bench.embeddings import compare_embedding_vectors
from llm_drift_bench.engine import DriftBenchmarkEngine
from llm_drift_bench.gates import evaluate_release_gates
from llm_drift_bench.models import ModelSample
from llm_drift_bench.testing import (
    assert_no_conversation_drift,
    assert_no_drift,
    assert_no_embedding_drift,
    assert_no_regression,
    assert_release_approved,
)


def test_assert_no_conversation_embedding_and_release_gate_helpers() -> None:
    def _rec(cid: str, reply: str) -> ConversationRecord:
        return ConversationRecord(
            conversation_id=cid,
            turns=(ConversationTurn("user", "hi"), ConversationTurn("assistant", reply)),
        )

    # Conversation drift assertions (pass, >5 failures truncation, wrong type)
    ok_conv = analyze_conversation_drift([_rec("c1", "ok")], [_rec("c1", "ok")])
    assert_no_conversation_drift(ok_conv)

    base_many = [_rec(f"c{i}", '{"ok": true}') for i in range(6)]
    cand_many = [_rec(f"c{i}", "I'm sorry, I cannot assist with that.") for i in range(6)]
    fail_conv = analyze_conversation_drift(base_many, cand_many)
    with pytest.raises(AssertionError, match="1 additional drifted conversation"):
        assert_no_conversation_drift(fail_conv)
    with pytest.raises(TypeError, match="ConversationDriftReport"):
        assert_no_conversation_drift({})  # type: ignore[arg-type]

    # Embedding drift assertions
    ok_emb = compare_embedding_vectors(
        {"p1": [1.0, 0.0]}, {"p1": [1.0, 0.0]}, min_mean_similarity=0.9
    )
    assert_no_embedding_drift(ok_emb)
    fail_emb = compare_embedding_vectors(
        {"p1": [1.0, 0.0]}, {"p1": [0.0, 1.0]}, min_mean_similarity=0.9
    )
    with pytest.raises(AssertionError, match="embedding similarity check failed"):
        assert_no_embedding_drift(fail_emb)
    with pytest.raises(TypeError, match="EmbeddingDriftReport"):
        assert_no_embedding_drift({})  # type: ignore[arg-type]

    # Release gate verdict assertions
    base_s = [ModelSample(prompt_id=f"p{i}", model_name="b", output='{"x": 1}') for i in range(6)]
    pass_s = [ModelSample(prompt_id=f"p{i}", model_name="ok", output='{"x": 1}') for i in range(6)]
    fail_s = [ModelSample(prompt_id=f"p{i}", model_name="bad", output="nope") for i in range(6)]
    engine = DriftBenchmarkEngine()
    rep_ok = engine.evaluate(base_s, pass_s)
    rep_fail = engine.evaluate(base_s, fail_s)

    assert_no_drift(rep_ok)
    assert_release_approved(evaluate_release_gates(rep_ok))
    with pytest.raises(AssertionError, match="NO-GO"):
        assert_release_approved(evaluate_release_gates(rep_fail))
    with pytest.raises(TypeError, match="ReleaseVerdict"):
        assert_release_approved({})  # type: ignore[arg-type]

    # assert_no_regression with >5 regressed prompts hits truncation line cleanly
    with pytest.raises(AssertionError, match="1 additional regressions"):
        assert_no_regression([rep_ok, rep_fail], against="ok")


def test_assert_no_tool_trace_drift_and_assert_audit_passed() -> None:
    from llm_drift_bench.audit import run_full_audit
    from llm_drift_bench.testing import assert_audit_passed, assert_no_tool_trace_drift
    from llm_drift_bench.tool_events import ToolEvent, compare_tool_traces

    base_ev = [ToolEvent(0, "read", "src/a.py@1", start=0, end=10)]
    cand_ev = [
        ToolEvent(0, "read", "src/a.py@1", start=0, end=10),
        ToolEvent(1, "edit", "src/b.py@1"),
    ]
    assert_no_tool_trace_drift(compare_tool_traces(base_ev, base_ev))
    fail_diff = compare_tool_traces(base_ev, cand_ev, fail_on_new_edit_resources=True)
    with pytest.raises(AssertionError, match="agent tool-trace drift failed"):
        assert_no_tool_trace_drift(fail_diff)
    with pytest.raises(TypeError, match="ToolTraceDiff"):
        assert_no_tool_trace_drift({})  # type: ignore[arg-type]

    same = [ModelSample(prompt_id="p1", output='{"x": 1}')]
    diff = [ModelSample(prompt_id="p1", output="bad")]
    assert_audit_passed(run_full_audit(same, same))
    with pytest.raises(AssertionError, match="unified audit failed"):
        assert_audit_passed(run_full_audit(same, diff))
    with pytest.raises(TypeError, match="AuditReport"):
        assert_audit_passed({})  # type: ignore[arg-type]


def test_assert_consumer_quality_and_rolling_helpers() -> None:
    from llm_drift_bench.consumer import ConsumerDecision, compare_consumer_decisions
    from llm_drift_bench.quality_checks import run_quality_checks
    from llm_drift_bench.rolling import analyze_rolling_distribution
    from llm_drift_bench.testing import (
        assert_no_consumer_regression,
        assert_no_quality_regressions,
        assert_no_rolling_drift,
    )

    # Consumer regression assertions
    base_d = [ConsumerDecision("p1", True), ConsumerDecision("p2", True)]
    cand_bad = [ConsumerDecision("p1", False, "bad"), ConsumerDecision("p2", True)]
    assert_no_consumer_regression(compare_consumer_decisions(base_d, base_d))
    with pytest.raises(AssertionError, match="downstream consumer check failed"):
        assert_no_consumer_regression(
            compare_consumer_decisions(base_d, cand_bad, max_rejection_increase=0.1)
        )
    with pytest.raises(TypeError, match="ConsumerComparison"):
        assert_no_consumer_regression({})  # type: ignore[arg-type]

    # Quality regression assertions (pass, >5 findings truncation, wrong type)
    base_q = [ModelSample(prompt_id=f"p{i}", output="Detailed valid answer " * 4) for i in range(6)]
    cand_q = [
        ModelSample(prompt_id=f"p{i}", output="I'm sorry, I cannot assist with that.")
        for i in range(6)
    ]
    assert_no_quality_regressions(run_quality_checks(base_q, base_q))
    with pytest.raises(AssertionError, match="additional quality finding"):
        assert_no_quality_regressions(run_quality_checks(base_q, cand_q))
    with pytest.raises(TypeError, match="QualityReport"):
        assert_no_quality_regressions({})  # type: ignore[arg-type]

    # Rolling distribution assertions
    w1 = [ModelSample(prompt_id="p1", output='{"label": "a"}')]
    w2 = [ModelSample(prompt_id="p1", output='{"label": "b"}')]
    assert_no_rolling_drift(analyze_rolling_distribution([w1], w1, max_tvd=0.1))
    with pytest.raises(AssertionError, match="rolling-baseline distribution check failed"):
        assert_no_rolling_drift(analyze_rolling_distribution([w1], w2, max_tvd=0.1))
    with pytest.raises(TypeError, match="RollingWindowReport"):
        assert_no_rolling_drift({})  # type: ignore[arg-type]


def test_assert_formats_logprobs_and_cost_helpers() -> None:
    import math

    from llm_drift_bench.cost import ModelPricing, estimate_cost_drift
    from llm_drift_bench.formats import check_field_formats
    from llm_drift_bench.logprobs import analyze_logprobs_drift
    from llm_drift_bench.testing import (
        assert_cost_within_budget,
        assert_formats_valid,
        assert_logprobs_within_budget,
    )

    # Format assertions (pass, >5 issues truncation, wrong type)
    ok_fmt = check_field_formats(
        [ModelSample(prompt_id="p1", output='{"email": "a@example.com"}')],
        {"$.email": "email"},
    )
    assert_formats_valid(ok_fmt)
    bad_fmt = check_field_formats(
        [ModelSample(prompt_id=f"p{i}", output='{"email": "bad"}') for i in range(6)],
        {"$.email": "email"},
    )
    with pytest.raises(AssertionError, match="1 additional format issue"):
        assert_formats_valid(bad_fmt)
    with pytest.raises(TypeError, match="FormatReport"):
        assert_formats_valid({})  # type: ignore[arg-type]

    # Logprobs assertions (no limits, pass, entropy fail, jsd fail, incomplete fail, bad args)
    b_lp = [
        ModelSample(
            prompt_id="p1",
            output="yes",
            metadata={"logprobs": [{"token": "yes", "logprob": math.log(0.95)}]},
        )
    ]
    c_lp = [
        ModelSample(
            prompt_id="p1",
            output="yes",
            metadata={"logprobs": [{"token": "yes", "logprob": math.log(0.55)}]},
        )
    ]
    lp_same = analyze_logprobs_drift(b_lp, b_lp)
    lp_diff = analyze_logprobs_drift(b_lp, c_lp)
    lp_inc = analyze_logprobs_drift([ModelSample("p1", "yes")], [ModelSample("p1", "yes")])

    assert_logprobs_within_budget(lp_inc)
    assert_logprobs_within_budget(lp_same, max_entropy_delta=0.05, max_top_k_jsd=0.05)
    with pytest.raises(AssertionError, match="abs\\(mean_entropy_delta_bits\\)"):
        assert_logprobs_within_budget(lp_diff, max_entropy_delta=0.01)
    with pytest.raises(AssertionError, match="max_top_k_jsd"):
        assert_logprobs_within_budget(lp_diff, max_top_k_jsd=0.01)
    with pytest.raises(AssertionError, match="Incomplete logprobs coverage"):
        assert_logprobs_within_budget(lp_inc, max_top_k_jsd=0.5)
    with pytest.raises(TypeError, match="LogprobDriftSummary"):
        assert_logprobs_within_budget({})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_entropy_delta"):
        assert_logprobs_within_budget(lp_same, max_entropy_delta=-0.1)
    with pytest.raises(ValueError, match="max_top_k_jsd"):
        assert_logprobs_within_budget(lp_same, max_top_k_jsd=1.5)

    # Cost assertions (pass, fail, wrong type)
    c_base = [ModelSample(prompt_id="p1", output="a", tokens_generated=100)]
    c_cand = [ModelSample(prompt_id="p1", output="a", tokens_generated=200)]
    comp = estimate_cost_drift(
        c_base,
        c_cand,
        baseline_pricing=ModelPricing(output_per_million_usd=10.0),
        candidate_pricing=ModelPricing(output_per_million_usd=10.0),
    )
    assert_cost_within_budget(comp, max_token_increase_pct=150.0, max_cost_increase_pct=150.0)
    with pytest.raises(AssertionError, match="LLM cost/token budget failed"):
        assert_cost_within_budget(comp, max_token_increase_pct=20.0)
    with pytest.raises(TypeError, match="CostComparison"):
        assert_cost_within_budget({})  # type: ignore[arg-type]
