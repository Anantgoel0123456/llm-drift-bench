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
