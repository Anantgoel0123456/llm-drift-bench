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
