"""Tests for the unified offline audit report and executive dashboard."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from llm_drift_bench.audit import (
    export_audit_html,
    format_audit_markdown,
    format_audit_text,
    run_full_audit,
)
from llm_drift_bench.models import ModelSample
from llm_drift_bench.packs import get_pack
from llm_drift_bench.policy import load_policy


def test_full_audit_passes_clean_pack_and_fails_drifted_pack(tmp_path: Path) -> None:
    pack = get_pack("extraction")
    policy_file = tmp_path / "drift.toml"
    policy_file.write_text(pack.policy_toml, encoding="utf-8")
    policy = load_policy(policy_file)

    ok_audit = run_full_audit(
        pack.baseline,
        pack.candidate_pass,
        prompt_cases=pack.prompts,
        policy=policy,
        max_tvd=0.1,
        max_wasserstein=0.1,
    )
    assert ok_audit.passed is True
    assert ok_audit.failure_reasons == ()
    assert ok_audit.cost.complete is True
    assert "PASSED" in format_audit_text(ok_audit)
    assert "# Unified Drift Audit: PASSED" in format_audit_markdown(ok_audit)

    bad_audit = run_full_audit(
        pack.baseline,
        pack.candidate_drift,
        prompt_cases=pack.prompts,
        policy=policy,
        max_tvd=0.1,
        max_jsd=0.05,
        max_wasserstein=0.1,
    )
    assert bad_audit.passed is False
    assert any("Distribution max_tvd" in r for r in bad_audit.failure_reasons)
    assert any("Distribution max_jsd" in r for r in bad_audit.failure_reasons)
    assert bad_audit.to_dict()["passed"] is False

    html_out = tmp_path / "audit.html"
    html = export_audit_html(bad_audit, html_out)
    assert html_out.read_text(encoding="utf-8") == html
    assert "default-src 'none'" in html
    assert "<script" not in html.lower()
    assert "FAILED" in html


def test_full_audit_quality_and_logprobs_gates() -> None:
    pack = get_pack("coding")
    coding_audit = run_full_audit(
        pack.baseline,
        pack.candidate_drift,
        prompt_cases=pack.prompts,
        fail_on_quality_issues=True,
    )
    assert coding_audit.passed is False
    assert any("Quality regressions" in r for r in coding_audit.failure_reasons)
    assert "## Quality findings" in format_audit_markdown(coding_audit)

    # Incomplete logprobs with threshold must fail honestly
    lp_missing = run_full_audit(
        pack.baseline,
        pack.candidate_pass,
        prompt_cases=pack.prompts,
        max_top_k_jsd=0.2,
    )
    assert lp_missing.passed is False
    assert any("Logprobs coverage incomplete" in r for r in lp_missing.failure_reasons)

    # Complete logprobs breaching threshold fails with exact reason
    lp_base = [
        ModelSample(
            prompt_id="p1",
            output="yes",
            model_name="fp16",
            metadata={
                "logprobs": [
                    {
                        "token": "yes",
                        "logprob": math.log(0.99),
                        "top_logprobs": [
                            {"token": "yes", "logprob": math.log(0.99)},
                            {"token": "no", "logprob": math.log(0.01)},
                        ],
                    }
                ]
            },
        )
    ]
    lp_cand = [
        ModelSample(
            prompt_id="p1",
            output="yes",
            model_name="q4",
            metadata={
                "logprobs": [
                    {
                        "token": "yes",
                        "logprob": math.log(0.55),
                        "top_logprobs": [
                            {"token": "yes", "logprob": math.log(0.55)},
                            {"token": "no", "logprob": math.log(0.45)},
                        ],
                    }
                ]
            },
        )
    ]
    lp_fail = run_full_audit(lp_base, lp_cand, max_entropy_delta=0.2, max_top_k_jsd=0.1)
    assert lp_fail.passed is False
    assert any("mean_entropy_delta_bits" in r for r in lp_fail.failure_reasons)
    assert any("max_top_k_jsd" in r for r in lp_fail.failure_reasons)

    with pytest.raises(ValueError, match="max_tvd"):
        run_full_audit(lp_base, lp_cand, max_tvd=1.5)
    with pytest.raises(ValueError, match="max_entropy_delta"):
        run_full_audit(lp_base, lp_cand, max_entropy_delta=-0.1)


def test_full_audit_segmented_widespread_and_consumer_integration() -> None:
    from llm_drift_bench.consumer import ConsumerDecision, compare_consumer_decisions

    base = [
        ModelSample(
            prompt_id="a1", output='{"label": "high"}', model_name="b", metadata={"route": "r1"}
        ),
        ModelSample(
            prompt_id="b1", output='{"label": "high"}', model_name="b", metadata={"route": "r2"}
        ),
    ]
    cand = [
        ModelSample(
            prompt_id="a1", output='{"label": "low"}', model_name="c", metadata={"route": "r1"}
        ),
        ModelSample(
            prompt_id="b1", output='{"label": "low"}', model_name="c", metadata={"route": "r2"}
        ),
    ]
    cons = compare_consumer_decisions(
        [ConsumerDecision("a1", True), ConsumerDecision("b1", True)],
        [ConsumerDecision("a1", False, "bad"), ConsumerDecision("b1", True)],
        max_rejection_increase=0.1,
    )
    rep = run_full_audit(
        base,
        cand,
        group_by="metadata:route",
        fail_on_widespread_shift=True,
        consumer_comparison=cons,
    )
    assert rep.passed is False
    assert rep.segmented_distribution is not None
    assert rep.segmented_distribution.shift_scope == "widespread"
    assert any("Widespread distribution shift" in r for r in rep.failure_reasons)
    assert any("Rejection rate increased" in r for r in rep.failure_reasons)
    assert "Blast radius : widespread" in format_audit_text(rep)
    assert "Downstream consumer" in format_audit_markdown(rep)
    as_dict = rep.to_dict()
    assert "segmented_distribution" in as_dict and "consumer" in as_dict


def test_full_audit_embedding_and_conversation_integration(tmp_path: Path) -> None:
    from llm_drift_bench.conversation import (
        ConversationRecord,
        ConversationTurn,
        analyze_conversation_drift,
    )
    from llm_drift_bench.embeddings import compare_embedding_vectors

    samples = [ModelSample(prompt_id="p1", output='{"ok": true}', model_name="m1")]
    emb_fail = compare_embedding_vectors(
        {"p1": [1.0, 0.0]},
        {"p1": [0.0, 1.0]},
        min_mean_similarity=0.85,
    )
    conv_fail = analyze_conversation_drift(
        [
            ConversationRecord(
                "c1",
                (
                    ConversationTurn("user", "hi"),
                    ConversationTurn("assistant", '{"ok": true}'),
                ),
            )
        ],
        [
            ConversationRecord(
                "c1",
                (
                    ConversationTurn("user", "hi"),
                    ConversationTurn("assistant", "I'm sorry, I cannot assist with that."),
                ),
            )
        ],
    )
    rep = run_full_audit(
        samples,
        samples,
        embedding_report=emb_fail,
        conversation_report=conv_fail,
    )
    assert rep.passed is False
    assert rep.embeddings is emb_fail and rep.conversations is conv_fail
    assert any("cosine similarity" in r for r in rep.failure_reasons)
    assert any("Refusal flip" in r or "exceeded trajectory drift" in r for r in rep.failure_reasons)

    txt = format_audit_text(rep)
    assert "Embeddings   : FAIL" in txt and "Conversations: FAIL" in txt
    md = format_audit_markdown(rep)
    assert "| Embedding similarity | FAIL |" in md
    assert "| Multi-turn conversations | FAIL |" in md
    html = export_audit_html(rep, tmp_path / "full.html")
    assert "Embedding Mean Cos" in html and "Multi-Turn Conversations" in html
    assert "<script" not in html.lower()
    payload = rep.to_dict()
    assert "embeddings" in payload and "conversations" in payload


def test_full_audit_tool_trace_integration_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from llm_drift_bench.cli import main
    from llm_drift_bench.tool_events import ToolEvent, compare_tool_traces

    samples = [ModelSample(prompt_id="p1", output='{"ok": true}', model_name="m1")]
    base_ev = [ToolEvent(0, "read", "src/a.py@1", start=0, end=10)]
    cand_ev = [
        ToolEvent(0, "read", "src/a.py@1", start=0, end=10),
        ToolEvent(1, "edit", "src/secret.py@1"),
    ]
    diff_fail = compare_tool_traces(base_ev, cand_ev, fail_on_new_edit_resources=True)
    rep = run_full_audit(samples, samples, tool_trace_diff=diff_fail)
    assert rep.passed is False
    assert rep.tool_trace is diff_fail
    assert any("new resource" in r for r in rep.failure_reasons)
    assert "Tool trace   : FAIL" in format_audit_text(rep)
    assert "| Agent tool trace | FAIL |" in format_audit_markdown(rep)
    html = export_audit_html(rep)
    assert "Agent Tool Calls" in html and "1 &rarr; 2 (FAIL)" in html
    assert "tool_trace" in rep.to_dict()

    examples = Path(__file__).resolve().parents[1] / "examples"
    b_trace = tmp_path / "b.jsonl"
    c_trace = tmp_path / "c.jsonl"
    b_trace.write_text(
        '{"kind": "read", "resource": "a@1", "start": 0, "end": 10}\n', encoding="utf-8"
    )
    c_trace.write_text(
        '{"kind": "read", "resource": "a@1", "start": 0, "end": 10}\n'
        '{"kind": "edit", "resource": "b@1"}\n',
        encoding="utf-8",
    )
    assert (
        main(
            [
                "audit",
                "-s",
                str(examples / "extraction-suite.json"),
                "-b",
                str(examples / "baseline.jsonl"),
                "-c",
                str(examples / "candidate-pass.jsonl"),
                "--baseline-trace",
                str(b_trace),
                "--candidate-trace",
                str(c_trace),
                "--fail-on-new-edit-resources",
                "--format",
                "json",
            ]
        )
        == 1
    )
    doc = json.loads(capsys.readouterr().out)
    assert doc["passed"] is False and doc["tool_trace"]["passed"] is False

    # Missing paired trace flag returns usage/input error (exit 2)
    assert (
        main(
            [
                "audit",
                "-b",
                str(examples / "baseline.jsonl"),
                "-c",
                str(examples / "candidate-pass.jsonl"),
                "--baseline-trace",
                str(b_trace),
            ]
        )
        == 2
    )


def test_export_audit_pr_comment_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_drift_bench.audit import export_audit_pr_comment
    from llm_drift_bench.cli import main
    from llm_drift_bench.conversation import (
        ConversationRecord,
        ConversationTurn,
        analyze_conversation_drift,
    )
    from llm_drift_bench.embeddings import compare_embedding_vectors
    from llm_drift_bench.tool_events import ToolEvent, compare_tool_traces

    samples = [ModelSample(prompt_id="p1", output='{"ok": true}', model_name="m1")]
    emb = compare_embedding_vectors({"p1": [1.0, 0.0]}, {"p1": [0.0, 1.0]}, min_mean_similarity=0.9)
    conv = analyze_conversation_drift(
        [ConversationRecord("c1", (ConversationTurn("assistant", "ok"),))],
        [ConversationRecord("c1", (ConversationTurn("assistant", "ok"),))],
    )
    tt = compare_tool_traces(
        [ToolEvent(0, "read", "a@1", start=0, end=5)],
        [ToolEvent(0, "read", "a@1", start=0, end=5)],
    )
    rep = run_full_audit(
        samples,
        samples,
        embedding_report=emb,
        conversation_report=conv,
        tool_trace_diff=tt,
    )
    comment = export_audit_pr_comment(rep)
    assert "<!-- llm-drift-bench:audit -->" in comment
    assert "| Embeddings | FAIL |" in comment
    assert "| Conversations | PASS |" in comment
    assert "| Agent tool trace | PASS |" in comment
    assert "<details><summary>Failure reasons" in comment

    examples = Path(__file__).resolve().parents[1] / "examples"
    assert (
        main(
            [
                "audit",
                "-s",
                str(examples / "extraction-suite.json"),
                "-b",
                str(examples / "baseline.jsonl"),
                "-c",
                str(examples / "candidate-pass.jsonl"),
                "--format",
                "pr-comment",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "### LLM Drift Bench Audit: **PASS**" in out


