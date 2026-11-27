"""Tests for multi-turn conversation trajectory drift analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.conversation import (
    ConversationRecord,
    ConversationTurn,
    analyze_conversation_drift,
    export_conversation_html,
    format_conversation_markdown,
    format_conversation_text,
    load_conversations,
)


def _conv(cid: str, assistant_replies: list[str], model: str = "m") -> ConversationRecord:
    turns: list[ConversationTurn] = []
    for idx, reply in enumerate(assistant_replies, start=1):
        turns.append(ConversationTurn(role="user", content=f"Question {idx}"))
        turns.append(ConversationTurn(role="assistant", content=reply))
    return ConversationRecord(conversation_id=cid, turns=tuple(turns), model_name=model)


def test_conversation_snowball_drift_escalation_and_refusal_flip(tmp_path: Path) -> None:
    base = [
        _conv(
            "conv-1",
            [
                '{"step": 1, "status": "ok"}',
                '{"step": 2, "status": "ok"}',
                "Here is the complete refund confirmation for invoice 42.",
            ],
            model="fp16",
        ),
        _conv("conv-2", ["Hello, how can I help?", "Your order ships tomorrow."], model="fp16"),
    ]
    cand = [
        _conv(
            "conv-1",
            [
                '{"step": 1, "status": "ok"}',
                '{"step": 2, "status": "pending"}',
                "I'm sorry, I cannot assist with that request.",
            ],
            model="q4",
        ),
        _conv("conv-2", ["Hello, how can I help?", "Your order ships tomorrow."], model="q4"),
    ]

    rep = analyze_conversation_drift(
        base,
        cand,
        max_turn_drift=0.15,
        max_mean_drift=0.15,
        max_final_turn_drift=0.30,
        max_turn_count_delta=0,
    )
    assert rep.passed is False
    assert rep.total_conversations == 2
    assert rep.drifted_conversations == 1
    assert rep.total_refusal_flip_turns == 1

    c1 = rep.conversations[0]
    assert c1.drifted is True
    assert c1.first_divergence_turn == 2
    assert c1.escalation_slope is not None and c1.escalation_slope > 0.5
    assert c1.refusal_flip_turns == (3,)
    assert rep.conversations[1].drifted is False

    as_dict = rep.to_dict()
    assert as_dict["passed"] is False
    assert as_dict["conversations"][0]["first_divergence_turn"] == 2

    txt = format_conversation_text(rep)
    assert "Multi-Turn Conversation Trajectory Drift [FAILED]" in txt
    assert "first_div=turn 2" in txt
    md = format_conversation_markdown(rep)
    assert "# Multi-Turn Conversation Trajectory Drift: FAILED" in md
    assert "| `conv-1` |" in md

    html_out = tmp_path / "dash" / "conv.html"
    html = export_conversation_html(rep, output_path=html_out)
    assert html_out.read_text(encoding="utf-8") == html
    assert "Multi-Turn Conversation Trajectory Drift" in html
    assert "default-src 'none'" in html
    assert "<svg viewBox=" in html and "<polyline" in html and "<title>T1=" in html
    assert "<script" not in html.lower()

    # Zero-assistant-turn conversation renders n/a trajectory cleanly
    user_only = [ConversationRecord("u1", (ConversationTurn("user", "ping"),))]
    empty_html = export_conversation_html(analyze_conversation_drift(user_only, user_only))
    assert "<span class='muted'>n/a</span>" in empty_html


def test_load_conversations_json_wrapper_array_and_jsonl(tmp_path: Path) -> None:
    rec = _conv("c1", ["answer one"])
    d = rec.to_dict()
    assert ConversationTurn(role="user", content="hi", metadata={"k": 1}).to_dict()["metadata"] == {
        "k": 1
    }

    j_arr = tmp_path / "arr.json"
    j_arr.write_text(json.dumps([d]), encoding="utf-8")
    assert len(load_conversations(j_arr)) == 1

    j_wrap = tmp_path / "wrap.json"
    j_wrap.write_text(json.dumps({"conversations": [d]}), encoding="utf-8")
    assert len(load_conversations(j_wrap)) == 1

    j_lines = tmp_path / "lines.jsonl"
    j_lines.write_text(json.dumps({**d, "metadata": {"route": "help"}}) + "\n\n", encoding="utf-8")
    loaded = load_conversations(j_lines)
    assert loaded[0].metadata == {"route": "help"} and loaded[0].to_dict()["metadata"] == {
        "route": "help"
    }

    # Passing identical trajectories + skip_levenshtein long-text path
    long_c = _conv("long", ["word " * 4000])
    rep_long = analyze_conversation_drift([long_c], [long_c], long_text_mode="skip_levenshtein")
    assert rep_long.passed is True and rep_long.mean_trajectory_drift == 0.0
    assert "PASSED" in format_conversation_text(rep_long)
    assert "PASSED" in format_conversation_markdown(rep_long)


def test_conversation_validation_and_error_guards(tmp_path: Path) -> None:
    ok = [_conv("c1", ["hi"])]

    with pytest.raises(ValueError, match="role"):
        ConversationTurn(role="narrator", content="hi")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="content must be a string"):
        ConversationTurn(role="user", content=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metadata must be a dictionary"):
        ConversationTurn(role="user", content="hi", metadata="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        ConversationRecord(conversation_id="  ", turns=(ConversationTurn("user", "hi"),))
    with pytest.raises(ValueError, match="turns must be non-empty"):
        ConversationRecord(conversation_id="c1", turns=())
    with pytest.raises(ValueError, match="model_name"):
        ConversationRecord(
            conversation_id="c1", turns=(ConversationTurn("user", "hi"),), model_name=" "
        )

    for fname, body, match in (
        ("empty.jsonl", "", "empty"),
        ("empty_arr.json", "[]", "no records"),
        ("bad_top.json", '{"bad": 1}', "JSON array"),
        ("non_dict.json", "[123]", "JSON object"),
        ("no_cid.jsonl", '{"turns": [{"role": "user", "content": "hi"}]}\n', "conversation_id"),
        ("no_turns.jsonl", '{"conversation_id": "c1", "turns": []}\n', "non-empty list"),
        ("bad_turn.jsonl", '{"conversation_id": "c1", "turns": [1]}\n', "turn #1"),
        (
            "bad_role.jsonl",
            '{"conversation_id": "c1", "turns": [{"role": "x", "content": "a"}]}\n',
            "invalid role",
        ),
        (
            "bad_content.jsonl",
            '{"conversation_id": "c1", "turns": [{"role": "user", "content": 1}]}\n',
            "content",
        ),
        (
            "bad_tmeta.jsonl",
            '{"conversation_id": "c1", '
            '"turns": [{"role": "user", "content": "a", "metadata": 1}]}\n',
            "metadata",
        ),
        (
            "bad_model.jsonl",
            '{"conversation_id": "c1", "model_name": "", '
            '"turns": [{"role": "user", "content": "a"}]}\n',
            "model_name",
        ),
        (
            "bad_meta.jsonl",
            '{"conversation_id": "c1", "metadata": [], '
            '"turns": [{"role": "user", "content": "a"}]}\n',
            "metadata",
        ),
        (
            "dup.jsonl",
            '{"conversation_id": "c1", "turns": [{"role": "user", "content": "a"}]}\n'
            '{"conversation_id": "c1", "turns": [{"role": "user", "content": "b"}]}\n',
            "duplicate",
        ),
    ):
        f = tmp_path / fname
        f.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_conversations(f)

    with pytest.raises(ValueError, match="non-empty"):
        analyze_conversation_drift([], ok)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        analyze_conversation_drift(ok, ok, max_turn_drift=2.0)
    with pytest.raises(ValueError, match="non-negative integer"):
        analyze_conversation_drift(ok, ok, max_turn_count_delta=-1)
    with pytest.raises(ValueError, match="long_text_mode"):
        analyze_conversation_drift(ok, ok, long_text_mode="bad")  # type: ignore[arg-type]


def test_diff_conversations_per_turn_json_and_text_diffs() -> None:
    from llm_drift_bench.conversation import diff_conversations, format_conversation_diffs_text

    base = [
        _conv("c1", ['{"amount": 10}', '{"status": "ok"}']),
        _conv("c2", ["identical"]),
    ]
    cand = [
        _conv("c1", ['{"amount": 10}', '{"status": "error"}']),
        _conv("c2", ["identical"]),
    ]
    all_diffs = diff_conversations(base, cand, only_changed=False)
    assert len(all_diffs) == 2
    assert all_diffs[0].changed is True and all_diffs[1].changed is False
    assert all_diffs[0].to_dict()["turn_diffs"][1]["turn_index"] == 2

    only_diffs = diff_conversations(base, cand, only_changed=True)
    assert [d.conversation_id for d in only_diffs] == ["c1"]
    assert [t.turn_index for t in only_diffs[0].turn_diffs] == [2]
    rendered = format_conversation_diffs_text(only_diffs)
    assert "Conversation c1 (CHANGED)" in rendered and "$.status" in rendered

    # Identical collection -> empty only_changed diff; user-only conversation has no compared turns
    assert format_conversation_diffs_text(diff_conversations(base, base, only_changed=True)) == (
        "No changed conversation turns.\n"
    )
    user_only = [ConversationRecord("u1", (ConversationTurn("user", "hi"),))]
    assert "no compared assistant turn diffs" in format_conversation_diffs_text(
        diff_conversations(user_only, user_only)
    )
    with pytest.raises(ValueError, match="non-empty"):
        diff_conversations([], base)


def test_export_conversation_pr_comment_and_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from llm_drift_bench.cli import main
    from llm_drift_bench.conversation import export_conversation_pr_comment

    def _rec(cid: str, reply: str) -> ConversationRecord:
        return ConversationRecord(
            cid, (ConversationTurn("user", "u"), ConversationTurn("assistant", reply))
        )

    pass_md = export_conversation_pr_comment(
        analyze_conversation_drift([_rec("c1", "ok")], [_rec("c1", "ok")])
    )
    assert "<!-- llm-drift-bench:conversations -->" in pass_md
    assert "**PASS**" in pass_md and "<details>" not in pass_md

    base_many = [_rec(f"c{i}", '{"ok": true}') for i in range(6)]
    cand_many = [_rec(f"c{i}", "I'm sorry, I cannot assist with that.") for i in range(6)]
    fail_md = export_conversation_pr_comment(analyze_conversation_drift(base_many, cand_many))
    assert "**FAIL**" in fail_md and "1 more drifted conversation(s)" in fail_md

    examples = Path(__file__).resolve().parents[1] / "examples"
    assert (
        main(
            [
                "check-conversations",
                "-b",
                str(examples / "conversations-baseline.jsonl"),
                "-c",
                str(examples / "conversations-candidate-drift.jsonl"),
                "--format",
                "pr-comment",
            ]
        )
        == 1
    )
    cli_out = capsys.readouterr().out
    assert "### Multi-Turn Conversation Drift: **FAIL**" in cli_out
    assert "ORD-1042" not in cli_out

