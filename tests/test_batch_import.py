"""Tests for offline OpenAI Batch, Anthropic Batch, and CSV capture importers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.batch_import import (
    import_anthropic_batch_jsonl,
    import_csv_captures,
    import_openai_batch_jsonl,
)


def test_import_openai_batch_jsonl_chat_and_responses(tmp_path: Path) -> None:
    batch_file = tmp_path / "openai_batch.jsonl"
    rows = [
        {
            "id": "batch_req_2",
            "custom_id": "p2",
            "response": {
                "status_code": 200,
                "body": {
                    "model": "gpt-4o-mini",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "B"}]}
                    ],
                    "usage": {"output_tokens": 7},
                },
            },
            "error": None,
        },
        {
            "id": "batch_req_1",
            "custom_id": "p1",
            "response": {
                "status_code": 200,
                "body": {
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"content": '{"total": 10}'}}],
                    "usage": {"completion_tokens": 12},
                },
            },
            "error": None,
        },
    ]
    batch_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    samples = import_openai_batch_jsonl(batch_file)
    assert [s.prompt_id for s in samples] == ["p1", "p2"]
    assert samples[0].output == '{"total": 10}' and samples[0].tokens_generated == 12
    assert samples[1].output == "B" and samples[1].tokens_generated == 7
    assert samples[0].model_name == "gpt-4o-mini"
    assert samples[0].latency_ms is None

    # Failed batch rows fail explicitly rather than being silently dropped.
    bad = tmp_path / "bad_openai.jsonl"
    bad.write_text(
        json.dumps({"custom_id": "p1", "response": {"status_code": 429, "body": {}}, "error": None})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="HTTP status 429"):
        import_openai_batch_jsonl(bad)


def test_import_anthropic_batch_jsonl_and_errors(tmp_path: Path) -> None:
    ant_file = tmp_path / "anthropic_batch.jsonl"
    ant_file.write_text(
        json.dumps(
            {
                "custom_id": "ticket-01",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "model": "claude-sonnet-4-5",
                        "content": [{"type": "text", "text": '{"priority": "high"}'}],
                        "usage": {"output_tokens": 9},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = import_anthropic_batch_jsonl(ant_file)
    assert len(samples) == 1
    assert samples[0].prompt_id == "ticket-01"
    assert samples[0].model_name == "claude-sonnet-4-5"
    assert samples[0].tokens_generated == 9

    errored = tmp_path / "ant_err.jsonl"
    errored.write_text(
        json.dumps({"custom_id": "p1", "result": {"type": "errored"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not succeed"):
        import_anthropic_batch_jsonl(errored)


def test_import_csv_captures_preserves_missing_metrics_and_rejects_dups(tmp_path: Path) -> None:
    csv_file = tmp_path / "captures.csv"
    csv_file.write_text(
        'prompt_id,model_name,latency_ms,tokens_generated,output\n'
        'p2,q4,45.5,20,"{""ok"": true}"\n'
        'p1,q4,,,plain text\n',
        encoding="utf-8",
    )
    samples = import_csv_captures(csv_file)
    assert [s.prompt_id for s in samples] == ["p1", "p2"]
    assert samples[0].latency_ms is None and samples[0].tokens_generated is None
    assert samples[1].latency_ms == 45.5 and samples[1].tokens_generated == 20
    assert samples[1].output == '{"ok": true}'

    dup_csv = tmp_path / "dup.csv"
    dup_csv.write_text("prompt_id,output\np1,a\np1,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        import_csv_captures(dup_csv)

    bad_hdr = tmp_path / "bad_hdr.csv"
    bad_hdr.write_text("id,text\np1,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header columns"):
        import_csv_captures(bad_hdr)

    # Multipart OpenAI chat message.content list is supported cleanly.
    mp = tmp_path / "multipart.jsonl"
    mp.write_text(
        json.dumps(
            {
                "custom_id": "m1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": "gpt-4o",
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"type": "text", "text": '{"a": '},
                                        {"type": "text", "text": "1}"},
                                    ]
                                }
                            }
                        ],
                    },
                },
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mp_samples = import_openai_batch_jsonl(mp)
    assert mp_samples[0].output == '{"a": 1}'

    # Invalid negative latency in CSV reports exact row location.
    neg_csv = tmp_path / "neg.csv"
    neg_csv.write_text("prompt_id,output,latency_ms\np1,ok,-5.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"row 2"):
        import_csv_captures(neg_csv)


def test_batch_import_error_branches_and_edge_cases(tmp_path: Path) -> None:
    def _write(name: str, text: str) -> Path:
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return p

    # Empty and non-object JSONL
    with pytest.raises(ValueError, match="empty"):
        import_openai_batch_jsonl(_write("empty.jsonl", "\n   \n"))
    with pytest.raises(ValueError, match="JSON object"):
        import_openai_batch_jsonl(_write("arr.jsonl", "[1, 2]\n"))

    # OpenAI error paths
    openai_bad_cases = [
        ({"custom_id": "", "response": {"status_code": 200, "body": {}}}, "custom_id"),
        ({"custom_id": "p1", "error": {"message": "boom"}}, "API error"),
        ({"custom_id": "p1", "response": "bad"}, "missing 'response'"),
        ({"custom_id": "p1", "response": {"status_code": 200, "body": "bad"}}, "response.body"),
        (
            {"custom_id": "p1", "response": {"status_code": 200, "body": {"choices": ["x"]}}},
            "choice entry",
        ),
        (
            {"custom_id": "p1", "response": {"status_code": 200, "body": {"choices": [{}]}}},
            "message object",
        ),
        (
            {
                "custom_id": "p1",
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": []}}]},
                },
            },
            "no text blocks",
        ),
        (
            {
                "custom_id": "p1",
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": 5}}]},
                },
            },
            "missing string",
        ),
        (
            {"custom_id": "p1", "response": {"status_code": 200, "body": {"output": ["bad"]}}},
            "neither chat",
        ),
    ]
    for idx, (row, match) in enumerate(openai_bad_cases):
        with pytest.raises(ValueError, match=match):
            import_openai_batch_jsonl(_write(f"oa_{idx}.jsonl", json.dumps(row) + "\n"))

    dup_oa = _write(
        "dup_oa.jsonl",
        "\n".join(
            json.dumps(
                {
                    "custom_id": "p1",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": "ok"}}]},
                    },
                }
            )
            for _ in range(2)
        )
        + "\n",
    )
    with pytest.raises(ValueError, match="duplicate custom_id"):
        import_openai_batch_jsonl(dup_oa)

    # Anthropic error paths
    ant_bad_cases = [
        ({"custom_id": ""}, "custom_id"),
        ({"custom_id": "p1", "result": "bad"}, "missing 'result'"),
        ({"custom_id": "p1", "result": {"type": "succeeded", "message": "bad"}}, "result.message"),
        (
            {"custom_id": "p1", "result": {"type": "succeeded", "message": {"content": []}}},
            "empty message.content",
        ),
        (
            {
                "custom_id": "p1",
                "result": {"type": "succeeded", "message": {"content": [{"type": "tool_use"}]}},
            },
            "no text content",
        ),
    ]
    for idx, (row, match) in enumerate(ant_bad_cases):
        with pytest.raises(ValueError, match=match):
            import_anthropic_batch_jsonl(_write(f"ant_{idx}.jsonl", json.dumps(row) + "\n"))

    dup_ant = _write(
        "dup_ant.jsonl",
        "\n".join(
            json.dumps(
                {
                    "custom_id": "p1",
                    "result": {
                        "type": "succeeded",
                        "message": {"content": [{"type": "text", "text": "ok"}]},
                    },
                }
            )
            for _ in range(2)
        )
        + "\n",
    )
    with pytest.raises(ValueError, match="duplicate custom_id"):
        import_anthropic_batch_jsonl(dup_ant)

    # CSV error paths
    with pytest.raises(ValueError, match="empty 'prompt_id'"):
        import_csv_captures(_write("csv_empty_id.csv", "prompt_id,output\n ,hello\n"))
    with pytest.raises(ValueError, match="missing 'output'"):
        import_csv_captures(_write("csv_short.csv", "prompt_id,output\np1\n"))
    with pytest.raises(ValueError, match="invalid 'latency_ms'"):
        import_csv_captures(_write("csv_lat.csv", "prompt_id,output,latency_ms\np1,ok,fast\n"))
    with pytest.raises(ValueError, match="invalid 'tokens_generated'"):
        import_csv_captures(_write("csv_tok.csv", "prompt_id,output,tokens_generated\np1,ok,ten\n"))
    with pytest.raises(ValueError, match="zero data rows"):
        import_csv_captures(_write("csv_nodata.csv", "prompt_id,output\n"))


def test_import_conversations_openai_anthropic_sharegpt_and_errors(tmp_path: Path) -> None:
    from llm_drift_bench.batch_import import (
        import_conversations_jsonl,
        serialize_conversations_jsonl,
    )
    from llm_drift_bench.conversation import load_conversations

    # 1. OpenAI chat format (with developer role, multipart blocks, and null tool-call content)
    oa = tmp_path / "oa_conv.jsonl"
    oa.write_text(
        json.dumps(
            {
                "id": "conv-b",
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "developer", "content": "Be brief."},
                    {"role": "user", "content": ["Hello ", {"type": "text", "text": "world"}]},
                    {"role": "assistant", "content": None},
                    {"role": "function", "content": '{"ok": true}'},
                    {"role": "assistant", "content": "Done!"},
                ],
            }
        )
        + "\n"
        + json.dumps(
            {
                "conversation_id": "conv-a",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    oa_recs = import_conversations_jsonl(oa, source_format="openai-chat")
    assert [r.conversation_id for r in oa_recs] == ["conv-a", "conv-b"]
    assert oa_recs[1].turns[0].role == "system"
    assert oa_recs[1].turns[1].content == "Hello world"
    assert oa_recs[1].turns[2].content == ""
    assert oa_recs[1].turns[3].role == "tool"
    assert oa_recs[1].model_name == "gpt-4o-mini"

    # Round-trip through serialize_conversations_jsonl -> load_conversations
    rt = tmp_path / "rt.jsonl"
    rt.write_text(serialize_conversations_jsonl(oa_recs), encoding="utf-8")
    assert len(load_conversations(rt)) == 2

    # 2. Anthropic messages format with top-level system prompt and content blocks
    ant = tmp_path / "ant_conv.jsonl"
    ant.write_text(
        json.dumps(
            {
                "custom_id": "c1",
                "model": "claude-sonnet-4-5",
                "system": "Extract JSON.",
                "messages": [
                    {"role": "user", "content": "Order 1"},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": '{"id": 1}'}],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ant_recs = import_conversations_jsonl(ant, source_format="anthropic-messages")
    assert ant_recs[0].turns[0].role == "system"
    assert ant_recs[0].turns[0].content == "Extract JSON."
    assert ant_recs[0].turns[2].content == '{"id": 1}'

    # 3. ShareGPT format
    sg = tmp_path / "sg.jsonl"
    sg.write_text(
        json.dumps(
            {
                "id": "sg-1",
                "conversations": [
                    {"from": "system", "value": "Helpful assistant"},
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello!"},
                    {"from": "observation", "value": "tool out"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sg_recs = import_conversations_jsonl(sg, source_format="sharegpt", model_name="llama-q4")
    assert [t.role for t in sg_recs[0].turns] == ["system", "user", "assistant", "tool"]
    assert sg_recs[0].model_name == "llama-q4"

    # Error branches
    with pytest.raises(ValueError, match="source_format"):
        import_conversations_jsonl(sg, source_format="unknown")  # type: ignore[arg-type]

    def _w(name: str, obj: object) -> Path:
        p = tmp_path / name
        p.write_text(
            obj if isinstance(obj, str) else (json.dumps(obj) + "\n"),
            encoding="utf-8",
        )
        return p

    bad_cases: list[tuple[str, object, str]] = [
        ("openai-chat", {"messages": [{"role": "user", "content": "a"}]}, "conversation_id"),
        ("openai-chat", {"id": "c1", "messages": []}, "messages"),
        ("openai-chat", {"id": "c1", "messages": ["bad"]}, "JSON object"),
        ("openai-chat", {"id": "c1", "messages": [{"role": "narrator", "content": "a"}]}, "role"),
        ("openai-chat", {"id": "c1", "messages": [{"role": "user", "content": 123}]}, "content"),
        ("sharegpt", {"id": "c1", "conversations": []}, "conversations"),
        ("sharegpt", {"id": "c1", "conversations": [1]}, "JSON object"),
        (
            "sharegpt",
            {"id": "c1", "conversations": [{"from": "bot", "value": "a"}]},
            "ShareGPT role",
        ),
        ("sharegpt", {"id": "c1", "conversations": [{"from": "human", "value": 9}]}, "value"),
    ]
    for idx, (fmt, body, match) in enumerate(bad_cases):
        with pytest.raises(ValueError, match=match):
            import_conversations_jsonl(
                _w(f"bad_conv_{idx}.jsonl", body),
                source_format=fmt,  # type: ignore[arg-type]
            )

    dup = _w(
        "dup_conv.jsonl",
        json.dumps({"id": "c1", "messages": [{"role": "user", "content": "a"}]})
        + "\n"
        + json.dumps({"id": "c1", "messages": [{"role": "user", "content": "b"}]})
        + "\n",
    )
    with pytest.raises(ValueError, match="duplicate conversation_id"):
        import_conversations_jsonl(dup, source_format="openai-chat")


