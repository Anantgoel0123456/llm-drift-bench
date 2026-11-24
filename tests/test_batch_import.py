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
