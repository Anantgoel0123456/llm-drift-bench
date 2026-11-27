"""Tests for deterministic PII and credential redaction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.models import ModelSample
from llm_drift_bench.redaction import redact_samples, redact_text


def test_redact_text_secrets_and_pii_preserves_valid_json() -> None:
    payload = json.dumps({
        "email": "alice.smith@example.com",
        "ssn": "123-45-6789",
        "ip": "192.168.1.42",
        "card": "4111-1111-1111-1111",
        "not_a_card": "4111-1111-1111-1112",
        "key": "sk-proj-1234567890abcdefghijklmnop",
        "ant": "sk-ant-api03-1234567890abcdefghij",
    })
    cleaned, counts = redact_text(payload)
    parsed = json.loads(cleaned)
    assert parsed["email"] == "[REDACTED:EMAIL]"
    assert parsed["ssn"] == "[REDACTED:SSN]"
    assert parsed["ip"] == "[REDACTED:IPV4]"
    assert parsed["card"] == "[REDACTED:CREDIT_CARD]"
    assert parsed["not_a_card"] == "4111-1111-1111-1112"
    assert parsed["key"] == "[REDACTED:OPENAI_KEY]"
    assert parsed["ant"] == "[REDACTED:ANTHROPIC_KEY]"
    assert counts["CREDIT_CARD"] == 1
    assert counts["OPENAI_KEY"] == 1
    assert counts["ANTHROPIC_KEY"] == 1

    # Secrets-only mode leaves PII untouched while still redacting credentials.
    secrets_only, sec_counts = redact_text(payload, include_pii=False)
    assert "alice.smith@example.com" in secrets_only
    assert "[REDACTED:OPENAI_KEY]" in secrets_only
    assert "EMAIL" not in sec_counts


def test_redact_samples_cleans_outputs_and_nested_metadata() -> None:
    samples = [
        ModelSample(
            prompt_id="p1",
            output="Contact bob@corp.io with key AKIA1234567890ABCDEF",
            metadata={"headers": ["Bearer abcdefghijklmnopqrstuvwxyz123456"]},
        ),
        ModelSample(prompt_id="p2", output='{"total": 100}'),
    ]
    cleaned, report = redact_samples(samples)
    assert report.total_samples == 2
    assert report.redacted_samples == 1
    assert report.total_replacements == 3
    assert "[REDACTED:EMAIL]" in cleaned[0].output
    assert "[REDACTED:AWS_ACCESS_KEY]" in cleaned[0].output
    assert cleaned[0].metadata["headers"] == ["[REDACTED:BEARER_TOKEN]"]
    assert cleaned[1].output == '{"total": 100}'


def test_redact_jwt_database_url_and_custom_safe_patterns() -> None:
    import pytest

    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue12345"
    db = "postgres://app:mysecret@db.internal:5432/orders"
    text = json.dumps({"jwt": jwt, "dsn": db, "acct": "CUST-12345678"})
    cleaned, counts = redact_text(
        text,
        include_pii=False,
        custom_patterns={"customer_id": r"CUST-[0-9]{8}"},
    )
    parsed = json.loads(cleaned)
    assert parsed["jwt"] == "[REDACTED:JWT]"
    assert parsed["dsn"] == "[REDACTED:DATABASE_URL]"
    assert parsed["acct"] == "[REDACTED:CUSTOMER_ID]"
    assert counts == {"CUSTOMER_ID": 1, "DATABASE_URL": 1, "JWT": 1}

    # ReDoS-unsafe custom pattern is rejected by the safe regex policy.
    with pytest.raises(ValueError, match="Unsafe custom redaction pattern"):
        redact_text("abc", custom_patterns={"bad": r"(a+)+$"})
    with pytest.raises(ValueError, match="alphanumeric"):
        redact_text("abc", custom_patterns={"bad label!": r"abc"})


def test_redaction_luhn_custom_errors_report_dict_and_scalar_metadata() -> None:
    import pytest

    from llm_drift_bench.redaction import _luhn_valid

    # Luhn invalid length/non-digit AND valid card whose doubled digits exceed 9 (5*2=10 -> 1)
    assert _luhn_valid("12345") is False
    assert _luhn_valid("5555555555554444") is True

    cleaned_card, card_counts = redact_text("Card: 5555-5555-5555-4444")
    assert cleaned_card == "Card: [REDACTED:CREDIT_CARD]"
    assert card_counts == {"CREDIT_CARD": 1}

    # Custom pattern validation: empty, >256 chars, invalid regex syntax
    with pytest.raises(ValueError, match="non-empty string"):
        redact_text("abc", custom_patterns={"empty": ""})
    with pytest.raises(ValueError, match="256 characters"):
        redact_text("abc", custom_patterns={"huge": "a" * 257})
    with pytest.raises(ValueError, match="Invalid custom redaction pattern"):
        redact_text("abc", custom_patterns={"broken": "[unclosed"})

    # Scalar metadata values pass through unchanged; RedactionReport.to_dict serializes cleanly
    samples = [
        ModelSample(
            prompt_id="p1",
            output="email: dev@example.com",
            metadata={"retries": 2, "cached": False, "note": "dev@example.com"},
        )
    ]
    cleaned, report = redact_samples(samples)
    assert cleaned[0].metadata == {"retries": 2, "cached": False, "note": "[REDACTED:EMAIL]"}
    assert report.to_dict() == {
        "total_samples": 1,
        "redacted_samples": 1,
        "total_replacements": 2,
        "counts_by_kind": {"EMAIL": 2},
    }


def test_redact_conversations_scrubs_turns_and_metadata() -> None:
    from llm_drift_bench.conversation import ConversationRecord, ConversationTurn
    from llm_drift_bench.redaction import redact_conversations

    convs = [
        ConversationRecord(
            conversation_id="c1",
            model_name="m1",
            turns=(
                ConversationTurn(
                    role="user",
                    content="My email is alice@corp.io and key sk-proj-1234567890abcdefghijklmnop",
                    metadata={"ip": "10.0.0.1"},
                ),
                ConversationTurn(role="assistant", content='{"status": "ok"}'),
            ),
            metadata={"owner": "bob@corp.io"},
        ),
        ConversationRecord(
            conversation_id="c2",
            model_name="m1",
            turns=(ConversationTurn(role="assistant", content="clean reply"),),
        ),
    ]
    cleaned, rep = redact_conversations(convs)
    assert rep.total_samples == 2
    assert rep.redacted_samples == 1
    assert rep.total_replacements == 4
    assert "[REDACTED:EMAIL]" in cleaned[0].turns[0].content
    assert "[REDACTED:OPENAI_KEY]" in cleaned[0].turns[0].content
    assert cleaned[0].turns[0].metadata["ip"] == "[REDACTED:IPV4]"
    assert cleaned[0].metadata["owner"] == "[REDACTED:EMAIL]"
    assert cleaned[1].turns[0].content == "clean reply"


def test_redact_tool_events_and_cli_traces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_drift_bench.cli import main
    from llm_drift_bench.redaction import redact_tool_events, serialize_tool_events_jsonl
    from llm_drift_bench.tool_events import ToolEvent
    from llm_drift_bench.tool_log import load_tool_events

    events = [
        ToolEvent(
            0,
            "read",
            "db/postgres://u:secret@db.internal/app sk-ant-api03-1234567890abcdefghij",
            start=0,
            end=20,
        ),
        ToolEvent(1, "edit", "src/clean.py@1"),
    ]
    cleaned, rep = redact_tool_events(events)
    assert rep.total_samples == 2 and rep.redacted_samples == 1
    assert "[REDACTED:ANTHROPIC_KEY]" in cleaned[0].resource
    assert cleaned[1].resource == "src/clean.py@1"
    assert serialize_tool_events_jsonl([]) == ""

    raw_trace = tmp_path / "raw_trace.jsonl"
    out_trace = tmp_path / "clean_trace.jsonl"
    raw_trace.write_text(
        '{"kind": "read", "resource": "mail/alice@corp.io/sk-ant-api03-1234567890abcdefghij", '
        '"start": 0, "end": 10}\n'
        '{"kind": "edit", "resource": "src/a.py@1"}\n',
        encoding="utf-8",
    )
    assert (
        main(["redact", "--traces", "-i", str(raw_trace), "-o", str(out_trace)]) == 0
    )
    assert "tool event(s)" in capsys.readouterr().out
    reloaded = load_tool_events(out_trace)
    assert "[REDACTED:EMAIL]" in reloaded[0].resource
    assert "[REDACTED:ANTHROPIC_KEY]" in reloaded[0].resource
    assert reloaded[1].resource == "src/a.py@1"

    # Reject combining --conversations and --traces
    assert (
        main(
            [
                "redact",
                "--conversations",
                "--traces",
                "-i",
                str(raw_trace),
                "-o",
                str(out_trace),
            ]
        )
        == 2
    )



