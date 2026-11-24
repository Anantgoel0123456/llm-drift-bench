"""Tests for deterministic PII and credential redaction."""

from __future__ import annotations

import json

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

