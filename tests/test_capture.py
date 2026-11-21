"""Unit and CLI tests for the OpenAI-compatible capture collector."""

from __future__ import annotations

import io
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from llm_drift_bench.capture import CaptureConfig, capture_suite, serialize_samples_jsonl
from llm_drift_bench.cli import main
from llm_drift_bench.engine import load_model_samples
from llm_drift_bench.models import PromptCase

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_capture_config_validation() -> None:
    cfg = CaptureConfig(endpoint="http://localhost:11434/v1", model="llama3:8b-q4")
    assert cfg.chat_completions_url == "http://localhost:11434/v1/chat/completions"
    assert cfg.endpoint_host == "localhost:11434"

    full = CaptureConfig(endpoint="https://api.example.com/v1/chat/completions", model="m")
    assert full.chat_completions_url == "https://api.example.com/v1/chat/completions"

    with pytest.raises(ValueError, match="http://"):
        CaptureConfig(endpoint="ftp://bad.host", model="m")
    with pytest.raises(ValueError, match="timeout_sec"):
        CaptureConfig(endpoint="http://localhost:8000/v1", model="m", timeout_sec=0)
    with pytest.raises(ValueError, match="max_retries"):
        CaptureConfig(endpoint="http://localhost:8000/v1", model="m", max_retries=10)


def test_capture_suite_retries_and_preserves_provenance_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "super-secret-token-value")
    calls = {"count": 0}
    sleeps: list[float] = []

    def fake_transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        calls["count"] += 1
        assert headers["Authorization"] == "Bearer super-secret-token-value"
        assert payload["seed"] == 42
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                url, 429, "Too Many Requests", hdrs={}, fp=io.BytesIO(b"")  # type: ignore[arg-type]
            )
        return {
            "choices": [
                {"message": {"content": '{"total": 100}'}, "finish_reason": "stop"}
            ],
            "usage": {"completion_tokens": 12},
            "system_fingerprint": "fp_abc123",
        }

    cfg = CaptureConfig(
        endpoint="http://localhost:11434/v1",
        model="llama3:8b",
        api_key_env="MY_TEST_KEY",
        seed=42,
        system_prompt="Return JSON only.",
        max_retries=2,
    )
    prompts = [PromptCase(prompt_id="p1", prompt="Extract total.")]
    samples = capture_suite(prompts, cfg, transport=fake_transport, sleep_fn=sleeps.append)
    assert calls["count"] == 2
    assert len(sleeps) == 1
    assert len(samples) == 1
    sample = samples[0]
    assert sample.output == '{"total": 100}'
    assert sample.tokens_generated == 12
    assert sample.latency_ms is not None and sample.latency_ms >= 0.0
    assert sample.metadata["system_fingerprint"] == "fp_abc123"
    serialized = serialize_samples_jsonl(samples)
    assert "super-secret-token-value" not in serialized


def test_capture_suite_missing_api_key_env_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cfg = CaptureConfig(endpoint="http://localhost:11434/v1", model="m", api_key_env="MISSING_KEY")
    with pytest.raises(ValueError, match="MISSING_KEY"):
        capture_suite([PromptCase(prompt_id="p1", prompt="hi")], cfg)


def test_cli_capture_writes_loadable_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def fake_transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        return {
            "choices": [
                {"message": {"content": '{"total": 99}'}, "finish_reason": "stop"}
            ],
            "usage": {"completion_tokens": 8},
        }

    monkeypatch.setattr("llm_drift_bench.capture._default_transport", fake_transport)
    out_file = tmp_path / "captures" / "cand.jsonl"
    rc = main(
        [
            "capture",
            "--suite",
            str(EXAMPLES / "extraction-suite.json"),
            "--endpoint",
            "http://localhost:11434/v1",
            "--model",
            "llama3:8b-q4",
            "--seed",
            "7",
            "-o",
            str(out_file),
        ]
    )
    assert rc == 0
    assert "Captured 2 prompt(s)" in capsys.readouterr().out
    loaded = load_model_samples(out_file)
    assert len(loaded) == 2
    assert all(s.model_name == "llama3:8b-q4" and s.tokens_generated == 8 for s in loaded)


def test_capture_concurrency_json_mode_and_multi_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    seen_payloads: list[dict[str, Any]] = []

    def fake_transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        seen_payloads.append(payload)
        return {
            "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 4},
        }

    monkeypatch.setattr("llm_drift_bench.capture._default_transport", fake_transport)
    out_base = tmp_path / "batch.jsonl"
    rc = main(
        [
            "capture",
            "--suite",
            str(EXAMPLES / "extraction-suite.json"),
            "--endpoint",
            "http://localhost:11434/v1",
            "--model",
            "llama3:8b-q4",
            "--json-mode",
            "--concurrency",
            "2",
            "--runs",
            "2",
            "-o",
            str(out_base),
        ]
    )
    assert rc == 0
    assert (tmp_path / "batch-run1.jsonl").is_file()
    assert (tmp_path / "batch-run2.jsonl").is_file()
    assert len(seen_payloads) == 4
    assert all(p.get("response_format") == {"type": "json_object"} for p in seen_payloads)
