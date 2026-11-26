"""Unit and CLI tests for the OpenAI-compatible capture collector."""

from __future__ import annotations

import io
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from llm_drift_bench.capture import (
    CaptureConfig,
    capture_suite,
    parse_header_spec,
    serialize_samples_jsonl,
)
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


def test_capture_resume_checkpointing_and_custom_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert parse_header_spec("X-Title: MyBench") == ("X-Title", "MyBench")
    assert parse_header_spec("HTTP-Referer=https://example.com") == (
        "HTTP-Referer",
        "https://example.com",
    )
    with pytest.raises(ValueError, match="CR or LF"):
        parse_header_spec("X-Bad: foo\r\nEvil: bar")
    with pytest.raises(ValueError, match="Authorization"):
        CaptureConfig(
            endpoint="http://localhost:11434/v1",
            model="m",
            api_key_env="KEY",
            extra_headers=(("Authorization", "Bearer nope"),),
        )

    out_file = tmp_path / "resumed.jsonl"
    calls = {"count": 0}
    seen_headers: list[dict[str, str]] = []

    def flaky_transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        calls["count"] += 1
        seen_headers.append(dict(headers))
        if calls["count"] == 2:
            raise urllib.error.HTTPError(
                url, 400, "Bad Request", hdrs={}, fp=io.BytesIO(b"")  # type: ignore[arg-type]
            )
        return {
            "choices": [{"message": {"content": f'{{"call": {calls["count"]}}}'}}],
            "usage": {"completion_tokens": 5},
        }

    monkeypatch.setattr("llm_drift_bench.capture._default_transport", flaky_transport)

    # First run captures prompt 1 to checkpoint then fails on prompt 2 (HTTP 400 non-retryable).
    rc_fail = main([
        "capture",
        "-s", str(EXAMPLES / "extraction-suite.json"),
        "--endpoint", "http://localhost:11434/v1",
        "--model", "llama3:8b",
        "--header", "X-Title: DriftBench",
        "--header", "HTTP-Referer=https://example.com",
        "--resume",
        "-o", str(out_file),
    ])
    assert rc_fail == 2
    assert "HTTP 400" in capsys.readouterr().err
    partial = load_model_samples(out_file)
    assert len(partial) == 1
    assert seen_headers[0]["X-Title"] == "DriftBench"
    assert seen_headers[0]["HTTP-Referer"] == "https://example.com"

    # Second run with --resume skips prompt 1 and only calls endpoint for prompt 2!
    rc_ok = main([
        "capture",
        "-s", str(EXAMPLES / "extraction-suite.json"),
        "--endpoint", "http://localhost:11434/v1",
        "--model", "llama3:8b",
        "--header", "X-Title: DriftBench",
        "--resume",
        "-o", str(out_file),
    ])
    assert rc_ok == 0
    assert "(resumed 1)" in capsys.readouterr().out
    final = load_model_samples(out_file)
    assert len(final) == 2
    assert calls["count"] == 3  # 1 succeeded + 1 failed + 1 resumed


def test_capture_error_branches_and_validation() -> None:
    cfg = CaptureConfig(endpoint="http://localhost:11434/v1", model="m", max_retries=1)
    prompts = [PromptCase(prompt_id="p1", prompt="hi")]

    def timeout_transport(*_: Any) -> dict[str, Any]:
        raise TimeoutError("timed out")

    with pytest.raises(RuntimeError, match="localhost:11434"):
        capture_suite(prompts, cfg, transport=timeout_transport, sleep_fn=lambda _: None)

    with pytest.raises(ValueError, match="missing non-empty 'choices'"):
        capture_suite(prompts, cfg, transport=lambda *_: {})

    with pytest.raises(ValueError, match="empty"):
        capture_suite([], cfg)


def test_capture_retry_after_connection_reset_and_anthropic_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "sk-ant-secret-value")
    sleeps: list[float] = []
    attempts = {"n": 0}
    seen: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def anthropic_transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        del timeout
        attempts["n"] += 1
        seen.append((url, payload, dict(headers)))
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(
                url,
                429,
                "Rate limited",
                hdrs={"Retry-After": "3"},  # type: ignore[arg-type]
                fp=io.BytesIO(b""),
            )
        if attempts["n"] == 2:
            raise ConnectionResetError("peer reset")
        return {
            "content": [{"type": "text", "text": '{"status": "ok"}'}],
            "stop_reason": "end_turn",
            "usage": {"output_tokens": 9},
        }

    cfg = CaptureConfig(
        endpoint="https://api.anthropic.com/v1",
        model="claude-3-5-sonnet",
        api_key_env="ANTHROPIC_TEST_KEY",
        provider="anthropic",
        system_prompt="Return JSON.",
        max_retries=3,
    )
    samples = capture_suite(
        [PromptCase(prompt_id="p1", prompt="Hi")],
        cfg,
        transport=anthropic_transport,
        sleep_fn=sleeps.append,
    )
    assert len(samples) == 1
    assert samples[0].output == '{"status": "ok"}'
    assert samples[0].tokens_generated == 9
    assert sleeps[0] == 3.0  # Honored Retry-After header!
    assert seen[0][0] == "https://api.anthropic.com/v1/messages"
    assert seen[0][2]["x-api-key"] == "sk-ant-secret-value"
    assert seen[0][2]["anthropic-version"] == "2023-06-01"
    assert seen[0][1]["system"] == "Return JSON."


def test_capture_gemini_provider_url_headers_json_mode_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_TEST_KEY", "AIza-test-secret-key")
    seen: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def gemini_transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        del timeout
        seen.append((url, payload, dict(headers)))
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": '{"ok": true}'}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"candidatesTokenCount": 11},
        }

    cfg = CaptureConfig(
        endpoint="https://generativelanguage.googleapis.com",
        model="gemini-2.5-flash",
        api_key_env="GEMINI_TEST_KEY",
        provider="gemini",
        system_prompt="Return strict JSON.",
        json_mode=True,
        temperature=0.0,
        seed=42,
        max_tokens=256,
    )
    samples = capture_suite(
        [PromptCase(prompt_id="g1", prompt="Hi")],
        cfg,
        transport=gemini_transport,
    )
    assert len(samples) == 1
    assert samples[0].output == '{"ok": true}'
    assert samples[0].tokens_generated == 11
    url, body, headers = seen[0]
    assert url == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert headers["x-goog-api-key"] == "AIza-test-secret-key"
    assert "AIza-test-secret-key" not in url
    assert body["systemInstruction"] == {"parts": [{"text": "Return strict JSON."}]}
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["maxOutputTokens"] == 256
    assert body["generationConfig"]["seed"] == 42

    with pytest.raises(ValueError, match="x-goog-api-key"):
        CaptureConfig(
            endpoint="https://generativelanguage.googleapis.com",
            model="gemini-2.5-flash",
            api_key_env="GEMINI_TEST_KEY",
            provider="gemini",
            extra_headers=(("x-goog-api-key", "dup"),),
        )


def test_capture_rejects_duplicates_handles_multipart_content_and_top_logprobs() -> None:
    with pytest.raises(ValueError, match="top_logprobs"):
        CaptureConfig(endpoint="http://localhost:11434/v1", model="m", top_logprobs=0)
    with pytest.raises(ValueError, match="top_logprobs"):
        CaptureConfig(endpoint="http://localhost:11434/v1", model="m", top_logprobs=21)
    with pytest.raises(ValueError, match="top_logprobs"):
        CaptureConfig(endpoint="http://localhost:11434/v1", model="m", top_logprobs=True)
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        CaptureConfig(
            endpoint="https://api.anthropic.com/v1",
            model="claude-3-5-sonnet",
            provider="anthropic",
            top_logprobs=5,
        )

    cfg = CaptureConfig(
        endpoint="http://localhost:11434/v1",
        model="llama3:8b",
        top_logprobs=3,
    )
    with pytest.raises(ValueError, match="Duplicate prompt_id in suite"):
        capture_suite(
            [PromptCase(prompt_id="p1", prompt="a"), PromptCase(prompt_id="p1", prompt="b")],
            cfg,
        )

    from llm_drift_bench.models import ModelSample

    with pytest.raises(ValueError, match="Duplicate prompt_id in existing_samples"):
        capture_suite(
            [PromptCase(prompt_id="p1", prompt="a")],
            cfg,
            existing_samples=[
                ModelSample(prompt_id="p1", output="ok", model_name="llama3:8b"),
                ModelSample(prompt_id="p1", output="ok2", model_name="llama3:8b"),
            ],
        )

    seen_payloads: list[dict[str, Any]] = []

    def multipart_logprobs_transport(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        del url, headers, timeout
        seen_payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"label": '},
                            {"type": "output_text", "text": '"high"}'},
                        ]
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "high",
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": "high", "logprob": -0.1},
                                    {"token": "low", "logprob": -2.3},
                                    {"token": "bad", "logprob": float("nan")},
                                ],
                            }
                        ]
                    },
                }
            ],
            "usage": {"completion_tokens": 5},
        }

    samples = capture_suite(
        [PromptCase(prompt_id="p1", prompt="classify")],
        cfg,
        transport=multipart_logprobs_transport,
    )
    assert seen_payloads[0]["logprobs"] is True
    assert seen_payloads[0]["top_logprobs"] == 3
    assert samples[0].output == '{"label": "high"}'
    lp = samples[0].metadata["logprobs"]
    assert len(lp) == 1
    assert lp[0]["token"] == "high"
    assert len(lp[0]["top_logprobs"]) == 2


def test_capture_config_url_variants_and_validation_edge_cases() -> None:
    assert (
        CaptureConfig(
            endpoint="https://api.anthropic.com/v1/messages",
            model="claude",
            provider="anthropic",
        ).chat_completions_url
        == "https://api.anthropic.com/v1/messages"
    )
    assert (
        CaptureConfig(
            endpoint="https://api.anthropic.com",
            model="claude",
            provider="anthropic",
        ).chat_completions_url
        == "https://api.anthropic.com/v1/messages"
    )
    assert (
        CaptureConfig(
            endpoint="https://generativelanguage.googleapis.com/v1beta/models/g:generateContent",
            model="models/g",
            provider="gemini",
        ).chat_completions_url
        == "https://generativelanguage.googleapis.com/v1beta/models/g:generateContent"
    )
    assert (
        CaptureConfig(
            endpoint="https://generativelanguage.googleapis.com/v1beta",
            model="models/gemini-2.5-flash",
            provider="gemini",
        ).chat_completions_url
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )

    with pytest.raises(ValueError, match="model must be"):
        CaptureConfig(endpoint="http://localhost:8000/v1", model="   ")
    with pytest.raises(ValueError, match="temperature"):
        CaptureConfig(endpoint="http://localhost:8000/v1", model="m", temperature=-0.1)
    with pytest.raises(ValueError, match="max_tokens"):
        CaptureConfig(endpoint="http://localhost:8000/v1", model="m", max_tokens=0)
    with pytest.raises(ValueError, match="json_mode"):
        CaptureConfig(endpoint="http://localhost:8000/v1", model="m", json_mode="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="concurrency"):
        CaptureConfig(endpoint="http://localhost:8000/v1", model="m", concurrency=0)
    with pytest.raises(ValueError, match="provider"):
        CaptureConfig(endpoint="http://localhost:8000/v1", model="m", provider="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="extra_headers"):
        CaptureConfig(endpoint="http://localhost:8000/v1", model="m", extra_headers=(("", "v"),))
    with pytest.raises(ValueError, match="CR or LF"):
        CaptureConfig(
            endpoint="http://localhost:8000/v1", model="m", extra_headers=(("X-Test", "a\nb"),)
        )
    with pytest.raises(ValueError, match="syntax"):
        parse_header_spec("NoSeparatorHeader")
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_header_spec(": value")


def test_default_transport_and_extract_completion_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_drift_bench.capture import (
        _default_transport,
        _extract_completion,
        _retry_after_delay,
    )
    from llm_drift_bench.models import ModelSample

    class _FakeResp:
        def __init__(self, raw: bytes) -> None:
            self._raw = raw

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def read(self, limit: int) -> bytes:
            return self._raw[:limit]

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout: _FakeResp(b'{"ok": true}')
    )
    assert _default_transport("http://localhost/v1", {"a": 1}, {}, 5.0) == {"ok": True}

    monkeypatch.setattr("llm_drift_bench.capture.MAX_INPUT_BYTES", 8)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout: _FakeResp(b'{"too_big": 12345}')
    )
    with pytest.raises(ValueError, match="limit"):
        _default_transport("http://localhost/v1", {}, {}, 5.0)

    monkeypatch.setattr("llm_drift_bench.capture.MAX_INPUT_BYTES", 1024)
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _FakeResp(b"\xff\xfe"))
    with pytest.raises(ValueError, match="UTF-8"):
        _default_transport("http://localhost/v1", {}, {}, 5.0)

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _FakeResp(b"{bad"))
    with pytest.raises(ValueError, match="valid JSON"):
        _default_transport("http://localhost/v1", {}, {}, 5.0)

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _FakeResp(b"[1, 2]"))
    with pytest.raises(ValueError, match="JSON object"):
        _default_transport("http://localhost/v1", {}, {}, 5.0)

    # Provider extraction error branches
    with pytest.raises(ValueError, match="Anthropic response is missing"):
        _extract_completion({}, provider="anthropic")
    with pytest.raises(ValueError, match="no text blocks"):
        _extract_completion({"content": [{"type": "tool_use"}]}, provider="anthropic")
    with pytest.raises(ValueError, match="missing non-empty 'candidates'"):
        _extract_completion({}, provider="gemini")
    with pytest.raises(ValueError, match="candidate entry"):
        _extract_completion({"candidates": ["bad"]}, provider="gemini")
    with pytest.raises(ValueError, match="content.parts"):
        _extract_completion({"candidates": [{}]}, provider="gemini")
    with pytest.raises(ValueError, match="no text blocks"):
        _extract_completion({"candidates": [{"content": {"parts": [{}]}}]}, provider="gemini")
    with pytest.raises(ValueError, match="choice entry"):
        _extract_completion({"choices": ["bad"]}, provider="openai")
    with pytest.raises(ValueError, match="missing string 'message.content'"):
        _extract_completion({"choices": [{"message": {"content": 123}}]}, provider="openai")
    with pytest.raises(ValueError, match="no text blocks"):
        _extract_completion(
            {"choices": [{"message": {"content": [{"type": "image"}]}}]}, provider="openai"
        )

    # Invalid Retry-After falls back to default delay
    err = urllib.error.HTTPError(
        "http://x", 429, "Too Many", hdrs={"Retry-After": "not-a-number"}, fp=io.BytesIO(b"")  # type: ignore[arg-type]
    )
    assert _retry_after_delay(err, 2.5) == 2.5

    # Existing samples mismatch and complete resume short-circuit
    cfg = CaptureConfig(endpoint="http://localhost:8000/v1", model="m1", max_tokens=64)
    prompts = [PromptCase(prompt_id="p1", prompt="hi")]
    with pytest.raises(ValueError, match="belongs to model"):
        capture_suite(
            prompts,
            cfg,
            existing_samples=[ModelSample(prompt_id="p1", output="a", model_name="other")],
        )
    reused = capture_suite(
        prompts,
        cfg,
        existing_samples=[ModelSample(prompt_id="p1", output="cached", model_name="m1")],
        transport=lambda *_: (_ for _ in ()).throw(AssertionError("should not call transport")),
    )
    assert reused[0].output == "cached"

    # Cover openai max_tokens body field, malformed message object, and noisy logprobs entries
    from llm_drift_bench.capture import _build_request_body, _extract_openai_logprobs

    body = _build_request_body(prompts[0], cfg)
    assert body["max_tokens"] == 64

    with pytest.raises(ValueError, match="message.content"):
        _extract_completion({"choices": [{"message": "not-a-dict"}]}, provider="openai")

    assert _extract_openai_logprobs({"logprobs": {"content": []}}) is None
    parsed_lp = _extract_openai_logprobs(
        {
            "logprobs": {
                "content": [
                    "not-a-dict",
                    {"token": "skip", "logprob": "nan"},
                    {
                        "token": "ok",
                        "logprob": -0.1,
                        "top_logprobs": ["bad", {"token": "ok", "logprob": -0.1}],
                    },
                ]
            }
        }
    )
    assert parsed_lp is not None and len(parsed_lp) == 1






