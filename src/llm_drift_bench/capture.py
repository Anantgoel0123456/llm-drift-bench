"""Reproducible capture collector for OpenAI-compatible endpoints (Ollama, vLLM, OpenAI)."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from llm_drift_bench.models import ModelSample, PromptCase

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


@dataclass(frozen=True)
class CaptureConfig:
    """Explicit timeout, retry, sampling, and secret-env configuration for endpoint capture."""

    endpoint: str
    model: str
    api_key_env: str | None = None
    timeout_sec: float = 60.0
    max_retries: int = 2
    temperature: float | None = 0.0
    seed: int | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlparse(self.endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"endpoint must be an http:// or https:// URL, got {self.endpoint!r}."
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string.")
        if (
            isinstance(self.timeout_sec, bool)
            or not math.isfinite(self.timeout_sec)
            or self.timeout_sec <= 0
        ):
            raise ValueError("timeout_sec must be a finite positive number.")
        if isinstance(self.max_retries, bool) or not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be an integer between 0 and 5.")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number or None.")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool) or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer or None.")

    @property
    def chat_completions_url(self) -> str:
        base = self.endpoint.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @property
    def endpoint_host(self) -> str:
        return urllib.parse.urlparse(self.endpoint).netloc


def _default_transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: float,
) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
    parsed: Any = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Endpoint response must be a JSON object.")
    return parsed


def _build_request_body(prompt: PromptCase, config: CaptureConfig) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt.prompt})
    body: dict[str, Any] = {"model": config.model, "messages": messages}
    if config.temperature is not None:
        body["temperature"] = config.temperature
    if config.seed is not None:
        body["seed"] = config.seed
    if config.max_tokens is not None:
        body["max_tokens"] = config.max_tokens
    return body


def _extract_completion(
    response: dict[str, Any],
) -> tuple[str, int | None, str | None, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Endpoint response is missing non-empty 'choices'.")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Endpoint choice entry must be a JSON object.")
    message = first.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Endpoint choice is missing string 'message.content'.")
    content = message["content"]
    finish_reason = (
        str(first["finish_reason"]) if first.get("finish_reason") is not None else None
    )
    usage = response.get("usage")
    tokens: int | None = None
    if isinstance(usage, dict):
        comp_tokens = usage.get("completion_tokens")
        if isinstance(comp_tokens, int) and not isinstance(comp_tokens, bool) and comp_tokens >= 0:
            tokens = comp_tokens
    fingerprint = (
        str(response["system_fingerprint"])
        if response.get("system_fingerprint") is not None
        else None
    )
    return content, tokens, finish_reason, fingerprint


def capture_suite(
    prompts: Sequence[PromptCase],
    config: CaptureConfig,
    *,
    transport: Transport | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[ModelSample]:
    """Capture responses for every PromptCase from an OpenAI-compatible chat endpoint."""
    if not prompts:
        raise ValueError("Prompt suite cannot be empty.")

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if config.api_key_env:
        token = os.environ.get(config.api_key_env)
        if not token:
            raise ValueError(
                f"Environment variable {config.api_key_env!r} is not set or empty."
            )
        headers["Authorization"] = f"Bearer {token}"

    send = transport or _default_transport
    url = config.chat_completions_url
    samples: list[ModelSample] = []

    for prompt in prompts:
        body = _build_request_body(prompt, config)
        last_exc: Exception | None = None
        response: dict[str, Any] | None = None
        latency_ms = 0.0

        for attempt in range(config.max_retries + 1):
            t0 = time.perf_counter()
            try:
                response = send(url, body, headers, config.timeout_sec)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                break
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code not in _RETRYABLE_STATUS or attempt == config.max_retries:
                    raise RuntimeError(
                        f"Capture failed for prompt {prompt.prompt_id!r}: HTTP {exc.code}."
                    ) from None
                sleep_fn(min(8.0, 0.5 * (2**attempt)))
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt == config.max_retries:
                    raise RuntimeError(
                        f"Capture failed for prompt {prompt.prompt_id!r}: {exc}."
                    ) from None
                sleep_fn(min(8.0, 0.5 * (2**attempt)))

        if response is None:
            raise RuntimeError(
                f"Capture failed for prompt {prompt.prompt_id!r}: {last_exc}"
            )

        content, comp_tokens, finish_reason, fingerprint = _extract_completion(response)
        metadata: dict[str, Any] = {
            "endpoint_host": config.endpoint_host,
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temperature": config.temperature,
            "seed": config.seed,
            "finish_reason": finish_reason,
            "system_fingerprint": fingerprint,
        }
        samples.append(
            ModelSample(
                prompt_id=prompt.prompt_id,
                output=content,
                model_name=config.model,
                latency_ms=latency_ms,
                tokens_generated=comp_tokens,
                metadata={k: v for k, v in metadata.items() if v is not None},
            )
        )
    return samples


def serialize_samples_jsonl(samples: Sequence[ModelSample]) -> str:
    """Serialize captured ModelSample objects into strict JSONL format."""
    lines: list[str] = []
    for sample in samples:
        row: dict[str, Any] = {
            "prompt_id": sample.prompt_id,
            "model_name": sample.model_name,
            "output": sample.output,
        }
        if sample.latency_ms is not None:
            row["latency_ms"] = sample.latency_ms
        if sample.tokens_generated is not None:
            row["tokens_generated"] = sample.tokens_generated
        if sample.metadata:
            row["metadata"] = sample.metadata
        lines.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")
