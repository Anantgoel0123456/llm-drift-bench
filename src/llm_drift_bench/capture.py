"""Reproducible capture collector for OpenAI-compatible endpoints (Ollama, vLLM, OpenAI)."""

from __future__ import annotations

import http.client
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from llm_drift_bench._file_io import MAX_INPUT_BYTES
from llm_drift_bench.models import ModelSample, PromptCase

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_MAX_RETRY_AFTER_SEC = 30.0
CaptureProvider = Literal["openai", "anthropic", "gemini"]
_VALID_PROVIDERS = ("openai", "anthropic", "gemini")
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
    json_mode: bool = False
    concurrency: int = 1
    extra_headers: tuple[tuple[str, str], ...] = ()
    provider: CaptureProvider = "openai"

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
        if not isinstance(self.json_mode, bool):
            raise ValueError("json_mode must be a boolean.")
        if isinstance(self.concurrency, bool) or not 1 <= self.concurrency <= 32:
            raise ValueError("concurrency must be an integer between 1 and 32.")
        if self.provider not in _VALID_PROVIDERS:
            raise ValueError(f"provider must be one of {_VALID_PROVIDERS}, got {self.provider!r}.")
        for name, val in self.extra_headers:
            if not isinstance(name, str) or not name.strip() or not isinstance(val, str):
                raise ValueError("extra_headers entries must be (non-empty name, string value).")
            if any(ch in name or ch in val for ch in ("\r", "\n")):
                raise ValueError(f"Header {name!r} must not contain CR or LF characters.")
            if self.api_key_env and name.strip().lower() in (
                "authorization",
                "x-api-key",
                "x-goog-api-key",
            ):
                raise ValueError(
                    "Do not pass an Authorization/x-api-key/x-goog-api-key header "
                    "when --api-key-env is set."
                )

    @property
    def chat_completions_url(self) -> str:
        base = self.endpoint.rstrip("/")
        if self.provider == "anthropic":
            if base.endswith("/messages"):
                return base
            if base.endswith("/v1"):
                return f"{base}/messages"
            return f"{base}/v1/messages"
        if self.provider == "gemini":
            if base.endswith(":generateContent"):
                return base
            model_name = self.model.removeprefix("models/")
            quoted = urllib.parse.quote(model_name, safe="")
            if base.endswith(("/v1beta", "/v1")):
                return f"{base}/models/{quoted}:generateContent"
            return f"{base}/v1beta/models/{quoted}:generateContent"
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
        raw_bytes = resp.read(MAX_INPUT_BYTES + 1)
    if len(raw_bytes) > MAX_INPUT_BYTES:
        raise ValueError(f"Endpoint response exceeded the {MAX_INPUT_BYTES}-byte limit.")
    try:
        body = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Endpoint response was not valid UTF-8.") from exc
    try:
        parsed: Any = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Endpoint response was not valid JSON: {exc.msg}.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Endpoint response must be a JSON object.")
    return parsed


def _build_request_body(prompt: PromptCase, config: CaptureConfig) -> dict[str, Any]:
    if config.provider == "anthropic":
        body: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens if config.max_tokens is not None else 1024,
            "messages": [{"role": "user", "content": prompt.prompt}],
        }
        if config.system_prompt:
            body["system"] = config.system_prompt
        if config.temperature is not None:
            body["temperature"] = config.temperature
        return body

    if config.provider == "gemini":
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt.prompt}]}],
        }
        if config.system_prompt:
            body["systemInstruction"] = {"parts": [{"text": config.system_prompt}]}
        gen_cfg: dict[str, Any] = {}
        if config.temperature is not None:
            gen_cfg["temperature"] = config.temperature
        if config.max_tokens is not None:
            gen_cfg["maxOutputTokens"] = config.max_tokens
        if config.seed is not None:
            gen_cfg["seed"] = config.seed
        if config.json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        if gen_cfg:
            body["generationConfig"] = gen_cfg
        return body

    messages: list[dict[str, str]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt.prompt})
    body = {"model": config.model, "messages": messages}
    if config.temperature is not None:
        body["temperature"] = config.temperature
    if config.seed is not None:
        body["seed"] = config.seed
    if config.max_tokens is not None:
        body["max_tokens"] = config.max_tokens
    if config.json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def _extract_completion(
    response: dict[str, Any],
    *,
    provider: CaptureProvider = "openai",
) -> tuple[str, int | None, str | None, str | None]:
    if provider == "anthropic":
        blocks = response.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("Anthropic response is missing non-empty 'content'.")
        parts = [
            block["text"]
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if not parts:
            raise ValueError("Anthropic response content contains no text blocks.")
        finish_reason = (
            str(response["stop_reason"]) if response.get("stop_reason") is not None else None
        )
        usage = response.get("usage")
        tokens: int | None = None
        if isinstance(usage, dict):
            out_toks = usage.get("output_tokens")
            if isinstance(out_toks, int) and not isinstance(out_toks, bool) and out_toks >= 0:
                tokens = out_toks
        return "".join(parts), tokens, finish_reason, None

    if provider == "gemini":
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Gemini response is missing non-empty 'candidates'.")
        first_cand = candidates[0]
        if not isinstance(first_cand, dict):
            raise ValueError("Gemini candidate entry must be a JSON object.")
        content_obj = first_cand.get("content")
        raw_parts = content_obj.get("parts") if isinstance(content_obj, dict) else None
        if not isinstance(raw_parts, list) or not raw_parts:
            raise ValueError("Gemini candidate is missing non-empty 'content.parts'.")
        text_parts = [
            part["text"]
            for part in raw_parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if not text_parts:
            raise ValueError("Gemini response parts contain no text blocks.")
        finish_reason = (
            str(first_cand["finishReason"]) if first_cand.get("finishReason") is not None else None
        )
        usage_meta = response.get("usageMetadata")
        tokens = None
        if isinstance(usage_meta, dict):
            cand_toks = usage_meta.get("candidatesTokenCount")
            if isinstance(cand_toks, int) and not isinstance(cand_toks, bool) and cand_toks >= 0:
                tokens = cand_toks
        return "".join(text_parts), tokens, finish_reason, None

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
    tokens = None
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


def _retry_after_delay(exc: urllib.error.HTTPError, default_delay: float) -> float:
    hdrs = getattr(exc, "headers", None)
    raw = hdrs.get("Retry-After") if hdrs is not None and hasattr(hdrs, "get") else None
    if isinstance(raw, str):
        try:
            parsed = float(raw.strip())
        except ValueError:
            return default_delay
        if math.isfinite(parsed) and parsed >= 0:
            return min(_MAX_RETRY_AFTER_SEC, parsed)
    return default_delay


def parse_header_spec(raw: str) -> tuple[str, str]:
    """Parse a 'Name: Value' or 'Name=Value' CLI header specification."""
    colon_idx = raw.find(":")
    eq_idx = raw.find("=")
    if colon_idx == -1 and eq_idx == -1:
        raise ValueError(f"Header must use 'Name: Value' or 'Name=Value' syntax: {raw!r}.")
    if colon_idx != -1 and (eq_idx == -1 or colon_idx < eq_idx):
        name, val = raw[:colon_idx], raw[colon_idx + 1 :]
    else:
        name, val = raw[:eq_idx], raw[eq_idx + 1 :]
    name = name.strip()
    val = val.strip()
    if not name:
        raise ValueError(f"Header name cannot be empty: {raw!r}.")
    if any(ch in name or ch in val for ch in ("\r", "\n")):
        raise ValueError(f"Header {name!r} must not contain CR or LF characters.")
    return name, val


def capture_suite(
    prompts: Sequence[PromptCase],
    config: CaptureConfig,
    *,
    transport: Transport | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    existing_samples: Sequence[ModelSample] = (),
    on_sample: Callable[[ModelSample], None] | None = None,
) -> list[ModelSample]:
    """Capture responses for every PromptCase from an OpenAI-compatible chat endpoint."""
    if not prompts:
        raise ValueError("Prompt suite cannot be empty.")

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if config.provider == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    for h_name, h_val in config.extra_headers:
        headers[h_name.strip()] = h_val
    if config.api_key_env:
        token = os.environ.get(config.api_key_env)
        if not token:
            raise ValueError(
                f"Environment variable {config.api_key_env!r} is not set or empty."
            )
        if config.provider == "anthropic":
            headers["x-api-key"] = token
        elif config.provider == "gemini":
            headers["x-goog-api-key"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"

    send = transport or _default_transport
    url = config.chat_completions_url
    callback_lock = threading.Lock()

    completed: dict[str, ModelSample] = {}
    for sample in existing_samples:
        if sample.model_name != config.model:
            raise ValueError(
                f"Existing sample {sample.prompt_id!r} belongs to model "
                f"{sample.model_name!r}, expected {config.model!r}."
            )
        completed[sample.prompt_id] = sample

    pending = [prompt for prompt in prompts if prompt.prompt_id not in completed]
    if not pending:
        return [completed[prompt.prompt_id] for prompt in prompts]

    def _capture_one(prompt: PromptCase) -> ModelSample:
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
                        f"Capture failed for prompt {prompt.prompt_id!r} "
                        f"on {config.endpoint_host}: HTTP {exc.code}."
                    ) from None
                backoff = min(8.0, 0.5 * (2**attempt))
                sleep_fn(_retry_after_delay(exc, backoff))
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.HTTPException,
            ) as exc:
                last_exc = exc
                if attempt == config.max_retries:
                    reason = getattr(exc, "reason", type(exc).__name__)
                    raise RuntimeError(
                        f"Capture failed for prompt {prompt.prompt_id!r} "
                        f"on {config.endpoint_host}: {reason}."
                    ) from None
                sleep_fn(min(8.0, 0.5 * (2**attempt)))

        if response is None:
            raise RuntimeError(
                f"Capture failed for prompt {prompt.prompt_id!r}: {last_exc}"
            )

        content, comp_tokens, finish_reason, fingerprint = _extract_completion(
            response, provider=config.provider
        )
        metadata: dict[str, Any] = {
            "endpoint_host": config.endpoint_host,
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temperature": config.temperature,
            "seed": config.seed,
            "finish_reason": finish_reason,
            "system_fingerprint": fingerprint,
        }
        sample = ModelSample(
            prompt_id=prompt.prompt_id,
            output=content,
            model_name=config.model,
            latency_ms=latency_ms,
            tokens_generated=comp_tokens,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )
        if on_sample is not None:
            with callback_lock:
                on_sample(sample)
        return sample

    if config.concurrency == 1 or len(pending) == 1:
        captured = [_capture_one(prompt) for prompt in pending]
    else:
        with ThreadPoolExecutor(max_workers=min(config.concurrency, len(pending))) as pool:
            captured = list(pool.map(_capture_one, pending))

    for sample in captured:
        completed[sample.prompt_id] = sample
    return [completed[prompt.prompt_id] for prompt in prompts]


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
