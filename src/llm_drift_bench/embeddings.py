"""Opt-in semantic embedding cosine similarity drift analyzer.

Works completely offline on precomputed vectors (or ``metadata["embedding"]`` inside
captured samples) and optionally fetches vectors from any OpenAI-compatible
``/v1/embeddings`` endpoint using Python's standard library. Never alters existing
deterministic lexical drift scores.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm_drift_bench._matching import index_records, require_same_ids
from llm_drift_bench._report_views import markdown_cell
from llm_drift_bench.models import ModelSample

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_RETRY_AFTER_SECONDS = 30.0
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

EmbeddingTransport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def _validate_vector(vec: Any, label: str) -> list[float]:
    if not isinstance(vec, Sequence) or isinstance(vec, (str, bytes, bytearray)) or not vec:
        raise ValueError(f"{label} must be a non-empty numeric sequence.")
    out: list[float] = []
    for item in vec:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{label} contains non-finite or non-numeric element: {item!r}.")
        out.append(float(item))
    return out


def _validate_similarity_limit(name: str, val: float | None) -> None:
    if val is not None and (
        isinstance(val, bool)
        or not isinstance(val, (int, float))
        or not math.isfinite(val)
        or not -1.0 <= val <= 1.0
    ):
        raise ValueError(f"{name} must be a finite number between -1.0 and 1.0.")


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute exact cosine similarity in [-1.0, 1.0] between two equal-length vectors."""
    a = _validate_vector(vec_a, "vec_a")
    b = _validate_vector(vec_b, "vec_b")
    if len(a) != len(b):
        raise ValueError(f"Embedding vector dimension mismatch: {len(a)} != {len(b)}.")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for opt-in OpenAI-compatible ``/v1/embeddings`` collection."""

    endpoint: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    batch_size: int = 32
    extra_headers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlparse(self.endpoint.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Embedding endpoint must be an http(s) URL, got {self.endpoint!r}.")
        if not self.model.strip():
            raise ValueError("EmbeddingConfig.model must be a non-empty string.")
        if (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0.0
        ):
            raise ValueError("timeout_seconds must be a finite positive number.")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

    @property
    def embeddings_url(self) -> str:
        base = self.endpoint.strip().rstrip("/")
        if base.endswith("/embeddings"):
            return base
        if base.endswith("/v1"):
            return f"{base}/embeddings"
        return f"{base}/v1/embeddings"


def _default_embedding_transport(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    encoded = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("Embedding endpoint response exceeded 16 MiB limit.")
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Embedding endpoint response must be a JSON object.")
    return parsed


def _parse_retry_after(header_val: str | None) -> float | None:
    if not header_val:
        return None
    try:
        secs = float(header_val.strip())
    except ValueError:
        return None
    if not math.isfinite(secs) or secs < 0.0:
        return None
    return min(secs, _MAX_RETRY_AFTER_SECONDS)


def fetch_embeddings(
    texts: Sequence[str],
    config: EmbeddingConfig,
    *,
    transport: EmbeddingTransport | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[list[float]]:
    """Fetch dense embedding vectors for texts from an OpenAI-compatible endpoint."""
    if not texts:
        return []
    sender = transport or _default_embedding_transport
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key_env:
        token = os.environ.get(config.api_key_env, "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    for key, val in config.extra_headers:
        headers[key] = val

    url = config.embeddings_url
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), config.batch_size):
        chunk = list(texts[start : start + config.batch_size])
        body: dict[str, Any] = {"model": config.model, "input": chunk}
        response: dict[str, Any] | None = None
        for attempt in range(config.max_retries + 1):
            try:
                response = sender(url, body, dict(headers), config.timeout_seconds)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE_STATUS and attempt < config.max_retries:
                    delay = _parse_retry_after(
                        exc.headers.get("Retry-After") if exc.headers else None
                    )
                    sleep_fn(delay if delay is not None else min(2.0**attempt, 8.0))
                    continue
                raise RuntimeError(
                    f"Embedding endpoint returned HTTP {exc.code} for model {config.model!r}."
                ) from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.HTTPException,
            ) as exc:
                if attempt < config.max_retries:
                    sleep_fn(min(2.0**attempt, 8.0))
                    continue
                raise RuntimeError(
                    f"Embedding request failed after {attempt + 1} attempt(s): {exc}"
                ) from exc

        assert response is not None
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(chunk):
            raise ValueError(
                f"Embedding response 'data' must contain {len(chunk)} item(s), "
                f"got {len(data) if isinstance(data, list) else type(data).__name__}."
            )
        indexed: list[tuple[int, list[float]]] = []
        for pos, item in enumerate(data):
            if not isinstance(item, dict) or "embedding" not in item:
                raise ValueError(f"Malformed embedding item at position {pos}.")
            idx = item.get("index", pos)
            if type(idx) is not int or not 0 <= idx < len(chunk):
                raise ValueError(f"Invalid embedding item index {idx!r}.")
            indexed.append((idx, _validate_vector(item["embedding"], f"embedding[{idx}]")))
        indexed.sort(key=lambda pair: pair[0])
        if [i for i, _ in indexed] != list(range(len(chunk))):
            raise ValueError("Embedding response returned duplicate or missing indices.")
        all_vectors.extend(vec for _, vec in indexed)

    return all_vectors


@dataclass(frozen=True)
class EmbeddingDriftItem:
    """Per-prompt semantic embedding cosine similarity between baseline and candidate."""

    prompt_id: str
    available: bool
    cosine_similarity: float | None
    cosine_distance: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "available": self.available,
            "cosine_similarity": (
                round(self.cosine_similarity, 6) if self.cosine_similarity is not None else None
            ),
            "cosine_distance": (
                round(self.cosine_distance, 6) if self.cosine_distance is not None else None
            ),
        }


@dataclass(frozen=True)
class EmbeddingDriftReport:
    """Aggregate semantic embedding similarity report across matched prompts."""

    model: str
    total_samples: int
    comparable_samples: int
    complete: bool
    mean_similarity: float | None
    min_similarity: float | None
    max_distance: float | None
    items: tuple[EmbeddingDriftItem, ...]
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "total_samples": self.total_samples,
            "comparable_samples": self.comparable_samples,
            "complete": self.complete,
            "mean_similarity": (
                round(self.mean_similarity, 6) if self.mean_similarity is not None else None
            ),
            "min_similarity": (
                round(self.min_similarity, 6) if self.min_similarity is not None else None
            ),
            "max_distance": (
                round(self.max_distance, 6) if self.max_distance is not None else None
            ),
            "items": [item.to_dict() for item in self.items],
        }


def _build_embedding_report(
    ordered_ids: Sequence[str],
    base_vecs: Mapping[str, Sequence[float] | None],
    cand_vecs: Mapping[str, Sequence[float] | None],
    *,
    model: str,
    min_mean_similarity: float | None,
    min_prompt_similarity: float | None,
) -> EmbeddingDriftReport:
    _validate_similarity_limit("min_mean_similarity", min_mean_similarity)
    _validate_similarity_limit("min_prompt_similarity", min_prompt_similarity)

    items: list[EmbeddingDriftItem] = []
    sims: list[float] = []
    dists: list[float] = []

    for pid in ordered_ids:
        va = base_vecs.get(pid)
        vb = cand_vecs.get(pid)
        if va is None or vb is None:
            items.append(
                EmbeddingDriftItem(
                    prompt_id=pid,
                    available=False,
                    cosine_similarity=None,
                    cosine_distance=None,
                )
            )
            continue
        sim = cosine_similarity(va, vb)
        dist = max(0.0, 1.0 - sim)
        sims.append(sim)
        dists.append(dist)
        items.append(
            EmbeddingDriftItem(
                prompt_id=pid,
                available=True,
                cosine_similarity=sim,
                cosine_distance=dist,
            )
        )

    total = len(ordered_ids)
    comparable = len(sims)
    complete = comparable == total
    mean_sim = (sum(sims) / comparable) if sims else None
    min_sim = min(sims) if sims else None
    max_dist = max(dists) if dists else None

    failures: list[str] = []
    has_limits = min_mean_similarity is not None or min_prompt_similarity is not None
    if has_limits and not complete:
        failures.append(
            f"Incomplete embedding coverage ({comparable}/{total}) for configured similarity gate."
        )
    if (
        min_mean_similarity is not None
        and mean_sim is not None
        and mean_sim < min_mean_similarity
    ):
        failures.append(
            f"Mean embedding cosine similarity ({mean_sim:.4f}) fell below "
            f"minimum ({min_mean_similarity:.4f})."
        )
    if (
        min_prompt_similarity is not None
        and min_sim is not None
        and min_sim < min_prompt_similarity
    ):
        worst = [
            it.prompt_id
            for it in items
            if it.cosine_similarity is not None and it.cosine_similarity < min_prompt_similarity
        ]
        failures.append(
            f"Min prompt embedding similarity ({min_sim:.4f}) fell below "
            f"minimum ({min_prompt_similarity:.4f}) on {len(worst)} prompt(s): "
            f"{', '.join(worst[:5])}."
        )

    return EmbeddingDriftReport(
        model=model,
        total_samples=total,
        comparable_samples=comparable,
        complete=complete,
        mean_similarity=mean_sim,
        min_similarity=min_sim,
        max_distance=max_dist,
        items=tuple(items),
        passed=len(failures) == 0,
        failure_reasons=tuple(failures),
    )


def compare_embedding_vectors(
    baseline_vectors: Mapping[str, Sequence[float]],
    candidate_vectors: Mapping[str, Sequence[float]],
    *,
    model: str = "precomputed",
    min_mean_similarity: float | None = None,
    min_prompt_similarity: float | None = None,
) -> EmbeddingDriftReport:
    """Compare precomputed prompt_id -> vector mappings completely offline."""
    if not baseline_vectors or not candidate_vectors:
        raise ValueError("Both baseline and candidate vector mappings must be non-empty.")
    require_same_ids(set(baseline_vectors), set(candidate_vectors), "Candidate vectors")
    ordered = sorted(baseline_vectors)
    return _build_embedding_report(
        ordered,
        baseline_vectors,
        candidate_vectors,
        model=model,
        min_mean_similarity=min_mean_similarity,
        min_prompt_similarity=min_prompt_similarity,
    )


def analyze_capture_embeddings(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
    *,
    config: EmbeddingConfig | None = None,
    transport: EmbeddingTransport | None = None,
    min_mean_similarity: float | None = None,
    min_prompt_similarity: float | None = None,
) -> EmbeddingDriftReport:
    """Evaluate semantic embedding similarity from capture metadata or an opt-in endpoint."""
    if not baseline_samples or not candidate_samples:
        raise ValueError("Both baseline and candidate sample collections must be non-empty.")
    base_map = index_records(baseline_samples, "baseline samples")
    cand_map = index_records(candidate_samples, "candidate samples")
    require_same_ids(set(base_map), set(cand_map), "Candidate samples")
    ordered = sorted(base_map)

    if config is not None:
        base_list = fetch_embeddings(
            [base_map[pid].output for pid in ordered], config, transport=transport
        )
        cand_list = fetch_embeddings(
            [cand_map[pid].output for pid in ordered], config, transport=transport
        )
        base_vecs: dict[str, Sequence[float] | None] = dict(zip(ordered, base_list, strict=True))
        cand_vecs: dict[str, Sequence[float] | None] = dict(zip(ordered, cand_list, strict=True))
        model_name = config.model
    else:
        base_vecs = {pid: base_map[pid].metadata.get("embedding") for pid in ordered}
        cand_vecs = {pid: cand_map[pid].metadata.get("embedding") for pid in ordered}
        model_name = "precomputed"

    return _build_embedding_report(
        ordered,
        base_vecs,
        cand_vecs,
        model=model_name,
        min_mean_similarity=min_mean_similarity,
        min_prompt_similarity=min_prompt_similarity,
    )


def format_embedding_drift_text(report: EmbeddingDriftReport) -> str:
    """Format an EmbeddingDriftReport as a readable terminal summary."""
    status = "PASSED" if report.passed else "FAILED"
    cov = "complete" if report.complete else "incomplete"
    mean_s = "n/a" if report.mean_similarity is None else f"{report.mean_similarity:.4f}"
    min_s = "n/a" if report.min_similarity is None else f"{report.min_similarity:.4f}"
    dist_s = "n/a" if report.max_distance is None else f"{report.max_distance:.4f}"
    lines = [
        f"Embedding Similarity Drift [{status}] "
        f"(model={report.model}, {report.comparable_samples}/{report.total_samples} {cov})",
        f"  Mean cosine similarity : {mean_s}",
        f"  Min prompt similarity  : {min_s}",
        f"  Max cosine distance    : {dist_s}",
    ]
    if report.failure_reasons:
        lines.append("")
        lines.append("Threshold failures:")
        for msg in report.failure_reasons:
            lines.append(f"  - {msg}")
    return "\n".join(lines) + "\n"


def format_embedding_drift_markdown(report: EmbeddingDriftReport) -> str:
    """Format an EmbeddingDriftReport as GitHub-flavored Markdown."""
    status = "PASSED" if report.passed else "FAILED"
    mean_s = "n/a" if report.mean_similarity is None else f"{report.mean_similarity:.4f}"
    min_s = "n/a" if report.min_similarity is None else f"{report.min_similarity:.4f}"
    lines = [
        f"# Embedding Similarity Drift: {status}",
        "",
        f"- **Embedding model**: `{markdown_cell(report.model)}`",
        f"- **Coverage**: {report.comparable_samples}/{report.total_samples} "
        f"({'complete' if report.complete else 'incomplete'})",
        f"- **Mean / Min cosine similarity**: `{mean_s}` / `{min_s}`",
        "",
        "| Prompt ID | Cosine Similarity | Cosine Distance |",
        "| --- | ---: | ---: |",
    ]
    for item in report.items:
        sim_s = "n/a" if item.cosine_similarity is None else f"{item.cosine_similarity:.4f}"
        dist_s = "n/a" if item.cosine_distance is None else f"{item.cosine_distance:.4f}"
        lines.append(f"| {markdown_cell(item.prompt_id)} | {sim_s} | {dist_s} |")
    if report.failure_reasons:
        lines += ["", "## Threshold failures", ""]
        lines.extend(f"- {markdown_cell(msg)}" for msg in report.failure_reasons)
    return "\n".join(lines) + "\n"
