"""Tests for opt-in semantic embedding cosine similarity drift analysis."""

from __future__ import annotations

import io
import urllib.error
from email.message import Message
from typing import Any

import pytest

from llm_drift_bench.embeddings import (
    EmbeddingConfig,
    analyze_capture_embeddings,
    compare_embedding_vectors,
    cosine_similarity,
    fetch_embeddings,
    format_embedding_drift_markdown,
    format_embedding_drift_text,
)
from llm_drift_bench.models import ModelSample


def test_cosine_similarity_and_precomputed_vector_comparison() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    rep = compare_embedding_vectors(
        {"p1": [1.0, 0.0], "p2": [0.0, 1.0]},
        {"p1": [0.9, 0.1], "p2": [1.0, 0.0]},
        model="text-embedding-3-small",
        min_mean_similarity=0.80,
        min_prompt_similarity=0.70,
    )
    assert rep.passed is False
    assert rep.complete is True
    assert rep.min_similarity == pytest.approx(0.0)
    assert len(rep.failure_reasons) == 2
    assert rep.to_dict()["passed"] is False
    assert "FAILED" in format_embedding_drift_text(rep)
    md = format_embedding_drift_markdown(rep)
    assert "# Embedding Similarity Drift: FAILED" in md and "| p2 |" in md


def test_analyze_capture_embeddings_offline_and_incomplete_honesty() -> None:
    base = [
        ModelSample(prompt_id="p1", output="hi", metadata={"embedding": [1.0, 0.0]}),
        ModelSample(prompt_id="p2", output="bye"),
    ]
    cand = [
        ModelSample(prompt_id="p1", output="hi", metadata={"embedding": [1.0, 0.0]}),
        ModelSample(prompt_id="p2", output="bye"),
    ]
    # Without thresholds, incomplete reports honestly with passed=True
    rep_info = analyze_capture_embeddings(base, cand)
    assert rep_info.complete is False
    assert rep_info.comparable_samples == 1
    assert rep_info.passed is True
    assert "incomplete" in format_embedding_drift_text(rep_info)

    # With a threshold, incomplete coverage must fail honestly
    rep_gated = analyze_capture_embeddings(base, cand, min_mean_similarity=0.5)
    assert rep_gated.passed is False
    assert any("Incomplete embedding coverage" in msg for msg in rep_gated.failure_reasons)


def test_fetch_embeddings_batching_retry_after_and_live_capture_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_EMB_KEY", "sk-emb-secret-123")
    calls: list[dict[str, Any]] = []
    sleeps: list[float] = []

    def flaky_transport(
        url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> dict[str, Any]:
        del timeout
        calls.append({"url": url, "body": body, "headers": headers})
        if len(calls) == 1:
            hdrs = Message()
            hdrs["Retry-After"] = "1.5"
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", hdrs, io.BytesIO(b""))
        # Return out-of-order indices to verify index sorting
        inputs = body["input"]
        return {
            "data": [
                {"index": idx, "embedding": [1.0, float(len(txt))]}
                for idx, txt in reversed(list(enumerate(inputs)))
            ]
        }

    cfg = EmbeddingConfig(
        endpoint="https://api.example.com",
        model="text-embedding-3-small",
        api_key_env="TEST_EMB_KEY",
        batch_size=2,
        max_retries=2,
        extra_headers=(("X-App", "drift"),),
    )
    assert cfg.embeddings_url == "https://api.example.com/v1/embeddings"
    vecs = fetch_embeddings(
        ["a", "bb", "ccc"], cfg, transport=flaky_transport, sleep_fn=sleeps.append
    )
    assert len(vecs) == 3
    assert vecs[0] == [1.0, 1.0] and vecs[2] == [1.0, 3.0]
    assert sleeps == [1.5]
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-emb-secret-123"
    assert calls[0]["headers"]["X-App"] == "drift"
    assert fetch_embeddings([], cfg) == []

    # Live analyze_capture_embeddings with mocked transport
    def ok_transport(
        url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> dict[str, Any]:
        del url, headers, timeout
        return {
            "data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(body["input"]))]
        }

    live_rep = analyze_capture_embeddings(
        [ModelSample(prompt_id="p1", output="ok")],
        [ModelSample(prompt_id="p1", output="fine")],
        config=cfg,
        transport=ok_transport,
        min_mean_similarity=0.95,
    )
    assert live_rep.passed is True and live_rep.mean_similarity == pytest.approx(1.0)


def test_embeddings_validation_and_error_paths() -> None:
    with pytest.raises(ValueError, match="non-empty numeric"):
        cosine_similarity([], [1.0])
    with pytest.raises(ValueError, match="non-finite"):
        cosine_similarity([float("nan")], [1.0])
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="between -1.0 and 1.0"):
        compare_embedding_vectors({"p1": [1.0]}, {"p1": [1.0]}, min_mean_similarity=2.0)
    with pytest.raises(ValueError, match="non-empty"):
        compare_embedding_vectors({}, {"p1": [1.0]})
    with pytest.raises(ValueError, match="non-empty"):
        analyze_capture_embeddings([], [ModelSample(prompt_id="p1", output="x")])

    with pytest.raises(ValueError, match="http"):
        EmbeddingConfig(endpoint="ftp://bad", model="m")
    with pytest.raises(ValueError, match="model"):
        EmbeddingConfig(endpoint="https://api.example.com", model="  ")
    with pytest.raises(ValueError, match="timeout_seconds"):
        EmbeddingConfig(endpoint="https://api.example.com", model="m", timeout_seconds=0)
    with pytest.raises(ValueError, match="max_retries"):
        EmbeddingConfig(endpoint="https://api.example.com", model="m", max_retries=-1)
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingConfig(endpoint="https://api.example.com", model="m", batch_size=0)

    cfg = EmbeddingConfig(endpoint="https://api.example.com/v1", model="m", max_retries=0)
    assert cfg.embeddings_url == "https://api.example.com/v1/embeddings"
    cfg2 = EmbeddingConfig(endpoint="https://api.example.com/v1/embeddings", model="m")
    assert cfg2.embeddings_url == "https://api.example.com/v1/embeddings"

    with pytest.raises(RuntimeError, match="HTTP 400"):
        fetch_embeddings(
            ["a"],
            cfg,
            transport=lambda u, b, h, t: (_ for _ in ()).throw(
                urllib.error.HTTPError(u, 400, "Bad Request", Message(), io.BytesIO(b""))
            ),
        )
    with pytest.raises(RuntimeError, match="after 1 attempt"):
        fetch_embeddings(
            ["a"],
            cfg,
            transport=lambda u, b, h, t: (_ for _ in ()).throw(TimeoutError("timed out")),
        )
    with pytest.raises(ValueError, match="must contain 1 item"):
        fetch_embeddings(["a"], cfg, transport=lambda u, b, h, t: {"data": []})
    with pytest.raises(ValueError, match="Malformed embedding item"):
        fetch_embeddings(["a"], cfg, transport=lambda u, b, h, t: {"data": [{"index": 0}]})
    with pytest.raises(ValueError, match="Invalid embedding item index"):
        fetch_embeddings(
            ["a"],
            cfg,
            transport=lambda u, b, h, t: {"data": [{"index": 5, "embedding": [1.0]}]},
        )
    with pytest.raises(ValueError, match="duplicate or missing indices"):
        fetch_embeddings(
            ["a", "b"],
            cfg,
            transport=lambda u, b, h, t: {
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            },
        )
