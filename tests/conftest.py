"""Shared pytest mock fixtures for the llm-drift-bench test suite."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from llm_drift_bench.models import ModelSample, PromptCase

F = TypeVar("F", bound=Callable[..., Any])

try:
    import pytest

    fixture = pytest.fixture
except ImportError:  # pragma: no cover

    def fixture(func: F) -> F:  # type: ignore[misc]
        return func


@fixture
def invoice_schema() -> dict[str, Any]:
    """Return a strict JSON Schema for an extracted invoice payload."""
    return {
        "type": "object",
        "required": ["invoice_id", "amount", "currency", "status", "items"],
        "properties": {
            "invoice_id": {
                "type": "string",
                "minLength": 4,
                "maxLength": 20,
                "pattern": r"^INV-\d+$",
            },
            "amount": {"type": "number", "minimum": 0.0, "maximum": 100000.0},
            "currency": {"type": "string", "const": "USD"},
            "status": {"type": "string", "enum": ["paid", "pending", "void"]},
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {"type": "string"},
            },
            "is_verified": {"type": "boolean"},
            "line_count": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": False,
    }


@fixture
def sample_prompt_cases(invoice_schema: dict[str, Any]) -> list[PromptCase]:
    """Return mock PromptCase fixtures covering JSON extraction and text summarization."""
    return [
        PromptCase(
            prompt_id="p_invoice_1",
            prompt="Extract invoice JSON.",
            schema=invoice_schema,
            category="extraction",
        ),
        PromptCase(
            prompt_id="p_summary_2",
            prompt="Summarize release notes in one sentence.",
            schema=None,
            category="summarization",
        ),
    ]


@fixture
def baseline_samples() -> list[ModelSample]:
    """Return baseline FP16 model samples."""
    return [
        ModelSample(
            prompt_id="p_invoice_1",
            model_name="model-fp16",
            latency_ms=110.0,
            tokens_generated=35,
            output=(
                '{"invoice_id": "INV-1001", "amount": 250.0, "currency": "USD", '
                '"status": "paid", "items": ["hosting"], "is_verified": true, "line_count": 1}'
            ),
        ),
        ModelSample(
            prompt_id="p_summary_2",
            model_name="model-fp16",
            latency_ms=90.0,
            tokens_generated=18,
            output="Version 2.1 improves inference latency and reduces memory overhead.",
        ),
    ]


@fixture
def candidate_minor_drift_samples() -> list[ModelSample]:
    """Return candidate Q4_K_M samples with negligible output drift."""
    return [
        ModelSample(
            prompt_id="p_invoice_1",
            model_name="model-q4_k_m",
            latency_ms=75.0,
            tokens_generated=35,
            output=(
                '{"invoice_id": "INV-1001", "amount": 250.0, "currency": "USD", '
                '"status": "paid", "items": ["hosting"], "is_verified": true, "line_count": 1}'
            ),
        ),
        ModelSample(
            prompt_id="p_summary_2",
            model_name="model-q4_k_m",
            latency_ms=65.0,
            tokens_generated=18,
            output="Version 2.1 improves inference latency and reduces memory overhead.",
        ),
    ]


@fixture
def candidate_major_drift_samples() -> list[ModelSample]:
    """Return candidate INT2 samples exhibiting severe schema violations and drift."""
    return [
        ModelSample(
            prompt_id="p_invoice_1",
            model_name="model-int2",
            latency_ms=260.0,
            tokens_generated=40,
            output=(
                '{"invoice_id": "X", "amount": 950.0, '
                '"currency": "EUR", "status": "unknown", "items": []}'
            ),
        ),
        ModelSample(
            prompt_id="p_summary_2",
            model_name="model-int2",
            latency_ms=240.0,
            tokens_generated=12,
            output="Unrelated hallucinated text about cooking pasta recipes.",
        ),
    ]
