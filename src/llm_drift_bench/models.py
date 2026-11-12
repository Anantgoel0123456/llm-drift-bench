"""Domain data models and schemas for llm-drift-bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptCase:
    """Represents a single benchmark prompt case and optional expected JSON schema."""

    prompt_id: str
    prompt: str
    schema: dict[str, Any] | None = None
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptCase:
        """Construct a PromptCase from a dictionary payload."""
        prompt_id = str(data.get("prompt_id") or data.get("id") or "")
        if not prompt_id:
            raise ValueError("PromptCase requires a non-empty 'prompt_id' or 'id'.")
        prompt = str(data.get("prompt", ""))
        schema = data.get("schema")
        if schema is not None and not isinstance(schema, dict):
            raise ValueError(f"PromptCase '{prompt_id}' schema must be a JSON object/dict.")
        category = str(data.get("category", "general"))
        metadata_raw = data.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        return cls(
            prompt_id=prompt_id,
            prompt=prompt,
            schema=schema,
            category=category,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ModelSample:
    """Represents a model's output sample for a specific prompt_id."""

    prompt_id: str
    output: str
    model_name: str = "unknown"
    latency_ms: float | None = None
    tokens_generated: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_model_name: str = "unknown") -> ModelSample:
        """Construct a ModelSample from a dictionary payload."""
        prompt_id = str(data.get("prompt_id") or data.get("id") or "")
        if not prompt_id:
            raise ValueError("ModelSample requires a non-empty 'prompt_id' or 'id'.")
        if "output" not in data and "response" not in data:
            raise ValueError(f"ModelSample '{prompt_id}' requires an 'output' or 'response' field.")
        raw_output = data.get("output", data.get("response", ""))
        output = raw_output if isinstance(raw_output, str) else str(raw_output)
        model_name = str(data.get("model_name") or data.get("model") or default_model_name)
        latency_raw = data.get("latency_ms")
        latency_ms = float(latency_raw) if latency_raw is not None else None
        tokens_raw = data.get("tokens_generated")
        tokens_generated = int(tokens_raw) if tokens_raw is not None else None
        metadata_raw = data.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        return cls(
            prompt_id=prompt_id,
            output=output,
            model_name=model_name,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            metadata=metadata,
        )
