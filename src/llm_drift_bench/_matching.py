"""Strict prompt identity checks shared by loaders and evaluators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar


class _HasPromptID(Protocol):
    @property
    def prompt_id(self) -> str: ...


Record = TypeVar("Record", bound=_HasPromptID)


def index_records(
    records: Sequence[Record], label: str, *, positions: Sequence[int] | None = None,
    unit: str = "record",
) -> dict[str, Record]:
    """Index without silently replacing duplicate or blank prompt IDs."""
    indexed: dict[str, Record] = {}
    for index, record in enumerate(records):
        prompt_id = record.prompt_id
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"{label} requires non-empty string prompt IDs.")
        if prompt_id in indexed:
            location = f" ({unit} {positions[index]})" if positions is not None else ""
            raise ValueError(f"Duplicate prompt_id {prompt_id!r} in {label}{location}.")
        indexed[prompt_id] = record
    return indexed


def require_same_ids(expected: set[str], actual: set[str], label: str) -> None:
    """Reject both missing and extra records with deterministic diagnostics."""
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    def preview(ids: list[str]) -> str:
        if len(ids) <= 5:
            return repr(ids)
        return f"{ids[:5]!r} (+{len(ids) - 5} more)"

    if missing or extra:
        raise ValueError(
            f"{label} prompt IDs do not match: missing={preview(missing)}, extra={preview(extra)}."
        )
