"""Bounded UTF-8 input and strict JSON decoding for saved datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_drift_bench._numbers import parse_finite_float, parse_finite_int, reject_constant

MAX_INPUT_BYTES = 16 * 1024 * 1024


def read_utf8(path: str | Path) -> str:
    source = Path(path)
    with source.open("rb") as stream:
        data = stream.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError(f"Input exceeds the {MAX_INPUT_BYTES}-byte limit: {source}")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Input must be UTF-8 (an initial BOM is allowed): {source}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def parse_json(text: str) -> Any:
    return json.loads(
        text, parse_constant=reject_constant, parse_float=parse_finite_float,
        parse_int=parse_finite_int, object_pairs_hook=_unique_object,
    )
