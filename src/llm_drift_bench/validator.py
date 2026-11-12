"""Zero-bloat JSON payload extraction and JSON Schema conformity validation."""

from __future__ import annotations

import json
import re
from typing import Any

_CODE_FENCE_PATTERN = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def extract_json_payload(raw_text: str) -> tuple[bool, Any, str | None]:
    """Extract and parse a JSON payload from raw LLM completion text.

    Supports pure JSON strings, markdown code fences (```json ... ```), and
    fallback extraction of the outermost JSON object/array when wrapped in prose.

    Args:
        raw_text: Raw text completion produced by an LLM.

    Returns:
        A tuple of (is_valid_json, parsed_value, error_message).
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return False, None, "Empty or non-string output."

    stripped = raw_text.strip()

    # 1. Try direct JSON parsing first.
    try:
        return True, json.loads(stripped), None
    except json.JSONDecodeError as direct_err:
        last_error = str(direct_err)

    # 2. Try extracting from markdown ```json ... ``` fences.
    fence_match = _CODE_FENCE_PATTERN.search(stripped)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return True, json.loads(candidate), None
        except json.JSONDecodeError as fence_err:
            last_error = str(fence_err)

    # 3. Try locating the first '{' or '[' and matching to last '}' or ']'.
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start_idx = stripped.find(open_char)
        end_idx = stripped.rfind(close_char)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            sub = stripped[start_idx : end_idx + 1]
            try:
                return True, json.loads(sub), None
            except json.JSONDecodeError:
                continue

    return False, None, f"Invalid JSON syntax: {last_error}"
