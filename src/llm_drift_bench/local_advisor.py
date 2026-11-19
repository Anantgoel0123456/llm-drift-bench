"""Optional local model advice based on a bounded summary of finding categories."""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from llm_drift_bench.diagnostics import Finding

_CHECKS = {
    "schema": "Inspect the field and schema rule identified in the diagnosis.",
    "invalid_json": "Check the candidate response for malformed or truncated JSON.",
    "numeric_change": "Compare units, decimal placement and rounding with the source record.",
    "drift": "Compare the baseline and candidate responses for the flagged prompt.",
    "suite": "Review the recorded threshold failures.",
    "missing_latency": "Collect missing timings before comparing latency.",
}


def build_advisor_prompt(findings: Sequence[Finding]) -> str:
    """Supply a closed menu without sending prompt IDs, field values or free text."""
    counts = Counter(item.code for item in findings if item.code in _CHECKS)
    menu = "\n".join(f"{code}={counts[code]}: {_CHECKS[code]}" for code in counts)
    return (
        "Select up to three checks in priority order. Start with prompt-specific failures. "
        "Reply with only the selected category names, separated by commas.\n" + menu
    )


def generate_advice(findings: Sequence[Finding], model: Path, device: str) -> str:
    """Generate plain text on the requested device with a local OpenVINO model."""
    if device not in ("CPU", "GPU", "NPU"):
        raise ValueError("Choose CPU, GPU or NPU.")
    if not model.is_dir():
        raise ValueError("Provide an existing directory containing an OpenVINO GenAI model.")
    checks: dict[str, str] = {}
    for item in findings:
        if item.code in _CHECKS:
            checks.setdefault(item.code, item.suggestion)
    if not checks:
        return "No diagnostic checks to prioritize."
    try:
        sdk = importlib.import_module("openvino_genai")
    except ImportError as exc:
        raise ValueError("Install the ai extra to enable OpenVINO GenAI.") from exc
    pipeline = sdk.LLMPipeline(str(model.resolve()), device)
    response = pipeline.generate(
        build_advisor_prompt(findings), max_new_tokens=32, do_sample=False
    )
    selected = [code.strip() for code in response.split(",")] if isinstance(response, str) else []
    if 1 <= len(selected) <= 3 and len(set(selected)) == len(selected) and all(
        code in checks for code in selected
    ):
        label = "Model-selected checks"
    else:
        label = "Model selection unavailable. Showing report checks"
        selected = list(checks)[:3]
    return label + ":\n" + "\n".join(f"- {checks[code]}" for code in selected)


def available_devices() -> list[str]:
    """List device identifiers reported by the installed OpenVINO runtime."""
    try:
        runtime = importlib.import_module("openvino")
    except ImportError as exc:
        raise ValueError("Install the ai extra to list OpenVINO devices.") from exc
    return sorted(str(name) for name in runtime.Core().available_devices)
