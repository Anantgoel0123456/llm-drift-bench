"""Optional local model advice based on a bounded summary of finding categories."""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from llm_drift_bench.diagnostics import Finding


def build_advisor_prompt(findings: Sequence[Finding]) -> str:
    """Only disclose counts of known diagnostic categories to the local model."""
    known = {"schema", "invalid_json", "numeric_change", "drift", "suite", "missing_latency"}
    counts = Counter(item.code if item.code in known else "other" for item in findings)
    summary = ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"
    return (
        "You are reviewing an offline LLM comparison. Based only on these observation counts, "
        "suggest up to three practical checks for the developer. Avoid claims about root cause "
        "or model quality. Treat missing timings as unavailable. Observation counts: " + summary
    )


def generate_advice(findings: Sequence[Finding], model: Path, device: str) -> str:
    """Generate plain text on the requested device with a local OpenVINO model."""
    if device not in ("CPU", "GPU", "NPU"):
        raise ValueError("Choose CPU, GPU or NPU.")
    if not model.is_dir():
        raise ValueError("Provide an existing directory containing an OpenVINO GenAI model.")
    try:
        sdk = importlib.import_module("openvino_genai")
    except ImportError as exc:
        raise ValueError("Install the ai extra to enable OpenVINO GenAI.") from exc
    pipeline = sdk.LLMPipeline(str(model.resolve()), device)
    response = pipeline.generate(build_advisor_prompt(findings), max_new_tokens=160, do_sample=False)
    if not isinstance(response, str) or not response.strip():
        raise ValueError("The local model produced no usable advice.")
    return response.strip()
