"""Curated starter benchmark packs for extraction, classification, coding, and RAG workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llm_drift_bench.capture import serialize_samples_jsonl
from llm_drift_bench.models import ModelSample, PromptCase
from llm_drift_bench.policy import DriftPolicy, FieldRule, render_policy_toml
from llm_drift_bench.wizard import (
    PRESET_ANSWERS,
    WizardAnswers,
    build_policy_from_answers,
    render_wizard_config,
)

PackName = Literal["extraction", "classification", "coding", "rag", "agent"]
_PACK_NAMES: tuple[PackName, ...] = ("extraction", "classification", "coding", "rag", "agent")


@dataclass(frozen=True)
class BenchmarkPack:
    """Self-contained starter benchmark suite with baseline, passing, and drifted captures."""

    name: PackName
    description: str
    prompts: tuple[PromptCase, ...]
    baseline: tuple[ModelSample, ...]
    candidate_pass: tuple[ModelSample, ...]
    candidate_drift: tuple[ModelSample, ...]
    policy_toml: str


def _case_to_dict(case: PromptCase) -> dict[str, Any]:
    row: dict[str, Any] = {
        "prompt_id": case.prompt_id,
        "category": case.category,
        "prompt": case.prompt,
    }
    if case.schema is not None:
        row["schema"] = case.schema
    return row


def _build_extraction_pack() -> BenchmarkPack:
    inv_schema: dict[str, Any] = {
        "type": "object",
        "required": ["invoice_id", "currency", "total", "paid"],
        "properties": {
            "invoice_id": {"type": "string", "minLength": 4},
            "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
            "total": {"type": "number", "minimum": 0},
            "paid": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    prompts = (
        PromptCase(
            prompt_id="ext_inv_01",
            category="extraction",
            prompt="Extract invoice_id, currency, total, and paid status for INV-1001 ($420.50).",
            schema=inv_schema,
        ),
        PromptCase(
            prompt_id="ext_inv_02",
            category="extraction",
            prompt="Extract invoice_id, currency, total, and paid status for INV-1002 (EUR 980).",
            schema=inv_schema,
        ),
        PromptCase(
            prompt_id="ext_inv_03",
            category="extraction",
            prompt="Extract invoice_id, currency, total, and paid status for INV-1003 (GBP 150).",
            schema=inv_schema,
        ),
    )
    baseline = (
        ModelSample(
            prompt_id="ext_inv_01",
            model_name="baseline-fp16",
            latency_ms=120.0,
            tokens_generated=24,
            output='{"invoice_id": "INV-1001", "currency": "USD", "total": 420.5, "paid": true}',
        ),
        ModelSample(
            prompt_id="ext_inv_02",
            model_name="baseline-fp16",
            latency_ms=115.0,
            tokens_generated=25,
            output='{"invoice_id": "INV-1002", "currency": "EUR", "total": 980.0, "paid": false}',
        ),
        ModelSample(
            prompt_id="ext_inv_03",
            model_name="baseline-fp16",
            latency_ms=110.0,
            tokens_generated=23,
            output='{"invoice_id": "INV-1003", "currency": "GBP", "total": 150.0, "paid": true}',
        ),
    )
    candidate_pass = (
        ModelSample(
            prompt_id="ext_inv_01",
            model_name="candidate-q8",
            latency_ms=85.0,
            tokens_generated=24,
            output='{"invoice_id": "INV-1001", "currency": "USD", "total": 420.5, "paid": true}',
        ),
        ModelSample(
            prompt_id="ext_inv_02",
            model_name="candidate-q8",
            latency_ms=82.0,
            tokens_generated=25,
            output='{"invoice_id": "INV-1002", "currency": "EUR", "total": 980.0, "paid": false}',
        ),
        ModelSample(
            prompt_id="ext_inv_03",
            model_name="candidate-q8",
            latency_ms=80.0,
            tokens_generated=23,
            output='{"invoice_id": "INV-1003", "currency": "GBP", "total": 150.0, "paid": true}',
        ),
    )
    candidate_drift = (
        ModelSample(
            prompt_id="ext_inv_01",
            model_name="candidate-q4",
            latency_ms=72.0,
            tokens_generated=24,
            output='{"invoice_id": "INV-1001", "currency": "USD", "total": 42.05, "paid": true}',
        ),
        ModelSample(
            prompt_id="ext_inv_02",
            model_name="candidate-q4",
            latency_ms=70.0,
            tokens_generated=20,
            output='{"invoice_id": "INV-1002", "currency": "EUR", "total": 980.0}',
        ),
        ModelSample(
            prompt_id="ext_inv_03",
            model_name="candidate-q4",
            latency_ms=68.0,
            tokens_generated=23,
            output='{"invoice_id": "INV-1003", "currency": "USD", "total": 150.0, "paid": true}',
        ),
    )
    return BenchmarkPack(
        name="extraction",
        description="Structured invoice JSON extraction with strict schema and numeric totals.",
        prompts=prompts,
        baseline=baseline,
        candidate_pass=candidate_pass,
        candidate_drift=candidate_drift,
        policy_toml=render_wizard_config(PRESET_ANSWERS["extraction"]),
    )


def _build_classification_pack() -> BenchmarkPack:
    cls_schema: dict[str, Any] = {
        "type": "object",
        "required": ["label", "confidence"],
        "properties": {
            "label": {"type": "string", "enum": ["urgent", "normal", "low"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "additionalProperties": False,
    }
    prompts = (
        PromptCase(
            prompt_id="cls_01",
            category="classification",
            prompt="Classify ticket: Production database primary node unreachable.",
            schema=cls_schema,
        ),
        PromptCase(
            prompt_id="cls_02",
            category="classification",
            prompt="Classify ticket: Payment gateway returning 502 for checkout.",
            schema=cls_schema,
        ),
        PromptCase(
            prompt_id="cls_03",
            category="classification",
            prompt="Classify ticket: Billing export PDF missing company logo.",
            schema=cls_schema,
        ),
        PromptCase(
            prompt_id="cls_04",
            category="classification",
            prompt="Classify ticket: Feature request for dark-mode calendar icon.",
            schema=cls_schema,
        ),
    )
    baseline = (
        ModelSample(
            prompt_id="cls_01",
            model_name="baseline-fp16",
            latency_ms=95.0,
            tokens_generated=12,
            output='{"label": "urgent", "confidence": 0.96}',
        ),
        ModelSample(
            prompt_id="cls_02",
            model_name="baseline-fp16",
            latency_ms=92.0,
            tokens_generated=12,
            output='{"label": "urgent", "confidence": 0.94}',
        ),
        ModelSample(
            prompt_id="cls_03",
            model_name="baseline-fp16",
            latency_ms=90.0,
            tokens_generated=12,
            output='{"label": "normal", "confidence": 0.88}',
        ),
        ModelSample(
            prompt_id="cls_04",
            model_name="baseline-fp16",
            latency_ms=88.0,
            tokens_generated=12,
            output='{"label": "low", "confidence": 0.92}',
        ),
    )
    candidate_pass = (
        ModelSample(
            prompt_id="cls_01",
            model_name="candidate-q8",
            latency_ms=70.0,
            tokens_generated=12,
            output='{"label": "urgent", "confidence": 0.95}',
        ),
        ModelSample(
            prompt_id="cls_02",
            model_name="candidate-q8",
            latency_ms=68.0,
            tokens_generated=12,
            output='{"label": "urgent", "confidence": 0.93}',
        ),
        ModelSample(
            prompt_id="cls_03",
            model_name="candidate-q8",
            latency_ms=66.0,
            tokens_generated=12,
            output='{"label": "normal", "confidence": 0.87}',
        ),
        ModelSample(
            prompt_id="cls_04",
            model_name="candidate-q8",
            latency_ms=65.0,
            tokens_generated=12,
            output='{"label": "low", "confidence": 0.91}',
        ),
    )
    candidate_drift = (
        ModelSample(
            prompt_id="cls_01",
            model_name="candidate-q4",
            latency_ms=60.0,
            tokens_generated=12,
            output='{"label": "normal", "confidence": 0.55}',
        ),
        ModelSample(
            prompt_id="cls_02",
            model_name="candidate-q4",
            latency_ms=59.0,
            tokens_generated=12,
            output='{"label": "normal", "confidence": 0.52}',
        ),
        ModelSample(
            prompt_id="cls_03",
            model_name="candidate-q4",
            latency_ms=58.0,
            tokens_generated=12,
            output='{"label": "normal", "confidence": 0.50}',
        ),
        ModelSample(
            prompt_id="cls_04",
            model_name="candidate-q4",
            latency_ms=57.0,
            tokens_generated=12,
            output='{"label": "normal", "confidence": 0.48}',
        ),
    )
    base_policy = build_policy_from_answers(PRESET_ANSWERS["classification"])
    cls_policy = DriftPolicy(
        thresholds=base_policy.thresholds,
        weights=base_policy.weights,
        field_rules={"$.confidence": FieldRule(path="$.confidence", tolerance_abs=0.03)},
        long_text_mode=base_policy.long_text_mode,
    )
    return BenchmarkPack(
        name="classification",
        description="Support triage routing with enum labels and calibrated confidence scores.",
        prompts=prompts,
        baseline=baseline,
        candidate_pass=candidate_pass,
        candidate_drift=candidate_drift,
        policy_toml="# Starter classification policy (allows +-0.03 confidence jitter)\n"
        + render_policy_toml(cls_policy),
    )


def _build_coding_pack() -> BenchmarkPack:
    prompts = (
        PromptCase(
            prompt_id="code_01",
            category="coding",
            prompt="Write a Python function sum_even(nums) returning the sum of even integers.",
        ),
        PromptCase(
            prompt_id="code_02",
            category="coding",
            prompt="Write a Python function normalize_slug(text) that lowercases and hyphenates.",
        ),
        PromptCase(
            prompt_id="code_03",
            category="coding",
            prompt="Write a Python function safe_div(a, b) returning None when b == 0.",
        ),
    )
    baseline = (
        ModelSample(
            prompt_id="code_01",
            model_name="baseline-fp16",
            latency_ms=140.0,
            tokens_generated=30,
            output=(
                "def sum_even(nums: list[int]) -> int:\n"
                "    return sum(n for n in nums if n % 2 == 0)\n"
            ),
        ),
        ModelSample(
            prompt_id="code_02",
            model_name="baseline-fp16",
            latency_ms=135.0,
            tokens_generated=32,
            output=(
                "def normalize_slug(text: str) -> str:\n"
                "    return '-'.join(text.strip().lower().split())\n"
            ),
        ),
        ModelSample(
            prompt_id="code_03",
            model_name="baseline-fp16",
            latency_ms=130.0,
            tokens_generated=28,
            output=(
                "def safe_div(a: float, b: float) -> float | None:\n"
                "    return None if b == 0 else a / b\n"
            ),
        ),
    )
    candidate_pass = (
        ModelSample(
            prompt_id="code_01",
            model_name="candidate-q8",
            latency_ms=105.0,
            tokens_generated=30,
            output=(
                "def sum_even(nums: list[int]) -> int:\n"
                "    return sum(n for n in nums if n % 2 == 0)\n"
            ),
        ),
        ModelSample(
            prompt_id="code_02",
            model_name="candidate-q8",
            latency_ms=100.0,
            tokens_generated=32,
            output=(
                "def normalize_slug(text: str) -> str:\n"
                "    return '-'.join(text.strip().lower().split())\n"
            ),
        ),
        ModelSample(
            prompt_id="code_03",
            model_name="candidate-q8",
            latency_ms=98.0,
            tokens_generated=28,
            output=(
                "def safe_div(a: float, b: float) -> float | None:\n"
                "    return None if b == 0 else a / b\n"
            ),
        ),
    )
    candidate_drift = (
        ModelSample(
            prompt_id="code_01",
            model_name="candidate-q4",
            latency_ms=90.0,
            tokens_generated=60,
            output=(
                "def sum_even(nums):\n"
                "    # verify the list of even numbers carefully\n" * 4
                + "    return 0\n"
            ),
        ),
        ModelSample(
            prompt_id="code_02",
            model_name="candidate-q4",
            latency_ms=88.0,
            tokens_generated=10,
            output="pass",
        ),
        ModelSample(
            prompt_id="code_03",
            model_name="candidate-q4",
            latency_ms=85.0,
            tokens_generated=15,
            output="I'm sorry, I cannot assist with writing division helpers.",
        ),
    )
    return BenchmarkPack(
        name="coding",
        description="Python code generation pack highlighting loops, truncation, and refusals.",
        prompts=prompts,
        baseline=baseline,
        candidate_pass=candidate_pass,
        candidate_drift=candidate_drift,
        policy_toml=render_wizard_config(PRESET_ANSWERS["coding"]),
    )


def _build_rag_pack() -> BenchmarkPack:
    rag_schema: dict[str, Any] = {
        "type": "object",
        "required": ["answer", "citations"],
        "properties": {
            "answer": {"type": "string", "minLength": 10},
            "citations": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
        },
        "additionalProperties": False,
    }
    prompts = (
        PromptCase(
            prompt_id="rag_01",
            category="rag",
            prompt="Answer from context: What is the SLA uptime target and credit cap?",
            schema=rag_schema,
        ),
        PromptCase(
            prompt_id="rag_02",
            category="rag",
            prompt="Answer from context: How many days are audit logs retained by default?",
            schema=rag_schema,
        ),
        PromptCase(
            prompt_id="rag_03",
            category="rag",
            prompt="Answer from context: Which TLS versions are accepted on the API gateway?",
            schema=rag_schema,
        ),
    )
    baseline = (
        ModelSample(
            prompt_id="rag_01",
            model_name="baseline-fp16",
            latency_ms=150.0,
            tokens_generated=35,
            output=(
                '{"answer": "The SLA guarantees 99.95% uptime with a 25% monthly credit cap.", '
                '"citations": ["doc-sla-1", "doc-sla-2"]}'
            ),
        ),
        ModelSample(
            prompt_id="rag_02",
            model_name="baseline-fp16",
            latency_ms=145.0,
            tokens_generated=30,
            output=(
                '{"answer": "Audit logs are retained for 90 days by default.", '
                '"citations": ["doc-sec-4"]}'
            ),
        ),
        ModelSample(
            prompt_id="rag_03",
            model_name="baseline-fp16",
            latency_ms=140.0,
            tokens_generated=32,
            output=(
                '{"answer": "The API gateway accepts TLS 1.2 and TLS 1.3 only.", '
                '"citations": ["doc-net-2", "doc-net-3"]}'
            ),
        ),
    )
    candidate_pass = (
        ModelSample(
            prompt_id="rag_01",
            model_name="candidate-q8",
            latency_ms=112.0,
            tokens_generated=35,
            output=(
                '{"answer": "The SLA guarantees 99.95% uptime with a 25% monthly credit cap.", '
                '"citations": ["doc-sla-2", "doc-sla-1"]}'
            ),
        ),
        ModelSample(
            prompt_id="rag_02",
            model_name="candidate-q8",
            latency_ms=108.0,
            tokens_generated=30,
            output=(
                '{"answer": "Audit logs are retained for 90 days by default.", '
                '"citations": ["doc-sec-4"]}'
            ),
        ),
        ModelSample(
            prompt_id="rag_03",
            model_name="candidate-q8",
            latency_ms=106.0,
            tokens_generated=32,
            output=(
                '{"answer": "The API gateway accepts TLS 1.2 and TLS 1.3 only.", '
                '"citations": ["doc-net-3", "doc-net-2"]}'
            ),
        ),
    )
    candidate_drift = (
        ModelSample(
            prompt_id="rag_01",
            model_name="candidate-q4",
            latency_ms=95.0,
            tokens_generated=34,
            output=(
                '{"answer": "The SLA guarantees 99.0% uptime with a 10% credit cap.", '
                '"citations": []}'
            ),
        ),
        ModelSample(
            prompt_id="rag_02",
            model_name="candidate-q4",
            latency_ms=92.0,
            tokens_generated=28,
            output='{"answer": "Audit logs are kept for 30 days.", "citations": ["doc-sec-4"]}',
        ),
        ModelSample(
            prompt_id="rag_03",
            model_name="candidate-q4",
            latency_ms=90.0,
            tokens_generated=25,
            output='{"answer": "All TLS versions are allowed.", "citations": ["doc-net-1"]}',
        ),
    )
    rag_answers = WizardAnswers(
        use_case="rag",
        numeric_sensitivity="moderate",
        unordered_array_paths=("$.citations",),
        track_latency=True,
        long_outputs=True,
    )
    return BenchmarkPack(
        name="rag",
        description="Grounded Q&A with order-insensitive citations and factual figure stability.",
        prompts=prompts,
        baseline=baseline,
        candidate_pass=candidate_pass,
        candidate_drift=candidate_drift,
        policy_toml=render_wizard_config(rag_answers),
    )


def _build_agent_pack() -> BenchmarkPack:
    tool_schema: dict[str, Any] = {
        "type": "object",
        "required": ["tool", "target", "dry_run", "max_retries"],
        "properties": {
            "tool": {
                "type": "string",
                "enum": ["read_file", "search_code", "run_tests", "edit_file"],
            },
            "target": {"type": "string", "minLength": 1},
            "dry_run": {"type": "boolean"},
            "max_retries": {"type": "integer", "minimum": 0, "maximum": 5},
        },
        "additionalProperties": False,
    }
    prompts = (
        PromptCase(
            prompt_id="agent_01",
            category="agent",
            prompt="Choose the safest first tool action before modifying src/auth.py.",
            schema=tool_schema,
        ),
        PromptCase(
            prompt_id="agent_02",
            category="agent",
            prompt="Dispatch a dry-run test execution on tests/test_auth.py with max 1 retry.",
            schema=tool_schema,
        ),
        PromptCase(
            prompt_id="agent_03",
            category="agent",
            prompt="Search the codebase for token expiry handlers without editing files.",
            schema=tool_schema,
        ),
    )
    baseline = (
        ModelSample(
            prompt_id="agent_01",
            model_name="baseline-fp16",
            latency_ms=110.0,
            tokens_generated=22,
            output=(
                '{"tool": "read_file", "target": "src/auth.py", '
                '"dry_run": true, "max_retries": 0}'
            ),
        ),
        ModelSample(
            prompt_id="agent_02",
            model_name="baseline-fp16",
            latency_ms=105.0,
            tokens_generated=23,
            output=(
                '{"tool": "run_tests", "target": "tests/test_auth.py", '
                '"dry_run": true, "max_retries": 1}'
            ),
        ),
        ModelSample(
            prompt_id="agent_03",
            model_name="baseline-fp16",
            latency_ms=100.0,
            tokens_generated=21,
            output=(
                '{"tool": "search_code", "target": "token_expiry", '
                '"dry_run": true, "max_retries": 0}'
            ),
        ),
    )
    candidate_pass = (
        ModelSample(
            prompt_id="agent_01",
            model_name="candidate-q8",
            latency_ms=82.0,
            tokens_generated=22,
            output=(
                '{"tool": "read_file", "target": "src/auth.py", '
                '"dry_run": true, "max_retries": 0}'
            ),
        ),
        ModelSample(
            prompt_id="agent_02",
            model_name="candidate-q8",
            latency_ms=79.0,
            tokens_generated=23,
            output=(
                '{"tool": "run_tests", "target": "tests/test_auth.py", '
                '"dry_run": true, "max_retries": 1}'
            ),
        ),
        ModelSample(
            prompt_id="agent_03",
            model_name="candidate-q8",
            latency_ms=76.0,
            tokens_generated=21,
            output=(
                '{"tool": "search_code", "target": "token_expiry", '
                '"dry_run": true, "max_retries": 0}'
            ),
        ),
    )
    candidate_drift = (
        ModelSample(
            prompt_id="agent_01",
            model_name="candidate-q4",
            latency_ms=68.0,
            tokens_generated=23,
            output=(
                '{"tool": "edit_file", "target": "src/auth.py", '
                '"dry_run": false, "max_retries": 0}'
            ),
        ),
        ModelSample(
            prompt_id="agent_02",
            model_name="candidate-q4",
            latency_ms=66.0,
            tokens_generated=24,
            output=(
                '{"tool": "run_tests", "target": "tests/test_auth.py", '
                '"dry_run": false, "max_retries": 10}'
            ),
        ),
        ModelSample(
            prompt_id="agent_03",
            model_name="candidate-q4",
            latency_ms=65.0,
            tokens_generated=22,
            output=(
                '{"tool": "shell_exec", "target": "token_expiry", '
                '"dry_run": true, "max_retries": 0}'
            ),
        ),
    )
    return BenchmarkPack(
        name="agent",
        description=(
            "Structured agent tool-call dispatch with strict action enums and safety flags."
        ),
        prompts=prompts,
        baseline=baseline,
        candidate_pass=candidate_pass,
        candidate_drift=candidate_drift,
        policy_toml=render_wizard_config(PRESET_ANSWERS["strict"]),
    )


def list_packs() -> tuple[str, ...]:
    """Return all available built-in starter benchmark pack names."""
    return _PACK_NAMES


def get_pack(name: str) -> BenchmarkPack:
    """Return a built-in starter BenchmarkPack by name."""
    key = name.strip().lower()
    if key == "extraction":
        return _build_extraction_pack()
    if key == "classification":
        return _build_classification_pack()
    if key == "coding":
        return _build_coding_pack()
    if key == "rag":
        return _build_rag_pack()
    if key == "agent":
        return _build_agent_pack()
    raise ValueError(f"Unknown starter pack {name!r}. Choose from {_PACK_NAMES}.")


def scaffold_pack(name: str, target_dir: str | Path) -> dict[str, Path]:
    """Write prompts, baseline, passing/drifted candidates, and drift.toml for a pack."""
    pack = get_pack(name)
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)
    files = {
        "prompts": dest / "prompts.json",
        "baseline": dest / "baseline.jsonl",
        "candidate_pass": dest / "candidate-pass.jsonl",
        "candidate_drift": dest / "candidate-drift.jsonl",
        "policy": dest / "drift.toml",
    }
    files["prompts"].write_text(
        json.dumps([_case_to_dict(c) for c in pack.prompts], indent=2) + "\n",
        encoding="utf-8",
    )
    files["baseline"].write_text(serialize_samples_jsonl(pack.baseline), encoding="utf-8")
    files["candidate_pass"].write_text(
        serialize_samples_jsonl(pack.candidate_pass), encoding="utf-8"
    )
    files["candidate_drift"].write_text(
        serialize_samples_jsonl(pack.candidate_drift), encoding="utf-8"
    )
    files["policy"].write_text(pack.policy_toml, encoding="utf-8")
    return files
