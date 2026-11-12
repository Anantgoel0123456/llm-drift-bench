# llm-drift-bench

[![CI](https://github.com/anant-goel/llm-drift-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/anant-goel/llm-drift-bench/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](pyproject.toml)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typed: Mypy Strict](https://img.shields.io/badge/typing-mypy%20strict-blue.svg)](http://mypy-lang.org/)

**`llm-drift-bench`** is a lightweight, zero-bloat CLI and Python library to benchmark LLM outputs, validate JSON Schema conformity, and detect behavioral or structural output drift across model updates, prompt revisions, and quantizations (e.g., `FP16` vs. `Q4_K_M` / `INT4`).

---

## Why `llm-drift-bench`?

When upgrading model checkpoints or quantizing weights for production inference, subtle regressions frequently slip past perplexity benchmarks:
- **Broken JSON Conformity**: Markdown fence artifacts or missing required keys in structured outputs.
- **Structural Key-Path Drift**: Schema-valid payloads that drop optional nested fields or alter value types.
- **Numeric Field Drift**: Extracted financial or confidence numbers drifting beyond acceptable tolerances.
- **Tail Latency Regressions**: P95 latency spikes across specific prompt categories.

`llm-drift-bench` catches these regressions deterministically in CI/CD with **zero third-party runtime dependencies**.

---

## 3-Line Quickstart

```bash
pip install llm-drift-bench
llm-drift-bench init-sample --dir ./demo
llm-drift-bench run -b ./demo/baseline_fp16.jsonl -c ./demo/candidate_int4.jsonl -p ./demo/prompts.json
```

---

## Visual Terminal Output

### 1. Drift Benchmark Comparison (`llm-drift-bench run`)

```text
==============================================================================
 LLM DRIFT BENCHMARK REPORT  [PASSED]
==============================================================================
 Baseline Model  : llama-3-8b-fp16
 Candidate Model : llama-3-8b-q4_k_m
 Total Samples   : 2 (Drifted: 0 / 0.0%)
------------------------------------------------------------------------------
 Metric                               | Baseline         | Candidate / Value 
------------------------------------------------------------------------------
 JSON Conformity Rate                 |          100.0% |            100.0%
 Mean Composite Drift                 |     0.0000 (ref) |             0.0298
 Max Composite Drift                  |     0.0000 (ref) |             0.0596
 Mean Token Jaccard Similarity        |     1.0000 (ref) |             0.9615
 Mean Levenshtein Similarity          |     1.0000 (ref) |             0.9932
 Mean JSON Structural Similarity      |     1.0000 (ref) |             1.0000
 Mean JSON Numeric Relative Drift     |     0.0000 (ref) |             0.0160
 P50 Latency (ms)                     |            130.2 |               82.8
 P95 Latency (ms)                     |            141.3 |               88.6
------------------------------------------------------------------------------
 Prompt ID          | Category     | JSON OK   | Struct Sim | Drift    | State  
------------------------------------------------------------------------------
 extract_invoice_01 | extraction   | YES       |     1.0000 |   0.0000 | OK     
 classify_ticket_02 | classificati | YES       |     1.0000 |   0.0596 | OK     
==============================================================================
```

### 2. Standalone JSON Conformity Validation (`llm-drift-bench validate`)

```bash
llm-drift-bench validate -i ./demo/candidate_int4.jsonl -p ./demo/prompts.json --strict
```

```text
========================================================================
 JSON CONFORMITY VALIDATION  [2/2 passed (100.0%)]
========================================================================
 [PASS] extract_invoice_01 | Conformant
 [PASS] classify_ticket_02 | Conformant
========================================================================
```

---

## Input & Configuration Formats

### Prompt Suite with JSON Schemas (`prompts.json`)

```json
[
  {
    "prompt_id": "extract_invoice_01",
    "category": "extraction",
    "prompt": "Extract invoice fields as JSON.",
    "schema": {
      "type": "object",
      "required": ["invoice_id", "total_usd", "status"],
      "properties": {
        "invoice_id": {"type": "string", "minLength": 3, "pattern": "^INV-"},
        "total_usd": {"type": "number", "minimum": 0},
        "status": {"type": "string", "enum": ["paid", "pending", "overdue"]}
      },
      "additionalProperties": false
    }
  }
]
```

### Model Outputs (`baseline_fp16.jsonl` / `candidate_int4.jsonl`)

```jsonl
{"prompt_id": "extract_invoice_01", "model_name": "llama-3-8b-fp16", "latency_ms": 142.5, "output": "{\"invoice_id\": \"INV-2026-09\", \"total_usd\": 1250.00, \"status\": \"paid\"}"}
```

---

## CLI Reference & CI Gating

Gate your CI/CD pipeline on model drift and JSON conformity thresholds using `--fail-on-drift`:

```bash
llm-drift-bench run \
  --baseline ./baseline_fp16.jsonl \
  --candidate ./candidate_int4.jsonl \
  --prompts ./prompts.json \
  --max-drift 0.15 \
  --min-conformity 0.98 \
  --max-numeric-drift 0.05 \
  --max-latency-regression 25.0 \
  --output ./artifacts/drift_report.json \
  --fail-on-drift
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-b`, `--baseline` | *(Required)* | Path to baseline model outputs (`.json` or `.jsonl`). |
| `-c`, `--candidate` | *(Required)* | Path to candidate model outputs (`.json` or `.jsonl`). |
| `-p`, `--prompts` | `None` | Optional prompt definitions with per-prompt JSON Schemas. |
| `--max-drift` | `0.25` | Maximum allowed mean/per-sample composite drift score `[0.0, 1.0]`. |
| `--min-conformity` | `0.95` | Minimum required candidate JSON Schema conformity rate `[0.0, 1.0]`. |
| `--max-numeric-drift` | `0.15` | Maximum allowed relative numeric drift on shared JSON number fields. |
| `--max-latency-regression` | `50.0` | Maximum allowed P95 latency regression percentage (`%`). |
| `-o`, `--output` | `None` | File path to save the structured JSON benchmark report. |
| `--format` | `table` | Output format on stdout (`table` or `json`). |
| `--fail-on-drift` | `False` | Exit with code `1` if any threshold is violated or sample drifts. |

---

## Python Library API

Use `llm-drift-bench` directly inside your evaluation pipelines or pytest suites:

```python
from llm_drift_bench import (
    DriftBenchmarkEngine,
    DriftThresholds,
    ModelSample,
    PromptCase,
    format_report_table,
)

prompts = [
    PromptCase(
        prompt_id="invoice_1",
        prompt="Extract invoice total.",
        schema={
            "type": "object",
            "required": ["total"],
            "properties": {"total": {"type": "number", "minimum": 0}},
        },
    )
]

baseline = [ModelSample(prompt_id="invoice_1", model_name="fp16", output='{"total": 100.0}')]
candidate = [ModelSample(prompt_id="invoice_1", model_name="int4", output='{"total": 99.5}')]

engine = DriftBenchmarkEngine(thresholds=DriftThresholds(max_composite_drift=0.10))
report = engine.evaluate(baseline, candidate, prompts)

print(format_report_table(report))
assert report.passed
```

---

## Contributing & License

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing instructions, and our `good first issue` roadmap.

Licensed under the [MIT License](LICENSE).
