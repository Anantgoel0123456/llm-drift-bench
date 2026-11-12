# llm-drift-bench

[![CI](https://github.com/anant-goel/llm-drift-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/anant-goel/llm-drift-bench/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Runtime dependencies](https://img.shields.io/badge/runtime-Pydantic%20%2B%20Rich-blue.svg)](pyproject.toml)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typed: Mypy Strict](https://img.shields.io/badge/typing-mypy%20strict-blue.svg)](http://mypy-lang.org/)

**`llm-drift-bench`** is a Python CLI and library for comparing captured LLM outputs across model updates, prompt revisions, and quantizations. It measures lexical differences, JSON structure and field agreement, numeric changes, and recorded latency.

---

## Why `llm-drift-bench`?

When upgrading model checkpoints or quantizing weights for production inference, subtle regressions frequently slip past perplexity benchmarks:
- **Broken JSON Conformity**: Markdown fence artifacts or missing required keys in structured outputs.
- **Structural Key-Path Drift**: Schema-valid payloads that drop optional nested fields or alter value types.
- **Numeric Field Drift**: Extracted financial or confidence numbers drifting beyond acceptable tolerances.
- **Tail Latency Regressions**: P95 latency spikes across specific prompt categories.

The runtime uses Pydantic and Rich, together with their transitive dependencies.

Similarity measures agreement with a baseline, not factual correctness or semantic equivalence. The validator implements a JSON Schema subset and rejects unsupported keywords before checking the response. It is not a complete JSON Schema implementation.

---

## Quickstart after publication

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
 Mean Composite Drift                 |     0.0000 (ref) |             0.0426
 Max Composite Drift                  |     0.0000 (ref) |             0.0852
 Mean Token Jaccard Similarity        |     1.0000 (ref) |             0.9412
 Mean Levenshtein Similarity          |     1.0000 (ref) |             0.9930
 Mean JSON Structural Similarity      |     1.0000 (ref) |             1.0000
 Mean JSON Numeric Relative Drift     |     0.0000 (ref) |             0.0160
 P50 Latency (ms)                     |            130.2 |               82.8
 P95 Latency (ms)                     |            141.3 |               88.6
------------------------------------------------------------------------------
 Prompt ID          | Category     | JSON OK   | Struct Sim | Drift    | State  
------------------------------------------------------------------------------
 extract_invoice_01 | extraction   | YES       |     1.0000 |   0.0000 | OK     
 classify_ticket_02 | classificati | YES       |     1.0000 |   0.0852 | OK
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
| `--format` | `table` | Output format on stdout (`table`, `rich`, `json`, or `markdown`). |
| `--fail-on-drift` | `False` | Exit with code `1` if any threshold is violated or sample drifts. |

### Current behavior of the unreleased revision

Levenshtein similarity always compares characters (Unicode code points),
including whitespace and case. Each response is limited to 16,384 characters
for this metric. Longer responses raise an error (CLI exit code `2`), rather
than being truncated or scored using a different unit. Unicode normalization
is not applied.

Commands are `run`, `report`, `validate`, and `init-sample`.
`--suite` and `--prompts` are aliases for the suite path.

Use existing captured-output files for both comparison targets.
The current model-name fallback generates synthetic data and does not
contact a model endpoint. A missing file can accidentally activate that
fallback.

Table and JSON output use the benchmark engine. Rich and Markdown output
currently recompute scores through a second evaluator with different
weights. Until these paths are unified, their scores are not interchangeable.

Saved engine reports also lose information when converted by `report`.
Do not treat converted throughput or mean-latency values as measurements.

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
