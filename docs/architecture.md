# Architecture Overview: `llm-drift-bench`

`llm-drift-bench` is designed as a lightweight, zero-bloat Python library and CLI tool to benchmark LLM outputs, validate JSON schema conformity, and quantify behavioral/structural drift across model updates or quantizations (e.g., `FP16` vs. `INT4` / `Q4_K_M`, or `v1` vs. `v2` checkpoints).

## Core Modules (`src/llm_drift_bench/`)

1. **`models.py` (Data Schemas & Domain Types)**
   - Immutable, strongly typed dataclasses representing benchmark prompts (`PromptCase`), model responses (`ModelSample`), JSON validation outcomes (`SchemaValidationResult`), pairwise drift metrics (`SampleDriftResult`), and aggregate benchmark reports (`DriftBenchmarkReport`).

2. **`validator.py` (Zero-Bloat JSON Extraction & Conformity Validator)**
   - Extracts JSON payloads from raw LLM completions (handling markdown code fences or pure JSON).
   - Validates JSON payloads against JSON Schema Draft-like rules (`type`, `required`, `properties`, `enum`, `minimum`, `maximum`, `minLength`, `maxLength`, `items`, `additionalProperties`) with detailed path-level error diagnostics—requiring zero heavy third-party dependencies.

3. **`metrics.py` (Output Drift & Similarity Calculators)**
   - Token-level Jaccard and N-gram overlap similarity.
   - Normalized Levenshtein character/token edit distance.
   - Structural JSON key-path & value-type stability score (`json_structural_similarity`).
   - Numeric field relative drift (`json_numeric_drift`) and semantic field exact-match ratio.
   - Latency & throughput delta calculations.

4. **`engine.py` (Benchmark & Drift Evaluation Engine)**
   - Orchestrates evaluation comparing a **Baseline** dataset/provider against a **Candidate** dataset/provider across a suite of prompts and optional JSON schemas.
   - Computes composite drift scores, pass/fail threshold checks, and summary statistics (P50/P95 latency, JSON conformity delta, mean/max drift).

5. **`cli.py` (Command-Line Interface)**
   - Built on standard `argparse` for zero-dependency execution (`llm-drift-bench`).
   - Supports `run` (compare baseline vs. candidate outputs), `validate` (check JSON conformity of outputs against a schema), and `inspect` (generate or inspect sample benchmark suites), with terminal table formatting, JSON report export, and CI fail-on-drift exit codes.
