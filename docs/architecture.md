# Architecture of llm-drift-bench

This document describes the current unreleased implementation.

The package compares captured LLM responses using lexical similarity,
JSON structure and field agreement, numeric drift, and recorded latency.
These metrics do not establish factual correctness or semantic equivalence.

Runtime dependencies are Pydantic and Rich, including their transitive
dependencies.

## Data models

`models.py` contains two model families:

- Frozen dataclasses: PromptCase, ModelSample, ValidationIssue,
  SchemaValidationResult, DriftThresholds, SampleDriftResult, and
  DriftBenchmarkReport.
- Pydantic models: TestPrompt, DriftMetrics, and BenchmarkResult.

Frozen dataclasses prevent field reassignment but do not make nested
dictionaries or lists immutable.

The two report families contain different metrics and are not losslessly
interchangeable.

## Input loaders

`engine.py` loads PromptCase and ModelSample records from JSON or JSONL.

PromptCase wrappers accept `prompts`, `cases`, or `items`.
ModelSample wrappers accept `samples`, `outputs`, or `results`.

`evaluator.py` separately loads TestPrompt records. Its wrappers accept
`prompts`, `cases`, or `suite`.

All current file readers use UTF-8 without explicit BOM support.
Files are read into memory.

## JSON extraction and validation

`validator.py` attempts direct JSON parsing, then a Markdown code fence,
then object or array substring extraction from prose.

A successful extraction does not imply the complete response was JSON.

The handwritten schema subset implements:

- type
- enum and const
- minLength, maxLength, and pattern
- minimum and maximum
- minItems, maxItems, and dictionary-valued items
- required and dictionary-valued properties
- boolean or dictionary-valued additionalProperties

This is not a complete JSON Schema implementation. A schema support check
rejects unsupported keywords before instance validation, including rules
for absent properties and empty arrays. Unsupported schema forms are
reported explicitly. Schema traversal is limited to depth 64 and 10,000
visited nodes.

## Metrics

`metrics.py` implements:

- Lowercased word and punctuation tokenization.
- Token-set Jaccard similarity.
- Token n-gram multiset Dice similarity.
- Normalized Levenshtein similarity.
- Typed JSON leaf-path Jaccard similarity.
- Exact JSON leaf-value agreement.
- Mean bounded symmetric numeric drift.
- A fixed-weight composite drift score.
- Linear-interpolated percentiles.

Levenshtein uses exact character-level edit distance at every supported
length, implemented with bit-vector dynamic programming. Characters are
Unicode code points; whitespace, case, and normalization differences count.
Similarity is `1 - distance / max(len(a), len(b))`. Both empty inputs score
1.0, while exactly one empty input scores 0.0. Each input is limited to
16,384 characters to bound computation and memory. Larger inputs raise
ValueError, even for identical strings. The CLI reports this as input error
exit code 2. Inputs are never silently truncated or compared as tokens.

The JSON path representation currently does not escape object keys.

`evaluator.py` additionally implements SequenceMatcher token difference,
a lexical and structural similarity blend, and throughput calculations.
Its similarity blend is not embedding-based semantic similarity.

## Engine evaluation

DriftBenchmarkEngine compares baseline and candidate records by prompt ID.
It validates responses, calculates pair metrics, aggregates results, and
applies thresholds.

Its report contains mean and maximum drift, conformity rates, latency
percentiles, individual drift flags, and aggregate failure reasons.

Current input matching does not consistently reject duplicate IDs,
extra records, or incomplete suite coverage.

## Alternate evaluator

`evaluate_suite` produces BenchmarkResult using a different scoring formula.

Targets must be existing captured-output files or explicit ModelSample
sequences. Missing files, directories, and model identifiers raise input
errors; evaluation never generates responses or calls model endpoints.
Inline outputs in TestPrompt do not bypass this requirement. Use explicit
ModelSample sequences for in-memory comparisons.

Missing latency and throughput remain None and serialize as JSON null.
Throughput requires an explicit model token count and positive measured
latency. Zero latency is retained but yields no throughput. Negative or
non-finite measurements and invalid token counts are rejected.

Aggregate measurement means and percentiles require complete data for the
respective side of the suite. Otherwise they are unavailable. Latency
regression thresholds are skipped when either side is incomplete or baseline
P95 is zero. The table explicitly identifies a skipped latency check; passing
other thresholds does not certify latency performance.

## CLI and reporting

Commands are:

- run
- report
- validate
- init-sample

There is no inspect command.

For file-based comparisons, table and JSON output use DriftBenchmarkReport.

Rich and Markdown output currently reevaluate the same samples through
evaluate_suite and render BenchmarkResult. Therefore output format can
change the displayed score.

The report command converts saved engine reports into BenchmarkResult.
Unavailable timing and throughput fields remain None and render as n/a.
Aggregate percentiles cannot reconstruct means or per-prompt observations.
Other converted similarity and exact-match fields are still approximations;
the conversion is not lossless.

The release implementation should calculate one canonical report and
render it without recomputing scores or manufacturing measurements.

## Entry points and packaging

The console entry point calls llm_drift_bench.cli.main.
The module entry point supports python -m llm_drift_bench.
The package includes a py.typed marker.

The version currently appears independently in pyproject.toml and
__init__.py. Release preparation should establish one source of truth.

## Testing limitations

Coverage is not evidence that all benchmark semantics are correct.

Release checks must include unsupported schemas, malformed inputs,
duplicate and mismatched IDs, missing measurements, format consistency,
module execution, and installation of built distributions outside the
source checkout.
