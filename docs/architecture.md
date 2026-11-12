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

This is not a complete JSON Schema implementation. Unsupported keywords
currently may be ignored. This behavior must be corrected before release.

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

Levenshtein currently switches from characters to tokens above 1,024
characters. This discontinuity must be removed before release.

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

A string target that is not an existing file currently activates synthetic
sample generation. No model endpoint is called. Missing latency may also
be replaced by synthetic values.

This fallback is unsuitable for measured benchmarks and must be removed
from normal comparison paths before release.

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
That conversion currently invents unavailable metrics and mislabels some
latency statistics. It is not lossless.

The release implementation should calculate one canonical report and
render it without recomputing scores or manufacturing measurements.

## Entry points and packaging

The console entry point calls llm_drift_bench.cli.main.
The module entry point supports python -m llm_drift_bench.
The package includes a py.typed marker.

The version currently appears independently in pyproject.toml and
__init__.py. Release preparation should establish one source of truth.

## Testing limitations

The current test suite has 36 tests. Coverage is not evidence that all
benchmark semantics are correct.

Release checks must include unsupported schemas, malformed inputs,
duplicate and mismatched IDs, missing measurements, format consistency,
module execution, and installation of built distributions outside the
source checkout.
