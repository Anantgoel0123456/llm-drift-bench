# Architecture of llm-drift-bench

This document describes the `0.4.0` implementation.

The package compares captured LLM responses using lexical similarity,
JSON structure and field agreement, numeric drift, and recorded latency.
These metrics do not establish factual correctness or semantic equivalence.

Runtime dependencies are Pydantic and Rich, including their transitive
dependencies.

## Data models

`models.py` exposes two model families while staying lightweight at import time:

- Frozen dataclasses (pure stdlib, loaded immediately): PromptCase, ModelSample,
  ValidationIssue, SchemaValidationResult, DriftThresholds, SampleDriftResult,
  and DriftBenchmarkReport.
- Pydantic models (defined in `_legacy_models.py` and lazy-loaded via PEP 562
  `__getattr__`): TestPrompt, DriftMetrics, and BenchmarkResult.

Package exports in `__init__.py`, Rich console rendering in `reporter.py`,
Pydantic report deserializers in `report_io.py`, and CLI subcommand handlers in
`cli.py` are all lazily loaded so canonical evaluation and CLI startup avoid
importing `pydantic` or `rich` unless actually needed.

Frozen dataclasses prevent field reassignment but do not make nested
dictionaries or lists immutable.

The two report families contain different metrics and are not losslessly
interchangeable.

## Input loaders

`engine.py` loads PromptCase and ModelSample records from JSON or JSONL.

PromptCase wrappers accept `prompts`, `cases`, `suite`, or `items`.
ModelSample wrappers accept `samples`, `outputs`, or `results`.

`evaluator.py` separately loads TestPrompt records. Its wrappers accept
`prompts`, `cases`, `suite`, or `items`.

All current file readers enforce UTF-8 with BOM (`utf-8-sig`) support and a
16 MiB file size cap. Files are read into memory.

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

Both `metrics.py` and `validator.py` quote object keys containing dots,
brackets, or special characters (for example `$["a.b"]`) to avoid colliding
with nested field paths (`$.a.b`).

`evaluator.py` additionally implements SequenceMatcher token difference,
a lexical and structural similarity blend, and throughput calculations.
Its similarity blend is not embedding-based semantic similarity.

## Engine evaluation

DriftBenchmarkEngine compares baseline and candidate records by prompt ID.
It validates responses, calculates pair metrics, aggregates results, and
applies thresholds.

Its report contains mean and maximum drift, conformity rates, latency
percentiles, individual drift flags, and aggregate failure reasons.

The report fails if any sample fails its individual drift checks or any
aggregate threshold fails. Thresholds reject non-finite and out-of-range
values. `run` exits 1 for a failed report by default, 0 for a passing report,
and 2 for usage or input errors. `--no-fail-on-drift` suppresses only exit
code 1; it never changes the report's pass/fail status.

Loaders and evaluators reject duplicate and blank prompt IDs. Baseline and
candidate must have identical ID sets; no extra records are discarded.
An explicit suite must be nonempty and cover exactly those IDs. The
validate command enforces the same suite-coverage requirement. Direct pair
evaluation also checks that both samples and the optional prompt share an ID.
Comparison results use sorted prompt-ID order for reproducibility.

## Alternate evaluator

`evaluate_suite` produces BenchmarkResult using a different scoring formula.

An empty prompt sequence in this library API requests inference from the
matching output sets. A nonempty prompt sequence must match them exactly.
Missing or extra outputs are rejected before any prompt inference.

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

- `run` (with `--config`, `--category`, `--long-text-mode`, `--include-diffs`, `--format pr-comment|csv|sarif`, `--junit-output`, `--sarif-output`, `--github-summary`)
- `audit` (unified offline audit combining per-prompt drift, field distributions with optional `--group-by`/`--fail-on-widespread` blast-radius checks, optional downstream consumer validation via decision logs or `--validator-module`, optional offline embeddings via `--min-embedding-similarity`, optional multi-turn conversations via `--baseline-conversations`/`--candidate-conversations`, optional agent tool-trace drift via `--baseline-trace`/`--candidate-trace`, quality checks, logprobs, and token volume in `text`, `json`, `markdown`, offline `html`, or `pr-comment`)
- `diff` (policy-aware JSON field-level and unified text diff inspector in `text`, `json`, or GFM `markdown`)
- `distribution` (cross-prompt JSON field distribution drift using categorical TVD/JSD and numeric quantiles + exact 1D Wasserstein, with optional `--group-by category|metadata:<key>` segmentation, `localized` vs `widespread` blast-radius classification, and `--fail-on-widespread`)
- `rolling-distribution` (unpaired rolling-baseline cross-window JSON field distribution analyzer pooling prior traffic windows `-w`, measuring consecutive-window baseline jitter, classifying segmented blast radius, and exporting `text`, `json`, `markdown`, or offline `html`)
- `logprobs` (optional `top_logprobs` token entropy and position-aligned top-k JSD analyzer)
- `check-embeddings` (opt-in semantic embedding cosine similarity drift analyzer working offline on `metadata.embedding` or via any OpenAI-compatible `/v1/embeddings` endpoint, exporting `text`, `json`, `markdown`, offline `html`, or `csv`)
- `check-conversations` (multi-turn assistant conversation trajectory drift analyzer detecting per-turn divergence, earliest divergence turn, escalation slope, turn-count mismatches, JSON adherence drops, refusal flips, optional `--show-diffs`, and `--diagnose` in `text`, `json`, `markdown`, offline `html` with inline SVG sparklines, `sarif` 2.1.0, `junit` XML, `csv`, or `pr-comment`)
- `check-quality` (deterministic repetition-loop, refusal-flip, length-truncation, and length-blowup regression checks)
- `check-formats` (standalone semantic format validator for `date`, `date-time`, `email`, `uri`, `uuid`, `ipv4`, and `iso-currency` on exact or `[*]` wildcard JSONPaths)
- `check-consumer` (downstream consumer validation rejection-rate comparator from pre-recorded decision logs or `--validator-module pkg.mod:fn`, exporting `text`, `json`, `markdown`, or offline `html`)
- `wizard` (guided plain-English interactive setup wizard and use-case preset generator for `drift.toml`)
- `score` (pluggable scorer CLI supporting `--list-scorers` and repeatable `--scorer`)
- `validate` (supports `--category` and `--strict`)
- `report` (supports `markdown`, `console`/`table`, `json`, `csv`, `sarif`, `html`, `junit`, `pr-comment`)
- `diagnose`
- `compare-many` (supports `--sort-by`, `--format rich`, `--html-output`, `--markdown-output`, `--json-output`, `--config`)
- `compare-reports` (supports `--sort-by`, `--format rich`, `--html-output`, `--markdown-output`, `--json-output`)
- `matrix` (Prompt x Candidate drift heatmap matrix in `text`, `json`, or offline `html`)
- `gate` (composite GO/NO-GO release gate combining drift, persistent history trends, statistical tests, tool checks, agent tool-trace drift, segmented distribution blast radius, offline embedding similarity, multi-turn conversation trajectories, and downstream consumer rejection logs)
- `cost` (explicit generated token volume, optional USD output cost drift comparator, and `--max-token-increase-pct` / `--max-cost-increase-pct` budgets)
- `badge` (offline shields-style static SVG status badge generator supporting saved canonical engine, unified audit, and multi-turn conversation JSON reports)
- `import-captures` (offline importers for OpenAI Batch API JSONL, Anthropic Message Batches JSONL, and CSV captures)
- `import-conversations` (offline importers converting OpenAI chat, Anthropic messages, and ShareGPT JSONL transcripts into canonical `ConversationRecord` JSONL)
- `snapshot` (`create` and `verify` offline SHA-256 baseline fixture integrity manifests, with optional `--conversations` and `--trace` pinning)
- `check-tools` (supports `--adapter normalized|openai|anthropic|gemini|otel`, `--max-failed-calls`, `--max-tool-calls`, and `--allow-resource`)
- `diff-tools` (offline baseline vs candidate agent tool-trace drift comparator across normalized, OpenAI, Anthropic, Gemini, and OTel transcripts in `text`, `json`, `markdown`, offline `html`, `sarif` 2.1.0, or `junit` XML)
- `stats` (bootstrap confidence intervals, multi-run flakiness, paired permutation with Cohen's `d`, and overflow-safe exact two-sided McNemar tests)
- `history` (`record`, `show` with `--candidate`/`--tag` filters, sparklines & SVG tooltips, `check` moving-window gate, `summary`, and atomic `prune`)
- `capture` (zero-dependency OpenAI-compatible, Anthropic `--provider anthropic`, and Gemini `--provider gemini` endpoint collector with `--concurrency`, `--runs`, `--json-mode`, `--resume`, `--header`, `Retry-After` backoff, and connection-drop retries)
- `capture-conversations` (sequential multi-turn live conversation collector replaying user/system prompt templates across OpenAI-compatible, Anthropic, and Gemini endpoints into canonical `ConversationRecord` JSONL)
- `redact` (deterministic credential and optional PII redactor for capture files and multi-turn `--conversations` transcripts)
- `subsample` (deterministic stratified prompt suite subsampling and lockstep capture slicing)
- `infer-schema` (automatic supported-subset JSON Schema suite inference from captured baseline JSON outputs)
- `lint-suite` (prompt suite quality and JSON Schema hygiene auditor)
- `calibrate` (automatic `drift.toml` threshold calibration, volatile JSON field detection with `[*]` wildcard collapse, and `--numeric-tolerances`)
- `init-sample` (optional `--with-policy` to scaffold `drift.toml`, or `--pack extraction|classification|coding|rag|agent` for curated starter suites)
- `doctor` (pre-flight validation of environment, report/history schema versions, policies, suite schemas/regex safety, blank capture outputs, consumer decision logs, multi-turn conversation transcripts, agent tool traces with credential scanning, and ID alignment)
- `ai-devices`

Additional library modules:

- `policy.py`: `drift.toml` / `drift.json` evaluation policies with custom metric weights and exact or `[*]` wildcard JSONPath `ignore` / `unordered_array` / `tolerance_abs` / `tolerance_rel` rules.
- `diffing.py`: policy-aware semantic JSON field diffing, summary counts, and unified text diff formatting.
- `csv_export.py`: RFC-4180 per-prompt CSV exporter with spreadsheet formula-injection sanitization for canonical engine reports.
- `sarif.py`: OASIS SARIF 2.1.0 exporter (`export_sarif_report`, `export_conversation_sarif_report`) with deterministic `partialFingerprints` and invocation metadata for GitHub Code Scanning and IDE diagnostics.
- `junit.py`: JUnit XML exporter (`export_junit_report`, `export_conversation_junit_report`) for canonical engine and multi-turn conversation reports without raw response-body leakage.
- `diagnostics.py`: deterministic root-cause diagnostics for canonical engine (`diagnose_report`) and multi-turn conversation (`diagnose_conversation_report`) reports.
- `doctor.py`: pre-flight environment, policy, suite hygiene linting, capture credential scanning, and ID-alignment validator.
- `stats.py`: deterministic bootstrap CIs, Wilson score binomial intervals, Holm-Bonferroni multiple-testing correction, multi-run stability, paired permutation significance tests (with Cohen's `d`), and overflow-safe exact two-sided McNemar conformity tests.
- `history.py`: append-only JSONL drift ledger with strict validation, moving-window regression detection, per-candidate summaries, atomic pruning, sparklines, and SVG HTML trend charts with tooltips.
- `capture.py`: OpenAI-compatible, Anthropic, and Google Gemini HTTP capture collector with bounded reads, sanitized errors, `Retry-After` support, `--concurrency`, `--runs`, `--json-mode`, `--resume` checkpointing, custom `--header` support, timeouts, and retries.
- `adapters.py`: offline transcript adapters for OpenAI Chat & Responses APIs (`output` wrapper), Anthropic/Claude `tool_use`/`tool_result` logs, Google Gemini `functionCall`/`functionResponse` contents, and OpenTelemetry GenAI / OTLP `resourceSpans` JSON (with `startTimeUnixNano` ordering).
- `scorers.py`: pluggable scorer protocol & registry (`char_ngram_cosine`, `json_key_jaccard`, `markdown_structure`, `numeric_prose`, `python_ast`, `tfidf_cosine`, plus `llm_drift_bench.scorers` entry-point discovery).
- `schema_infer.py`: automatic JSON Schema inference & merging (including nullable object/array constraint preservation and deep-copied single schemas).
- `calibrate.py`: baseline noise threshold calibration, volatile field detection with `[*]` wildcard collapse, numeric tolerance suggestion, and `drift.toml` synthesis.
- `redaction.py`: deterministic credential and optional PII redaction for text and `ModelSample` captures.
- `sampling.py`: deterministic stratified prompt suite subsampler, suite filter (`filter_prompt_suite`), and lockstep capture slicer.
- `cost.py`: explicit token volume and USD output-cost drift estimator (returns `complete=False` if any sample omits `tokens_generated`).
- `matrix.py`: multi-candidate Prompt x Candidate drift heatmap matrix (text, JSON, offline HTML).
- `prompt_lint.py`: prompt suite and JSON Schema hygiene linter.
- `gates.py`: composite GO/NO-GO release gate evaluator aggregating canonical drift, history trend, statistical, and tool-trace evidence.
- `snapshot.py`: offline SHA-256 baseline fixture integrity manifests and tamper verification with path-traversal protection.
- `batch_import.py`: offline importers converting OpenAI Batch API JSONL, Anthropic Message Batches JSONL, and CSV exports into validated `ModelSample` records.
- `badge.py`: offline script-free shields-style static SVG drift status badge generator.
- `distribution.py`: cross-prompt JSON field distribution drift analyzer (categorical TVD/JSD, numeric quantiles & exact normalized 1D Wasserstein, plus text/Markdown/offline HTML exporters).
- `wizard.py`: guided 5-question plain-English `drift.toml` wizard and use-case preset builder (`extraction`, `classification`, `coding`, `rag`, `strict`).
- `logprobs.py`: optional `top_logprobs` Shannon entropy and position-aligned top-k base-2 JSD drift analyzer with honest missing-data reporting (`complete=False`).
- `quality_checks.py`: deterministic regression checks for consecutive line/n-gram repetition loops, refusal flips, length truncation, and verbosity blowups.
- `formats.py`: standalone semantic JSON field format validator (`date`, `date-time`, `email`, `uri`, `uuid`, `ipv4`, `iso-currency`) on exact or `[*]` wildcard paths.
- `packs.py`: curated starter benchmark packs (`extraction`, `classification`, `coding`, `rag`, `agent`) with lint-clean prompts, baseline/passing/drifted captures, and policies.
- `audit.py`: unified offline audit report and script-free HTML dashboard aggregating drift, distributions, quality checks, logprobs, and token volume.
- `consumer.py`: downstream consumer validation and rejection-rate creep checker supporting pre-recorded decision logs, Python validator callables, and offline HTML exports.
- `rolling.py`: rolling-baseline cross-window distribution drift analyzer supporting unpaired traffic windows, consecutive-window baseline jitter measurement, segmented blast-radius classification, and offline HTML dashboards.
- `embeddings.py`: opt-in zero-dependency semantic embedding cosine similarity drift analyzer supporting offline precomputed vectors (`metadata.embedding`), OpenAI-compatible `/v1/embeddings` endpoints, and standalone offline HTML reports without altering lexical scores.
- `conversation.py`: multi-turn conversation trajectory drift analyzer comparing stateful assistant turns for earliest divergence, snowballing escalation slope, turn-count mismatches, JSON adherence drops, and refusal flips, with text, Markdown, and offline HTML exports.
- `_comparison_md.py` and `_cli_extras.py`: GitHub-Flavored Markdown multi-candidate comparison exporter and lazily imported extra CLI subcommand handlers.

All CLI run formats render one DriftBenchmarkReport. Rich, Markdown, and HTML
use the canonical summary/detail values without calling evaluate_suite. The
engine report stores the active thresholds; old saved reports without them keep
thresholds unavailable. JSON rounding does not trigger pass/fail recomputation.

report_io.load_report validates complete engine and legacy evaluator reports.
It preserves the report model instead of manufacturing semantic or exact-match
values. The compatibility name load_benchmark_result delegates to this loader.
Incomplete and unrecognized reports fail. The alternate evaluator library API
retains its historical formula; its reports are explicitly separate from engine
reports. HTML requires engine reports and includes no scripts or network assets.

## Entry points and packaging

The console entry point calls llm_drift_bench.cli.main.
The module entry point supports python -m llm_drift_bench.
The package includes a py.typed marker.

Hatch reads the version from __init__.py, which is the single version source.

## Testing limitations

Coverage is not evidence that all benchmark semantics are correct.

Release checks must include unsupported schemas, malformed inputs,
duplicate and mismatched IDs, missing measurements, format consistency,
module execution, and installation of built distributions outside the
source checkout.
