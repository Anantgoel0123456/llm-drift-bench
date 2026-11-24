# Changelog

## [Unreleased]

### Added

- Cross-prompt JSON field distribution drift analyzer (`distribution.py`, `CategoricalFieldDrift`, `NumericFieldDrift`, `DistributionReport`, `analyze_field_distributions`, `format_distribution_text`, `format_distribution_markdown`, `export_distribution_html`, and `llm-drift-bench distribution`) catching soft slides across valid JSON outputs using categorical TVD / base-2 JSD and numeric quantiles (p10/p50/p90) + exact normalized 1D Wasserstein distance with optional `--max-tvd`, `--max-jsd`, and `--max-wasserstein` gates.
- Guided plain-English setup wizard and use-case presets (`wizard.py`, `WizardAnswers`, `PRESET_ANSWERS`, `build_policy_from_answers`, `render_wizard_config`, `run_guided_wizard`, `format_wizard_next_steps`, and `llm-drift-bench wizard [--preset extraction|classification|coding|rag|strict] [--non-interactive] [-o drift.toml]`) generating round-trippable `drift.toml` policies without eval jargon.
- Optional OpenAI-compatible `top_logprobs` capture (`CaptureConfig.top_logprobs`, `capture --top-logprobs K`) and token uncertainty / top-k distribution drift analyzer (`logprobs.py`, `PromptLogprobDrift`, `LogprobDriftSummary`, `analyze_logprobs_drift`, `format_logprobs_text`, and `llm-drift-bench logprobs`) computing Shannon entropy in bits and position-aligned top-k base-2 JSD even before greedy decoding flips tokens while reporting incomplete coverage honestly (`complete=False`).
- Deterministic quantization & model-upgrade quality regression checks (`quality_checks.py`, `QualityFinding`, `QualityReport`, `detect_repetition_loop`, `is_likely_refusal`, `run_quality_checks`, `format_quality_report_text`, and `llm-drift-bench check-quality`) detecting consecutive line/n-gram repetition loops, refusal flips, length truncation, and verbosity blowups.
- Offline batch and table capture importers (`batch_import.py`: `import_openai_batch_jsonl`, `import_anthropic_batch_jsonl`, `import_csv_captures`, and `llm-drift-bench import-captures --from openai-batch|anthropic-batch|csv`) converting OpenAI Chat/Responses batch JSONL, Anthropic Message Batches JSONL, and CSV files into canonical `ModelSample` JSONL while rejecting failed items and duplicate IDs.
- Offline shields-style static SVG drift status badge generator (`badge.py`, `export_drift_badge_svg`, and `llm-drift-bench badge -i REPORT [-o badge.svg] [--label ...]`).
- Policy composition and round-trippable serialization (`merge_policies`, `render_policy_toml` in `policy.py`).
- GitHub-Flavored Markdown semantic diff exporter (`format_diffs_markdown` and `llm-drift-bench diff --format markdown`).
- Multi-candidate cost ranking and budget gate checks (`rank_candidate_costs`, `check_cost_budget`, and `--max-token-increase-pct` / `--max-cost-increase-pct` on `llm-drift-bench cost`).
- Tool-call step budget (`check_max_tool_calls`, `--max-tool-calls`), resource prefix boundary auditing (`check_allowed_resources`, `--allow-resource`), and JWT + database URL secret detection in `tool_events.py` and `redaction.py` (plus ReDoS-safe `custom_patterns` redaction rules).
- Multi-category breakdown mini-table inside GitHub PR comments (`canonical_pr_comment`) whenever a benchmark spans 2+ prompt categories.
- Offline baseline snapshot integrity manifests (`snapshot.py`, `SnapshotArtifact`, `SnapshotManifest`, `SnapshotVerification`, `create_snapshot_manifest`, `verify_snapshot_manifest`, and `llm-drift-bench snapshot create|verify`) with SHA-256 artifact digests, prompt-ID hashing, and strict path-traversal rejection.
- Composite release gate CLI (`llm-drift-bench gate`) combining canonical drift reports, history ledgers, paired statistical tests, and saved tool checks into GO/NO-GO exit codes, plus token/USD output cost CLI (`llm-drift-bench cost`).
- Native Google Gemini support across both live capture (`capture --provider gemini` using `/v1beta/models/{model}:generateContent` and `x-goog-api-key`) and offline tool-trace auditing (`from_gemini_transcript` and `check-tools --adapter gemini`).
- Pure-stdlib `numeric_prose` built-in scorer (`numeric_prose_similarity`) catching silent numeric hallucinations in free-form RAG/financial prose while ignoring alphanumeric identifiers.
- Step-down Holm-Bonferroni multiple-testing correction (`holm_bonferroni_adjust`), deterministic Wilson score binomial confidence intervals (`wilson_score_interval`), and optional `require_same_candidate` enforcement in `summarize_multi_run`.
- Deterministic SHA-256 `partialFingerprints` (`primaryLocationLineHash`) and `invocations` metadata in SARIF 2.1.0 exports for deduplicated GitHub Code Scanning alerts.
- Integrated prompt-suite hygiene linting (`suite_hygiene`) and capture credential leak scanning (`secrets_scan`) in `doctor.py`.
- Per-candidate drift history summaries (`summarize_history`, `history summary`) and atomic ledger compaction (`prune_history`, `history prune --keep-last N`).
- Prompt suite filtering helper (`filter_prompt_suite`) by category, schema presence, and excluded prompt IDs.
- Deterministic PII and credential redactor (`redaction.py`, `redact_text`, `redact_samples`, `RedactionReport`, and `llm-drift-bench redact [--secrets-only]`) supporting PEM private keys, Anthropic/OpenAI/GitHub/AWS/Google/Slack/Bearer credentials, and optional PII (email, SSN, IPv4, Luhn-checked credit cards) while preserving valid JSON.
- Deterministic stratified prompt suite subsampler and lockstep capture slicer (`sampling.py`, `sample_prompt_suite`, `slice_samples_for_suite`, and `llm-drift-bench subsample --size N [--seed] [--slice-capture IN OUT]`).
- Explicit token and output-cost drift estimator (`cost.py`, `ModelPricing`, `CostComparison`, `estimate_cost_drift`) that never fabricates token counts when `tokens_generated` is missing.
- Multi-candidate Prompt x Candidate drift heatmap matrix (`matrix.py`, `DriftMatrix`, `MatrixRow`, `build_drift_matrix`, `format_drift_matrix_text`, `export_drift_matrix_html`, and `llm-drift-bench matrix`).
- Prompt suite quality and JSON Schema hygiene linter (`prompt_lint.py`, `LintIssue`, `lint_prompt_suite`, `format_lint_issues_text`, and `llm-drift-bench lint-suite [--strict]`) detecting duplicate IDs/prompts, unsatisfiable `required` + `additionalProperties: false` rules, permissive/unconstrained objects, and unsupported schema keywords.
- Composite GO/NO-GO release gate evaluator (`gates.py`, `GateCheck`, `ReleaseVerdict`, `evaluate_release_gates`) combining canonical drift, moving-window history trends, statistical significance tests, and tool-trace evidence without mutating canonical reports.
- GitHub-Flavored Markdown multi-candidate comparison leaderboard exporter (`export_comparison_markdown` and `--markdown-output FILE` on `compare-reports` and `compare-many`).
- Native Anthropic Messages API provider (`provider="anthropic"` and `capture --provider anthropic`) with `/v1/messages` routing, `x-api-key`, `anthropic-version`, `Retry-After` backoff honoring (capped at 30 s), and connection-reset retry handling.
- Automatic numeric tolerance rule suggestions (`detect_numeric_tolerances`, `CalibrationResult.numeric_tolerances`, and `calibrate --numeric-tolerances`) emitting `tolerance_abs` rules in `drift.toml` for jittering numeric fields.
- Multi-candidate regression assertion helper (`assert_no_regression(reports, against=...)`) alongside explicit `DriftBenchmarkReport` type validation in `assert_no_drift`.
- Sanitized `<properties>` model metadata (`baseline_model`, `candidate_model`, `mean_composite_drift`, `candidate_json_conformity_rate`) embedded in JUnit XML `<testsuite>` exports.
- Per-prompt semantic diff summary counts (`PromptDiff.summary_counts()` and `"summary"` in `PromptDiff.to_dict()`).

### Fixed

- Enforced `FieldRule` dictionary-key / `rule.path` parity in `DriftPolicy.__post_init__` and hardened the Python 3.10 fallback TOML parser in `policy.py` to reject empty keys and embedded single quotes in single-quoted strings.
- Added multipart OpenAI `message.content` block-list support in both `capture.py` and `batch_import.py`, rejected duplicate `prompt_id` entries in `capture_suite` (`prompts` and `existing_samples`), and included CSV row numbers on `ModelSample` validation errors in `import_csv_captures`.
- Made Python AST traversal in `scorers.py` iterative (preventing `RecursionError` on deeply nested expressions), caught `ValueError`/`RecursionError` cleanly, hardened Markdown fence parsing for nested backtick and tilde fences, and rejected duplicate `prompt_id` entries in `score_captures(..., prompt_cases=...)`.
- Enforced strict boolean validation in `mcnemar_exact_test`, rejected boolean seeds in `stats.py` and `sampling.py`, validated `record_report` input types and finite `sparkline` inputs in `history.py`, and stably sorted OpenTelemetry GenAI spans by `startTimeUnixNano` when timestamps are present.
- Replaced recursive `flatten_json_paths` with an iterative stack to prevent `RecursionError` on deeply nested JSON (1,500+ levels), widened `compute_percentile` to accept `Sequence[float]`, and hardened numeric type checks in `compute_composite_drift_score`.
- Fixed the Python 3.10 fallback TOML parser in `policy.py` to support single-quoted section headers/keys and `#` inside single-quoted strings.
- Rejected boolean `version` values (`True == 1`) in `tool_log.py` and added helpful `--adapter openai|anthropic|otel` hints when vendor transcript wrapper keys are detected.
- Added default spreadsheet formula injection protection (`sanitize_formulas=True`) to string columns in `csv_export.py` while preserving negative numeric columns like `latency_delta_ms`.
- Fixed `diffing.py` reporting spurious `$` root removals when comparing empty containers (`{}` / `[]`) against populated containers.
- Deep-copied single-element schemas in `merge_schemas` and consistently enforced `strict_additional=True` recursively.
- Rejected boolean limit arguments (`True`/`False`) in `latency_increases`, `numeric_changes`, and `_hotspots` breakdowns.
- Prevented output-file collisions when `--json-output`, `--html-output`, and `--markdown-output` point to the same path.

## [0.3.0] - 2026-11-23

### Added

- OASIS SARIF 2.1.0 report exporter (`sarif.py`, `--format sarif`, and `run --sarif-output FILE`) for GitHub Code Scanning and IDE diagnostic viewers.
- Pre-flight workspace validator (`doctor.py` and `llm-drift-bench doctor`) checking Python runtime, optional packages, `drift.toml` policies, prompt suite schema/regex safety, and capture `prompt_id` alignment.
- Repeatable `--category` filtering on `run` and `validate` commands.
- `json_key_jaccard` built-in scorer (`json_key_jaccard_similarity`), `markdown_structure` scorer (`markdown_structure_similarity`), and `llm-drift-bench score` CLI (`--list-scorers`, repeatable `--scorer`).
- Expanded secret-leak detection in `tool_events.py` (distinct `anthropic_key`, `openai_key`, `google_api_key`, `slack_token`, `bearer_token`, `aws_access_key`, `github_pat`, `private_key_pem`) and `--max-failed-calls` retry budget gating on `check-tools`.
- Paired Cohen's `d` effect size (`effect_size_d`) on `paired_permutation_test` and `PairedSignificance`.
- Targeted `diagnose` suggestions for `minimum`, `maximum`, `minLength`, `maxLength`, `pattern`, `pattern_limit`, `minItems`, `maxItems`, `const`, and `unsupported_keyword` schema rules.
- `[*]` array wildcard JSONPath matching in `drift.toml` field rules (exact-path precedence over wildcards for `ignore`, `unordered_array`, `tolerance_abs`, and `tolerance_rel`), plus automatic collapsing of repeated volatile array indices in `calibrate`.
- Semantic JSON field-level and unified text diff engine (`diffing.py` and `llm-drift-bench diff`) respecting `drift.toml` policies (`--only-changed`, `--prompt-id`, `--limit`, `--format text|json`).
- RFC-4180 CSV exporter (`csv_export.py` and `--format csv` on `run` and `report`).
- Crash-resilient incremental `--resume` checkpointing and custom `--header` support (`parse_header_spec`) on `capture`.
- Moving-window drift trend regression gate (`detect_history_regression`, `HistoryTrendCheck`, and `llm-drift-bench history check`), plus Y-axis gridlines and accessible `<title>` tooltips in history SVG trend charts.
- `init-sample --with-policy` option to scaffold a starter `drift.toml` policy file.
- Automatic JSON Schema inference (`schema_infer.py` and `llm-drift-bench infer-schema`) generating supported-subset `prompts.json` suites directly from captured baseline JSON outputs (`--strict-additional`, `--shared-schema`, multi-run merging, and nullable object/array constraint preservation).
- Automatic threshold calibration & volatile field detection (`calibrate.py` and `llm-drift-bench calibrate`) generating ready-to-use `drift.toml` policies with configurable safety headroom from repeated baseline runs or saved reports.
- Pluggable scorer registry (`scorers.py`: `Scorer`, `ScorerResult`, `register_scorer`, `get_scorer`, `list_scorers` with `llm_drift_bench.scorers` entry-point discovery) plus pure-stdlib `char_ngram_cosine_similarity`, `tfidf_cosine_similarity`, and `python_ast_similarity` scorers.
- Exact two-sided McNemar test (`mcnemar_exact_test`, `McNemarResult`) wired into `stats --compare-against`, plus baseline-model mismatch protection.
- Order-insensitive array rules (`FieldRule.unordered_array = true`) in `drift.toml` policies, overflow-safe absolute tolerance math, and single-quoted TOML string support in the Python 3.10 fallback parser.
- Capture collector upgrades (`--concurrency`, `--runs`, `--json-mode`), bounded HTTP response reads, and sanitized HTTP error reporting.
- OpenAI Responses API (`function_call` / `function_call_output`, `output` wrapper) and standard OTLP `resourceSpans` JSON support in `adapters.py`, `load_adapted_tool_events`, and `check-tools --adapter normalized|openai|anthropic|otel`.
- `history show --candidate` and `--tag` filters, plus SVG Pareto frontier polyline, axis grid/ticks, and label collision avoidance in comparison HTML dashboards.
- `drift.toml` / `drift.json` evaluation policies (`load_policy`, `DriftPolicy`, `MetricWeights`, `FieldRule`, and `run`/`compare-many --config`) with custom metric weights, per-JSONPath `ignore` / `tolerance_abs` / `tolerance_rel` rules, and `--long-text-mode skip_levenshtein` for >16,384-character outputs.
- Multi-run statistical analysis (`stats.py` and `llm-drift-bench stats`): deterministic bootstrap confidence intervals (`bootstrap_mean_ci`), paired sign-flip permutation significance tests (`paired_permutation_test`), and flaky-prompt detection (`summarize_multi_run`).
- Visual reporting upgrades: GitHub PR comment formatter (`--format pr-comment` with `--previous` deltas), opt-in pure-CSS side-by-side HTML JSON field diffs (`run --format html --include-diffs`), and multi-candidate HTML dashboards with inline SVG Pareto charts (`compare-reports`/`compare-many --html-output`).
- Append-only drift history ledger (`history.py` and `llm-drift-bench history record|show`) with terminal Unicode sparklines and standalone SVG trend HTML dashboards.
- Zero-dependency OpenAI-compatible capture collector (`capture.py` and `llm-drift-bench capture`) supporting Ollama, vLLM, llama.cpp server, and OpenAI endpoints with bounded retries, timeouts, provenance metadata, and env-var secret handling.
- Offline agent transcript adapters (`adapters.py`: `from_openai_tool_calls`, `from_anthropic_transcript`, `from_otel_genai_spans`) plus `check_max_tool_retries` and `check_no_secret_in_tool_args` workflow checks.
- `--sort-by` (`drift`, `passed`, `conformity`, `latency`) and `--format rich` on `compare-reports` and `compare-many`, plus logical workflow subcommand ordering in `--help`.
- `run --github-summary`, `run --junit-output FILE`, and JUnit XML export (`run/report --format junit`).
- `compare-many` / `evaluate_many()`, `compare-reports` / `compare_reports()`, `prompt_changes()`, `--by-category`, `--rule-summary`, `--slowdowns`, `--numeric-changes`, `--hotspots`, `--inspect-prompt`, `--against`, `--fail-on-regression`, and `diagnose` (with optional local OpenVINO `ai` extra and `ai-devices`).

### Fixed

- Prevented integer-to-float `OverflowError` in `mcnemar_exact_test` when discordant prompt counts exceed 1,000 by using log-space binomial tail evaluation (`math.lgamma`).
- Fixed `adapters.py` to support OpenAI Responses `output` wrappers, `failed`/`incomplete` statuses, and standard OTLP `resourceSpans` / `[{key, value}]` attributes.
- HTML response diffs (`--include-diffs`) now apply active `drift.toml` policy rules (`ignore`, `unordered_array`, numeric tolerances) and preserve readable Unicode (`ensure_ascii=False`).
- PEP 562 lazy exports in `__init__.py` and `models.py` (moving legacy Pydantic models to `_legacy_models.py`), lazy Rich imports in `reporter.py`, cached `TypeAdapter` in `report_io.py`, and lazy CLI subcommand handler imports — cutting `import llm_drift_bench` latency from ~411 ms to ~8 ms and keeping canonical CLI runs completely free of eager `pydantic` and `rich` imports.
- `export_comparison_html` now respects the `sort_by` ranking order passed from `compare-reports` and `compare-many`.
- `HistoryEntry.from_dict` now strictly validates boolean/numeric types and unit ranges (fixing `bool("false") -> True`), and `export_history_html` renders a clean placeholder SVG when history is empty.
- `_write_comparison_json` (and all CLI file writers) now create missing parent directories automatically and write atomically via temporary files + `os.replace`.
- `extract_json_payload` now scans opening `{` and `[` in document order (fixing prose-wrapped JSON arrays of objects like `[{"a": 1}]`), iterates all markdown fences prioritizing ```` ```json ```` blocks, and handles unrelated prose braces cleanly while keeping strict duplicate-key and non-finite rejection.
- `_check_json_type` now rejects `NaN` and `±Infinity` for JSON Schema `"type": "number"`, and `validator.py` quotes dotted/special object keys consistently with `metrics.py`.
- `_patterns.py` now properly handles leading `]` inside regex character classes (`[^]...]` and `[]...]`).
- `diagnose_report` now reports per-prompt conformity failures even when baseline also failed, and reports co-occurring `numeric_change` alongside `schema` issues.
- `DriftBenchmarkEngine.evaluate` now rejects mixed `model_name` values within a single capture, matching `evaluate_many`.
- Dataset parsing errors identify the source file, line/column, and record index without echoing sensitive values; `check-tools` refuses to overwrite its input trace; suite files accept the `suite` wrapper; Opik exports copy nested metadata safely.

## [0.2.0] - 2026-11-15

### Changed

- CI and publishing use commit-pinned Node.js 24 actions for checkout, Python
  setup and artifact transfers, replacing deprecated Node.js 20 runtimes.

- Captured outputs must be strings. `ModelSample` and `ModelSample.from_dict` now
  reject objects, arrays, numbers, booleans and null with an explicit error
  instead of converting them to Python text such as `{'ok': True}` or `None`.
  Serialize structured values with `json.dumps` before capture.
- The `opik` extra is now `opik>=1.10.24,<3`, the range tested in CI. Earlier SDK
  releases do not accept the `category_name` argument the adapter passes.
- Reports distinguish schema conformity from JSON syntax, display candidate
  validation issues, and explain numeric drift normalization and sample coverage.
- `run` fails by default on drift or threshold failure. An individual failing
  sample fails the suite. `--no-fail-on-drift` enables informational runs.
- Character-level Levenshtein now uses the same unit at every supported length.
- Missing timings and token counts remain unavailable instead of being invented.
- Package metadata and CLI version use one version source.

### Fixed

- CI's newest-SDK job respects the supported `opik>=1.10.24,<3` range.
- Length and item-count schema limits accept and enforce integral floats such
  as `3.0`, while rejecting booleans, fractions and invalid numeric values.
- Saved engine reports validate every per-prompt metric (range 0 to 1), JSON
  comparison scores against validation results, validation fields against their
  issues, the `drifted` flag against recorded thresholds, every summary mean and
  maximum against the per-prompt results, conformity rates against the validation
  outcomes, latency summaries against per-prompt deltas, and a passing verdict
  against the thresholds it records. A report claiming `passed: true` with a
  per-prompt drift score of 99 is rejected. Legacy reports are checked the same way.
- Captured output parsing rejects duplicate object keys in every extraction mode
  (direct, fenced and prose-wrapped, including nested objects), as dataset files
  already did. `{"total":100,"total":1}` no longer silently becomes `{"total":1}`.
- Schema type `integer` accepts finite numbers without a fractional part such as
  `1.0` and `1e2`, as JSON Schema defines, while still rejecting booleans.
- Dataset, schema and report loading share bounded UTF-8/BOM handling and reject
  duplicate keys and nonfinite numbers. JSONL errors identify their line.
- Saved engine reports reject contradictory pass/fail status and out-of-range
  aggregate similarity scores.
- Schema patterns now enforce a bounded, conservative regex subset before
  matching; unsafe expressions and oversized inputs fail explicitly.
- Malformed supported schema keywords now produce explicit validation issues,
  including in absent properties. Enum and const distinguish booleans from numbers.
- Structural metric paths quote special object keys to prevent collisions with
  nested fields and array indices.
- Output extraction rejects nonstandard numeric constants and numbers outside
  the finite floating-point range, including fenced and prose-wrapped payloads.
- Numeric drift rejects invalid numbers instead of returning a plausible score,
  and handles opposite-sign finite extremes without subtraction overflow.
- All CLI run formats now use the canonical engine result without reevaluation.
- Saved engine reports retain scores and thresholds without invented metrics.
- Report loading rejects incomplete formats; see migration notes for the updated
  load_benchmark_result return types.

- Duplicate and mismatched prompt IDs cannot silently drop records.
- Missing files cannot generate synthetic evaluation results.
- Unsupported schema keywords produce explicit validation issues.
- Nonfinite and out-of-range thresholds cannot disable checks silently.
- Empty validation datasets now fail with an input error.
- Empty per-prompt schemas correctly override a global schema.
- Ruff import ordering and strict typing issues in reviewed code.

### Added

- `llm-drift-bench check-tools` runs the offline read-before-edit check on a
  normalized trace (`.jsonl` or `.json`) with distinct exit codes: 0 PASS, 1 FAIL,
  3 UNKNOWN, 2 error. `-o` saves a validated verdict file.
- Strict trace import (`load_tool_events`, `parse_tool_events`) that rejects unknown
  fields, duplicate keys, nonfinite and non-integer offsets, naming the bad line.
- `ToolCheckRecord`, `check_trace`, `check_trace_file` and `load_tool_check` keep the
  question beside each verdict and reject saved verdicts their own evidence
  contradicts.
- `--tool-check` on `run` and `report`, and a `tool_checks` argument on
  `export_html_report` and `export_markdown_report`, add a separate Tool evidence
  section that never affects the drift verdict.

- Post-upload registry verification jobs download the exact published wheel and
  run clean-environment installation checks, separately from upload permissions.
- Optional `opik` extra and explicit `EvidenceScore.to_opik()` adapter that
  constructs the installed SDK result and rejects unavailable evidence.
- Local `EvidenceScore` result model for tool-event verdicts with explicit
  unavailable values; no vendored SDK source is packaged.
- CI tests on Windows and Linux, plus isolated wheel and sdist installation checks
  on Windows and Linux for Python 3.10 to 3.13, built once and installed from the
  same artifacts.
- Distribution content checks: wheel and sdist must include `py.typed` and exclude
  retired vendored code and local commit helper scripts.
- Integration tests against the real installed Opik SDK, run in CI against the
  minimum supported and the latest release. `REQUIRE_OPIK=1` turns a missing SDK
  into a failure instead of a skip.
- Coverage for the module CLI entry point.
- Offline `ToolEvent` and `check_read_before_edit` Python API with PASS, FAIL,
  and UNKNOWN verdicts, revision matching, and deduplicated content coverage.
- Self-contained offline HTML reports, with escaped input text and no scripts
  or external assets: `report --format html --output report.html`.

- Framework-independent `assert_no_drift(report)` helper for automated regression
  tests, preserving canonical engine gates and summarizing failed prompt IDs.
- Reproducible passing and failing offline examples with expected reports.
- Migration guidance and a prioritized roadmap.

## [0.1.0]

Initial published release. See the linked PyPI release for its artifacts.

[Unreleased]: https://github.com/anant-goel/llm-drift-bench/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/anant-goel/llm-drift-bench/releases/tag/v0.3.0
[0.2.0]: https://github.com/anant-goel/llm-drift-bench/releases/tag/v0.2.0
[0.1.0]: https://pypi.org/project/llm-drift-bench/0.1.0/
