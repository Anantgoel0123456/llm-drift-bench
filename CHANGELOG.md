# Changelog

## [Unreleased]

- Explain saved comparisons with `diagnose` or the Python `diagnose_report` API.
  Output includes evidence, suggested checks and missing timing measurements.
- Opt in to local model suggestions with the `ai` extra. Use `ai-devices` to
  inspect OpenVINO devices before choosing CPU, GPU or NPU.

- Suite files accept the `suite` wrapper across both evaluation APIs. Files with
  conflicting ID aliases or multiple dataset containers now raise an input error.
- Exact edit distance skips shared character spans. Invalid n-gram widths and
  nonfinite percentile inputs raise a validation error.
- Large prompt-set mismatches display the first five missing and extra IDs plus
  omitted counts. Deeply nested input gets a JSON decoder diagnostic.
- Opik exports copy nested metadata so SDK-side edits preserve the source score.

Target: 0.3.0 (development version 0.3.0.dev0).

### Added

- `run --github-summary` appends the comparison result to a GitHub Actions job summary.
- `run --junit-output FILE` saves XML from the same comparison as the console
  and JSON reports, including when drift makes the command exit with code 1.
- `check-tools` rejects an output path that would overwrite the input trace.

- JUnit XML export for canonical engine reports through the Python API and
  `run/report --format junit`. Includes one case per prompt and an overall
  benchmark gate so suite-only failures remain visible to CI test viewers.

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

[Unreleased]: https://github.com/anant-goel/llm-drift-bench/commits/main/
[0.2.0]: https://github.com/anant-goel/llm-drift-bench/releases/tag/v0.2.0
[0.1.0]: https://pypi.org/project/llm-drift-bench/0.1.0/
