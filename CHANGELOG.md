# Changelog

## [Unreleased]

Target: **0.2.0**. Development version: `0.2.0.dev0`.

### Changed

- Reports distinguish schema conformity from JSON syntax, display candidate
  validation issues, and explain numeric drift normalization and sample coverage.
- `run` fails by default on drift or threshold failure. An individual failing
  sample fails the suite. `--no-fail-on-drift` enables informational runs.
- Character-level Levenshtein now uses the same unit at every supported length.
- Missing timings and token counts remain unavailable instead of being invented.
- Package metadata and CLI version use one version source.

### Fixed

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

- CI tests on Windows and Linux, plus isolated wheel and sdist installation checks.
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

Initial release. Release date and exact artifact contents still need reconciliation
with the local history before publishing the next release notes.

[Unreleased]: https://github.com/anant-goel/llm-drift-bench/commits/main/
[0.1.0]: https://pypi.org/project/llm-drift-bench/0.1.0/
