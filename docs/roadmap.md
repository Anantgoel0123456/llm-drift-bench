# Roadmap

This is a prioritized plan, not a list of shipped capabilities or release dates.

## 0.2.0: trustworthy comparisons

- [x] Reject duplicate and mismatched prompt IDs.
- [x] Preserve missing measurements and reject invalid thresholds.
- [x] Make CI drift failures explicit by default.
- [x] Provide reproducible passing and failing examples.
- [x] Add a framework-independent assertion helper for automated tests.
- [x] Unify CLI engine, saved report, Rich, and Markdown scores and failure reasons.
- [x] Add standalone HTML exports of canonical engine reports.
- [x] Validate supported schema keyword shapes and enforce a bounded regex subset.
- [x] Reject nonfinite and out-of-range numbers in captured output parsing and numeric drift.
- [x] Remove ambiguous JSON key paths in structural, field-match, and numeric metrics.
- [x] Add an offline read-before-edit library check for normalized tool events.
- [x] Handle BOM input and document encoding and size limits consistently.
- [x] Allow manual checks or an explicit publish to the selected registry. Published releases target PyPI.
- [x] Verify built wheels outside the checkout on Windows/Linux and Python 3.10-3.13.

## Remaining verification and integrations

- Hosted CI passed, as confirmed by the maintainer's run screenshot on
  2026-11-15: eight source-test jobs, eight installed-package jobs, two SDK
  integration jobs, lint/type checking, and package build checks. Keep these
  checks green on subsequent changes; this does not verify registry publishing.
- Trusted Publishing: verify registry and GitHub environment settings using
  `docs/publishing.md`, then perform the authorized TestPyPI release checks.
- Vendor transcript adapters: the checker accepts normalized tool events, but
  does not import Claude Code or OpenAI transcripts. Implementation needs a
  sanitized sample and tool-result semantics. Line ranges alone cannot establish
  exact character coverage without the corresponding file revision/content;
  retain UNKNOWN when the evidence is insufficient. This is a future integration,
  not a blocker for the existing offline output-comparison API.

The CI version constraint and integral-float schema-limit issues are fixed locally.

## 0.3.0 development (implemented on `main`)

- [x] JUnit XML exports with per-prompt cases and an overall benchmark gate (`--junit-output`, `--format junit`).
- [x] GitHub step summaries (`run --github-summary`) and compact PR comments (`--format pr-comment --previous`).
- [x] Evaluate and rank multiple candidates (`compare-many`, `compare-reports`, `--sort-by`, `--format rich`, `--against`, `--hotspots`, `--inspect-prompt`, offline SVG Pareto frontier HTML).
- [x] Configurable evaluation policies (`drift.toml` / `drift.json`): custom metric weights, per-JSONPath `ignore` / `unordered_array` / `tolerance_abs` / `tolerance_rel` rules, and `--long-text-mode skip_levenshtein`.
- [x] Zero-dependency OpenAI-compatible capture collector (`capture` with `--concurrency`, `--runs`, `--json-mode`, bounded retries, timeouts, and env-var secret handling).
- [x] Multi-run statistics (`stats`): deterministic bootstrap confidence intervals, flaky-prompt detection, paired sign-flip permutation tests, and exact two-sided McNemar conformity tests.
- [x] Append-only drift history ledger (`history record|show`) with `--candidate`/`--tag` filters, Unicode sparklines, and offline SVG trend HTML.
- [x] Offline vendor transcript adapters (`adapters.py` + `check-tools --adapter normalized|openai|anthropic|otel`, including OpenAI Responses API) plus retry and secret-leak workflow checks.
- [x] Automatic JSON Schema inference (`infer-schema`) and threshold auto-calibration + volatile field detection (`calibrate`).
- [x] Pluggable scorer registry (`scorers.py`: `char_ngram_cosine`, `tfidf_cosine`, `python_ast`, and `llm_drift_bench.scorers` entry-point discovery).
- [x] Lightweight PEP 562 lazy exports and lazy CLI/Rich/Pydantic loading (~8 ms package import).

## Future exploration

| Feature | Value | Relative effort |
| :--- | :--- | :--- |
| Optional embedding scorers | Add an opt-in semantic similarity signal alongside lexical scorers | Medium |
| Hosted CI benchmark case study | Publish a reproducible real-model quantization benchmark report | Medium |
| Versioned report schema migrations | Formal schema version tags for long-lived history ledgers | Small |

Embedding scorers (when added as an optional extra or pluggable scorer) must never silently alter existing deterministic lexical scores.

## Build useful community evidence

1. Recruit a few users with a concrete model migration or quantization problem.
2. Publish one reproducible case study with licensed, sanitized raw captures,
   model revisions, inference settings, hardware, repeats, and limitations.
3. Turn their failures into scoped issues with acceptance criteria.
4. Help external contributors get meaningful fixes reviewed and merged.
5. Track repeat users, dependent projects, and external contributions alongside
   downloads. Downloads are not a count of unique people.

Do not set adoption promises or treat commit volume as impact. A small tool that
reliably prevents a real regression is a stronger foundation than a broad feature
list with inconsistent scores.

## Observability direction, not current product claims

The local HTML report now consumes canonical saved report data without a
second scoring path. Add a versioned report schema and run
provenance before storing history: dataset identity, model revision, evaluation
version, thresholds, collection settings, and which measurements are absent.

Trace integration should initially reference existing trace IDs rather than
implement another tracing backend. Decide how to redact sensitive prompts and
tool arguments before collecting or sharing them. Exporting a final agent output
is not equivalent to tracing agent execution.

Production monitoring needs sampling, retention, authentication, resource
limits, alert policy, and deployment maintenance. It is outside the 0.2.0
release scope. No hosted dashboard or monitoring service is currently provided.
