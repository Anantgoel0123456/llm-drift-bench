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

## 0.3.0

- [x] JUnit XML (`--junit-output`), OASIS SARIF 2.1.0 (`--sarif-output`, `--format sarif`), and RFC-4180 CSV (`--format csv`) exports.
- [x] GitHub step summaries (`run --github-summary`) and compact PR comments (`--format pr-comment --previous`).
- [x] Policy-aware semantic JSON/text `diff` CLI and policy-aware HTML response diffs (`--include-diffs`).
- [x] Evaluate and rank multiple candidates (`compare-many`, `compare-reports`, `--sort-by`, `--format rich`, `--against`, `--hotspots`, `--inspect-prompt`, offline SVG Pareto frontier HTML) plus `--category` filtering on `run`/`validate`.
- [x] Configurable evaluation policies (`drift.toml` / `drift.json`): custom metric weights, exact and `[*]` wildcard JSONPath `ignore` / `unordered_array` / `tolerance_abs` / `tolerance_rel` rules, and `--long-text-mode skip_levenshtein`.
- [x] Zero-dependency OpenAI-compatible capture collector (`capture` with `--concurrency`, `--runs`, `--json-mode`, `--resume`, `--header`, bounded retries, timeouts, and env-var secret handling).
- [x] Multi-run statistics (`stats`): deterministic bootstrap CIs, flaky-prompt detection, paired sign-flip permutation tests with Cohen's `d`, and overflow-safe exact two-sided McNemar conformity tests.
- [x] Append-only drift history ledger (`history record|show|check`) with moving-window regression alerts, `--candidate`/`--tag` filters, Unicode sparklines, and offline SVG trend HTML with tooltips.
- [x] Offline vendor transcript adapters (`adapters.py` + `check-tools --adapter normalized|openai|anthropic|otel`, supporting OpenAI Responses `output` & OTLP `resourceSpans`) plus `--max-failed-calls` retry budgets and secret-leak checks.
- [x] Automatic JSON Schema inference (`infer-schema`), threshold auto-calibration with `[*]` wildcard collapsing (`calibrate`), and pre-flight validation (`doctor`).
- [x] Pluggable scorer registry & `score` CLI (`char_ngram_cosine`, `json_key_jaccard`, `markdown_structure`, `python_ast`, `tfidf_cosine`, and `llm_drift_bench.scorers` entry-point discovery).
- [x] Lightweight PEP 562 lazy exports and lazy CLI/Rich/Pydantic loading (~8 ms package import).

## 0.4.0

- [x] Cross-prompt JSON field distribution drift (`distribution`: categorical TVD/JSD, numeric p10/p50/p90 & normalized 1D Wasserstein, offline HTML/Markdown).
- [x] Guided plain-English setup wizard & presets (`wizard`), curated starter benchmark packs (`init-sample --pack`), and unified offline audit dashboard (`audit`).
- [x] Optional `top_logprobs` capture & token entropy / top-k JSD drift (`capture --top-logprobs`, `logprobs`), deterministic quantization quality regression checks (`check-quality`), and standalone semantic field format checks (`check-formats`).
- [x] Composite GO/NO-GO release gates (`gate`), multi-candidate Prompt x Candidate drift heatmap matrix (`matrix`), explicit token/cost drift & budgets (`cost`), offline SHA-256 baseline snapshot manifests (`snapshot`), deterministic credential/PII redaction (`redact`), stratified subsampling (`subsample`), suite linting (`lint-suite`), offline batch/CSV importers (`import-captures`), and static SVG status badges (`badge`).
- [x] Native Anthropic Messages & Google Gemini capture + Gemini tool-trace adapter, reusable composite GitHub Action (`action.yml`), and automated GitHub Release publishing in `publish.yml`.

## Unreleased (heading toward 0.5.0)

- [x] Rolling-baseline cross-window distribution drift with natural window-to-window jitter measurement (`rolling.py`, `rolling-distribution`).
- [x] Segmented field distributions (`--group-by category|metadata:<key>`) with `localized` vs `widespread` blast-radius classification (`--fail-on-widespread`).
- [x] Multi-window persistence gating (`--persist-windows`) and schema-versioned drift history ledgers (`HISTORY_SCHEMA_VERSION = 1`).
- [x] Downstream consumer rejection-rate creep checks (`check-consumer`, `audit`, `gate`, and `doctor` pre-flight validation).
- [x] Reusable GitHub Action offline audit dashboard and blast-radius inputs (`audit-html-output`, `group-by`, `fail-on-widespread`).

## Future exploration

| Feature | Value | Relative effort |
| :--- | :--- | :--- |
| Optional embedding scorers | Add an opt-in semantic similarity signal alongside lexical scorers | Medium |
| Hosted CI benchmark case study | Publish a reproducible real-model quantization benchmark report | Medium |

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
