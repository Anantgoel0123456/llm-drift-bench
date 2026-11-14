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
- [ ] Complete schema validation policy, malformed-schema checks, and regex safety.
- [x] Reject nonfinite and out-of-range numbers in captured output parsing and numeric drift.
- [x] Remove ambiguous JSON key paths in structural, field-match, and numeric metrics.
- [x] Add an offline read-before-edit library check for normalized tool events.
- [x] Handle BOM input and document encoding and size limits consistently.
- [x] Separate TestPyPI manual publishing from production release publishing.
- [ ] Verify built wheels outside the checkout on Windows and Python 3.10-3.13.

## After the reliability release

| Feature | Value | Relative effort |
| --- | --- | --- |
| GitHub step summary and JUnit | Make failures visible where developers work | Medium |
| Configurable weights and drift.toml | Reuse task-specific evaluation policies | Medium |
| Compare one baseline with many candidates | Inspect a quantization sweep | Medium |
| Endpoint collection with provenance | Capture reproducible inputs | Large |
| Optional embeddings | Add a distinct semantic signal | Large |
| Run history and trends | Compare versioned runs over time | Large |
| Trace import and correlation | Link failures to externally captured spans | Large |

Collection needs explicit timeouts, bounded retries, secret handling, model and
sampling metadata, and documented token accounting. Embedding scores must not
silently change the interpretation of existing lexical scores.

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
