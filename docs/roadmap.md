# Roadmap

This is a prioritized plan, not a list of shipped capabilities or release dates.

## 0.2.0: trustworthy comparisons

- [x] Reject duplicate and mismatched prompt IDs.
- [x] Preserve missing measurements and reject invalid thresholds.
- [x] Make CI drift failures explicit by default.
- [x] Provide reproducible passing and failing examples.
- [ ] Unify engine, saved report, Rich, and Markdown scores and failure reasons.
- [ ] Complete schema validation policy, malformed-schema checks, and regex safety.
- [ ] Reject nonfinite JSON numbers; remove ambiguous JSON key paths.
- [ ] Handle BOM input and document encoding and size limits consistently.
- [ ] Separate TestPyPI manual publishing from production release publishing.
- [ ] Verify built wheels outside the checkout on Windows and Python 3.10-3.13.

## After the reliability release

| Feature | Value | Relative effort |
| --- | --- | --- |
| pytest assertion helper | Put regression checks next to application tests | Small |
| GitHub step summary and JUnit | Make failures visible where developers work | Medium |
| Configurable weights and drift.toml | Reuse task-specific evaluation policies | Medium |
| Compare one baseline with many candidates | Inspect a quantization sweep | Medium |
| Endpoint collection with provenance | Capture reproducible inputs | Large |
| Optional embeddings | Add a distinct semantic signal | Large |

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
