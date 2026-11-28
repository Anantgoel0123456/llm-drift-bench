# Workflow recipes

These recipes describe the development checkout, not all features in 0.1.0.
The benchmark reads captured responses. It does not call an LLM, run an agent,
retrieve documents, or instrument an application.

## Model swap or quantization

Run the same prompt suite against both configurations using your inference
runner. Save the raw response strings under matching prompt IDs. Record actual
latency and token counts if the runner provides them. Keep model revisions,
sampling settings, and hardware details with your experiment.

Run the comparison with the same thresholds. Inspect individual failures before
changing a gate. A model that agrees with a baseline can still be wrong.

## RAG answer regression

Keep a fixed question set and export each pipeline's final answer as `output`.
Use stable question IDs across runs. Compare responses after changing the model,
prompt, retriever, or document corpus, ideally changing one factor at a time.

This detects output changes. It does not evaluate retrieval relevance,
faithfulness to documents, or citation correctness. Those need reference data
and task-specific evaluation. A useful paraphrase may trigger lexical drift.

## Agent final outputs and tool arguments

Choose an explicit checkpoint, such as the final answer or a tool-argument
payload. Export the checkpoint value as a response string and use a supported
JSON schema for structured output. For multiple checkpoints, give each a unique
case ID and ensure both runs capture exactly the same cases.

The benchmark does not execute tool calls or inspect intermediate spans. A
schema-valid tool argument can still be unsafe or incorrect. Use separate
execution and permission checks in the application.

## Automated regression tests

Evaluate with `DriftBenchmarkEngine`, then pass the resulting report to
`assert_no_drift` (or use `assert_audit_passed`, `assert_formats_valid`,
`assert_no_quality_regressions`, `assert_logprobs_within_budget`,
`assert_cost_within_budget`, `assert_no_conversation_drift`,
`assert_no_tool_trace_drift`, `assert_no_consumer_regression`, and
`assert_no_rolling_drift` for specialized checks). Passing reports return
normally. Failing reports raise `AssertionError` with suite reasons and up to
five failed prompt IDs. Reports with no evaluated samples raise `ValueError`.

For debugging, inspect `report.results` and `report.failure_reasons`. For stored
results use `report.to_json()`. Preserve the original captures separately and
review them for sensitive content before sharing. Do not equate absent timing
data with good latency.

## Local report viewer

Use `report -i report.json --format html -o report.html` to view an engine report
in a browser. This is a standalone offline document, not a monitoring service.

## Multi-turn conversation trajectory drift

When assistant behavior drifts gradually over multi-turn dialogs (snowballing
changes, dropped JSON formatting on turn 3, or late-turn refusal flips), import
existing OpenAI chat, Anthropic messages, or ShareGPT exports, scrub secrets/PII,
pin the baseline snapshot, and run trajectory checks offline:

```bash
# 1. Convert existing chat logs to canonical ConversationRecord JSONL
llm-drift-bench import-conversations --from openai-chat -i raw-baseline.jsonl -o baseline-conv.jsonl

# 2. Scrub API keys and PII before committing golden transcripts
llm-drift-bench redact --conversations -i baseline-conv.jsonl -o baseline-conv-clean.jsonl

# 3. Pre-flight check conversation_id alignment and export HTML, SARIF 2.1.0, JUnit XML, or CSV reports
llm-drift-bench doctor --baseline-conversations baseline-conv-clean.jsonl --candidate-conversations candidate-conv.jsonl
llm-drift-bench check-conversations -b baseline-conv-clean.jsonl -c candidate-conv.jsonl --diagnose --format html -o conversation-drift.html
llm-drift-bench check-conversations -b baseline-conv-clean.jsonl -c candidate-conv.jsonl --format sarif -o conversation-drift.sarif
llm-drift-bench check-conversations -b baseline-conv-clean.jsonl -c candidate-conv.jsonl --format junit -o conversation-drift.junit.xml
llm-drift-bench check-conversations -b baseline-conv-clean.jsonl -c candidate-conv.jsonl --format csv -o conversation-drift.csv
```

You can also collect live multi-turn candidate transcripts directly from OpenAI-compatible,
Anthropic, or Gemini endpoints by replaying user/system templates with
`llm-drift-bench capture-conversations -i baseline-conv-clean.jsonl --endpoint URL --model MODEL -o candidate-conv.jsonl`.

To gate releases in one command, pass `--baseline-conversations` and
`--candidate-conversations` (and optional `--baseline-trace` / `--candidate-trace` for
agent tool-trace drift, or standalone
`llm-drift-bench diff-tools -b examples/trace-baseline.jsonl -c examples/trace-candidate-drift.jsonl --fail-on-new-edit-resources --diagnose --format html|sarif|junit|pr-comment`)
directly to `llm-drift-bench audit` (add `--diagnose` for cross-dimension root causes)
or `llm-drift-bench gate`. Use `redact --traces` to scrub credentials from agent traces
before committing them, track multi-turn drift over time via
`history record --conversation-drift`, export PR comments from `audit`,
`check-conversations`, `diff-tools`, `check-consumer`, or `rolling-distribution` with
`--format pr-comment`, and generate offline SVG status badges from any saved JSON report
via `llm-drift-bench badge -i report.json -o badge.svg`.
