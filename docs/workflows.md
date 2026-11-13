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
`assert_no_drift`. Passing reports return normally. Failing reports raise
`AssertionError` with suite reasons and up to five failed prompt IDs. Reports
with no evaluated samples raise `ValueError`.

For debugging, inspect `report.results` and `report.failure_reasons`. For stored
results use `report.to_json()`. Preserve the original captures separately and
review them for sensitive content before sharing. Do not equate absent timing
data with good latency.

## Local report viewer

Use `report -i report.json --format html -o report.html` to view an engine report
in a browser. This is a standalone offline document, not a monitoring service.
Run history and trace correlation remain roadmap items.
