---
name: Bug Report
about: Report a bug, unexpected drift score, or schema validation issue in llm-drift-bench
title: "[BUG] "
labels: ["bug", "triage"]
assignees: ""
---

## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Run command or Python snippet:
   ```bash
   llm-drift-bench run -b ... -c ...
   ```
2. Provide minimal sample input (`prompts.json` / `.jsonl` snippet):
   ```json
   {"prompt_id": "example_1", "output": "{...}"}
   ```
3. See error or unexpected output.

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Output / Traceback
```text
Paste terminal output or stack trace here
```

## Environment
- **`llm-drift-bench` version**: (e.g., `0.1.0`)
- **Python version**: (e.g., `3.11.8`)
- **Operating System**: (e.g., `Ubuntu 22.04`, `macOS 14`, `Windows 11`)
