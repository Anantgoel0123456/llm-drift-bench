"""Security and structural checks for the reusable composite GitHub Action (action.yml)."""

from __future__ import annotations

import re
from pathlib import Path

ACTION_YAML = Path(__file__).resolve().parents[1] / "action.yml"


def test_action_yaml_structure_and_security_hygiene() -> None:
    text = ACTION_YAML.read_text(encoding="utf-8")
    assert 'using: "composite"' in text

    # Every `uses:` action reference must be pinned to a full 40-char commit SHA.
    uses_lines = [
        line.strip() for line in text.splitlines() if line.strip().startswith("uses:")
    ]
    assert uses_lines, "Expected at least one pinned action step in action.yml."
    sha_re = re.compile(r"^uses:\s+[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}\b")
    for line in uses_lines:
        assert sha_re.match(line), f"Unpinned GitHub Action reference: {line}"

    # Verify required inputs and outputs are declared.
    for required_key in (
        "baseline:",
        "candidate:",
        "suite:",
        "config:",
        "junit-output:",
        "sarif-output:",
        "pr-comment-output:",
        "passed:",
        "mean-drift:",
        "report-path:",
    ):
        assert required_key in text

    # Ensure inputs are never interpolated directly inside `run:` blocks (shell injection guard).
    in_run_block = False
    run_indent = 0
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if stripped.startswith("run:"):
            in_run_block = True
            run_indent = indent
            continue
        if in_run_block:
            if stripped and indent <= run_indent:
                in_run_block = False
            else:
                assert "${{ inputs." not in line, f"Unsafe input interpolation in run block: {line}"
