"""Static security and contract tests for the reusable composite GitHub Action."""

from __future__ import annotations

import re
from pathlib import Path

ACTION_YML = Path(__file__).resolve().parents[1] / "action.yml"


def test_action_yml_pins_sha_avoids_shell_interpolation_and_declares_audit_inputs() -> None:
    text = ACTION_YML.read_text(encoding="utf-8")

    # actions/setup-python must be pinned to a full 40-char commit SHA
    assert re.search(r"uses:\s*actions/setup-python@[0-9a-f]{40}\b", text)

    # All required and newly added inputs/outputs are declared
    for key in (
        "baseline:",
        "candidate:",
        "suite:",
        "config:",
        "junit-output:",
        "sarif-output:",
        "pr-comment-output:",
        "audit-html-output:",
        "group-by:",
        "fail-on-widespread:",
        "passed:",
        "mean-drift:",
        "report-path:",
    ):
        assert key in text

    # Never interpolate ${{ inputs.* }} directly inside shell run: blocks (shell-injection guard)
    in_run = False
    run_indent = 0
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if stripped.startswith("run:"):
            in_run = True
            run_indent = indent
            assert "${{ inputs." not in stripped
            continue
        if in_run:
            if stripped and indent <= run_indent:
                in_run = False
            else:
                assert "${{ inputs." not in raw_line, f"Unsafe interpolation in run: {raw_line!r}"
