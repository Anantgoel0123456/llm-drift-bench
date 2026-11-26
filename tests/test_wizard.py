"""Tests for the guided plain-English setup wizard and drift.toml presets."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_drift_bench.policy import load_policy
from llm_drift_bench.wizard import (
    PRESET_ANSWERS,
    WizardAnswers,
    build_policy_from_answers,
    format_wizard_next_steps,
    render_wizard_config,
    run_guided_wizard,
)


def test_all_presets_build_valid_round_trippable_drift_toml(tmp_path: Path) -> None:
    for name, preset in PRESET_ANSWERS.items():
        policy = build_policy_from_answers(preset)
        toml_text = render_wizard_config(preset)
        cfg_file = tmp_path / f"{name}.toml"
        cfg_file.write_text(toml_text, encoding="utf-8")
        reloaded = load_policy(cfg_file)
        assert reloaded == policy
        steps = format_wizard_next_steps(preset, config_path=str(cfg_file))
        assert "evaluate" in steps


def test_guided_wizard_interactive_script_normalizes_paths_and_handles_defaults() -> None:
    responses = iter([
        "2",                       # classification
        "strict",                  # numeric sensitivity
        "request_id, $.meta.ts",   # ignored fields
        "tags, items[*].labels",   # unordered arrays
        "n",                       # relax latency guard
        "y",                       # long outputs
    ])
    printed: list[str] = []
    answers, policy = run_guided_wizard(
        input_fn=lambda _prompt: next(responses),
        output_fn=printed.append,
    )
    assert answers.use_case == "classification"
    assert answers.numeric_sensitivity == "strict"
    assert answers.ignored_paths == ("$.request_id", "$.meta.ts")
    assert answers.unordered_array_paths == ("$.tags", "$.items[*].labels")
    assert answers.track_latency is False
    assert answers.long_outputs is True
    assert policy.thresholds.max_numeric_relative_drift == 0.02
    assert policy.thresholds.max_latency_regression_pct == 500.0
    assert policy.long_text_mode == "skip_levenshtein"
    assert policy.field_rules["$.request_id"].ignore is True
    assert policy.field_rules["$.tags"].unordered_array is True
    assert any("setup wizard" in line for line in printed)


def test_guided_wizard_blank_inputs_use_extraction_defaults() -> None:
    responses = iter(["", "", "", "", "", ""])
    answers, policy = run_guided_wizard(
        input_fn=lambda _prompt: next(responses),
        output_fn=lambda _msg: None,
    )
    assert answers == PRESET_ANSWERS["extraction"]
    assert policy == build_policy_from_answers(PRESET_ANSWERS["extraction"])


def test_wizard_answers_validation() -> None:
    with pytest.raises(ValueError, match="use_case"):
        WizardAnswers(use_case="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numeric_sensitivity"):
        WizardAnswers(use_case="extraction", numeric_sensitivity="extreme")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="starting with '\\$'"):
        WizardAnswers(use_case="extraction", ignored_paths=("request_id",))
    with pytest.raises(ValueError, match="boolean"):
        WizardAnswers(use_case="extraction", track_latency="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        WizardAnswers(use_case="extraction", long_outputs="no")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="WizardAnswers"):
        build_policy_from_answers({"use_case": "extraction"})  # type: ignore[arg-type]

    # Leading-dot/bracket path normalization + unrecognized interactive inputs fallback cleanly
    responses = iter(["bogus_case", "moderate", ".req_id, [0].ts", "", "maybe", "maybe"])
    printed: list[str] = []
    ans, _ = run_guided_wizard(input_fn=lambda _p: next(responses), output_fn=printed.append)
    assert ans.use_case == "extraction"
    assert ans.ignored_paths == ("$.req_id", "$[0].ts")
    assert any("Unrecognized choice" in line for line in printed)

