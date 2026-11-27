"""Tests for curated starter benchmark packs."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_drift_bench.engine import DriftBenchmarkEngine, load_model_samples, load_prompt_cases
from llm_drift_bench.packs import get_pack, list_packs, scaffold_pack
from llm_drift_bench.policy import load_policy
from llm_drift_bench.prompt_lint import lint_prompt_suite


@pytest.mark.parametrize("pack_name", list_packs())
def test_every_starter_pack_is_lint_clean_and_separates_pass_from_drift(
    pack_name: str, tmp_path: Path
) -> None:
    pack = get_pack(pack_name)
    assert pack.name == pack_name
    assert pack.description

    files = scaffold_pack(pack_name, tmp_path / pack_name)
    assert set(files) == {"prompts", "baseline", "candidate_pass", "candidate_drift", "policy"}

    prompts = load_prompt_cases(files["prompts"])
    baseline = load_model_samples(files["baseline"])
    cand_pass = load_model_samples(files["candidate_pass"])
    cand_drift = load_model_samples(files["candidate_drift"])
    policy = load_policy(files["policy"])

    # Every pack suite must have zero lint errors or warnings.
    issues = lint_prompt_suite(prompts)
    assert issues == []

    engine = DriftBenchmarkEngine(policy=policy)
    pass_report = engine.evaluate(baseline, cand_pass, prompt_cases=prompts)
    drift_report = engine.evaluate(baseline, cand_drift, prompt_cases=prompts)
    assert pass_report.passed is True, pass_report.failure_reasons
    assert drift_report.passed is False


def test_unknown_pack_raises() -> None:
    assert "agent" in list_packs()
    with pytest.raises(ValueError, match="Unknown starter pack"):
        get_pack("nonexistent")
