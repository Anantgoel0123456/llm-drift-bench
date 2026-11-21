"""Unit and CLI tests for bootstrap CIs, permutation tests, and multi-run stability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main
from llm_drift_bench.engine import DriftBenchmarkEngine
from llm_drift_bench.models import ModelSample
from llm_drift_bench.stats import (
    bootstrap_mean_ci,
    mcnemar_exact_test,
    paired_permutation_test,
    summarize_multi_run,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_bootstrap_mean_ci_bounds_and_validation() -> None:
    ci = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4, 0.5], confidence=0.95, n_resamples=500, seed=42)
    assert ci.mean == pytest.approx(0.3)
    assert 0.1 <= ci.lower <= ci.mean <= ci.upper <= 0.5
    assert ci.to_dict()["sample_size"] == 5

    single = bootstrap_mean_ci([0.25])
    assert single.lower == single.mean == single.upper == 0.25

    with pytest.raises(ValueError, match="empty"):
        bootstrap_mean_ci([])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_mean_ci([0.1], confidence=1.0)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_mean_ci([0.1], n_resamples=5)


def test_paired_permutation_test_detects_significant_shift() -> None:
    base_scores = [0.01, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03]
    worse_scores = [0.40, 0.45, 0.50, 0.42, 0.48, 0.39, 0.44, 0.47]
    sig = paired_permutation_test(base_scores, worse_scores, n_permutations=1000, seed=7)
    assert sig.observed_diff > 0.35
    assert sig.p_value < 0.05
    assert sig.significant is True

    equal = paired_permutation_test(base_scores, base_scores, n_permutations=100, seed=7)
    assert equal.observed_diff == 0.0
    assert equal.p_value == 1.0
    assert equal.significant is False

    with pytest.raises(ValueError, match="equal length"):
        paired_permutation_test([0.1], [0.1, 0.2])


def test_summarize_multi_run_detects_flaky_prompts() -> None:
    base = [
        ModelSample(prompt_id="stable", model_name="fp16", output='{"v": 1}'),
        ModelSample(prompt_id="flaky", model_name="fp16", output='{"v": 100}'),
    ]
    run1 = DriftBenchmarkEngine().evaluate(
        base,
        [
            ModelSample(prompt_id="stable", model_name="q4", output='{"v": 1}'),
            ModelSample(prompt_id="flaky", model_name="q4", output='{"v": 100}'),
        ],
    )
    run2 = DriftBenchmarkEngine().evaluate(
        base,
        [
            ModelSample(prompt_id="stable", model_name="q4", output='{"v": 1}'),
            ModelSample(prompt_id="flaky", model_name="q4", output='{"v": 999}'),
        ],
    )
    summary = summarize_multi_run([run1, run2], n_resamples=200, seed=1)
    assert summary.runs == 2
    assert summary.flaky_prompts == ("flaky",)
    by_id = {item.prompt_id: item for item in summary.prompt_stability}
    assert by_id["flaky"].flaky is True
    assert by_id["flaky"].drifted_runs == 1
    assert by_id["stable"].flaky is False
    assert by_id["stable"].std_drift == 0.0


def test_cli_stats_text_and_json_with_significance(capsys) -> None:
    pass_file = str(EXAMPLES / "expected-pass.json")
    drift_file = str(EXAMPLES / "expected-drift.json")
    assert main(["stats", drift_file, pass_file, "--format", "text"]) == 0
    out = capsys.readouterr().out
    assert "Multi-run stability (2 run(s), 2 prompt(s))" in out
    assert "Flaky prompts (2): invoice-01, ticket-01" in out

    assert main(
        [
            "stats",
            drift_file,
            "--compare-against",
            pass_file,
            "--resamples",
            "200",
            "--format",
            "json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "mean_drift_ci" in payload
    assert "significance" in payload
    assert "mcnemar_conformity" in payload
    assert payload["significance"]["observed_diff"] > 0


def test_mcnemar_exact_test_binary_shifts() -> None:
    # 10 prompts regressed (True -> False), 0 recovered -> exact p = 2 * (1/1024) = 0.001953
    res = mcnemar_exact_test([True] * 12, [True, True] + [False] * 10)
    assert res.b_regressed == 10 and res.c_recovered == 0
    assert res.significant is True
    assert res.p_value == pytest.approx(2 / 1024)

    same = mcnemar_exact_test([True, False], [True, False])
    assert same.p_value == 1.0 and same.significant is False
