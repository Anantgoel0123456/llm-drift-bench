"""Multi-candidate evaluation and prompt-level verdict changes."""

from dataclasses import replace

import pytest

from llm_drift_bench import evaluate_many
from llm_drift_bench.models import ModelSample

BASE = [ModelSample("item", '{"price":1}', "reference")]
CHANGED = [replace(BASE[0], output='{"price":8}')]


def test_batch_reuses_single_run_scoring_with_labels() -> None:
    from llm_drift_bench import DriftBenchmarkEngine

    reports = evaluate_many(BASE, [("Q8", BASE), ("Q4", CHANGED)])
    assert reports[0].passed
    expected = DriftBenchmarkEngine().evaluate(BASE, CHANGED)
    assert reports[1] == replace(expected, candidate_model="Q4")


def test_require_multiple_captures() -> None:
    with pytest.raises(ValueError, match="two"):
        evaluate_many(BASE, [("Q8", BASE)])


@pytest.mark.parametrize("label", ["Q8", "  "])
def test_rejects_ambiguous_labels(label: str) -> None:
    with pytest.raises(ValueError, match="labels"):
        evaluate_many(BASE, [("Q8", BASE), (label, BASE)])


def test_alignment_error_names_candidate_and_prompt() -> None:
    with pytest.raises(ValueError, match="Q4.*missing=.*item"):
        evaluate_many(BASE, [("Q8", BASE), ("Q4", [replace(BASE[0], prompt_id="other")])])


def test_mixed_model_names_are_reported_on_either_side() -> None:
    mixed = [*BASE, replace(BASE[0], prompt_id="next", model_name="alien")]
    with pytest.raises(ValueError, match="Baseline.*multiple"):
        evaluate_many(mixed, [])
    with pytest.raises(ValueError, match="Q4.*multiple"):
        evaluate_many(BASE, [("Q8", BASE), ("Q4", mixed)])
