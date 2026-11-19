"""Boundary cases for public evaluation APIs."""
import pytest

from llm_drift_bench import ngram_overlap_similarity


@pytest.mark.parametrize("size", [True, 1.5, 0])
def test_invalid_width_on_identical_inputs(size):
    with pytest.raises(ValueError):
        ngram_overlap_similarity("same", "same", size)


def test_percentile_rejects_nonfinite_input():
    from llm_drift_bench import compute_percentile

    for values, rank in (([float("nan")], 0), ([], float("inf"))):
        with pytest.raises(ValueError):
            compute_percentile(values, rank)
