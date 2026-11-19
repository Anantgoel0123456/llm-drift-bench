"""Boundary cases for public evaluation APIs."""
import pytest

from llm_drift_bench import ngram_overlap_similarity


@pytest.mark.parametrize("size", [True, 1.5, 0])
def test_invalid_width_on_identical_inputs(size):
    with pytest.raises(ValueError):
        ngram_overlap_similarity("same", "same", size)
