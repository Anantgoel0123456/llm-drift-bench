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


def test_insert_inside_long_shared_context():
    from llm_drift_bench import normalized_levenshtein_similarity

    a = "a" * 5000 + "z" * 5000
    b = "a" * 5000 + "x" + "z" * 5000
    assert normalized_levenshtein_similarity(a, b) == 1 - 1 / len(b)


def test_suite_wrapper(tmp_path):
    from llm_drift_bench import load_prompt_cases

    path = tmp_path / "suite.json"
    path.write_text('{"suite":[{"id":"abc","prompt":"test"}]}')
    assert load_prompt_cases(path)[0].prompt_id == "abc"


def test_two_output_containers(tmp_path):
    from llm_drift_bench import load_model_samples

    path = tmp_path / "samples.json"
    path.write_text('{"samples":[],"outputs":[{"id":"abc","output":"x"}]}')
    with pytest.raises(ValueError, match="Ambiguous dataset containers"):
        load_model_samples(path)
