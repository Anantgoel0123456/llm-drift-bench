from llm_drift_bench.metrics import compute_percentile


def test_interpolation_between_opposite_finite_extremes() -> None:
    assert compute_percentile([-1e308, 1e308], 50) == 0
    assert compute_percentile([-1e308, 1e308], 25) == -5e307
