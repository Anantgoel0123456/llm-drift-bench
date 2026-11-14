"""Saved reports must be internally consistent, not merely well typed."""

import copy
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from llm_drift_bench import (
    DriftBenchmarkEngine,
    DriftThresholds,
    ModelSample,
    PromptCase,
    evaluate_suite,
    load_report,
)

SCHEMA = {
    "type": "object", "required": ["n"],
    "properties": {"n": {"type": "integer"}, "tag": {"type": "string"}},
}


def _samples(latency: bool) -> list[ModelSample]:
    timing = (10, 12, 11) if latency else (None, None, None)
    outputs = ('{"n": 1, "tag": "x"}', '{"n": 2}', "plain text")
    return [
        ModelSample(prompt_id=pid, output=output, latency_ms=ms)
        for pid, output, ms in zip("abc", outputs, timing, strict=True)
    ]


def _report(latency: bool = True) -> dict[str, Any]:
    cases = [PromptCase("a", "", SCHEMA), PromptCase("b", "", SCHEMA), PromptCase("c", "")]
    report = DriftBenchmarkEngine().evaluate(_samples(latency), _samples(latency), cases)
    assert report.passed
    return report.to_dict()


def _write(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_exact_reproduction_drift_score_99_with_passed_true(tmp_path: Path) -> None:
    raw = _report()
    raw["results"][0]["composite_drift_score"] = 99
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_report(_write(tmp_path, raw))


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def apply(raw: dict[str, Any]) -> None:
        target: Any = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return apply


def _drop(path: tuple[Any, ...]) -> Callable[[dict[str, Any]], None]:
    def apply(raw: dict[str, Any]) -> None:
        target: Any = raw
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return apply


MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "composite-high": _set(("results", 0, "composite_drift_score"), 99),
    "composite-negative": _set(("results", 0, "composite_drift_score"), -0.5),
    "jaccard-high": _set(("results", 1, "token_jaccard_similarity"), 1.5),
    "ngram-negative": _set(("results", 1, "ngram_similarity"), -1),
    "levenshtein-high": _set(("results", 2, "levenshtein_similarity"), 2),
    "structural-high": _set(("results", 0, "json_structural_similarity"), 7),
    "field-match-negative": _set(("results", 0, "json_exact_field_match"), -0.1),
    "numeric-high": _set(("results", 0, "json_numeric_drift"), 3),
    "mean-composite-mismatch": _set(("mean_composite_drift",), 0.2),
    "max-composite-mismatch": _set(("max_composite_drift",), 0.2),
    "mean-jaccard-mismatch": _set(("mean_token_jaccard",), 0.1),
    "mean-levenshtein-mismatch": _set(("mean_levenshtein_similarity",), 0.1),
    "mean-structural-mismatch": _set(("mean_json_structural_similarity",), 0.1),
    "mean-structural-dropped": _set(("mean_json_structural_similarity",), None),
    "numeric-mean-invented": _set(("mean_json_numeric_drift",), 0.05),
    "rate-mismatch": _set(("candidate_json_conformity_rate",), 0.3),
    "baseline-rate-mismatch": _set(("baseline_json_conformity_rate",), 0.1),
    "passed-flag-contradicts-validity": _set(
        ("results", 0, "candidate_validation", "passed"), False),
    "conformant-without-valid-json": _set(
        ("results", 2, "candidate_validation", "is_schema_conformant"), True),
    "invalid-json-without-issue": _set(("results", 2, "candidate_validation", "issues"), []),
    "conformant-with-issues": _set(("results", 0, "candidate_validation", "issues"), [
        {"path": "$", "message": "x", "rule": "type"}]),
    "comparison-scores-for-invalid-json": _set(("results", 2, "json_structural_similarity"), 0.5),
    "missing-comparison-scores": _set(("results", 0, "json_structural_similarity"), None),
    "latency-p50-above-p95": _set(("baseline_p50_latency_ms",), 99.0),
    "latency-negative": _set(("candidate_p95_latency_ms",), -1.0),
    "latency-half-present": _set(("candidate_p50_latency_ms",), None),
    "latency-delta-missing": _set(("results", 0, "latency_delta_ms"), None),
    "passes-despite-mean-over-threshold": lambda raw: (
        raw["thresholds"].__setitem__("max_composite_drift", 0.0)
        or raw["results"][0].__setitem__("composite_drift_score", 0.1)
        or raw.__setitem__("mean_composite_drift", 0.0333)
        or raw.__setitem__("max_composite_drift", 0.1)
    ),
    "passes-despite-low-conformity": _set(("thresholds", "min_json_conformity_rate"), 1.0),
    "threshold-out-of-range": _set(("thresholds", "max_composite_drift"), 5),
    "missing-per-result-field": _drop(("results", 0, "composite_drift_score")),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_inconsistent_reports_are_rejected(name: str, tmp_path: Path) -> None:
    raw = _report()
    if name == "passes-despite-low-conformity":
        raw["candidate_json_conformity_rate"] = 0.5
    MUTATIONS[name](raw)
    with pytest.raises(ValueError):
        load_report(_write(tmp_path, raw))


def test_untouched_report_round_trips_exactly(tmp_path: Path) -> None:
    raw = _report()
    assert load_report(_write(tmp_path, raw)).to_dict() == raw


def test_drifted_flag_must_match_thresholds(tmp_path: Path) -> None:
    base = [ModelSample(prompt_id="p", output='{"n": 1}')]
    cand = [ModelSample(prompt_id="p", output='{"n": 9, "extra": true}')]
    raw = DriftBenchmarkEngine().evaluate(base, cand).to_dict()
    assert raw["results"][0]["drifted"] and not raw["passed"]
    load_report(_write(tmp_path, raw))
    hidden = copy.deepcopy(raw)
    hidden["results"][0]["drifted"] = False
    hidden["drifted_samples"] = 0
    with pytest.raises(ValueError):
        load_report(_write(tmp_path, hidden))
    quiet = DriftBenchmarkEngine().evaluate(base, base).to_dict()
    quiet["results"][0]["drifted"] = True
    quiet["drifted_samples"] = 1
    quiet["passed"] = False
    quiet["failure_reasons"] = ["forged"]
    with pytest.raises(ValueError, match="without exceeding"):
        load_report(_write(tmp_path, quiet))


def test_schema_regression_must_be_marked_drifted(tmp_path: Path) -> None:
    case = [PromptCase("p", "", SCHEMA)]
    base = [ModelSample(prompt_id="p", output='{"n": 1}')]
    cand = [ModelSample(prompt_id="p", output='{"n": "one"}')]
    raw = DriftBenchmarkEngine().evaluate(base, cand, case).to_dict()
    raw["results"][0]["drifted"] = False
    raw["drifted_samples"] = 0
    with pytest.raises(ValueError, match="not marked drifted"):
        load_report(_write(tmp_path, raw))


def test_legacy_reports_must_match_their_metrics(tmp_path: Path) -> None:
    samples = [
        ModelSample(prompt_id="a", output='{"n": 1}', latency_ms=5, tokens_generated=10),
        ModelSample(prompt_id="b", output="text", latency_ms=7, tokens_generated=12),
    ]
    shifted = [
        ModelSample(prompt_id="a", output='{"n": 2}', latency_ms=6, tokens_generated=10),
        ModelSample(prompt_id="b", output="text!", latency_ms=9, tokens_generated=12),
    ]
    legacy = evaluate_suite([], samples, shifted).model_dump(mode="json")
    assert load_report(_write(tmp_path, legacy)).total_prompts == 2
    for mutate in (
        lambda r: r.__setitem__("total_prompts", 3),
        lambda r: r.__setitem__("mean_drift_score", 0.0 if r["mean_drift_score"] else 0.9),
        lambda r: r.__setitem__("exact_match_rate", 0.99),
        lambda r: r.__setitem__("candidate_json_adherence_rate", 0.0),
        lambda r: r.__setitem__("mean_baseline_latency_ms", None),
        lambda r: r["metrics"][0].__setitem__("latency_delta_ms", 500.0),
        lambda r: r["metrics"][0].__setitem__("prompt_id", "b"),
        lambda r: r["metrics"][0].__setitem__("drift_score", 5.0),
        lambda r: r.update(metrics=[], total_prompts=0),
    ):
        broken = copy.deepcopy(legacy)
        mutate(broken)
        with pytest.raises(ValueError):
            load_report(_write(tmp_path, broken))


def _random_json(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.1:
        return "not json at all"
    if kind < 0.2:
        return '{"n": "wrong", "tag": 3}'
    return json.dumps({
        "n": rng.choice([1, 2, 3, 40, 1000, 7.0]),
        "tag": rng.choice(["a", "b", "ccc"]),
        "score": round(rng.uniform(0, 100), rng.randint(0, 6)),
        "nested": {"v": rng.choice([0, 1, 2.5, -4.25])},
    })


@pytest.mark.parametrize("seed", range(60))
def test_every_engine_generated_report_loads(seed: int, tmp_path: Path) -> None:
    """Strict validation must never reject what the engine itself produces."""
    rng = random.Random(seed)
    count = rng.randint(1, 8)
    timed = rng.random() < 0.7
    partial = rng.random() < 0.2
    base, cand, cases = [], [], []
    for index in range(count):
        pid = f"p{index}"
        has_latency = timed and not (partial and index == 0)
        base.append(ModelSample(
            prompt_id=pid, output=_random_json(rng),
            latency_ms=round(rng.uniform(0, 500), 3) if has_latency else None))
        cand.append(ModelSample(
            prompt_id=pid, output=_random_json(rng),
            latency_ms=round(rng.uniform(0, 800), 3) if has_latency else None))
        cases.append(PromptCase(pid, "", SCHEMA if rng.random() < 0.6 else None))
    thresholds = DriftThresholds(
        max_composite_drift=rng.choice([0.05, 0.25, 0.5, 1.0]),
        min_json_conformity_rate=rng.choice([0.0, 0.5, 0.95, 1.0]),
        max_numeric_relative_drift=rng.choice([0.0, 0.15, 1.0]),
        max_latency_regression_pct=rng.choice([0.0, 50.0, 500.0]),
    )
    use_cases = rng.random() < 0.8
    report = DriftBenchmarkEngine(thresholds).evaluate(base, cand, cases if use_cases else None)
    path = tmp_path / "saved.json"
    path.write_text(report.to_json(), encoding="utf-8")
    assert load_report(path).to_dict() == report.to_dict()


def _forge_pass(raw: dict[str, Any]) -> dict[str, Any]:
    raw["passed"] = True
    raw["failure_reasons"] = []
    return raw


def test_latency_regression_cannot_be_hidden_by_forging_pass(tmp_path: Path) -> None:
    base = [ModelSample(prompt_id="p", output="{}", latency_ms=10)]
    cand = [ModelSample(prompt_id="p", output="{}", latency_ms=100)]
    raw = DriftBenchmarkEngine().evaluate(base, cand).to_dict()
    assert not raw["passed"] and raw["drifted_samples"] == 0
    load_report(_write(tmp_path, raw))
    with pytest.raises(ValueError, match="latency regression"):
        load_report(_write(tmp_path, _forge_pass(raw)))


def test_low_conformity_cannot_be_hidden_by_forging_pass(tmp_path: Path) -> None:
    case = [PromptCase("p", "", SCHEMA)]
    broken = [ModelSample(prompt_id="p", output='{"n": "one"}')]
    raw = DriftBenchmarkEngine().evaluate(broken, broken, case).to_dict()
    assert not raw["passed"] and raw["drifted_samples"] == 0
    load_report(_write(tmp_path, raw))
    with pytest.raises(ValueError, match="conformity is below"):
        load_report(_write(tmp_path, _forge_pass(raw)))


def test_missing_latency_zero_baseline_and_older_reports_still_load(tmp_path: Path) -> None:
    zero = DriftBenchmarkEngine().evaluate(
        [ModelSample(prompt_id="p", output="{}", latency_ms=0)],
        [ModelSample(prompt_id="p", output="{}", latency_ms=500)],
    ).to_dict()
    assert zero["passed"]
    assert load_report(_write(tmp_path, zero)).passed
    older = _report(latency=False)
    older.pop("thresholds")
    assert load_report(_write(tmp_path, older)).thresholds is None
