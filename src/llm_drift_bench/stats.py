"""Statistical analysis: bootstrap CIs, permutation tests, and multi-run variance."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import Any

from llm_drift_bench._matching import index_records, require_same_ids
from llm_drift_bench._numbers import finite_number
from llm_drift_bench.metrics import compute_percentile
from llm_drift_bench.models import DriftBenchmarkReport


@dataclass(frozen=True)
class BootstrapCI:
    """Non-parametric percentile bootstrap confidence interval for a sample mean."""

    mean: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "confidence_level": self.confidence_level,
            "resamples": self.resamples,
            "sample_size": self.sample_size,
        }


@dataclass(frozen=True)
class PairedSignificance:
    """Two-sided paired sign-flip permutation test comparing candidate B vs candidate A."""

    observed_diff: float
    p_value: float
    alpha: float
    significant: bool
    permutations: int
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_diff": round(self.observed_diff, 4),
            "p_value": round(self.p_value, 6),
            "alpha": self.alpha,
            "significant": self.significant,
            "permutations": self.permutations,
            "sample_size": self.sample_size,
        }


@dataclass(frozen=True)
class McNemarResult:
    """Exact two-sided McNemar test on paired binary outcomes (e.g., pass/fail or conformity)."""

    b_regressed: int
    c_recovered: int
    p_value: float
    alpha: float
    significant: bool
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "b_regressed": self.b_regressed,
            "c_recovered": self.c_recovered,
            "p_value": round(self.p_value, 6),
            "alpha": self.alpha,
            "significant": self.significant,
            "sample_size": self.sample_size,
        }


def mcnemar_exact_test(
    passed_a: Sequence[bool],
    passed_b: Sequence[bool],
    *,
    alpha: float = 0.05,
) -> McNemarResult:
    """Compute exact two-sided McNemar binomial p-value for paired binary pass/fail outcomes.

    b_regressed counts items that passed in A and failed in B;
    c_recovered counts items that failed in A and passed in B.
    """
    if not passed_a or len(passed_a) != len(passed_b):
        raise ValueError("passed_a and passed_b must be non-empty sequences of equal length.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1.")

    b_reg = sum(1 for a, b in zip(passed_a, passed_b, strict=True) if a and not b)
    c_rec = sum(1 for a, b in zip(passed_a, passed_b, strict=True) if not a and b)
    discordant = b_reg + c_rec
    if discordant == 0:
        return McNemarResult(0, 0, 1.0, alpha, False, len(passed_a))

    tail = min(b_reg, c_rec)
    denom = 2**discordant
    one_sided = sum(math.comb(discordant, k) for k in range(tail + 1)) / denom
    p_val = min(1.0, 2.0 * one_sided)
    return McNemarResult(
        b_regressed=b_reg,
        c_recovered=c_rec,
        p_value=p_val,
        alpha=alpha,
        significant=p_val < alpha,
        sample_size=len(passed_a),
    )


@dataclass(frozen=True)
class PromptStability:
    """Per-prompt drift variance and flakiness across repeated runs."""

    prompt_id: str
    runs: int
    drifted_runs: int
    drift_rate: float
    mean_drift: float
    std_drift: float
    min_drift: float
    max_drift: float
    conformity_rate: float

    @property
    def flaky(self) -> bool:
        """Return True if this prompt flipped between drifting and passing across runs."""
        return 0 < self.drifted_runs < self.runs


@dataclass(frozen=True)
class MultiRunSummary:
    """Aggregate variance, confidence intervals, and prompt flakiness across repeated runs."""

    baseline_model: str
    candidate_model: str
    runs: int
    prompts: int
    mean_drift_ci: BootstrapCI
    conformity_ci: BootstrapCI
    flaky_prompts: tuple[str, ...]
    prompt_stability: tuple[PromptStability, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "runs": self.runs,
            "prompts": self.prompts,
            "mean_drift_ci": self.mean_drift_ci.to_dict(),
            "conformity_ci": self.conformity_ci.to_dict(),
            "flaky_prompts": list(self.flaky_prompts),
            "prompt_stability": [
                {**asdict(item), "flaky": item.flaky} for item in self.prompt_stability
            ],
        }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> BootstrapCI:
    """Compute a deterministic percentile bootstrap confidence interval for the mean."""
    if not values:
        raise ValueError("values cannot be empty.")
    cleaned = [finite_number(v) for v in values]
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie strictly between 0 and 1.")
    if isinstance(n_resamples, bool) or n_resamples < 10:
        raise ValueError("n_resamples must be an integer >= 10.")

    n = len(cleaned)
    observed_mean = fmean(cleaned)
    if n == 1:
        return BootstrapCI(observed_mean, observed_mean, observed_mean, confidence, n_resamples, 1)

    rng = random.Random(seed)
    boot_means = [
        fmean(cleaned[rng.randrange(n)] for _ in range(n))
        for _ in range(n_resamples)
    ]
    alpha = (1.0 - confidence) / 2.0
    lower = compute_percentile(boot_means, alpha * 100.0)
    upper = compute_percentile(boot_means, (1.0 - alpha) * 100.0)
    assert lower is not None and upper is not None
    return BootstrapCI(
        mean=observed_mean,
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        resamples=n_resamples,
        sample_size=n,
    )


def paired_permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    alpha: float = 0.05,
    n_permutations: int = 2000,
    seed: int = 0,
) -> PairedSignificance:
    """Run a two-sided paired sign-flip permutation test on mean(scores_b - scores_a)."""
    if not scores_a or len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must be non-empty sequences of equal length.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if isinstance(n_permutations, bool) or n_permutations < 10:
        raise ValueError("n_permutations must be an integer >= 10.")

    diffs = [finite_number(b) - finite_number(a) for a, b in zip(scores_a, scores_b, strict=True)]
    observed = fmean(diffs)
    abs_obs = abs(observed)
    if math.isclose(abs_obs, 0.0, abs_tol=1e-12):
        return PairedSignificance(0.0, 1.0, alpha, False, n_permutations, len(diffs))

    rng = random.Random(seed)
    extreme = 0
    n = len(diffs)
    for _ in range(n_permutations):
        signed_sum = sum(d if rng.getrandbits(1) else -d for d in diffs)
        if abs(signed_sum / n) >= abs_obs - 1e-12:
            extreme += 1

    p_value = (extreme + 1) / (n_permutations + 1)
    return PairedSignificance(
        observed_diff=observed,
        p_value=p_value,
        alpha=alpha,
        significant=p_value < alpha,
        permutations=n_permutations,
        sample_size=n,
    )


def summarize_multi_run(
    reports: Sequence[DriftBenchmarkReport],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> MultiRunSummary:
    """Aggregate repeated runs of a candidate to measure stability, variance, and CIs."""
    if not reports:
        raise ValueError("Provide at least one engine report.")
    reference = reports[0]
    expected_ids = set(index_records(reference.results, "reference report"))
    indexed_runs = []
    for index, report in enumerate(reports, start=1):
        if report.baseline_model != reference.baseline_model:
            raise ValueError("All runs must share the same baseline model.")
        mapping = index_records(report.results, f"run {index}")
        require_same_ids(expected_ids, set(mapping), f"Run {index}")
        indexed_runs.append(mapping)

    prompt_stats: list[PromptStability] = []
    prompt_mean_drifts: list[float] = []
    prompt_conformities: list[float] = []
    flaky: list[str] = []

    run_count = len(indexed_runs)
    for pid in sorted(expected_ids):
        rows = [run[pid] for run in indexed_runs]
        drifts = [row.composite_drift_score for row in rows]
        drifted_runs = sum(1 for row in rows if row.drifted)
        conformant = sum(1 for row in rows if row.candidate_validation.passed) / run_count
        mean_d = fmean(drifts)
        std_d = pstdev(drifts) if run_count > 1 else 0.0
        item = PromptStability(
            prompt_id=pid,
            runs=run_count,
            drifted_runs=drifted_runs,
            drift_rate=round(drifted_runs / run_count, 4),
            mean_drift=round(mean_d, 4),
            std_drift=round(std_d, 4),
            min_drift=round(min(drifts), 4),
            max_drift=round(max(drifts), 4),
            conformity_rate=round(conformant, 4),
        )
        prompt_stats.append(item)
        prompt_mean_drifts.append(mean_d)
        prompt_conformities.append(conformant)
        if item.flaky:
            flaky.append(pid)

    return MultiRunSummary(
        baseline_model=reference.baseline_model,
        candidate_model=reference.candidate_model,
        runs=run_count,
        prompts=len(expected_ids),
        mean_drift_ci=bootstrap_mean_ci(
            prompt_mean_drifts, confidence=confidence, n_resamples=n_resamples, seed=seed
        ),
        conformity_ci=bootstrap_mean_ci(
            prompt_conformities, confidence=confidence, n_resamples=n_resamples, seed=seed
        ),
        flaky_prompts=tuple(flaky),
        prompt_stability=tuple(prompt_stats),
    )
