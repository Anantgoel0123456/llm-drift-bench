"""Deterministic stratified prompt suite subsampling and lockstep capture slicing."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from llm_drift_bench.models import ModelSample, PromptCase


def sample_prompt_suite(
    cases: Sequence[PromptCase],
    *,
    size: int,
    seed: int = 0,
    stratify_by_category: bool = True,
) -> list[PromptCase]:
    """Deterministically draw a subset of PromptCases (optionally stratified by category)."""
    if not cases:
        raise ValueError("Prompt suite cannot be empty.")
    if len({c.prompt_id for c in cases}) != len(cases):
        raise ValueError("Prompt suite contains duplicate prompt_id entries.")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer.")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer, not a boolean.")
    if size > len(cases):
        raise ValueError(f"Requested sample size {size} exceeds suite size {len(cases)}.")
    if size == len(cases):
        return sorted(cases, key=lambda c: c.prompt_id)

    rng = random.Random(seed)
    ordered = sorted(cases, key=lambda c: c.prompt_id)
    if not stratify_by_category:
        chosen = rng.sample(ordered, size)
        return sorted(chosen, key=lambda c: c.prompt_id)

    buckets: dict[str, list[PromptCase]] = defaultdict(list)
    for case in ordered:
        buckets[case.category].append(case)
    categories = sorted(buckets)
    total = len(ordered)

    # Compute largest-remainder proportional quotas with >=1 floor when size >= len(categories).
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    if size >= len(categories):
        for cat in categories:
            exact = (len(buckets[cat]) / total) * size
            base = max(1, min(len(buckets[cat]), int(exact)))
            quotas[cat] = base
            remainders.append((exact - base, cat))
    else:
        for cat in categories:
            quotas[cat] = 0
            remainders.append((len(buckets[cat]) / total, cat))

    allocated = sum(quotas.values())
    if allocated > size:
        # Trim largest allocations above 1 first.
        for cat in sorted(categories, key=lambda c: (-quotas[c], c)):
            while allocated > size and quotas[cat] > 1:
                quotas[cat] -= 1
                allocated -= 1
    elif allocated < size:
        for _, cat in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if allocated == size:
                break
            if quotas[cat] < len(buckets[cat]):
                quotas[cat] += 1
                allocated += 1

    selected: list[PromptCase] = []
    for cat in categories:
        k = quotas[cat]
        if k > 0:
            selected.extend(rng.sample(buckets[cat], k))
    return sorted(selected, key=lambda c: c.prompt_id)


def slice_samples_for_suite(
    samples: Sequence[ModelSample],
    cases: Sequence[PromptCase],
) -> list[ModelSample]:
    """Return samples matching the prompt IDs in `cases`, sorted by prompt_id."""
    by_id = {sample.prompt_id: sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError("Capture contains duplicate prompt_id entries.")
    wanted = sorted({case.prompt_id for case in cases})
    missing = [pid for pid in wanted if pid not in by_id]
    if missing:
        raise ValueError(f"Capture is missing prompt_id(s) required by suite: {missing[:5]!r}.")
    return [by_id[pid] for pid in wanted]


def filter_prompt_suite(
    cases: Sequence[PromptCase],
    *,
    categories: Sequence[str] | None = None,
    require_schema: bool = False,
    exclude_ids: Sequence[str] | None = None,
) -> list[PromptCase]:
    """Filter a PromptCase suite by category, schema presence, and excluded prompt IDs."""
    if not cases:
        raise ValueError("Prompt suite cannot be empty.")
    selected = sorted(cases, key=lambda c: c.prompt_id)
    if categories is not None:
        available = sorted({c.category for c in selected})
        wanted_cats = set(categories)
        unknown = sorted(wanted_cats - set(available))
        if unknown:
            raise ValueError(
                f"Unknown category filter(s): {unknown!r}. Available: {available!r}."
            )
        selected = [c for c in selected if c.category in wanted_cats]
    if require_schema:
        selected = [c for c in selected if c.schema is not None]
    if exclude_ids is not None:
        excluded = set(exclude_ids)
        selected = [c for c in selected if c.prompt_id not in excluded]
    return selected
