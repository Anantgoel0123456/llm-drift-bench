"""Optional top_logprobs token entropy and Jensen-Shannon divergence drift analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llm_drift_bench._matching import index_records, require_same_ids
from llm_drift_bench.models import ModelSample

_TAIL_BUCKET = "\x00<other>"
_LOGPROB_EPS = 1e-6


def _as_valid_logprob(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value <= _LOGPROB_EPS
    ):
        return min(0.0, float(value))
    return None


def _parse_token_distributions(
    sample: ModelSample,
) -> tuple[list[float], list[dict[str, float]]] | None:
    """Extract chosen-token logprobs and top-k (+ tail) probability distributions."""
    raw = sample.metadata.get("logprobs")
    if not isinstance(raw, list) or not raw:
        return None
    chosen_logprobs: list[float] = []
    distributions: list[dict[str, float]] = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        token = item.get("token")
        lp = _as_valid_logprob(item.get("logprob"))
        if not isinstance(token, str) or lp is None:
            continue

        chosen_logprobs.append(lp)
        probs: dict[str, float] = {token: math.exp(lp)}
        raw_top = item.get("top_logprobs")
        if isinstance(raw_top, list):
            for cand in raw_top:
                if not isinstance(cand, dict):
                    continue
                cand_tok = cand.get("token")
                cand_lp = _as_valid_logprob(cand.get("logprob"))
                if isinstance(cand_tok, str) and cand_lp is not None:
                    p = math.exp(cand_lp)
                    if p > probs.get(cand_tok, 0.0):
                        probs[cand_tok] = p

        mass = sum(probs.values())
        if mass > 1.0:
            probs = {tok: p / mass for tok, p in probs.items()}
        else:
            tail = 1.0 - mass
            if tail > 1e-12:
                probs[_TAIL_BUCKET] = tail
        distributions.append(probs)

    if not chosen_logprobs:
        return None
    return chosen_logprobs, distributions


def _shannon_entropy_bits(dist: dict[str, float]) -> float:
    return max(0.0, -sum(p * math.log2(p) for p in dist.values() if p > 0.0))


def _jsd_bits(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Base-2 Jensen-Shannon divergence bounded in [0.0, 1.0]."""
    kl_a = 0.0
    kl_b = 0.0
    for tok in set(dist_a) | set(dist_b):
        p = dist_a.get(tok, 0.0)
        q = dist_b.get(tok, 0.0)
        m = 0.5 * (p + q)
        if p > 0.0:
            kl_a += p * math.log2(p / m)
        if q > 0.0:
            kl_b += q * math.log2(q / m)
    return max(0.0, min(1.0, 0.5 * (kl_a + kl_b)))


def _round_opt(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(frozen=True)
class PromptLogprobDrift:
    """Per-prompt token uncertainty and top-k distribution divergence summary."""

    prompt_id: str
    available: bool
    baseline_tokens: int
    candidate_tokens: int
    baseline_mean_entropy_bits: float | None
    candidate_mean_entropy_bits: float | None
    entropy_delta_bits: float | None
    baseline_mean_logprob: float | None
    candidate_mean_logprob: float | None
    logprob_delta: float | None
    aligned_positions: int
    mean_top_k_jsd: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "available": self.available,
            "baseline_tokens": self.baseline_tokens,
            "candidate_tokens": self.candidate_tokens,
            "baseline_mean_entropy_bits": _round_opt(self.baseline_mean_entropy_bits),
            "candidate_mean_entropy_bits": _round_opt(self.candidate_mean_entropy_bits),
            "entropy_delta_bits": _round_opt(self.entropy_delta_bits),
            "baseline_mean_logprob": _round_opt(self.baseline_mean_logprob),
            "candidate_mean_logprob": _round_opt(self.candidate_mean_logprob),
            "logprob_delta": _round_opt(self.logprob_delta),
            "aligned_positions": self.aligned_positions,
            "mean_top_k_jsd": _round_opt(self.mean_top_k_jsd),
        }


@dataclass(frozen=True)
class LogprobDriftSummary:
    """Aggregate logprobs uncertainty and JS divergence summary across a capture pair."""

    total_samples: int
    comparable_samples: int
    complete: bool
    mean_baseline_entropy_bits: float | None
    mean_candidate_entropy_bits: float | None
    mean_entropy_delta_bits: float | None
    mean_baseline_logprob: float | None
    mean_candidate_logprob: float | None
    mean_logprob_delta: float | None
    mean_top_k_jsd: float | None
    max_top_k_jsd: float | None
    results: tuple[PromptLogprobDrift, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "comparable_samples": self.comparable_samples,
            "complete": self.complete,
            "mean_baseline_entropy_bits": _round_opt(self.mean_baseline_entropy_bits),
            "mean_candidate_entropy_bits": _round_opt(self.mean_candidate_entropy_bits),
            "mean_entropy_delta_bits": _round_opt(self.mean_entropy_delta_bits),
            "mean_baseline_logprob": _round_opt(self.mean_baseline_logprob),
            "mean_candidate_logprob": _round_opt(self.mean_candidate_logprob),
            "mean_logprob_delta": _round_opt(self.mean_logprob_delta),
            "mean_top_k_jsd": _round_opt(self.mean_top_k_jsd),
            "max_top_k_jsd": _round_opt(self.max_top_k_jsd),
            "results": [item.to_dict() for item in self.results],
        }


def analyze_logprobs_drift(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
) -> LogprobDriftSummary:
    """Compare token entropy and position-aligned top-k JSD from metadata['logprobs'].

    Detects probability-distribution and uncertainty shifts even before greedy decoding
    flips a chosen token. Samples lacking valid logprobs remain marked unavailable
    (complete=False) instead of fabricating values.
    """
    if not baseline_samples or not candidate_samples:
        raise ValueError("Both baseline and candidate sample collections must be non-empty.")

    base_map = index_records(baseline_samples, "baseline samples")
    cand_map = index_records(candidate_samples, "candidate samples")
    require_same_ids(set(base_map), set(cand_map), "Candidate")

    results: list[PromptLogprobDrift] = []
    base_entropies: list[float] = []
    cand_entropies: list[float] = []
    ent_deltas: list[float] = []
    base_lps: list[float] = []
    cand_lps: list[float] = []
    lp_deltas: list[float] = []
    jsds: list[float] = []

    for prompt_id in sorted(base_map):
        parsed_b = _parse_token_distributions(base_map[prompt_id])
        parsed_c = _parse_token_distributions(cand_map[prompt_id])
        if parsed_b is None or parsed_c is None:
            results.append(
                PromptLogprobDrift(
                    prompt_id=prompt_id,
                    available=False,
                    baseline_tokens=0 if parsed_b is None else len(parsed_b[0]),
                    candidate_tokens=0 if parsed_c is None else len(parsed_c[0]),
                    baseline_mean_entropy_bits=None,
                    candidate_mean_entropy_bits=None,
                    entropy_delta_bits=None,
                    baseline_mean_logprob=None,
                    candidate_mean_logprob=None,
                    logprob_delta=None,
                    aligned_positions=0,
                    mean_top_k_jsd=None,
                )
            )
            continue

        b_chosen, b_dists = parsed_b
        c_chosen, c_dists = parsed_c
        b_ent = sum(_shannon_entropy_bits(d) for d in b_dists) / len(b_dists)
        c_ent = sum(_shannon_entropy_bits(d) for d in c_dists) / len(c_dists)
        ent_delta = c_ent - b_ent
        b_lp = sum(b_chosen) / len(b_chosen)
        c_lp = sum(c_chosen) / len(c_chosen)
        lp_delta = c_lp - b_lp
        aligned = min(len(b_dists), len(c_dists))
        mean_jsd = (
            sum(_jsd_bits(b_dists[i], c_dists[i]) for i in range(aligned)) / aligned
            if aligned > 0
            else None
        )

        base_entropies.append(b_ent)
        cand_entropies.append(c_ent)
        ent_deltas.append(ent_delta)
        base_lps.append(b_lp)
        cand_lps.append(c_lp)
        lp_deltas.append(lp_delta)
        if mean_jsd is not None:
            jsds.append(mean_jsd)

        results.append(
            PromptLogprobDrift(
                prompt_id=prompt_id,
                available=True,
                baseline_tokens=len(b_chosen),
                candidate_tokens=len(c_chosen),
                baseline_mean_entropy_bits=b_ent,
                candidate_mean_entropy_bits=c_ent,
                entropy_delta_bits=ent_delta,
                baseline_mean_logprob=b_lp,
                candidate_mean_logprob=c_lp,
                logprob_delta=lp_delta,
                aligned_positions=aligned,
                mean_top_k_jsd=mean_jsd,
            )
        )

    comparable = len(base_entropies)
    total = len(results)

    def _avg(values: list[float]) -> float | None:
        return (sum(values) / len(values)) if values else None

    return LogprobDriftSummary(
        total_samples=total,
        comparable_samples=comparable,
        complete=(comparable == total and total > 0),
        mean_baseline_entropy_bits=_avg(base_entropies),
        mean_candidate_entropy_bits=_avg(cand_entropies),
        mean_entropy_delta_bits=_avg(ent_deltas),
        mean_baseline_logprob=_avg(base_lps),
        mean_candidate_logprob=_avg(cand_lps),
        mean_logprob_delta=_avg(lp_deltas),
        mean_top_k_jsd=_avg(jsds),
        max_top_k_jsd=max(jsds) if jsds else None,
        results=tuple(results),
    )


def format_logprobs_text(summary: LogprobDriftSummary) -> str:
    """Format a LogprobDriftSummary as a clear human-readable text report."""
    def _fmt(val: float | None, spec: str = ".4f") -> str:
        return "n/a" if val is None else format(val, spec)

    status = "complete" if summary.complete else "incomplete (missing logprobs on some prompts)"
    lines = [
        f"Logprobs Distribution & Uncertainty Drift "
        f"({summary.comparable_samples}/{summary.total_samples} comparable, {status})",
        f"Mean entropy (bits): {_fmt(summary.mean_baseline_entropy_bits)} -> "
        f"{_fmt(summary.mean_candidate_entropy_bits)} "
        f"(delta {_fmt(summary.mean_entropy_delta_bits, '+.4f')})",
        f"Mean chosen logprob: {_fmt(summary.mean_baseline_logprob)} -> "
        f"{_fmt(summary.mean_candidate_logprob)} "
        f"(delta {_fmt(summary.mean_logprob_delta, '+.4f')})",
        f"Position-aligned top-k JSD: mean={_fmt(summary.mean_top_k_jsd)}, "
        f"max={_fmt(summary.max_top_k_jsd)}",
    ]
    if summary.results:
        lines.append("")
        lines.append("Per-prompt breakdown:")
        for item in summary.results:
            if not item.available:
                lines.append(f"  {item.prompt_id}: unavailable (logprobs missing)")
            else:
                lines.append(
                    f"  {item.prompt_id}: entropy "
                    f"{_fmt(item.baseline_mean_entropy_bits)}->"
                    f"{_fmt(item.candidate_mean_entropy_bits)} "
                    f"({_fmt(item.entropy_delta_bits, '+.4f')} bits), "
                    f"top_k_jsd={_fmt(item.mean_top_k_jsd)} "
                    f"(aligned={item.aligned_positions} toks)"
                )
    return "\n".join(lines) + "\n"
