"""Explicit token and output-cost drift estimation (never fabricates token counts)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

from llm_drift_bench._matching import require_same_ids
from llm_drift_bench._numbers import finite_number
from llm_drift_bench.models import ModelSample


@dataclass(frozen=True)
class ModelPricing:
    """Explicit USD-per-1M-tokens pricing supplied by the caller."""

    output_per_million_usd: float
    input_per_million_usd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("output_per_million_usd", "input_per_million_usd"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number.")
            num = finite_number(val)
            if num < 0:
                raise ValueError(f"{name} must be a finite non-negative number.")


@dataclass(frozen=True)
class CostComparison:
    """Token volume and explicit output-cost comparison between baseline and candidate."""

    total_samples: int
    complete: bool
    baseline_tokens: int | None
    candidate_tokens: int | None
    token_delta: int | None
    token_delta_pct: float | None
    baseline_output_cost_usd: float | None
    candidate_output_cost_usd: float | None
    cost_delta_usd: float | None
    cost_delta_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_cost_drift(
    baseline_samples: Sequence[ModelSample],
    candidate_samples: Sequence[ModelSample],
    *,
    baseline_pricing: ModelPricing | None = None,
    candidate_pricing: ModelPricing | None = None,
) -> CostComparison:
    """Compare explicit generated token counts and optional USD output costs.

    If any sample lacks an explicit `tokens_generated` measurement, token and cost
    totals remain None (complete=False) rather than estimating tokens from text.
    """
    if not baseline_samples or not candidate_samples:
        raise ValueError("Baseline and candidate captures must both be non-empty.")
    base_map = {s.prompt_id: s for s in baseline_samples}
    cand_map = {s.prompt_id: s for s in candidate_samples}
    if len(base_map) != len(baseline_samples) or len(cand_map) != len(candidate_samples):
        raise ValueError("Captures contain duplicate prompt_id entries.")
    require_same_ids(set(base_map), set(cand_map), "Candidate capture")

    base_complete = all(s.tokens_generated is not None for s in baseline_samples)
    cand_complete = all(s.tokens_generated is not None for s in candidate_samples)
    if not (base_complete and cand_complete):
        return CostComparison(
            total_samples=len(base_map),
            complete=False,
            baseline_tokens=None,
            candidate_tokens=None,
            token_delta=None,
            token_delta_pct=None,
            baseline_output_cost_usd=None,
            candidate_output_cost_usd=None,
            cost_delta_usd=None,
            cost_delta_pct=None,
        )

    b_toks = sum(s.tokens_generated for s in baseline_samples if s.tokens_generated is not None)
    c_toks = sum(s.tokens_generated for s in candidate_samples if s.tokens_generated is not None)
    tok_delta = c_toks - b_toks
    tok_pct = round((tok_delta / b_toks) * 100.0, 2) if b_toks > 0 else None

    b_cost = (
        round((b_toks / 1_000_000.0) * baseline_pricing.output_per_million_usd, 6)
        if baseline_pricing is not None
        else None
    )
    c_cost = (
        round((c_toks / 1_000_000.0) * candidate_pricing.output_per_million_usd, 6)
        if candidate_pricing is not None
        else None
    )
    cost_delta = round(c_cost - b_cost, 6) if (b_cost is not None and c_cost is not None) else None
    cost_pct = (
        round(((c_cost - b_cost) / b_cost) * 100.0, 2)
        if (b_cost is not None and c_cost is not None and b_cost > 0)
        else None
    )
    return CostComparison(
        total_samples=len(base_map),
        complete=True,
        baseline_tokens=b_toks,
        candidate_tokens=c_toks,
        token_delta=tok_delta,
        token_delta_pct=tok_pct,
        baseline_output_cost_usd=b_cost,
        candidate_output_cost_usd=c_cost,
        cost_delta_usd=cost_delta,
        cost_delta_pct=cost_pct,
    )


def check_cost_budget(
    comparison: CostComparison,
    *,
    max_token_increase_pct: float | None = None,
    max_cost_increase_pct: float | None = None,
) -> tuple[bool, str]:
    """Check whether a CostComparison stays within explicit token/cost increase budgets."""
    if not isinstance(comparison, CostComparison):
        raise TypeError("check_cost_budget requires a CostComparison instance.")
    for name, limit in (
        ("max_token_increase_pct", max_token_increase_pct),
        ("max_cost_increase_pct", max_cost_increase_pct),
    ):
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number.")
            if finite_number(limit) < 0:
                raise ValueError(f"{name} must be a finite non-negative number.")

    if max_token_increase_pct is None and max_cost_increase_pct is None:
        return True, "No token or cost budget thresholds configured."
    if not comparison.complete:
        return False, "Cannot verify budget: explicit tokens_generated measurements are incomplete."

    failures: list[str] = []
    if max_token_increase_pct is not None:
        if comparison.token_delta_pct is None:
            failures.append("Token percentage delta unavailable (baseline tokens = 0).")
        elif comparison.token_delta_pct > max_token_increase_pct:
            failures.append(
                f"Output tokens increased by {comparison.token_delta_pct:+.2f}% "
                f"(limit +{max_token_increase_pct:.2f}%)."
            )
    if max_cost_increase_pct is not None:
        if comparison.cost_delta_pct is None:
            failures.append(
                "Cost percentage delta unavailable (supply non-zero baseline and candidate prices)."
            )
        elif comparison.cost_delta_pct > max_cost_increase_pct:
            failures.append(
                f"Output cost increased by {comparison.cost_delta_pct:+.2f}% "
                f"(limit +{max_cost_increase_pct:.2f}%)."
            )
    if failures:
        return False, "; ".join(failures)
    return True, "Token and cost changes stayed within configured budgets."


def rank_candidate_costs(
    baseline_samples: Sequence[ModelSample],
    candidates: Mapping[str, Sequence[ModelSample]],
    *,
    baseline_pricing: ModelPricing | None = None,
    candidate_pricing: Mapping[str, ModelPricing] | None = None,
) -> list[dict[str, Any]]:
    """Compare and rank multiple candidates by explicit output cost (or token delta)."""
    if not candidates:
        raise ValueError("Provide at least one candidate capture to rank.")
    pricing_map = candidate_pricing or {}
    rows: list[dict[str, Any]] = []
    for label, samples in candidates.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Candidate labels must be non-empty strings.")
        comp = estimate_cost_drift(
            baseline_samples,
            samples,
            baseline_pricing=baseline_pricing,
            candidate_pricing=pricing_map.get(label),
        )
        rows.append({"candidate": label, **comp.to_dict()})

    def _sort_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
        if not row["complete"]:
            return (2, float("inf"), 0, str(row["candidate"]))
        if row["candidate_output_cost_usd"] is not None:
            return (0, float(row["candidate_output_cost_usd"]), int(row["candidate_tokens"]),
                    str(row["candidate"]))
        return (1, float(row["candidate_tokens"]), 0, str(row["candidate"]))

    rows.sort(key=_sort_key)
    return rows


def _fmt_usd(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"${value:+.6f}" if signed else f"${value:.6f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"


def format_cost_text(
    comparison: CostComparison,
    *,
    budget_ok: bool | None = None,
    budget_detail: str | None = None,
) -> str:
    """Format a CostComparison as human-readable text without fabricating missing counts."""
    if not comparison.complete:
        lines = [
            f"Token & Cost Drift ({comparison.total_samples} sample(s)): "
            "n/a (incomplete explicit tokens_generated measurements)."
        ]
    else:
        pct_s = _fmt_pct(comparison.token_delta_pct)
        delta_s = "n/a" if comparison.token_delta is None else f"{comparison.token_delta:+d}"
        lines = [
            f"Token & Cost Drift ({comparison.total_samples} sample(s))",
            f"  Output tokens : baseline={comparison.baseline_tokens}, "
            f"candidate={comparison.candidate_tokens} (delta={delta_s}, {pct_s})",
        ]
        if (
            comparison.baseline_output_cost_usd is not None
            or comparison.candidate_output_cost_usd is not None
        ):
            b_s = _fmt_usd(comparison.baseline_output_cost_usd)
            c_s = _fmt_usd(comparison.candidate_output_cost_usd)
            d_s = _fmt_usd(comparison.cost_delta_usd, signed=True)
            lines.append(f"  Output cost   : baseline={b_s}, candidate={c_s} (delta={d_s})")
    if budget_detail is not None:
        tag = "PASS" if budget_ok else "FAIL"
        lines.append(f"  Budget check  : {tag} ({budget_detail})")
    return "\n".join(lines) + "\n"


def format_cost_markdown(
    comparison: CostComparison,
    *,
    budget_ok: bool | None = None,
    budget_detail: str | None = None,
) -> str:
    """Format a CostComparison as GitHub-Flavored Markdown."""
    status = "Complete" if comparison.complete else "Incomplete"
    b_tok = "n/a" if comparison.baseline_tokens is None else str(comparison.baseline_tokens)
    c_tok = "n/a" if comparison.candidate_tokens is None else str(comparison.candidate_tokens)
    d_tok = "n/a" if comparison.token_delta is None else f"{comparison.token_delta:+d}"
    lines = [
        f"## Token & Cost Drift ({status})",
        "",
        f"Samples evaluated: **{comparison.total_samples}**",
        "",
        "| Metric | Baseline | Candidate | Delta | Delta % |",
        "| :--- | ---: | ---: | ---: | ---: |",
        f"| Output tokens | {b_tok} | {c_tok} | {d_tok} | "
        f"{_fmt_pct(comparison.token_delta_pct)} |",
        f"| Output cost (USD) | {_fmt_usd(comparison.baseline_output_cost_usd)} | "
        f"{_fmt_usd(comparison.candidate_output_cost_usd)} | "
        f"{_fmt_usd(comparison.cost_delta_usd, signed=True)} | "
        f"{_fmt_pct(comparison.cost_delta_pct)} |",
    ]
    if budget_detail is not None:
        tag = "PASS" if budget_ok else "FAIL"
        lines.extend(["", f"Budget check: **{tag}** — {budget_detail}"])
    return "\n".join(lines) + "\n"


def export_cost_html(
    comparison: CostComparison,
    output_path: str | Path | None = None,
    *,
    budget_ok: bool | None = None,
    budget_detail: str | None = None,
) -> str:
    """Export a standalone script-free offline HTML report for a CostComparison."""
    if budget_ok is False:
        badge_text, badge_bg = "BUDGET FAIL", "#d73a49"
    elif not comparison.complete:
        badge_text, badge_bg = "INCOMPLETE", "#b45309"
    else:
        badge_text, badge_bg = "COMPLETE", "#2ea44f"

    b_tok = "n/a" if comparison.baseline_tokens is None else str(comparison.baseline_tokens)
    c_tok = "n/a" if comparison.candidate_tokens is None else str(comparison.candidate_tokens)
    d_tok = "n/a" if comparison.token_delta is None else f"{comparison.token_delta:+d}"
    budget_html = ""
    if budget_detail is not None:
        tag = "PASS" if budget_ok else "FAIL"
        budget_html = f"<p><strong>Budget check ({tag}):</strong> {escape(budget_detail)}</p>"

    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta http-equiv='Content-Security-Policy' "
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>LLM Drift Bench — Token &amp; Cost Drift</title>"
        "<style>"
        "body{font-family:system-ui,-apple-system,sans-serif;max-width:860px;"
        "margin:2rem auto;padding:0 1rem;color:#1f2937}"
        f".badge{{display:inline-block;padding:0.2rem 0.65rem;border-radius:999px;"
        f"color:#fff;font-weight:600;background:{badge_bg}}}"
        "table{width:100%;border-collapse:collapse;margin-top:1rem}"
        "th,td{border:1px solid #e5e7eb;padding:0.5rem 0.75rem;text-align:left}"
        "th{background:#f9fafb}"
        "</style></head><body>"
        f"<h1>Token &amp; Cost Drift <span class='badge'>{badge_text}</span></h1>"
        f"<p>Samples evaluated: <strong>{comparison.total_samples}</strong></p>"
        "<table><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th>"
        "<th>Delta</th><th>Delta %</th></tr></thead><tbody>"
        f"<tr><td>Output tokens</td><td>{b_tok}</td><td>{c_tok}</td>"
        f"<td>{d_tok}</td><td>{_fmt_pct(comparison.token_delta_pct)}</td></tr>"
        f"<tr><td>Output cost (USD)</td>"
        f"<td>{_fmt_usd(comparison.baseline_output_cost_usd)}</td>"
        f"<td>{_fmt_usd(comparison.candidate_output_cost_usd)}</td>"
        f"<td>{_fmt_usd(comparison.cost_delta_usd, signed=True)}</td>"
        f"<td>{_fmt_pct(comparison.cost_delta_pct)}</td></tr>"
        f"</tbody></table>{budget_html}</body></html>\n"
    )
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
    return html
