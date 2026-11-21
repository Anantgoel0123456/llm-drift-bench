"""Core benchmark and drift evaluation engine for llm-drift-bench."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_drift_bench._file_io import parse_source_json, read_utf8
from llm_drift_bench._matching import index_records, require_same_ids
from llm_drift_bench.metrics import (
    compute_composite_drift_score,
    compute_percentile,
    jaccard_similarity,
    json_exact_field_match,
    json_numeric_drift,
    json_structural_similarity,
    ngram_overlap_similarity,
    normalized_levenshtein_similarity,
)
from llm_drift_bench.models import (
    DriftBenchmarkReport,
    DriftThresholds,
    ModelSample,
    PromptCase,
    SampleDriftResult,
)
from llm_drift_bench.validator import validate_output


def evaluate_sample_pair(
    baseline: ModelSample,
    candidate: ModelSample,
    prompt_case: PromptCase | None = None,
    thresholds: DriftThresholds | None = None,
) -> SampleDriftResult:
    """Compare a baseline ModelSample and candidate ModelSample for a single prompt_id.

    Args:
        baseline: Baseline model completion sample.
        candidate: Candidate model completion sample.
        prompt_case: Optional PromptCase with expected JSON Schema and category.
        thresholds: Optional DriftThresholds configuration.

    Returns:
        SampleDriftResult detailing validation, similarity, and composite drift metrics.
    """
    index_records([baseline], "baseline")
    index_records([candidate], "candidate")
    if baseline.prompt_id != candidate.prompt_id:
        raise ValueError("Baseline and candidate prompt IDs must match.")
    if prompt_case is not None and prompt_case.prompt_id != baseline.prompt_id:
        raise ValueError("Prompt case ID must match the sample prompt ID.")
    active_thresholds = thresholds or DriftThresholds()
    schema = prompt_case.schema if prompt_case is not None else None
    category = prompt_case.category if prompt_case is not None else "general"

    base_val = validate_output(baseline.output, schema=schema)
    cand_val = validate_output(candidate.output, schema=schema)

    tok_jaccard = jaccard_similarity(baseline.output, candidate.output)
    ngram_sim = ngram_overlap_similarity(baseline.output, candidate.output, n=2)
    lev_sim = normalized_levenshtein_similarity(baseline.output, candidate.output)

    struct_sim: float | None = None
    field_match: float | None = None
    num_drift: float | None = None

    if base_val.is_valid_json and cand_val.is_valid_json:
        struct_sim = json_structural_similarity(base_val.parsed_json, cand_val.parsed_json)
        field_match = json_exact_field_match(base_val.parsed_json, cand_val.parsed_json)
        num_drift = json_numeric_drift(base_val.parsed_json, cand_val.parsed_json)

    composite_drift = compute_composite_drift_score(
        token_jaccard=tok_jaccard,
        ngram_sim=ngram_sim,
        levenshtein_sim=lev_sim,
        json_structural_sim=struct_sim,
        json_field_match=field_match,
        json_num_drift=num_drift,
        baseline_schema_passed=base_val.passed,
        candidate_schema_passed=cand_val.passed,
    )

    latency_delta: float | None = None
    if baseline.latency_ms is not None and candidate.latency_ms is not None:
        latency_delta = candidate.latency_ms - baseline.latency_ms

    drifted = (
        composite_drift > active_thresholds.max_composite_drift
        or (base_val.passed and not cand_val.passed)
        or (
            num_drift is not None
            and num_drift > active_thresholds.max_numeric_relative_drift
        )
    )

    return SampleDriftResult(
        prompt_id=baseline.prompt_id,
        category=category,
        baseline_validation=base_val,
        candidate_validation=cand_val,
        token_jaccard_similarity=tok_jaccard,
        ngram_similarity=ngram_sim,
        levenshtein_similarity=lev_sim,
        json_structural_similarity=struct_sim,
        json_exact_field_match=field_match,
        json_numeric_drift=num_drift,
        composite_drift_score=composite_drift,
        latency_delta_ms=latency_delta,
        drifted=drifted,
    )


class DriftBenchmarkEngine:
    """Orchestrates benchmark comparisons between baseline and candidate LLM outputs."""

    def __init__(self, thresholds: DriftThresholds | None = None) -> None:
        self.thresholds = thresholds or DriftThresholds()

    def evaluate(
        self,
        baseline_samples: Sequence[ModelSample],
        candidate_samples: Sequence[ModelSample],
        prompt_cases: Sequence[PromptCase] | None = None,
    ) -> DriftBenchmarkReport:
        """Evaluate drift across matched prompt_ids between baseline and candidate runs."""
        if not baseline_samples:
            raise ValueError("baseline_samples cannot be empty.")
        if not candidate_samples:
            raise ValueError("candidate_samples cannot be empty.")
        if len({sample.model_name for sample in baseline_samples}) > 1:
            raise ValueError("Baseline capture contains multiple model names.")
        if len({sample.model_name for sample in candidate_samples}) > 1:
            raise ValueError("Candidate capture contains multiple model names.")

        base_map = index_records(baseline_samples, "baseline")
        cand_map = index_records(candidate_samples, "candidate")
        require_same_ids(set(base_map), set(cand_map), "Candidate")
        prompt_map: dict[str, PromptCase] = {}
        if prompt_cases is not None:
            if not prompt_cases:
                raise ValueError("An explicitly supplied prompt suite cannot be empty.")
            prompt_map = index_records(prompt_cases, "prompt suite")
            require_same_ids(set(base_map), set(prompt_map), "Prompt suite")

        pair_results: list[SampleDriftResult] = []
        base_latencies: list[float] = []
        cand_latencies: list[float] = []

        for prompt_id in sorted(base_map):
            base_sample = base_map[prompt_id]
            cand_sample = cand_map[prompt_id]
            prompt_case = prompt_map.get(base_sample.prompt_id)
            res = evaluate_sample_pair(
                baseline=base_sample,
                candidate=cand_sample,
                prompt_case=prompt_case,
                thresholds=self.thresholds,
            )
            pair_results.append(res)
            if base_sample.latency_ms is not None:
                base_latencies.append(base_sample.latency_ms)
            if cand_sample.latency_ms is not None:
                cand_latencies.append(cand_sample.latency_ms)

        total = len(pair_results)
        drifted_count = sum(1 for r in pair_results if r.drifted)

        json_expected_results = [
            r
            for r in pair_results
            if (
                (r.prompt_id in prompt_map and prompt_map[r.prompt_id].schema is not None)
                or r.baseline_validation.is_valid_json
                or r.candidate_validation.is_valid_json
            )
        ]
        if json_expected_results:
            json_total = len(json_expected_results)
            base_conformant = (
                sum(1 for r in json_expected_results if r.baseline_validation.passed)
                / json_total
            )
            cand_conformant = (
                sum(1 for r in json_expected_results if r.candidate_validation.passed)
                / json_total
            )
        else:
            base_conformant = 1.0
            cand_conformant = 1.0

        mean_drift = sum(r.composite_drift_score for r in pair_results) / total
        max_drift = max(r.composite_drift_score for r in pair_results)
        mean_jaccard = sum(r.token_jaccard_similarity for r in pair_results) / total
        mean_lev = sum(r.levenshtein_similarity for r in pair_results) / total

        struct_vals = [
            r.json_structural_similarity
            for r in pair_results
            if r.json_structural_similarity is not None
        ]
        mean_struct = (sum(struct_vals) / len(struct_vals)) if struct_vals else None

        num_vals = [
            r.json_numeric_drift for r in pair_results if r.json_numeric_drift is not None
        ]
        mean_num = (sum(num_vals) / len(num_vals)) if num_vals else None

        # A partial timing population is not a suite-level measurement.
        if len(base_latencies) != total:
            base_latencies = []
        if len(cand_latencies) != total:
            cand_latencies = []
        base_p50 = compute_percentile(base_latencies, 50.0)
        cand_p50 = compute_percentile(cand_latencies, 50.0)
        base_p95 = compute_percentile(base_latencies, 95.0)
        cand_p95 = compute_percentile(cand_latencies, 95.0)

        failure_reasons: list[str] = []
        if drifted_count:
            failure_reasons.append(
                f"{drifted_count} of {total} samples failed individual drift checks."
            )
        if mean_drift > self.thresholds.max_composite_drift:
            failure_reasons.append(
                f"Mean composite drift ({mean_drift:.4f}) exceeded threshold "
                f"({self.thresholds.max_composite_drift:.4f})."
            )
        if cand_conformant < self.thresholds.min_json_conformity_rate:
            failure_reasons.append(
                f"Candidate JSON conformity rate ({cand_conformant:.2%}) fell below "
                f"minimum ({self.thresholds.min_json_conformity_rate:.2%})."
            )
        if mean_num is not None and mean_num > self.thresholds.max_numeric_relative_drift:
            failure_reasons.append(
                f"Mean numeric relative drift ({mean_num:.4f}) exceeded threshold "
                f"({self.thresholds.max_numeric_relative_drift:.4f})."
            )
        if (
            base_p95 is not None
            and cand_p95 is not None
            and base_p95 > 0
            and ((cand_p95 - base_p95) / base_p95) * 100.0
            > self.thresholds.max_latency_regression_pct
        ):
            reg_pct = ((cand_p95 - base_p95) / base_p95) * 100.0
            failure_reasons.append(
                f"Candidate P95 latency regression (+{reg_pct:.1f}%) exceeded threshold "
                f"(+{self.thresholds.max_latency_regression_pct:.1f}%)."
            )

        baseline_model = baseline_samples[0].model_name
        candidate_model = candidate_samples[0].model_name

        return DriftBenchmarkReport(
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            total_samples=total,
            drifted_samples=drifted_count,
            baseline_json_conformity_rate=base_conformant,
            candidate_json_conformity_rate=cand_conformant,
            mean_composite_drift=mean_drift,
            max_composite_drift=max_drift,
            mean_token_jaccard=mean_jaccard,
            mean_levenshtein_similarity=mean_lev,
            mean_json_structural_similarity=mean_struct,
            mean_json_numeric_drift=mean_num,
            baseline_p50_latency_ms=base_p50,
            candidate_p50_latency_ms=cand_p50,
            baseline_p95_latency_ms=base_p95,
            candidate_p95_latency_ms=cand_p95,
            passed=len(failure_reasons) == 0,
            failure_reasons=tuple(failure_reasons),
            results=tuple(pair_results),
            thresholds=self.thresholds,
        )


def _record_unit(path: str | Path) -> str:
    return "line" if Path(path).suffix.lower() == ".jsonl" else "record"


def _record_location(path: str | Path, number: int) -> str:
    return f"{path} ({_record_unit(path)} {number})"


def _load_records(
    path: str | Path,
    container_keys: tuple[str, ...],
) -> tuple[list[tuple[int, dict[str, Any]]], str | None]:
    """Read a JSON or JSONL file into a list of dicts and optional top-level model_name."""
    file_path = Path(path)
    raw = read_utf8(file_path)
    if not raw.strip(" \t\r\n"):
        return [], None

    if file_path.suffix.lower() == ".jsonl":
        records: list[tuple[int, dict[str, Any]]] = []
        for idx, line in enumerate(raw.split("\n"), start=1):
            stripped = line.strip(" \t\r")
            if not stripped:
                continue
            item = parse_source_json(line, file_path, line=idx)
            if not isinstance(item, dict):
                raise ValueError(f"Line {idx} in {file_path} must be a JSON object.")
            records.append((idx, item))
        return records, None

    parsed = parse_source_json(raw, file_path)
    model_str = None
    if isinstance(parsed, dict):
        top_model = parsed.get("model_name") or parsed.get("model")
        model_str = str(top_model) if top_model is not None else None
        present = [key for key in container_keys if key in parsed]
        if len(present) > 1:
            raise ValueError(f"Ambiguous dataset containers in {file_path}: {present!r}.")
        if present:
            key = present[0]
            parsed = parsed[key]
            if not isinstance(parsed, list):
                raise ValueError(f"{file_path}: dataset container '{key}' must be a JSON array.")
    if isinstance(parsed, list):
        records = []
        for idx, item in enumerate(parsed, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{_record_location(path, idx)}: records must be JSON objects.")
            records.append((idx, item))
        return records, model_str

    raise ValueError(f"Unsupported dataset format in {file_path}.")


def load_model_samples(
    path: str | Path,
    default_model_name: str = "unknown",
) -> list[ModelSample]:
    """Load a list of ModelSample objects from a JSON or JSONL file."""
    file_path = Path(path)
    records, top_model = _load_records(file_path, ("samples", "outputs", "results"))
    fallback_model = top_model or (
        default_model_name if default_model_name != "unknown" else file_path.stem
    )
    samples = []
    for number, rec in records:
        try:
            samples.append(ModelSample.from_dict(rec, default_model_name=fallback_model))
        except ValueError as exc:
            raise ValueError(f"{_record_location(path, number)}: {exc}") from exc
    index_records(samples, str(file_path), positions=[n for n, _ in records],
                  unit=_record_unit(file_path))
    return samples


def load_prompt_cases(path: str | Path) -> list[PromptCase]:
    """Load a list of PromptCase objects from a JSON or JSONL file."""
    records, _ = _load_records(path, ("prompts", "cases", "suite", "items"))
    prompts = []
    for number, rec in records:
        try:
            prompts.append(PromptCase.from_dict(rec))
        except ValueError as exc:
            raise ValueError(f"{_record_location(path, number)}: {exc}") from exc
    index_records(prompts, str(path), positions=[n for n, _ in records], unit=_record_unit(path))
    return prompts


def format_report_table(report: DriftBenchmarkReport) -> str:
    """Format a DriftBenchmarkReport as a human-readable terminal summary table."""
    status_label = "PASSED" if report.passed else "FAILED"
    base_conf = report.baseline_json_conformity_rate
    cand_conf = report.candidate_json_conformity_rate
    lines: list[str] = [
        "=" * 78,
        f" LLM DRIFT BENCHMARK REPORT  [{status_label}]",
        "=" * 78,
        f" Baseline Model  : {report.baseline_model}",
        f" Candidate Model : {report.candidate_model}",
        (
            f" Total Samples   : {report.total_samples} "
            f"(Drifted: {report.drifted_samples} / {report.drift_rate:.1%})"
        ),
        "-" * 78,
        f" {'Metric':<36} | {'Baseline':<16} | {'Candidate / Value':<18}",
        "-" * 78,
        f" {'JSON Conformity Rate':<36} | {base_conf:>15.1%} | {cand_conf:>17.1%}",
        (
            f" {'Mean Composite Drift':<36} | {'0.0000 (ref)':>16} | "
            f"{report.mean_composite_drift:>18.4f}"
        ),
        (
            f" {'Max Composite Drift':<36} | {'0.0000 (ref)':>16} | "
            f"{report.max_composite_drift:>18.4f}"
        ),
        (
            f" {'Mean Token Jaccard Similarity':<36} | {'1.0000 (ref)':>16} | "
            f"{report.mean_token_jaccard:>18.4f}"
        ),
        (
            f" {'Mean Levenshtein Similarity':<36} | {'1.0000 (ref)':>16} | "
            f"{report.mean_levenshtein_similarity:>18.4f}"
        ),
    ]

    if report.mean_json_structural_similarity is not None:
        lines.append(
            f" {'Mean JSON Structural Similarity':<36} | {'1.0000 (ref)':>16} | "
            f"{report.mean_json_structural_similarity:>18.4f}"
        )
    if report.mean_json_numeric_drift is not None:
        lines.append(
            f" {'Mean JSON Numeric Relative Drift':<36} | {'0.0000 (ref)':>16} | "
            f"{report.mean_json_numeric_drift:>18.4f}"
        )
    for label, baseline_ms, candidate_ms in (
        ("P50 Latency (ms)", report.baseline_p50_latency_ms, report.candidate_p50_latency_ms),
        ("P95 Latency (ms)", report.baseline_p95_latency_ms, report.candidate_p95_latency_ms),
    ):
        baseline_text = "n/a" if baseline_ms is None else f"{baseline_ms:.1f}"
        candidate_text = "n/a" if candidate_ms is None else f"{candidate_ms:.1f}"
        lines.append(f" {label:<36} | {baseline_text:>16} | {candidate_text:>18}")
    if report.baseline_p95_latency_ms is None or report.candidate_p95_latency_ms is None:
        lines.append(" Latency regression check: n/a (incomplete measurements).")
    elif report.baseline_p95_latency_ms == 0:
        lines.append(" Latency regression check: n/a (zero baseline P95).")

    lines.extend(
        [
            "-" * 78,
            (
                f" {'Prompt ID':<18} | {'Category':<14} | {'JSON OK':<7} | "
                f"{'Struct Sim':<10} | {'Drift':<8} | {'State':<7}"
            ),
            "-" * 78,
        ]
    )

    for res in report.results:
        json_ok = "YES" if res.candidate_validation.passed else "NO"
        struct_str = (
            f"{res.json_structural_similarity:.4f}"
            if res.json_structural_similarity is not None
            else "N/A"
        )
        state_str = "DRIFT" if res.drifted else "OK"
        prompt_id = res.prompt_id if len(res.prompt_id) <= 18 else res.prompt_id[:15] + "..."
        category = res.category if len(res.category) <= 14 else res.category[:11] + "..."
        lines.append(
            f" {prompt_id:<18} | {category:<14} | {json_ok:<7} | "
            f"{struct_str:>10} | {res.composite_drift_score:>8.4f} | {state_str:<7}"
        )

    lines.append("=" * 78)
    if report.failure_reasons:
        lines.append(" Threshold Violations:")
        for reason in report.failure_reasons:
            lines.append(f"  - {reason}")
        lines.append("=" * 78)

    return "\n".join(lines)



