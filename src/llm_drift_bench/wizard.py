"""Guided plain-English setup wizard and use-case presets for generating drift.toml."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from llm_drift_bench.models import DriftThresholds
from llm_drift_bench.policy import DriftPolicy, FieldRule, MetricWeights, render_policy_toml

UseCase = Literal["extraction", "classification", "coding", "rag", "strict"]
NumericSensitivity = Literal["strict", "moderate", "loose"]

_VALID_USE_CASES: tuple[UseCase, ...] = (
    "extraction",
    "classification",
    "coding",
    "rag",
    "strict",
)
_VALID_NUMERIC_SENSITIVITIES: tuple[NumericSensitivity, ...] = ("strict", "moderate", "loose")


@dataclass(frozen=True)
class WizardAnswers:
    """Pragmatic plain-English choices that map directly to a DriftPolicy."""

    use_case: UseCase
    numeric_sensitivity: NumericSensitivity = "moderate"
    ignored_paths: tuple[str, ...] = ()
    unordered_array_paths: tuple[str, ...] = ()
    track_latency: bool = True
    long_outputs: bool = False

    def __post_init__(self) -> None:
        if self.use_case not in _VALID_USE_CASES:
            raise ValueError(
                f"use_case must be one of {_VALID_USE_CASES}, got {self.use_case!r}."
            )
        if self.numeric_sensitivity not in _VALID_NUMERIC_SENSITIVITIES:
            raise ValueError(
                f"numeric_sensitivity must be one of {_VALID_NUMERIC_SENSITIVITIES}, "
                f"got {self.numeric_sensitivity!r}."
            )
        if not isinstance(self.track_latency, bool):
            raise ValueError("track_latency must be a boolean.")
        if not isinstance(self.long_outputs, bool):
            raise ValueError("long_outputs must be a boolean.")
        for label, paths in (
            ("ignored_paths", self.ignored_paths),
            ("unordered_array_paths", self.unordered_array_paths),
        ):
            for path in paths:
                if not isinstance(path, str) or not path.startswith("$"):
                    raise ValueError(
                        f"{label} entries must be JSONPath strings starting with '$', got {path!r}."
                    )


PRESET_ANSWERS: dict[str, WizardAnswers] = {
    "extraction": WizardAnswers(
        use_case="extraction",
        numeric_sensitivity="strict",
        track_latency=True,
        long_outputs=False,
    ),
    "classification": WizardAnswers(
        use_case="classification",
        numeric_sensitivity="moderate",
        track_latency=True,
        long_outputs=False,
    ),
    "coding": WizardAnswers(
        use_case="coding",
        numeric_sensitivity="loose",
        track_latency=True,
        long_outputs=True,
    ),
    "rag": WizardAnswers(
        use_case="rag",
        numeric_sensitivity="moderate",
        track_latency=True,
        long_outputs=True,
    ),
    "strict": WizardAnswers(
        use_case="strict",
        numeric_sensitivity="strict",
        track_latency=True,
        long_outputs=False,
    ),
}

_USE_CASE_DESCRIPTIONS: dict[UseCase, str] = {
    "extraction": "Structured JSON extraction (invoices, forms, entities, tool args)",
    "classification": "Labeling, routing, triage, or scoring outputs",
    "coding": "Code generation, diffs, patches, or structured refactors",
    "rag": "Grounded Q&A, summaries, and citation-backed answers",
    "strict": "High-assurance deterministic gate (tight drift and 100% schema pass)",
}

_NUMERIC_THRESHOLDS: dict[NumericSensitivity, float] = {
    "strict": 0.02,
    "moderate": 0.08,
    "loose": 0.20,
}


def _normalize_user_paths(raw: str) -> tuple[str, ...]:
    """Parse comma-separated user field names into normalized '$' JSONPaths."""
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        if cleaned.startswith("$"):
            path = cleaned
        elif cleaned.startswith(".") or cleaned.startswith("["):
            path = f"${cleaned}"
        else:
            path = f"$.{cleaned}"
        if path not in seen:
            seen.add(path)
            out.append(path)
    return tuple(out)


def build_policy_from_answers(answers: WizardAnswers) -> DriftPolicy:
    """Translate WizardAnswers into a validated DriftPolicy."""
    if not isinstance(answers, WizardAnswers):
        raise TypeError("build_policy_from_answers requires a WizardAnswers instance.")

    num_limit = _NUMERIC_THRESHOLDS[answers.numeric_sensitivity]
    if answers.use_case == "extraction":
        max_drift, min_conf = 0.12, 0.98
        weights = MetricWeights(text_weight=0.2, json_weight=0.8)
    elif answers.use_case == "classification":
        max_drift, min_conf = 0.10, 0.98
        weights = MetricWeights(
            text_weight=0.25,
            json_weight=0.75,
            json_structure=0.25,
            json_exact_field=0.60,
            json_numeric=0.15,
        )
    elif answers.use_case == "coding":
        max_drift, min_conf = 0.22, 0.90
        weights = MetricWeights(text_weight=0.70, json_weight=0.30)
    elif answers.use_case == "rag":
        max_drift, min_conf = 0.25, 0.90
        weights = MetricWeights(text_weight=0.65, json_weight=0.35)
    else:  # strict
        max_drift, min_conf = 0.05, 1.0
        weights = MetricWeights(text_weight=0.30, json_weight=0.70)

    if answers.track_latency:
        max_latency_pct = 15.0 if answers.use_case == "strict" else 25.0
    else:
        max_latency_pct = 500.0

    thresholds = DriftThresholds(
        max_composite_drift=max_drift,
        min_json_conformity_rate=min_conf,
        max_numeric_relative_drift=num_limit,
        max_latency_regression_pct=max_latency_pct,
    )

    field_rules: dict[str, FieldRule] = {}
    for path in answers.unordered_array_paths:
        field_rules[path] = FieldRule(path=path, unordered_array=True)
    for path in answers.ignored_paths:
        # Ignore supersedes unordered_array on the exact same path.
        field_rules[path] = FieldRule(path=path, ignore=True)

    return DriftPolicy(
        thresholds=thresholds,
        weights=weights,
        field_rules=field_rules,
        long_text_mode="skip_levenshtein" if answers.long_outputs else "error",
    )


def render_wizard_config(answers: WizardAnswers) -> str:
    """Render a commented, round-trippable drift.toml generated from WizardAnswers."""
    policy = build_policy_from_answers(answers)
    header = [
        f"# Generated by `llm-drift-bench wizard` (preset: {answers.use_case})",
        f"# Focus: {_USE_CASE_DESCRIPTIONS[answers.use_case]}",
        f"# Numeric sensitivity: {answers.numeric_sensitivity} | "
        f"Latency guard: {'on' if answers.track_latency else 'relaxed'}",
        "",
    ]
    return "\n".join(header) + render_policy_toml(policy)


def _prompt_choice(
    prompt_text: str,
    choices: Sequence[tuple[str, str]],
    default_key: str,
    *,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    output_fn(prompt_text)
    for idx, (key, desc) in enumerate(choices, start=1):
        marker = " (default)" if key == default_key else ""
        output_fn(f"  {idx}. {key} - {desc}{marker}")
    raw = input_fn(f"Choose [1-{len(choices)} or name, default={default_key}]: ").strip().lower()
    if not raw:
        return default_key
    if raw.isdigit():
        idx_num = int(raw)
        if 1 <= idx_num <= len(choices):
            return choices[idx_num - 1][0]
    for key, _ in choices:
        if raw == key:
            return key
    output_fn(f"Unrecognized choice {raw!r}; using default {default_key!r}.")
    return default_key


def _prompt_yes_no(
    question: str,
    default: bool,
    *,
    input_fn: Callable[[str], str],
) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input_fn(f"{question} [{hint}]: ").strip().lower()
    if not raw:
        return default
    if raw in ("y", "yes", "true", "1"):
        return True
    if raw in ("n", "no", "false", "0"):
        return False
    return default


def run_guided_wizard(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> tuple[WizardAnswers, DriftPolicy]:
    """Ask 5 plain-English questions about the user's outputs and build a DriftPolicy."""
    output_fn("llm-drift-bench setup wizard")
    output_fn("Answer 5 short questions about your own outputs (press Enter to keep defaults).")
    output_fn("")

    use_case_raw = _prompt_choice(
        "1) What kind of outputs matter most for your app?",
        [(key, _USE_CASE_DESCRIPTIONS[key]) for key in _VALID_USE_CASES],
        default_key="extraction",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    use_case: UseCase = use_case_raw  # type: ignore[assignment]
    preset = PRESET_ANSWERS[use_case]

    num_raw = _prompt_choice(
        "\n2) If numbers inside your outputs shift slightly, how strict should checks be?",
        [
            ("strict", "Catch small shifts (<= 2% relative drift, e.g. totals, prices, counts)"),
            ("moderate", "Allow minor rounding noise (<= 8% relative drift, e.g. scores)"),
            ("loose", "Only flag large swings (<= 20% relative drift)"),
        ],
        default_key=preset.numeric_sensitivity,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    numeric_sensitivity: NumericSensitivity = num_raw  # type: ignore[assignment]

    output_fn(
        "\n3) Are any JSON fields noisy on purpose (like timestamps, request_id, trace_id)?"
    )
    ignored_raw = input_fn(
        "Comma-separated field paths to ignore (leave blank for none, e.g. request_id,meta.ts): "
    )
    ignored_paths = _normalize_user_paths(ignored_raw)

    output_fn(
        "\n4) Do any JSON list fields have items where order does not matter "
        "(like tags or citations)?"
    )
    unordered_raw = input_fn(
        "Comma-separated array paths (leave blank for none, e.g. tags,items[*].labels): "
    )
    unordered_paths = _normalize_user_paths(unordered_raw)

    output_fn("\n5) Runtime & output length preferences:")
    track_latency = _prompt_yes_no(
        "   Flag candidate models if P95 latency regresses?",
        default=preset.track_latency,
        input_fn=input_fn,
    )
    long_outputs = _prompt_yes_no(
        "   Do your prompts produce long outputs (> 4,000 chars, e.g. full code files or essays)?",
        default=preset.long_outputs,
        input_fn=input_fn,
    )

    answers = WizardAnswers(
        use_case=use_case,
        numeric_sensitivity=numeric_sensitivity,
        ignored_paths=ignored_paths,
        unordered_array_paths=unordered_paths,
        track_latency=track_latency,
        long_outputs=long_outputs,
    )
    return answers, build_policy_from_answers(answers)


def format_wizard_next_steps(answers: WizardAnswers, *, config_path: str = "drift.toml") -> str:
    """Return concrete, jargon-free follow-up CLI commands tailored to the chosen use case."""
    lines = [
        f"Next steps using {config_path}:",
        "  1. Run per-prompt regression gate:",
        f"     llm-drift-bench evaluate -p prompts.json -b baseline.jsonl -c candidate.jsonl "
        f"--config {config_path}",
    ]
    if answers.use_case in ("extraction", "classification", "strict"):
        lines.append("  2. Check cross-prompt field distribution slides (scores & label shares):")
        lines.append(
            f"     llm-drift-bench distribution -b baseline.jsonl -c candidate.jsonl "
            f"--config {config_path}"
        )
    else:
        lines.append(
            "  2. Check for quantization failure modes (loops, refusal flips, length jumps):"
        )
        lines.append(
            "     llm-drift-bench check-quality -b baseline.jsonl -c candidate.jsonl"
        )
    lines.append("  3. Inspect exact field/text diffs for any drifted prompt:")
    lines.append("     llm-drift-bench diff -b baseline.jsonl -c candidate.jsonl")
    return "\n".join(lines) + "\n"
