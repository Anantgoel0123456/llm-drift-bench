"""Task-specific evaluation policies (drift.toml / drift.json): weights and per-field rules."""

from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from llm_drift_bench._file_io import parse_source_json, read_utf8
from llm_drift_bench._numbers import finite_number
from llm_drift_bench.metrics import _json_type_tag, _values_equal, flatten_json_paths
from llm_drift_bench.models import DriftThresholds

LongTextMode = Literal["error", "skip_levenshtein"]
_VALID_LONG_TEXT_MODES = ("error", "skip_levenshtein")


def _check_weight(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Weight {name!r} must be a finite non-negative number.")


def _check_optional_nonneg(name: str, value: float | None) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number.")


@dataclass(frozen=True)
class MetricWeights:
    """Configurable component weights for the composite drift score."""

    text_weight: float = 0.35
    json_weight: float = 0.65
    token_jaccard: float = 0.35
    ngram: float = 0.30
    levenshtein: float = 0.35
    json_structure: float = 0.40
    json_exact_field: float = 0.35
    json_numeric: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "text_weight", "json_weight", "token_jaccard", "ngram", "levenshtein",
            "json_structure", "json_exact_field", "json_numeric",
        ):
            _check_weight(name, getattr(self, name))
        if self.text_weight + self.json_weight <= 0:
            raise ValueError("text_weight + json_weight must be positive.")
        if self.token_jaccard + self.ngram + self.levenshtein <= 0:
            raise ValueError("Text metric weights must sum to a positive value.")
        if self.json_structure + self.json_exact_field + self.json_numeric <= 0:
            raise ValueError("JSON metric weights must sum to a positive value.")


@dataclass(frozen=True)
class FieldRule:
    """Per-JSONPath override for ignoring fields, unordered arrays, or numeric tolerances."""

    path: str
    ignore: bool = False
    unordered_array: bool = False
    tolerance_abs: float | None = None
    tolerance_rel: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.startswith("$"):
            raise ValueError(f"FieldRule path must start with '$': {self.path!r}.")
        if not isinstance(self.ignore, bool):
            raise ValueError("FieldRule.ignore must be a boolean.")
        if not isinstance(self.unordered_array, bool):
            raise ValueError("FieldRule.unordered_array must be a boolean.")
        _check_optional_nonneg("tolerance_abs", self.tolerance_abs)
        _check_optional_nonneg("tolerance_rel", self.tolerance_rel)


@dataclass(frozen=True)
class DriftPolicy:
    """Complete reusable evaluation policy loaded from drift.toml or drift.json."""

    thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    weights: MetricWeights = field(default_factory=MetricWeights)
    field_rules: dict[str, FieldRule] = field(default_factory=dict)
    long_text_mode: LongTextMode = "error"

    def __post_init__(self) -> None:
        if self.long_text_mode not in _VALID_LONG_TEXT_MODES:
            raise ValueError(
                f"long_text_mode must be one of {_VALID_LONG_TEXT_MODES}, "
                f"got {self.long_text_mode!r}."
            )


def _strip_toml_comment(line: str) -> str:
    in_quotes = False
    escaped = False
    for index, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_quotes:
            escaped = True
        elif ch == '"':
            in_quotes = not in_quotes
        elif ch == "#" and not in_quotes:
            return line[:index]
    return line


def _parse_toml_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return json.loads(value)
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return value[1:-1]
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?|[+-]?\d+[eE][+-]?\d+", value):
        return finite_number(float(value))
    raise ValueError(f"Unsupported TOML scalar value: {raw!r}.")


def _parse_toml_fallback(text: str) -> dict[str, Any]:
    """Minimal strict TOML parser for Python 3.10 environments without tomllib."""
    root: dict[str, Any] = {}
    current: dict[str, Any] = root
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_toml_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            header = line[1:-1].strip()
            parts: list[str] = []
            for match in re.finditer(r'"((?:\\.|[^"\\])*)"|([^.\s]+)', header):
                if match.group(1) is not None:
                    parts.append(json.loads(f'"{match.group(1)}"'))
                else:
                    parts.append(match.group(2))
            if not parts:
                raise ValueError(f"Line {lineno}: empty TOML section header.")
            node = root
            for part in parts:
                nxt = node.setdefault(part, {})
                if not isinstance(nxt, dict):
                    raise ValueError(f"Line {lineno}: conflicting TOML table {part!r}.")
                node = nxt
            current = node
            continue
        if "=" not in line:
            raise ValueError(f"Line {lineno}: expected 'key = value' assignment.")
        key_raw, val_raw = line.split("=", 1)
        key = key_raw.strip()
        if key.startswith('"') and key.endswith('"'):
            key = json.loads(key)
        if key in current:
            raise ValueError(f"Line {lineno}: duplicate TOML key {key!r}.")
        current[key] = _parse_toml_scalar(val_raw)
    return root


def _parse_toml(text: str, source: Path) -> dict[str, Any]:
    try:
        tomllib = importlib.import_module("tomllib")
        parsed: Any = tomllib.loads(text)
    except ModuleNotFoundError:
        try:
            parsed = _parse_toml_fallback(text)
        except ValueError as exc:
            raise ValueError(f"Invalid TOML in {source}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Invalid TOML in {source}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Policy file must decode to a table/object: {source}")
    return parsed


def _check_keys(section: str, data: Mapping[str, Any], allowed: set[str]) -> None:
    extra = sorted(set(data) - allowed)
    if extra:
        raise ValueError(f"Unknown key(s) in policy {section}: {extra}.")


def load_policy(path: str | Path) -> DriftPolicy:
    """Load and validate a DriftPolicy from a .toml or .json policy file."""
    source = Path(path)
    text = read_utf8(source)
    if source.suffix.lower() == ".json":
        raw = parse_source_json(text, source)
        if not isinstance(raw, dict):
            raise ValueError(f"Policy file must contain a JSON object: {source}")
    else:
        raw = _parse_toml(text, source)

    _check_keys("root", raw, {"thresholds", "weights", "fields", "long_text_mode"})

    thresholds_raw = raw.get("thresholds", {})
    if not isinstance(thresholds_raw, dict):
        raise ValueError("Policy 'thresholds' must be a table/object.")
    _check_keys(
        "thresholds",
        thresholds_raw,
        {
            "max_composite_drift",
            "min_json_conformity_rate",
            "max_numeric_relative_drift",
            "max_latency_regression_pct",
        },
    )
    thresholds = DriftThresholds(**thresholds_raw)

    weights_raw = raw.get("weights", {})
    if not isinstance(weights_raw, dict):
        raise ValueError("Policy 'weights' must be a table/object.")
    _check_keys(
        "weights",
        weights_raw,
        {
            "text_weight", "json_weight", "token_jaccard", "ngram", "levenshtein",
            "json_structure", "json_exact_field", "json_numeric",
        },
    )
    weights = MetricWeights(**weights_raw)

    fields_raw = raw.get("fields", {})
    if not isinstance(fields_raw, dict):
        raise ValueError("Policy 'fields' must be a table/object.")
    field_rules: dict[str, FieldRule] = {}
    for field_path, spec in fields_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Policy field rule {field_path!r} must be a table/object.")
        _check_keys(
            f"fields.{field_path}",
            spec,
            {"ignore", "unordered_array", "tolerance_abs", "tolerance_rel"},
        )
        field_rules[field_path] = FieldRule(path=field_path, **spec)

    long_text_mode = raw.get("long_text_mode", "error")
    if not isinstance(long_text_mode, str) or long_text_mode not in _VALID_LONG_TEXT_MODES:
        raise ValueError(f"long_text_mode must be one of {_VALID_LONG_TEXT_MODES}.")

    return DriftPolicy(
        thresholds=thresholds,
        weights=weights,
        field_rules=field_rules,
        long_text_mode=long_text_mode,  # type: ignore[arg-type]
    )


def _path_is_ignored(path: str, ignored_prefixes: tuple[str, ...]) -> bool:
    for prefix in ignored_prefixes:
        if path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "["):
            return True
    return False


def _normalize_unordered_arrays(
    data: Any,
    unordered_paths: frozenset[str],
    prefix: str = "$",
) -> Any:
    """Recursively sort arrays whose JSONPath is configured as order-insensitive."""
    if isinstance(data, dict):
        result: dict[str, Any] = {}
        for key, val in data.items():
            seg = (
                f".{key}" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                else f"[{json.dumps(key, ensure_ascii=True)}]"
            )
            result[key] = _normalize_unordered_arrays(val, unordered_paths, f"{prefix}{seg}")
        return result
    if isinstance(data, list):
        items = [
            _normalize_unordered_arrays(item, unordered_paths, f"{prefix}[{i}]")
            for i, item in enumerate(data)
        ]
        if prefix in unordered_paths:
            return sorted(items, key=lambda elem: json.dumps(elem, sort_keys=True))
        return items
    return data


def _numeric_within_tolerance(
    val_a: float,
    val_b: float,
    rule: FieldRule | None,
) -> tuple[bool, float]:
    denom = max(abs(val_a), abs(val_b), 1e-9)
    opposite_sign = (val_a < 0) != (val_b < 0)
    rel_diff = abs(val_a / denom - val_b / denom) if opposite_sign else abs(val_a - val_b) / denom

    if rule is not None:
        if rule.tolerance_abs is not None:
            tol_abs = rule.tolerance_abs
            if opposite_sign:
                within_abs = (
                    abs(val_a) <= tol_abs
                    and abs(val_b) <= tol_abs
                    and (abs(val_a) + abs(val_b)) <= tol_abs
                )
            else:
                within_abs = abs(val_a - val_b) <= tol_abs
            if within_abs:
                return True, 0.0
        if rule.tolerance_rel is not None and rel_diff <= rule.tolerance_rel:
            return True, 0.0
    return False, min(1.0, rel_diff)


def evaluate_json_with_policy(
    json_a: Any,
    json_b: Any,
    field_rules: Mapping[str, FieldRule],
) -> tuple[float, float, float | None]:
    """Compute (structural_sim, exact_field_match, numeric_drift) applying per-field rules."""
    unordered = frozenset(path for path, rule in field_rules.items() if rule.unordered_array)
    if unordered:
        json_a = _normalize_unordered_arrays(json_a, unordered)
        json_b = _normalize_unordered_arrays(json_b, unordered)
    flat_a_raw = flatten_json_paths(json_a)
    flat_b_raw = flatten_json_paths(json_b)

    # Validate all numeric values even on ignored/unmatched paths for safety consistency.
    for values in (flat_a_raw, flat_b_raw):
        for value in values.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                finite_number(value)

    ignored = tuple(path for path, rule in field_rules.items() if rule.ignore)
    flat_a = {k: v for k, v in flat_a_raw.items() if not _path_is_ignored(k, ignored)}
    flat_b = {k: v for k, v in flat_b_raw.items() if not _path_is_ignored(k, ignored)}

    if not flat_a and not flat_b:
        return 1.0, 1.0, None
    if not flat_a or not flat_b:
        return 0.0, 0.0, None

    typed_a = {(path, _json_type_tag(val)) for path, val in flat_a.items()}
    typed_b = {(path, _json_type_tag(val)) for path, val in flat_b.items()}
    struct_sim = len(typed_a & typed_b) / len(typed_a | typed_b)

    all_paths = set(flat_a) | set(flat_b)
    matched = 0
    drifts: list[float] = []
    for path in sorted(all_paths):
        if path not in flat_a or path not in flat_b:
            continue
        val_a, val_b = flat_a[path], flat_b[path]
        rule = field_rules.get(path)
        both_numeric = (
            isinstance(val_a, (int, float))
            and not isinstance(val_a, bool)
            and isinstance(val_b, (int, float))
            and not isinstance(val_b, bool)
        )
        if both_numeric:
            num_a, num_b = finite_number(val_a), finite_number(val_b)
            within_tol, bounded_rel = _numeric_within_tolerance(num_a, num_b, rule)
            drifts.append(bounded_rel)
            if within_tol or _values_equal(val_a, val_b):
                matched += 1
        elif _values_equal(val_a, val_b):
            matched += 1

    exact_match = matched / len(all_paths)
    numeric_drift = (sum(drifts) / len(drifts)) if drifts else None
    return struct_sim, exact_match, numeric_drift


def compute_weighted_composite_drift(
    *,
    weights: MetricWeights,
    token_jaccard: float,
    ngram_sim: float,
    levenshtein_sim: float | None,
    json_structural_sim: float | None = None,
    json_field_match: float | None = None,
    json_num_drift: float | None = None,
    baseline_schema_passed: bool = True,
    candidate_schema_passed: bool = True,
) -> float:
    """Compute composite drift score in [0.0, 1.0] using configurable MetricWeights."""
    text_parts: list[tuple[float, float]] = [
        (weights.token_jaccard, 1.0 - token_jaccard),
        (weights.ngram, 1.0 - ngram_sim),
    ]
    if levenshtein_sim is not None:
        text_parts.append((weights.levenshtein, 1.0 - levenshtein_sim))
    text_weight_sum = sum(w for w, _ in text_parts)
    if text_weight_sum <= 0:
        text_drift = 0.5 * (1.0 - token_jaccard) + 0.5 * (1.0 - ngram_sim)
    else:
        text_drift = sum(w * val for w, val in text_parts) / text_weight_sum

    if json_structural_sim is not None and json_field_match is not None:
        json_parts: list[tuple[float, float]] = [
            (weights.json_structure, 1.0 - json_structural_sim),
            (weights.json_exact_field, 1.0 - json_field_match),
        ]
        if json_num_drift is not None:
            json_parts.append((weights.json_numeric, json_num_drift))
        json_weight_sum = sum(w for w, _ in json_parts)
        if json_weight_sum <= 0:
            json_drift = 0.5 * (1.0 - json_structural_sim) + 0.5 * (1.0 - json_field_match)
        else:
            json_drift = sum(w * val for w, val in json_parts) / json_weight_sum
        total_mix = weights.text_weight + weights.json_weight
        combined = (
            weights.text_weight * text_drift + weights.json_weight * json_drift
        ) / total_mix
    else:
        combined = text_drift

    if baseline_schema_passed and not candidate_schema_passed:
        combined = max(combined, 0.75) + 0.25 * (1.0 - max(combined, 0.75))

    return max(0.0, min(1.0, combined))
