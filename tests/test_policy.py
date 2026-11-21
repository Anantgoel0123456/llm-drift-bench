"""Unit and CLI tests for drift.toml / drift.json policies, custom weights, and field rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_drift_bench.cli import main
from llm_drift_bench.engine import DriftBenchmarkEngine
from llm_drift_bench.models import ModelSample
from llm_drift_bench.policy import (
    DriftPolicy,
    FieldRule,
    MetricWeights,
    _parse_toml_fallback,
    load_policy,
)


def test_load_toml_and_json_policy_with_field_rules(tmp_path: Path) -> None:
    toml_file = tmp_path / "drift.toml"
    toml_file.write_text(
        """
# Task-specific policy
long_text_mode = "skip_levenshtein"

[thresholds]
max_composite_drift = 0.20
min_json_conformity_rate = 0.99
max_numeric_relative_drift = 0.05
max_latency_regression_pct = 30.0

[weights]
text_weight = 0.20
json_weight = 0.80
json_structure = 0.50
json_exact_field = 0.30
json_numeric = 0.20

[fields."$.timestamp"]
ignore = true

[fields."$.confidence"]
tolerance_abs = 0.05

[fields."$.total_usd"]
tolerance_rel = 0.01
""".strip() + "\n",
        encoding="utf-8",
    )
    policy = load_policy(toml_file)
    assert policy.thresholds.max_composite_drift == 0.20
    assert policy.weights.json_weight == 0.80
    assert policy.field_rules["$.timestamp"].ignore is True
    assert policy.field_rules["$.confidence"].tolerance_abs == 0.05
    assert policy.field_rules["$.total_usd"].tolerance_rel == 0.01
    assert policy.long_text_mode == "skip_levenshtein"

    # Exercise fallback TOML parser directly (used on Python 3.10)
    fallback = _parse_toml_fallback(toml_file.read_text(encoding="utf-8"))
    assert fallback["fields"]["$.timestamp"]["ignore"] is True
    assert fallback["thresholds"]["max_composite_drift"] == 0.20

    # JSON policy format works identically
    json_file = tmp_path / "drift.json"
    json_file.write_text(
        json.dumps(
            {
                "thresholds": {"max_composite_drift": 0.12},
                "fields": {"$.request_id": {"ignore": True}},
            }
        ),
        encoding="utf-8",
    )
    j_policy = load_policy(json_file)
    assert j_policy.thresholds.max_composite_drift == 0.12
    assert j_policy.field_rules["$.request_id"].ignore is True


def test_ignored_volatile_fields_and_numeric_tolerances() -> None:
    policy = DriftPolicy(
        weights=MetricWeights(text_weight=0.0, json_weight=1.0),
        field_rules={
            "$.timestamp": FieldRule(path="$.timestamp", ignore=True),
            "$.confidence": FieldRule(path="$.confidence", tolerance_abs=0.05),
            "$.total": FieldRule(path="$.total", tolerance_rel=0.01),
        },
    )
    base = [
        ModelSample(
            prompt_id="p1",
            model_name="fp16",
            output='{"timestamp": "2026-01-01T00:00:00Z", "confidence": 0.92, "total": 1000.0}',
        )
    ]
    cand = [
        ModelSample(
            prompt_id="p1",
            model_name="int4",
            output='{"timestamp": "2026-11-21T22:00:00Z", "confidence": 0.95, "total": 1005.0}',
        )
    ]
    # Without policy, timestamp + confidence + total changes produce significant drift
    raw_report = DriftBenchmarkEngine().evaluate(base, cand)
    assert raw_report.mean_composite_drift > 0.15

    # With policy (ignore timestamp, +-0.05 confidence, 1% total, JSON-only weight), drift is 0
    pol_report = DriftBenchmarkEngine(policy=policy).evaluate(base, cand)
    assert pol_report.passed
    assert pol_report.mean_composite_drift == 0.0
    assert pol_report.mean_json_numeric_drift == 0.0


def test_long_text_mode_skip_levenshtein_handles_large_outputs(tmp_path: Path, capsys) -> None:
    big_a = json.dumps({"body": "word " * 4000})
    big_b = json.dumps({"body": "word " * 4000})
    bf = tmp_path / "base.jsonl"
    cf = tmp_path / "cand.jsonl"
    bf.write_text(json.dumps({"prompt_id": "big", "output": big_a}) + "\n", encoding="utf-8")
    cf.write_text(json.dumps({"prompt_id": "big", "output": big_b}) + "\n", encoding="utf-8")

    # Default mode fails on >16384 chars
    assert main(["run", "-b", str(bf), "-c", str(cf)]) == 2
    capsys.readouterr()

    # skip_levenshtein mode succeeds cleanly
    assert main(["run", "-b", str(bf), "-c", str(cf), "--long-text-mode", "skip_levenshtein"]) == 0
    assert "PASSED" in capsys.readouterr().out


def test_cli_config_flag_and_flag_override(tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "drift.toml"
    cfg.write_text(
        '[thresholds]\nmax_composite_drift = 0.01\n\n[fields."$.ts"]\nignore = true\n',
        encoding="utf-8",
    )
    bf = tmp_path / "b.jsonl"
    cf = tmp_path / "c.jsonl"
    bf.write_text(
        '{"prompt_id": "p1", "output": "{\\"ts\\": \\"1\\", \\"v\\": 100}"}\n', encoding="utf-8"
    )
    cf.write_text(
        '{"prompt_id": "p1", "output": "{\\"ts\\": \\"999\\", \\"v\\": 100}"}\n', encoding="utf-8"
    )

    # With strict max_composite_drift=0.01 from config (text still differs slightly), fails
    assert main(["run", "-b", str(bf), "-c", str(cf), "--config", str(cfg)]) == 1
    capsys.readouterr()

    # Explicit CLI --max-drift overrides config threshold
    rc = main(
        ["run", "-b", str(bf), "-c", str(cf), "--config", str(cfg), "--max-drift", "0.5"]
    )
    assert rc == 0


def test_invalid_policy_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[unknown_section]\nx = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        load_policy(bad)

    with pytest.raises(ValueError, match="must start with"):
        FieldRule(path="timestamp", ignore=True)

    with pytest.raises(ValueError, match="positive"):
        MetricWeights(text_weight=0.0, json_weight=0.0)
