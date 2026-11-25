"""CLI tests for redact, matrix, lint-suite, subsample, and new flags on existing commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_drift_bench.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_cli_redact_secrets_and_pii(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = tmp_path / "raw.jsonl"
    out_all = tmp_path / "clean_all.jsonl"
    out_sec = tmp_path / "clean_sec.jsonl"
    sample = {
        "prompt_id": "p1",
        "model_name": "m1",
        "output": json.dumps(
            {
                "contact": "alice@example.com",
                "token": "sk-proj-abcdefghijklmnopqrstuvwxyz1234",
            }
        ),
    }
    raw.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    code = main(["redact", "-i", str(raw), "-o", str(out_all), "--format", "json"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["total_replacements"] == 2
    cleaned_all = json.loads(out_all.read_text(encoding="utf-8").strip())
    assert "alice@example.com" not in cleaned_all["output"]
    assert "[REDACTED:OPENAI_KEY]" in cleaned_all["output"]

    code2 = main(["redact", "-i", str(raw), "-o", str(out_sec), "--secrets-only"])
    assert code2 == 0
    assert "Redacted 1 occurrence(s)" in capsys.readouterr().out
    cleaned_sec = json.loads(out_sec.read_text(encoding="utf-8").strip())
    assert "alice@example.com" in cleaned_sec["output"]
    assert "[REDACTED:OPENAI_KEY]" in cleaned_sec["output"]

    # Refuses overwriting input file
    assert main(["redact", "-i", str(raw), "-o", str(raw)]) == 2


def test_cli_matrix_text_json_and_html(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pass_path = str(EXAMPLES / "expected-pass.json")
    drift_path = str(EXAMPLES / "expected-drift.json")

    assert main(["matrix", drift_path, pass_path]) == 0
    txt = capsys.readouterr().out
    assert "Drift Matrix" in txt and "ticket-01" in txt

    assert main(["matrix", drift_path, pass_path, "--sort-prompts", "prompt_id",
                 "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [row["prompt_id"] for row in data["rows"]] == ["invoice-01", "ticket-01"]

    html_out = tmp_path / "heatmap.html"
    assert main(["matrix", drift_path, pass_path, "--format", "html", "-o", str(html_out)]) == 0
    _ = capsys.readouterr()
    html = html_out.read_text(encoding="utf-8")
    assert "<table" in html and "<script" not in html.lower()


def test_cli_lint_suite_normal_and_strict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    warn_suite = tmp_path / "warn_suite.json"
    warn_suite.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "p1",
                    "category": "extraction",
                    "prompt": "Extract total",
                    "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                }
            ]
        ),
        encoding="utf-8",
    )

    assert main(["lint-suite", "-s", str(warn_suite)]) == 0
    assert "permissive_object" in capsys.readouterr().out

    assert main(["lint-suite", "-s", str(warn_suite), "--strict", "--format", "json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["warnings"] == 1 and payload["errors"] == 0

    err_suite = tmp_path / "err_suite.json"
    err_suite.write_text(
        json.dumps(
            [
                {
                    "prompt_id": "p1",
                    "category": "extraction",
                    "prompt": "Extract",
                    "schema": {"type": "object", "anyOf": [{"type": "string"}]},
                }
            ]
        ),
        encoding="utf-8",
    )
    assert main(["lint-suite", "-s", str(err_suite)]) == 1


def test_cli_subsample_and_lockstep_slice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "suite.json"
    cap = tmp_path / "capture.jsonl"
    cases = [
        {"prompt_id": f"ext_{i}", "category": "extraction", "prompt": f"E {i}"}
        for i in range(4)
    ] + [
        {"prompt_id": f"cls_{i}", "category": "classification", "prompt": f"C {i}"}
        for i in range(4)
    ]
    suite.write_text(json.dumps(cases), encoding="utf-8")
    cap.write_text(
        "\n".join(
            json.dumps({"prompt_id": c["prompt_id"], "model_name": "m", "output": "ok"})
            for c in cases
        )
        + "\n",
        encoding="utf-8",
    )

    out_suite = tmp_path / "mini_suite.json"
    out_cap = tmp_path / "mini_capture.jsonl"
    code = main(
        [
            "subsample",
            "-s",
            str(suite),
            "--size",
            "4",
            "--seed",
            "7",
            "-o",
            str(out_suite),
            "--slice-capture",
            str(cap),
            str(out_cap),
        ]
    )
    assert code == 0
    assert "Sampled 4/8 prompt(s)" in capsys.readouterr().out
    sub_cases = json.loads(out_suite.read_text(encoding="utf-8"))
    assert len(sub_cases) == 4
    cats = [c["category"] for c in sub_cases]
    assert cats.count("extraction") == 2 and cats.count("classification") == 2
    sub_rows = [json.loads(line) for line in out_cap.read_text(encoding="utf-8").splitlines()]
    assert [r["prompt_id"] for r in sub_rows] == [c["prompt_id"] for c in sub_cases]

    # Colliding output paths fail cleanly
    assert (
        main(
            [
                "subsample",
                "-s",
                str(suite),
                "--size",
                "2",
                "-o",
                str(out_suite),
                "--slice-capture",
                str(cap),
                str(out_suite),
            ]
        )
        == 2
    )


def test_cli_compare_reports_markdown_output_and_collision_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pass_path = str(EXAMPLES / "expected-pass.json")
    drift_path = str(EXAMPLES / "expected-drift.json")
    md_out = tmp_path / "leaderboard.md"

    code = main(
        [
            "compare-reports",
            drift_path,
            pass_path,
            "--against",
            "demo-candidate-pass",
            "--markdown-output",
            str(md_out),
        ]
    )
    assert code == 0
    _ = capsys.readouterr()
    md = md_out.read_text(encoding="utf-8")
    assert "# Candidate Drift Comparison" in md
    assert "| Rank | Candidate |" in md
    assert "Prompt transitions vs `demo-candidate-pass`" in md

    # Same file for --html-output and --markdown-output must fail
    same = tmp_path / "same.out"
    assert (
        main(
            [
                "compare-reports",
                drift_path,
                pass_path,
                "--html-output",
                str(same),
                "--markdown-output",
                str(same),
            ]
        )
        == 2
    )


def test_cli_calibrate_numeric_tolerances(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_a = tmp_path / "a.jsonl"
    base_b = tmp_path / "b.jsonl"
    base_a.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "m", "output": '{"score": 0.90}'}) + "\n",
        encoding="utf-8",
    )
    base_b.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "m", "output": '{"score": 0.92}'}) + "\n",
        encoding="utf-8",
    )

    code = main(
        [
            "calibrate",
            "--baseline-a",
            str(base_a),
            "--baseline-b",
            str(base_b),
            "--numeric-tolerances",
        ]
    )
    assert code == 0
    toml_out = capsys.readouterr().out
    assert "tolerance_abs =" in toml_out
    assert "ignore = true" not in toml_out

    # --numeric-tolerances without baseline captures raises exit code 2
    assert main(["calibrate", str(EXAMPLES / "expected-pass.json"), "--numeric-tolerances"]) == 2


def test_cli_capture_provider_anthropic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "prompts.json"
    suite.write_text(
        json.dumps([{"prompt_id": "p1", "prompt": "Say hi"}]),
        encoding="utf-8",
    )
    out = tmp_path / "anthropic.jsonl"
    seen: list[dict[str, Any]] = []

    def fake_post(
        url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> dict[str, Any]:
        seen.append({"url": url, "body": body, "headers": headers})
        return {
            "content": [{"type": "text", "text": '{"ok": true}'}],
            "usage": {"output_tokens": 5},
        }

    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-test-key")
    monkeypatch.setattr("llm_drift_bench.capture._default_transport", fake_post)

    code = main(
        [
            "capture",
            "-s",
            str(suite),
            "--endpoint",
            "https://api.anthropic.com",
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-5",
            "--api-key-env",
            "TEST_ANTHROPIC_KEY",
            "-o",
            str(out),
        ]
    )
    assert code == 0
    assert "Captured 1 prompt(s)" in capsys.readouterr().out
    assert seen[0]["url"].endswith("/v1/messages")
    assert seen[0]["headers"]["x-api-key"] == "sk-ant-test-key"
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["output"] == '{"ok": true}'
    assert row["tokens_generated"] == 5


def test_cli_gate_cost_snapshot_gemini_and_history_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    # 1. gate command: GO on passing report, NO-GO on drifted report
    assert main(["gate", "-i", str(EXAMPLES / "expected-pass.json")]) == 0
    assert "GO (APPROVED)" in capsys.readouterr().out

    assert main(["gate", "-i", str(EXAMPLES / "expected-drift.json"), "--format", "json"]) == 1
    gate_data = json.loads(capsys.readouterr().out)
    assert gate_data["approved"] is False

    # gate with --compare-against runs significance tests cleanly
    assert (
        main(
            [
                "gate",
                "-i",
                str(EXAMPLES / "expected-pass.json"),
                "--compare-against",
                str(EXAMPLES / "expected-pass.json"),
                "--resamples",
                "100",
                "--format",
                "json",
            ]
        )
        == 0
    )
    gate_sig = json.loads(capsys.readouterr().out)
    assert {c["name"] for c in gate_sig["checks"]} >= {
        "benchmark_drift",
        "statistical_drift_shift",
        "mcnemar_conformity",
    }

    # 2. cost command: incomplete tokens reported honestly as n/a, complete tokens compute USD
    assert (
        main(
            [
                "cost",
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-pass.jsonl"),
            ]
        )
        == 0
    )
    assert "incomplete explicit tokens_generated" in capsys.readouterr().out

    tok_b = tmp_path / "tok_b.jsonl"
    tok_c = tmp_path / "tok_c.jsonl"
    tok_b.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "b", "output": "hi", "tokens_generated": 1000})
        + "\n",
        encoding="utf-8",
    )
    tok_c.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "c", "output": "hi", "tokens_generated": 1500})
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "cost",
                "-b",
                str(tok_b),
                "-c",
                str(tok_c),
                "--baseline-price",
                "2.0",
                "--candidate-price",
                "1.0",
                "--format",
                "json",
            ]
        )
        == 0
    )
    cost_out = json.loads(capsys.readouterr().out)
    assert cost_out["complete"] is True
    assert cost_out["token_delta"] == 500
    assert cost_out["cost_delta_usd"] == pytest.approx(-0.0005)

    # 3. snapshot create & verify
    snap_dir = tmp_path / "snap"
    snap_dir.mkdir()
    shutil.copy2(EXAMPLES / "baseline.jsonl", snap_dir / "baseline.jsonl")
    shutil.copy2(EXAMPLES / "extraction-suite.json", snap_dir / "extraction-suite.json")
    manifest_path = snap_dir / "snapshot.manifest.json"
    assert (
        main(
            [
                "snapshot",
                "create",
                "-b",
                str(snap_dir / "baseline.jsonl"),
                "-s",
                str(snap_dir / "extraction-suite.json"),
                "-o",
                str(manifest_path),
            ]
        )
        == 0
    )
    assert "Created snapshot manifest" in capsys.readouterr().out
    assert main(["snapshot", "verify", "-i", str(manifest_path)]) == 0
    assert "VALID" in capsys.readouterr().out
    (snap_dir / "baseline.jsonl").write_text("tampered\n", encoding="utf-8")
    assert main(["snapshot", "verify", "-i", str(manifest_path), "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False

    # 4. check-tools --adapter gemini
    gem_trace = tmp_path / "gemini_trace.json"
    gem_trace.write_text(
        json.dumps(
            {
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "read_file", "args": {"path": "g@1"}}},
                            {"functionCall": {"name": "edit_file", "args": {"path": "a@1"}}},
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "read_file",
                                    "response": {"range": {"start": 0, "end": 50}},
                                }
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "check-tools",
                "-i",
                str(gem_trace),
                "--adapter",
                "gemini",
                "--resource",
                "g@1",
                "--end",
                "50",
                "--trace-complete",
            ]
        )
        == 0
    )
    _ = capsys.readouterr()

    # 5. capture --provider gemini
    suite = tmp_path / "g_suite.json"
    suite.write_text(json.dumps([{"prompt_id": "p1", "prompt": "Hi"}]), encoding="utf-8")
    g_out = tmp_path / "g_cap.jsonl"

    def fake_gemini(
        url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> dict[str, Any]:
        del body, headers, timeout
        assert ":generateContent" in url
        return {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

    monkeypatch.setattr("llm_drift_bench.capture._default_transport", fake_gemini)
    assert (
        main(
            [
                "capture",
                "-s",
                str(suite),
                "--endpoint",
                "https://generativelanguage.googleapis.com",
                "--provider",
                "gemini",
                "--model",
                "gemini-2.5-flash",
                "-o",
                str(g_out),
            ]
        )
        == 0
    )
    _ = capsys.readouterr()
    assert json.loads(g_out.read_text(encoding="utf-8").strip())["output"] == "hello"

    # 6. history summary & prune
    ledger = tmp_path / "hist.jsonl"
    for tag in ("v1", "v2", "v3"):
        assert (
            main(
                [
                    "history",
                    "record",
                    "--ledger",
                    str(ledger),
                    "-i",
                    str(EXAMPLES / "expected-pass.json"),
                    "--tag",
                    tag,
                ]
            )
            == 0
        )
    _ = capsys.readouterr()
    assert main(["history", "summary", "--ledger", str(ledger), "--format", "json"]) == 0
    hist_sum = json.loads(capsys.readouterr().out)
    assert hist_sum["total_entries"] == 3
    assert main(["history", "prune", "--ledger", str(ledger), "--keep-last", "1",
                 "--format", "json"]) == 0
    prune_out = json.loads(capsys.readouterr().out)
    assert prune_out["kept"] == 1 and prune_out["removed"] == 2


def test_cli_import_captures_badge_diff_markdown_cost_and_tool_budgets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. import-captures (csv, openai-batch, anthropic-batch)
    csv_in = tmp_path / "raw.csv"
    csv_in.write_text(
        'prompt_id,output,tokens_generated\np1,"{""ok"": true}",10\n', encoding="utf-8"
    )
    csv_out = tmp_path / "from_csv.jsonl"
    assert (
        main(
            [
                "import-captures",
                "--from",
                "csv",
                "-i",
                str(csv_in),
                "-o",
                str(csv_out),
                "--model",
                "q4",
                "--format",
                "json",
            ]
        )
        == 0
    )
    imp_meta = json.loads(capsys.readouterr().out)
    assert imp_meta["imported_samples"] == 1 and imp_meta["model_names"] == ["q4"]

    # 2. badge command (file output & stdout)
    svg_file = tmp_path / "drift.svg"
    assert main(["badge", "-i", str(EXAMPLES / "expected-pass.json"), "-o", str(svg_file)]) == 0
    _ = capsys.readouterr()
    assert "PASS 0.00" in svg_file.read_text(encoding="utf-8")

    # 3. diff --format markdown
    assert (
        main(
            [
                "diff",
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-drift.jsonl"),
                "--format",
                "markdown",
            ]
        )
        == 0
    )
    md_diff = capsys.readouterr().out
    assert "## Semantic Response Diff" in md_diff and "| Field Path |" in md_diff

    # 4. cost budget breach returns exit 1
    b_tok = tmp_path / "b.jsonl"
    c_tok = tmp_path / "c.jsonl"
    b_tok.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "b", "output": "a", "tokens_generated": 100})
        + "\n",
        encoding="utf-8",
    )
    c_tok.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "c", "output": "a", "tokens_generated": 200})
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "cost",
                "-b",
                str(b_tok),
                "-c",
                str(c_tok),
                "--max-token-increase-pct",
                "25",
                "--format",
                "json",
            ]
        )
        == 1
    )
    cost_payload = json.loads(capsys.readouterr().out)
    assert cost_payload["budget_passed"] is False

    # 5. check-tools --max-tool-calls & --allow-resource
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"kind": "read", "resource": "src/a.py@1", "start": 0, "end": 10}\n'
        '{"kind": "edit", "resource": "src/a.py@1"}\n'
        '{"kind": "edit", "resource": "secret/k.env@1"}\n',
        encoding="utf-8",
    )
    assert (
        main(
            [
                "check-tools",
                "-i",
                str(trace),
                "--resource",
                "src/a.py@1",
                "--end",
                "10",
                "--trace-complete",
                "--max-tool-calls",
                "2",
                "--allow-resource",
                "src/",
                "--format",
                "json",
            ]
        )
        == 1
    )
    tool_out = json.loads(capsys.readouterr().out)
    assert tool_out["max_tool_calls_check"]["status"] == "FAIL"
    assert tool_out["allowed_resources_check"]["status"] == "FAIL"


def test_cli_distribution_wizard_logprobs_check_quality_and_capture_top_logprobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import math

    # 1. distribution command (json, markdown, html, threshold exit 1)
    base_d = tmp_path / "dist_base.jsonl"
    cand_d = tmp_path / "dist_cand.jsonl"
    base_d.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt_id": f"p{i}",
                    "model_name": "b",
                    "output": json.dumps({"label": "high", "score": 0.9}),
                }
            )
            for i in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    cand_d.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt_id": f"p{i}",
                    "model_name": "c",
                    "output": json.dumps({"label": "high" if i < 2 else "low", "score": 0.5}),
                }
            )
            for i in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["distribution", "-b", str(base_d), "-c", str(cand_d), "--format", "json"]) == 0
    dist_json = json.loads(capsys.readouterr().out)
    assert dist_json["max_tvd"] == pytest.approx(0.5)

    assert (
        main(
            [
                "distribution",
                "-b",
                str(base_d),
                "-c",
                str(cand_d),
                "--max-tvd",
                "0.2",
                "--format",
                "json",
            ]
        )
        == 1
    )
    dist_fail = json.loads(capsys.readouterr().out)
    assert dist_fail["passed"] is False and dist_fail["threshold_failures"]

    dist_html = tmp_path / "dist.html"
    assert (
        main(
            [
                "distribution",
                "-b",
                str(base_d),
                "-c",
                str(cand_d),
                "--format",
                "html",
                "-o",
                str(dist_html),
            ]
        )
        == 0
    )
    _ = capsys.readouterr()
    assert "<table" in dist_html.read_text(encoding="utf-8")

    # 2. wizard command (--preset + -o file, and --non-interactive stdout)
    wiz_out = tmp_path / "drift.toml"
    assert main(["wizard", "--preset", "classification", "-o", str(wiz_out)]) == 0
    wiz_stdout = capsys.readouterr().out
    assert "Next steps" in wiz_stdout
    assert "max_composite_drift = 0.1" in wiz_out.read_text(encoding="utf-8")

    assert main(["wizard", "--non-interactive"]) == 0
    assert "preset: extraction" in capsys.readouterr().out

    # 3. logprobs command (passes on clean, exits 1 when threshold breached or incomplete)
    lp_b = tmp_path / "lp_b.jsonl"
    lp_c = tmp_path / "lp_c.jsonl"
    lp_b.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "model_name": "fp16",
                "output": "yes",
                "metadata": {
                    "logprobs": [
                        {
                            "token": "yes",
                            "logprob": math.log(0.99),
                            "top_logprobs": [
                                {"token": "yes", "logprob": math.log(0.99)},
                                {"token": "no", "logprob": math.log(0.01)},
                            ],
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lp_c.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "model_name": "q4",
                "output": "yes",
                "metadata": {
                    "logprobs": [
                        {
                            "token": "yes",
                            "logprob": math.log(0.55),
                            "top_logprobs": [
                                {"token": "yes", "logprob": math.log(0.55)},
                                {"token": "no", "logprob": math.log(0.45)},
                            ],
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["logprobs", "-b", str(lp_b), "-c", str(lp_c), "--format", "json"]) == 0
    lp_res = json.loads(capsys.readouterr().out)
    assert lp_res["complete"] is True and lp_res["max_top_k_jsd"] > 0.2

    assert (
        main(
            [
                "logprobs",
                "-b",
                str(lp_b),
                "-c",
                str(lp_c),
                "--max-jsd",
                "0.1",
                "--format",
                "json",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["passed"] is False

    # Incomplete logprobs with a limit must fail honestly
    assert (
        main(
            [
                "logprobs",
                "-b",
                str(base_d),
                "-c",
                str(cand_d),
                "--max-jsd",
                "0.5",
            ]
        )
        == 1
    )
    assert "incomplete logprobs coverage" in capsys.readouterr().out

    # 4. check-quality command
    q_b = tmp_path / "q_b.jsonl"
    q_c = tmp_path / "q_c.jsonl"
    q_b.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "model_name": "fp16",
                "output": "Here is the complete technical answer with clear reasoning steps.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    q_c.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "model_name": "q4",
                "output": "I'm sorry, I cannot assist with that request.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["check-quality", "-b", str(q_b), "-c", str(q_c), "--format", "json"]) == 1
    q_payload = json.loads(capsys.readouterr().out)
    assert q_payload["passed"] is False and q_payload["counts_by_check"]["refusal_flip"] == 1

    assert (
        main(["check-quality", "-b", str(q_b), "-c", str(q_c), "--no-fail-on-issues"]) == 0
    )
    assert "ISSUES DETECTED" in capsys.readouterr().out

    # 5. capture --top-logprobs
    suite = tmp_path / "lp_suite.json"
    suite.write_text(json.dumps([{"prompt_id": "p1", "prompt": "Say yes"}]), encoding="utf-8")
    lp_out = tmp_path / "lp_captured.jsonl"
    seen_bodies: list[dict[str, Any]] = []

    def fake_lp_post(
        url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> dict[str, Any]:
        del url, headers, timeout
        seen_bodies.append(body)
        return {
            "choices": [
                {
                    "message": {"content": "yes"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "yes",
                                "logprob": -0.05,
                                "top_logprobs": [{"token": "yes", "logprob": -0.05}],
                            }
                        ]
                    },
                }
            ]
        }

    monkeypatch.setattr("llm_drift_bench.capture._default_transport", fake_lp_post)
    assert (
        main(
            [
                "capture",
                "-s",
                str(suite),
                "--endpoint",
                "http://localhost:11434/v1",
                "--model",
                "llama3:8b",
                "--top-logprobs",
                "5",
                "-o",
                str(lp_out),
            ]
        )
        == 0
    )
    _ = capsys.readouterr()
    assert seen_bodies[0]["logprobs"] is True and seen_bodies[0]["top_logprobs"] == 5
    captured_row = json.loads(lp_out.read_text(encoding="utf-8").strip())
    assert captured_row["metadata"]["logprobs"][0]["token"] == "yes"


def test_cli_init_sample_pack_audit_and_check_formats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack_dir = tmp_path / "rag_pack"
    assert main(["init-sample", "--dir", str(pack_dir), "--pack", "rag"]) == 0
    assert "Created 'rag' starter pack" in capsys.readouterr().out
    assert (pack_dir / "drift.toml").is_file()

    # Refuses overwriting existing pack directory
    assert main(["init-sample", "--dir", str(pack_dir), "--pack", "rag"]) == 2
    _ = capsys.readouterr()

    # audit passes on candidate-pass and fails on candidate-drift
    audit_html = tmp_path / "audit.html"
    assert (
        main(
            [
                "audit",
                "-s",
                str(pack_dir / "prompts.json"),
                "-b",
                str(pack_dir / "baseline.jsonl"),
                "-c",
                str(pack_dir / "candidate-pass.jsonl"),
                "--config",
                str(pack_dir / "drift.toml"),
                "--format",
                "html",
                "-o",
                str(audit_html),
            ]
        )
        == 0
    )
    _ = capsys.readouterr()
    assert "PASSED" in audit_html.read_text(encoding="utf-8")

    assert (
        main(
            [
                "audit",
                "-s",
                str(pack_dir / "prompts.json"),
                "-b",
                str(pack_dir / "baseline.jsonl"),
                "-c",
                str(pack_dir / "candidate-drift.jsonl"),
                "--config",
                str(pack_dir / "drift.toml"),
                "--format",
                "json",
            ]
        )
        == 1
    )
    audit_payload = json.loads(capsys.readouterr().out)
    assert audit_payload["passed"] is False and audit_payload["failure_reasons"]

    # check-formats pass and fail
    fmt_file = tmp_path / "fmt.jsonl"
    fmt_file.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "model_name": "m",
                "output": json.dumps({"email": "dev@example.com", "due": "2026-11-25"}),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "check-formats",
                "-i",
                str(fmt_file),
                "--rule",
                "$.email=email",
                "--rule",
                "$.due=date",
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["passed"] is True

    bad_fmt = tmp_path / "bad_fmt.jsonl"
    bad_fmt.write_text(
        json.dumps(
            {
                "prompt_id": "p1",
                "model_name": "m",
                "output": json.dumps({"email": "not-an-email"}),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["check-formats", "-i", str(bad_fmt), "--rule", "$.email=email"]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_cli_extras_text_outputs_stdout_modes_and_error_guards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # subsample stdout (no -o)
    assert main(["subsample", "-s", str(EXAMPLES / "extraction-suite.json"), "--size", "1"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1

    # cost text mode with explicit prices + budget check text output
    b_tok = tmp_path / "b_tok.jsonl"
    c_tok = tmp_path / "c_tok.jsonl"
    b_tok.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "b", "output": "x", "tokens_generated": 100})
        + "\n",
        encoding="utf-8",
    )
    c_tok.write_text(
        json.dumps({"prompt_id": "p1", "model_name": "c", "output": "x", "tokens_generated": 150})
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "cost",
                "-b",
                str(b_tok),
                "-c",
                str(c_tok),
                "--baseline-price",
                "2.0",
                "--candidate-price",
                "2.0",
                "--max-token-increase-pct",
                "10",
            ]
        )
        == 1
    )
    cost_txt = capsys.readouterr().out
    assert "Output tokens" in cost_txt and "Output cost" in cost_txt and "FAIL" in cost_txt

    # snapshot missing sub-action returns input error (2)
    assert main(["snapshot"]) == 2
    _ = capsys.readouterr()

    # badge stdout mode (no -o)
    assert main(["badge", "-i", str(EXAMPLES / "expected-pass.json")]) == 0
    assert capsys.readouterr().out.startswith("<svg ")

    # import-captures text mode for openai-batch and anthropic-batch
    oa_in = tmp_path / "oa.jsonl"
    oa_in.write_text(
        json.dumps(
            {
                "custom_id": "p1",
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": "ok"}}]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    oa_out = tmp_path / "oa_out.jsonl"
    assert (
        main(
            ["import-captures", "--from", "openai-batch", "-i", str(oa_in), "-o", str(oa_out)]
        )
        == 0
    )
    assert "Imported 1 sample(s)" in capsys.readouterr().out

    ant_in = tmp_path / "ant.jsonl"
    ant_in.write_text(
        json.dumps(
            {
                "custom_id": "p1",
                "result": {
                    "type": "succeeded",
                    "message": {"content": [{"type": "text", "text": "ok"}]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ant_out = tmp_path / "ant_out.jsonl"
    assert (
        main(
            [
                "import-captures",
                "--from",
                "anthropic-batch",
                "-i",
                str(ant_in),
                "-o",
                str(ant_out),
            ]
        )
        == 0
    )
    assert "anthropic-batch" in capsys.readouterr().out

    # distribution markdown, text failure, and out-of-range limit error
    assert (
        main(
            [
                "distribution",
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-drift.jsonl"),
                "--format",
                "markdown",
            ]
        )
        == 0
    )
    assert "# Cross-Prompt JSON Field Distribution Drift" in capsys.readouterr().out

    assert (
        main(
            [
                "distribution",
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-drift.jsonl"),
                "--max-wasserstein",
                "0.01",
            ]
        )
        == 1
    )
    assert "Threshold failures:" in capsys.readouterr().out

    assert (
        main(
            [
                "distribution",
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-drift.jsonl"),
                "--max-tvd",
                "2.0",
            ]
        )
        == 2
    )
    _ = capsys.readouterr()

    # logprobs invalid negative entropy delta
    assert (
        main(
            [
                "logprobs",
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-pass.jsonl"),
                "--max-entropy-delta",
                "-1.0",
            ]
        )
        == 2
    )
    _ = capsys.readouterr()

    # audit markdown and text stdout outputs
    assert (
        main(
            [
                "audit",
                "-s",
                str(EXAMPLES / "extraction-suite.json"),
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-pass.jsonl"),
                "--format",
                "markdown",
            ]
        )
        == 0
    )
    assert "# Unified Drift Audit: PASSED" in capsys.readouterr().out

    assert (
        main(
            [
                "audit",
                "-s",
                str(EXAMPLES / "extraction-suite.json"),
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-pass.jsonl"),
            ]
        )
        == 0
    )
    assert "LLM Drift Bench Unified Audit [PASSED]" in capsys.readouterr().out


def test_cli_segmented_distribution_persist_windows_and_check_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    # 1. Segmented distribution (--group-by category with -s suite)
    assert (
        main(
            [
                "distribution",
                "-s",
                str(EXAMPLES / "extraction-suite.json"),
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-drift.jsonl"),
                "--group-by",
                "category",
                "--shift-threshold",
                "0.15",
                "--max-wasserstein",
                "0.1",
                "--format",
                "json",
            ]
        )
        == 1
    )
    seg_json = json.loads(capsys.readouterr().out)
    assert seg_json["group_by"] == "category"
    assert seg_json["passed"] is False
    assert seg_json["shift_scope"] == "localized"
    assert seg_json["shifted_segments"] == ["extraction"]
    assert "extraction" in seg_json["segments"]

    seg_html = tmp_path / "seg.html"
    assert (
        main(
            [
                "distribution",
                "-s",
                str(EXAMPLES / "extraction-suite.json"),
                "-b",
                str(EXAMPLES / "baseline.jsonl"),
                "-c",
                str(EXAMPLES / "candidate-pass.jsonl"),
                "--group-by",
                "category",
                "--format",
                "html",
                "-o",
                str(seg_html),
            ]
        )
        == 0
    )
    _ = capsys.readouterr()
    assert "Segmented JSON Field Distribution" in seg_html.read_text(encoding="utf-8")

    # 2. History persistence windows & distribution TVD tracking
    ledger = tmp_path / "persist.jsonl"
    for tvd in (0.02, 0.03, 0.02, 0.03):
        assert (
            main(
                [
                    "history",
                    "record",
                    "--ledger",
                    str(ledger),
                    "-i",
                    str(EXAMPLES / "expected-pass.json"),
                    "--distribution-tvd",
                    str(tvd),
                ]
            )
            == 0
        )
    # One noisy spike -> stable when persist_windows=2
    assert (
        main(
            [
                "history",
                "record",
                "--ledger",
                str(ledger),
                "-i",
                str(EXAMPLES / "expected-drift.json"),
                "--distribution-tvd",
                "0.40",
            ]
        )
        == 0
    )
    _ = capsys.readouterr()
    assert (
        main(
            [
                "history",
                "check",
                "--ledger",
                str(ledger),
                "--window",
                "3",
                "--persist-windows",
                "2",
                "--max-tvd-increase",
                "0.10",
            ]
        )
        == 0
    )
    assert "Transient shift (1/2 consecutive runs)" in capsys.readouterr().out

    # Second consecutive shifted run -> regressed!
    assert (
        main(
            [
                "history",
                "record",
                "--ledger",
                str(ledger),
                "-i",
                str(EXAMPLES / "expected-drift.json"),
                "--distribution-tvd",
                "0.45",
            ]
        )
        == 0
    )
    _ = capsys.readouterr()
    assert (
        main(
            [
                "history",
                "check",
                "--ledger",
                str(ledger),
                "--window",
                "3",
                "--persist-windows",
                "2",
                "--max-tvd-increase",
                "0.10",
                "--format",
                "json",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["persistent_runs"] == 2

    # 3. check-consumer via pre-recorded decision files AND via --validator-module
    b_dec = tmp_path / "b_dec.jsonl"
    c_dec = tmp_path / "c_dec.jsonl"
    b_dec.write_text(
        '{"prompt_id": "p1", "accepted": true}\n{"prompt_id": "p2", "accepted": true}\n',
        encoding="utf-8",
    )
    c_dec.write_text(
        '{"prompt_id": "p1", "accepted": true}\n'
        '{"prompt_id": "p2", "accepted": false, "reason": "invalid total"}\n',
        encoding="utf-8",
    )
    assert (
        main(
            [
                "check-consumer",
                "--baseline-decisions",
                str(b_dec),
                "--candidate-decisions",
                str(c_dec),
                "--max-rejection-increase",
                "0.1",
                "--format",
                "markdown",
            ]
        )
        == 1
    )
    assert "# Downstream Consumer Rejection Check: FAILED" in capsys.readouterr().out

    # Via --validator-module
    val_mod = tmp_path / "cli_consumer_val.py"
    val_mod.write_text(
        "def accept_all(_sample):\n    return True\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert (
            main(
                [
                    "check-consumer",
                    "-b",
                    str(EXAMPLES / "baseline.jsonl"),
                    "-c",
                    str(EXAMPLES / "candidate-pass.jsonl"),
                    "--validator-module",
                    "cli_consumer_val:accept_all",
                    "--max-rejection-rate",
                    "0.0",
                    "--format",
                    "json",
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out)["passed"] is True
    finally:
        sys.modules.pop("cli_consumer_val", None)

    # Misconfigured flags return exit 2
    assert main(["check-consumer"]) == 2
    _ = capsys.readouterr()


def test_cli_distribution_and_audit_fail_on_widespread_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "routes.json"
    suite.write_text(
        json.dumps(
            [
                {
                    "prompt_id": f"r{r}_p{i}",
                    "category": "classification",
                    "prompt": f"Classify {r}-{i}",
                    "metadata": {"route": f"route_{r}"},
                }
                for r in ("a", "b")
                for i in range(2)
            ]
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base.jsonl"
    cand_local = tmp_path / "cand_local.jsonl"
    cand_wide = tmp_path / "cand_wide.jsonl"
    base.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt_id": f"r{r}_p{i}",
                    "model_name": "b",
                    "output": json.dumps({"tier": "high"}),
                }
            )
            for r in ("a", "b")
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    cand_local.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt_id": f"r{r}_p{i}",
                    "model_name": "c",
                    "output": json.dumps({"tier": "low" if r == "a" else "high"}),
                }
            )
            for r in ("a", "b")
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    cand_wide.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt_id": f"r{r}_p{i}",
                    "model_name": "c",
                    "output": json.dumps({"tier": "low"}),
                }
            )
            for r in ("a", "b")
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    # distribution --fail-on-widespread without --group-by -> exit 2
    assert (
        main(["distribution", "-b", str(base), "-c", str(cand_wide), "--fail-on-widespread"]) == 2
    )
    _ = capsys.readouterr()

    # localized shift passes --fail-on-widespread
    assert (
        main(
            [
                "distribution",
                "-s",
                str(suite),
                "-b",
                str(base),
                "-c",
                str(cand_local),
                "--group-by",
                "metadata:route",
                "--fail-on-widespread",
                "--format",
                "json",
            ]
        )
        == 0
    )
    local_out = json.loads(capsys.readouterr().out)
    assert local_out["shift_scope"] == "localized" and local_out["passed"] is True

    # widespread shift fails --fail-on-widespread (exit 1)
    assert (
        main(
            [
                "distribution",
                "-s",
                str(suite),
                "-b",
                str(base),
                "-c",
                str(cand_wide),
                "--group-by",
                "metadata:route",
                "--fail-on-widespread",
                "--format",
                "json",
            ]
        )
        == 1
    )
    wide_out = json.loads(capsys.readouterr().out)
    assert wide_out["shift_scope"] == "widespread" and wide_out["passed"] is False
    assert any("widespread shift" in msg for msg in wide_out["threshold_failures"])

    # audit --fail-on-widespread without --group-by -> exit 2
    assert main(["audit", "-b", str(base), "-c", str(cand_wide), "--fail-on-widespread"]) == 2
    _ = capsys.readouterr()

    # audit --group-by with --fail-on-widespread fails on widespread shift
    assert (
        main(
            [
                "audit",
                "-s",
                str(suite),
                "-b",
                str(base),
                "-c",
                str(cand_wide),
                "--group-by",
                "metadata:route",
                "--shift-threshold",
                "0.15",
                "--fail-on-widespread",
                "--format",
                "json",
            ]
        )
        == 1
    )
    audit_wide = json.loads(capsys.readouterr().out)
    assert audit_wide["passed"] is False
    assert audit_wide["segmented_distribution"]["shift_scope"] == "widespread"
    assert any("widespread" in r.lower() for r in audit_wide["failure_reasons"])


def test_cli_gate_with_consumer_decisions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    b_dec = tmp_path / "b_dec.jsonl"
    c_ok = tmp_path / "c_ok.jsonl"
    c_bad = tmp_path / "c_bad.jsonl"
    b_dec.write_text(
        '{"prompt_id": "p1", "accepted": true}\n{"prompt_id": "p2", "accepted": true}\n',
        encoding="utf-8",
    )
    c_ok.write_text(
        '{"prompt_id": "p1", "accepted": true}\n{"prompt_id": "p2", "accepted": true}\n',
        encoding="utf-8",
    )
    c_bad.write_text(
        '{"prompt_id": "p1", "accepted": true}\n'
        '{"prompt_id": "p2", "accepted": false, "reason": "bad_sku"}\n',
        encoding="utf-8",
    )

    # Half-configured consumer files fail cleanly with exit 2
    assert (
        main(
            [
                "gate",
                "-i",
                str(EXAMPLES / "expected-pass.json"),
                "--baseline-decisions",
                str(b_dec),
            ]
        )
        == 2
    )
    _ = capsys.readouterr()

    # Passing consumer decisions approve release
    assert (
        main(
            [
                "gate",
                "-i",
                str(EXAMPLES / "expected-pass.json"),
                "--baseline-decisions",
                str(b_dec),
                "--candidate-decisions",
                str(c_ok),
                "--max-rejection-increase",
                "0.1",
                "--format",
                "json",
            ]
        )
        == 0
    )
    ok_gate = json.loads(capsys.readouterr().out)
    assert ok_gate["approved"] is True
    assert any(c["name"] == "consumer_rejection" and c["passed"] for c in ok_gate["checks"])

    # Regressed consumer decisions block release by default
    assert (
        main(
            [
                "gate",
                "-i",
                str(EXAMPLES / "expected-pass.json"),
                "--baseline-decisions",
                str(b_dec),
                "--candidate-decisions",
                str(c_bad),
                "--max-rejection-increase",
                "0.1",
                "--format",
                "json",
            ]
        )
        == 1
    )
    bad_gate = json.loads(capsys.readouterr().out)
    assert bad_gate["approved"] is False
    assert any(
        c["name"] == "consumer_rejection" and not c["passed"] and c["blocking"]
        for c in bad_gate["checks"]
    )

    # --no-fail-on-consumer downgrades consumer check to advisory WARN
    assert (
        main(
            [
                "gate",
                "-i",
                str(EXAMPLES / "expected-pass.json"),
                "--baseline-decisions",
                str(b_dec),
                "--candidate-decisions",
                str(c_bad),
                "--max-rejection-increase",
                "0.1",
                "--no-fail-on-consumer",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "GO (APPROVED)" in out and "WARN" in out








