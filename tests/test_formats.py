"""Tests for standalone semantic JSON field format validation."""

from __future__ import annotations

import json

import pytest

from llm_drift_bench.formats import (
    SUPPORTED_FORMATS,
    check_field_formats,
    format_check_report_text,
    parse_format_rule_spec,
    validate_format_value,
)
from llm_drift_bench.models import ModelSample


@pytest.mark.parametrize(
    ("fmt", "good", "bad"),
    [
        ("date", "2026-11-25", "2026-02-30"),
        ("date-time", "2026-11-25T14:30:00Z", "2026-11-25 99:00:00"),
        ("email", "ops+alerts@example.co.uk", "bad..dot@example.com"),
        ("uri", "https://api.example.com/v1/items?a=1", "ftp://not-http.example"),
        ("uuid", "550e8400-e29b-41d4-a716-446655440000", "not-a-uuid"),
        ("ipv4", "192.168.1.42", "999.1.2.3"),
        ("iso-currency", "USD", "usd"),
    ],
)
def test_validate_format_value_accepts_good_and_rejects_bad(
    fmt: str, good: str, bad: str
) -> None:
    ok_good, err_good = validate_format_value(good, fmt)  # type: ignore[arg-type]
    assert ok_good is True and err_good is None

    ok_bad, err_bad = validate_format_value(bad, fmt)  # type: ignore[arg-type]
    assert ok_bad is False and err_bad is not None

    ok_non_str, err_non_str = validate_format_value(123, fmt)  # type: ignore[arg-type]
    assert ok_non_str is False and "Expected string" in (err_non_str or "")


def test_check_field_formats_exact_and_wildcard_paths() -> None:
    samples = [
        ModelSample(
            prompt_id="p1",
            model_name="m",
            output=json.dumps(
                {
                    "due_date": "2026-12-01",
                    "currency": "USD",
                    "contact": "billing@acme.io",
                    "links": [
                        {
                            "url": "https://acme.io/inv/1",
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                        },
                        {"url": "not a url", "id": "bad-id"},
                    ],
                }
            ),
        ),
        ModelSample(
            prompt_id="p2",
            model_name="m",
            output=json.dumps({"due_date": None, "currency": "EUR", "contact": "a@b.co"}),
        ),
    ]
    rules = {
        "$.due_date": "date",
        "$.currency": "iso-currency",
        "$.contact": "email",
        "$.links[*].url": "uri",
        "$.links[*].id": "uuid",
    }

    strict = check_field_formats(samples, rules)  # type: ignore[arg-type]
    assert strict.passed is False
    assert strict.checked_values == 10
    assert strict.failed_values == 3
    failed_paths = {(i.prompt_id, i.path, i.expected_format) for i in strict.issues}
    assert ("p1", "$.links[1].url", "uri") in failed_paths
    assert ("p1", "$.links[1].id", "uuid") in failed_paths
    assert ("p2", "$.due_date", "date") in failed_paths

    nullable = check_field_formats(samples, rules, allow_null=True)  # type: ignore[arg-type]
    assert nullable.failed_values == 2
    assert "FAILED" in format_check_report_text(nullable)
    assert nullable.to_dict()["failed_values"] == 2


def test_parse_format_rule_spec_and_errors() -> None:
    assert parse_format_rule_spec("$.items[*].date = DATE") == ("$.items[*].date", "date")
    assert len(SUPPORTED_FORMATS) == 7

    with pytest.raises(ValueError, match="JSONPATH=FORMAT"):
        parse_format_rule_spec("$.email")
    with pytest.raises(ValueError, match="start with"):
        parse_format_rule_spec("email=email")
    with pytest.raises(ValueError, match="Unsupported format"):
        parse_format_rule_spec("$.x=credit-card")
    with pytest.raises(ValueError, match="cannot be empty"):
        check_field_formats([], {"$.x": "email"})
    with pytest.raises(ValueError, match="At least one"):
        check_field_formats([ModelSample(prompt_id="p1", output="{}")], {})

    # Additional validation & non-JSON skip branches
    with pytest.raises(ValueError, match="Unsupported format"):
        validate_format_value("x", "unknown")  # type: ignore[arg-type]
    assert validate_format_value("   ", "email")[0] is False
    assert validate_format_value("not-a-date", "date")[0] is False
    assert validate_format_value("not-a-datetime", "date-time")[0] is False

    sample_ok = [ModelSample(prompt_id="p1", output="plain non-json text")]
    with pytest.raises(ValueError, match="start with"):
        check_field_formats(sample_ok, {"no_dollar": "email"})
    with pytest.raises(ValueError, match="Unsupported format"):
        check_field_formats(sample_ok, {"$.x": "bogus"})  # type: ignore[arg-type]
    skipped = check_field_formats(sample_ok, {"$.email": "email"})
    assert skipped.passed is True and skipped.checked_values == 0


def test_format_markdown_csv_and_cli(
    tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    import csv
    import io
    from pathlib import Path

    from llm_drift_bench.cli import main
    from llm_drift_bench.csv_export import FORMAT_CSV_COLUMNS, export_format_csv_report
    from llm_drift_bench.formats import format_check_report_markdown

    assert isinstance(tmp_path, Path)
    clean_rep = check_field_formats(
        [ModelSample("p1", '{"email": "dev@example.com"}')], {"$.email": "email"}
    )
    assert "## Semantic Field Format Check (PASS)" in format_check_report_markdown(clean_rep)

    bad_rep = check_field_formats(
        [ModelSample("=p1", '{"email": "=bad"}')], {"$.email": "email"}
    )
    md = format_check_report_markdown(bad_rep)
    assert "## Semantic Field Format Check (FAIL)" in md and "`$.email`" in md

    csv_file = tmp_path / "formats.csv"
    csv_text = export_format_csv_report(bad_rep, output_path=csv_file)
    assert csv_file.read_text(encoding="utf-8") == csv_text
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert tuple(rows[0].keys()) == FORMAT_CSV_COLUMNS
    assert rows[0]["prompt_id"] == "'=p1"

    with pytest.raises(ValueError, match="FormatReport"):
        export_format_csv_report({})  # type: ignore[arg-type]

    cap = tmp_path / "cap.jsonl"
    cap.write_text('{"prompt_id": "p1", "output": "{\\"email\\": \\"not-an-email\\"}"}\n', "utf-8")
    md_out = tmp_path / "fmt.md"
    assert (
        main(
            [
                "check-formats",
                "-i",
                str(cap),
                "--rule",
                "$.email=email",
                "--format",
                "markdown",
                "-o",
                str(md_out),
            ]
        )
        == 1
    )
    assert "Semantic Field Format Check (FAIL)" in md_out.read_text(encoding="utf-8")
    capsys.readouterr()

    assert (
        main(
            [
                "check-formats",
                "-i",
                str(cap),
                "--rule",
                "$.email=email",
                "--format",
                "csv",
            ]
        )
        == 1
    )
    cli_rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(cli_rows) == 1 and cli_rows[0]["expected_format"] == "email"
