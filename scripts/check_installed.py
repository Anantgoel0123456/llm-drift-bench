"""Install a distribution in a fresh venv and exercise it outside the checkout."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

FORBIDDEN_FRAGMENTS = ("_vendor", "OPIK_LICENSE", "OPIK_NOTICE", "commit-reviewed-changes")


def archive_file_sizes(artifact: Path) -> dict[str, int]:
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as wheel:
            return {item.filename: item.file_size for item in wheel.infolist() if not item.is_dir()}
    with tarfile.open(artifact) as sdist:
        return {item.name: item.size for item in sdist.getmembers() if item.isfile()}


def check_archive_contents(artifact: Path) -> None:
    """Inspect the distribution itself, before and independent of installation."""
    files = archive_file_sizes(artifact)
    leaked = [n for n in files if any(part in n for part in FORBIDDEN_FRAGMENTS)]
    if leaked:
        raise SystemExit(f"{artifact.name} contains excluded files: {leaked}")
    is_wheel = artifact.suffix == ".whl"
    package_root = "" if is_wheel else artifact.name.removesuffix(".tar.gz") + "/src/"
    if package_root + "llm_drift_bench/py.typed" not in files:
        raise SystemExit(f"{artifact.name} is missing llm_drift_bench/py.typed")
    if is_wheel:
        metadata_root = "-".join(artifact.name.split("-")[:2]) + ".dist-info"
        if not files.get(metadata_root + "/licenses/LICENSE"):
            raise SystemExit(f"{artifact.name} is missing its packaged LICENSE")
    elif not files.get(artifact.name.removesuffix(".tar.gz") + "/LICENSE"):
        raise SystemExit(f"{artifact.name} is missing its root LICENSE")
    print(f"Archive contents verified: {artifact.name} ({len(files)} files)")


def main() -> None:
    artifact = Path(sys.argv[1]).resolve(strict=True)
    check_archive_contents(artifact)
    with tempfile.TemporaryDirectory(prefix="drift-wheel-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        cli = scripts / ("llm-drift-bench.exe" if os.name == "nt" else "llm-drift-bench")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        subprocess.run([str(python), "-m", "pip", "install", str(artifact)],
                       cwd=root, env=env, check=True)
        for command in (
            [str(cli), "--version"], [str(cli), "--help"],
            [str(python), "-m", "llm_drift_bench", "--help"],
            [str(python), "-c", "import llm_drift_bench; print(llm_drift_bench.__file__)"],
            [
                str(python),
                "-c",
                "import sys, llm_drift_bench, llm_drift_bench.cli; "
                "assert 'pydantic' not in sys.modules and 'rich' not in sys.modules",
            ],
        ):
            subprocess.run(command, cwd=root, env=env, check=True)
        # Exercise the installed engine, exports and package data, never source imports.
        code = '''
from pathlib import Path
from llm_drift_bench import DriftBenchmarkEngine, ModelSample, export_html_report
import llm_drift_bench
package = Path(llm_drift_bench.__file__).parent
assert (package / "py.typed").is_file()
assert not (package / "_vendor").exists()
baseline = [ModelSample(prompt_id="p", output='{"x":1}')]
candidate = [ModelSample(prompt_id="p", output='{"x":9}')]
engine = DriftBenchmarkEngine()
assert engine.evaluate(baseline, baseline).passed
failed = engine.evaluate(baseline, candidate)
assert not failed.passed
assert "FAILED" in export_html_report(failed)

from llm_drift_bench import validate_output
assert not validate_output('{"total":100,"total":1}').is_valid_json
assert validate_output('{"n":1.0}', {"type":"object","properties":{"n":{"type":"integer"}}}).passed
try:
    ModelSample.from_dict({"prompt_id": "p", "output": {"ok": True}})
except ValueError:
    pass
else:
    raise AssertionError("structured output must be rejected")
'''
        subprocess.run([str(python), "-c", code], cwd=root, env=env, check=True)
        examples = Path(__file__).resolve().parents[1] / "examples"
        benchmark = [str(cli), "run", "-s", str(examples / "extraction-suite.json"),
                     "-b", str(examples / "baseline.jsonl"), "--format", "json"]
        for name, expected in (("pass", 0), ("drift", 1)):
            outcome = subprocess.run(
                [*benchmark, "-c", str(examples / f"candidate-{name}.jsonl")],
                cwd=root, env=env, capture_output=True, text=True,
            )
            if outcome.returncode != expected:
                raise SystemExit(f"run {name}: exit {outcome.returncode}, want {expected}")
            assert json.loads(outcome.stdout)["passed"] is (expected == 0)
        invalid = root / "invalid capture.jsonl"
        invalid.write_text('\n{"id":"p","output":{"private":"hidden sample"}}', encoding="utf-8")
        outcome = subprocess.run([*benchmark, "-c", str(invalid)], cwd=root, env=env,
                                 capture_output=True, text=True)
        if outcome.returncode != 2 or outcome.stdout:
            raise SystemExit("Installed run must return exit 2 and empty stdout for invalid input")
        assert str(invalid) in outcome.stderr and "line 2" in outcome.stderr
        assert "hidden sample" not in outcome.stderr
        comparison = subprocess.run(
            [str(cli), "compare-reports", str(examples / "expected-drift.json"),
             str(examples / "expected-pass.json"), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        passing, failing = json.loads(comparison.stdout)["candidates"]
        assert passing["passed"] and not failing["passed"]
        assert failing["worst_prompt_id"] == "ticket-01"
        batch = subprocess.run(
            [str(cli), "compare-many", "-b", str(examples / "baseline.jsonl"),
             "-s", str(examples / "extraction-suite.json"),
             "-c", "pass", str(examples / "candidate-pass.jsonl"),
             "-c", "drift", str(examples / "candidate-drift.jsonl"),
             "--against", "pass", "--format", "json", "--no-fail-on-drift"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        changes = json.loads(batch.stdout)["changes"]
        assert [item["prompt_id"] for item in changes] == ["invoice-01", "ticket-01"]
        assert all(item["state"] == "regressed" for item in changes)
        # The installed CLI must map PASS, FAIL and UNKNOWN to distinct exit codes.
        trace = root / "trace.jsonl"
        trace.write_text(
            '{"kind": "read", "resource": "g@1", "start": 0, "end": 10}\n'
            '{"kind": "edit", "resource": "a@1"}\n', encoding="utf-8")
        for flags, expected in (
            (["--end", "10", "--trace-complete"], 0),
            (["--end", "20", "--trace-complete"], 1),
            (["--end", "10"], 3),
        ):
            outcome = subprocess.run(
                [str(cli), "check-tools", "-i", str(trace), "--resource", "g@1", *flags],
                cwd=root, env=env, check=False, capture_output=True,
            )
            if outcome.returncode != expected:
                raise SystemExit(f"check-tools {flags}: exit {outcome.returncode}, want {expected}")
        # Exercise 0.3.0 commands in the isolated installed environment.
        diff_out = subprocess.run(
            [str(cli), "diff", "-b", str(examples / "baseline.jsonl"),
             "-c", str(examples / "candidate-drift.jsonl"), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(diff_out.stdout)["changed_prompts"] == 2
        scorers_out = subprocess.run(
            [str(cli), "score", "--list-scorers", "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "python_ast" in json.loads(scorers_out.stdout)["scorers"]
        inferred = root / "inferred.json"
        subprocess.run(
            [str(cli), "infer-schema", "-i", str(examples / "baseline.jsonl"),
             "-o", str(inferred), "--strict-additional"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [str(cli), "validate", "-i", str(examples / "baseline.jsonl"),
             "-p", str(inferred), "--strict"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        doc_out = subprocess.run(
            [str(cli), "doctor", "--config", str(examples / "drift.toml"),
             "-s", str(examples / "extraction-suite.json"),
             "-b", str(examples / "baseline.jsonl"),
             "-c", str(examples / "candidate-pass.jsonl"), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(doc_out.stdout)["ok"] is True
        sarif_out = subprocess.run(
            [str(cli), "report", "-i", str(examples / "expected-drift.json"),
             "--format", "sarif"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(sarif_out.stdout)["version"] == "2.1.0"
        csv_out = subprocess.run(
            [str(cli), "report", "-i", str(examples / "expected-drift.json"),
             "--format", "csv"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "invoice-01" in csv_out.stdout
        # Exercise 0.4.0 commands in the isolated installed environment.
        assert "numeric_prose" in json.loads(scorers_out.stdout)["scorers"]
        mat_out = subprocess.run(
            [str(cli), "matrix", str(examples / "expected-drift.json"),
             str(examples / "expected-pass.json"), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert len(json.loads(mat_out.stdout)["rows"]) == 2
        lint_out = subprocess.run(
            [str(cli), "lint-suite", "-s", str(examples / "extraction-suite.json"),
             "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(lint_out.stdout)["errors"] == 0
        sub_suite = root / "sub.json"
        subprocess.run(
            [str(cli), "subsample", "-s", str(examples / "extraction-suite.json"),
             "--size", "1", "-o", str(sub_suite)],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert len(json.loads(sub_suite.read_text(encoding="utf-8"))) == 1
        red_out = root / "redacted.jsonl"
        subprocess.run(
            [str(cli), "redact", "-i", str(examples / "baseline.jsonl"), "-o", str(red_out)],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        gate_ok = subprocess.run(
            [str(cli), "gate", "-i", str(examples / "expected-pass.json"), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(gate_ok.stdout)["approved"] is True
        cost_out = subprocess.run(
            [str(cli), "cost", "-b", str(examples / "baseline.jsonl"),
             "-c", str(examples / "candidate-pass.jsonl"), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(cost_out.stdout)["complete"] is False
        snap_dir = root / "snap"
        snap_dir.mkdir()
        shutil.copy2(examples / "baseline.jsonl", snap_dir / "baseline.jsonl")
        manifest_file = snap_dir / "snapshot.manifest.json"
        subprocess.run(
            [str(cli), "snapshot", "create", "-b", str(snap_dir / "baseline.jsonl"),
             "-o", str(manifest_file)],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        ver_out = subprocess.run(
            [str(cli), "snapshot", "verify", "-i", str(manifest_file), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(ver_out.stdout)["valid"] is True
        badge_out = subprocess.run(
            [str(cli), "badge", "-i", str(examples / "expected-pass.json")],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert badge_out.stdout.startswith("<svg ") and "PASS 0.00" in badge_out.stdout
        csv_in = root / "in.csv"
        csv_in.write_text('prompt_id,output\np1,"{""ok"": true}"\n', encoding="utf-8")
        imp_out = root / "imported.jsonl"
        subprocess.run(
            [str(cli), "import-captures", "--from", "csv", "-i", str(csv_in), "-o", str(imp_out)],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert '"prompt_id": "p1"' in imp_out.read_text(encoding="utf-8")
        diff_md = subprocess.run(
            [str(cli), "diff", "-b", str(examples / "baseline.jsonl"),
             "-c", str(examples / "candidate-drift.jsonl"), "--format", "markdown"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "## Semantic Response Diff" in diff_md.stdout
        dist_out = subprocess.run(
            [str(cli), "distribution", "-b", str(examples / "baseline.jsonl"),
             "-c", str(examples / "candidate-drift.jsonl"), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(dist_out.stdout)["comparable_json_prompts"] == 2
        wiz_out = subprocess.run(
            [str(cli), "wizard", "--preset", "extraction", "--non-interactive"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "[thresholds]" in wiz_out.stdout and "preset: extraction" in wiz_out.stdout
        lp_out = subprocess.run(
            [str(cli), "logprobs", "-b", str(examples / "baseline.jsonl"),
             "-c", str(examples / "candidate-pass.jsonl"), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(lp_out.stdout)["complete"] is False
        qual_out = subprocess.run(
            [str(cli), "check-quality", "-b", str(examples / "baseline.jsonl"),
             "-c", str(examples / "candidate-pass.jsonl"), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(qual_out.stdout)["passed"] is True
        pack_dir = root / "pack_extraction"
        subprocess.run(
            [str(cli), "init-sample", "--dir", str(pack_dir), "--pack", "extraction"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        audit_out = subprocess.run(
            [
                str(cli), "audit",
                "-s", str(pack_dir / "prompts.json"),
                "-b", str(pack_dir / "baseline.jsonl"),
                "-c", str(pack_dir / "candidate-pass.jsonl"),
                "--config", str(pack_dir / "drift.toml"),
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(audit_out.stdout)["passed"] is True
        fmt_out = subprocess.run(
            [
                str(cli), "check-formats",
                "-i", str(pack_dir / "baseline.jsonl"),
                "--rule", "$.currency=iso-currency",
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(fmt_out.stdout)["passed"] is True
        seg_out = subprocess.run(
            [
                str(cli), "distribution",
                "-s", str(examples / "extraction-suite.json"),
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--group-by", "category",
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "extraction" in json.loads(seg_out.stdout)["segments"]
        cons_b = root / "cons_b.jsonl"
        cons_c = root / "cons_c.jsonl"
        cons_b.write_text('{"prompt_id": "p1", "accepted": true}\n', encoding="utf-8")
        cons_c.write_text('{"prompt_id": "p1", "accepted": true}\n', encoding="utf-8")
        cons_out = subprocess.run(
            [
                str(cli), "check-consumer",
                "--baseline-decisions", str(cons_b),
                "--candidate-decisions", str(cons_c),
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(cons_out.stdout)["passed"] is True
        roll_out = subprocess.run(
            [
                str(cli), "rolling-distribution",
                "-w", str(examples / "baseline.jsonl"),
                "-w", str(examples / "candidate-pass.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(roll_out.stdout)["passed"] is True
        gate_full = subprocess.run(
            [
                str(cli), "gate",
                "-i", str(examples / "expected-pass.json"),
                "--baseline-decisions", str(cons_b),
                "--candidate-decisions", str(cons_c),
                "--baseline-capture", str(examples / "baseline.jsonl"),
                "--candidate-capture", str(examples / "candidate-pass.jsonl"),
                "-s", str(examples / "extraction-suite.json"),
                "--group-by", "category",
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(gate_full.stdout)["approved"] is True
        cons_html = root / "consumer.html"
        subprocess.run(
            [
                str(cli), "check-consumer",
                "--baseline-decisions", str(cons_b),
                "--candidate-decisions", str(cons_c),
                "--format", "html", "-o", str(cons_html),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<table" in cons_html.read_text(encoding="utf-8")
        roll_html = root / "rolling.html"
        subprocess.run(
            [
                str(cli), "rolling-distribution",
                "-w", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "html", "-o", str(roll_html),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<table" in roll_html.read_text(encoding="utf-8")
        emb_out = subprocess.run(
            [
                str(cli), "check-embeddings",
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(emb_out.stdout)["complete"] is False
        case_dir = examples / "case-study"
        case_doc = subprocess.run(
            [
                str(cli), "doctor",
                "--config", str(case_dir / "drift.toml"),
                "-s", str(case_dir / "prompts.json"),
                "-b", str(case_dir / "baseline-fp16.jsonl"),
                "-c", str(case_dir / "candidate-q8.jsonl"),
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(case_doc.stdout)["ok"] is True
    print(f"Installed artifact checks passed: {artifact.name}")


if __name__ == "__main__":
    main()
