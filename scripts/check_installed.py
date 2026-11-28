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
        emb_html = root / "embeddings.html"
        subprocess.run(
            [
                str(cli), "check-embeddings",
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "html", "-o", str(emb_html),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<table" in emb_html.read_text(encoding="utf-8")
        conv_pass = subprocess.run(
            [
                str(cli), "check-conversations",
                "-b", str(examples / "conversations-baseline.jsonl"),
                "-c", str(examples / "conversations-baseline.jsonl"),
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(conv_pass.stdout)["passed"] is True
        conv_html = root / "conversations.html"
        conv_drift = subprocess.run(
            [
                str(cli), "check-conversations",
                "-b", str(examples / "conversations-baseline.jsonl"),
                "-c", str(examples / "conversations-candidate-drift.jsonl"),
                "--format", "html", "-o", str(conv_html),
            ],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if conv_drift.returncode != 1:
            raise SystemExit(
                f"check-conversations drift: exit {conv_drift.returncode}, want 1"
            )
        assert "<table" in conv_html.read_text(encoding="utf-8")
        conv_doc = subprocess.run(
            [
                str(cli), "doctor",
                "--baseline-conversations", str(examples / "conversations-baseline.jsonl"),
                "--candidate-conversations", str(examples / "conversations-candidate-drift.jsonl"),
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(conv_doc.stdout)["ok"] is True
        sg_in = root / "sharegpt.jsonl"
        sg_in.write_text(
            json.dumps(
                {
                    "id": "conv-1",
                    "conversations": [
                        {"from": "human", "value": "Email dev@example.com"},
                        {"from": "gpt", "value": '{"ok": true}'},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        sg_imp = root / "imported_conv.jsonl"
        subprocess.run(
            [
                str(cli), "import-conversations",
                "--from", "sharegpt",
                "-i", str(sg_in),
                "-o", str(sg_imp),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        sg_red = root / "redacted_conv.jsonl"
        subprocess.run(
            [
                str(cli), "redact", "--conversations",
                "-i", str(sg_imp), "-o", str(sg_red),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "[REDACTED:EMAIL]" in sg_red.read_text(encoding="utf-8")
        audit_conv_html = root / "audit_conv.html"
        subprocess.run(
            [
                str(cli), "audit",
                "-s", str(examples / "extraction-suite.json"),
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--baseline-conversations", str(examples / "conversations-baseline.jsonl"),
                "--candidate-conversations", str(examples / "conversations-baseline.jsonl"),
                "--format", "html", "-o", str(audit_conv_html),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "Multi-Turn Conversations" in audit_conv_html.read_text(encoding="utf-8")
        shutil.copy2(examples / "conversations-baseline.jsonl", snap_dir / "conversations.jsonl")
        snap_conv_manifest = snap_dir / "snapshot-conv.manifest.json"
        subprocess.run(
            [
                str(cli), "snapshot", "create",
                "-b", str(snap_dir / "baseline.jsonl"),
                "--conversations", str(snap_dir / "conversations.jsonl"),
                "-o", str(snap_conv_manifest),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        ver_conv = subprocess.run(
            [str(cli), "snapshot", "verify", "-i", str(snap_conv_manifest), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(ver_conv.stdout)["valid"] is True
        diff_tools_out = subprocess.run(
            [str(cli), "diff-tools", "-b", str(trace), "-c", str(trace), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(diff_tools_out.stdout)["passed"] is True
        agent_pack_dir = root / "pack_agent"
        subprocess.run(
            [str(cli), "init-sample", "--dir", str(agent_pack_dir), "--pack", "agent"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [str(cli), "lint-suite", "-s", str(agent_pack_dir / "prompts.json"), "--strict"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        conv_diff_out = subprocess.run(
            [
                str(cli), "check-conversations",
                "-b", str(examples / "conversations-baseline.jsonl"),
                "-c", str(examples / "conversations-candidate-drift.jsonl"),
                "--show-diffs", "--format", "json",
            ],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if conv_diff_out.returncode != 1:
            raise SystemExit(f"check-conversations --show-diffs: exit {conv_diff_out.returncode}")
        assert len(json.loads(conv_diff_out.stdout)["turn_diffs"]) == 1
        conv_sarif = subprocess.run(
            [
                str(cli), "check-conversations",
                "-b", str(examples / "conversations-baseline.jsonl"),
                "-c", str(examples / "conversations-candidate-drift.jsonl"),
                "--format", "sarif",
            ],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if conv_sarif.returncode != 1:
            raise SystemExit(f"check-conversations --format sarif: exit {conv_sarif.returncode}")
        assert json.loads(conv_sarif.stdout)["version"] == "2.1.0"
        conv_junit = subprocess.run(
            [
                str(cli), "check-conversations",
                "-b", str(examples / "conversations-baseline.jsonl"),
                "-c", str(examples / "conversations-candidate-drift.jsonl"),
                "--format", "junit",
            ],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if conv_junit.returncode != 1:
            raise SystemExit(f"check-conversations --format junit: exit {conv_junit.returncode}")
        assert "llm-drift-bench-conversations" in conv_junit.stdout
        gate_trace = subprocess.run(
            [
                str(cli), "gate",
                "-i", str(examples / "expected-pass.json"),
                "--baseline-trace", str(trace),
                "--candidate-trace", str(trace),
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        gate_trace_data = json.loads(gate_trace.stdout)
        assert gate_trace_data["approved"] is True
        assert any(c["name"] == "tool_trace_drift" for c in gate_trace_data["checks"])
        trace_html = root / "trace-diff.html"
        subprocess.run(
            [
                str(cli), "diff-tools",
                "-b", str(trace), "-c", str(trace),
                "--format", "html", "-o", str(trace_html),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<!DOCTYPE html>" in trace_html.read_text(encoding="utf-8")
        audit_trace_json = root / "audit-trace.json"
        subprocess.run(
            [
                str(cli), "audit",
                "-s", str(examples / "extraction-suite.json"),
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--baseline-trace", str(trace),
                "--candidate-trace", str(trace),
                "--format", "json", "-o", str(audit_trace_json),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        audit_trace_data = json.loads(audit_trace_json.read_text(encoding="utf-8"))
        assert audit_trace_data["passed"] is True and "tool_trace" in audit_trace_data
        audit_badge = subprocess.run(
            [str(cli), "badge", "-i", str(audit_trace_json)],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "llm-audit: PASS" in audit_badge.stdout
        conv_csv = subprocess.run(
            [
                str(cli), "check-conversations",
                "-b", str(examples / "conversations-baseline.jsonl"),
                "-c", str(examples / "conversations-candidate-drift.jsonl"),
                "--format", "csv",
            ],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if conv_csv.returncode != 1:
            raise SystemExit(f"check-conversations --format csv: exit {conv_csv.returncode}")
        assert "conversation_id,conversation_drifted,turn_index" in conv_csv.stdout
        conv_diag = subprocess.run(
            [
                str(cli), "check-conversations",
                "-b", str(examples / "conversations-baseline.jsonl"),
                "-c", str(examples / "conversations-candidate-drift.jsonl"),
                "--diagnose", "--format", "json",
            ],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if conv_diag.returncode != 1:
            raise SystemExit(f"check-conversations --diagnose: exit {conv_diag.returncode}")
        assert len(json.loads(conv_diag.stdout)["diagnostics"]) >= 1
        doc_trace = subprocess.run(
            [
                str(cli), "doctor",
                "--baseline-trace", str(trace),
                "--candidate-trace", str(trace),
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(doc_trace.stdout)["ok"] is True
        ex_trace_sarif = subprocess.run(
            [
                str(cli), "diff-tools",
                "-b", str(examples / "trace-baseline.jsonl"),
                "-c", str(examples / "trace-candidate-drift.jsonl"),
                "--fail-on-new-edit-resources",
                "--format", "sarif",
            ],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if ex_trace_sarif.returncode != 1:
            raise SystemExit(f"diff-tools --format sarif: exit {ex_trace_sarif.returncode}")
        assert json.loads(ex_trace_sarif.stdout)["version"] == "2.1.0"
        ex_trace_junit = subprocess.run(
            [
                str(cli), "diff-tools",
                "-b", str(examples / "trace-baseline.jsonl"),
                "-c", str(examples / "trace-candidate-drift.jsonl"),
                "--fail-on-new-edit-resources",
                "--format", "junit",
            ],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if ex_trace_junit.returncode != 1:
            raise SystemExit(f"diff-tools --format junit: exit {ex_trace_junit.returncode}")
        assert "llm-drift-bench-tool-traces" in ex_trace_junit.stdout
        emb_csv = subprocess.run(
            [
                str(cli), "check-embeddings",
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "csv",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "model,prompt_id,available,cosine_similarity" in emb_csv.stdout
        audit_pr = subprocess.run(
            [
                str(cli), "audit",
                "-s", str(examples / "extraction-suite.json"),
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "pr-comment",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<!-- llm-drift-bench:audit -->" in audit_pr.stdout
        conv_pr = subprocess.run(
            [
                str(cli), "check-conversations",
                "-b", str(examples / "conversations-baseline.jsonl"),
                "-c", str(examples / "conversations-baseline.jsonl"),
                "--format", "pr-comment",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<!-- llm-drift-bench:conversations -->" in conv_pr.stdout
        shutil.copy2(examples / "trace-baseline.jsonl", snap_dir / "trace.jsonl")
        snap_trace_manifest = snap_dir / "snapshot-trace.manifest.json"
        subprocess.run(
            [
                str(cli), "snapshot", "create",
                "-b", str(snap_dir / "baseline.jsonl"),
                "--trace", str(snap_dir / "trace.jsonl"),
                "-o", str(snap_trace_manifest),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        ver_trace = subprocess.run(
            [str(cli), "snapshot", "verify", "-i", str(snap_trace_manifest), "--format", "json"],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(ver_trace.stdout)["valid"] is True
        red_trace_in = root / "secret_trace.jsonl"
        red_trace_in.write_text(
            '{"kind": "read", "resource": "mail/dev@example.com", "start": 0, "end": 5}\n',
            encoding="utf-8",
        )
        red_trace_out = root / "clean_trace.jsonl"
        subprocess.run(
            [str(cli), "redact", "--traces", "-i", str(red_trace_in), "-o", str(red_trace_out)],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "[REDACTED:EMAIL]" in red_trace_out.read_text(encoding="utf-8")
        tt_diag_json = root / "tt-diag.json"
        tt_diag = subprocess.run(
            [
                str(cli), "diff-tools",
                "-b", str(examples / "trace-baseline.jsonl"),
                "-c", str(examples / "trace-candidate-drift.jsonl"),
                "--fail-on-new-edit-resources",
                "--diagnose", "--format", "json", "-o", str(tt_diag_json),
            ],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if tt_diag.returncode != 1:
            raise SystemExit(f"diff-tools --diagnose: exit {tt_diag.returncode}")
        assert len(json.loads(tt_diag_json.read_text(encoding="utf-8"))["diagnostics"]) >= 1
        tt_badge = subprocess.run(
            [str(cli), "badge", "-i", str(tt_diag_json)],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "tool-trace: FAIL" in tt_badge.stdout
        tt_pr = subprocess.run(
            [
                str(cli), "diff-tools",
                "-b", str(examples / "trace-baseline.jsonl"),
                "-c", str(examples / "trace-baseline.jsonl"),
                "--format", "pr-comment",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<!-- llm-drift-bench:tool-trace -->" in tt_pr.stdout
        dist_csv = subprocess.run(
            [
                str(cli), "distribution",
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-drift.jsonl"),
                "--format", "csv",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "segment,field_type,path,baseline_count" in dist_csv.stdout
        conv_ledger = root / "conv-history.jsonl"
        for conv_val in ("0.02", "0.03"):
            subprocess.run(
                [
                    str(cli), "history", "record",
                    "--ledger", str(conv_ledger),
                    "-i", str(examples / "expected-pass.json"),
                    "--conversation-drift", conv_val,
                ],
                cwd=root, env=env, check=True, capture_output=True, text=True,
            )
        hist_chk = subprocess.run(
            [
                str(cli), "history", "check",
                "--ledger", str(conv_ledger),
                "--window", "1",
                "--max-conversation-drift-increase", "0.10",
                "--format", "json",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(hist_chk.stdout)["regressed"] is False
        audit_diag = subprocess.run(
            [
                str(cli), "audit",
                "-s", str(examples / "extraction-suite.json"),
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-drift.jsonl"),
                "--diagnose", "--format", "json",
            ],
            cwd=root, env=env, check=False, capture_output=True, text=True,
        )
        if audit_diag.returncode != 1:
            raise SystemExit(f"audit --diagnose: exit {audit_diag.returncode}")
        assert len(json.loads(audit_diag.stdout)["diagnostics"]) >= 1
        qual_html = root / "quality.html"
        subprocess.run(
            [
                str(cli), "check-quality",
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "html", "-o", str(qual_html),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<!doctype html>" in qual_html.read_text(encoding="utf-8")
        fmt_csv = subprocess.run(
            [
                str(cli), "check-formats",
                "-i", str(pack_dir / "baseline.jsonl"),
                "--rule", "$.currency=iso-currency",
                "--format", "csv",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "prompt_id,path,expected_format,actual_preview,message" in fmt_csv.stdout
        cons_json = root / "cons.json"
        subprocess.run(
            [
                str(cli), "check-consumer",
                "--baseline-decisions", str(cons_b),
                "--candidate-decisions", str(cons_c),
                "--format", "json", "-o", str(cons_json),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        cons_badge = subprocess.run(
            [str(cli), "badge", "-i", str(cons_json)],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "consumer: PASS" in cons_badge.stdout
        cons_pr = subprocess.run(
            [
                str(cli), "check-consumer",
                "--baseline-decisions", str(cons_b),
                "--candidate-decisions", str(cons_c),
                "--format", "pr-comment",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<!-- llm-drift-bench:consumer -->" in cons_pr.stdout
        roll_json = root / "roll.json"
        subprocess.run(
            [
                str(cli), "rolling-distribution",
                "-w", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "json", "-o", str(roll_json),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        roll_badge = subprocess.run(
            [str(cli), "badge", "-i", str(roll_json)],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "rolling-dist: PASS" in roll_badge.stdout
        roll_pr = subprocess.run(
            [
                str(cli), "rolling-distribution",
                "-w", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "pr-comment",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<!-- llm-drift-bench:rolling-distribution -->" in roll_pr.stdout
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
        qual_sarif = subprocess.run(
            [
                str(cli), "check-quality",
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "sarif",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(qual_sarif.stdout)["version"] == "2.1.0"
        qual_junit = subprocess.run(
            [
                str(cli), "check-quality",
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "junit",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "llm-drift-bench-quality" in qual_junit.stdout
        qual_csv = subprocess.run(
            [
                str(cli), "check-quality",
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "csv",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "prompt_id,check,severity,detail" in qual_csv.stdout
        fmt_sarif = subprocess.run(
            [
                str(cli), "check-formats",
                "-i", str(pack_dir / "baseline.jsonl"),
                "--rule", "$.currency=iso-currency",
                "--format", "sarif",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert json.loads(fmt_sarif.stdout)["version"] == "2.1.0"
        fmt_junit = subprocess.run(
            [
                str(cli), "check-formats",
                "-i", str(pack_dir / "baseline.jsonl"),
                "--rule", "$.currency=iso-currency",
                "--format", "junit",
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "llm-drift-bench-formats" in fmt_junit.stdout
        lp_html = root / "logprobs.html"
        subprocess.run(
            [
                str(cli), "logprobs",
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "html", "-o", str(lp_html),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<!doctype html>" in lp_html.read_text(encoding="utf-8")
        cost_html = root / "cost.html"
        subprocess.run(
            [
                str(cli), "cost",
                "-b", str(examples / "baseline.jsonl"),
                "-c", str(examples / "candidate-pass.jsonl"),
                "--format", "html", "-o", str(cost_html),
            ],
            cwd=root, env=env, check=True, capture_output=True, text=True,
        )
        assert "<!doctype html>" in cost_html.read_text(encoding="utf-8")
    print(f"Installed artifact checks passed: {artifact.name}")


if __name__ == "__main__":
    main()
