"""Install a distribution in a fresh venv and exercise it outside the checkout."""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

FORBIDDEN_FRAGMENTS = ("_vendor", "OPIK_LICENSE", "OPIK_NOTICE", "commit-reviewed-changes")


def archive_names(artifact: Path) -> list[str]:
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as wheel:
            return [item.filename for item in wheel.infolist() if not item.is_dir()]
    with tarfile.open(artifact) as sdist:
        return [item.name for item in sdist.getmembers() if item.isfile()]


def check_archive_contents(artifact: Path) -> None:
    """Inspect the distribution itself, before and independent of installation."""
    names = archive_names(artifact)
    leaked = [n for n in names if any(part in n for part in FORBIDDEN_FRAGMENTS)]
    if leaked:
        raise SystemExit(f"{artifact.name} contains excluded files: {leaked}")
    is_wheel = artifact.suffix == ".whl"
    package_root = "" if is_wheel else artifact.name.removesuffix(".tar.gz") + "/src/"
    if package_root + "llm_drift_bench/py.typed" not in names:
        raise SystemExit(f"{artifact.name} is missing llm_drift_bench/py.typed")
    if is_wheel:
        metadata_root = "-".join(artifact.name.split("-")[:2]) + ".dist-info"
        if metadata_root + "/licenses/LICENSE" not in names:
            raise SystemExit(f"{artifact.name} is missing its packaged LICENSE")
    print(f"Archive contents verified: {artifact.name} ({len(names)} files)")


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
    print(f"Installed artifact checks passed: {artifact.name}")


if __name__ == "__main__":
    main()
