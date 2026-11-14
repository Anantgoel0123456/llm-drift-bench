"""Install a distribution in a fresh venv and exercise it outside the checkout."""

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
            return wheel.namelist()
    with tarfile.open(artifact) as sdist:
        return sdist.getnames()


def check_archive_contents(artifact: Path) -> None:
    """Inspect the distribution itself, before and independent of installation."""
    names = archive_names(artifact)
    leaked = [n for n in names if any(part in n for part in FORBIDDEN_FRAGMENTS)]
    if leaked:
        raise SystemExit(f"{artifact.name} contains excluded files: {leaked}")
    typed = [n for n in names if n.endswith("llm_drift_bench/py.typed")]
    if not typed:
        raise SystemExit(f"{artifact.name} is missing llm_drift_bench/py.typed")
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
    print(f"Installed artifact checks passed: {artifact.name}")


if __name__ == "__main__":
    main()
