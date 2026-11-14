"""Install a distribution in a fresh venv and exercise it outside the checkout."""

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def main() -> None:
    artifact = Path(sys.argv[1]).resolve(strict=True)
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
baseline = [ModelSample(prompt_id="p", output='{"x":1}')]
candidate = [ModelSample(prompt_id="p", output='{"x":9}')]
engine = DriftBenchmarkEngine()
assert engine.evaluate(baseline, baseline).passed
failed = engine.evaluate(baseline, candidate)
assert not failed.passed
assert "FAILED" in export_html_report(failed)
'''
        subprocess.run([str(python), "-c", code], cwd=root, env=env, check=True)
    print(f"Installed artifact checks passed: {artifact.name}")


if __name__ == "__main__":
    main()
