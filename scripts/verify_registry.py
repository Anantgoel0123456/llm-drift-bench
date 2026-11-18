"""Download an exact registry release and run the isolated installation checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

INDEXES = {
    "testpypi": "https://test.pypi.org/simple/",
    "pypi": "https://pypi.org/simple/",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", choices=INDEXES)
    parser.add_argument("version")
    args = parser.parse_args()
    # Accept ordinary PEP 440 version characters, never pip requirement options/URLs.
    if not args.version or any(
        c not in "0123456789abcdefghijklmnopqrstuvwxyz.+!-" for c in args.version
    ):
        parser.error("Use an exact lowercase package version.")
    checker = Path(__file__).resolve().with_name("check_installed.py")
    with tempfile.TemporaryDirectory(prefix="drift-registry-") as directory:
        destination = Path(directory)
        command = [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "download",
            "--no-cache-dir",
            "--no-deps",
            "--only-binary=:all:",
            "--index-url",
            INDEXES[args.registry],
            "--dest",
            str(destination),
            f"llm-drift-bench=={args.version}",
        ]
        for attempt in range(3):
            try:
                result = subprocess.run(command, check=False, timeout=180)
            except subprocess.TimeoutExpired:
                print("Registry download timed out after 180 seconds.", flush=True)
            else:
                if result.returncode == 0:
                    break
            if attempt == 2:
                raise SystemExit("Release download failed after three attempts.")
            print("Waiting 15 seconds for registry availability.", flush=True)
            time.sleep(15)
        wheels = list(destination.glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit("Expected exactly one downloaded project wheel.")
        # check_installed creates a fresh venv; dependencies use the normal PyPI index.
        subprocess.run([sys.executable, str(checker), str(wheels[0])], check=True)


if __name__ == "__main__":
    main()
