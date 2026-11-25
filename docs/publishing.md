# Publishing releases

Editing the workflow file alone does not publish a package.

Manual `workflow_dispatch` has a Yes/No choice, defaulting to No. No runs tests,
builds and checks artifacts without uploading. With Yes, choose the target:
`TestPyPI` (default), `PyPI`, or `GitHub Release only`. Open Actions, Publish
distributions, Run workflow to select these options. Selecting `GitHub Release only`
runs the full build and distribution checks and creates/updates the `v<version>`
GitHub Release with changelog notes and wheel/sdist assets without uploading to
PyPI (useful when a version is already published on PyPI). Manual `PyPI` publishing
does not require creating a GitHub release or tag beforehand: once the PyPI upload
and `verify-pypi` checks pass, the `github-release` job automatically creates the
`v<version>` GitHub Release using the matching `CHANGELOG.md` section and attaches
the verified wheel and sdist files. It still rejects development versions and
respects environment approval rules. A published GitHub release also publishes to
PyPI and attaches the built wheel and sdist assets to that release. Both paths test
and build once, check metadata, install the wheel
and sdist in isolated environments, and pass the same artifacts to a separate
upload job. Production tags must equal `v` plus the package version; development
versions cannot pass the production guard. Actions are pinned to commits.

After each upload, a separate verification job downloads that exact version's
wheel from the target registry and runs the isolated installation checks. It
retries failed downloads three times to allow brief index propagation delays.
The project wheel is downloaded with `--no-deps`; its dependencies are then
installed using the normal PyPI index rather than mixing TestPyPI and PyPI
resolution. A verification failure does not undo the completed upload.

To verify an already uploaded version without uploading anything:

```powershell
& .\.venv\Scripts\python.exe scripts/verify_registry.py testpypi 0.2.0
```

Use the actual uploaded version; this command cannot verify an unpublished version.

Configure Trusted Publishing separately on each registry for owner `Anantgoel0123456`,
repository `llm-drift-bench`, workflow `publish.yml`:

- TestPyPI: environment `testpypi`.
- PyPI: environment `pypi`.

If the project does not yet exist on TestPyPI, use its pending-publisher setup.
Create matching GitHub environments. Require maintainer approval on the production
`pypi` environment. These are account settings, not settings this repository can
establish by itself. Do not trigger publishing until these settings are verified.

## Trusted Publisher checklist

Each value below must match exactly, including case. A mismatch fails the upload
with an OIDC token error. Registry and GitHub settings cannot be verified from this
repository, so tick each item in the web UI before triggering a publish.

| Setting | TestPyPI | PyPI |
| --- | --- | --- |
| Project name | `llm-drift-bench` | `llm-drift-bench` |
| Owner | `Anantgoel0123456` | `Anantgoel0123456` |
| Repository | `llm-drift-bench` | `llm-drift-bench` |
| Workflow filename | `publish.yml` | `publish.yml` |
| Environment name | `testpypi` | `pypi` |
| Trigger | manual: Yes + TestPyPI | manual: Yes + PyPI, or published GitHub release |

- [ ] `git remote -v` shows `Anantgoel0123456/llm-drift-bench`, the owner in the table.
- [ ] GitHub, Settings, Environments: `testpypi` and `pypi` both exist.
- [ ] `pypi` requires a reviewer; the publisher cannot approve their own run if
      "prevent self-review" is on, so add a second reviewer or leave it off for a
      solo project.
- [ ] TestPyPI Publishing page lists the publisher (or a pending publisher) for
      the row above; PyPI likewise.
- [ ] Only the `testpypi` and `pypi` jobs request `id-token: write`, and only
      the post-verification `github-release` job requests `contents: write`;
      the `build` job has `contents: read` only. No API token secret exists or is referenced.

## Distribution checks

CI builds the wheel and sdist once, then installs both on Windows and Linux for
Python 3.10 to 3.13. `scripts/check_installed.py` also inspects the archives:
`py.typed` must be present, and retired vendored code and local commit helper
scripts must be absent. The same script runs in `publish.yml` before upload.
The Opik adapter is tested against the real SDK at the minimum (1.10.24) and the
latest version in a separate CI job.

Before production, update the version in `src/llm_drift_bench/__init__.py`, reconcile
the changelog, pass CI, and verify a TestPyPI install. Each registry rejects reuse
of previously uploaded distribution filenames; increment the development version
for another TestPyPI upload. Do not delete releases to try to reuse versions.

Local checks in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -m build
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
& .\.venv\Scripts\python.exe -m twine check dist/*
if ($LASTEXITCODE -ne 0) { throw 'Metadata validation failed' }
# Select the exact NEW version's artifacts, not stale files from older builds.
$Wheel = Get-Item 'dist/llm_drift_bench-0.4.0-py3-none-any.whl'
$Sdist = Get-Item 'dist/llm_drift_bench-0.4.0.tar.gz'
& .\.venv\Scripts\python.exe scripts/check_installed.py $Wheel.FullName
if ($LASTEXITCODE -ne 0) { throw 'Wheel smoke test failed' }
& .\.venv\Scripts\python.exe scripts/check_installed.py $Sdist.FullName
if ($LASTEXITCODE -ne 0) { throw 'Sdist smoke test failed' }
```

Reference: https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
