# Run after reviewing the diff. Creates local commits only; never pushes.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$GitCommand = Get-Command git -ErrorAction SilentlyContinue
$GitExe = if ($GitCommand) { $GitCommand.Source } else {
    'C:\Users\Anant Goel\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'
}
if (-not (Test-Path -LiteralPath $GitExe)) { throw 'Git executable not found.' }
$Branch = & $GitExe branch --show-current
if ($LASTEXITCODE -ne 0 -or $Branch -ne 'main') { throw 'Expected branch main.' }
& $GitExe diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Review existing staged changes before running this script.' }

function Save-ReviewedCommit {
    param([string]$Message, [string[]]$Paths)
    & $GitExe add -- @Paths
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed. No commit attempted.' }
    & $GitExe diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { return }
    if ($LASTEXITCODE -ne 1) { throw 'Could not inspect staged changes.' }
    & $GitExe commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
}

Save-ReviewedCommit 'fix(cli): enforce drift gates and validation input contracts' @(
    'src/llm_drift_bench/cli.py', 'src/llm_drift_bench/engine.py',
    'src/llm_drift_bench/models.py', 'tests/test_cli.py', 'tests/test_engine.py',
    'tests/test_validation_cli.py', 'docs/architecture.md'
)
Save-ReviewedCommit 'build: prepare single-source 0.2.0 development version' @(
    'pyproject.toml', 'src/llm_drift_bench/__init__.py'
)
Save-ReviewedCommit 'docs: refresh project front page and add reproducible examples' @(
    'README.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'docs/current-limitations.md',
    'docs/migration-0.2.md', 'docs/roadmap.md', 'docs/commit-reviewed-changes.ps1',
    'examples/README.md', 'examples/baseline.jsonl', 'examples/candidate-pass.jsonl',
    'examples/candidate-drift.jsonl', 'examples/extraction-suite.json',
    'examples/expected-pass.json', 'examples/expected-drift.json', 'tests/test_examples.py'
)
& $GitExe --no-pager status --short --branch
