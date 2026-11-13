param([switch]$Push)

# Commit only the reviewed reporting/test-helper changes. Never force-push or publish.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$GitCommand = Get-Command git -ErrorAction SilentlyContinue
$GitExe = if ($GitCommand) { $GitCommand.Source } else {
    'C:\Users\Anant Goel\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'
}
if (-not (Test-Path -LiteralPath $GitExe)) { throw 'Git executable not found.' }
$Branch = & $GitExe --no-pager branch --show-current
if ($LASTEXITCODE -ne 0 -or $Branch -ne 'main') { throw 'Expected branch main.' }
& $GitExe --no-pager diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Review existing staged changes before continuing.' }
& $GitExe --no-pager diff --check
if ($LASTEXITCODE -ne 0) {
    throw 'Git whitespace check failed. Review the Git output above before continuing.'
}

function Save-ReviewedCommit {
    param([string]$Message, [string[]]$Paths)
    foreach ($Path in $Paths) {
        if (-not (Test-Path -LiteralPath $Path)) { throw "Missing expected file: $Path" }
    }
    & $GitExe --no-pager add -- @Paths
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed. Stop here.' }
    & $GitExe --no-pager diff --cached --quiet
    $DiffCode = $LASTEXITCODE
    if ($DiffCode -eq 0) { return }
    if ($DiffCode -ne 1) { throw 'Could not inspect staged changes.' }
    & $GitExe --no-pager commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed. Stop here.' }
}

Save-ReviewedCommit 'feat(reports): unify CLI scores and add offline HTML reports' @(
    'src/llm_drift_bench/__init__.py', 'src/llm_drift_bench/cli.py',
    'src/llm_drift_bench/engine.py', 'src/llm_drift_bench/evaluator.py',
    'src/llm_drift_bench/models.py', 'src/llm_drift_bench/reporter.py',
    'src/llm_drift_bench/_report_views.py', 'src/llm_drift_bench/report_io.py',
    'src/llm_drift_bench/testing.py', 'tests/test_evaluator.py',
    'tests/test_canonical_reports.py', 'tests/test_testing.py',
    'examples/expected-drift.json', 'examples/expected-pass.json'
)
Save-ReviewedCommit 'docs: explain canonical reports and automated evaluation workflows' @(
    'CHANGELOG.md', 'README.md', 'docs/architecture.md', 'docs/current-limitations.md',
    'docs/migration-0.2.md', 'docs/roadmap.md', 'docs/workflows.md',
    'docs/assets/example-report.html', 'scripts/commit-report-upgrade.ps1'
)
if ($Push) {
    & $GitExe --no-pager push origin main
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Local commits are preserved.' }
}
& $GitExe --no-pager status --short --branch
