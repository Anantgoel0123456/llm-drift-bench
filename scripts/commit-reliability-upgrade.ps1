param([switch]$Push)

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
if ($LASTEXITCODE -ne 0) { throw 'Review existing staged changes first.' }
& $GitExe --no-pager diff --check
if ($LASTEXITCODE -ne 0) { throw 'Git diff check failed; see output above.' }

function Save-Change {
    param([string]$Message, [string[]]$Paths)
    foreach ($Path in $Paths) {
        if (-not (Test-Path -LiteralPath $Path)) { throw "Missing file: $Path" }
    }
    & $GitExe --no-pager add -- @Paths
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed.' }
    & $GitExe --no-pager diff --cached --quiet
    $DiffCode = $LASTEXITCODE
    if ($DiffCode -eq 0) { return }
    if ($DiffCode -ne 1) { throw 'Cannot inspect staged changes.' }
    & $GitExe --no-pager commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed; staged changes are preserved.' }
}

Save-Change 'fix(validation): bound regex and reject ambiguous numeric comparisons' @(
    'src/llm_drift_bench/_numbers.py', 'src/llm_drift_bench/_patterns.py',
    'src/llm_drift_bench/metrics.py', 'src/llm_drift_bench/validator.py',
    'tests/test_json_paths.py', 'tests/test_numeric_safety.py',
    'tests/test_pattern_safety.py', 'tests/test_schema_shapes.py'
)
Save-Change 'fix(io): validate bounded UTF-8 datasets and saved reports' @(
    'src/llm_drift_bench/_file_io.py', 'src/llm_drift_bench/cli.py',
    'src/llm_drift_bench/engine.py', 'src/llm_drift_bench/evaluator.py',
    'src/llm_drift_bench/report_io.py', 'tests/test_file_io.py'
)
Save-Change 'feat(evaluation): check read-before-edit tool event evidence' @(
    'src/llm_drift_bench/tool_events.py', 'src/llm_drift_bench/__init__.py',
    'tests/test_tool_events.py', 'docs/tool-events.md'
)
Save-Change 'ci: verify installed distributions and separate publishing targets' @(
    '.github/workflows/ci.yml', '.github/workflows/publish.yml',
    'scripts/check_installed.py', 'docs/publishing.md'
)
Save-Change 'docs: document reliability changes and ignore local issue drafts' @(
    '.gitignore', 'CHANGELOG.md', 'README.md', 'docs/current-limitations.md',
    'docs/roadmap.md', 'scripts/commit-reliability-upgrade.ps1'
)
if ($Push) {
    & $GitExe --no-pager push origin main
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Local commits are preserved.' }
}
& $GitExe --no-pager status --short --branch
