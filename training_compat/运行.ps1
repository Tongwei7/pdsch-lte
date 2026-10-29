param(
    [Parameter(Position=0)]
    [ValidateSet('check','inspect','train','evaluate','predict')]
    [string]$Command = 'check',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Options
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if ($env:PDSCH_PYTHON) {
    $taskPython = $env:PDSCH_PYTHON
} elseif (Test-Path -LiteralPath $projectPython) {
    $taskPython = $projectPython
} else {
    throw 'No configured Python environment. Follow README.md to create .venv, or set PDSCH_PYTHON.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Command -eq 'check') {
        & $taskPython 'check_network.py' @Options
    } else {
        & $taskPython 'run.py' $Command @Options
    }
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
