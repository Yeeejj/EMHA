<#
.SYNOPSIS
    Dev commands for INSIDE-OUT / EMHA (Windows-friendly; no `make` required).

.USAGE
    .\scripts\dev.ps1 install   # pip install runtime + dev requirements
    .\scripts\dev.ps1 format    # black .
    .\scripts\dev.ps1 lint      # black --check . ; flake8
    .\scripts\dev.ps1 test      # pytest -q
    .\scripts\dev.ps1 all       # install, format, lint, test (default)
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "format", "lint", "test", "all")]
    [string]$Command = "all"
)

$ErrorActionPreference = "Stop"

function Invoke-Install {
    pip install -r requirements.txt -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

function Invoke-Format {
    python -m black .
    if ($LASTEXITCODE -ne 0) { throw "black failed" }
}

function Invoke-Lint {
    python -m black --check .
    if ($LASTEXITCODE -ne 0) { throw "black --check failed" }
    python -m flake8 .
    if ($LASTEXITCODE -ne 0) { throw "flake8 failed" }
}

function Invoke-Test {
    python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
}

switch ($Command) {
    "install" { Invoke-Install }
    "format"  { Invoke-Format }
    "lint"    { Invoke-Lint }
    "test"    { Invoke-Test }
    "all"     {
        Invoke-Install
        Invoke-Format
        Invoke-Lint
        Invoke-Test
    }
}
