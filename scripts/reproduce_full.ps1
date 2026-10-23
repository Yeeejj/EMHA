param(
    [string]$Device = "auto",
    [switch]$SkipExploratory,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

$Py = if ($env:EMHA_PYTHON) { $env:EMHA_PYTHON } else { "python" }
$Prefix = @()
if ($env:EMHA_PYTHON_PREFIX) {
    $Prefix = @($env:EMHA_PYTHON_PREFIX -split "\s+" | Where-Object { $_ })
}
$DataRoot = if ($env:EMHA_DATA_ROOT) { $env:EMHA_DATA_ROOT } else { "DATASET" }
$Meta = Join-Path $DataRoot "metadata"

function Step {
    Write-Host "==> python -m $($args -join ' ')"
    if ($DryRun) { return }
    & $Py @Prefix -m @args
    if ($LASTEXITCODE -ne 0) {
        Write-Error "step failed (exit $LASTEXITCODE): python -m $($args -join ' ')" -ErrorAction Continue
        exit $LASTEXITCODE
    }
}

function Get-Sha([string]$Name) {
    (Get-FileHash -Algorithm SHA256 (Join-Path $Meta $Name)).Hash.ToLower()
}

function Test-Hash([string]$Name, [string]$Expected = "") {
    Write-Host "==> verify $Name SHA-256"
    if ($DryRun) { return }
    $h = Get-Sha $Name
    Write-Host "    $h"
    if ($Expected -and $h -ne $Expected) {
        Write-Error "$Name changed during the run (was $Expected)" -ErrorAction Continue
        exit 1
    }
    if (Test-Path (Join-Path $DataRoot ".synthetic_root")) {
        Write-Host "    synthetic root: not checked against EVALUATION_PROTOCOL.md"
    }
    elseif (-not (Select-String -Path "EVALUATION_PROTOCOL.md" -SimpleMatch $h -Quiet)) {
        Write-Error "$Name SHA-256 $h is not recorded in EVALUATION_PROTOCOL.md" -ErrorAction Continue
        exit 1
    }
}

Test-Hash "folds.csv"
Test-Hash "labels.csv"
$LabelsSha = if ($DryRun) { "" } else { Get-Sha "labels.csv" }
$FoldsSha = if ($DryRun) { "" } else { Get-Sha "folds.csv" }

Step src.data.labeler
Test-Hash "labels.csv" $LabelsSha
Step src.analysis.questionnaire_report

Step src.data.ingest

Step src.preprocessing.pipeline --overwrite
Step src.features.handcrafted
Step src.features.embeddings --device $Device
if (-not $SkipExploratory) {
    Step src.features.embeddings_handwriting --device $Device
}

Step src.training.splits
Test-Hash "folds.csv" $FoldsSha

Step src.training.run_baselines --analysis primary --predict-middle
Step src.training.run_baselines --analysis full
Step src.training.run_cnn --analysis primary --predict-middle --device $Device
Step src.training.run_cnn --analysis primary --backbone simple --device $Device
Step src.training.run_hybrid --analysis primary --predict-middle --device $Device
Step src.training.run_ensemble --analysis primary

Step src.training.evaluator
foreach ($analysis in "primary", "full") {
    foreach ($model in "lr_handcrafted", "lr_embeddings") {
        Step src.analysis.permutation --model $model --analysis $analysis
    }
}
foreach ($model in "cnn_head_fused", "cnn_hmm_fused", "ensemble") {
    Step src.analysis.permutation --model $model --analysis primary --device $Device
}
Step src.analysis.secondary
Step src.analysis.gradcam
Step src.analysis.report

Write-Host "Done: results/final/RESULTS.txt"
