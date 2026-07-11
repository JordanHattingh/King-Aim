#Requires -Version 5.1
<#
.SYNOPSIS
    E040 rehearsal validation for the King Aim YOLOv8 baseline detector.

.DESCRIPTION
    Runs all automated validation gates for the epoch-40 baseline artifact:
    - Python unit tests (29 tests)
    - CPU ONNX parity  (validate_detector_onnx.py, CPUExecutionProvider)
    - DirectML ONNX parity if C:\Tmp\dml_venv exists (run_dml_parity.py)

    Artifacts must be present at C:\KingAimTraining\baseline\yolov8-e040\
    Does NOT touch the active training run at C:\KingAimTraining\

.NOTES
    Run from the repo root:  .\RUN_E040_WINDOWS_VALIDATION.ps1
    Training artifacts live outside the repo; do NOT commit .pt or .onnx files.
#>

$ErrorActionPreference = "Stop"

# -- Paths -------------------------------------------------------------------
$RepoRoot    = $PSScriptRoot
$TrainingDir = Join-Path $RepoRoot "training"
$BaselineDir = "C:\KingAimTraining\baseline\yolov8-e040"
$ParityDir   = Join-Path $BaselineDir "parity"
$PtFile      = Join-Path $BaselineDir "kingaim-yolov8-baseline-e040.pt"
$OnnxFile    = Join-Path $BaselineDir "kingaim-yolov8-baseline-e040-fp32.onnx"
$TestImage   = Join-Path $RepoRoot "readme_assets\DT1.png"
$DmlScript   = Join-Path $TrainingDir "tools\run_dml_parity.py"
$DmlVenv     = "C:\Tmp\dml_venv"

# -- Helpers -----------------------------------------------------------------
$PassList = [System.Collections.Generic.List[string]]::new()
$FailList = [System.Collections.Generic.List[string]]::new()

function Gate {
    param([string]$Name, [scriptblock]$Block)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    try {
        & $Block
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
        $script:PassList.Add($Name)
        Write-Host "PASS: $Name" -ForegroundColor Green
    } catch {
        $script:FailList.Add($Name)
        Write-Host "FAIL: $Name  ($_)" -ForegroundColor Red
    }
}

# -- Preflight ---------------------------------------------------------------
Write-Host "King Aim - E040 rehearsal validation"
Write-Host "Repo  : $RepoRoot"
Write-Host "Epoch : 40"
Write-Host ("Date  : " + (Get-Date -Format "yyyy-MM-dd HH:mm"))
Write-Host ""

if (-not (Test-Path $PtFile)) {
    Write-Host "ERROR: artifact not found: $PtFile" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $OnnxFile)) {
    Write-Host "ERROR: artifact not found: $OnnxFile" -ForegroundColor Red
    exit 1
}

# -- Gate 1: Python unit tests -----------------------------------------------
Gate "Python unit tests (29)" {
    python -m pytest (Join-Path $TrainingDir "tests") -q --tb=short
}

# -- Gate 2: CPU ONNX parity -------------------------------------------------
Gate "CPU ONNX parity (CPUExecutionProvider)" {
    $imgArgs = @()
    if (Test-Path $TestImage) { $imgArgs = @("--image", $TestImage) }
    python (Join-Path $TrainingDir "validate_detector_onnx.py") `
        --pt          $PtFile   `
        --onnx        $OnnxFile `
        --imgsz       512       `
        --class-count 1         `
        --provider    CPUExecutionProvider `
        --output-dir  $ParityDir `
        @imgArgs
}

# -- Gate 3: DirectML ONNX parity (optional) ---------------------------------
$DmlPy    = Join-Path $DmlVenv "Scripts\python.exe"
$DmlReady = (Test-Path $DmlPy) -and (Test-Path $DmlScript)

Write-Host ""
Write-Host "=== DirectML ONNX parity ===" -ForegroundColor Cyan
if ($DmlReady) {
    $imgArgs = @()
    if (Test-Path $TestImage) { $imgArgs = @("--image", $TestImage) }
    try {
        & $DmlPy $DmlScript --onnx $OnnxFile --output-dir $ParityDir @imgArgs
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
        $PassList.Add("DirectML ONNX parity (DmlExecutionProvider)")
        Write-Host "PASS: DirectML ONNX parity (DmlExecutionProvider)" -ForegroundColor Green
    } catch {
        $FailList.Add("DirectML ONNX parity (DmlExecutionProvider)")
        Write-Host "FAIL: DirectML ONNX parity ($_)" -ForegroundColor Red
    }
} else {
    Write-Host "  SKIPPED -- $DmlVenv not found." -ForegroundColor Yellow
    Write-Host "  To enable:"
    Write-Host "    python -m venv C:\Tmp\dml_venv"
    Write-Host "    C:\Tmp\dml_venv\Scripts\pip install onnxruntime-directml numpy pillow"
}

# -- Summary -----------------------------------------------------------------
Write-Host ""
Write-Host "=================================================" -ForegroundColor White
Write-Host "E040 Rehearsal Validation Summary" -ForegroundColor White
Write-Host "================================================="
foreach ($g in $PassList) { Write-Host "  PASS  $g" -ForegroundColor Green }
foreach ($g in $FailList) { Write-Host "  FAIL  $g" -ForegroundColor Red }

if ($FailList.Count -gt 0) {
    Write-Host ""
    Write-Host "$($FailList.Count) gate(s) failed." -ForegroundColor Red
    Write-Host ""
    exit 1
} else {
    Write-Host ""
    Write-Host "All $($PassList.Count) gate(s) passed." -ForegroundColor Green
    Write-Host ""
    exit 0
}
