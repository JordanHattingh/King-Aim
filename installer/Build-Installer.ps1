[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Container })]
    [string]$StableModelDirectory,
    [string]$PublishDirectory = (Join-Path $PSScriptRoot '..\publish\Aimmy2'),
    [string]$IsccPath = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'output'),
    [switch]$SkipRedist
)

$ErrorActionPreference = 'Stop'
$stableRoot = (Resolve-Path -LiteralPath $StableModelDirectory).Path
$publishRoot = (Resolve-Path -LiteralPath $PublishDirectory).Path
$onnxPath = Join-Path $stableRoot 'model.onnx'
$manifestPath = Join-Path $stableRoot 'manifest.json'
foreach ($required in @($onnxPath, $manifestPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Stable model bundle is missing: $required" }
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($field in @('id', 'version', 'architecture', 'task', 'output_schema')) {
    if ([string]::IsNullOrWhiteSpace([string]$manifest.$field)) { throw "Stable model manifest requires '$field'." }
}
if ($manifest.task -notin @('detect', 'pose')) { throw "Unsupported model task '$($manifest.task)'." }
if (($manifest.task -eq 'pose') -ne [bool]$manifest.is_pose_model) {
    throw 'Manifest task and is_pose_model disagree; detector and pose bundles cannot be substituted silently.'
}

$payloadRoot = Join-Path $PSScriptRoot 'payload\stable-detector'
if (Test-Path -LiteralPath $payloadRoot) { Remove-Item -LiteralPath $payloadRoot -Recurse -Force }
New-Item -ItemType Directory -Path $payloadRoot | Out-Null
Copy-Item -LiteralPath $onnxPath -Destination (Join-Path $payloadRoot 'model.onnx')
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $payloadRoot 'manifest.json')
$modelHash = (Get-FileHash -LiteralPath $onnxPath -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
@(
    "model_sha256=$modelHash"
    "manifest_sha256=$manifestHash"
    "model_id=$($manifest.id)"
    "model_version=$($manifest.version)"
    "model_architecture=$($manifest.architecture)"
    "model_task=$($manifest.task)"
) | Set-Content -LiteralPath (Join-Path $payloadRoot 'checksums.ini') -Encoding ascii
@{
    schema_version = 1
    model_id = [string]$manifest.id
    model_version = [string]$manifest.version
    architecture = [string]$manifest.architecture
    task = [string]$manifest.task
    files = @{
        'model.onnx' = $modelHash
        'manifest.json' = $manifestHash
    }
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $payloadRoot 'checksums.json') -Encoding utf8

if (-not (Test-Path -LiteralPath $IsccPath -PathType Leaf)) { throw "Inno Setup compiler not found: $IsccPath" }
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$defines = @("/DPublishDir=$publishRoot", "/DStableBundleDir=$payloadRoot", "/DInstallerOutputDir=$outputRoot")
if ($SkipRedist) { $defines += '/DSkipRedist=1' }
& $IsccPath @defines (Join-Path $PSScriptRoot 'Aimmy2Setup.iss')
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed with exit code $LASTEXITCODE." }
