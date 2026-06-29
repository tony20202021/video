# 5_train_residents.ps1 — обучение идентификатора жителей (Модель 2)
#
# Usage:
#   .\sh\train\5_train_residents.ps1 -Data ".output\training\export_identify.zip"
#   .\sh\train\5_train_residents.ps1 -Data export_dir -Backbone ".models\classify\backbone.pt"
#   .\sh\train\5_train_residents.ps1 -Data export_dir -InitFrom ".models\identify\v1.pt"

param(
    [Parameter(Mandatory=$true)]
    [string]  $Data,
    [string]  $Backbone  = "",   # backbone.pt от 2_train_groups (первый цикл)
    [string]  $InitFrom  = "",   # v<N>.pt от предыдущей Модели 2 (повторное обучение)
    [int]     $Epochs    = 30,
    [int]     $BatchSize = 32,
    [float]   $Lr        = 5e-4,
    [float]   $ValSplit  = 0.2,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\train\5_train_residents.py"

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

# Разворачиваем относительные пути от корня репо
if (-not [System.IO.Path]::IsPathRooted($Data))    { $Data     = "$Repo\$Data" }
if ($Backbone -and -not [System.IO.Path]::IsPathRooted($Backbone)) { $Backbone = "$Repo\$Backbone" }
if ($InitFrom -and -not [System.IO.Path]::IsPathRooted($InitFrom)) { $InitFrom = "$Repo\$InitFrom" }

if ($Backbone -and $InitFrom) {
    Write-Host "[!] Use either -Backbone or -InitFrom, not both." -ForegroundColor Red
    exit 1
}

Write-Host "=== 5_train_residents (Model 2) ===" -ForegroundColor Cyan
Write-Host "Repo:   $Repo"
Write-Host "Script: $Script"
Write-Host "Data:   $Data"
if ($Backbone)  { Write-Host "Backbone:  $Backbone  (first cycle)" }
if ($InitFrom)  { Write-Host "Init-from: $InitFrom  (retrain)" }
if (-not $Backbone -and -not $InitFrom) { Write-Host "Init: random (no backbone/init-from)" }
Write-Host "Epochs=$Epochs  BatchSize=$BatchSize  LR=$Lr"
Write-Host "Model output: $Repo\.models\identify\"
Write-Host ""

if (-not (Test-Path $Data)) {
    Write-Host "[!] Not found: $Data" -ForegroundColor Red
    exit 1
}

$AllArgs = @(
    "--data", $Data,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--val-split", "$ValSplit"
)
if ($Backbone) { $AllArgs += @("--backbone", $Backbone) }
if ($InitFrom) { $AllArgs += @("--init-from", $InitFrom) }
$AllArgs += $ExtraArgs

Write-Host "Args:" ($AllArgs -join " ")
Write-Host ""

& $Python $Script @AllArgs
