# 2_train_groups.ps1 — обучение классификатора группы (Модель 1)
#
# Usage:
#   .\sh\train\2_train_groups.ps1 -Data ".output\training\export_classify.zip"
#   .\sh\train\2_train_groups.ps1 -Data ".output\training\export_dir" -Epochs 30

param(
    [Parameter(Mandatory=$false)]
    [string]  $Data             = "E:\_Home\Tony\pet projects\video\.data\groups\v1",
    [int]     $Epochs           = 20,
    [int]     $BatchSize        = 32,
    [float]   $Lr               = 1e-3,
    [float]   $ValSplit         = 0.2,
    [bool]    $ClassWeights      = $true,   # взвешенная функция потерь
    [bool]    $WeightedSampling = $true,   # WeightedRandomSampler
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\train\3_train_groups.py"

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Разворачиваем относительный путь к данным от корня репо
if (-not [System.IO.Path]::IsPathRooted($Data)) {
    $Data = "$Repo\$Data"
}

Write-Host "=== 3_train_groups (Model 1) ===" -ForegroundColor Cyan
Write-Host "Repo:   $Repo"
Write-Host "Script: $Script"
Write-Host "Data:   $Data"
Write-Host "Epochs=$Epochs  BatchSize=$BatchSize  LR=$Lr"
Write-Host "Model output: $Repo\.models\classify\"
Write-Host "Backbone:     $Repo\.models\classify\backbone.pt"
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
if ($ClassWeights)      { $AllArgs += "--class-weights" }
if ($WeightedSampling)  { $AllArgs += "--weighted-sampling" }
$AllArgs += $ExtraArgs

Write-Host "Args:" ($AllArgs -join " ")
Write-Host ""

& $Python $Script @AllArgs
