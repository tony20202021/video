# 5_track_direction.ps1 — трекинг людей и определение направления (домой/из дома)
#
# Читает detections.csv из выхода 2_yolo_boxes_files.
# Связывает боксы в треки, классифицирует: домой / из дома / неизвестно.
# Зоны «дверь» и «лифт» — в config.yaml (tracking.zones).
#
# Usage:
#   .\sh\pipeline\5_track_direction.ps1
#   .\sh\pipeline\5_track_direction.ps1 -Config "config.yaml"

param(
    [string]   $Config    = "",
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Repo   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\pipeline\5_track_direction.py"

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null
$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Выход скрипта 2_yolo_boxes_files — конкретные run_*
$S2Dir = "$Repo\.output\pipeline\2_yolo_boxes_files"

$InputDirs = @(
    "$S2Dir\run_20260629_210753_msk"
    # Добавляй нужные прогоны:
    # "$S2Dir\run_20260630_120000_msk"
)

Write-Host "=== 5_track_direction ===" -ForegroundColor Magenta
Write-Host "Repo:   $Repo"
Write-Host "Script: $Script"
Write-Host "Input:"
foreach ($d in $InputDirs) { Write-Host "  $d" }
Write-Host ""

$Missing = $InputDirs | Where-Object { -not (Test-Path $_) }
if ($Missing.Count -gt 0) {
    Write-Host "[!] Dirs not found:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

$ExtraFlags = @()
if ($Config) { $ExtraFlags += @("--config", $Config) }
$ExtraFlags += $ExtraArgs

foreach ($InputDir in $InputDirs) {
    Write-Host "--- $($InputDir | Split-Path -Leaf) ---" -ForegroundColor DarkCyan
    $AllArgs = @($InputDir) + $ExtraFlags
    Write-Host "Args:" ($AllArgs -join " ")
    Write-Host ""
    & $Python $Script @AllArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
