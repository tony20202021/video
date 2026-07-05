# 4_label_ui.ps1 — веб-разметчик кропов для обучения Модели 1
#
# Usage:
#   .\sh\train\4_label_ui.ps1
#   .\sh\train\4_label_ui.ps1 -Input ".output\cameras\5_2_yolo_boxes_files\run_20260628_202843_msk"
#   .\sh\train\4_label_ui.ps1 -Port 8080

param(
    [string] $InputDir      = "E:\_Home\Tony\pet projects\video\.data\groups\new",   # каталог кропов для разметки
    [string] $Labels        = "",   # файл меток; по умолчанию .output\train\2_label_ui\labels.json
    [string] $Dataset       = "E:\_Home\Tony\pet projects\video\.data\groups\v1",   # каталог датасета для чтения классов; по умолчанию .data\groups\последняя версия
    [int]    $Port          = 8750,
    [bool]   $UnlabeledOnly = $true,        # показывать только неразмеченные; -UnlabeledOnly $false — все
    [string] $Ext           = "jpg"         # расширения файлов через запятую
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\train\2_label_ui.py"

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null
$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Ищем последний run_* в 5_2 если Input не задан
if (-not $InputDir) {
    $S2 = "$Repo\.output\pipeline\2_yolo_boxes_files"
    if (Test-Path $S2) {
        $Latest = Get-ChildItem $S2 -Directory |
                  Where-Object { $_.Name -match '^run_' } |
                  Sort-Object LastWriteTime -Descending |
                  Select-Object -First 1
        if ($Latest) { $InputDir = $Latest.FullName }
    }
}

if (-not $InputDir) {
    Write-Host "[!] No run_* found in .output\pipeline\2_yolo_boxes_files" -ForegroundColor Red
    Write-Host "    Set explicitly: -Input <path to run>" -ForegroundColor Red
    exit 1
}

if (-not [System.IO.Path]::IsPathRooted($InputDir)) { $InputDir = "$Repo\$InputDir" }

if (-not $Labels) { $Labels = "$Repo\.output\train\2_label_ui\labels.json" }
if (-not [System.IO.Path]::IsPathRooted($Labels)) { $Labels = "$Repo\$Labels" }

Write-Host "=== 2_label_ui ===" -ForegroundColor Cyan
Write-Host "Repo:   $Repo"
Write-Host "Input:  $InputDir"
Write-Host "Labels: $Labels"
Write-Host "Port:   $Port"
Write-Host ""

if (-not (Test-Path $InputDir)) {
    Write-Host "[!] Not found: $InputDir" -ForegroundColor Red
    exit 1
}

if ($Dataset -and -not [System.IO.Path]::IsPathRooted($Dataset)) { $Dataset = "$Repo\$Dataset" }

$AllArgs = @("--input", $InputDir, "--labels", $Labels, "--port", "$Port")
if ($Dataset)       { $AllArgs += @("--dataset", $Dataset) }
if ($UnlabeledOnly) { $AllArgs += "--unlabeled-only" }
if ($Ext)           { $AllArgs += @("--ext", $Ext) }

& $Python $Script @AllArgs
