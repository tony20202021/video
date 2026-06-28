# 6_2_identify_people_files.ps1 — офлайн YOLO+ML-переобработка сохранённых прогонов S5
#
# Usage:
#   .\sh\cameras\6_2_identify_people_files.ps1
#   .\sh\cameras\6_2_identify_people_files.ps1 -YoloMaxFps 1.0
#   .\sh\cameras\6_2_identify_people_files.ps1 -NoMl

param(
    [float]   $YoloMaxFps = $(if ($env:YOLO_MAX_FPS) { [float]$env:YOLO_MAX_FPS } else { 2.0 }),
    [switch]  $NoMl,
    [string[]]$ExtraArgs  = @()
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$Conda  = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
$Env    = "conda_video"
$Script = "$Repo\scripts\cameras\6_2_identify_people_files.py"

$S5Dir  = "$Repo\.output\cameras\5_1_diff_yolo_boxes_low"

# Явный список прогонов для обработки
$InputDirs = @(
    "$S5Dir\run_20260627_174648_msk",
    "$S5Dir\run_20260627_191131_msk",
    "$S5Dir\run_20260627_205503_msk"
)

Write-Host "=== 6_2_identify_people_files ===" -ForegroundColor Magenta
Write-Host "Repo:   $Repo"
Write-Host "Script: $Script"
Write-Host "Прогоны:"
foreach ($d in $InputDirs) { Write-Host "  $d" }
Write-Host ""

$Missing = $InputDirs | Where-Object { -not (Test-Path $_) }
if ($Missing.Count -gt 0) {
    Write-Host "[!] Не найдены каталоги:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

$AllArgs = $InputDirs + @("--yolo-max-fps", "$YoloMaxFps")
if ($NoMl) { $AllArgs += "--no-ml" }
$AllArgs += $ExtraArgs

Write-Host "Аргументы:" ($AllArgs -join " ")
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"
& $Conda run -n $Env --no-capture-output python $Script @AllArgs
