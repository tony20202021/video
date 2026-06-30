# 5_2_yolo_boxes_files.ps1 — YOLO-переобработка сохранённых прогонов S4 и S5
#
# Запускает 5_2_yolo_boxes_files.py на всех run_* из:
#   .output/cameras/4_motion_diff_low
#   .output/cameras/5_diff_yolo_boxes_low
#
# Usage:
#   .\sh\cameras\5_2_yolo_boxes_files.ps1
#   .\sh\cameras\5_2_yolo_boxes_files.ps1 --conf 0.4
#   .\sh\cameras\5_2_yolo_boxes_files.ps1 .output\cameras\5_diff_yolo_boxes_low\run_20260628_003445_msk

param(
    [Nullable[float]] $YoloMaxFps,
    [Nullable[float]] $Conf,
    [Nullable[float]] $Nms,
    [string[]]        $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# Читаем .env — KEY=VALUE, игнорируем комментарии
$_dotenv = @{}
$_envFile = "$Repo\.env"
if (Test-Path $_envFile) {
    Get-Content $_envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $_dotenv[$Matches[1]] = $Matches[2].Trim()
        }
    }
}
function _ef([string]$key, [float]$default) {
    if ($_dotenv.ContainsKey($key) -and $_dotenv[$key] -ne '') { return [float]$_dotenv[$key] }
    return $default
}

if ($null -eq $YoloMaxFps) { $YoloMaxFps = _ef "YOLO_MAX_FPS" 2.0 }
if ($null -eq $Conf)       { $Conf       = _ef "YOLO_CONF"    0.35 }
if ($null -eq $Nms)        { $Nms        = _ef "YOLO_NMS"     0.45 }

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

$Env     = "conda_video"
$Python  = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script  = "$Repo\scripts\pipeline\2_yolo_boxes_files.py"

$S1Dir   = "$Repo\.output\pipeline\1_motion_diff"

# Явный список прогонов для обработки (все run_* из 1_motion_diff)
# $S1Dir
# Или конкретные прогоны:
$InputDirs = @(
    "$S1Dir\run_20260630_073847_msk",
    "$S1Dir\run_20260630_084142_msk"
)

Write-Host "=== 2_yolo_boxes_files ===" -ForegroundColor Cyan
Write-Host "Repo:   $Repo"
Write-Host "Script: $Script"
Write-Host "Runs:"
foreach ($d in $InputDirs) { Write-Host "  $d" }
Write-Host ""

$Missing = $InputDirs | Where-Object { -not (Test-Path $_) }
if ($Missing.Count -gt 0) {
    Write-Host "[!] Dirs not found:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

$ExtraFlags = @("--yolo-max-fps", "$YoloMaxFps", "--conf", "$Conf", "--nms", "$Nms") + $ExtraArgs

foreach ($InputDir in $InputDirs) {
    Write-Host "--- $($InputDir | Split-Path -Leaf) ---" -ForegroundColor DarkCyan
    $AllArgs = @($InputDir) + $ExtraFlags
    Write-Host "Args:" ($AllArgs -join " ")
    Write-Host ""
    & $Python $Script @AllArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
