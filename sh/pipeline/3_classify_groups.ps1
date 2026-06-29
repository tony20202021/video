# 6_2_classify_groups_files.ps1 — офлайн классификация кропов по группам (Модель 1)
#
# Usage:
#   .\sh\cameras\6_2_classify_groups_files.ps1
#   .\sh\cameras\6_2_classify_groups_files.ps1 -ClassifyConf 0.70

param(
    [Nullable[float]] $ClassifyConf,
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

if ($null -eq $ClassifyConf) { $ClassifyConf = _ef "CLASSIFY_CONF" 0.65 }

$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\pipeline\3_classify_groups.py"

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

# Входные данные — выход скрипта 5_2 (кропы уже вырезаны)
$S2Dir = "$Repo\.output\pipeline\2_yolo_boxes_files"

# Обрабатываем все прогоны 2_yolo_boxes_files разом (или укажи конкретные ниже)
$InputDirs = @(
    $S2Dir
    # Или конкретные прогоны:
    # "$S2Dir\run_20260628_202843_msk"
)

Write-Host "=== 3_classify_groups ===" -ForegroundColor Magenta
Write-Host "Repo:   $Repo"
Write-Host "Script: $Script"
Write-Host "classify_conf=$ClassifyConf"
Write-Host "Input:"
foreach ($d in $InputDirs) { Write-Host "  $d" }
Write-Host ""

$Missing = $InputDirs | Where-Object { -not (Test-Path $_) }
if ($Missing.Count -gt 0) {
    Write-Host "[!] Dirs not found:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

$AllArgs = [System.Collections.Generic.List[string]]$InputDirs
$AllArgs.Add("--classify-conf"); $AllArgs.Add("$ClassifyConf")
$AllArgs.AddRange($ExtraArgs)

Write-Host "Args:" ($AllArgs -join " ")
Write-Host ""

& $Python $Script @AllArgs
