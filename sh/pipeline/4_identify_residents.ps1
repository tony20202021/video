# 6_3_identify_residents_files.ps1 — офлайн идентификация жителей из кропов 6_2 (Модель 2)
#
# Usage:
#   .\sh\cameras\6_3_identify_residents_files.ps1
#   .\sh\cameras\6_3_identify_residents_files.ps1 -IdentifyConf 0.75
#   .\sh\cameras\6_3_identify_residents_files.ps1 -Model ".models\identify\v1.onnx"

param(
    [Nullable[float]] $IdentifyConf,
    [string]          $Model     = "",
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

if ($null -eq $IdentifyConf) { $IdentifyConf = _ef "IDENTIFY_CONF" 0.70 }

$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\pipeline\4_identify_residents.py"

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

# Входные данные — выход скрипта 6_2
$S3Dir = "$Repo\.output\pipeline\3_classify_groups"

# Обрабатываем все прогоны 3_classify_groups разом (или укажи конкретные ниже)
$InputDirs = @(
    $S3Dir
    # Или конкретные прогоны:
    # "$S3Dir\run_20260628_202843_msk"
)

Write-Host "=== 4_identify_residents ===" -ForegroundColor Magenta
Write-Host "Repo:   $Repo"
Write-Host "Script: $Script"
Write-Host "identify_conf=$IdentifyConf"
if ($Model) { Write-Host "model=$Model" }
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
$AllArgs.Add("--identify-conf"); $AllArgs.Add("$IdentifyConf")
if ($Model) { $AllArgs.Add("--model"); $AllArgs.Add($Model) }
$AllArgs.AddRange($ExtraArgs)

Write-Host "Args:" ($AllArgs -join " ")
Write-Host ""

& $Python $Script @AllArgs
