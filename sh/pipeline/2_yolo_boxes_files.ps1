# 2_yolo_boxes_files.ps1 — YOLO-переобработка прогонов из 1_motion_diff
#
# Batch (по умолчанию):
#   .\sh\pipeline\2_yolo_boxes_files.ps1
#   .\sh\pipeline\2_yolo_boxes_files.ps1 -Conf 0.4
#
# Watch-режим:
#   .\sh\pipeline\2_yolo_boxes_files.ps1 -Watch
#   .\sh\pipeline\2_yolo_boxes_files.ps1 -Watch -PollSec 3
#   .\sh\pipeline\2_yolo_boxes_files.ps1 -Watch -DeleteAfter   # удалять обработанные файлы

param(
    [Nullable[float]] $YoloMaxFps,
    [Nullable[float]] $Conf,
    [Nullable[float]] $Nms,
    [switch]          $Watch,
    [int]             $PollSec     = 5,
    [switch]          $DeleteAfter,
    [string[]]        $ExtraArgs   = @()
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# Читаем .env
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
$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\pipeline\2_yolo_boxes_files.py"

$S1Dir  = "$Repo\.output\pipeline\1_motion_diff"

$InputDirs = @(
    "$Repo\.output\transfer\diff"
)

$ExtraFlags = @("--yolo-max-fps", "$YoloMaxFps", "--conf", "$Conf", "--nms", "$Nms") + $ExtraArgs

Write-Host "=== 2_yolo_boxes_files ===" -ForegroundColor Cyan
Write-Host "Repo:   $Repo"
Write-Host "Script: $Script"
Write-Host "Runs:"
foreach ($d in $InputDirs) { Write-Host "  $d" }
if ($Watch) {
    $delLabel = if ($DeleteAfter) { "да" } else { "нет" }
    Write-Host "Режим:  watch (poll ${PollSec}s, delete-after=$delLabel)"
}
Write-Host ""

# ── Вспомогательная функция запуска ─────────────────────────────────────────
function Invoke-Yolo([string]$InputDir) {
    $AllArgs = @($InputDir) + $ExtraFlags
    & $Python $Script @AllArgs
    return $LASTEXITCODE
}

# ── Batch-режим ──────────────────────────────────────────────────────────────
if (-not $Watch) {
    $Missing = $InputDirs | Where-Object { -not (Test-Path $_) }
    if ($Missing.Count -gt 0) {
        Write-Host "[!] Dirs not found:" -ForegroundColor Red
        $Missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        exit 1
    }

    foreach ($InputDir in $InputDirs) {
        Write-Host "--- $($InputDir | Split-Path -Leaf) ---" -ForegroundColor DarkCyan
        $rc = Invoke-Yolo $InputDir
        if ($rc -ne 0) { exit $rc }
        Write-Host ""
    }
    exit 0
}

# ── Watch-режим ──────────────────────────────────────────────────────────────
# Состояние обработанных файлов (для режима без -DeleteAfter)
$SeenFiles = @{}
foreach ($d in $InputDirs) { $SeenFiles[$d] = [System.Collections.Generic.HashSet[string]]::new() }

# Загружаем ранее сохранённые состояния
foreach ($d in $InputDirs) {
    $stateFile = Join-Path $d ".yolo_seen"
    if (Test-Path $stateFile) {
        Get-Content $stateFile | ForEach-Object { [void]$SeenFiles[$d].Add($_) }
    }
}

Write-Host "[watch] Ctrl+C для остановки" -ForegroundColor DarkGray
Write-Host ""

while ($true) {
    $anyFound = $false

    foreach ($InputDir in $InputDirs) {
        if (-not (Test-Path $InputDir -PathType Container)) { continue }

        $AllFiles = Get-ChildItem -Path $InputDir -Recurse -Filter "*.jpg" -File |
                    Select-Object -ExpandProperty FullName | Sort-Object

        if (-not $AllFiles -or $AllFiles.Count -eq 0) { continue }

        if ($DeleteAfter) {
            $NewFiles = $AllFiles
        } else {
            $NewFiles = $AllFiles | Where-Object { -not $SeenFiles[$InputDir].Contains($_) }
        }

        if (-not $NewFiles -or @($NewFiles).Count -eq 0) { continue }

        $anyFound = $true
        $ts = (Get-Date).ToString("HH:mm:ss")
        $dirName = Split-Path $InputDir -Leaf
        Write-Host "[$ts] ${dirName}: $(@($NewFiles).Count) новых файлов" -ForegroundColor Green

        $rc = Invoke-Yolo $InputDir

        if ($rc -eq 0) {
            if ($DeleteAfter) {
                $NewFiles | ForEach-Object { Remove-Item $_ -Force -ErrorAction SilentlyContinue }
                $ts2 = (Get-Date).ToString("HH:mm:ss")
                Write-Host "[$ts2] Удалено: $(@($NewFiles).Count) файлов"
            } else {
                $stateFile = Join-Path $InputDir ".yolo_seen"
                $NewFiles | ForEach-Object {
                    [void]$SeenFiles[$InputDir].Add($_)
                    Add-Content -Path $stateFile -Value $_
                }
            }
        } else {
            Write-Host "[!] Скрипт вернул ошибку ($rc), файлы не удалены" -ForegroundColor Red
        }
        Write-Host ""
    }

    if (-not $anyFound) { Start-Sleep -Seconds $PollSec }
}
