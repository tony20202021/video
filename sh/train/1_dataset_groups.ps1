# 1_dataset_groups.ps1 — управление датасетом групп (Модель 1)
#
# Usage:
#   .\sh\train\1_dataset_groups.ps1 build
#   .\sh\train\1_dataset_groups.ps1 build -Labels .output\train\2_label_ui\labels.json
#   .\sh\train\1_dataset_groups.ps1 apply -Labels .data\groups\v1\new\labels.json -Dataset .data\groups\v1
#   .\sh\train\1_dataset_groups.ps1 apply -Labels .data\groups\v1\new\labels.json -Dataset .data\groups\v1 -Move
#   .\sh\train\1_dataset_groups.ps1 check -Src .data\groups\new -Dataset .data\groups\v1
#   .\sh\train\1_dataset_groups.ps1 add -Src .output\pipeline\2_yolo_boxes_files\run_xxx -Dataset .data\groups\v1
#   .\sh\train\1_dataset_groups.ps1 status -Dataset .data\groups\v1

param(
    [Parameter(Position=0)]
    [ValidateSet("build","apply","check","add","status")]
    [string] $Cmd = "apply",

    [string] $Labels  = "E:\_Home\Tony\pet projects\video\.data\groups\new\labels.json",   # build/apply: путь к labels.json
    [string] $Version = "",   # build: версия датасета (v1, v2 ...)
    [string] $Out     = "",   # build: явный output dir
    [switch] $Move    = $true,       # apply: переместить вместо копирования
    [string] $Src     = "",   # check/add: каталог источника
    [string] $Dataset = "E:\_Home\Tony\pet projects\video\.data\groups\v1"    # apply/check/add/status: каталог датасета
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\train\dataset_groups.py"

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

# Разворачиваем относительные пути
function Abs([string]$p) {
    if ($p -and -not [System.IO.Path]::IsPathRooted($p)) { return "$Repo\$p" }
    return $p
}

$AllArgs = @($Cmd)

switch ($Cmd) {
    "build" {
        if ($Labels)  { $AllArgs += @("--labels",  (Abs $Labels)) }
        if ($Version) { $AllArgs += @("--version", $Version) }
        if ($Out)     { $AllArgs += @("--out",     (Abs $Out)) }
    }
    "apply" {
        if (-not $Labels)  { Write-Host "[!] -Labels required" -ForegroundColor Red; exit 1 }
        if (-not $Dataset) { Write-Host "[!] -Dataset required" -ForegroundColor Red; exit 1 }
        $AllArgs += @("--labels", (Abs $Labels), "--dataset", (Abs $Dataset))
        if ($Move) { $AllArgs += "--move" }
    }
    "check" {
        if (-not $Src)     { Write-Host "[!] -Src required" -ForegroundColor Red; exit 1 }
        if (-not $Dataset) { Write-Host "[!] -Dataset required" -ForegroundColor Red; exit 1 }
        $AllArgs += @("--src", (Abs $Src), "--dataset", (Abs $Dataset))
    }
    "add" {
        if (-not $Src)     { Write-Host "[!] -Src required" -ForegroundColor Red; exit 1 }
        if (-not $Dataset) { Write-Host "[!] -Dataset required" -ForegroundColor Red; exit 1 }
        $AllArgs += @("--src", (Abs $Src), "--dataset", (Abs $Dataset))
    }
    "status" {
        if (-not $Dataset) { Write-Host "[!] -Dataset required" -ForegroundColor Red; exit 1 }
        $AllArgs += @("--dataset", (Abs $Dataset))
    }
}

Write-Host "=== dataset_groups: $Cmd ===" -ForegroundColor Cyan
Write-Host ""

& $Python $Script @AllArgs
