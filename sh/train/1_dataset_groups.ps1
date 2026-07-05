# 1_dataset_groups.ps1 — управление датасетом групп (Модель 1)
#
# Команды:
#   build   — собрать zip-архив датасета из labels.json для обучения
#   apply   — разложить кропы из new/ по папкам классов (1_resident, 2_delivery, ...)
#   check   — найти дубли: сравнить Src с датасетом → new/unique/ и new/double/
#   add     — скопировать новые кропы из пайплайна в new/ датасета (перед разметкой)
#   status  — показать сколько изображений в каждом классе
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
    [string] $Src     = "E:\_Home\Tony\pet projects\video\.data\groups\new",   # check/add: каталог источника
    [string] $Dataset = "E:\_Home\Tony\pet projects\video\.data\groups\v1",    # apply/check/add/status: каталог датасета
    [string] $Ext     = "jpg"        # check/add: расширения файлов через запятую
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\train\dataset_groups.py"

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null
$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Разворачиваем относительные пути
function Abs([string]$p) {
    if ($p -and -not [System.IO.Path]::IsPathRooted($p)) { return "$Repo\$p" }
    return $p
}

$AllArgs = @($Cmd)

switch ($Cmd) {
    "build" {
        # Собрать датасет из labels.json → zip-архив для обучения
        if ($Labels)  { $AllArgs += @("--labels",  (Abs $Labels)) }
        if ($Version) { $AllArgs += @("--version", $Version) }
        if ($Out)     { $AllArgs += @("--out",     (Abs $Out)) }
    }
    "apply" {
        # Применить разметку: переложить кропы из new/ в папки классов датасета
        if (-not $Labels)  { Write-Host "[!] -Labels required" -ForegroundColor Red; exit 1 }
        if (-not $Dataset) { Write-Host "[!] -Dataset required" -ForegroundColor Red; exit 1 }
        $AllArgs += @("--labels", (Abs $Labels), "--dataset", (Abs $Dataset))
        if ($Move) { $AllArgs += "--move" }   # -Move: переместить (не копировать)
    }
    "check" {
        # Проверить дубли: сравнить Src с датасетом → unique/ и double/
        if (-not $Src)     { Write-Host "[!] -Src required" -ForegroundColor Red; exit 1 }
        if (-not $Dataset) { Write-Host "[!] -Dataset required" -ForegroundColor Red; exit 1 }
        $AllArgs += @("--src", (Abs $Src), "--dataset", (Abs $Dataset), "--ext", $Ext)
    }
    "add" {
        # Добавить новые кропы из Src в new/ датасета (для последующей разметки)
        if (-not $Src)     { Write-Host "[!] -Src required" -ForegroundColor Red; exit 1 }
        if (-not $Dataset) { Write-Host "[!] -Dataset required" -ForegroundColor Red; exit 1 }
        $AllArgs += @("--src", (Abs $Src), "--dataset", (Abs $Dataset), "--ext", $Ext)
    }
    "status" {
        # Показать статистику датасета: кол-во изображений по классам
        if (-not $Dataset) { Write-Host "[!] -Dataset required" -ForegroundColor Red; exit 1 }
        $AllArgs += @("--dataset", (Abs $Dataset))
    }
}

Write-Host "=== dataset_groups: $Cmd ===" -ForegroundColor Cyan
Write-Host ""

& $Python $Script @AllArgs
