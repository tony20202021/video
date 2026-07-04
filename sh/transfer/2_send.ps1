# 2_send.ps1 — постоянно следит за каталогом и отправляет новые файлы на Transfer Server
#
# Usage:
#   .\sh\transfer\2_send.ps1
#   .\sh\transfer\2_send.ps1 -WatchDir .output\pipeline\1_motion_diff\
#   .\sh\transfer\2_send.ps1 -WatchDir .output\pipeline\1_motion_diff\run_XXX\images
#   .\sh\transfer\2_send.ps1 -WatchDir <path> -Server http://1.2.3.4:8765 -Key SECRET
#
# Маршрутизация (client.py, per-file):
#   WatchDir = 1_motion_diff/  → parent=pipeline, step=1_motion_diff, run из пути файла
#   WatchDir = run_XXX/images   → parent=pipeline, step=1_motion_diff, run=run_XXX
#
# Настройки адаптивной скорости берутся из .env:
#   TRANSFER_POLL_SEC, TRANSFER_MAX_RATE, TRANSFER_MIN_RATE,
#   TRANSFER_ADAPT_WINDOW, TRANSFER_ADAPT_HIGH, TRANSFER_ADAPT_LOW, TRANSFER_ADAPT_FACTOR

param(
    [Parameter(Mandatory=$false)]
    [string] $WatchDir = "E:\_Home\Tony\pet projects\video\.output\pipeline\1_motion_diff\",
    [string] $RunRoot   = "",   # default: parent of WatchDir
    [string] $Step      = "",   # default: parent of RunRoot
    [string] $Run       = "",   # default: RunRoot.Name
    [string] $Server    = "",
    [string] $Key       = "",
    [string] $Ext       = "jpg",
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Repo   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\transfer\client.py"

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

if (-not [System.IO.Path]::IsPathRooted($WatchDir)) {
    $WatchDir = "$Repo\$WatchDir"
}
if ($RunRoot -and -not [System.IO.Path]::IsPathRooted($RunRoot)) {
    $RunRoot = "$Repo\$RunRoot"
}

Write-Host "=== 2_send (transfer watch) ===" -ForegroundColor Cyan
Write-Host "WatchDir: $WatchDir"

$AllArgs = @("watch", $WatchDir)
if ($RunRoot)  { $AllArgs += @("--run-root", $RunRoot) }
if ($Step)     { $AllArgs += @("--step",     $Step) }
if ($Run)      { $AllArgs += @("--run",      $Run) }
if ($Server)   { $AllArgs += @("--server",   $Server) }
if ($Key)      { $AllArgs += @("--key",      $Key) }
if ($Ext)      { $AllArgs += @("--ext",      $Ext) }
$AllArgs += $ExtraArgs

& $Python $Script @AllArgs
