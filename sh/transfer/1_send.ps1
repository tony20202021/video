# 1_send.ps1 — постоянно следит за каталогом и отправляет новые файлы на Transfer Server
#
# Usage:
#   .\sh\transfer\1_send.ps1 -WatchDir .output\pipeline\1_motion_diff\run_XXX\images
#   .\sh\transfer\1_send.ps1 -WatchDir <path> -Server http://1.2.3.4:8765 -Key SECRET
#   .\sh\transfer\1_send.ps1 -WatchDir <path> -Ext "jpg,png"
#
# Параметры (RunRoot, Step, Run) по умолчанию выводятся из пути:
#   WatchDir → RunRoot = parent(WatchDir) = run_XXX
#   Step = parent(RunRoot) = 1_motion_diff
#   Run  = RunRoot.Name   = run_XXX
#
# Настройки адаптивной скорости берутся из .env:
#   TRANSFER_POLL_SEC, TRANSFER_MAX_RATE, TRANSFER_MIN_RATE,
#   TRANSFER_ADAPT_WINDOW, TRANSFER_ADAPT_HIGH, TRANSFER_ADAPT_LOW, TRANSFER_ADAPT_FACTOR

param(
    [Parameter(Mandatory=$false)]
    [string] $WatchDir = "E:\_Home\Tony\pet projects\video\.output\pipeline\1_motion_diff\run_20260702_200143_msk",
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

Write-Host "=== 1_send (transfer watch) ===" -ForegroundColor Cyan
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
