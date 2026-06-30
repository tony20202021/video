# send.ps1 — отправить run_*-каталог на Transfer Server
#
# Usage:
#   .\sh\transfer\send.ps1 -Src .output\pipeline\1_motion_diff\run_20260629_XXX
#   .\sh\transfer\send.ps1 -Src run_XXX -Server http://1.2.3.4:8765 -Key SECRET

param(
    [Parameter(Mandatory=$true)]
    [string] $Src,
    [string] $Server = "",
    [string] $Key    = "",
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Repo   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\transfer\client.py"

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

# Разворачиваем относительный путь от корня репо
if (-not [System.IO.Path]::IsPathRooted($Src)) {
    $Src = "$Repo\$Src"
}

if (-not (Test-Path $Src)) {
    Write-Host "[!] Не найдено: $Src" -ForegroundColor Red
    exit 1
}

Write-Host "=== transfer send ===" -ForegroundColor Cyan
Write-Host "Src: $Src"

$AllArgs = @("send", $Src)
if ($Server) { $AllArgs += @("--server", $Server) }
if ($Key)    { $AllArgs += @("--key",    $Key) }
$AllArgs += $ExtraArgs

& $Python $Script @AllArgs
