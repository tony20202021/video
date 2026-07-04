# 1_start_server.ps1 — запуск Transfer Server на Windows
#
# Usage:
#   .\sh\transfer\1_start_server.ps1
#   .\sh\transfer\1_start_server.ps1 -Port 8765
#   .\sh\transfer\1_start_server.ps1 -Output D:\transfer_output

param(
    [string] $BindHost = "",    # default: из .env TRANSFER_HOST или 0.0.0.0
    [int]    $Port     = 0,     # default: из .env TRANSFER_PORT или 8765
    [string] $Output   = "E:\_Home\Tony\pet projects\video\.output\transfer",    # override каталог для принятых файлов
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Repo   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
$Script = "$Repo\scripts\transfer\server.py"

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
function _ef([string]$key, [string]$default) {
    if ($_dotenv.ContainsKey($key) -and $_dotenv[$key] -ne '') { return $_dotenv[$key] }
    return $default
}

if (-not $BindHost)  { $BindHost = _ef "TRANSFER_HOST" "0.0.0.0" }
if ($Port -eq 0) { $Port = [int](_ef "TRANSFER_PORT" "8765") }

$ApiKey = _ef "TRANSFER_API_KEY" ""
if ($ApiKey) {
    $env:TRANSFER_API_KEY = $ApiKey
} else {
    Write-Host "[!] TRANSFER_API_KEY не задан — сервер открыт без аутентификации" -ForegroundColor Yellow
}

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

Write-Host "=== Transfer Server (Windows) ===" -ForegroundColor Cyan
Write-Host "Repo:   $Repo"
Write-Host "Python: $Python"
Write-Host "Host:   $BindHost"
Write-Host "Port:   $Port"
Write-Host ""

$AllArgs = @("--host", $BindHost, "--port", "$Port")
if ($Output) { $AllArgs += @("--output", $Output) }
$AllArgs += $ExtraArgs

& $Python $Script @AllArgs
