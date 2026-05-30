# Start Cursor with Psiphon proxy env vars (for Claude Code panel).
# Usage: .\sh\start_cursor_with_proxy.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
& (Join-Path $repoRoot "sh\apply_psiphon_proxy_settings.ps1")

$proxyUrl = $env:HTTP_PROXY
if (-not $proxyUrl) {
    $settings = Get-Content (Join-Path $env:USERPROFILE ".claude\settings.json") -Raw | ConvertFrom-Json
    $proxyUrl = $settings.env.HTTPS_PROXY
}

$env:HTTP_PROXY  = $proxyUrl
$env:HTTPS_PROXY = $proxyUrl
$env:NO_PROXY    = "localhost,127.0.0.1"

$cursorExe = Join-Path $env:LOCALAPPDATA "Programs\cursor\Cursor.exe"
if (-not (Test-Path $cursorExe)) {
    throw "Cursor not found: $cursorExe"
}

Write-Host "Starting Cursor with proxy $proxyUrl" -ForegroundColor Cyan
Start-Process -FilePath $cursorExe
