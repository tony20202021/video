# Sync Psiphon HTTP proxy into Cursor + Claude Code settings.
# Run after Psiphon start or when ports change: .\sh\apply_psiphon_proxy_settings.ps1
# Undo: .\sh\remove_psiphon_proxy_settings.ps1

$ErrorActionPreference = "Stop"

function Get-PsiphonHttpProxyUrl {
    $proc = Get-Process "psiphon-tunnel-core" -ErrorAction SilentlyContinue
    if (-not $proc) {
        throw "Psiphon is not running (psiphon-tunnel-core)."
    }

    $ports = Get-NetTCPConnection -State Listen |
        Where-Object { $_.OwningProcess -eq $proc.Id } |
        Select-Object -ExpandProperty LocalPort |
        Sort-Object

    if (-not $ports -or $ports.Count -lt 2) {
        throw "Psiphon is running but HTTP/SOCKS ports were not found."
    }

    # Same order as psiphon_proxy.ps1: [0]=SOCKS, [1]=HTTP
    $httpPort = $ports[1]
    return "http://127.0.0.1:$httpPort"
}

$proxyUrl = Get-PsiphonHttpProxyUrl
$noProxy = "localhost,127.0.0.1"

$claudeSettingsPath = Join-Path $env:USERPROFILE ".claude\settings.json"
$claudeDir = Split-Path $claudeSettingsPath -Parent
if (-not (Test-Path $claudeDir)) {
    New-Item -ItemType Directory -Path $claudeDir -Force | Out-Null
}

$claudeSettings = @{
    theme = "auto"
    env   = @{
        HTTP_PROXY  = $proxyUrl
        HTTPS_PROXY = $proxyUrl
        NO_PROXY    = $noProxy
    }
}
$claudeSettings | ConvertTo-Json -Depth 5 | Set-Content -Path $claudeSettingsPath -Encoding utf8

$cursorSettingsPath = Join-Path $env:APPDATA "Cursor\User\settings.json"
if (-not (Test-Path $cursorSettingsPath)) {
    throw "Cursor settings not found: $cursorSettingsPath"
}

$cursor = Get-Content $cursorSettingsPath -Raw -Encoding utf8 | ConvertFrom-Json
$cursor | Add-Member -NotePropertyName "http.proxy" -NotePropertyValue $proxyUrl -Force
$cursor | Add-Member -NotePropertyName "http.proxySupport" -NotePropertyValue "on" -Force
$cursor | Add-Member -NotePropertyName "http.proxyStrictSSL" -NotePropertyValue $false -Force
$cursor | Add-Member -NotePropertyName "claudeCode.environmentVariables" -NotePropertyValue @(
    @{ name = "HTTP_PROXY";  value = $proxyUrl },
    @{ name = "HTTPS_PROXY"; value = $proxyUrl },
    @{ name = "NO_PROXY";    value = $noProxy }
) -Force

$cursor | ConvertTo-Json -Depth 10 | Set-Content -Path $cursorSettingsPath -Encoding utf8

Write-Host ""
Write-Host "Proxy applied: $proxyUrl" -ForegroundColor Green
Write-Host "  Claude: $claudeSettingsPath"
Write-Host "  Cursor: $cursorSettingsPath"
Write-Host ""
Write-Host "Restart Cursor completely, then open Claude Code panel." -ForegroundColor Cyan
