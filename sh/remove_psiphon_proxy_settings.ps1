# Remove Psiphon proxy from Cursor + Claude Code settings.
# Usage: .\sh\remove_psiphon_proxy_settings.ps1

$ErrorActionPreference = "Stop"

$proxyEnvNames = @(
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY"
)

$cursorProxyProperties = @(
    "http.proxy",
    "http.proxySupport",
    "http.proxyStrictSSL",
    "claudeCode.environmentVariables"
)

function Remove-JsonNoteProperty {
    param(
        [Parameter(Mandatory = $true)]
        $Object,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if ($null -eq $Object) {
        return $false
    }

    $prop = $Object.PSObject.Properties[$Name]
    if ($prop) {
        $Object.PSObject.Properties.Remove($Name)
        return $true
    }

    return $false
}

# --- Claude Code (~/.claude/settings.json) ---
$claudeSettingsPath = Join-Path $env:USERPROFILE ".claude\settings.json"
$claudeChanged = $false

if (Test-Path $claudeSettingsPath) {
    $claude = Get-Content $claudeSettingsPath -Raw -Encoding utf8 | ConvertFrom-Json

    if ($claude.env) {
        foreach ($name in $proxyEnvNames) {
            if (Remove-JsonNoteProperty -Object $claude.env -Name $name) {
                $claudeChanged = $true
            }
        }

        if ($claude.env.PSObject.Properties.Count -eq 0) {
            if (Remove-JsonNoteProperty -Object $claude -Name "env") {
                $claudeChanged = $true
            }
        }
    }

    if ($claudeChanged) {
        $claude | ConvertTo-Json -Depth 5 | Set-Content -Path $claudeSettingsPath -Encoding utf8
    }
}

# --- Cursor ---
$cursorSettingsPath = Join-Path $env:APPDATA "Cursor\User\settings.json"
$cursorChanged = $false

if (-not (Test-Path $cursorSettingsPath)) {
    throw "Cursor settings not found: $cursorSettingsPath"
}

$cursor = Get-Content $cursorSettingsPath -Raw -Encoding utf8 | ConvertFrom-Json

foreach ($name in $cursorProxyProperties) {
    if (Remove-JsonNoteProperty -Object $cursor -Name $name) {
        $cursorChanged = $true
    }
}

if ($cursorChanged) {
    $cursor | ConvertTo-Json -Depth 10 | Set-Content -Path $cursorSettingsPath -Encoding utf8
}

Write-Host ""
if ($claudeChanged -or $cursorChanged) {
    Write-Host "Proxy settings removed." -ForegroundColor Green
} else {
    Write-Host "No proxy settings found (already clean)." -ForegroundColor Yellow
}

if (Test-Path $claudeSettingsPath) {
    Write-Host "  Claude: $claudeSettingsPath"
}
Write-Host "  Cursor: $cursorSettingsPath"
Write-Host ""
Write-Host "Restart Cursor for changes to take effect." -ForegroundColor Cyan
Write-Host "Current shell: run  Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:NO_PROXY -ErrorAction SilentlyContinue" -ForegroundColor DarkGray
