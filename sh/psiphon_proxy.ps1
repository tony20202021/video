# Shows current Psiphon proxy ports

$proc = Get-Process "psiphon-tunnel-core" -ErrorAction SilentlyContinue

if (-not $proc) {
    Write-Host "Psiphon is not running" -ForegroundColor Red
    exit 1
}

$ports = Get-NetTCPConnection -State Listen |
    Where-Object { $_.OwningProcess -eq $proc.Id } |
    Select-Object -ExpandProperty LocalPort |
    Sort-Object

if (-not $ports) {
    Write-Host "Psiphon is running but no ports found" -ForegroundColor Yellow
    exit 1
}

$socks = $ports[0]
$http  = $ports[1]

Write-Host ""
Write-Host "Psiphon proxy:" -ForegroundColor Cyan
Write-Host "  HTTP:  http://127.0.0.1:$http" -ForegroundColor Green
Write-Host "  SOCKS: socks5://127.0.0.1:$socks" -ForegroundColor Green
Write-Host ""
Write-Host "Claude Code:" -ForegroundColor Cyan
Write-Host "  `$env:HTTPS_PROXY = `"http://127.0.0.1:$http`""
Write-Host ""
Write-Host "Cursor (Settings -> Http: Proxy):" -ForegroundColor Cyan
Write-Host "  http://127.0.0.1:$http"
