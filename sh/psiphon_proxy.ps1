# Показывает текущий адрес и порт прокси Psiphon

$proc = Get-Process "psiphon-tunnel-core" -ErrorAction SilentlyContinue

if (-not $proc) {
    Write-Host "Psiphon не запущен" -ForegroundColor Red
    exit 1
}

$ports = Get-NetTCPConnection -State Listen |
    Where-Object { $_.OwningProcess -eq $proc.Id } |
    Select-Object -ExpandProperty LocalPort |
    Sort-Object

if (-not $ports) {
    Write-Host "Psiphon запущен, но порты не найдены" -ForegroundColor Yellow
    exit 1
}

# Меньший порт — SOCKS, больший — HTTP (Psiphon всегда открывает оба)
$sorted = $ports | Sort-Object
$socks = $sorted[0]
$http  = $sorted[1]

Write-Host ""
Write-Host "Psiphon proxy ports:" -ForegroundColor Cyan
Write-Host "  HTTP:  http://127.0.0.1:$http" -ForegroundColor Green
Write-Host "  SOCKS: socks5://127.0.0.1:$socks" -ForegroundColor Green
Write-Host ""
Write-Host "Для Claude Code (PowerShell):" -ForegroundColor Cyan
Write-Host "  `$env:HTTPS_PROXY = `"http://127.0.0.1:$http`"" -ForegroundColor White
Write-Host ""
Write-Host "Для Cursor (Settings → Http: Proxy):" -ForegroundColor Cyan
Write-Host "  http://127.0.0.1:$http" -ForegroundColor White
