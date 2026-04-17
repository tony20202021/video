# Показывает текущий адрес и порт прокси Psiphon

$logPath = "$env:APPDATA\Psiphon3\psiphon3.log"

if (-not (Test-Path $logPath)) {
    Write-Host "Лог Psiphon не найден: $logPath" -ForegroundColor Red
    exit 1
}

$content = Get-Content $logPath -Encoding UTF8

$http  = $content | Select-String "HTTP proxy is running on localhost port (\d+)"  | Select-Object -Last 1
$socks = $content | Select-String "SOCKS proxy is running on localhost port (\d+)" | Select-Object -Last 1

if ($http) {
    $httpPort = $http.Matches[0].Groups[1].Value
    Write-Host "HTTP:  http://127.0.0.1:$httpPort" -ForegroundColor Green
} else {
    Write-Host "HTTP:  не найден" -ForegroundColor Yellow
}

if ($socks) {
    $socksPort = $socks.Matches[0].Groups[1].Value
    Write-Host "SOCKS: socks5://127.0.0.1:$socksPort" -ForegroundColor Green
} else {
    Write-Host "SOCKS: не найден" -ForegroundColor Yellow
}
