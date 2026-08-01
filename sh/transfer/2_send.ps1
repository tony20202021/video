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
    [string] $WatchDir = "",   # default: $Repo\.output\pipeline\1_motion_diff
    [string] $RunRoot   = "",   # default: parent of WatchDir
    [string] $Step      = "",   # default: parent of RunRoot
    [string] $Run       = "",   # default: RunRoot.Name
    [string] $Server    = "",
    [string] $Key       = "",
    [string] $Ext       = "jpg",
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Continue"
$Repo   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Env    = "conda_video"
$Python = "$env:USERPROFILE\miniconda3\envs\$Env\python.exe"
# Fallback для запуска под SYSTEM (без пользовательской сессии, USERPROFILE ≠ профиль с conda)
if (-not (Test-Path $Python)) {
    $found = Get-Item "C:\Users\*\miniconda3\envs\$Env\python.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $Python = $found.FullName }
}
$Script = "$Repo\scripts\transfer\client.py"

$env:PYTHONIOENCODING  = "utf-8"
$env:PYTHONUNBUFFERED = "1"
try { chcp 65001 | Out-Null } catch {}
try {
    $OutputEncoding            = [System.Text.Encoding]::UTF8
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    [Console]::InputEncoding  = [System.Text.Encoding]::UTF8
} catch {}

if (-not $WatchDir) {
    $WatchDir = Join-Path $Repo ".output\pipeline\1_motion_diff"
} elseif (-not [System.IO.Path]::IsPathRooted($WatchDir)) {
    $WatchDir = Join-Path $Repo $WatchDir
}
$WatchDir = [System.IO.Path]::GetFullPath($WatchDir.TrimEnd('\', '/'))
if ($RunRoot -and -not [System.IO.Path]::IsPathRooted($RunRoot)) {
    $RunRoot = "$Repo\$RunRoot"
}

# Лог ПО ДАТАМ (как run.log у motion_diff): transfer_client/meta/<date>/2_send.log — рядом с
# cpu.csv клиента, не растёт бесконечно. Раньше был единый .output/logs/2_send.log → рос до 14 МБ
# и тормозил status_win.ps1 (читал весь файл). Ротация — на полночь в цикле записи ниже.
$LogBase = Join-Path $Repo ".output\transfer_client\meta"
function New-SendWriter([string]$dateStr) {
    $dir = Join-Path $LogBase $dateStr
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    # FileShare.ReadWrite — не мешает статусу читать; Append — дописываем при рестарте в тот же день
    $fs = [System.IO.FileStream]::new((Join-Path $dir "2_send.log"),
              [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    $w = [System.IO.StreamWriter]::new($fs, [System.Text.Encoding]::UTF8)
    $w.AutoFlush = $true
    return $w
}
$LogFile = Join-Path (Join-Path $LogBase (Get-Date -Format "yyyyMMdd")) "2_send.log"

# Снижаем приоритет — motion_diff (Normal) всегда получает CPU первым
[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass =
    [System.Diagnostics.ProcessPriorityClass]::BelowNormal

Write-Host "=== 2_send (transfer watch) ===" -ForegroundColor Cyan
Write-Host "WatchDir: $WatchDir"
Write-Host "Log:      $LogFile"

$AllArgs = @("watch", $WatchDir)
if ($RunRoot)  { $AllArgs += @("--run-root", $RunRoot) }
if ($Step)     { $AllArgs += @("--step",     $Step) }
if ($Run)      { $AllArgs += @("--run",      $Run) }
if ($Server)   { $AllArgs += @("--server",   $Server) }
if ($Key)      { $AllArgs += @("--key",      $Key) }
if ($Ext)      { $AllArgs += @("--ext",      $Ext) }
$AllArgs += $ExtraArgs

# Состояние лога — в hashtable (ссылочный тип), чтобы менять из ForEach-Object без проблем со scope.
$log = @{ date = (Get-Date -Format "yyyyMMdd"); writer = $null }
$log.writer = New-SendWriter $log.date
$log.writer.WriteLine("$(Get-Date -Format 'HH:mm:ss')  INFO      === 2_send start ===")
try {
    & $Python $Script @AllArgs 2>&1 | ForEach-Object {
        Write-Host $_
        $d = Get-Date -Format "yyyyMMdd"
        if ($d -ne $log.date) {                 # смена суток → новый файл (ротация по датам)
            $log.writer.WriteLine("$(Get-Date -Format 'HH:mm:ss')  INFO      === 2_send rotate -> $d ===")
            $log.writer.Close()
            $log.date   = $d
            $log.writer = New-SendWriter $d
        }
        $log.writer.WriteLine($_)
    }
} finally {
    $log.writer.WriteLine("$(Get-Date -Format 'HH:mm:ss')  INFO      === 2_send stop ===")
    $log.writer.Close()
}
