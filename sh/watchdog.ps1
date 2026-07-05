# watchdog.ps1 — мониторинг и автоперезапуск процессов видеопайплайна
#
# Запускает и перезапускает при сбое/гибернации:
#   Transfer Server  (scripts/transfer/server.py)
#   Motion Diff      (scripts/pipeline/1_motion_diff.py)
#   Transfer Client  (scripts/transfer/client.py watch)
#
# Каждый процесс пишет в свой лог-файл (logs\*.log).
#
# Usage:
#   .\sh\watchdog.ps1                  — запустить watchdog (окно остаётся открытым)
#   .\sh\watchdog.ps1 -Register        — зарегистрировать как задачу при пробуждении системы
#   .\sh\watchdog.ps1 -Unregister      — удалить задачу

param(
    [switch] $Register,
    [switch] $Unregister,
    [int]    $CheckSec = 30       # интервал проверки живости процессов
)

$ErrorActionPreference = "Stop"
$Repo    = Split-Path -Parent $PSScriptRoot
$Conda   = "conda_video"
$Python  = "$env:USERPROFILE\miniconda3\envs\$Conda\python.exe"
$LogDir  = "$Repo\logs"
$TaskName = "VideoWatchdog"

$env:PYTHONIOENCODING = "utf-8"

# ─── Scheduled Task ────────────────────────────────────────────────────────────

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Задача '$TaskName' удалена."
    exit
}

if ($Register) {
    $ps   = (Get-Command powershell.exe).Source
    $self = $PSCommandPath
    $action = New-ScheduledTaskAction -Execute $ps `
        -Argument "-NonInteractive -WindowStyle Hidden -File `"$self`""
    # Запуск при старте системы
    $t1 = New-ScheduledTaskTrigger -AtStartup
    # Запуск при пробуждении: через XML (нет прямого командлета)
    $xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <EventTrigger>
      <Enabled>true</Enabled>
      <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="System"&gt;&lt;Select Path="System"&gt;*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
    </EventTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>$ps</Command>
      <Arguments>-NonInteractive -WindowStyle Hidden -File "$self"</Arguments>
    </Exec>
  </Actions>
  <Principals>
    <Principal id="Author">
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
  </Settings>
</Task>
"@
    $xmlPath = "$env:TEMP\watchdog_task.xml"
    $xml | Out-File $xmlPath -Encoding Unicode
    schtasks /Create /TN $TaskName /XML $xmlPath /F | Out-Null
    Remove-Item $xmlPath -ErrorAction SilentlyContinue
    Write-Host "Задача '$TaskName' зарегистрирована (при старте системы + при пробуждении)."
    Write-Host "Для удаления: .\sh\watchdog.ps1 -Unregister"
    exit
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

function Log([string]$msg) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[$ts] $msg"
}

New-Item -ItemType Directory -Force $LogDir | Out-Null

function Find-LatestImagesDir {
    $base = "$Repo\.output\pipeline\1_motion_diff"
    if (-not (Test-Path $base)) { return $null }
    $latest = Get-ChildItem $base -Filter "run_*" -Directory -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) { return $null }
    return "$($latest.FullName)\images"
}

function Start-BgProcess([string]$label, [string[]]$argList, [string]$logFile) {
    $proc = Start-Process $Python -ArgumentList $argList `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError  ($logFile -replace '\.log$', '_err.log') `
        -NoNewWindow -PassThru
    Log "  started $label  (PID $($proc.Id))  → $logFile"
    return $proc
}

function Is-Alive([System.Diagnostics.Process]$p) {
    if (-not $p)          { return $false }
    if ($p.HasExited)     { return $false }
    return $true
}

# ─── Основной цикл ────────────────────────────────────────────────────────────

Log "=== watchdog start ==="

$srvProc    = $null
$diffProc   = $null
$clientProc = $null

while ($true) {

    # ── Transfer Server ───────────────────────────────────────────────────────
    if (-not (Is-Alive $srvProc)) {
        if ($srvProc) { Log "server вышел (код $($srvProc.ExitCode))" }
        $srvProc = Start-BgProcess "server" @(
            "$Repo\scripts\transfer\server.py",
            "--host", "0.0.0.0",
            "--port", "8765",
            "--output", "$Repo\.output\transfer"
        ) "$LogDir\server.log"
        Start-Sleep 2
    }

    # ── Motion Diff ───────────────────────────────────────────────────────────
    if (-not (Is-Alive $diffProc)) {
        if ($diffProc) {
            Log "motion_diff вышел (код $($diffProc.ExitCode))"
            # Старый watch_dir больше не используется — убиваем клиента
            if (Is-Alive $clientProc) { $clientProc.Kill(); $clientProc = $null }
        }
        $diffProc = Start-BgProcess "motion_diff" @(
            "$Repo\scripts\pipeline\1_motion_diff.py"
        ) "$LogDir\motion_diff.log"

        # Ждём создания run_* (до 60с)
        Log "  ожидание run_* ..."
        $deadline = (Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep 3
            if (Find-LatestImagesDir) { break }
        }
    }

    # ── Transfer Client ───────────────────────────────────────────────────────
    if (-not (Is-Alive $clientProc)) {
        if ($clientProc) { Log "transfer client вышел (код $($clientProc.ExitCode))" }
        $wd = Find-LatestImagesDir
        if ($wd) {
            $clientProc = Start-BgProcess "client" @(
                "$Repo\scripts\transfer\client.py",
                "watch", $wd, "--ext", "jpg"
            ) "$LogDir\client.log"
        } else {
            Log "  watch_dir не найден, client не запущен (повтор через $CheckSec с)"
        }
    }

    Start-Sleep $CheckSec
}
