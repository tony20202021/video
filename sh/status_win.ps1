# Windows pipeline status
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File sh\status_win.ps1

$ErrorActionPreference = "SilentlyContinue"
# UTF-8 через SSH — иначе cp1251-консоль искажает кириллицу
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding            = [System.Text.Encoding]::UTF8
$REPO = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host ""
Write-Host ("  WINDOWS STATUS    {0}    {1}" -f $ts, $env:COMPUTERNAME)
Write-Host ""

# ── System ───────────────────────────────────────────────────────────────────
$cpu      = (Get-CimInstance Win32_Processor).LoadPercentage
$os       = Get-CimInstance Win32_OperatingSystem
$ramTotal = [int]($os.TotalVisibleMemorySize / 1KB)
$ramFree  = [int]($os.FreePhysicalMemory / 1KB)
$ramUsed  = $ramTotal - $ramFree
$disk     = Get-PSDrive C
$diskUsed = [int]($disk.Used / 1GB)
$diskFree = [int]($disk.Free / 1GB)

Write-Host ("  CPU: {0}%    RAM: {1}/{2} MB    Disk C: {3} GB used / {4} GB free" `
    -f $cpu, $ramUsed, $ramTotal, $diskUsed, $diskFree)
Write-Host ""

# ── Pipeline scripts ─────────────────────────────────────────────────────────
$scriptDefs = @(
    @{ file = "1_motion_diff.ps1"; label = "1_motion_diff";
       input = "RTSP cameras"; output = ".output\pipeline\1_motion_diff" },
    @{ file = "2_send.ps1";        label = "2_send";
       input = "1_motion_diff"; output = "transfer server (HTTP POST)" }
)

$scriptPids = @{}
Write-Host "  Scripts:"
foreach ($s in $scriptDefs) {
    $procs = Get-WmiObject Win32_Process |
             Where-Object { $_.CommandLine -like "*$($s.file)*" }
    if ($procs) {
        $ids = ($procs | ForEach-Object { $_.ProcessId }) -join ", "
        $scriptPids[$s.label] = $ids
        Write-Host ("    {0,-22} OK  PID: {1}" -f $s.label, $ids)
    } else {
        $scriptPids[$s.label] = $null
        Write-Host ("    {0,-22} NOT RUNNING" -f $s.label)
    }
}

# ── Python processes ──────────────────────────────────────────────────────────
$pyProcs = Get-Process python
Write-Host ""
if ($pyProcs) {
    Write-Host ("  Python: {0} process(es)" -f @($pyProcs).Count)
    @($pyProcs) | ForEach-Object {
        Write-Host ("    PID {0,6}  CPU {1,7:N1}s  RAM {2,5} MB" `
            -f $_.Id, $_.CPU, [int]($_.WorkingSet / 1MB))
    }
} else {
    Write-Host "  Python: not running"
}

# ── Latest motion_diff files ──────────────────────────────────────────────────
$motionDir   = Join-Path $REPO ".output\pipeline\1_motion_diff"
$lastFileTime = "--"; $lastFileName = "--"

Write-Host ""
Write-Host "  Latest files (.output\pipeline\1_motion_diff):"
if (Test-Path $motionDir) {
    $files = Get-ChildItem $motionDir -Recurse -Filter *.jpg |
             Sort-Object LastWriteTime -Descending |
             Select-Object -First 5
    if ($files) {
        $top = @($files)[0]
        $lastFileTime = $top.LastWriteTime.ToString("MM-dd HH:mm")
        $lastFileName = $top.Name
        $files | ForEach-Object {
            Write-Host ("    {0}  {1}" -f $_.LastWriteTime.ToString("MM-dd HH:mm:ss"), $_.Name)
        }
    } else {
        Write-Host "    (no files)"
    }
} else {
    Write-Host "    (directory not found)"
}

# ── motion_diff log ───────────────────────────────────────────────────────────
$today   = Get-Date -Format "yyyyMMdd"
$logFile = Join-Path $motionDir "meta\$today\run.log"
$lastLogLine = "--"; $lastEventTime = "--"; $count10m = 0

Write-Host ""
Write-Host "  Log motion_diff (last 5 lines):"
if (Test-Path $logFile) {
    $allLogLines = Get-Content $logFile -Encoding UTF8
    Get-Content $logFile -Tail 5 -Encoding UTF8 | ForEach-Object { Write-Host "    $_" }

    # Последнее событие (без cpu.csv) — для колонок "Файл" и "Лог: работа"
    $eventLines = $allLogLines | Where-Object { $_ -match "^\d{2}:\d{2}:\d{2}\s+INFO" -and $_ -notmatch "cpu\.csv" }
    if ($eventLines) {
        $le = @($eventLines)[-1]
        $lastLogLine = ($le.Trim() -replace '\|', ':')
        if ($le -match "^(\d{2}:\d{2})") {
            $lastEventTime = (Get-Date -Format "MM-dd") + " " + $Matches[1]
        }
    }

    # Количество событий за последние 10 минут (без строк cpu.csv)
    $cutoff   = (Get-Date).AddMinutes(-10)
    $todayStr = Get-Date -Format "yyyy-MM-dd"
    foreach ($ln in $allLogLines) {
        if ($ln -match "^(\d{2}:\d{2}:\d{2})\s+INFO" -and $ln -notmatch "cpu\.csv") {
            $ts2 = $Matches[1]
            try {
                $lt = [DateTime]::ParseExact("$todayStr $ts2", "yyyy-MM-dd HH:mm:ss", $null)
                if ($lt -ge $cutoff) { $count10m++ }
            } catch {}
        }
    }
} else {
    Write-Host "    (log not found: $logFile)"
}
Write-Host ""

# ── 2_send log ────────────────────────────────────────────────────────────────
$sendLogFile = Join-Path $REPO ".output\logs\2_send.log"
$sendLastLogLine = "--"; $sendLastEventTime = "--"; $sendCount10m = 0

Write-Host "  Log 2_send (last 5 lines):"
if (Test-Path $sendLogFile) {
    $sendAllLines = Get-Content $sendLogFile -Encoding UTF8
    Get-Content $sendLogFile -Tail 5 -Encoding UTF8 | ForEach-Object { Write-Host "    $_" }

    # match sent-file lines: contain a file extension (ASCII-safe, avoids non-ASCII arrow char)
    $sendEventLines = $sendAllLines | Where-Object { $_ -match "^\d{2}:\d{2}:\d{2}\s+INFO" -and $_ -match "\.(jpg|png|jpeg|bmp|webp)" }
    if ($sendEventLines) {
        $le = @($sendEventLines)[-1]
        $sendLastLogLine = ($le.Trim() -replace '\|', ':')
        if ($le -match "^(\d{2}:\d{2})") {
            $sendLastEventTime = (Get-Date -Format "MM-dd") + " " + $Matches[1]
        }
    }

    $sendCutoff   = (Get-Date).AddMinutes(-10)
    $sendTodayStr = Get-Date -Format "yyyy-MM-dd"
    foreach ($ln in $sendAllLines) {
        if ($ln -match "^(\d{2}:\d{2}:\d{2})\s+INFO" -and $ln -match "\.(jpg|png|jpeg|bmp|webp)") {
            $ts2 = $Matches[1]
            try {
                $lt = [DateTime]::ParseExact("$sendTodayStr $ts2", "yyyy-MM-dd HH:mm:ss", $null)
                if ($lt -ge $sendCutoff) { $sendCount10m++ }
            } catch {}
        }
    }
} else {
    Write-Host "    (log not found - 2_send not yet restarted with new version)"
}
Write-Host ""

# ── WIN_SVC: структурированные данные для master status.sh ───────────────────
# Строки с префиксом "# WIN_SVC|" отфильтровываются из вывода на терминал,
# но используются status.sh для построения таблицы сервисов в .md

$winHost    = $env:COMPUTERNAME.ToLower()
$stats10m   = if ($count10m -gt 0) { "${count10m}x" } else { "--" }
$sendStats10m = if ($sendCount10m -gt 0) { "${sendCount10m}x" } else { "--" }

foreach ($s in $scriptDefs) {
    $pid2  = $scriptPids[$s.label]
    $stat  = if ($pid2) { "OK" } else { "NOK" }
    if ($s.label -eq "1_motion_diff") {
        $file2 = $lastEventTime; $log2 = $lastLogLine;     $st2 = $stats10m
    } elseif ($s.label -eq "2_send") {
        $file2 = $sendLastEventTime; $log2 = $sendLastLogLine; $st2 = $sendStats10m
    } else {
        $file2 = "--"; $log2 = "--"; $st2 = "--"
    }
    Write-Host ("# WIN_SVC|{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}" -f `
        $s.label, $winHost, $stat, $s.input, $s.output, $file2, $log2, $st2)
}
