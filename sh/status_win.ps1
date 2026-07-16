# Windows pipeline status
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File sh\status_win.ps1

$ErrorActionPreference = "SilentlyContinue"
# UTF-8 через SSH — иначе cp1251-консоль искажает кириллицу
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding            = [System.Text.Encoding]::UTF8

function Shorten-WinLog($line) {
    if (-not $line -or $line -eq "--") { return $line }
    $msg = $line.Trim()
    # Убрать "HH:MM:SS INFO  " в начале, оставить только время
    if ($msg -match '^(\d{2}:\d{2}:\d{2})\s+\w+\s+(.+)$') {
        $msg = $Matches[1] + "  " + $Matches[2].Trim()
    }
    # "Готово. Время: X с." → "Готово Xс"
    $msg = $msg -replace 'Готово\. Время:\s*([\d.]+)\s*с\.?', 'Готово $1с'
    # "images/20260715/cam_01_9_u/cam_01_9_u_..._heartbeat.jpg" → "cam_01_9_u"
    $msg = $msg -replace 'images/\d{8}/(cam_\d+_\d+_[a-z]+)/\S+', '$1'
    # "cam_01_9_u_20260715_083849_168730_msk_heartbeat.jpg" → "cam_01_9_u"
    $msg = $msg -replace '(cam_\d+_\d+_[a-z]+)_\d{8}_\S+', '$1'
    # Убрать " diff=0.25" и подобное
    $msg = $msg -replace '\s+diff=[\d.]+', ''
    # Сократить "Файлов нет в C:\...\path"
    $msg = $msg -replace '\(client\)\s+Файлов нет в \S+', 'Файлов нет'
    return $msg.Trim()
}
function Build-Stats($cnt, $times, $windowSec, $suffix, $cpuLine = "") {
    $s = "${cnt}x${suffix}"
    if ($times.Count -gt 0) {
        $mn  = [math]::Round(($times | Measure-Object -Minimum).Minimum, 1)
        $avg = [math]::Round(($times | Measure-Object -Average).Average, 1)
        $mx  = [math]::Round(($times | Measure-Object -Maximum).Maximum, 1)
        $lst = [math]::Round($times[$times.Count - 1], 1)
        $s  += "<br>${mn}с/${avg}с/${mx}с/${lst}с"
        $avgIv = $windowSec / $cnt
        $pArr  = @($times | ForEach-Object { [int][math]::Round($_ / $avgIv * 100) })
        $bp_mn  = ($pArr | Measure-Object -Minimum).Minimum
        $bp_avg = [int][math]::Round(($pArr | Measure-Object -Average).Average)
        $bp_mx  = ($pArr | Measure-Object -Maximum).Maximum
        $bp_lst = $pArr[$pArr.Count - 1]
        $s += "<br>${bp_mn}%/${bp_avg}%/${bp_mx}%/${bp_lst}%"
    }
    if ($cpuLine -ne "") { $s += "<br>${cpuLine}" }
    return $s
}
$REPO = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host ""
Write-Host ("  WINDOWS STATUS    {0}    {1}" -f $ts, $env:COMPUTERNAME)
Write-Host ""

# ── System ───────────────────────────────────────────────────────────────────
$cpu      = (Get-CimInstance Win32_Processor).LoadPercentage
$os       = Get-CimInstance Win32_OperatingSystem
$ic         = [System.Globalization.CultureInfo]::InvariantCulture
$ramTotalGB = ([math]::Round($os.TotalVisibleMemorySize / 1MB, 1)).ToString("F1", $ic)
$ramFreeGB  = ([math]::Round($os.FreePhysicalMemory  / 1MB, 1)).ToString("F1", $ic)
$ramUsedGB  = ([math]::Round([double]$ramTotalGB - [double]$ramFreeGB, 1)).ToString("F1", $ic)

$allDrives = Get-PSDrive -PSProvider FileSystem |
             Where-Object { $_.Used -ne $null -and $_.Free -ne $null -and ($_.Used + $_.Free) -gt 0 } |
             Sort-Object Name
$diskStr = ($allDrives | ForEach-Object {
    "{0}: {1}/{2} GB" -f $_.Name, [int]($_.Used/1GB), [int](($_.Used+$_.Free)/1GB)
}) -join "<br>"

$repoLetter  = [System.IO.Path]::GetPathRoot($REPO).TrimEnd('\:')
$repoDrive   = $allDrives | Where-Object { $_.Name -eq $repoLetter } | Select-Object -First 1
$repoDiskStr = if ($repoDrive) {
    "{0}: {1}/{2} GB" -f $repoDrive.Name, [int]($repoDrive.Used/1GB), [int](($repoDrive.Used+$repoDrive.Free)/1GB)
} else { "?" }

Write-Host ("  CPU: {0}%    RAM: {1}/{2} GB" -f $cpu, $ramUsedGB, $ramTotalGB)
Write-Host ("  Disks: {0}" -f $diskStr)
Write-Host ("  Work:  {0}  [{1}]" -f $REPO, $repoDiskStr)
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
        $lastFileTime = $top.LastWriteTime.ToString("yyyy-MM-dd HH:mm")
        $lastFileName = $top.Name
        $files | ForEach-Object {
            Write-Host ("    {0}  {1}" -f $_.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"), $_.Name)
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

# Если лог за сегодня ещё не создан — берём самую свежую доступную дату
if (-not (Test-Path $logFile)) {
    $metaRoot = Join-Path $motionDir "meta"
    if (Test-Path $metaRoot) {
        $latestDir = Get-ChildItem $metaRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($latestDir) { $logFile = Join-Path $latestDir.FullName "run.log" }
    }
}

$lastLogLine = "--"; $lastEventTime = "--"; $motionIdleLine = "--"; $count10m = 0

Write-Host ""
Write-Host "  Log motion_diff (last 5 lines):"
if (Test-Path $logFile) {
    $allLogLines = Get-Content $logFile -Encoding UTF8
    Get-Content $logFile -Tail 5 -Encoding UTF8 | ForEach-Object { Write-Host "    $_" }

    # Последнее событие работы (без cpu.csv и без heartbeat) — для колонок "Файл" и "Лог: работа"
    $eventLines = $allLogLines | Where-Object { $_ -match "^\d{2}:\d{2}:\d{2}\s+INFO\s+\S" -and $_ -notmatch "cpu\.csv" -and $_ -notmatch "heartbeat" -and $_ -notmatch "\[день\]" }
    if ($eventLines) {
        $le = @($eventLines)[-1]
        $lastLogLine = ($le.Trim() -replace '\|', ':')
        if ($le -match "^(\d{2}:\d{2})") {
            $lastEventTime = (Get-Date -Format "yyyy-MM-dd") + "<br>" + $Matches[1]
        }
    }

    # Лог: ожид. для motion_diff — последний heartbeat (камера жива, движения нет)
    $hbLines = $allLogLines | Where-Object { $_ -match "^\d{2}:\d{2}:\d{2}\s+INFO" -and $_ -match "heartbeat" }
    if ($hbLines) {
        $motionIdleLine = Shorten-WinLog (@($hbLines)[-1])
    }

    # Количество событий (diff-кадры): 10м + 60м fallback
    $cutoff   = (Get-Date).AddMinutes(-10)
    $cutoff60 = (Get-Date).AddMinutes(-60)
    $todayStr = Get-Date -Format "yyyy-MM-dd"
    $times10m = [System.Collections.Generic.List[double]]::new()
    $count60m = 0; $times60m = [System.Collections.Generic.List[double]]::new()
    foreach ($ln in $allLogLines) {
        # Одна комбинированная regex — $Matches[1]=время, diff= в строке
        if ($ln -match "^(\d{2}:\d{2}:\d{2})\s+INFO.+diff=") {
            $ts2 = $Matches[1]
            try {
                $lt  = [DateTime]::ParseExact("$todayStr $ts2", "yyyy-MM-dd HH:mm:ss", $null)
                $t_v = $null
                if ($ln -match 'Готово\. Время:\s*([\d.,]+)\s*с') { $t_v = [double]($Matches[1] -replace ',', '.') }
                if ($lt -ge $cutoff)   { $count10m++; if ($t_v -ne $null) { $times10m.Add($t_v) } }
                if ($lt -ge $cutoff60) { $count60m++; if ($t_v -ne $null) { $times60m.Add($t_v) } }
            } catch {}
        }
    }
} else {
    Write-Host "    (log not found: $logFile)"
}
Write-Host ""

# ── 2_send log ────────────────────────────────────────────────────────────────
$sendLogFile = Join-Path $REPO ".output\logs\2_send.log"
$sendLastLogLine = "--"; $sendLastEventTime = "--"; $sendLastIdleLine = "--"; $sendCount10m = 0

Write-Host "  Log 2_send (last 5 lines):"
if (Test-Path $sendLogFile) {
    $sendAllLines = Get-Content $sendLogFile -Encoding UTF8
    Get-Content $sendLogFile -Tail 5 -Encoding UTF8 | ForEach-Object { Write-Host "    $_" }

    # Последний отправленный файл (содержит расширение изображения)
    $sendEventLines = $sendAllLines | Where-Object { $_ -match "^\d{2}:\d{2}:\d{2}\s+INFO" -and $_ -match "\.(jpg|png|jpeg|bmp|webp)" }
    if ($sendEventLines) {
        $le = @($sendEventLines)[-1]
        $sendLastLogLine = ($le.Trim() -replace '\|', ':')
        if ($le -match "^(\d{2}:\d{2})") {
            $sendLastEventTime = (Get-Date -Format "yyyy-MM-dd") + "<br>" + $Matches[1]
        }
    }

    # Последнее сообщение ожидания ("нет" / "ожид")
    $sendIdleLines = $sendAllLines | Where-Object { $_ -match "^\d{2}:\d{2}:\d{2}\s+INFO" -and ($_ -match "ожид|нет|wait|empty") }
    if ($sendIdleLines) {
        $sendLastIdleLine = Shorten-WinLog (@($sendIdleLines)[-1])
    }

    $sendCutoff   = (Get-Date).AddMinutes(-10)
    $sendTodayStr = Get-Date -Format "yyyy-MM-dd"
    $sendTimes10m = [System.Collections.Generic.List[double]]::new()
    foreach ($ln in $sendAllLines) {
        # Одна комбинированная regex — $Matches[1]=время, расширение файла в строке
        if ($ln -match "^(\d{2}:\d{2}:\d{2})\s+INFO.+\.(jpg|png|jpeg|bmp|webp)") {
            $ts2 = $Matches[1]
            try {
                $lt = [DateTime]::ParseExact("$sendTodayStr $ts2", "yyyy-MM-dd HH:mm:ss", $null)
                if ($lt -ge $sendCutoff) {
                    $sendCount10m++
                    # "(47.8 КБ  297мс)" → 0.297с
                    if ($ln -match '\([\d.,]+\s*КБ\s+([\d.,]+)мс\)') {
                        $sendTimes10m.Add([double]($Matches[1] -replace ',', '.') / 1000.0)
                    }
                }
            } catch {}
        }
    }
} else {
    Write-Host "    (log not found - 2_send not yet restarted with new version)"
}
Write-Host ""

# ── CPU из cpu.csv (1_motion_diff) ───────────────────────────────────────────
$motionCpuLine = ""
$cpuCsvFile = Join-Path $motionDir "meta\$today\cpu.csv"
if (Test-Path $cpuCsvFile) {
    $cpuCutoff = (Get-Date).AddMinutes(-10)
    $cpuRows = Import-Csv $cpuCsvFile | Where-Object {
        try { [datetime]$_.ts_msk -ge $cpuCutoff } catch { $false }
    }
    if ($cpuRows -and @($cpuRows).Count -gt 0) {
        $cpuVals = @($cpuRows | ForEach-Object { [int][math]::Round([double]$_.cpu_pct) })
        $cpuMn  = ($cpuVals | Measure-Object -Minimum).Minimum
        $cpuAvg = [int][math]::Round(($cpuVals | Measure-Object -Average).Average)
        $cpuMx  = ($cpuVals | Measure-Object -Maximum).Maximum
        $cpuLst = $cpuVals[$cpuVals.Count - 1]
        $motionCpuLine = "${cpuMn}%/${cpuAvg}%/${cpuMx}%/${cpuLst}%"
    }
}

# ── Кадры из diffs.csv (1_motion_diff) — показывать когда нет событий движения
$frameCount10m = 0
$diffsCsvFile = Join-Path $motionDir "meta\$today\diffs.csv"
if (Test-Path $diffsCsvFile) {
    $diffsCutoff = (Get-Date).AddMinutes(-10)
    try {
        $diffsRows = Import-Csv $diffsCsvFile | Where-Object {
            try { [datetime]$_.ts_msk -ge $diffsCutoff } catch { $false }
        }
        if ($diffsRows) { $frameCount10m = @($diffsRows).Count }
    } catch {}
}

# ── WIN_SVC: структурированные данные для master status.sh ───────────────────
# Строки с префиксом "# WIN_SVC|" отфильтровываются из вывода на терминал,
# но используются status.sh для построения таблицы сервисов в .md

$winHost    = $env:COMPUTERNAME.ToLower()

# stats для 1_motion_diff: 10м, fallback 60м, fallback кадры из diffs.csv
if ($count10m -gt 0) {
    $stats10m = Build-Stats $count10m $times10m 600 " (10м)" $motionCpuLine
} elseif ($count60m -gt 0) {
    $stats10m = Build-Stats $count60m $times60m 3600 " (60м)" $motionCpuLine
} elseif ($frameCount10m -gt 0) {
    # нет событий движения, но кадры обрабатываются — показываем счётчик кадров
    $stats10m = "${frameCount10m}× кадров (10м)"
    if ($motionCpuLine -ne "") { $stats10m += "<br>${motionCpuLine}" }
} else { $stats10m = "--" }

# stats для 2_send: 10м с таймингом
if ($sendCount10m -gt 0) {
    $sendStats10m = Build-Stats $sendCount10m $sendTimes10m 600 " (10м)"
} else { $sendStats10m = "--" }

foreach ($s in $scriptDefs) {
    $pid2  = $scriptPids[$s.label]
    $stat  = if ($pid2) { "OK" } else { "NOK" }
    if ($s.label -eq "1_motion_diff") {
        $file2  = $lastEventTime
        $log2   = Shorten-WinLog $lastLogLine
        $idle2  = $motionIdleLine
        $st2    = $stats10m
    } elseif ($s.label -eq "2_send") {
        $file2  = $sendLastEventTime
        $log2   = Shorten-WinLog $sendLastLogLine
        $idle2  = $sendLastIdleLine
        $st2    = $sendStats10m
    } else {
        $file2 = "--"; $log2 = "--"; $idle2 = "--"; $st2 = "--"
    }
    Write-Host ("# WIN_SVC|{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}" -f `
        $s.label, $winHost, $stat, $s.input, $s.output, $file2, $log2, $idle2, $st2)
}
