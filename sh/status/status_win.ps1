# Windows pipeline status
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File sh\status\status_win.ps1

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
    }
    if ($cpuLine -ne "") { $s += "<br>${cpuLine}" }
    return $s
}

function _Plural($n, $one, $few, $many) {
    $n100 = [math]::Abs($n) % 100
    if ($n100 -ge 11 -and $n100 -le 14) { return $many }
    $n10 = $n100 % 10
    if ($n10 -eq 1) { return $one }
    if ($n10 -ge 2 -and $n10 -le 4) { return $few }
    return $many
}

# Единый формат «N прогонов (X файлов)» (как у Linux-сервисов). motion_diff → runs=1
# (непрерывный цикл), 2_send → runs=число батчей отправки.
function Fmt-Runs($runs, $files, $suffix, $times, $cpuLine = "", $procMs = $null) {
    $rw = _Plural $runs "прогон" "прогона" "прогонов"
    $fw = _Plural $files "файл" "файла" "файлов"
    $s = "${runs} ${rw}${suffix} (${files} ${fw})"
    # ЧИСТОЕ время обработки кадра (gray+diff) — из 'счёт/кадр: min/avg/max мс' в run.log
    if ($procMs -ne $null) {
        $s += "<br>счёт/кадр $([math]::Round($procMs[0]))/$([math]::Round($procMs[1]))/$([math]::Round($procMs[2])) мс"
    }
    # 'Готово. Время' у motion_diff — это ИНТЕРВАЛ между сохранёнными кадрами (тишина), не обработка
    if ($times.Count -gt 0) {
        $mn  = [math]::Round(($times | Measure-Object -Minimum).Minimum, 1)
        $avg = [math]::Round(($times | Measure-Object -Average).Average, 1)
        $mx  = [math]::Round(($times | Measure-Object -Maximum).Maximum, 1)
        $lst = [math]::Round($times[$times.Count - 1], 1)
        $s  += "<br>интервал ${mn}с/${avg}с/${mx}с/${lst}с"
    }
    if ($cpuLine -ne "") { $s += "<br>${cpuLine}" }
    return $s
}
$REPO = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
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
       # Ловим и штатный .ps1-обёртку, и прямой python-прогон (напр. бенчмарк
       # `python 1_motion_diff.py --env _bench.env` в каталоге 1_motion_diff_3cam),
       # иначе идущий бенчмарк показывается ложным NOK.
       match = @("1_motion_diff.ps1", "1_motion_diff.py");
       input = "RTSP cameras"; output = ".output\pipeline\1_motion_diff" },
    @{ file = "2_send.ps1";        label = "2_send";
       match = @("2_send.ps1");
       input = "1_motion_diff"; output = "transfer server (HTTP POST)" }
)

$scriptPids = @{}
Write-Host "  Scripts:"
foreach ($s in $scriptDefs) {
    $pats  = if ($s.match) { $s.match } else { @($s.file) }
    $procs = Get-WmiObject Win32_Process |
             Where-Object { $cl = $_.CommandLine; $cl -and (@($pats | Where-Object { $cl -like "*$_*" }).Count -gt 0) }
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

# Дата лога = дата последней записи файла. Для выключенной машины это прошлый запуск,
# а не сегодня — иначе старые строки лога считаются как сегодняшняя активность.
$logDate = if (Test-Path $logFile) { (Get-Item $logFile).LastWriteTime.ToString("yyyy-MM-dd") } else { Get-Date -Format "yyyy-MM-dd" }

$lastLogLine = "--"; $lastEventTime = "--"; $motionIdleLine = "--"; $count10m = 0
$motionRuns10m = 1; $motionRuns60m = 1

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
            $lastEventTime = $logDate + "<br>" + $Matches[1]
        }
    }

    # Лог: ожид. для motion_diff — последний heartbeat (камера жива, движения нет)
    $hbLines = $allLogLines | Where-Object { $_ -match "^\d{2}:\d{2}:\d{2}\s+INFO" -and $_ -match "heartbeat" }
    if ($hbLines) {
        $motionIdleLine = Shorten-WinLog (@($hbLines)[-1])
    }

    # Diff-кадры (файлы) И число прогонов (стартов «Порог:») за 10м/60м.
    # Рестарт motion_diff / переход через полночь → строки прогона попадают в РАЗНЫЕ
    # date-каталоги → читаем 2 свежих каталога, у каждой строки своя дата (имя каталога).
    $cutoff   = (Get-Date).AddMinutes(-10)
    $cutoff60 = (Get-Date).AddMinutes(-60)
    $times10m = [System.Collections.Generic.List[double]]::new()
    $count60m = 0; $times60m = [System.Collections.Generic.List[double]]::new()
    $runs10m  = 0; $runs60m  = 0
    $procMs10m = $null; $procMs60m = $null   # последняя тройка 'счёт/кадр' (реальная обработка) в окне
    $metaRoot2  = Join-Path $motionDir "meta"
    $recentDirs = if (Test-Path $metaRoot2) { Get-ChildItem $metaRoot2 -Directory | Sort-Object Name -Descending | Select-Object -First 2 } else { @() }
    foreach ($d in $recentDirs) {
        $rf = Join-Path $d.FullName "run.log"
        if (-not (Test-Path $rf)) { continue }
        try { $dIso = [datetime]::ParseExact($d.Name, "yyyyMMdd", $null).ToString("yyyy-MM-dd") } catch { continue }
        foreach ($ln in (Get-Content $rf -Encoding UTF8)) {
            if ($ln -notmatch "^(\d{2}:\d{2}:\d{2})\s+INFO") { continue }
            $ts2 = $Matches[1]
            try { $lt = [DateTime]::ParseExact("$dIso $ts2", "yyyy-MM-dd HH:mm:ss", $null) } catch { continue }
            $isStart = ($ln -match "Порог:")          # старт прогона motion_diff
            $isDiff  = ($ln -match "diff=")           # сохранённый diff-кадр
            $t_v = $null
            if ($isDiff -and ($ln -match 'Готово\. Время:\s*([\d.,]+)\s*с')) { $t_v = [double]($Matches[1] -replace ',', '.') }
            # реальная обработка кадра: 'счёт/кадр: min/avg/max мс' (есть и на diff-, и на пульс-строках)
            $pm = $null
            if ($ln -match 'счёт/кадр:\s*([\d.,]+)/([\d.,]+)/([\d.,]+)\s*мс') {
                $pm = @([double]($Matches[1] -replace ',','.'), [double]($Matches[2] -replace ',','.'), [double]($Matches[3] -replace ',','.'))
            }
            if ($lt -ge $cutoff)   { if ($isDiff) { $count10m++; if ($t_v -ne $null) { $times10m.Add($t_v) } }; if ($isStart) { $runs10m++ }; if ($pm) { $procMs10m = $pm } }
            if ($lt -ge $cutoff60) { if ($isDiff) { $count60m++; if ($t_v -ne $null) { $times60m.Add($t_v) } }; if ($isStart) { $runs60m++ }; if ($pm) { $procMs60m = $pm } }
        }
    }
    # нет стартов в окне, но есть активность → 1 непрерывный прогон; иначе N рестартов
    $motionRuns10m = if ($runs10m -gt 0) { $runs10m } else { 1 }
    $motionRuns60m = if ($runs60m -gt 0) { $runs60m } else { 1 }
} else {
    Write-Host "    (log not found: $logFile)"
}
Write-Host ""

# ── 2_send log ────────────────────────────────────────────────────────────────
$sendLogFile = Join-Path $REPO ".output\logs\2_send.log"
$sendLastLogLine = "--"; $sendLastEventTime = "--"; $sendLastIdleLine = "--"; $sendCount10m = 0; $sendBatches10m = 0

Write-Host "  Log 2_send (last 5 lines):"
if (Test-Path $sendLogFile) {
    $sendLogDate = (Get-Item $sendLogFile).LastWriteTime.ToString("yyyy-MM-dd")
    $sendAllLines = Get-Content $sendLogFile -Encoding UTF8
    Get-Content $sendLogFile -Tail 5 -Encoding UTF8 | ForEach-Object { Write-Host "    $_" }

    # Последний отправленный файл (содержит расширение изображения)
    $sendEventLines = $sendAllLines | Where-Object { $_ -match "^\d{2}:\d{2}:\d{2}\s+INFO" -and $_ -match "\.(jpg|png|jpeg|bmp|webp)" }
    if ($sendEventLines) {
        $le = @($sendEventLines)[-1]
        $sendLastLogLine = ($le.Trim() -replace '\|', ':')
        if ($le -match "^(\d{2}:\d{2})") {
            $sendLastEventTime = $sendLogDate + "<br>" + $Matches[1]
        }
    }

    # Последнее сообщение ожидания ("нет" / "ожид")
    $sendIdleLines = $sendAllLines | Where-Object { $_ -match "^\d{2}:\d{2}:\d{2}\s+INFO" -and ($_ -match "ожид|нет|wait|empty") }
    if ($sendIdleLines) {
        $sendLastIdleLine = Shorten-WinLog (@($sendIdleLines)[-1])
    }

    $sendCutoff   = (Get-Date).AddMinutes(-10)
    $sendTodayStr = $sendLogDate
    $sendTimes10m = [System.Collections.Generic.List[double]]::new()
    # 2_send.log — ЕДИНЫЙ файл за все дни; все строки датируются одним днём, поэтому старые
    # отправки с временем суток позже cutoff ложно попадают в 10-мин окно (баг «1997x»).
    # Считаем только строки ТЕКУЩЕГО запуска — после последнего маркера '=== 2_send start ==='.
    $sendStartIdx = -1
    for ($i = $sendAllLines.Count - 1; $i -ge 0; $i--) {
        if ($sendAllLines[$i] -match '=== 2_send start ===') { $sendStartIdx = $i; break }
    }
    $sendRunLines = if ($sendStartIdx -ge 0 -and $sendStartIdx -lt ($sendAllLines.Count - 1)) {
        $sendAllLines[($sendStartIdx + 1)..($sendAllLines.Count - 1)]
    } else { $sendAllLines }
    foreach ($ln in $sendRunLines) {
        # Одна комбинированная regex — $Matches[1]=время, расширение файла в строке
        if ($ln -match "^(\d{2}:\d{2}:\d{2})\s+INFO.+\.(jpg|png|jpeg|bmp|webp)") {
            $ts2 = $Matches[1]
            try {
                $lt = [DateTime]::ParseExact("$sendTodayStr $ts2", "yyyy-MM-dd HH:mm:ss", $null)
                if ($lt -ge $sendCutoff) {
                    $sendCount10m++
                    if ($ln -match 'в батче 1/') { $sendBatches10m++ }
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
    # ts_msk формат: 20260716_084240_891271_msk → парсим YYYYMMDD_HHMMSS
    $cpuRows = Import-Csv $cpuCsvFile | Where-Object {
        try {
            $ts = $_.ts_msk
            if ($ts -match '^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})') {
                [datetime]::ParseExact("$($Matches[1])-$($Matches[2])-$($Matches[3]) $($Matches[4]):$($Matches[5]):$($Matches[6])", 'yyyy-MM-dd HH:mm:ss', $null) -ge $cpuCutoff
            } else { [datetime]$ts -ge $cpuCutoff }
        } catch { $false }
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
            try {
                $ts = $_.ts_msk
                if ($ts -match '^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})') {
                    [datetime]::ParseExact("$($Matches[1])-$($Matches[2])-$($Matches[3]) $($Matches[4]):$($Matches[5]):$($Matches[6])", 'yyyy-MM-dd HH:mm:ss', $null) -ge $diffsCutoff
                } else { [datetime]$ts -ge $diffsCutoff }
            } catch { $false }
        }
        if ($diffsRows) { $frameCount10m = @($diffsRows).Count }
    } catch {}
}

# ── CPU из cpu.csv (2_send / transfer client) ────────────────────────────────
$sendCpuLine = ""
$sendCpuCsv = Join-Path $REPO ".output\transfer_client\meta\$today\cpu.csv"
if (Test-Path $sendCpuCsv) {
    $sCut = (Get-Date).AddMinutes(-10)
    $sRows = Import-Csv $sendCpuCsv | Where-Object {
        try {
            $ts = $_.ts_msk
            if ($ts -match '^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})') {
                [datetime]::ParseExact("$($Matches[1])-$($Matches[2])-$($Matches[3]) $($Matches[4]):$($Matches[5]):$($Matches[6])", 'yyyy-MM-dd HH:mm:ss', $null) -ge $sCut
            } else { [datetime]$ts -ge $sCut }
        } catch { $false }
    }
    if ($sRows -and @($sRows).Count -gt 0) {
        $sVals = @($sRows | ForEach-Object { [int][math]::Round([double]$_.cpu_pct) })
        $sMn = ($sVals | Measure-Object -Minimum).Minimum
        $sAvg = [int][math]::Round(($sVals | Measure-Object -Average).Average)
        $sMx = ($sVals | Measure-Object -Maximum).Maximum
        $sLst = $sVals[$sVals.Count - 1]
        $sendCpuLine = "${sMn}%/${sAvg}%/${sMx}%/${sLst}%"
    }
}

# ── WIN_SVC: структурированные данные для master status.sh ───────────────────
# Строки с префиксом "# WIN_SVC|" отфильтровываются из вывода на терминал,
# но используются status.sh для построения таблицы сервисов в .md

$winHost    = $env:COMPUTERNAME.ToLower()

# stats для 1_motion_diff: 10м, fallback 60м, fallback кадры из diffs.csv
# motion_diff — непрерывный цикл: обычно «1 прогон (X файлов)», но при рестарте(ах) в окне —
# «N прогонов» (N = число стартов «Порог:», в т.ч. из соседнего date-каталога у полуночи).
if ($count10m -gt 0) {
    $stats10m = Fmt-Runs $motionRuns10m $count10m " (10м)" $times10m $motionCpuLine $procMs10m
} elseif ($count60m -gt 0) {
    $stats10m = Fmt-Runs $motionRuns60m $count60m " (60м)" $times60m $motionCpuLine $procMs60m
} elseif ($frameCount10m -gt 0) {
    # нет событий движения, но кадры обрабатываются
    $stats10m = Fmt-Runs $motionRuns10m $frameCount10m " (10м)" @() $motionCpuLine $procMs10m
} elseif ($motionCpuLine -ne "") {
    # нет ни движения ни diffs.csv, но CPU есть — хотя бы покажем CPU
    $stats10m = "—<br>${motionCpuLine}"
} else { $stats10m = "--" }

# stats для 2_send: 10м с таймингом + CPU (cpu.csv клиента). CPU есть даже без отправок.
# 2_send — батчи отправки: «N прогонов (X файлов)», N = число батчей [в батче 1/K]
if ($sendCount10m -gt 0) {
    $sb = if ($sendBatches10m -gt 0) { $sendBatches10m } else { 1 }
    $sendStats10m = Fmt-Runs $sb $sendCount10m " (10м)" $sendTimes10m $sendCpuLine
} elseif ($sendCpuLine -ne "") {
    $sendStats10m = "—<br>${sendCpuLine}"
} else { $sendStats10m = "--" }

# Состояние каталога: "N файл.<br>дата-время последнего" или "—" (пусто = очередь не копится)
function Get-DirState($dir) {
    if (-not $dir -or -not (Test-Path $dir)) { return "—" }
    $files = @(Get-ChildItem -Path $dir -Recurse -File -Filter *.jpg -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) { return "—" }
    $dirs = @(Get-ChildItem -Path $dir -Recurse -Directory -ErrorAction SilentlyContinue).Count
    $last = ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime.ToString("yyyy-MM-dd HH:mm")
    return ("{0} файл. · {1} кат.<br>{2}" -f $files.Count, $dirs, $last)
}
# Очередь кадров: выход 1_motion_diff = вход 2_send (те же файлы до отправки+удаления)
$motionState = Get-DirState $motionDir

# Адрес transfer-сервера (куда шлёт 2_send) — из .env TRANSFER_SERVER (без http:// и /)
$transferAddr = "transfer server"
$envFileTS = Join-Path $REPO ".env"
if (Test-Path $envFileTS) {
    $tl = (Get-Content $envFileTS | Where-Object { $_ -match '^\s*TRANSFER_SERVER\s*=' } | Select-Object -First 1)
    if ($tl) {
        $tv = ((($tl -replace '^\s*TRANSFER_SERVER\s*=\s*','') -replace '\s*#.*$','').Trim() -replace '^https?://','') -replace '/+$',''
        if ($tv) { $transferAddr = $tv }
    }
}

# Префикс даты лог-файла к строке лога — время в логе есть, а дата может быть старой
# (у выключенной машины показывает реальную дату последней активности).
function _WithDate($line, $date) {
    if (-not $line -or $line -in @('--', '—', '-') -or -not $date) { return $line }
    return "$date<br>$line"
}

foreach ($s in $scriptDefs) {
    $pid2  = $scriptPids[$s.label]
    $stat  = if ($pid2) { "OK" } else { "NOK" }
    if ($s.label -eq "1_motion_diff") {
        $in2  = "RTSP камеры<br>—"
        $out2 = "1_motion_diff/images/<br>$motionState"
        $log2 = _WithDate (Shorten-WinLog $lastLogLine) $logDate
        $idle2 = _WithDate $motionIdleLine $logDate
        $st2  = $stats10m
    } elseif ($s.label -eq "2_send") {
        $in2  = "1_motion_diff/images/<br>$motionState"
        $out2 = "→ $transferAddr<br>—"
        $log2 = _WithDate (Shorten-WinLog $sendLastLogLine) $sendLogDate
        $idle2 = _WithDate $sendLastIdleLine $sendLogDate
        $st2  = $sendStats10m
    } else {
        $in2 = "--"; $out2 = "--"; $log2 = "--"; $idle2 = "--"; $st2 = "--"
    }
    Write-Host ("# WIN_SVC|{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}" -f `
        $s.label, $winHost, $stat, $in2, $out2, $log2, $idle2, $st2)
}
