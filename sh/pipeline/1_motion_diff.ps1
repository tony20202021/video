# 1_motion_diff.ps1 — motion detection, сохранение LOW-кадров и heartbeat (без YOLO/ML)
#
# Использование:
#   .\sh\pipeline\1_motion_diff.ps1              — запустить бесконечно
#   .\sh\pipeline\1_motion_diff.ps1 7200         — запустить на 2 часа (секунды)
#   .\sh\pipeline\1_motion_diff.ps1 0 путь\к\run — перегенерировать графики из старого прогона
#
# Все настройки читаются из .env — менять только там, не здесь.

param(
    [int]    $Duration   = 0,    # секунды; 0 = бесконечно
    [string] $RegenFrom  = ""    # путь к run_xxx для --regen-from
)

$ErrorActionPreference = "Stop"
$Repo      = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$CONDA_ENV = "conda_video"
$PY        = "$env:USERPROFILE\miniconda3\envs\$CONDA_ENV\python.exe"
$SCRIPT    = "$Repo\scripts\pipeline\1_motion_diff.py"

# ─── Текущие значения из .env (для справки; менять в .env, не здесь) ──────────
#   MOTION_DIFF_THRESHOLD = 3.3      # порог motion diff
#   MOTION_HEARTBEAT_SEC  = 600      # пульс каждые 10 мин

# ─── Дополнительные флаги (раскомментировать при необходимости) ───────────────
$EXTRA = @(
    # "--tcp"       # RTSP через TCP (если UDP нестабилен)
    # "--cam-ts"    # добавлять время камеры из RTCP NTP в имя файла
)

# ═══════════════════════════════════════════════════════════════════════════════

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null
$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$py_args = [System.Collections.Generic.List[string]]::new()
$py_args.Add($SCRIPT)

if ($RegenFrom -ne "") {
    $py_args.Add("--regen-from")
    $py_args.Add($RegenFrom)
} elseif ($Duration -gt 0) {
    $py_args.Add("--duration")
    $py_args.Add("$Duration")
}
foreach ($a in $EXTRA) { if ($a -ne "") { $py_args.Add($a) } }

$dur_label = if ($RegenFrom -ne "")  { "regen: $RegenFrom" }
             elseif ($Duration -gt 0) { "${Duration}s ($('{0:0.0}' -f ($Duration/3600)) ч)" }
             else                      { "∞ (бесконечно)" }

Write-Host ""
Write-Host "  1_motion_diff  •  conda:$CONDA_ENV  •  $dur_label" -ForegroundColor Green
Write-Host ""

& $PY @py_args
