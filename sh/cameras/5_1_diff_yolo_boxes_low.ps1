# Запуск 5_1_diff_yolo_boxes_low.py — motion detection → YOLOv8n → кадры с людьми
#
# Использование:
#   .\sh\cam5.ps1              — запустить бесконечно
#   .\sh\cam5.ps1 7200         — запустить на 2 часа (секунды)
#   .\sh\cam5.ps1 0 путь\к\run — перегенерировать графики из старого прогона
#
# Все настройки читаются из .env — менять только там, не здесь.

param(
    [int]    $Duration   = 0,     # секунды; 0 = бесконечно
    [string] $RegenFrom  = ""     # путь к run_xxx для --regen-from
)

# ─── Пути (менять только при переезде проекта) ─────────────────────────────────
$CONDA_EXE  = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
$CONDA_ENV  = "conda_video"
$PROJECT    = "E:\_Home\Tony\pet projects\video"
$SCRIPT     = "scripts\cameras\5_1_diff_yolo_boxes_low.py"

# ─── Текущие значения из .env (для справки; менять в .env, не здесь) ──────────
#   MOTION_DIFF_THRESHOLD = 3.3      # порог motion diff
#   MOTION_HEARTBEAT_SEC  = 600      # пульс каждые 10 мин
#   YOLO_CONF             = 0.25     # порог уверенности YOLO
#   YOLO_NMS              = 0.55     # NMS IoU
#   YOLO_MAX_FPS          = 3        # макс. частота YOLO на камеру

# ─── Дополнительные флаги (раскомментировать при необходимости) ───────────────
$EXTRA = @(
    # "--tcp"          # RTSP через TCP (если UDP нестабилен)
    # "--save-raw"     # сохранять сырой кадр до YOLO при каждом срабатывании
    # "--cam-ts"       # добавлять время камеры из OSD в имя файла
)

# ═══════════════════════════════════════════════════════════════════════════════

$env:PYTHONIOENCODING = "utf-8"
Set-Location $PROJECT

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
Write-Host "  cam5  •  conda:$CONDA_ENV  •  $dur_label" -ForegroundColor Cyan
Write-Host ""

& $CONDA_EXE run -n $CONDA_ENV python @py_args
