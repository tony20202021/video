#!/usr/bin/env bash
# YOLO-переобработка сохранённых прогонов из 1_motion_diff
#
# По умолчанию: watch-режим + удаление обработанных файлов.
#
#   ./sh/pipeline/2_yolo_boxes_files.sh                  # watch + delete (default)
#   ./sh/pipeline/2_yolo_boxes_files.sh --poll-sec 3     # изменить интервал опроса
#   ./sh/pipeline/2_yolo_boxes_files.sh --conf 0.4       # пробросить флаг в Python
#   ./sh/pipeline/2_yolo_boxes_files.sh --no-delete-after  # watch без удаления (отладка)
#   ./sh/pipeline/2_yolo_boxes_files.sh --batch            # разовый прогон без удаления

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/pipeline/2_yolo_boxes_files.py"

export PYTHONIOENCODING=utf-8

# Читаем .env
_ef() {
    local key="$1" default="$2"
    local val
    val=$(grep -E "^\s*${key}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$default}"
}

YOLO_MAX_FPS="$(_ef YOLO_MAX_FPS 2.0)"
CONF="$(_ef YOLO_CONF 0.35)"
NMS="$(_ef YOLO_NMS 0.45)"

# Список прогонов для обработки
INPUT_DIRS=(
    "/home/tony/repos/video/.output/transfer/diff"
)

# Параметры watch-режима
WATCH=1
POLL_SEC=5
DELETE_AFTER=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)            WATCH=0; DELETE_AFTER=0; shift ;;
        --no-delete-after)  DELETE_AFTER=0; shift ;;
        --poll-sec)         POLL_SEC="$2"; shift 2 ;;
        *)                  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== 2_yolo_boxes_files ==="
echo "Repo:   $REPO"
echo "Script: $SCRIPT"
echo "Runs:"
for d in "${INPUT_DIRS[@]}"; do echo "  $d"; done
if [[ "$WATCH" -eq 1 ]]; then
    echo "Режим:  watch (poll ${POLL_SEC}s, delete-after=$([ "$DELETE_AFTER" -eq 1 ] && echo да || echo нет))"
fi
echo ""

# ── Вспомогательная функция: запуск скрипта на каталог ──────────────────────
_run_dir() {
    local input_dir="$1"
    "$PYTHON" "$SCRIPT" "$input_dir" \
        --yolo-max-fps "$YOLO_MAX_FPS" \
        --conf "$CONF" \
        --nms "$NMS" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

# ── Batch-режим ──────────────────────────────────────────────────────────────
if [[ "$WATCH" -eq 0 ]]; then
    missing=0
    for d in "${INPUT_DIRS[@]}"; do
        if [[ ! -e "$d" ]]; then
            echo "[!] Not found: $d" >&2
            missing=1
        fi
    done
    [[ "$missing" -eq 1 ]] && exit 1

    for input_dir in "${INPUT_DIRS[@]}"; do
        echo "--- $(basename "$input_dir") ---"
        _run_dir "$input_dir"
        echo ""
    done
    exit 0
fi

# ── Watch-режим ──────────────────────────────────────────────────────────────
# Файлы-состояния (для режима без --delete-after): хранят список обработанных путей
declare -A _STATE
for d in "${INPUT_DIRS[@]}"; do
    _STATE[$d]="$d/.yolo_seen"
done

echo "[watch] Ctrl+C для остановки"
echo ""

while true; do
    _any=0
    for input_dir in "${INPUT_DIRS[@]}"; do
        [[ ! -d "$input_dir" ]] && continue

        # Найти jpg-файлы
        mapfile -t _all < <(find "$input_dir" -maxdepth 4 -name "*.jpg" -type f 2>/dev/null | sort)
        [[ ${#_all[@]} -eq 0 ]] && continue

        if [[ "$DELETE_AFTER" -eq 1 ]]; then
            # В delete-режиме: все найденные файлы — новые
            _new=("${_all[@]}")
        else
            # Отфильтровываем уже обработанные
            _statefile="${_STATE[$input_dir]}"
            if [[ -f "$_statefile" ]]; then
                mapfile -t _seen < "$_statefile"
            else
                _seen=()
            fi
            _new=()
            for f in "${_all[@]}"; do
                # shellcheck disable=SC2076
                if [[ ! " ${_seen[*]+${_seen[*]}} " =~ " $f " ]]; then
                    _new+=("$f")
                fi
            done
        fi

        [[ ${#_new[@]} -eq 0 ]] && continue

        _any=1
        echo "[$(date '+%H:%M:%S')] $(basename "$input_dir"): ${#_new[@]} новых файлов"

        _run_dir "$input_dir"
        _rc=$?

        if [[ $_rc -eq 0 ]]; then
            if [[ "$DELETE_AFTER" -eq 1 ]]; then
                rm -f "${_new[@]}"
                echo "[$(date '+%H:%M:%S')] Удалено: ${#_new[@]} файлов"
            else
                printf '%s\n' "${_new[@]}" >> "${_STATE[$input_dir]}"
            fi
        else
            echo "[!] Скрипт вернул ошибку ($_rc), файлы не удалены" >&2
        fi
        echo ""
    done

    [[ "$_any" -eq 0 ]] && sleep "$POLL_SEC"
done
