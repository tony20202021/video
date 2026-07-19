#!/usr/bin/env bash
# Идентификация жителей и гостей из кропов 3_classify_groups (Модель 2) — watch-режим.
#
# Следит за .data/groups/<ver>/inference/images/<date>/ (v4: single/{1_resident,4_guest} + multi/),
# запускает идентификацию при появлении новых кропов.
# Пропуск уже обработанных файлов — на стороне Python (skip-if-exists).
#
# Структура вывода:
#   .data/residents/v1/inference/
#     images/<date>/<person_id>/   — идентифицированные кропы
#     meta/<date>/<run_ts>/        — CSV, charts, run.log
#
# Usage:
#   ./sh/pipeline/4_identify_residents.sh
#   ./sh/pipeline/4_identify_residents.sh --poll-sec 30
#   ./sh/pipeline/4_identify_residents.sh --once
#   ./sh/pipeline/4_identify_residents.sh --identify-conf 0.75
#   ./sh/pipeline/4_identify_residents.sh --model .models/identify/v2.onnx

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/pipeline/4_identify_residents.py"

export PYTHONIOENCODING=utf-8

ENV_FILE="$REPO/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "$ENV_FILE" | grep '=' | grep -v '<')
    set +a
fi

_ef() {
    local key="$1" default="$2"
    local val
    val=$(grep -E "^\s*${key}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$default}"
}

GROUPS_VER="$(_ef GROUPS_VER v3)"
INFERENCE_IMAGES="$REPO/.data/groups/$GROUPS_VER/inference/images"
OUT_DIR="$REPO/.data/residents/v1/inference"
IDENTIFY_CONF="$(_ef IDENTIFY_CONF 0.70)"
POLL_SEC=60
ONCE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --poll-sec)      POLL_SEC="$2"; shift 2 ;;
        --once)          ONCE=1; shift ;;
        --identify-conf) IDENTIFY_CONF="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

_ts() { date '+%H:%M:%S'; }
SCRIPT_NAME="$(basename "$0" .sh)"

MODEL_TAG=$(basename "$(_ef IDENTIFY_MODEL ?)" .onnx)

echo "=== 4_identify_residents ==="
echo "  Input:          $INFERENCE_IMAGES"
echo "  Output:         $OUT_DIR"
echo "  identify_conf:  $IDENTIFY_CONF"
echo "  модель:         $MODEL_TAG"
echo "  poll: ${POLL_SEC}s"
echo "  (пропуск уже обработанных — на стороне Python)"
echo ""
echo "[identify] Ctrl+C для остановки"
echo ""

_count_crops() {
    local date_dir="$1"
    local n=0
    # v4 multi-label раскладка Модели 1: single/{1_resident,4_guest} + multi/ (триггер;
    # фильтрацию multi/ по labels.json делает Python). Плюс старый плоский формат <class>/.
    for d in "$date_dir/single/1_resident" "$date_dir/single/4_guest" "$date_dir/multi" \
             "$date_dir/1_resident" "$date_dir/4_guest"; do
        if [[ -d "$d" ]]; then
            n=$(( n + $(find "$d" -name "*.jpg" -type f 2>/dev/null | wc -l) ))
        fi
    done
    echo "$n"
}

while true; do
    _any=0

    mapfile -t date_dirs < <(
        find "$INFERENCE_IMAGES" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort
    )

    for date_dir in "${date_dirs[@]+"${date_dirs[@]}"}"; do
        n=$(_count_crops "$date_dir")
        [[ "$n" -eq 0 ]] && continue

        date=$(basename "$date_dir")

        # Пропустить дату если нет новых кропов с момента последнего прогона.
        # find по несуществующему каталогу возвращает !=0 → под set -o pipefail это
        # роняет весь скрипт (exit 1); поэтому сначала проверяем наличие каталога.
        latest_meta=""
        if [[ -d "$OUT_DIR/meta/$date" ]]; then
            latest_meta=$(find "$OUT_DIR/meta/$date" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
                          | sort | tail -1)
        fi
        if [[ -n "$latest_meta" ]]; then
            # find | head -1: head закрывает пайп после первой строки → find получает
            # SIGPIPE (141) → под set -o pipefail это роняет скрипт; глушим через || true
            new_crops=$(find "$date_dir" -name "*.jpg" -newer "$latest_meta" -type f \
                        2>/dev/null | head -1 || true)
            if [[ -z "$new_crops" ]]; then
                continue
            fi
        fi

        _any=1
        echo "$(_ts)  INFO      $date: $n кропов — запуск идентификации…"

        "$PYTHON" "$SCRIPT" "$date_dir" \
            --identify-conf "$IDENTIFY_CONF" \
            --output        "$OUT_DIR" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
            || echo "$(_ts)  WARN      $date: идентификация завершилась с ошибкой" >&2

        echo ""
    done

    if [[ "$_any" -eq 0 ]]; then
        echo "$(_ts)  INFO      ($SCRIPT_NAME) Кропов нет в $INFERENCE_IMAGES — ожидание ${POLL_SEC}s…"
    fi

    if [[ "$ONCE" -eq 1 ]]; then
        break
    fi

    sleep "$POLL_SEC"
done
