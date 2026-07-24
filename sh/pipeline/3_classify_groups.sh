#!/usr/bin/env bash
# Классификация кропов по группам (Модель 1) — watch-режим.
#
# Непрерывно следит за кропами в 2_yolo_boxes_files/images/,
# запускает инференс и перемещает результаты в inference/images/<date>/<class>/.
#
# Usage:
#   ./sh/pipeline/3_classify_groups.sh
#   ./sh/pipeline/3_classify_groups.sh --poll-sec 30
#   ./sh/pipeline/3_classify_groups.sh --once
#   ./sh/pipeline/3_classify_groups.sh --classify-conf 0.70
#   ./sh/pipeline/3_classify_groups.sh --copy   # копировать, не перемещать

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/pipeline/3_classify_groups.py"

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

S2DIR="$REPO/.output/pipeline/2_yolo_boxes_files/images"
GROUPS_VER="$(_ef GROUPS_VER v3)"
OUT_DIR="$REPO/.data/groups/$GROUPS_VER/inference"
CLASSIFY_CONF="$(_ef CLASSIFY_CONF 0.65)"
POLL_SEC=60
ONCE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --poll-sec)      POLL_SEC="$2"; shift 2 ;;
        --once)          ONCE=1; shift ;;
        --classify-conf) CLASSIFY_CONF="$2"; shift 2 ;;
        --copy)          EXTRA_ARGS+=("--copy"); shift ;;
        *) echo "[!] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

_ts() { date '+%H:%M:%S'; }
SCRIPT_NAME="$(basename "$0" .sh)"

_count_crops() {
    find "$S2DIR" -name "*.jpg" -type f 2>/dev/null | wc -l
}

MODEL_TAG=$(basename "$(_ef CLASSIFY_MODEL ?)" .onnx)

# ID сессии — ОДИН на запуск сервиса: все поллинги пишут в один meta/<date>/<RUN_TS>/ (run.log
# дописывается). Новый каталог только при рестарте (новый RUN_TS) и полночи (python сменит <date>).
RUN_TS="$(TZ='Europe/Moscow' date '+%Y%m%d_%H%M%S')_msk"

echo "=== 3_classify_groups ==="
echo "  Input:         $S2DIR"
echo "  Output:        $OUT_DIR"
echo "  classify_conf: $CLASSIFY_CONF"
echo "  модель:        $MODEL_TAG"
echo "  poll: ${POLL_SEC}s"
echo ""
echo "[classify] Ctrl+C для остановки"
echo ""

while true; do
    n=$(_count_crops)

    if [[ "$n" -gt 0 ]]; then
        echo "$(_ts)  INFO      Найдено кропов: $n — запуск инференса…"

        "$PYTHON" "$SCRIPT" "$S2DIR" \
            --classify-conf "$CLASSIFY_CONF" \
            --output        "$OUT_DIR" \
            --run-ts        "$RUN_TS" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

        echo ""
    else
        echo "$(_ts)  INFO      ($SCRIPT_NAME) Кропов нет в $S2DIR — ожидание ${POLL_SEC}s…"
    fi

    if [[ "$ONCE" -eq 1 ]]; then
        break
    fi

    sleep "$POLL_SEC"
done
