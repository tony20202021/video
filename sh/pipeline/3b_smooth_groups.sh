#!/usr/bin/env bash
# Темпоральное сглаживание классов Модели 1 (между classify и identify) — watch-режим.
#
# Следит за .data/groups/<ver>/inference/images/, сглаживает предсказания по времени
# (visit-HMM), перекладывает кропы в single/<class>/ по сглаженному классу и правит labels.json.
# Скорость не важна — важна точность. Запускать ПОСЛЕ 3_classify_groups и ДО 4_identify_residents.
#
# ВНИМАНИЕ: 3b владеет labels.json живого инференса (модель+сглаживание). Ручную разметку под
# датасет вести на снапшоте (копии), иначе 3b затрёт правки.
#
# Usage:
#   ./sh/pipeline/3b_smooth_groups.sh
#   ./sh/pipeline/3b_smooth_groups.sh --once
#   ./sh/pipeline/3b_smooth_groups.sh --gap 60 --p-stay 0.95 --gate 0.8

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/pipeline/3b_smooth_groups.py"

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
IMAGES="$REPO/.data/groups/$GROUPS_VER/inference/images"
GAP="$(_ef SMOOTH_GAP_SEC 60)"
P_STAY="$(_ef SMOOTH_P_STAY 0.95)"
GATE="$(_ef SMOOTH_GATE 0.8)"                 # уверенные предсказания не трогаем (замер: сохраняет меньшинства)
MAX_PERSONS="$(_ef SMOOTH_MAX_PERSONS 1)"     # многолюдные кадры (разные люди) не сглаживаем
POLL_SEC=120
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)        EXTRA_ARGS+=("--once"); shift ;;
        --poll-sec)    POLL_SEC="$2"; shift 2 ;;
        --gap)         GAP="$2"; shift 2 ;;
        --p-stay)      P_STAY="$2"; shift 2 ;;
        --gate)        GATE="$2"; shift 2 ;;
        --max-persons) MAX_PERSONS="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== 3b_smooth_groups ==="
echo "  Каталог:  $IMAGES"
echo "  gap=${GAP}с  p_stay=${P_STAY}  gate=${GATE}  max_persons=${MAX_PERSONS}"
echo "  poll: ${POLL_SEC}s"
echo ""

args=("$IMAGES" "--gap" "$GAP" "--p-stay" "$P_STAY" "--max-persons" "$MAX_PERSONS" "--poll-sec" "$POLL_SEC")
[[ -n "$GATE" ]] && args+=("--gate" "$GATE")

exec "$PYTHON" "$SCRIPT" "${args[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
