#!/usr/bin/env bash
# Темпоральное сглаживание идентификации жителей (Модель 2) — watch-режим.
#
# Следит за .data/residents/<ver>/inference/images/, сглаживает person_id по времени внутри визита
# (то же ядро smooth_core, что у 3b для групп). images/ НЕ мутирует — пишет сайдкар smoothed/<date>/.
# Запускать ПОСЛЕ 4_identify_residents. Набор жителей открытый → берётся из p_* колонок CSV.
#
# Конфиг общий с 3b (SMOOTH_* в .env) — при желании позже развести отдельными SMOOTH_ID_*.
#
# Usage:
#   ./sh/pipeline/4b_smooth_identity.sh
#   ./sh/pipeline/4b_smooth_identity.sh --once
#   ./sh/pipeline/4b_smooth_identity.sh --prob-window 0 --gate 0 --p-stay 0.9 --viz

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/pipeline/4b_smooth_identity.py"

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

RESIDENTS_VER="$(_ef RESIDENTS_VER v1)"
IMAGES="$REPO/.data/residents/$RESIDENTS_VER/inference/images"
POLL_SEC="$(_ef SMOOTH_POLL_SEC 120)"
ONCE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)     ONCE=1; shift ;;
        --poll-sec) POLL_SEC="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Аргументы python из .env (SMOOTH_*) — читаются ЗАНОВО каждый поллинг → правки конфига без рестарта.
_build_args() {
    local gap p_stay gate maxp viz pw pwk tri mz hv
    gap="$(_ef SMOOTH_GAP_SEC 5)";   p_stay="$(_ef SMOOTH_P_STAY 0.95)"
    gate="$(_ef SMOOTH_GATE 0.8)";   maxp="$(_ef SMOOTH_MAX_PERSONS 1)"
    viz="$(_ef SMOOTH_VIZ 1)";       pw="$(_ef SMOOTH_PROB_WINDOW 0)"
    pwk="$(_ef SMOOTH_PROB_WINDOW_K 0)"
    tri="$(_ef SMOOTH_PROB_WINDOW_TRI 0)"
    mz="$(_ef SMOOTH_MERGE_ZONES 0)"             # склейка зон d/u: тот же житель в один поток
    hv="$(_ef SMOOTH_HARD_VOTE 0)"               # жёсткий голос: 1 житель на визит (общий флаг с 3b)
    ARGS=("$IMAGES" "--gap" "$gap" "--p-stay" "$p_stay" "--max-persons" "$maxp")
    [[ -n "$gate" ]] && ARGS+=("--gate" "$gate")
    [[ "$viz" == "1" ]] && ARGS+=("--viz")
    [[ -n "$pw" && "$pw" != "0" ]] && ARGS+=("--prob-window" "$pw")
    [[ -n "$pwk" && "$pwk" != "0" ]] && ARGS+=("--prob-window-k" "$pwk")
    [[ "$tri" == "1" ]] && ARGS+=("--prob-window-tri")
    [[ "$mz" == "1" ]] && ARGS+=("--merge-zones")
    [[ "$hv" == "1" ]] && ARGS+=("--hard-vote")
    return 0     # иначе последний ложный [[ ]] && … вернёт 1 → set -e убьёт сервис (флаги=0)
}

echo "=== 4b_smooth_identity ==="
echo "  Каталог:  $IMAGES   poll: ${POLL_SEC}s   (config из .env SMOOTH_*, авто-применение)"
echo ""

if [[ "$ONCE" -eq 1 ]]; then
    _build_args
    exec "$PYTHON" "$SCRIPT" "${ARGS[@]}" --once "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi

while true; do
    _build_args
    "$PYTHON" "$SCRIPT" "${ARGS[@]}" --once "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        || echo "[4b] прогон завершился с ошибкой"
    sleep "$POLL_SEC"
done
