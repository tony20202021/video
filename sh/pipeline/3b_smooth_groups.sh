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
#   ./sh/pipeline/3b_smooth_groups.sh --gap 5 --prob-window 3 --prob-window-k 7

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
POLL_SEC="$(_ef SMOOTH_POLL_SEC 120)"
ONCE=0
EXTRA_ARGS=()   # всё прочее (напр. --gap 5) пробрасывается в python и переопределяет env

while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)     ONCE=1; shift ;;
        --poll-sec) POLL_SEC="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Аргументы python из .env — читаются ЗАНОВО каждый поллинг → правки конфига (SMOOTH_*)
# применяются без рестарта (как и правки кода). SMOOTH_PROB_WINDOW=N — режим бегущего окна.
_build_args() {
    local gap p_stay gate maxp viz pw pwk tri mz hv
    gap="$(_ef SMOOTH_GAP_SEC 5)";   p_stay="$(_ef SMOOTH_P_STAY 0.95)"
    gate="$(_ef SMOOTH_GATE 0.8)";   maxp="$(_ef SMOOTH_MAX_PERSONS 1)"
    viz="$(_ef SMOOTH_VIZ 1)";       pw="$(_ef SMOOTH_PROB_WINDOW 0)"
    pwk="$(_ef SMOOTH_PROB_WINDOW_K 0)"          # адаптивное окно: ≤K ближайших кадров (0=все в окне)
    tri="$(_ef SMOOTH_PROB_WINDOW_TRI 0)"        # взвешенное (треугольное) среднее по окну (1=вкл)
    mz="$(_ef SMOOTH_MERGE_ZONES 0)"             # склейка зон d/u одной камеры в один поток
    hv="$(_ef SMOOTH_HARD_VOTE 0)"               # жёсткий голос: 1 класс на визит (вместо Viterbi)
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

echo "=== 3b_smooth_groups ==="
echo "  Каталог:  $IMAGES   poll: ${POLL_SEC}s   (config из .env, авто-применение)"
echo ""

if [[ "$ONCE" -eq 1 ]]; then
    _build_args
    exec "$PYTHON" "$SCRIPT" "${ARGS[@]}" --once "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi

# Watch-режим: bash-цикл заново вызывает python --once каждый поллинг → правки кода И конфига
# подхватываются БЕЗ рестарта (python-процесс не живёт между поллингами).
while true; do
    _build_args
    "$PYTHON" "$SCRIPT" "${ARGS[@]}" --once "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        || echo "[3b] прогон завершился с ошибкой"
    sleep "$POLL_SEC"
done
