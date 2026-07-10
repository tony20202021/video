#!/usr/bin/env bash
# Авторазметка кадров внутри сцен по 1 вручную размеченному кадру.
#
# Шаг 3 из 4 в пайплайне полу-авторазметки:
#   1. ./4_collect_scenes.sh       → scene_pool/ + scenes.json + representatives/
#   2. Разметить representatives/  → labels.json (1 метка на сцену)
#   3. [этот скрипт]               → дозаполнить labels.json для всей сцены
#   4. ./2_filter_residents.sh --min-diff N → финальная дедупликация
#
# Usage:
#   ./sh/train/residents/5_propagate_scenes.sh
#   ./sh/train/residents/5_propagate_scenes.sh --intra-diff 25
#   ./sh/train/residents/5_propagate_scenes.sh --pool .data/residents/v0/scene_pool --dry-run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/conda_video/bin/python"
SCRIPT="$REPO/scripts/train/propagate_scenes.py"

export PYTHONIOENCODING=utf-8

_ef() {
    local val
    val=$(grep -E "^\s*${1}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$2}"
}

RESIDENTS_VER="$(_ef RESIDENTS_VER v1)"
POOL="$REPO/.data/residents/$RESIDENTS_VER/scene_pool"
INTRA_DIFF=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pool)       POOL="$2";       shift 2 ;;
        --intra-diff) INTRA_DIFF="$2"; shift 2 ;;
        --dry-run)    EXTRA_ARGS+=("--dry-run"); shift ;;
        --report)     EXTRA_ARGS+=("--report");  shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== 5_propagate_scenes ==="
echo "  Пул:         $POOL"
if [[ "$INTRA_DIFF" != "0" ]]; then
    echo "  intra-diff:  $INTRA_DIFF"
fi
echo ""

INTRA_ARG=()
if [[ "$INTRA_DIFF" != "0" ]]; then
    INTRA_ARG=(--intra-diff "$INTRA_DIFF")
fi

"$PYTHON" "$SCRIPT" "$POOL" \
    "${INTRA_ARG[@]+"${INTRA_ARG[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "Следующий шаг:"
echo "  Проверьте labels.json, при необходимости разметьте граничные сцены вручную"
echo "  Финальный отбор: ./sh/train/residents/2_filter_residents.sh --min-diff 40"
