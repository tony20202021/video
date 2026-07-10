#!/usr/bin/env bash
# Финальная фильтрация scene_pool после авторазметки (шаг 1_b_3).
#
# Тот же filter_residents.py, но дефолты для scene_pool:
#   - путь: .data/residents/$RESIDENTS_VER/scene_pool
#   - --min-diff 40  (без diff-фильтра при сборке → теперь нужен строже)
#
# Шаг 4 из 4 в пайплайне полу-авторазметки:
#   1_b_1_collect_scenes.sh       → scene_pool/
#   1_b_2_propagate_scenes.sh     → labels.json
#   ручная разметка representatives/ при необходимости
#   1_b_3_filter_scenes.sh [этот] → финальная дедупликация
#   3_train_residents.sh          → .models/identify/v<N>.onnx
#
# Usage:
#   ./sh/train/residents/1_b_3_filter_scenes.sh
#   ./sh/train/residents/1_b_3_filter_scenes.sh --min-diff 30
#   ./sh/train/residents/1_b_3_filter_scenes.sh --dry-run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/conda_video/bin/python"
SCRIPT="$REPO/scripts/train/filter_residents.py"

export PYTHONIOENCODING=utf-8

_ef() {
    local val
    val=$(grep -E "^\s*${1}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$2}"
}

RESIDENTS_VER="$(_ef RESIDENTS_VER v1)"
DIR="$REPO/.data/residents/$RESIDENTS_VER/scene_pool"
MIN_DIFF=40
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-diff) MIN_DIFF="$2"; shift 2 ;;
        --dir)      DIR="$2";      shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== 1_b_3_filter_scenes ==="
echo "  dir:      $DIR"
echo "  min-diff: $MIN_DIFF"
echo ""
exec "$PYTHON" "$SCRIPT" "$DIR" --min-diff "$MIN_DIFF" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
