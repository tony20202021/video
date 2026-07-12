#!/usr/bin/env bash
# Экспорт размеченных сцен + фильтрация (шаг 1_b_3).
#
# 1. Читает labels.json из scene_pool
# 2. Копирует файлы в training_export/<person_id>/
# 3. Запускает filter_residents.py на каждой классовой подпапке
#
# ВАЖНО: filter_residents.py работает in-place на training_export, НЕ на scene_pool.
#
# Шаг 3 из 4 в пайплайне полу-авторазметки:
#   1_b_1_collect_scenes.sh     → scene_pool/
#   1_b_2_propagate_scenes.sh   → labels.json
#   ручная разметка representatives/
#   1_b_3_filter_scenes.sh [этот] → training_export/<person_id>/
#   3_train_residents.sh          → .models/identify/v<N>.onnx
#
# Usage:
#   ./sh/train/residents/1_b_3_filter_scenes.sh
#   ./sh/train/residents/1_b_3_filter_scenes.sh --min-diff 30
#   ./sh/train/residents/1_b_3_filter_scenes.sh --dry-run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/conda_video/bin/python"
EXPORT_SCRIPT="$REPO/scripts/train/4_export_scenes.py"
FILTER_SCRIPT="$REPO/scripts/train/filter_residents.py"

export PYTHONIOENCODING=utf-8

_ef() {
    local val
    val=$(grep -E "^\s*${1}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$2}"
}

RESIDENTS_VER="$(_ef RESIDENTS_VER v1)"
POOL="$REPO/.data/residents/$RESIDENTS_VER/scene_pool"
OUT="$REPO/.data/residents/$RESIDENTS_VER/training_export"
MIN_DIFF=40
DRY_RUN=0
FILTER_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-diff)  MIN_DIFF="$2"; shift 2 ;;
        --pool)      POOL="$2";     shift 2 ;;
        --out)       OUT="$2";      shift 2 ;;
        --dry-run)   DRY_RUN=1; FILTER_EXTRA+=("--dry-run"); shift ;;
        *) FILTER_EXTRA+=("$1"); shift ;;
    esac
done

echo "=== 1_b_3_filter_scenes ==="
echo "  pool:     $POOL"
echo "  out:      $OUT"
echo "  min-diff: $MIN_DIFF"
[[ "$DRY_RUN" -eq 1 ]] && echo "  [dry-run]"
echo ""

# Step 1: export labeled files to class-organized output directory
EXPORT_ARGS=("--pool" "$POOL" "--out" "$OUT")
[[ "$DRY_RUN" -eq 1 ]] && EXPORT_ARGS+=("--dry-run")
"$PYTHON" "$EXPORT_SCRIPT" "${EXPORT_ARGS[@]}"

[[ "$DRY_RUN" -eq 1 ]] && echo "" && echo "[dry-run] фильтрация пропущена" && exit 0

echo ""
echo "--- фильтрация по классам ---"

# Step 2: run filter_residents.py on each class subdirectory
for class_dir in "$OUT"/*/; do
    [[ -d "$class_dir" ]] || continue
    class_name="$(basename "$class_dir")"
    n_files=$(find "$class_dir" -maxdepth 1 -name '*.jpg' | wc -l)
    echo ""
    echo "  [$class_name] ($n_files файлов)"
    "$PYTHON" "$FILTER_SCRIPT" "$class_dir" --min-diff "$MIN_DIFF" "${FILTER_EXTRA[@]+"${FILTER_EXTRA[@]}"}"
done

echo ""
echo "=== готово ==="
