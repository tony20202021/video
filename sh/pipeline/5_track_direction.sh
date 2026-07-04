#!/usr/bin/env bash
# Трекинг людей и определение направления (домой/из дома)
#
# Читает detections.csv из выхода 2_yolo_boxes_files.
# Связывает боксы в треки, классифицирует: домой / из дома / неизвестно.
# Зоны «дверь» и «лифт» — в config.yaml (tracking.zones).
#
# Usage:
#   ./sh/pipeline/5_track_direction.sh
#   ./sh/pipeline/5_track_direction.sh --config config.yaml

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/pipeline/5_track_direction.py"

export PYTHONIOENCODING=utf-8

CONFIG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *)        EXTRA_ARGS+=("$1"); shift ;;
    esac
done

S2DIR="$REPO/.output/pipeline/2_yolo_boxes_files"

# Список входных каталогов (run_* из 2_yolo_boxes_files)
INPUT_DIRS=(
    "$S2DIR/run_20260629_210753_msk"
    # Добавляй нужные прогоны:
    # "$S2DIR/run_20260630_120000_msk"
)

echo "=== 5_track_direction ==="
echo "Repo:   $REPO"
echo "Script: $SCRIPT"
echo "Input:"
for d in "${INPUT_DIRS[@]}"; do echo "  $d"; done
echo ""

missing=0
for d in "${INPUT_DIRS[@]}"; do
    if [[ ! -e "$d" ]]; then
        echo "[!] Not found: $d" >&2
        missing=1
    fi
done
[[ "$missing" -eq 1 ]] && exit 1

extra_flags=()
[[ -n "$CONFIG" ]] && extra_flags+=(--config "$CONFIG")

for input_dir in "${INPUT_DIRS[@]}"; do
    echo "--- $(basename "$input_dir") ---"
    "$PYTHON" "$SCRIPT" "$input_dir" "${extra_flags[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    echo ""
done
