#!/usr/bin/env bash
# YOLO-переобработка сохранённых прогонов из 1_motion_diff
#
# Usage:
#   ./sh/pipeline/2_yolo_boxes_files.sh
#   ./sh/pipeline/2_yolo_boxes_files.sh --yolo-max-fps 1.0 --conf 0.4

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

S1DIR="$REPO/.output/pipeline/1_motion_diff"

# Список прогонов для обработки
INPUT_DIRS=(
    "$S1DIR/run_20260629_081339_msk"
    # Или весь каталог:
    # "$S1DIR"
)

echo "=== 2_yolo_boxes_files ==="
echo "Repo:   $REPO"
echo "Script: $SCRIPT"
echo "Runs:"
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

exec "$PYTHON" "$SCRIPT" "${INPUT_DIRS[@]}" \
    --yolo-max-fps "$YOLO_MAX_FPS" \
    --conf "$CONF" \
    --nms "$NMS" \
    "$@"
