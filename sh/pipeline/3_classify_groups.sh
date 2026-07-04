#!/usr/bin/env bash
# Классификация кропов по группам (Модель 1)
#
# Usage:
#   ./sh/pipeline/3_classify_groups.sh
#   ./sh/pipeline/3_classify_groups.sh --classify-conf 0.70

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/pipeline/3_classify_groups.py"

export PYTHONIOENCODING=utf-8

# Читаем .env
_ef() {
    local key="$1" default="$2"
    local val
    val=$(grep -E "^\s*${key}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$default}"
}

CLASSIFY_CONF="$(_ef CLASSIFY_CONF 0.65)"

S2DIR="$REPO/.output/pipeline/2_yolo_boxes_files"

# Список входных каталогов (весь 2_yolo_boxes_files или конкретные прогоны)
INPUT_DIRS=(
    "$S2DIR"
    # "$S2DIR/run_20260628_202843_msk"
)

echo "=== 3_classify_groups ==="
echo "Repo:   $REPO"
echo "Script: $SCRIPT"
echo "classify_conf=$CLASSIFY_CONF"
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

exec "$PYTHON" "$SCRIPT" "${INPUT_DIRS[@]}" \
    --classify-conf "$CLASSIFY_CONF" \
    "$@"
