#!/usr/bin/env bash
# Идентификация жителей из кропов 3_classify_groups (Модель 2)
#
# Usage:
#   ./sh/pipeline/4_identify_residents.sh
#   ./sh/pipeline/4_identify_residents.sh --identify-conf 0.75
#   ./sh/pipeline/4_identify_residents.sh --model .models/identify/v1.onnx

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/pipeline/4_identify_residents.py"

export PYTHONIOENCODING=utf-8

# Читаем .env
_ef() {
    local key="$1" default="$2"
    local val
    val=$(grep -E "^\s*${key}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$default}"
}

IDENTIFY_CONF="$(_ef IDENTIFY_CONF 0.70)"

S3DIR="$REPO/.output/pipeline/3_classify_groups"

# Список входных каталогов
INPUT_DIRS=(
    "$S3DIR"
    # "$S3DIR/run_20260628_202843_msk"
)

echo "=== 4_identify_residents ==="
echo "Repo:   $REPO"
echo "Script: $SCRIPT"
echo "identify_conf=$IDENTIFY_CONF"
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

for input_dir in "${INPUT_DIRS[@]}"; do
    echo "--- $(basename "$input_dir") ---"
    "$PYTHON" "$SCRIPT" "$input_dir" \
        --identify-conf "$IDENTIFY_CONF" \
        "$@"
    echo ""
done
