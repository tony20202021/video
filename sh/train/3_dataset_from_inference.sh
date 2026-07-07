#!/usr/bin/env bash
# Стратегии 1,2,3,5,8: отбор кропов из одного каталога инференса.
#
# Вызывается для каждой даты отдельно (или из 3_dataset_build.sh циклом).
#
# Usage:
#   ./sh/train/3_dataset_from_inference.sh <date_dir> --output <output_dir>
#   ./sh/train/3_dataset_from_inference.sh .data/groups/v1/inference/images/20260705 \
#       --output .data/groups/v2/dataset
#   ./sh/train/3_dataset_from_inference.sh ... --strategies 1,2 --dry-run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/dataset_v2_from_inference.py"

export PYTHONIOENCODING=utf-8

# Читаем .env
_ef() {
    local key="$1" default="$2"
    local val
    val=$(grep -E "^\s*${key}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$default}"
}

# Пороги из .env
CLASSIFY_CONF="$(_ef CLASSIFY_CONF 0.65)"
CLASSIFY_CONF_HIGH="$(_ef CLASSIFY_CONF_HIGH 0.85)"
MARGIN_THRESH="$(_ef MARGIN_THRESH 0.20)"

export CLASSIFY_CONF CLASSIFY_CONF_HIGH MARGIN_THRESH

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <inference/images/YYYYMMDD> --output <dataset_dir> [options]" >&2
    exit 1
fi

exec "$PYTHON" "$SCRIPT" "$@" \
    --conf-low     "$CLASSIFY_CONF" \
    --conf-high    "$CLASSIFY_CONF_HIGH" \
    --margin-thresh "$MARGIN_THRESH"
