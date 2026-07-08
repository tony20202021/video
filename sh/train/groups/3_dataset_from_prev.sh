#!/usr/bin/env bash
# Стратегии 4,7: прогон инференса на предыдущем датасете.
#
# Вызывается один раз для старого датасета (или из 3_dataset_build.sh).
#
# Usage:
#   ./sh/train/groups/3_dataset_from_prev.sh <dataset_dir> --output <output_dir>
#   ./sh/train/groups/3_dataset_from_prev.sh .data/groups/v1/dataset \
#       --output .data/groups/v2/dataset
#   ./sh/train/groups/3_dataset_from_prev.sh ... --strategies 7 --dry-run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/dataset_v2_from_prev.py"

export PYTHONIOENCODING=utf-8

_ef() {
    local key="$1" default="$2"
    local val
    val=$(grep -E "^\s*${key}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$default}"
}

CLASSIFY_CONF_HIGH="$(_ef CLASSIFY_CONF_HIGH 0.85)"
export CLASSIFY_CONF_HIGH

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <dataset_dir> --output <dataset_dir> [options]" >&2
    exit 1
fi

exec "$PYTHON" "$SCRIPT" "$@" \
    --conf-high "$CLASSIFY_CONF_HIGH"
