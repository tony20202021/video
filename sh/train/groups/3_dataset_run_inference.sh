#!/usr/bin/env bash
# Прогоняет GroupClassifier по датасету, создаёт inference-style каталог.
#
# Эквивалент: scripts/train/dataset_run_inference.py
# После выполнения полученный каталог можно передать в 3_dataset_from_inference.sh.
#
# Usage:
#   ./sh/train/groups/3_dataset_run_inference.sh \
#       .data/groups/v1/dataset \
#       --output .data/groups/v1/inference/images/dataset
#
#   ./sh/train/groups/3_dataset_run_inference.sh \
#       .data/groups/v2/dataset \
#       --output .data/groups/v2/inference/images/dataset \
#       --dry-run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/dataset_run_inference.py"

export PYTHONIOENCODING=utf-8

_ef() {
    local key="$1" default="$2"
    local val
    val=$(grep -E "^\s*${key}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$default}"
}

CLASSIFY_CONF="$(_ef CLASSIFY_CONF 0.65)"
export CLASSIFY_CONF

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <dataset_dir> --output <output_dir> [--copy] [--dry-run]" >&2
    exit 1
fi

exec "$PYTHON" "$SCRIPT" "$@" --conf-low "$CLASSIFY_CONF"
