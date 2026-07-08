#!/usr/bin/env bash
# Стратегия F: дополнение минорных классов из предыдущего датасета.
# Для каждого класса где new < prev — копирует все файлы из prev (дедуп по имени).
#
# Запускается ПОСЛЕ всех остальных стратегий.
#
# Usage:
#   ./sh/train/groups/3_dataset_fill_minor.sh <prev_dataset> --output <new_dataset>
#   ./sh/train/groups/3_dataset_fill_minor.sh .data/groups/v1/dataset \
#       --output .data/groups/v2/dataset [--dry-run]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/dataset_fill_minor.py"

export PYTHONIOENCODING=utf-8

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <prev_dataset> --output <new_dataset> [--dry-run]" >&2
    exit 1
fi

exec "$PYTHON" "$SCRIPT" "$@"
