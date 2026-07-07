#!/usr/bin/env bash
# Сбор кропов жителей из датасета и инференса групп → .data/residents/v1/new/
#
# Usage:
#   ./sh/train/1_collect_residents.sh
#   ./sh/train/1_collect_residents.sh --dry-run
#   ./sh/train/1_collect_residents.sh --src .data/groups/v1/dataset/1_resident
#   ./sh/train/1_collect_residents.sh --out .data/residents/v1/new

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/collect_residents.py"

export PYTHONIOENCODING=utf-8

echo "=== 1_collect_residents ==="
exec "$PYTHON" "$SCRIPT" "$@"
