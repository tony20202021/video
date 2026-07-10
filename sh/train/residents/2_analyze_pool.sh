#!/usr/bin/env bash
# Анализ полного пула кадров для датасета жителей (до дедупликации).
# Строит графики распределений в .data/residents/v0/analysis/
#
# Usage:
#   ./sh/train/residents/3_analyze_pool.sh
#   ./sh/train/residents/3_analyze_pool.sh --interval 1
#   ./sh/train/residents/3_analyze_pool.sh --out .data/residents/v0/analysis

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/analyze_residents_pool.py"

export PYTHONIOENCODING=utf-8

exec "$PYTHON" "$SCRIPT" "$@"
