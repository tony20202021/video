#!/usr/bin/env bash
# Фильтрация кропов жителей.
#
# Фильтры (дефолты):
#   --min-conf 0.5      — по confidence Model 1 из имени файла
#   --min-blur 40       — резкость (дисперсия Лапласиана)
#   --yolo-persons 1    — перезапускает YOLO на кропах, удаляет если >1 чел
#   --min-coverage 0.15 — bbox человека должен занимать ≥15% кадра
#   --min-diff 15       — жадный pixel-diff: следующий кадр берётся только если
#                         отличается от предыдущего оставленного на ≥15 (mean abs)
#
# Usage:
#   ./sh/train/residents/2_filter_residents.sh                        # дефолт: v0/new
#   ./sh/train/residents/2_filter_residents.sh --dry-run
#   ./sh/train/residents/2_filter_residents.sh --min-diff 20
#   ./sh/train/residents/2_filter_residents.sh --min-diff 0           # отключить diff
#   ./sh/train/residents/2_filter_residents.sh .data/residents/v1/new

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/filter_residents.py"

export PYTHONIOENCODING=utf-8

DIR="${1:-$REPO/.data/residents/v0/new}"
MIN_DIFF=20

EXTRA_ARGS=()
shift || true   # сдвигаем DIR
while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-diff) MIN_DIFF="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1");     shift ;;
    esac
done

echo "=== 2_filter_residents ==="
echo "  dir:      $DIR"
echo "  min-diff: $MIN_DIFF"
echo ""
exec "$PYTHON" "$SCRIPT" "$DIR" --min-diff "$MIN_DIFF" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
