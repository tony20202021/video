#!/usr/bin/env bash
# Сбор кропов жителей из датасета и инференса групп.
#
# Usage:
#   ./sh/train/residents/1_collect_residents.sh
#   ./sh/train/residents/1_collect_residents.sh --dry-run
#   ./sh/train/residents/1_collect_residents.sh --out .data/residents/v0/new
#   ./sh/train/residents/1_collect_residents.sh --interval 10

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/collect_residents.py"

export PYTHONIOENCODING=utf-8

OUT="$REPO/.data/residents/v0/new"
INTERVAL=1          # один кадр на N секунд на камеру
CONF_DELTA=0.1      # брать все кадры в окне с conf ≥ max(окна) − CONF_DELTA
MAX_PERSONS=1       # только одиночные кадры (p1of1)
CLASSES="1_resident 4_guest"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)         OUT="$2";         shift 2 ;;
        --interval)    INTERVAL="$2";    shift 2 ;;
        --conf-delta)  CONF_DELTA="$2";  shift 2 ;;
        --max-persons) MAX_PERSONS="$2"; shift 2 ;;
        --classes)     CLASSES="$2";     shift 2 ;;
        *) EXTRA_ARGS+=("$1");           shift ;;
    esac
done

echo "=== 1_collect_residents ==="
echo "  out:         $OUT"
echo "  interval:    ${INTERVAL}s"
echo "  conf-delta:  ${CONF_DELTA}"
echo "  max-persons: $MAX_PERSONS"
echo "  classes:     $CLASSES"
echo ""

# shellcheck disable=SC2086
exec "$PYTHON" "$SCRIPT" \
    --out "$OUT" \
    --interval "$INTERVAL" \
    --conf-delta "$CONF_DELTA" \
    --max-persons "$MAX_PERSONS" \
    --classes $CLASSES \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
