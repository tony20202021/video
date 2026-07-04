#!/usr/bin/env bash
# Motion detection — сохранение LOW-кадров и heartbeat (без YOLO/ML)
#
# Usage:
#   ./sh/pipeline/1_motion_diff.sh
#   ./sh/pipeline/1_motion_diff.sh 7200
#   ./sh/pipeline/1_motion_diff.sh 0 /path/to/run_dir   # --regen-from
#
# Все настройки читаются из .env — менять только там.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="scripts/pipeline/1_motion_diff.py"

# Настройки из .env (для справки):
#   MOTION_DIFF_THRESHOLD = 3.3
#   MOTION_HEARTBEAT_SEC  = 600

# Дополнительные флаги (раскомментировать при необходимости):
EXTRA=(
    # "--tcp"
    # "--cam-ts"
)

# ─────────────────────────────────────────────────────────────────────────────

export PYTHONIOENCODING=utf-8

DURATION="${1:-0}"
REGEN_FROM="${2:-}"

cd "$REPO"

py_args=("$SCRIPT")

if [[ -n "$REGEN_FROM" ]]; then
    py_args+=("--regen-from" "$REGEN_FROM")
    dur_label="regen: $REGEN_FROM"
elif [[ "$DURATION" -gt 0 ]]; then
    py_args+=("--duration" "$DURATION")
    dur_label="${DURATION}s"
else
    dur_label="∞"
fi

for a in "${EXTRA[@]+"${EXTRA[@]}"}"; do
    [[ -n "$a" ]] && py_args+=("$a")
done

echo ""
echo "  1_motion_diff  •  conda:$CONDA_ENV  •  $dur_label"
echo ""

exec "$PYTHON" "${py_args[@]}"
