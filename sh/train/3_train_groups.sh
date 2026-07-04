#!/usr/bin/env bash
# Обучение классификатора групп (Модель 1)
#
# Usage:
#   ./sh/train/3_train_groups.sh --data .data/groups/v1
#   ./sh/train/3_train_groups.sh --data .output/training/export.zip --epochs 30

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/3_train_groups.py"

export PYTHONIOENCODING=utf-8

DATA=""
EPOCHS=20
BATCH_SIZE=32
LR=1e-3
VAL_SPLIT=0.2
CLASS_WEIGHTS=1
WEIGHTED_SAMPLING=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)               DATA="$2";             shift 2 ;;
        --epochs)             EPOCHS="$2";           shift 2 ;;
        --batch-size)         BATCH_SIZE="$2";       shift 2 ;;
        --lr)                 LR="$2";               shift 2 ;;
        --val-split)          VAL_SPLIT="$2";        shift 2 ;;
        --class-weights)      CLASS_WEIGHTS=1;       shift ;;
        --no-class-weights)   CLASS_WEIGHTS=0;       shift ;;
        --weighted-sampling)  WEIGHTED_SAMPLING=1;   shift ;;
        --no-weighted-sampling) WEIGHTED_SAMPLING=0; shift ;;
        *)                    EXTRA_ARGS+=("$1");    shift ;;
    esac
done

if [[ -z "$DATA" ]]; then
    echo "[!] --data required" >&2
    exit 1
fi

[[ "$DATA" != /* ]] && DATA="$REPO/$DATA"

echo "=== 3_train_groups (Model 1) ==="
echo "Repo:   $REPO"
echo "Script: $SCRIPT"
echo "Data:   $DATA"
echo "Epochs=$EPOCHS  BatchSize=$BATCH_SIZE  LR=$LR"
echo "Model output: $REPO/.models/classify/"
echo "Backbone:     $REPO/.models/classify/backbone.pt"
echo ""

if [[ ! -e "$DATA" ]]; then
    echo "[!] Not found: $DATA" >&2
    exit 1
fi

args=(
    --data "$DATA"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --val-split "$VAL_SPLIT"
)
[[ "$CLASS_WEIGHTS"     -eq 1 ]] && args+=(--class-weights)
[[ "$WEIGHTED_SAMPLING" -eq 1 ]] && args+=(--weighted-sampling)

exec "$PYTHON" "$SCRIPT" "${args[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
