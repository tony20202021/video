#!/usr/bin/env bash
# Обучение идентификатора жителей (Модель 2)
#
# Usage:
#   ./sh/train/residents/5_train_residents.sh --data .output/training/export.zip
#   ./sh/train/residents/5_train_residents.sh --data export_dir --backbone .models/classify/backbone.pt
#   ./sh/train/residents/5_train_residents.sh --data export_dir --init-from .models/identify/v1.pt

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/5_train_residents.py"

export PYTHONIOENCODING=utf-8

DATA=""
BACKBONE=""
INIT_FROM=""
EPOCHS=30
BATCH_SIZE=32
LR=5e-4
VAL_SPLIT=0.2
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)       DATA="$2";      shift 2 ;;
        --backbone)   BACKBONE="$2";  shift 2 ;;
        --init-from)  INIT_FROM="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2";    shift 2 ;;
        --batch-size) BATCH_SIZE="$2";shift 2 ;;
        --lr)         LR="$2";        shift 2 ;;
        --val-split)  VAL_SPLIT="$2"; shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$DATA" ]]; then
    echo "[!] --data required" >&2
    exit 1
fi

if [[ -n "$BACKBONE" && -n "$INIT_FROM" ]]; then
    echo "[!] Use either --backbone or --init-from, not both." >&2
    exit 1
fi

[[ "$DATA"      != /* ]] && DATA="$REPO/$DATA"
[[ -n "$BACKBONE"  && "$BACKBONE"  != /* ]] && BACKBONE="$REPO/$BACKBONE"
[[ -n "$INIT_FROM" && "$INIT_FROM" != /* ]] && INIT_FROM="$REPO/$INIT_FROM"

echo "=== 5_train_residents (Model 2) ==="
echo "Repo:   $REPO"
echo "Script: $SCRIPT"
echo "Data:   $DATA"
[[ -n "$BACKBONE"  ]] && echo "Backbone:  $BACKBONE  (first cycle)"
[[ -n "$INIT_FROM" ]] && echo "Init-from: $INIT_FROM  (retrain)"
[[ -z "$BACKBONE" && -z "$INIT_FROM" ]] && echo "Init: random (no backbone/init-from)"
echo "Epochs=$EPOCHS  BatchSize=$BATCH_SIZE  LR=$LR"
echo "Model output: $REPO/.models/identify/"
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
[[ -n "$BACKBONE"  ]] && args+=(--backbone  "$BACKBONE")
[[ -n "$INIT_FROM" ]] && args+=(--init-from "$INIT_FROM")

exec "$PYTHON" "$SCRIPT" "${args[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
