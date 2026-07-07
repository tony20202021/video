#!/usr/bin/env bash
# Веб-разметчик кропов (Модель 1)
#
# Usage:
#   ./sh/train/2_label_ui.sh
#   ./sh/train/2_label_ui.sh --input .output/pipeline/2_yolo_boxes_files/run_xxx
#   ./sh/train/2_label_ui.sh --port 8080
#   ./sh/train/2_label_ui.sh --unlabeled-only
#   ./sh/train/2_label_ui.sh --dataset .data/groups/v1

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="$REPO/.env"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/2_label_ui.py"

export PYTHONIOENCODING=utf-8

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "$ENV_FILE" | grep '=' | grep -v '<')
    set +a
fi

INPUT="$REPO/.data/groups/v1/inference/images/20260706"
LABELS="$REPO/.data/groups/v1/inference/images/20260706/labels.json"
DATASET="$REPO/.data/groups/v1/dataset"
PORT="${LABEL_UI_PORT:-8750}"
UNLABELED_ONLY=0
EXT="jpg"
PROBS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)          INPUT="$2";   shift 2 ;;
        --labels)         LABELS="$2";  shift 2 ;;
        --dataset)        DATASET="$2"; shift 2 ;;
        --port)           PORT="$2";    shift 2 ;;
        --ext)            EXT="$2";     shift 2 ;;
        --unlabeled-only) UNLABELED_ONLY=1; shift ;;
        --all)            UNLABELED_ONLY=0; shift ;;
        --probs)          PROBS=1; shift ;;
        *) echo "[!] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$LABELS" ]] && LABELS="$REPO/.output/train/2_label_ui/labels.json"

echo "=== 2_label_ui ==="
echo "Repo:   $REPO"
echo "Input:  $INPUT"
echo "Labels: $LABELS"
echo "Port:   $PORT"
echo ""

if [[ ! -e "$INPUT" ]]; then
    echo "[!] Not found: $INPUT" >&2
    exit 1
fi

args=("--input" "$INPUT" "--labels" "$LABELS" "--port" "$PORT")
[[ -n "$DATASET" ]]           && args+=("--dataset" "$DATASET")
[[ -n "$EXT" ]]               && args+=("--ext" "$EXT")
[[ "$UNLABELED_ONLY" -eq 1 ]] && args+=("--unlabeled-only")
[[ "$PROBS"          -eq 1 ]] && args+=("--probs")

exec "$PYTHON" "$SCRIPT" "${args[@]}"
