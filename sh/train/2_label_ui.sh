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
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/2_label_ui.py"

export PYTHONIOENCODING=utf-8

INPUT=""
LABELS=""
DATASET=""
PORT=5050
UNLABELED_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)       INPUT="$2";   shift 2 ;;
        --labels)      LABELS="$2";  shift 2 ;;
        --dataset)     DATASET="$2"; shift 2 ;;
        --port)        PORT="$2";    shift 2 ;;
        --unlabeled-only) UNLABELED_ONLY=1; shift ;;
        *) echo "[!] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Ищем последний run_* если --input не задан
if [[ -z "$INPUT" ]]; then
    S2="$REPO/.output/pipeline/2_yolo_boxes_files"
    if [[ -d "$S2" ]]; then
        INPUT=$(ls -1dt "$S2"/run_* 2>/dev/null | head -1 || true)
    fi
fi

if [[ -z "$INPUT" ]]; then
    echo "[!] No run_* found in .output/pipeline/2_yolo_boxes_files" >&2
    echo "    Set explicitly: --input <path>" >&2
    exit 1
fi

# Относительные пути → абсолютные
[[ "$INPUT"   != /* ]] && INPUT="$REPO/$INPUT"
[[ -z "$LABELS" ]] && LABELS="$REPO/.output/train/2_label_ui/labels.json"
[[ "$LABELS"  != /* ]] && LABELS="$REPO/$LABELS"
[[ -n "$DATASET" && "$DATASET" != /* ]] && DATASET="$REPO/$DATASET"

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
[[ -n "$DATASET" ]]       && args+=("--dataset" "$DATASET")
[[ "$UNLABELED_ONLY" -eq 1 ]] && args+=("--unlabeled-only")

exec "$PYTHON" "$SCRIPT" "${args[@]}"
