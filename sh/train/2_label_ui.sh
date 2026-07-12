#!/usr/bin/env bash
# Веб-разметчик кропов (общий — Модель 1 и Модель 2)
#
# Usage (жители — дефолт):
#   ./sh/train/2_label_ui.sh
# Usage (группы):
#   ./sh/train/2_label_ui.sh --input .data/groups/v1/inference/images/20260707 --dataset .data/groups/v1/dataset

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

# groups
INPUT="$REPO/.data/groups/v1/inference/images/20260707"
LABELS="$REPO/.data/groups/v1/inference/images/20260707/labels.json"
DATASET="$REPO/.data/groups/v1/dataset"

# # residents v1
# INPUT="$REPO/.data/residents/v1/inference/images/20260711"
# LABELS="$REPO/.data/residents/v1/inference/images/20260711/labels.json"
# DATASET="$REPO/.data/residents/v1/dataset"
# PROBS_CSV="$REPO/.data/residents/v1/inference/images/20260711/identifications.csv"

PORT="${LABEL_UI_PORT:-8750}"
UNLABELED_ONLY=1
EXT="jpg"
PROBS=1
PROBS_CSV=""

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
        --probs-csv)      PROBS_CSV="$2"; shift 2 ;;
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
[[ -n "$PROBS_CSV"   ]]       && args+=("--probs-csv" "$PROBS_CSV")
[[ -z "$PROBS_CSV" && "$PROBS" -eq 1 ]] && args+=("--probs")

exec "$PYTHON" "$SCRIPT" "${args[@]}"
