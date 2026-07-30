#!/usr/bin/env bash
# Веб-разметчик кропов (общий — Модель 1 и Модель 2)
#
# Usage (результат сглаживания — дефолт): достаточно каталога сайдкара smoothed/<date>,
# INPUT/LABELS/PROBS_CSV/DATASET выведутся сами (кропы из images/, метки+probs из smoothed/):
#   ./sh/train/2_label_ui.sh --smoothed .data/groups/v4/inference/smoothed/20260720
# Usage (произвольный вход вручную):
#   ./sh/train/2_label_ui.sh --input .data/groups/v4/inference/images/20260720 --dataset .data/groups/v4/dataset
# Usage (просмотр готового датасета):
#   ./sh/train/2_label_ui.sh --smoothed "" --input .data/groups/v4/dataset

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

# РАЗМЕТКА РЕЗУЛЬТАТА СГЛАЖИВАНИЯ (по умолчанию).
# Достаточно каталога сайдкара smoothed/<date> (SMOOTHED= или --smoothed) — остальное выведется:
#   INPUT     = <inference>/images/<date>                     кропы (images/ НЕ мутируется)
#   LABELS    = smoothed/<date>/labels.json                   сглаженные классы — их и ведём разметкой
#   PROBS_CSV = smoothed/<date>/classifications_smoothed.csv  там же p_* (probs)
#   DATASET   = <vX>/dataset                                  набор классов
SMOOTHED="${SMOOTHED:-$REPO/.data/groups/v4/inference/smoothed/20260720}"
INPUT=""; LABELS=""; DATASET=""; PROBS_CSV=""

# # residents: SMOOTHED="$REPO/.data/residents/v1/inference/smoothed/20260714"
# # просмотр готового датасета вместо сглаживания: --smoothed "" --input .data/groups/v4/dataset

PORT="${LABEL_UI_PORT:-8750}"
UNLABELED_ONLY=0
EXT="jpg"
PROBS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoothed)       SMOOTHED="$2"; shift 2 ;;
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

INPUT="${INPUT:-}"; DATASET="${DATASET:-}"; PROBS_CSV="${PROBS_CSV:-}"; LABELS="${LABELS:-}"

# Вывод путей из каталога сглаживания smoothed/<date> (что явно задано через флаги — не трогаем).
SMOOTHED="${SMOOTHED:-}"
if [[ -n "$SMOOTHED" && -z "$INPUT" ]]; then
    _date="$(basename "$SMOOTHED")"
    _inf="$(dirname "$(dirname "$SMOOTHED")")"   # .../inference
    _ver="$(dirname "$_inf")"                    # .../vX
    INPUT="$_inf/images/$_date"
    [[ -z "$LABELS"    ]] && LABELS="$SMOOTHED/labels.json"
    [[ -z "$PROBS_CSV" ]] && PROBS_CSV="$SMOOTHED/classifications_smoothed.csv"
    [[ -z "$DATASET"   ]] && DATASET="$_ver/dataset"
fi

# Удобство: если задан только DATASET (смотрим датасет) — берём его как вход.
[[ -z "$INPUT" && -n "$DATASET" ]] && INPUT="$DATASET"
[[ -z "$LABELS" && -n "$INPUT" ]] && LABELS="$INPUT/labels.json"

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
